#!/usr/bin/env python3
"""First-30s original/Demucs/Mel-Band comparison with Whisper turbo. No cue writes."""
import argparse
import copy
import csv
import datetime
import fcntl
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import time

import song_language
import whisper_experiment as baseline

REPORT_POLICY = 'stem-comparison-label-filter-v1'
LABEL_NAMES = r'(?:music|instrumental|silence|applause|laughter|singing|background music)'
LABEL = re.compile(r'\[\s*' + LABEL_NAMES + r'\s*\]|\(\s*' + LABEL_NAMES + r'\s*\)', re.I)


def describe_words(result, duration, reference):
    """Filter explicit event labels only, preserving raw ASR and short exclamations."""
    cleaned = copy.deepcopy(result)
    ignored = []
    for segment in cleaned.get('segments', []):
        words = segment.get('words', [])
        text = ''.join(word.get('word', '') for word in words)
        spans = [match.span() for match in LABEL.finditer(text)]
        position, kept = 0, []
        for word in words:
            token = word.get('word', '')
            end = position + len(token)
            remainder = ''.join(char for i, char in enumerate(token, position)
                                if not any(a <= i < b for a, b in spans) and char not in '♪♫♬')
            if not any(c.isalnum() for c in remainder):
                if token.strip():
                    ignored.append(token.strip())
            else:
                kept.append({**word, 'word': remainder})
            position = end
        segment['words'] = kept
    summary = baseline.describe_words(cleaned, 0, duration, reference)
    summary['ignored_labels'] = ignored
    summary['first_words'] = ''.join(w['word'] for w in summary['words'][:8]).strip()
    if ignored:
        summary['quality_flags'] = sorted(set(summary['quality_flags'] + ['nonlexical_tokens_ignored']))
    return summary


def key_for(identity):
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def baseline_source_digest():
    # Leave the baseline script unchanged so its existing raw evidence can be
    # reused. Reporting improvements do not invalidate audio or ASR inference.
    return hashlib.sha256(b''.join(Path(__file__).with_name(name).read_bytes()
        for name in ('whisper_experiment.py', 'demucs_backend.py', 'model_worker.py'))).hexdigest()


def asr_identity(item, variant, source_digest, versions, language):
    # Matches the existing --windows 30 --models turbo experiment exactly.
    return dict(audio=item['identity'], audio_hash=item['hashes'][variant + '.wav'],
                variant=variant, duration=min(30.0, item['duration']), requested_window=30.0,
                model='turbo', options=baseline.decoding_options(language),
                source_digest=source_digest, packages=versions)


def prepare_melband(song, parent, root, separator, versions, threads):
    identity = dict(sha256=parent['sha256'], start=0, duration=30.0,
                    separator=separator.identity, packages=versions,
                    decoder_sha256=baseline.sha256(Path(__file__).with_name('model_worker.py')))
    directory = root / 'audio' / key_for(identity)
    directory.mkdir(exist_ok=True)
    path, manifest = directory / 'vocals.wav', directory / 'audio.json'
    stored = baseline.cached_json(manifest, identity)
    if stored and path.is_file() and baseline.sha256(path) == stored['hashes']['vocals.wav']:
        return {**stored, 'directory': str(directory), 'reused': True, 'prepare_seconds_this_run': 0}
    began = time.monotonic()
    print(f'Mel-Band: first 30s of {parent["song"]}', flush=True)
    separation = separator.separate(song, path, 30.0, threads)
    if abs(separation['audio_seconds'] - parent['duration']) > .03:
        raise ValueError('Separator durations differ; refusing an unequal-window comparison')
    stored = dict(status='complete', identity=identity, song=parent['song'], sha256=parent['sha256'],
                  duration=parent['duration'], start=0, separation=separation,
                  hashes={'vocals.wav': baseline.sha256(path)}, preparation_seconds=time.monotonic() - began)
    baseline.atomic_json(manifest, stored)
    return {**stored, 'directory': str(directory), 'reused': False,
            'prepare_seconds_this_run': stored['preparation_seconds']}


def save_report(root, report):
    rows = report['rows']
    report['fresh_asr_jobs'] = sum(not row['reused'] for row in rows)
    report['reused_asr_jobs'] = sum(row['reused'] for row in rows)
    report['asr_seconds_this_run'] = sum(row['seconds_this_run'] for row in rows)
    baseline.atomic_json(root / 'comparison-report.json', report)
    baseline.atomic_json(root / 'runs' / f'comparison-{report["run_id"]}.json', report)
    with (root / 'comparison-details.tsv').open('w', newline='') as stream:
        writer = csv.writer(stream, delimiter='\t')
        writer.writerow(['song', 'input', 'first_word_s', 'reference_s', 'delta_s', 'word_count',
                         'seconds_this_run', 'cached', 'quality_flags', 'first_words', 'raw_transcript'])
        for row in rows:
            ref = row.get('reference') or {}
            writer.writerow([row['song'], row['input'], row['first_word_start'],
                             ref.get('vocal_start') if ref.get('audio_matches') else None,
                             row['first_word_minus_reference_seconds'], row['word_count'],
                             round(row['seconds_this_run'], 3), row['reused'],
                             ','.join(row['quality_flags']), row['first_words'], row['transcript']])
    with (root / 'comparison.tsv').open('w', newline='') as stream:
        writer = csv.writer(stream, delimiter='\t')
        writer.writerow(['song', 'original_s', 'demucs_s', 'melband_s', 'reference_s',
                         'demucs_flags', 'melband_flags'])
        for song in report['songs']:
            group = {row['input']: row for row in rows if row['song'] == song}
            ref = next((r['reference'] for r in group.values() if r.get('reference')), {})
            writer.writerow([song, *(group.get(name, {}).get('first_word_start')
                                     for name in ('original', 'demucs', 'melband')),
                             ref.get('vocal_start') if ref.get('audio_matches') else None,
                             *((','.join(group[name]['quality_flags']) or 'none') if name in group else 'FAILED'
                               for name in ('demucs', 'melband'))])


def run(args, root):
    import torch
    import whisper
    import demucs_backend
    import melband_backend
    from model_worker import decode
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; CPU fallback disabled')
    torch.set_num_threads(args.threads)
    versions = baseline.packages()
    mel_versions = {**versions, **{name: importlib.metadata.version(name)
                    for name in ('einops', 'rotary-embedding-torch', 'beartype', 'librosa', 'soundfile', 'PyYAML')}}
    digest = baseline_source_digest()
    references = json.loads(args.references.read_text()) if args.references else {}
    songs = args.songs.resolve()
    selected = baseline.select_songs(songs, args.song_like, args.all)
    began = time.monotonic()
    report = dict(run_id=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'),
                  status='running', policy=REPORT_POLICY, source_digest=digest,
                  comparison_source_sha256=baseline.sha256(Path(__file__)),
                  gpu=torch.cuda.get_device_name(0), packages=mel_versions,
                  songs=[s.relative_to(songs).as_posix() for s in selected], window=30.0,
                  models=['turbo'], inputs=['original', 'demucs', 'melband'], language=args.language,
                  notice=baseline.NOTICE, asset_setup_seconds=0, asr_model_load_seconds=0,
                  audio_preparation=[], rows=[], errors=[])

    def save():
        report['elapsed_seconds'] = time.monotonic() - began
        report['gpu_peak_allocated_mib'] = torch.cuda.max_memory_allocated() / 2**20
        save_report(root, report)

    prepared, jobs = [], []
    for song in selected:
        try:
            item = baseline.prepare_audio(song, songs, root, 0, 30.0, args.threads, digest, versions)
            ref = baseline.reference_for(references, item['song'], item['sha256'])
            prepared.append((song, item, ref))
            report['audio_preparation'].append(dict(item, input='demucs'))
            jobs.extend((item, variant, label, ref, versions)
                        for variant, label in (('original', 'original'), ('vocals', 'demucs')))
        except Exception as error:
            report['errors'].append(dict(song=song.name, stage='demucs', error=str(error)))
            print(f'ERROR Demucs {song.name}: {error}', flush=True)
        save()
    demucs_backend._model = None
    gc.collect()
    torch.cuda.empty_cache()

    try:
        setup = time.monotonic()
        source, weights = melband_backend.ensure_assets(args.melband_cache)
        report['asset_setup_seconds'] = time.monotonic() - setup
        separator = melband_backend.Separator(source, weights)
        for song, parent, ref in prepared:
            try:
                item = prepare_melband(song, parent, root, separator, mel_versions, args.threads)
                report['audio_preparation'].append(dict(item, input='melband'))
                jobs.append((item, 'vocals', 'melband', ref, mel_versions))
            except Exception as error:
                report['errors'].append(dict(song=song.name, stage='melband', error=str(error)))
                print(f'ERROR Mel-Band {song.name}: {error}', flush=True)
            save()
        separator.release()
    except Exception as error:
        report['errors'].append(dict(stage='melband_setup', error=str(error)))
        print(f'ERROR Mel-Band setup: {error}', flush=True)
    gc.collect()
    torch.cuda.empty_cache()

    model = None

    def load_model():
        nonlocal model
        if model is None:
            started = time.monotonic()
            print('Whisper turbo: downloading/loading once on CUDA', flush=True)
            model = whisper.load_model('turbo', device='cuda', download_root=str(args.model_cache))
            torch.cuda.synchronize()
            report['asr_model_load_seconds'] = time.monotonic() - started
        return model

    languages = {}
    if args.language == song_language.PER_SONG:
        report['languages'] = {}
        for song, parent, ref in prepared:
            stems = {label: (item, variant) for item, variant, label, *_ in jobs if item['song'] == parent['song']}
            # Prefer the Mel-Band stem; Demucs only when Mel-Band separation failed.
            source_input = 'melband' if 'melband' in stems else 'demucs'
            item, variant = stems[source_input]
            try:
                detected = song_language.song_language(
                    root, Path(item['directory']) / f'{variant}.wav', item['hashes'][f'{variant}.wav'],
                    min(30.0, item['duration']), versions, source_input, load_model)
                languages[parent['song']] = detected['language']
                report['languages'][parent['song']] = detected
                print(f'Language {parent["song"]}: {detected["language"]} '
                      f'({detected["probability"]:.2f}, {source_input})', flush=True)
            except Exception as error:
                report['errors'].append(dict(song=parent['song'], stage='language', error=str(error)))
                print(f'ERROR language {parent["song"]}: {error}', flush=True)
            save()

    for item, variant, label, reference, job_versions in jobs:
        if args.language == song_language.PER_SONG:
            if item['song'] not in languages:
                continue  # Error already recorded; never fall back to a fixed language.
            language = languages[item['song']]
        else:
            language = args.language

        def attempt(language):
            identity = asr_identity(item, variant, digest, job_versions, language)
            path = root / 'transcripts' / f'{key_for(identity)}.json'
            stored = None if args.force else baseline.cached_json(path, identity)
            reused = stored is not None
            if stored is None:
                load_model()
                print(f'turbo | {item["song"]} | {label} | {language} | 30s', flush=True)
                started = time.monotonic()
                samples = decode(Path(item['directory']) / f'{variant}.wav', 16000,
                                 duration=identity['duration'])
                result = model.transcribe(samples, **baseline.decoding_options(language))
                torch.cuda.synchronize()
                stored = dict(status='complete', identity=identity, result=result,
                              elapsed_seconds=time.monotonic() - started)
                baseline.atomic_json(path, stored)
            return identity, path, stored, reused, describe_words(stored['result'], identity['duration'], reference)

        try:
            language, fallback_from, (identity, path, stored, reused, summary), seconds = song_language.with_fallback(
                attempt, language, args.language == song_language.PER_SONG)
            report['rows'].append(dict(song=item['song'], sha256=item['sha256'], input=label,
                model='turbo', window=30.0, actual_duration=identity['duration'], language=language,
                language_fallback_from=fallback_from,
                reference=reference, transcript=str(path.relative_to(root)), reused=reused,
                audio=str(Path(item['directory']) / f'{variant}.wav'),
                seconds_this_run=seconds, **summary))
            print(f'  {label}: first={summary["first_word_start"]}; words={summary["word_count"]}; '
                  f'{"cached" if reused else str(round(stored["elapsed_seconds"], 2)) + "s"}'
                  f'{"; fallback " + fallback_from + "->" + language if fallback_from else ""}', flush=True)
        except Exception as error:
            report['errors'].append(dict(song=item['song'], stage='asr', input=label, error=str(error)))
            print(f'ERROR ASR: {error}', flush=True)
            if model is None:
                break
        save()
    report['status'] = 'failed' if report['errors'] else 'complete'
    save()
    print(f'Comparison: {root / "comparison.tsv"}\nDetails: {root / "comparison-details.tsv"}\n'
          f'Report: {root / "comparison-report.json"}', flush=True)
    if report['errors']:
        raise SystemExit('Some comparison jobs failed; completed evidence retained. See report errors.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--songs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--song-like', action='append')
    group.add_argument('--all', action='store_true')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--language', default=song_language.PER_SONG,
                        help="'song' detects one language per recording; or a fixed Whisper code such as en")
    parser.add_argument('--references', type=Path)
    parser.add_argument('--model-cache', type=Path, default=Path('/workspace/cache/whisper'))
    parser.add_argument('--melband-cache', type=Path, default=Path('/workspace/cache/melband'))
    parser.add_argument('--force', action='store_true', help='Repeat ASR only; reuse verified stems')
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive')
    baseline.select_songs(args.songs.resolve(), args.song_like, args.all)
    root = baseline.experiment_root(args.songs, args.output)
    # Same lock as the original experiment: caches never have concurrent writers.
    with (root / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args, root)


if __name__ == '__main__':
    main()
