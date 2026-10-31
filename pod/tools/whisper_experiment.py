#!/usr/bin/env python3
"""Isolated CUDA ASR evidence comparison. Never writes or approves playout cues."""
import argparse
import csv
import datetime
import fcntl
import gc
import hashlib
import importlib.metadata
import json
import math
import pathlib
import time

EXTENSIONS = {'.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac', '.opus'}
DEFAULT_CASES = ('Thunderstruck', 'Teenage Dirtbag', "Don't Lose Your Head", 'Get Married')
NOTICE = ('ASR evidence only. Empty transcription does not establish absence of vocals; '
          'returned words and timestamps may be hallucinated. No playout cue is produced.')


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, data):
    temporary = path.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def cached_json(path, identity):
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return data if data.get('identity') == identity and data.get('status') == 'complete' else None


def select_songs(songs, patterns, all_songs=False):
    files = sorted(p for p in songs.rglob('*') if p.is_file() and p.suffix.lower() in EXTENSIONS)
    if not files:
        raise ValueError('No audio files found')
    if all_songs:
        return files
    selected = []
    for pattern in patterns or DEFAULT_CASES:
        matches = [p for p in files if pattern.casefold() in p.relative_to(songs).as_posix().casefold()]
        if len(matches) != 1:
            raise ValueError(f'{pattern!r}: expected one song, found {len(matches)}; use --song-like or --all')
        if matches[0] not in selected:
            selected.append(matches[0])
    return selected


def reference_for(references, song, fingerprint):
    reference = references.get(song)
    if not reference:
        return None
    if reference.get('sha256') != fingerprint:
        return {'audio_matches': False, 'reason': 'Different recording; reference not compared'}
    return {**reference, 'audio_matches': True}


def describe_words(result, offset, duration, reference=None):
    """Preserve all model output separately; only summarize valid timed words here."""
    words, flags = [], []
    for segment in result.get('segments', []):
        for key, predicate in (
            ('avg_logprob', lambda v: v < -1),
            ('no_speech_prob', lambda v: v > .6),
            ('compression_ratio', lambda v: v > 2.4),
        ):
            value = segment.get(key)
            if isinstance(value, (float, int)) and predicate(value):
                flags.append(key)
        for word in segment.get('words', []):
            start, end = word.get('start'), word.get('end')
            if not word.get('word', '').strip():
                continue
            if (not isinstance(start, (float, int)) or not isinstance(end, (float, int))
                    or not math.isfinite(start) or not math.isfinite(end)
                    or not 0 <= start < end <= duration + .02):
                flags.append('invalid_word_timing')
                continue
            words.append({**word, 'start_original': round(offset + start, 4),
                          'end_original': round(offset + end, 4)})
    words.sort(key=lambda w: w['start_original'])
    first = words[0]['start_original'] if words else None
    reference_start = reference.get('vocal_start') if reference and reference.get('audio_matches') else None
    return {
        'evidence': 'transcribed_words' if words else 'no_transcribed_words_inconclusive',
        'first_word_start': first, 'word_count': len(words), 'words': words,
        'quality_flags': sorted(set(flags)),
        'reference_in_window': bool(reference_start is not None and offset <= reference_start < offset + duration),
        'first_word_minus_reference_seconds': round(first - reference_start, 4)
        if first is not None and reference_start is not None and offset <= reference_start < offset + duration else None,
        'notice': NOTICE,
    }


def decoding_options(language):
    return dict(language=None if language == 'auto' else language, task='transcribe',
                word_timestamps=True, temperature=0.0, beam_size=5, fp16=True,
                condition_on_previous_text=False, initial_prompt=None,
                # Investigate quiet/vocoder vocals without suppressing them using a
                # speech-only VAD or Whisper's no-speech decision. Keep its metrics.
                no_speech_threshold=None, logprob_threshold=-1.0,
                compression_ratio_threshold=2.4, hallucination_silence_threshold=None,
                verbose=None)


def experiment_root(songs, output):
    songs, output = songs.resolve(), output.resolve()
    if output == songs or output.is_relative_to(songs) or songs.is_relative_to(output):
        raise ValueError('Input and output must be separate directory trees')
    root = output / 'whisper-evidence'
    if root.is_symlink():
        raise ValueError('Experiment root cannot be a symlink')
    root.mkdir(parents=True, exist_ok=True)
    for name in ('audio', 'transcripts', 'runs'):
        directory = root / name
        if directory.is_symlink():
            raise ValueError('Experiment subdirectories cannot be symlinks')
        directory.mkdir(exist_ok=True)
    return root


def packages():
    return {name: importlib.metadata.version(name)
            for name in ('openai-whisper', 'torch', 'torchaudio', 'demucs', 'numpy')}


def prepare_audio(song, songs, root, start, duration, threads, source_digest, package_versions):
    from demucs_backend import PROFILE, separate
    from model_worker import decode
    import soundfile as sf
    fingerprint = sha256(song)
    identity = dict(sha256=fingerprint, start=start, duration=duration,
                    demucs_profile=PROFILE, source_digest=source_digest, packages=package_versions)
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    directory = root / 'audio' / key
    directory.mkdir(exist_ok=True)
    manifest = directory / 'audio.json'
    cached = cached_json(manifest, identity)
    original, vocals = directory / 'original.wav', directory / 'vocals.wav'
    if cached and all(p.is_file() and sha256(p) == cached['hashes'].get(p.name) for p in (original, vocals)):
        return {**cached, 'song': song.relative_to(songs).as_posix(),
                'directory': str(directory), 'reused': True, 'prepare_seconds_this_run': 0}
    began = time.monotonic()
    print(f'Preparing {song.name}: {start:g}–{start + duration:g}s', flush=True)
    samples = decode(song, 16000, start=start, duration=duration)
    if not len(samples):
        raise ValueError(f'Empty audio: {song}')
    sf.write(original, samples, 16000, subtype='FLOAT')
    separation = separate(dict(song=str(song), left=start, right=start + duration,
                               stem=str(vocals), threads=threads), decode)
    # Downstream ASR sees the same real duration for both variants, with no
    # normalization, silence trimming or stitching from overlapping old stems.
    available = min(len(samples) / 16000, sf.info(vocals).duration)
    data = dict(status='complete', identity=identity, song=song.relative_to(songs).as_posix(),
                sha256=fingerprint, start=start, duration=available, separation=separation,
                hashes={p.name: sha256(p) for p in (original, vocals)},
                preparation_seconds=time.monotonic() - began)
    atomic_json(manifest, data)
    return {**data, 'directory': str(directory), 'reused': False,
            'prepare_seconds_this_run': data['preparation_seconds']}


def run(args, root):
    import torch
    import whisper
    import demucs_backend
    from model_worker import decode
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; CPU fallback is disabled')
    torch.set_num_threads(args.threads)
    versions = packages()
    source_digest = hashlib.sha256(b''.join(pathlib.Path(__file__).with_name(name).read_bytes()
        for name in ('whisper_experiment.py', 'demucs_backend.py', 'model_worker.py'))).hexdigest()
    references = json.loads(args.references.read_text()) if args.references else {}
    songs = args.songs.resolve()
    selected = select_songs(songs, args.song_like, args.all)
    run_id = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    report_path = root / 'runs' / f'{run_id}.json'
    report = dict(run_id=run_id, status='running', gpu=torch.cuda.get_device_name(0), packages=versions,
                  source_digest=source_digest, windows=args.windows, start=args.start,
                  models=args.models, language=args.language, notice=NOTICE,
                  model_load_seconds={}, audio_preparation=[], rows=[], errors=[])
    began = time.monotonic()

    def save():
        report['elapsed_seconds'] = time.monotonic() - began
        report['fresh_asr_jobs'] = sum(not r['reused'] for r in report['rows'])
        report['reused_asr_jobs'] = sum(r['reused'] for r in report['rows'])
        report['asr_seconds_this_run'] = sum(r['seconds_this_run'] for r in report['rows'])
        report['gpu_peak_allocated_mib'] = torch.cuda.max_memory_allocated() / 2**20
        atomic_json(report_path, report)
        atomic_json(root / 'report.json', report)
        with (root / 'summary.tsv').open('w', newline='') as stream:
            writer = csv.writer(stream, delimiter='\t')
            writer.writerow(['song', 'model', 'input', 'window_s', 'first_word_s',
                             'reference_s', 'word_count', 'seconds_this_run', 'cached', 'quality_flags'])
            for row in report['rows']:
                reference = row.get('reference') or {}
                writer.writerow([row['song'], row['model'], row['variant'], row['window'],
                                 row['first_word_start'], reference.get('vocal_start') if reference.get('audio_matches') else None,
                                 row['word_count'], round(row['seconds_this_run'], 2), row['reused'],
                                 ','.join(row['quality_flags'])])

    prepared = []
    for song in selected:
        try:
            item = prepare_audio(song, songs, root, args.start, max(args.windows), args.threads,
                                 source_digest, versions)
            item['reference'] = reference_for(references, item['song'], item['sha256'])
            prepared.append(item)
            report['audio_preparation'].append(item)
        except Exception as error:
            report['errors'].append(dict(song=song.name, stage='prepare', error=str(error)))
            print(f'ERROR preparing {song.name}: {error}', flush=True)
        save()
    # Separation is finished for the entire selection. Free its GPU weights
    # before loading one Whisper model, which then stays resident across jobs.
    demucs_backend._model = None
    gc.collect()
    torch.cuda.empty_cache()
    for model_name in args.models:
        model = None
        failed_model = False
        for item in prepared:
            for window in args.windows:
                duration = min(window, item['duration'])
                for variant in ('original', 'vocals'):
                    identity = dict(audio=item['identity'], audio_hash=item['hashes'][variant + '.wav'],
                                    variant=variant, duration=duration, requested_window=window,
                                    model=model_name, options=decoding_options(args.language),
                                    source_digest=source_digest, packages=versions)
                    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
                    path = root / 'transcripts' / f'{key}.json'
                    stored = None if args.force else cached_json(path, identity)
                    reused = stored is not None
                    try:
                        if stored is None:
                            if model is None:
                                load_began = time.monotonic()
                                print(f'Whisper {model_name}: downloading/loading on CUDA (once)', flush=True)
                                model = whisper.load_model(model_name, device='cuda', download_root=str(args.model_cache))
                                torch.cuda.synchronize()
                                report['model_load_seconds'][model_name] = time.monotonic() - load_began
                            print(f'{model_name} | {item["song"]} | {variant} | {window:g}s', flush=True)
                            inference_began = time.monotonic()
                            samples = decode(pathlib.Path(item['directory']) / f'{variant}.wav', 16000, duration=duration)
                            result = model.transcribe(samples, **decoding_options(args.language))
                            torch.cuda.synchronize()
                            stored = dict(status='complete', identity=identity, result=result,
                                          elapsed_seconds=time.monotonic() - inference_began)
                            atomic_json(path, stored)
                        summary = describe_words(stored['result'], args.start, duration, item['reference'])
                        row = dict(song=item['song'], sha256=item['sha256'], model=model_name, variant=variant,
                                   window=window, actual_duration=duration, reference=item['reference'],
                                   transcript=str(path.relative_to(root)), reused=reused,
                                   seconds_this_run=0 if reused else stored['elapsed_seconds'], **summary)
                        report['rows'].append(row)
                        first = summary['first_word_start']
                        print(f'  first word: {first if first is not None else "none (inconclusive)"}; '
                              f'{summary["word_count"]} words; '
                              f'{"cached" if reused else str(round(stored["elapsed_seconds"], 2)) + "s"}', flush=True)
                    except Exception as error:
                        report['errors'].append(dict(song=item['song'], model=model_name, variant=variant,
                                                     window=window, error=str(error)))
                        print(f'ERROR: {error}', flush=True)
                        # A failed model load should not retry a large download for every item.
                        if model is None:
                            failed_model = True
                    save()
                    if failed_model:
                        break
                if failed_model:
                    break
            if failed_model:
                break
        del model
        gc.collect()
        torch.cuda.empty_cache()
    report['status'] = 'failed' if report['errors'] else 'complete'
    save()
    print(f'Report: {root / "report.json"}\nSummary: {root / "summary.tsv"}\nArchived run: {report_path}', flush=True)
    if report['errors']:
        raise SystemExit('Some experiments failed; successful results retained. See report errors.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--songs', type=pathlib.Path, required=True)
    parser.add_argument('--output', type=pathlib.Path, required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--song-like', action='append', help='Repeat for each selected recording')
    group.add_argument('--all', action='store_true', help='Test bounded excerpts of every song')
    parser.add_argument('--models', nargs='+', choices=['large-v3', 'turbo'], default=['large-v3'])
    parser.add_argument('--windows', nargs='+', type=float, default=[10, 30])
    parser.add_argument('--start', type=float, default=0)
    parser.add_argument('--language', default='en', help='Known language, or auto; default en for test songs')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--references', type=pathlib.Path)
    parser.add_argument('--model-cache', type=pathlib.Path, default=pathlib.Path('/workspace/cache/whisper'))
    parser.add_argument('--force', action='store_true', help='Repeat ASR; retains identical separated audio')
    args = parser.parse_args()
    if (not args.songs.is_dir() or args.threads < 1 or not math.isfinite(args.start) or args.start < 0
            or any(not math.isfinite(w) or not 0 < w <= 30 for w in args.windows)):
        parser.error('Use an existing songs folder, positive threads, nonnegative start and windows <=30s')
    args.windows = sorted(set(args.windows))
    args.models = list(dict.fromkeys(args.models))
    # Validate selection before downloading any weights or loading CUDA models.
    try:
        select_songs(args.songs.resolve(), args.song_like, args.all)
        root = experiment_root(args.songs, args.output)
    except ValueError as error:
        parser.error(str(error))
    with (root / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args, root)


if __name__ == '__main__':
    main()
