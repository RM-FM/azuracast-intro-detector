#!/usr/bin/env python3
"""Compare Qwen ASR/alignment on verified, cached first-30s audio. No cue writes."""
import argparse
import csv
import dataclasses
import datetime
import fcntl
import hashlib
import importlib.metadata
import io
import json
import math
from pathlib import Path
import re
import subprocess
import time

import whisper_experiment as baseline
import song_language
import stem_comparison as comparison

ASR_MODEL = 'Qwen/Qwen3-ASR-1.7B'
ALIGN_MODEL = 'Qwen/Qwen3-ForcedAligner-0.6B'
PROFILE = 'qwen-cached-30s-v1'
SOURCE_DIGEST = '165b67272f7eabb3e69e3dcb86ea8be2add08eb6c772b77834c8f1baf7aac750'
INPUTS = ('original', 'demucs', 'melband')


def finite(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def beneath(root, value):
    path = Path(value)
    path = (path if path.is_absolute() else root / path).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f'Path outside evidence directory: {value}')
    return path


def select_jobs(root, songs, source, patterns, inputs, references):
    """Use the report as an index, then verify manifests and actual audio hashes."""
    names = source.get('songs', [])
    if patterns:
        chosen = []
        for pattern in patterns:
            matches = [name for name in names if pattern.casefold() in name.casefold()]
            if len(matches) != 1:
                raise ValueError(f'{pattern!r}: expected one song, found {len(matches)}')
            if matches[0] not in chosen:
                chosen.append(matches[0])
        names = chosen
    if not names or len(set(names)) != len(names):
        raise ValueError('Missing or duplicated songs in comparison report')
    jobs, errors = [], []
    for song in names:
        try:
            song_path = beneath(songs, song)
            song_hash = baseline.sha256(song_path)
        except Exception as error:
            errors.append(dict(song=song, stage='preflight', error=str(error)))
            continue
        for variant in inputs:
            try:
                matches = [r for r in source['rows'] if r['song'] == song and r['input'] == variant]
                if len(matches) != 1:
                    raise ValueError(f'Expected one cached {variant} row, found {len(matches)}')
                row = matches[0]
                path = beneath(root, row['audio'])
                if not path.is_relative_to((root / 'audio').resolve()):
                    raise ValueError('Input must be a cached audio asset')
                manifest = json.loads((path.parent / 'audio.json').read_text())
                duration = row['actual_duration']
                identity = manifest['identity']
                if (row.get('window') != 30 or not finite(duration) or not 0 < duration <= 30.03
                        or manifest.get('status') != 'complete' or manifest.get('start') != 0
                        or identity.get('start') != 0 or identity.get('duration') != 30
                        or abs(manifest['duration'] - duration) > .03
                        or row['sha256'] != song_hash or manifest['sha256'] != song_hash
                        or identity['sha256'] != song_hash):
                    raise ValueError('Changed recording or incompatible 0–30s audio manifest')
                audio_hash = baseline.sha256(path)
                if audio_hash != manifest['hashes'].get(path.name):
                    raise ValueError('Cached audio checksum mismatch; rerun separator comparison')
                # Verify the old transcript too; do not present a stale Whisper
                # timestamp beside a newly replaced audio file.
                transcript = beneath(root, row['transcript'])
                old = json.loads(transcript.read_text())
                old_id = old.get('identity', {})
                if (old.get('status') != 'complete' or old_id.get('audio_hash') != audio_hash
                        or old_id.get('audio') != identity or old_id.get('requested_window') != 30
                        or old_id.get('model') != 'turbo'):
                    raise ValueError('Whisper transcript does not match the cached audio')
                reference = baseline.reference_for(references, song, song_hash)
                old_summary = comparison.describe_words(old['result'], duration, reference)
                jobs.append(dict(song=song, input=variant, audio=str(path), audio_hash=audio_hash,
                    sha256=song_hash, duration=duration, reference=reference,
                    whisper_first_word_s=old_summary['first_word_start'],
                    whisper_text=old['result'].get('text', ''),
                    whisper_flags=old_summary['quality_flags'],
                    whisper_language=old_id.get('options', {}).get('language')))
            except Exception as error:
                errors.append(dict(song=song, input=variant, stage='preflight', error=str(error)))
    return jobs, errors


def model_lock(root):
    """Resolve model branches once; later invocations retain these exact revisions."""
    path = root / 'models.json'
    if path.exists():
        locked = json.loads(path.read_text())
    else:
        from huggingface_hub import HfApi
        api = HfApi()
        locked = {name: api.model_info(name).sha for name in (ASR_MODEL, ALIGN_MODEL)}
    if set(locked) != {ASR_MODEL, ALIGN_MODEL} or any(
            not isinstance(v, str) or not re.fullmatch(r'[0-9a-f]{40}', v) for v in locked.values()):
        raise ValueError('Invalid Qwen model revision lock')
    if not path.exists():
        baseline.atomic_json(path, locked)
    return locked


def asr_identity(job, revisions, packages, language, source_digest, ffmpeg_version):
    return dict(profile=PROFILE, audio_sha256=job['audio_hash'], duration=job['duration'],
                start=0, model=ASR_MODEL, revision=revisions[ASR_MODEL], packages=packages,
                language=language, context='', max_new_tokens=512, do_sample=False,
                dtype='bfloat16', attention='sdpa', source_digest=source_digest,
                decoder=ffmpeg_version, sample_rate=16000)


def alignment_identity(asr_id, result, revisions):
    identity = dict(asr=asr_id, result=result, aligner=ALIGN_MODEL, revision=revisions[ALIGN_MODEL])
    language = song_language.aligner_language(result['language'])
    # Only a substituted word-splitting language changes the key; existing caches stay valid.
    if language != result['language']:
        identity['aligner_language'] = language
    return identity


def decode(job):
    import numpy as np
    command = ['ffmpeg', '-v', 'error', '-nostdin', '-i', job['audio'], '-t',
               str(min(30.0, job['duration'])), '-ac', '1', '-ar', '16000', '-f', 'f32le', '-']
    samples = np.frombuffer(subprocess.check_output(command), dtype='<f4').copy()
    if not samples.size or not np.isfinite(samples).all():
        raise ValueError('Empty/non-finite input audio')
    if abs(samples.size / 16000 - job['duration']) > .03:
        raise ValueError('Decoded duration differs from the cached comparison')
    # No silence trimming or volume boosting. Qwen's own input adapter may
    # scale peaks above 1.0; record this rather than concealing preprocessing.
    return samples, dict(decoded_seconds=len(samples) / 16000,
                         input_peak=float(np.max(np.abs(samples))))


def summarize(result, alignment, duration, reference):
    words = [{'word': ' ' + item['text'], 'start': item['start_time'], 'end': item['end_time']}
             for item in (alignment or {}).get('items', [])]
    summary = comparison.describe_words({'segments': [{'words': words}]}, duration, reference)
    flags = summary['quality_flags']
    if not result['text'].strip():
        flags.append('empty_transcript_inconclusive')
    elif not words:
        flags.append('text_without_alignment')
    if alignment is not None and alignment.get('error'):
        flags.append('alignment_failed')
    previous = -1.0
    for word in words:
        start, end = word['start'], word['end']
        if finite(start) and finite(end):
            if start < previous:
                flags.append('nonmonotonic_word_timing')
            previous = end
    # A later valid word must not silently replace an invalid opening word.
    if any(f in flags for f in ('invalid_word_timing', 'nonmonotonic_word_timing')):
        summary['first_word_start'] = None
        summary['first_word_minus_reference_seconds'] = None
    summary['quality_flags'] = sorted(set(flags))
    summary['text'] = result['text']
    summary['language'] = result['language']
    summary['evidence'] = ('transcribed_text_unvalidated' if result['text'].strip()
                           else 'no_transcribed_words_inconclusive')
    return summary


def save_report(root, report):
    baseline.atomic_json(root / 'report.json', report)
    baseline.atomic_json(root / 'runs' / f'{report["run_id"]}.json', report)
    columns = ['song', 'input', 'whisper_first_word_s', 'qwen_first_word_s', 'reference_s',
               'delta_s', 'asr_s', 'alignment_s', 'asr_cached', 'alignment_cached',
               'flags', 'qwen_text', 'whisper_text']
    stream = io.StringIO(newline='')
    writer = csv.writer(stream, delimiter='\t')
    writer.writerow(columns)
    for row in report['rows']:
        ref = row.get('reference') or {}
        writer.writerow([row['song'], row['input'], row['whisper_first_word_s'],
            row['first_word_start'], ref.get('vocal_start') if ref.get('audio_matches') else None,
            row['first_word_minus_reference_seconds'], round(row['asr_seconds_this_run'], 3),
            round(row['alignment_seconds_this_run'], 3), row['asr_cached'], row['alignment_cached'],
            ','.join(row['quality_flags']) or 'none', row['text'], row['whisper_text']])
    temporary = root / 'comparison.tmp.tsv'
    temporary.write_text(stream.getvalue())
    temporary.replace(root / 'comparison.tsv')


class QwenRunner:
    """Lazy, resident models; keep transcription if alignment later fails."""
    def __init__(self, revisions):
        self.revisions = revisions
        self.asr = self.aligner = None
        self.load_seconds = {'asr': 0.0, 'aligner': 0.0}

    def load(self, name):
        import torch
        from huggingface_hub import snapshot_download
        from qwen_asr import Qwen3ASRModel, Qwen3ForcedAligner
        is_asr = name == 'asr'
        repo = ASR_MODEL if is_asr else ALIGN_MODEL
        began = time.monotonic()
        print(f'Loading {repo} once (downloading if absent)', flush=True)
        local = snapshot_download(repo, revision=self.revisions[repo],
                                  ignore_patterns=['*.bin', '*.pt', '*.pth'])
        kwargs = dict(dtype=torch.bfloat16, device_map='cuda:0', attn_implementation='sdpa')
        if is_asr:
            self.asr = Qwen3ASRModel.from_pretrained(local, max_inference_batch_size=1,
                                                    max_new_tokens=512, **kwargs)
            self.asr.model.generation_config.do_sample = False
        else:
            self.aligner = Qwen3ForcedAligner.from_pretrained(local, **kwargs)
        torch.cuda.synchronize()
        self.load_seconds[name] += time.monotonic() - began

    def transcribe(self, samples, language):
        if self.asr is None:
            self.load('asr')
        import torch
        began = time.monotonic()
        result = self.asr.transcribe(audio=(samples, 16000), context='', language=language,
                                     return_time_stamps=False)[0]
        torch.cuda.synchronize()
        return dict(language=result.language, text=result.text), time.monotonic() - began

    def align(self, samples, result):
        if not result['text'].strip():
            return {'items': []}, 0.0
        if self.aligner is None:
            self.load('aligner')
        import torch
        began = time.monotonic()
        aligned = self.aligner.align(audio=(samples, 16000), text=result['text'],
                                     language=song_language.aligner_language(result['language']))[0]
        torch.cuda.synchronize()
        return dataclasses.asdict(aligned), time.monotonic() - began


def process_job(job, root, runner, revisions, packages, language, digest, decoder,
                force=False, force_alignment=False):
    if language == song_language.PER_SONG:
        # Same language as this input's Whisper transcript; None lets Qwen detect it.
        language = song_language.qwen_language(job.get('whisper_language'))
    identity = asr_identity(job, revisions, packages, language, digest, decoder)
    asr_path = root / 'transcripts' / f'{comparison.key_for(identity)}.json'
    stored = None if force else baseline.cached_json(asr_path, identity)
    reused_asr = stored is not None
    samples = None
    if stored is None:
        samples, audio_info = decode(job)
        result, seconds = runner.transcribe(samples, language)
        stored = dict(status='complete', identity=identity, result=result,
                      elapsed_seconds=seconds, audio_info=audio_info)
        baseline.atomic_json(asr_path, stored)
    result = stored['result']
    align_id = alignment_identity(identity, result, revisions)
    align_path = root / 'alignments' / f'{comparison.key_for(align_id)}.json'
    aligned = None if force or force_alignment else baseline.cached_json(align_path, align_id)
    reused_align = aligned is not None
    alignment_error = None
    if aligned is None:
        began = time.monotonic()
        try:
            if samples is None and result['text'].strip():
                samples, _ = decode(job)
            output, seconds = runner.align(samples, result)
            aligned = dict(status='complete', identity=align_id, result=output,
                           elapsed_seconds=seconds)
            baseline.atomic_json(align_path, aligned)
        except Exception as error:
            alignment_error = str(error)
            aligned = dict(result={'items': [], 'error': alignment_error},
                           elapsed_seconds=time.monotonic() - began)
            # Save failure evidence separately; failed alignment is retried.
            baseline.atomic_json(root / 'runs' / f'failed-alignment-{comparison.key_for(align_id)}.json', aligned)
            # A CUDA OOM is a process-level failure, not an ordinary bad span.
            if type(error).__name__ == 'OutOfMemoryError':
                raise
    summary = summarize(result, aligned['result'], job['duration'], job['reference'])
    if job.get('whisper_language') and language != song_language.qwen_language(job['whisper_language']):
        summary['quality_flags'].append('different_asr_language_settings')
    return dict(**job, **summary, qwen_language=language, asr_cached=reused_asr, alignment_cached=reused_align,
                asr_seconds_this_run=0 if reused_asr else stored['elapsed_seconds'],
                alignment_seconds_this_run=0 if reused_align else aligned['elapsed_seconds'],
                audio_info=stored['audio_info'], alignment_error=alignment_error,
                transcript=str(asr_path.relative_to(root)),
                alignment=str(align_path.relative_to(root)) if not alignment_error else None)


def run(args, root):
    began = time.monotonic()
    source_path = getattr(args, 'source_report', None) or args.evidence / 'comparison-report.json'
    source = json.loads(source_path.read_text())
    references = json.loads(args.references.read_text()) if args.references else {}
    jobs, errors = select_jobs(args.evidence, args.songs, source, args.song_like, args.inputs, references)
    report = dict(run_id=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'),
                  status='running', notice=baseline.NOTICE, profile=PROFILE,
                  source_report_sha256=baseline.sha256(source_path),
                  language=args.language, rows=[], errors=errors, model_load_seconds={},
                  timestamp_notice='Forced alignment is not independent confirmation of vocals.')
    def save():
        report['elapsed_seconds'] = time.monotonic() - began
        save_report(root, report)
    save()
    if not jobs:
        report['status'] = 'failed'
        save()
        raise SystemExit('No valid cached audio. See qwen/report.json preflight errors.')
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; CPU fallback disabled')
    torch.set_num_threads(args.threads)
    torch.cuda.reset_peak_memory_stats()
    revisions = model_lock(root)
    packages = {name: importlib.metadata.version(name) for name in
                ('qwen-asr', 'torch', 'transformers', 'numpy', 'librosa', 'soundfile', 'accelerate')}
    # Pinned cache key: the source digest that produced the existing Qwen evidence.
    # The per-song language is part of the identity itself, so routing it does not
    # require new ASR. Change PROFILE whenever transcription or alignment behaviour changes.
    digest = SOURCE_DIGEST
    decoder = subprocess.check_output(['ffmpeg', '-version'], text=True).splitlines()[0]
    report.update(models=revisions, packages=packages, source_digest=digest,
                  source_files_sha256=hashlib.sha256(b''.join(Path(__file__).with_name(name).read_bytes()
                      for name in ('qwen_experiment.py', 'whisper_experiment.py', 'stem_comparison.py',
                                   'song_language.py'))).hexdigest(),
                  gpu=torch.cuda.get_device_name(0), decoder=decoder)
    runner = QwenRunner(revisions)
    for number, job in enumerate(jobs, 1):
        print(f'[{number}/{len(jobs)}] {job["song"]} | {job["input"]}', flush=True)
        try:
            row = process_job(job, root, runner, revisions, packages, args.language, digest, decoder,
                              args.force, args.force_alignment)
            report['rows'].append(row)
            if row['alignment_error']:
                report['errors'].append(dict(song=job['song'], input=job['input'], stage='alignment',
                                             error=row['alignment_error']))
            print(f'  first={row["first_word_start"]}; ASR={row["asr_seconds_this_run"]:.2f}s; '
                  f'align={row["alignment_seconds_this_run"]:.2f}s; text={row["text"]!r}', flush=True)
        except Exception as error:
            report['errors'].append(dict(song=job['song'], input=job['input'], stage='asr', error=str(error)))
            print(f'  ERROR: {error}', flush=True)
            if isinstance(error, torch.cuda.OutOfMemoryError):
                save()
                raise
        report['model_load_seconds'] = runner.load_seconds
        report['gpu_peak_allocated_mib'] = torch.cuda.max_memory_allocated() / 2**20
        save()
    report['status'] = 'failed' if report['errors'] else 'complete'
    save()
    print(f'Comparison: {root / "comparison.tsv"}\nReport: {root / "report.json"}', flush=True)
    if report['errors']:
        raise SystemExit('Some jobs failed; completed text/alignment caches retained. See report errors.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--source-report', type=Path, help='Alternative verified comparison manifest')
    parser.add_argument('--songs', type=Path, required=True)
    parser.add_argument('--references', type=Path)
    parser.add_argument('--song-like', action='append', help='Select one unambiguous song per flag; default all')
    parser.add_argument('--inputs', nargs='+', choices=INPUTS, default=list(INPUTS[:3]))
    parser.add_argument('--language', default=song_language.PER_SONG,
                        help="'song' follows each input's Whisper language; a full Qwen language name; or auto")
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--force', action='store_true', help='Repeat Qwen ASR and alignment; retain audio')
    parser.add_argument('--force-alignment', action='store_true', help='Reuse Qwen text; repeat alignment only')
    args = parser.parse_args()
    if args.threads < 1 or len(set(args.inputs)) != len(args.inputs):
        parser.error('Threads must be positive; inputs must be unique')
    if args.language.lower() == 'auto':
        args.language = None
    elif args.language.lower() != song_language.PER_SONG:
        args.language = args.language.capitalize()
    args.evidence, args.songs = args.evidence.resolve(), args.songs.resolve()
    if args.evidence.is_relative_to(args.songs) or args.songs.is_relative_to(args.evidence):
        parser.error('Audio input and evidence directories must be separate')
    root = args.evidence / 'qwen'
    # Lock with the separator/Whisper experiments before reading any manifests.
    with (args.evidence / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for directory in (root, root / 'transcripts', root / 'alignments', root / 'runs'):
            if directory.is_symlink():
                raise ValueError('Experiment output directories cannot be symlinks')
            directory.mkdir(exist_ok=True)
        try:
            run(args, root)
        except (Exception, KeyboardInterrupt) as error:
            report_path = root / 'report.json'
            if report_path.exists():
                report = json.loads(report_path.read_text())
                report['status'] = 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed'
                report['errors'].append(dict(stage='run', error=str(error) or type(error).__name__))
                save_report(root, report)
            raise


if __name__ == '__main__':
    main()
