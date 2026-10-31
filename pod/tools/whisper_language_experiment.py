#!/usr/bin/env python3
"""Compare multilingual Whisper language auto-detection on the cached four inputs."""
import argparse
import csv
import datetime
import fcntl
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path

import whisper_experiment as base
import stem_comparison as stems
import qwen_experiment as qwen
from ultimate_inference import INPUTS, read

PROFILE = 'whisper-turbo-auto-language-30s-v1'


def run(args):
    import torch
    import whisper
    from model_worker import decode

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; CPU fallback disabled')
    torch.set_num_threads(args.threads)
    root = args.evidence / 'ultimate'
    source = read(root / 'matrix.json')
    names = source.get('songs', [])
    if len(names) != args.expected_count or len(set(names)) != args.expected_count:
        raise ValueError(f'Expected {args.expected_count} unique songs in the verified matrix; found {len(names)}')
    jobs, errors = qwen.select_jobs(args.evidence, args.songs, source, None, INPUTS, {})
    if errors or len(jobs) != args.expected_count * len(INPUTS):
        raise ValueError(f'Expected {args.expected_count * len(INPUTS)} verified cached inputs; got {len(jobs)} jobs and {len(errors)} errors')
    versions = {name: importlib.metadata.version(name)
                for name in ('openai-whisper', 'torch', 'numpy', 'torchaudio')}
    source_digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    out_dir = root / 'language'
    out_dir.mkdir(exist_ok=True)
    report = dict(status='running', profile=PROFILE, model='turbo', language='auto',
        input_window_seconds=30, expected_songs=args.expected_count, inputs=list(INPUTS),
        gpu=torch.cuda.get_device_name(0), packages=versions, source_sha256=source_digest,
        model_load_seconds=0, rows=[], errors=[])
    report_path = out_dir / 'report.json'
    started = time.monotonic()

    def save():
        report['elapsed_seconds'] = time.monotonic() - started
        base.atomic_json(report_path, report)
        temp = out_dir / 'comparison.tmp.tsv'
        with temp.open('w', newline='') as stream:
            writer = csv.writer(stream, delimiter='\t')
            writer.writerow(['song', 'input', 'detected_language', 'first_word_s', 'word_count',
                             'quality_flags', 'seconds_this_run', 'cached', 'text'])
            for row in report['rows']:
                writer.writerow([row['song'], row['input'], row['language'], row['first_word_start'],
                    row['word_count'], ','.join(row['quality_flags']) or 'none',
                    round(row['seconds_this_run'], 3), row['reused'], row['text']])
        temp.replace(out_dir / 'comparison.tsv')

    model = None
    for index, job in enumerate(jobs, 1):
        identity = dict(profile=PROFILE, song=job['song'], sha256=job['sha256'],
            input=job['input'], audio_hash=job['audio_hash'], duration=job['duration'],
            model='turbo', options=base.decoding_options('auto'), packages=versions,
            source_sha256=source_digest)
        key = stems.key_for(identity)
        target = out_dir / f'{key}.json'
        stored = base.cached_json(target, identity)
        reused = stored is not None
        try:
            if stored is None:
                if model is None:
                    load_started = time.monotonic()
                    print('Whisper turbo multilingual: loading once on CUDA', flush=True)
                    model = whisper.load_model('turbo', device='cuda', download_root=str(args.model_cache))
                    torch.cuda.synchronize()
                    report['model_load_seconds'] = time.monotonic() - load_started
                print(f'Whisper auto [{index}/{len(jobs)}] {job["song"]} | {job["input"]}', flush=True)
                samples = decode(Path(job['audio']), 16000, duration=job['duration'])
                inference_started = time.monotonic()
                result = model.transcribe(samples, **base.decoding_options('auto'))
                torch.cuda.synchronize()
                stored = dict(status='complete', identity=identity, result=result,
                              elapsed_seconds=time.monotonic() - inference_started)
                base.atomic_json(target, stored)
            summary = stems.describe_words(stored['result'], job['duration'], None)
            report['rows'].append(dict(song=job['song'], sha256=job['sha256'], input=job['input'],
                audio_hash=job['audio_hash'], transcript=str(target.relative_to(args.evidence)),
                language=stored['result'].get('language'), text=stored['result'].get('text', ''),
                reused=reused, seconds_this_run=0 if reused else stored.get('elapsed_seconds', 0), **summary))
        except Exception as error:
            report['errors'].append(dict(song=job['song'], input=job['input'], error=str(error)))
        save()
    del model
    torch.cuda.empty_cache()
    report['status'] = 'failed' if report['errors'] or len(report['rows']) != len(jobs) else 'complete'
    save()
    print(f'Whisper language report: {report_path}', flush=True)
    if report['status'] != 'complete':
        raise SystemExit('Whisper auto-language stage incomplete; inspect language/report.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--songs', type=Path, default=Path('/workspace/input/songs'))
    parser.add_argument('--evidence', type=Path, default=Path('/workspace/experiments/whisper-evidence'))
    parser.add_argument('--model-cache', type=Path, default=Path('/workspace/cache/whisper'))
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--expected-count', type=int, default=40)
    args = parser.parse_args()
    if args.threads < 1 or args.expected_count < 1:
        parser.error('threads and expected count must be positive')
    args.songs, args.evidence = args.songs.resolve(), args.evidence.resolve()
    if not args.songs.is_dir() or args.songs.is_relative_to(args.evidence) or args.evidence.is_relative_to(args.songs):
        parser.error('Songs and evidence must be separate existing directory trees')
    args.evidence.mkdir(parents=True, exist_ok=True)
    with (args.evidence / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args)


if __name__ == '__main__':
    main()
