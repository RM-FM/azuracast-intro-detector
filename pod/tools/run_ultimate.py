#!/usr/bin/env python3
"""One resumable comparison. Processes release GPU models between stages."""
import argparse
import datetime
import os
from pathlib import Path
import subprocess
import time

import whisper_experiment as base
import song_language


def language_args(args):
    """Whisper and Qwen language options for one ASR language policy."""
    language = getattr(args, 'asr_language', song_language.PER_SONG)
    if language == song_language.PER_SONG:
        return [language], [language]
    qwen = song_language.qwen_language(language)
    if qwen is None:
        raise ValueError(f'Qwen cannot force language {language!r}')
    return [language], [qwen]


def commands(args):
    tools = Path(__file__).resolve().parent
    common = ['--songs', str(args.songs), '--evidence', str(args.evidence)]
    whisper_language, qwen_language = language_args(args)
    return [
        ('demucs-melband-whisper', [args.stem_python, str(tools/'stem_comparison.py'), '--all',
                                  '--songs', str(args.songs), '--output', str(args.evidence.parent),
                                  '--language', *whisper_language]),
        ('matrix', [args.stem_python, str(tools/'ultimate_inference.py'), 'matrix', *common]),
        ('musical-timing', [args.stem_python, str(tools/'musical_intro.py'), *common]),
        ('panns-levels', [args.stem_python, str(tools/'ultimate_inference.py'), 'acoustic', *common]),
        ('qwen', [args.qwen_python, str(tools/'qwen_experiment.py'), *common,
                  '--source-report', str(args.evidence/'ultimate/matrix.json'),
                  '--inputs', 'original', 'demucs', 'melband', '--language', *qwen_language]),
        ('whisper-auto-language', [args.stem_python, str(tools/'whisper_language_experiment.py'), *common,
                                   '--model-cache', str(args.whisper_cache)]),
        ('smad-singing-classifier', [args.qwen_python, str(tools/'smad_experiment.py'), *common]),
        ('evaluate', [args.stem_python, str(tools/'intro_benchmark.py'), *common]),
    ]


def stages_for(args, selected):
    return [(name, command + ['--expected-count', str(len(selected))]
             if name in ('whisper-auto-language', 'smad-singing-classifier') else command)
            for name, command in commands(args)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stem-python', default='/workspace/stem-comparison-venv/bin/python')
    parser.add_argument('--qwen-python', default='/workspace/qwen-asr-venv/bin/python')
    parser.add_argument('--songs', type=Path, default=Path('/workspace/input/songs'))
    parser.add_argument('--evidence', type=Path, default=Path('/workspace/experiments/whisper-evidence'))
    parser.add_argument('--whisper-cache', type=Path, default=Path('/workspace/cache/whisper'))
    parser.add_argument('--asr-language', default=song_language.PER_SONG,
                        help="'song' (default): one language per recording, detected on its vocal stem. "
                             "A Whisper code such as en reproduces a fixed-language run.")
    parser.add_argument('--plan', action='store_true', help='Print stage commands without starting inference')
    args = parser.parse_args()
    if args.evidence.name != 'whisper-evidence':
        parser.error('Evidence basename must be whisper-evidence for baseline cache compatibility')
    try:
        language_args(args)
    except ValueError as error:
        parser.error(str(error))
    if args.plan:
        import shlex
        for name, command in commands(args):
            print(name+': '+shlex.join(command))
        return
    root = base.experiment_root(args.songs, args.evidence.parent)/'ultimate'
    root.mkdir(exist_ok=True)
    (root/'runs').mkdir(exist_ok=True)
    selected = base.select_songs(args.songs.resolve(), None, True)
    if not selected:
        parser.error('No recordings found')
    print(f'{len(selected)} fingerprinted recordings; first 30s; stem, acoustic, multilingual ASR and singing-classifier checks; resumable caches.', flush=True)
    stages = stages_for(args, selected)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    report = dict(run_id=stamp, launcher_run_id=os.environ.get('POC_RUN_ID'),
                  song_count=len(selected), total_stages=len(stages), status='running', stages=[],
                  notice='Wall time includes cache verification and subprocess startup; excludes pip/provisioning/transfers.')
    began = time.monotonic()
    def save():
        report['elapsed_seconds'] = time.monotonic()-began
        base.atomic_json(root/'run.json', report)
        base.atomic_json(root/'runs'/f'run-{stamp}.json', report)
    try:
        for name, command in stages:
            print(f'\n=== {name} ===', flush=True)
            stage = dict(stage=name, status='running', command=command)
            report['stages'].append(stage)
            save()
            started = time.monotonic()
            result = subprocess.run(command, check=False)
            stage.update(status='complete' if result.returncode == 0 else 'failed',
                         exit_code=result.returncode, seconds=time.monotonic()-started)
            save()
            # Preserve partial model results and still produce an honest report.
            if name == 'matrix' and not (root/'matrix.json').exists():
                break
        report['status'] = 'failed' if any(s['status'] != 'complete' for s in report['stages']) else 'complete'
    except KeyboardInterrupt:
        report['status'] = 'interrupted'
        raise
    except Exception as error:
        report['status'] = 'failed'
        report['error'] = str(error)
        raise
    finally:
        save()
    print(f'Run report: {root/"run.json"}', flush=True)
    if report['status'] != 'complete':
        raise SystemExit('Some stages failed; review run.json and rerun the same command to resume.')


if __name__ == '__main__':
    main()

