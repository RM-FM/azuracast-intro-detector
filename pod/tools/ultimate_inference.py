#!/usr/bin/env python3
"""Additional bounded inference stages. Never imported by production playout."""
import argparse
import datetime
import fcntl
import importlib.metadata
import json
from pathlib import Path
import shutil
import time
import urllib.request

import whisper_experiment as base
import song_language
import stem_comparison as stems
import qwen_experiment as qwen

INPUTS = ('original', 'demucs', 'melband')
VOCAL = {'Speech', 'Singing', 'Male singing', 'Female singing', 'Child singing',
         'Choir', 'Rapping', 'Humming', 'Chant', 'Mantra', 'Vocal music'}
SINGING = VOCAL - {'Speech'}
ACOUSTIC_SOURCE_SHA256 = 'e5d9487620fa8a52ed6475cd5b0b778002e12a7d8eb5b5d44655804ab754d488'


def read(path):
    return json.loads(Path(path).read_text())


def root_for(evidence):
    root = evidence / 'ultimate'
    for path in (root, root / 'runs', root / 'acoustic'):
        if path.is_symlink():
            raise ValueError('Output directories must not be symlinks')
        path.mkdir(exist_ok=True)
    return root


def matrix_stage(args, root):
    source = read(args.evidence / 'comparison-report.json')
    # Enumerate the current library, including tracks whose every stage failed.
    names = [p.relative_to(args.songs).as_posix() for p in base.select_songs(args.songs, None, True)]
    rows = list(source['rows'])
    jobs, errors = qwen.select_jobs(args.evidence, args.songs, dict(songs=names, rows=rows), None, INPUTS, {})
    valid = {(j['song'], j['input']) for j in jobs}
    matrix = dict(songs=names, inputs=INPUTS, window=30, status='failed' if errors else 'complete',
                  rows=[r for r in rows if (r['song'], r['input']) in valid], errors=errors,
                  fingerprints={name: base.sha256(qwen.beneath(args.songs, name)) for name in names})
    base.atomic_json(root / 'matrix.json', matrix)
    return matrix


def checkpoint(cache):
    directory = cache / 'panns'
    directory.mkdir(parents=True, exist_ok=True)
    path, lock = directory / 'Cnn14_DecisionLevelMax.pth', directory / 'weights.json'
    if not path.exists():
        candidates = [Path('/cache/panns_data') / path.name, Path.home() / 'panns_data' / path.name]
        previous = next((p for p in candidates if p.is_file() and p.stat().st_size > 300000000), None)
        temporary = path.with_suffix('.part')
        try:
            if previous:
                shutil.copyfile(previous, temporary)
            else:
                url = 'https://zenodo.org/record/3987831/files/Cnn14_DecisionLevelMax_mAP%3D0.385.pth?download=1'
                with urllib.request.urlopen(url, timeout=120) as response, temporary.open('wb') as out:
                    shutil.copyfileobj(response, out)
            if temporary.stat().st_size < 300000000:
                raise ValueError('Incomplete PANNs checkpoint')
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    digest = base.sha256(path)
    if lock.exists() and read(lock)['sha256'] != digest:
        raise ValueError('PANNs checkpoint changed from recorded hash')
    if not lock.exists():
        base.atomic_json(lock, dict(sha256=digest))
    return path, digest


def classify_audio(path, duration, model, labels, seconds):
    """10ms timeline, full coverage, fixed cores/context; keep levels unnormalized."""
    import numpy as np
    import torch
    from model_worker import decode
    from frame_alignment import align_frames
    mono = decode(path, 32000, duration=duration)
    if not mono.size or not np.isfinite(mono).all() or abs(len(mono)/32000-duration) > .03:
        raise ValueError('Invalid classifier input duration/samples')
    frames = int(np.ceil(len(mono)/320))
    blocks = np.pad(mono, (0, frames*320-len(mono))).reshape(-1, 320)
    rms = 20*np.log10(np.maximum(np.sqrt(np.mean(blocks**2, axis=1)), 1e-10))
    groups = {'vocal': VOCAL, 'singing': SINGING, 'speech': {'Speech'}}
    indices = {key: [i for i, label in enumerate(labels) if label in names] for key, names in groups.items()}
    if any(not i for i in indices.values()):
        raise ValueError('PANNs taxonomy mismatch')
    curves = {key: np.zeros(frames, dtype=np.float32) for key in groups}
    windows = []
    full, core_size, context = seconds*32000, seconds*16000, seconds*8000
    for core in range(0, len(mono), core_size):
        end = min(len(mono), core+core_size)
        left, right = max(0, core-context), min(len(mono), end+context)
        clip = np.pad(mono[left:right], (0, full-(right-left)))
        with torch.inference_mode():
            prediction = model.inference(clip[None])[0]
        if not np.isfinite(prediction).all():
            raise ValueError('Non-finite classifier output')
        offset, count, lo = core//320, int(np.ceil((end-core)/320)), (core-left)//320
        for key, ids in indices.items():
            aligned, _ = align_frames(prediction[:, ids].max(axis=1).tolist(), [-100.]*(seconds*100), seconds*100)
            curves[key][offset:offset+count] = aligned[lo:lo+count]
        # Scores are multi-label: a strong Music/instrument label is not proof
        # that vocals are absent. Retain competing classes for inspection.
        core_prediction = prediction[min(lo, len(prediction)-1):min(lo+count, len(prediction))]
        top = core_prediction.mean(axis=0).argsort()[-8:][::-1]
        windows.append(dict(start=core/32000, end=end/32000,
                            top_classes=[dict(label=labels[i], score=float(core_prediction[:, i].mean())) for i in top]))
    return dict(duration=duration, frame_seconds=.01, rms_db=rms.tolist(),
                curves={k: v.tolist() for k, v in curves.items()}, windows=windows,
                input_peak=float(np.max(np.abs(mono))), input_seconds=seconds)


def acoustic_stage(args, root, torch):
    from panns_inference import SoundEventDetection, labels
    from checkpoint_loading import load_weights_only
    jobs, errors = qwen.select_jobs(args.evidence, args.songs, read(root/'matrix.json'), None, INPUTS, {})
    originals = {j['song']: j for j in jobs if j['input'] == 'original'}
    weights, digest = checkpoint(args.cache)
    versions = {p: importlib.metadata.version(p) for p in ('torch', 'panns-inference', 'numpy', 'librosa')}
    report = dict(status='running', rows=[], errors=errors, model_load_seconds=0)
    model = None
    for job in jobs:
        for seconds in (4, 8):
            identity = dict(profile='full-panns-rms-v1', audio_sha256=job['audio_hash'], duration=job['duration'],
                original_audio_sha256=originals.get(job['song'], {}).get('audio_hash'),
                input_seconds=seconds, weight_sha256=digest, packages=versions,
                source_sha256=ACOUSTIC_SOURCE_SHA256,
                decoder_sha256=base.sha256(Path(__file__).with_name('model_worker.py')),
                alignment_sha256=base.sha256(Path(__file__).with_name('frame_alignment.py')))
            target = root / 'acoustic' / (stems.key_for(identity)+'.json')
            try:
                stored = base.cached_json(target, identity)
                reused = stored is not None
                if stored is None:
                    if model is None:
                        began = time.monotonic()
                        model = load_weights_only(SoundEventDetection, checkpoint_path=str(weights), device='cuda')
                        torch.cuda.synchronize()
                        report['model_load_seconds'] = time.monotonic()-began
                    print(f'PANNs {seconds}s | {job["song"]} | {job["input"]}', flush=True)
                    began = time.monotonic()
                    result = classify_audio(job['audio'], job['duration'], model, labels, seconds)
                    if job['song'] in originals:
                        import numpy as np
                        from model_worker import decode
                        mix = decode(originals[job['song']]['audio'], 32000, duration=job['duration'])
                        count = len(result['rms_db'])
                        if not count*320-320 < len(mix) <= count*320:
                            raise ValueError('Original/stem level timeline differs')
                        blocks = np.pad(mix, (0, count*320-len(mix))).reshape(-1, 320)
                        mix_db = 20*np.log10(np.maximum(np.sqrt(np.mean(blocks**2, axis=1)), 1e-10))
                        result['original_rms_db'] = mix_db.tolist()
                        result['stem_to_mix_db'] = (np.asarray(result['rms_db'])-mix_db).tolist()
                    torch.cuda.synchronize()
                    stored = dict(status='complete', identity=identity, result=result, elapsed_seconds=time.monotonic()-began)
                    base.atomic_json(target, stored)
                report['rows'].append(dict(song=job['song'], sha256=job['sha256'], input=job['input'],
                    audio_hash=job['audio_hash'], input_seconds=seconds, evidence=str(target.relative_to(root)),
                    reused=reused, seconds_this_run=0 if reused else stored['elapsed_seconds']))
            except Exception as error:
                report['errors'].append(dict(song=job['song'], input=job['input'], input_seconds=seconds, error=str(error)))
            base.atomic_json(root/'acoustic-report.json', report)
    report['status'] = 'failed' if report['errors'] else 'complete'
    base.atomic_json(root/'acoustic-report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['matrix', 'acoustic'])
    parser.add_argument('--songs', type=Path, default=Path('/workspace/input/songs'))
    parser.add_argument('--evidence', type=Path, default=Path('/workspace/experiments/whisper-evidence'))
    parser.add_argument('--cache', type=Path, default=Path('/workspace/cache'))
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--language', default=song_language.PER_SONG,
                        help="'song' uses each recording's detected language; or a fixed Whisper code")
    args = parser.parse_args()
    args.songs, args.evidence = args.songs.resolve(), args.evidence.resolve()
    if args.threads < 1 or args.evidence.is_relative_to(args.songs) or args.songs.is_relative_to(args.evidence):
        parser.error('Positive threads and separate input/output trees required')
    root = root_for(args.evidence)
    with (args.evidence/'.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.stage == 'matrix':
            result = matrix_stage(args, root)
        else:
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA required; no CPU fallback')
            torch.set_num_threads(args.threads)
            torch.cuda.reset_peak_memory_stats()
            began = time.monotonic()
            result = acoustic_stage(args, root, torch)
            result.update(elapsed_seconds=time.monotonic()-began,
                          gpu=torch.cuda.get_device_name(0), gpu_peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
            base.atomic_json(root/'acoustic-report.json', result)
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        base.atomic_json(root/'runs'/f'{args.stage}-{stamp}.json', result)
        if result['errors']:
            raise SystemExit('Stage has incomplete jobs; completed caches retained. See report errors.')


if __name__ == '__main__':
    main()
