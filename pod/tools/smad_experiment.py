#!/usr/bin/env python3
"""Evaluate a pinned clip-level singing detector on every song. No cue writes."""
import argparse
import datetime
import fcntl
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import subprocess
import time

import whisper_experiment as base

MODEL_ID = 'duclvQ/smad'
MODEL_REVISION = '157a6316366bcf8831baf23df742edb71f07eb47'
PROFILE = 'smad-crnn-4s-sliding-v2-first-30s'
SAMPLE_RATE = 16000
WINDOW_SECONDS = 4.0
HOP_SECONDS = 0.5
LABELS = ('speech_noise', 'speech_music', 'singing_music', 'none')


def window_starts(duration, window=WINDOW_SECONDS, hop=HOP_SECONDS):
    """Return covering window starts, including the final right-aligned crop."""
    if not all(math.isfinite(v) for v in (duration, window, hop)) or duration <= 0 or window <= 0 or hop <= 0:
        raise ValueError('Invalid SMAD window settings')
    if duration <= window:
        return [0.0]
    last = duration - window
    starts = [i * hop for i in range(int(last // hop) + 1)]
    if last - starts[-1] > 1e-6:
        starts.append(last)
    return [round(v, 6) for v in starts]


def smad_model(torch, cache):
    """Load the audited tiny CRNN and safetensors without executing Hub code."""
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    import torch.nn as nn

    class ConvBlock(nn.Module):
        def __init__(self, in_ch, out_ch, pool):
            super().__init__()
            self.block = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                                       nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True), nn.MaxPool2d(pool))

        def forward(self, x):
            return self.block(x)

    class TinyAudioCRNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer('feat_mean', torch.zeros(80))
            self.register_buffer('feat_std', torch.ones(80))
            channels = (32, 64, 128, 128)
            pools = [(2, 2)] * 3 + [(2, 1)]
            blocks, in_ch = [], 1
            for out_ch, pool in zip(channels, pools):
                blocks.append(ConvBlock(in_ch, out_ch, pool))
                in_ch = out_ch
            self.conv = nn.Sequential(*blocks)
            rnn_in = channels[-1] * (80 // 16)
            self.dropout = nn.Dropout(0.2)
            self.rnn = nn.GRU(rnn_in, 128, num_layers=1, batch_first=True, bidirectional=True)
            self.classifier = nn.Linear(128 * 2 * 2, 4)

        def forward(self, features):
            x = (features - self.feat_mean) / self.feat_std
            x = x.transpose(1, 2).unsqueeze(1)
            x = self.conv(x)
            batch, channels, bins, frames = x.shape
            x = x.permute(0, 3, 1, 2).reshape(batch, frames, channels * bins)
            x = self.dropout(x)
            x, _ = self.rnn(x)
            x = torch.cat((x.mean(dim=1), x.max(dim=1).values), dim=-1)
            return self.classifier(self.dropout(x))

    cache.mkdir(parents=True, exist_ok=True)
    config_path = hf_hub_download(MODEL_ID, 'config.json', revision=MODEL_REVISION, cache_dir=str(cache))
    weights_path = hf_hub_download(MODEL_ID, 'model.safetensors', revision=MODEL_REVISION, cache_dir=str(cache))
    config = json.loads(Path(config_path).read_text())
    if config.get('id2label') != {str(i): label for i, label in enumerate(LABELS)} or \
       config.get('sample_rate') != SAMPLE_RATE or config.get('segment_seconds') != WINDOW_SECONDS:
        raise ValueError('Pinned SMAD model configuration changed unexpectedly')
    model = TinyAudioCRNN()
    state = load_file(weights_path, device='cpu')
    # Hugging Face wraps this module as `smad.*`; load only that documented prefix.
    prefixed = ({key.removeprefix('smad.'): value for key, value in state.items()
                 if key.startswith('smad.')} if any(key.startswith('smad.') for key in state) else state)
    missing, unexpected = model.load_state_dict(prefixed, strict=False)
    if missing or unexpected:
        raise ValueError(f'SMAD weight structure mismatch: missing={missing}, unexpected={unexpected}')
    model.eval().to('cuda')
    return model, float(config['temperature']), dict(
        model=MODEL_ID, revision=MODEL_REVISION, config_sha256=base.sha256(Path(config_path)),
        weights_sha256=base.sha256(Path(weights_path)), temperature=float(config['temperature']))


def features_for(wave, starts, librosa, np):
    rows = []
    width = int(WINDOW_SECONDS * SAMPLE_RATE)
    for start in starts:
        left = int(round(start * SAMPLE_RATE))
        clip = wave[left:left + width]
        if len(clip) < width:
            clip = np.pad(clip, (0, width - len(clip)))
        mel = librosa.feature.melspectrogram(y=np.asarray(clip, dtype=np.float32), sr=SAMPLE_RATE,
            n_fft=400, hop_length=160, n_mels=80, power=2.0)
        rows.append(librosa.power_to_db(mel).T.astype(np.float32))
    return rows


def infer_song(song, duration, model, temperature, batch_size, torch, np, librosa):
    from model_worker import decode
    wave = decode(song, SAMPLE_RATE, start=0, duration=duration)
    if wave.size == 0 or not np.isfinite(wave).all() or abs(wave.size / SAMPLE_RATE - duration) > .03:
        raise ValueError('Invalid 0–45s decoded waveform')
    starts = window_starts(duration)
    feature_rows = features_for(wave, starts, librosa, np)
    probs = []
    with torch.inference_mode():
        for begin in range(0, len(feature_rows), batch_size):
            batch = torch.from_numpy(np.stack(feature_rows[begin:begin + batch_size])).to('cuda')
            logits = model(batch)
            result = torch.softmax(logits / temperature, dim=-1).detach().cpu().numpy()
            probs.extend(result.tolist())
    if len(probs) != len(starts) or any(not np.isfinite(row).all() for row in np.asarray(probs)):
        raise ValueError('SMAD returned invalid probabilities')
    return dict(duration=duration, window_seconds=WINDOW_SECONDS, hop_seconds=HOP_SECONDS,
        class_order=list(LABELS), window_starts=starts, probabilities=probs,
        notice='Clip-level probabilities, not frame-aligned onsets. Candidate offsets are benchmark hypotheses only.')


def run(args):
    import numpy as np
    import librosa
    import torch
    from whisper_experiment import atomic_json
    torch.set_num_threads(args.threads)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; CPU fallback disabled')
    songs = sorted(p for p in args.songs.rglob('*') if p.is_file() and p.suffix.lower() in base.EXTENSIONS)
    if len(songs) != args.expected_count:
        raise ValueError(f'Expected {args.expected_count} songs; found {len(songs)}')
    source_digest = base.sha256(Path(__file__))
    cache = args.evidence / 'model-cache'
    model, temperature, model_info = smad_model(torch, cache)
    evidence_root = args.evidence / 'ultimate'
    out_dir = evidence_root / 'smad'
    out_dir.mkdir(parents=True, exist_ok=True)
    versions = {name: importlib.metadata.version(name) for name in ('torch', 'librosa', 'numpy', 'safetensors', 'huggingface-hub')}
    report = dict(status='running', model=model_info, gpu=torch.cuda.get_device_name(0), packages=versions,
        source_sha256=source_digest, expected_songs=args.expected_count, window_seconds=WINDOW_SECONDS,
        hop_seconds=HOP_SECONDS, rows=[], errors=[])
    began = time.monotonic()
    for index, song in enumerate(songs, 1):
        name = song.relative_to(args.songs).as_posix()
        audio_hash = base.sha256(song)
        duration = min(args.window_seconds, float(subprocess.check_output([
            'ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', str(song)
        ], text=True).strip()))
        identity = dict(profile=PROFILE, song=name, audio_sha256=audio_hash, duration=duration,
                        model=model_info, packages=versions, source_sha256=source_digest,
                        sample_rate=SAMPLE_RATE, window_seconds=WINDOW_SECONDS, hop_seconds=HOP_SECONDS,
                        analyzed_seconds=duration)
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        target = out_dir / f'{key}.json'
        stored = None if args.force else base.cached_json(target, identity)
        reused = stored is not None
        try:
            if stored is None:
                print(f'SMAD [{index}/{len(songs)}] {name} | 0–{duration:g}s', flush=True)
                started = time.monotonic()
                result = infer_song(song, duration, model, temperature, args.batch_size, torch, np, librosa)
                torch.cuda.synchronize()
                stored = dict(status='complete', identity=identity, result=result,
                              elapsed_seconds=time.monotonic() - started)
                base.atomic_json(target, stored)
            report['rows'].append(dict(song=name, sha256=audio_hash, evidence=str(target.relative_to(evidence_root)),
                reused=reused,
                seconds_this_run=stored.get('elapsed_seconds', 0)))
        except Exception as error:
            report['errors'].append(dict(song=name, error=str(error)))
        report['elapsed_seconds'] = time.monotonic() - began
        atomic_json(evidence_root / 'smad-report.json', report)
    report['status'] = 'failed' if report['errors'] or len(report['rows']) != args.expected_count else 'complete'
    report['elapsed_seconds'] = time.monotonic() - began
    atomic_json(evidence_root / 'smad-report.json', report)
    print(f'SMAD report: {evidence_root / "smad-report.json"}', flush=True)
    if report['status'] != 'complete':
        raise SystemExit('SMAD stage incomplete; see smad-report.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--songs', type=Path, default=Path('/workspace/input/songs'))
    parser.add_argument('--evidence', type=Path, default=Path('/workspace/experiments/whisper-evidence'))
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--expected-count', type=int, default=40)
    parser.add_argument('--window-seconds', type=float, default=30.0,
                        help='Analyze only the first N seconds (30s matches the comparison matrix)')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    if args.threads < 1 or args.batch_size < 1 or args.expected_count < 1 or args.window_seconds <= 0:
        parser.error('threads, batch size, expected count, and window must be positive')
    args.songs, args.evidence = args.songs.resolve(), args.evidence.resolve()
    if not args.songs.is_dir() or args.songs.is_relative_to(args.evidence) or args.evidence.is_relative_to(args.songs):
        parser.error('Songs and evidence must be separate existing directory trees')
    evidence_root = args.evidence / 'ultimate'
    evidence_root.mkdir(parents=True, exist_ok=True)
    with (args.evidence / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args)


if __name__ == '__main__':
    main()
