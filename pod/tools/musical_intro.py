#!/usr/bin/env python3
"""Measure musical handovers in verified first-30s originals, without vocal decisions."""
import argparse
import json
import importlib.metadata
from pathlib import Path

PROFILE = 'percussion-energy-30s-v1'


def measure(samples, sr):
    import numpy as np
    from scipy.ndimage import median_filter
    from scipy.signal import stft, find_peaks
    y = np.asarray(samples, dtype=float)
    if y.ndim != 1 or not len(y) or not np.isfinite(y).all() or sr < 8000:
        raise ValueError('Invalid mono audio')
    y = y[:int(30*sr)]
    if len(y) < sr*3:
        return dict(candidate=None, reason='short_audio', profile=PROFILE)
    hop, nfft = int(sr*.02), 1024
    _, times, spectrum = stft(y, sr, nperseg=nfft, noverlap=nfft-hop,
                              boundary='zeros', padded=True)
    power = abs(spectrum)**2
    # Median-filter HPSS soft mask. Sustained tones are suppressed in percussion.
    harmonic = median_filter(power, size=(1, 31))
    percussive = median_filter(power, size=(31, 1))
    mask = percussive**2 / np.maximum(percussive**2+harmonic**2, 1e-20)
    energy = power.sum(axis=0)
    drum = (power*mask).sum(axis=0)
    db = 10*np.log10(np.maximum(energy, 1e-15))
    drum_db = 10*np.log10(np.maximum(drum, 1e-15))
    flux = np.maximum(np.diff(np.sqrt(power*mask), axis=1), 0).sum(axis=0)
    flux = np.r_[0, flux]
    peaks, _ = find_peaks(flux, distance=max(1, int(.18/.02)),
                         prominence=max(float(np.max(flux))*.07, 1e-12))
    attacks = times[peaks]
    smooth = median_filter(db, size=11)
    smooth_drum = median_filter(drum_db, size=11)
    candidates = []
    for i in range(100, len(times)-100):
        t = float(times[i])
        if not 3 <= t <= 27:
            continue
        before, after = slice(i-100, i-15), slice(i+5, i+100)
        gain = float(np.median(db[after])-np.median(db[before]))
        percussion_gain = float(np.median(drum_db[after])-np.median(drum_db[before]))
        following = attacks[(attacks >= t) & (attacks <= t+2)]
        sustained = float(np.median(db[after])) > float(np.max(db))-24
        local_rise = float(smooth[i+5]-smooth[i-5])
        local_drum = float(smooth_drum[i+5]-smooth_drum[i-5])
        if sustained and ((gain >= 5 and local_rise >= 2) or
                          (percussion_gain >= 8 and local_drum >= 3 and len(following) >= 3)):
            near = attacks[(attacks >= t-.4) & (attacks <= t+.5)]
            cue = float(near[np.argmin(abs(near-t))]) if len(near) else t
            candidates.append(dict(time_s=round(cue, 3),
                kind='percussion_entrance' if percussion_gain >= 8 and len(following) >= 3 else 'energy_rise',
                strength=max(gain/5, percussion_gain/8), energy_gain_db=round(gain, 2),
                percussion_gain_db=round(percussion_gain, 2)))
    if candidates:
        # Group the broad transition response; choose the strongest sustained change.
        best = max(candidates, key=lambda c: c['strength'])
    else:
        # A beat already running is useful too, but only a stable repeated pattern.
        best = None
        for start in attacks[(attacks >= 6) & (attacks <= 24)]:
            group = attacks[(attacks >= start) & (attacks <= start+3)]
            gaps = np.diff(group)
            if len(gaps) >= 4 and .2 <= np.median(gaps) <= .8 and np.std(gaps)/np.mean(gaps) < .18:
                best = dict(time_s=round(float(start), 3), kind='steady_percussion', strength=1.)
                break
    return dict(profile=PROFILE, candidate=best, duration=len(y)/sr,
                reason='musical_handover' if best else 'no_stable_musical_change',
                frame_seconds=.02, energy_db=np.round(db, 2).tolist(),
                percussion_db=np.round(drum_db, 2).tolist(),
                onset_times_s=[round(float(t), 3) for t in attacks])


def main():
    import whisper_experiment as base
    import qwen_experiment as qwen
    from model_worker import decode
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--songs', type=Path, required=True)
    ap.add_argument('--evidence', type=Path, required=True)
    args = ap.parse_args()
    root = args.evidence/'ultimate'
    matrix = json.loads((root/'matrix.json').read_text())
    jobs, errors = qwen.select_jobs(args.evidence, args.songs, matrix, None, ['original'], {})
    folder = root/'music'; folder.mkdir(exist_ok=True)
    report = dict(profile=PROFILE, rows=[], errors=errors)
    import hashlib
    for job in jobs:
        identity = dict(profile=PROFILE, audio_sha256=job['audio_hash'], sha256=job['sha256'],
                        duration=min(30, job['duration']),
                        packages={p:importlib.metadata.version(p) for p in ('numpy','scipy')},
                        source=base.sha256(Path(__file__)))
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        target = folder/(key+'.json')
        try:
            stored = base.cached_json(target, identity)
            if stored is None:
                print(f'Musical timing | {job["song"]} | first 30s', flush=True)
                stored = dict(status='complete', identity=identity,
                              result=measure(decode(job['audio'], 22050, duration=min(30, job['duration'])), 22050))
                base.atomic_json(target, stored)
            report['rows'].append(dict(song=job['song'], sha256=job['sha256'],
                audio_hash=job['audio_hash'], **stored['result']))
        except Exception as error:
            report['errors'].append(dict(song=job['song'], error=str(error)))
    report['status'] = 'failed' if report['errors'] else 'complete'
    base.atomic_json(root/'music-report.json', report)
    if report['errors']:
        raise SystemExit('Incomplete musical timing; see music-report.json')


if __name__ == '__main__':
    main()
