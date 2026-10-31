"""Experimental CUDA Demucs; not imported by the production scanner."""
import pathlib
import time

PROFILE = 'htdemucs-4.0.1-fp32-seg7-overlap025-shifts0-v1'
_model = None


def separate(job, decode):
    global _model
    import numpy as np
    import soundfile as sf
    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    if not torch.cuda.is_available():
        raise RuntimeError('Demucs experiment requires CUDA; no silent CPU fallback')
    torch.set_num_threads(job['threads'])
    started = time.monotonic()
    if _model is None:
        print('Demucs: loading official htdemucs once on CUDA', flush=True)
        _model = get_model('htdemucs').to('cuda').eval()
    torch.cuda.synchronize()
    load_seconds = time.monotonic() - started
    rate = _model.samplerate
    channels = _model.audio_channels
    samples = decode(job['song'], rate, channels, job['left'],
                     job['right'] - job['left']).reshape(-1, channels).T.copy()
    if not samples.shape[1] or not np.isfinite(samples).all():
        raise ValueError('Empty or non-finite Demucs input')
    # Match upstream separation normalization, then restore original amplitude.
    # Never clip, peak-normalize or fade the stem: fusion uses its actual RMS.
    wave = torch.from_numpy(samples)
    ref = wave.mean(0)
    mean, std = ref.mean(), ref.std()
    started = time.monotonic()
    if float(std) <= 1e-8:
        vocals = np.zeros_like(samples)
    else:
        with torch.inference_mode():
            sources = apply_model(_model, ((wave - mean) / std)[None],
                                  device='cuda', shifts=0, split=True,
                                  overlap=0.25, segment=7, progress=False,
                                  num_workers=0)[0]
            vocals = (sources[_model.sources.index('vocals')] * std + mean).cpu().numpy()
        del sources
    torch.cuda.synchronize()
    inference_seconds = time.monotonic() - started
    if vocals.shape != samples.shape or not np.isfinite(vocals).all():
        raise ValueError('Demucs output alignment/non-finite error')
    target = pathlib.Path(job['stem'])
    temporary = target.with_suffix('.tmp.wav')
    sf.write(temporary, vocals.T, rate, subtype='FLOAT')
    temporary.replace(target)
    return dict(separator='demucs', separation_profile=PROFILE, stem=str(target),
                load_seconds=load_seconds, inference_seconds=inference_seconds,
                audio_seconds=samples.shape[1] / rate,
                gpu_peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20)
