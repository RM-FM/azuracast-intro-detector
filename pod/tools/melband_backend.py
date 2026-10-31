"""Pinned Mel-Band RoFormer GPU experiment; never imported by playout."""
import contextlib
import hashlib
import importlib
from pathlib import Path
import sys
import time
import types
import urllib.request

REVISION = '9430afc97432ba22cdae8171a0567759c229df82'
WEIGHT_REVISION = 'ac9b0614ab3cd7f77219e18ba494dfd93956c348'
WEIGHT_SHA256 = '87201f4d31afb5bc79993230fc49446918425574db48c01c405e44f365c7559e'
PROFILE = 'msst-melband-9430afc-legacy-sdp-amp-fp16-chunk8-overlap2-v1'
CONFIG = 'configs/KimberleyJensen/config_vocals_mel_band_roformer_kj.yaml'
# Git blob IDs verified against the pinned upstream revision.
SOURCE_FILES = {
    'models/bs_roformer/mel_band_roformer.py': '9a3c0d644cdf6b219ed27501fa3817a6ddad6bdd',
    'models/bs_roformer/attend.py': 'd6dc4b3079cff5b3c8c90cea8df2301afd18918b',
    CONFIG: 'f7f511025573acaa7a53f75c957b878c0aaa8205',
    'LICENSE': '9d7186e88bca9975edd65956cd499fa60bd04251',
}


def file_digest(path, git_blob=False):
    digest = hashlib.sha1() if git_blob else hashlib.sha256()
    if git_blob:
        digest.update(f'blob {path.stat().st_size}\0'.encode())
    with path.open('rb') as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def download_verified(url, target, expected, git_blob=False):
    if target.is_file() and file_digest(target, git_blob) == expected:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + '.part')
    print(f'Downloading {target.name}', flush=True)
    try:
        request = urllib.request.Request(url, headers={'User-Agent': 'azuracast-intro-detector'})
        with urllib.request.urlopen(request, timeout=120) as source, temporary.open('wb') as out:
            while block := source.read(1024 * 1024):
                out.write(block)
        if file_digest(temporary, git_blob) != expected:
            raise ValueError(f'Checksum mismatch: {target.name}')
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def ensure_assets(cache):
    source = Path(cache) / ('source-' + REVISION)
    for name, digest in SOURCE_FILES.items():
        download_verified(
            f'https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/{REVISION}/{name}',
            source / name, digest, git_blob=True)
    weights = Path(cache) / 'MelBandRoformer.ckpt'
    download_verified(
        f'https://huggingface.co/KimberleyJSN/melbandroformer/resolve/{WEIGHT_REVISION}/MelBandRoformer.ckpt',
        weights, WEIGHT_SHA256)
    return source, weights


def load_model(source, weights, device):
    """Load only the verified inference modules, not MSST's training imports."""
    import torch
    import yaml
    source = Path(source)
    for name, path in [('models', 'models'), ('models.bs_roformer', 'models/bs_roformer')]:
        package = types.ModuleType(name)
        package.__path__ = [str(source / path)]
        sys.modules[name] = package
    module = importlib.import_module('models.bs_roformer.mel_band_roformer')
    config = yaml.load((source / CONFIG).read_text(), Loader=yaml.FullLoader)
    model = module.MelBandRoformer(**config['model'])
    model.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True))
    if torch.device(device).type == 'cuda':
        attend = importlib.import_module('models.bs_roformer.attend')
        properties = torch.cuda.get_device_properties(device)
        flash = (properties.major, properties.minor) == (8, 0)
        for layer in model.modules():
            if isinstance(layer, attend.Attend) and layer.flash:
                # Preserve Kimberley's A100-only kernel choice, including on RTX 4090.
                layer.cuda_config = attend.FlashAttentionConfig(flash, not flash, not flash)
    return model.to(device).eval(), int(config['audio']['chunk_size']), int(config['inference']['num_overlap'])


def overlap_add(model, mix, chunk_size, overlap=2, amp=True):
    """Upstream overlap-add with explicit pad tracking and no silent NaN repair.

    Track-level time zero and length are preserved, including clips shorter
    than one chunk. Only padded context is removed; there is no cue trimming.
    """
    import torch
    import torch.nn.functional as F
    length = mix.shape[-1]
    if length < 1 or chunk_size < 10 or overlap < 1 or overlap > chunk_size:
        raise ValueError('Invalid overlap-add dimensions')
    step = chunk_size // overlap
    border = chunk_size - step
    padded = border > 0 and length > 2 * border
    if padded:
        mix = F.pad(mix, (border, border), mode='reflect')
    total = mix.shape[-1]
    result = torch.zeros_like(mix, dtype=torch.float32)
    counter = torch.zeros(total, device=mix.device, dtype=torch.float32)
    fade = chunk_size // 10
    window = torch.ones(chunk_size, device=mix.device)
    window[:fade] = torch.linspace(0, 1, fade, device=mix.device)
    window[-fade:] = torch.linspace(1, 0, fade, device=mix.device)
    autocast = torch.autocast('cuda', dtype=torch.float16) if amp else contextlib.nullcontext()
    with torch.inference_mode(), autocast:
        for start in range(0, total, step):
            part = mix[:, start:start + chunk_size]
            size = part.shape[-1]
            if size < chunk_size:
                mode = 'reflect' if size > chunk_size // 2 + 1 else 'constant'
                part = F.pad(part, (0, chunk_size - size), mode=mode)
            estimate = model(part[None])[0]
            if estimate.shape != part.shape or not torch.isfinite(estimate).all():
                raise ValueError('Invalid Mel-Band output shape or non-finite samples')
            weight = window.clone()
            if start == 0:
                weight[:fade] = 1
            if start + chunk_size >= total:
                weight[-fade:] = 1
            result[:, start:start + size] += estimate[:, :size].float() * weight[:size]
            counter[start:start + size] += weight[:size]
    if not (counter > 0).all():
        raise ValueError('Uncovered samples in Mel-Band overlap-add')
    result /= counter
    if padded:
        result = result[:, border:-border]
    if result.shape[-1] != length or not torch.isfinite(result).all():
        raise ValueError('Mel-Band output alignment error')
    return result


class Separator:
    def __init__(self, source, weights):
        self.source, self.weights = Path(source), Path(weights)
        self.model = None
        self.identity = dict(profile=PROFILE, source_revision=REVISION,
                             weight_sha256=WEIGHT_SHA256,
                             backend_sha256=file_digest(Path(__file__)))

    def separate(self, song, output, duration, threads):
        import numpy as np
        import soundfile as sf
        import torch
        from model_worker import decode
        if not torch.cuda.is_available():
            raise RuntimeError('Mel-Band experiment requires CUDA; no CPU fallback')
        torch.set_num_threads(threads)
        began = time.monotonic()
        if self.model is None:
            print('Mel-Band RoFormer: loading verified weights once on CUDA', flush=True)
            self.model, self.chunk_size, self.overlap = load_model(self.source, self.weights, 'cuda')
        torch.cuda.synchronize()
        load_seconds = time.monotonic() - began
        began = time.monotonic()
        samples = decode(song, 44100, 2, 0, duration).reshape(-1, 2).T.copy()
        if not samples.shape[1] or not np.isfinite(samples).all():
            raise ValueError('Empty/non-finite Mel-Band input')
        decode_seconds = time.monotonic() - began
        wave = torch.from_numpy(samples).to('cuda')
        torch.cuda.synchronize()
        began = time.monotonic()
        vocals = overlap_add(self.model, wave, self.chunk_size, self.overlap).cpu().numpy()
        torch.cuda.synchronize()
        inference_seconds = time.monotonic() - began
        temporary = Path(output).with_suffix('.tmp.wav')
        sf.write(temporary, vocals.T, 44100, subtype='FLOAT')
        temporary.replace(output)
        return dict(separator='melband', load_seconds=load_seconds,
                    decode_seconds=decode_seconds, inference_seconds=inference_seconds,
                    audio_seconds=samples.shape[1] / 44100, identity=self.identity)

    def release(self):
        self.model = None
