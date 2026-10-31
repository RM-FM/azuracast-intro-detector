"""Bake every pinned model into the paths the vendored Pod pipeline reads.

Runs at image build time without a GPU or private evaluation helpers.
Re-running it offline proves nothing is fetched at runtime.
"""
import json
import hashlib
from pathlib import Path
import shutil
import sys
import urllib.request

sys.path.insert(0, '/app/tools')

CACHE = Path('/workspace/cache')
EVIDENCE = Path('/workspace/experiments/whisper-evidence')
# Same revisions as intro_detector.gpu.QWEN_REVISIONS.
QWEN_REVISIONS = {'Qwen/Qwen3-ASR-1.7B': '7278e1e70fe206f11671096ffdd38061171dd6e5',
                  'Qwen/Qwen3-ForcedAligner-0.6B': 'c7cbfc2048c462b0d63a45797104fc9db3ad62b7'}
SMAD_MODEL = 'duclvQ/smad'
SMAD_REVISION = '157a6316366bcf8831baf23df742edb71f07eb47'
PANNS_URL = 'https://zenodo.org/record/3987831/files/Cnn14_DecisionLevelMax_mAP%3D0.385.pth?download=1'
PANNS_SHA256 = 'dd3b4043a87d4ec13df8082c0fcfee3fb5084151808e47e060987a95eabdd142'
MODEL_FILES = ['*.safetensors', '*.json', '*.txt', '*.model', '*.tiktoken', '*.jinja', '*.md', 'LICENSE*', 'NOTICE*']


def panns_checkpoint(cache):
    directory = Path(cache) / 'panns'
    directory.mkdir(parents=True, exist_ok=True)
    path, lock = directory / 'Cnn14_DecisionLevelMax.pth', directory / 'weights.json'
    if not path.exists():
        candidates = (Path('/cache/panns_data') / path.name, Path.home() / 'panns_data' / path.name)
        previous = next((candidate for candidate in candidates
                         if candidate.is_file() and candidate.stat().st_size > 300000000), None)
        temporary = path.with_suffix('.part')
        try:
            if previous:
                shutil.copyfile(previous, temporary)
            else:
                with urllib.request.urlopen(PANNS_URL, timeout=120) as response, temporary.open('wb') as output:
                    shutil.copyfileobj(response, output)
            if temporary.stat().st_size < 300000000:
                raise ValueError('Incomplete PANNs checkpoint')
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    value = digest.hexdigest()
    if value != PANNS_SHA256:
        raise ValueError('PANNs checkpoint does not match the pinned release')
    if lock.exists() and json.loads(lock.read_text())['sha256'] != value:
        raise ValueError('PANNs checkpoint changed from recorded hash')
    if not lock.exists():
        lock.write_text(json.dumps(dict(sha256=value)) + '\n')
    return path, value


def stem():
    import whisper
    print('Whisper turbo:', whisper._download(whisper._MODELS['turbo'], str(CACHE / 'whisper'), False), flush=True)
    from demucs.pretrained import get_model
    get_model('htdemucs')
    print('Demucs htdemucs ready', flush=True)
    import melband_backend
    print('Mel-Band RoFormer:', melband_backend.ensure_assets(CACHE / 'melband'), flush=True)
    print('PANNs:', panns_checkpoint(CACHE), flush=True)
    import panns_inference  # noqa: F401  (fetches its label CSV into $HOME/panns_data)


def qwen():
    from huggingface_hub import hf_hub_download, snapshot_download
    for repo, revision in QWEN_REVISIONS.items():
        # Same call as qwen_experiment.QwenRunner.load.
        print(repo, snapshot_download(repo, revision=revision, allow_patterns=MODEL_FILES), flush=True)
    lock = EVIDENCE / 'qwen' / 'models.json'
    lock.parent.mkdir(parents=True, exist_ok=True)
    if lock.exists() and json.loads(lock.read_text()) != QWEN_REVISIONS:
        raise SystemExit('Existing Qwen revision lock differs')
    lock.write_text(json.dumps(QWEN_REVISIONS, indent=2) + '\n')
    # smad_experiment uses <evidence>/model-cache as its Hub cache.
    for name in ('config.json', 'model.safetensors'):
        print('SMAD', hf_hub_download(SMAD_MODEL, name, revision=SMAD_REVISION,
                                      cache_dir=str(EVIDENCE / 'model-cache')), flush=True)


if __name__ == '__main__':
    {'stem': stem, 'qwen': qwen}[sys.argv[1]]()
