"""Fetch hash-pinned upstream sources needed by the worker source bundle."""
import argparse
import gzip
import hashlib
import shutil
import subprocess
import tempfile
from pathlib import Path
from urllib.request import urlopen

from build_source_bundle import NATIVE_BUILD_SOURCES


DIRECT_SOURCES = {
    'av-18.1.0.tar.gz': (
        'https://files.pythonhosted.org/packages/8d/f4/f22114d30d3435e38c6af2b4870f37b864403dca6ae7af747a289ce0a18e/av-18.1.0.tar.gz',
        '47bfc286e1bc9de7ab4681fc2b575cd2460a66919d31ffe1bd5aa54fae531a28'),
    'ffmpeg-8.1.2.tar.xz': (
        'https://ffmpeg.org/releases/ffmpeg-8.1.2.tar.xz',
        '464beb5e7bf0c311e68b45ae2f04e9cc2af88851abb4082231742a74d97b524c'),
    'ffmpeg-8.1.2.tar.xz.asc': (
        'https://ffmpeg.org/releases/ffmpeg-8.1.2.tar.xz.asc',
        '0a0963fccd70597838073f3e31b20f4a4d8cc2b5e577472c9a5a1f22624246f8'),
    'ffmpeg-devel.asc': (
        'https://ffmpeg.org/ffmpeg-devel.asc',
        '397b3becedcd5a98769967ff1ff8501ddc89f8368b8f766e4701377d7dbaabe5'),
    'fribidi-1.0.8.tar.bz2': (
        'https://github.com/fribidi/fribidi/releases/download/v1.0.8/fribidi-1.0.8.tar.bz2',
        '94c7b68d86ad2a9613b4dcffe7bbeb03523d63b5b37918bdf2e4ef34195c1e6c'),
    'lame-4.0.tar.gz': (
        'https://downloads.sourceforge.net/project/lame/lame/4.0/lame-4.0.tar.gz',
        '3df5124d5ad3a98312ffd7ba6a9b36230e4f8a3e66d3ce0f425e336c32d216eb'),
    'libsndfile-1.2.2.tar.xz': (
        'https://github.com/libsndfile/libsndfile/releases/download/1.2.2/libsndfile-1.2.2.tar.xz',
        '3799ca9924d3125038880367bf1468e53a1b7e3686a934f098b7e1d286cdb80e'),
    'soxr-1.1.0.tar.gz': (
        'https://files.pythonhosted.org/packages/ed/11/27cebce4a108f77afea7c80545115536b45e3f11ebfb914f638fdd9ba847/soxr-1.1.0.tar.gz',
        '9f228ae21c78fa9359ca98d8a5e8e91f30639e438e574133dace62c5b5309e44'),
    'soundfile-0.13.1.tar.gz': (
        'https://files.pythonhosted.org/packages/e1/41/9b873a8c055582859b239be17902a85339bec6a30ad162f98c9b0288a2cc/soundfile-0.13.1.tar.gz',
        'b2c68dab1e30297317080a5b43df57e302584c49e2942defdde0acccc53f0e5b'),
}
GIT_ARCHIVES = {
    'lameenc-1.9.0.tar.gz': (
        'https://github.com/chrisstaite/lameenc.git',
        '77aba334b8a2ca4a8bb8c5997f3d8c11ca5d90b1',
        'chrisstaite-lameenc-77aba33',
        '958644e5223607903309ce6d2d8c9d34e05c3d6eb0960e4eb954c33b6939c4bd'),
    'libsoxr-a66f3eee.tar.gz': (
        'https://github.com/dofuuz/soxr.git',
        'a66f3eeeeb62a32403ff143b756eed92b1ec6b62',
        'dofuuz-soxr-a66f3ee',
        'c41adb9b67992e5af1fba7287a196a26000bd7703a7a3f30fa5adb7bfcd0f0e4'),
    'pillow-13.0.0-build-sources.tar.gz': (
        'https://github.com/python-pillow/Pillow.git',
        '293218ad059d994baf1777d882911bed6a49704a',
        'python-pillow-Pillow-293218a',
        'd6b77f3924012a8ac332e913f180edc5309bf8b0c66b3a325748ff965d411f6d'),
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path, expected):
    actual = sha256(path)
    if actual != expected:
        Path(path).unlink(missing_ok=True)
        raise ValueError(f'Source checksum mismatch for {Path(path).name}: {actual}')


def download(url, output):
    with urlopen(url, timeout=60) as response:
        if not response.geturl().startswith('https://'):
            raise ValueError(f'Insecure source redirect: {response.geturl()}')
        with Path(output).open('wb') as destination:
            shutil.copyfileobj(response, destination)


def git_archive(repository, commit, prefix, output, temporary_root):
    checkout = Path(temporary_root) / Path(output).stem
    checkout.mkdir()
    subprocess.run(['git', 'init', '--quiet', str(checkout)], check=True)
    subprocess.run(['git', '-C', str(checkout), 'remote', 'add', 'origin', repository], check=True)
    subprocess.run(['git', '-C', str(checkout), 'fetch', '--quiet', '--depth=1', 'origin', commit], check=True)
    tar_path = checkout.parent / f'{checkout.name}.tar'
    with tar_path.open('wb') as archive:
        subprocess.run([
            'git', '-C', str(checkout), 'archive', '--format=tar', f'--prefix={prefix}/', 'FETCH_HEAD'],
            stdout=archive, check=True)
    with Path(output).open('wb') as archive:
        subprocess.run(['gzip', '-n', '-c', str(tar_path)], stdout=archive, check=True)


def fetch(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    expected_names = set(DIRECT_SOURCES) | set(GIT_ARCHIVES)
    if expected_names != set(NATIVE_BUILD_SOURCES):
        raise ValueError('Native source definitions differ from the source-bundle allowlist')
    with tempfile.TemporaryDirectory(prefix='intro-native-source-') as temporary_root:
        for name, (url, expected) in DIRECT_SOURCES.items():
            destination = output / name
            if destination.exists():
                verify(destination, expected)
                continue
            download(url, destination)
            verify(destination, expected)
        for name, (repository, commit, prefix, expected) in GIT_ARCHIVES.items():
            destination = output / name
            if destination.exists():
                verify(destination, expected)
                continue
            git_archive(repository, commit, prefix, destination, temporary_root)
            verify(destination, expected)
    return sorted(expected_names)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    arguments = parser.parse_args()
    names = fetch(arguments.output)
    print(f'Verified native source archives: {len(names)}')