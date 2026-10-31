"""Prepare operator configuration without exposing credentials in process arguments."""
import argparse
from pathlib import Path
import re
import shutil
import sys
import tempfile

from dotenv import dotenv_values, set_key
import tomlkit

from .config import Runpod, load


def inspect(root):
    root = Path(root)
    path = root / 'config.toml'
    document = tomlkit.parse(path.read_text()) if path.exists() else {}
    keys = dotenv_values(root / '.env', interpolate=False)
    return (str(document.get('azuracast', {}).get('base_url', '')),
            bool(keys.get('AZURACAST_API_KEY')), bool(keys.get('RUNPOD_API_KEY')))


def worker_image(root):
    root = Path(root)
    path = root / 'config.toml'
    if not path.exists():
        path = root / 'config.example.toml'
    settings = tomlkit.parse(path.read_text()).get('runpod', {})
    image = settings.get('image', Runpod.image)
    if not isinstance(image, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._:/@-]*', image):
        raise ValueError('Invalid worker image reference')
    return image, bool(settings.get('registry_auth_id'))


def prepare(root, url, azura_key, runpod_key):
    root = Path(root)
    config_path, env_path = root / 'config.toml', root / '.env'
    for path in (config_path, env_path, root / 'config.example.toml'):
        if path.is_symlink():
            raise ValueError('Configuration files must not be symlinks')
    existing = dotenv_values(env_path, interpolate=False)
    keys = dict(AZURACAST_API_KEY=azura_key or existing.get('AZURACAST_API_KEY') or '',
                RUNPOD_API_KEY=runpod_key or existing.get('RUNPOD_API_KEY') or '')
    for value in keys.values():
        if not value or any(character.isspace() or character in '\\"\x00' for character in value):
            raise ValueError('Both API keys must be non-empty single-line tokens')
    document = tomlkit.parse((config_path if config_path.exists() else root / 'config.example.toml').read_text())
    document.setdefault('azuracast', tomlkit.table())['base_url'] = url.rstrip('/')
    if not config_path.exists():
        document.setdefault('station_defaults', tomlkit.table())['mode'] = 'shadow'
        document.setdefault('scan', tomlkit.table())['auto_process'] = False
    with tempfile.TemporaryDirectory(prefix='.setup-', dir=root) as directory:
        temporary = Path(directory)
        candidate = temporary / 'config.toml'
        candidate.write_text(tomlkit.dumps(document))
        candidate.chmod(0o640)
        load(candidate, environ=keys)
        env = temporary / '.env'
        if env_path.exists():
            shutil.copyfile(env_path, env)
        else:
            env.touch(mode=0o600)
        env.chmod(0o600)
        for name, value in keys.items():
            set_key(env, name, value, quote_mode='always')
        env.replace(env_path)
        candidate.replace(config_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('inspect', 'prepare', 'worker-image'))
    parser.add_argument('--root', type=Path, default=Path('/setup'))
    args = parser.parse_args()
    if args.action in ('inspect', 'worker-image'):
        values = inspect(args.root) if args.action == 'inspect' else worker_image(args.root)
        for value in values:
            print(int(value) if isinstance(value, bool) else value)
    else:
        fields = sys.stdin.buffer.read().split(b'\0')
        if len(fields) != 4 or fields[-1] != b'':
            raise ValueError('Expected three setup input fields')
        prepare(args.root, *(field.decode('utf-8') for field in fields[:-1]))


if __name__ == '__main__':
    main()