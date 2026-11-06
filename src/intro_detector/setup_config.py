"""Prepare operator configuration without exposing credentials in process arguments."""
import argparse
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import sys
import tempfile

from dotenv import dotenv_values, set_key
import tomlkit

from .config import Runpod, load

TOKEN = 'INTRO_DETECTOR_API_TOKEN'
# Address of the management API on AzuraCast's Docker network (alias set by setup).
API_URL = 'http://intro-detector:8080'


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


def prepare(root, url, azura_key, runpod_key=''):
    """Write config.toml and .env; empty keys keep existing ones. The Runpod key is optional."""
    root = Path(root)
    config_path, env_path = root / 'config.toml', root / '.env'
    url = url.strip()
    if url and '://' not in url:
        url = f'https://{url}'
    for path in (config_path, env_path, root / 'config.example.toml'):
        if path.is_symlink():
            raise ValueError('Configuration files must not be symlinks')
    existing = dotenv_values(env_path, interpolate=False)
    keys = dict(AZURACAST_API_KEY=azura_key or existing.get('AZURACAST_API_KEY') or '')
    runpod = runpod_key or existing.get('RUNPOD_API_KEY') or ''
    if runpod:
        keys['RUNPOD_API_KEY'] = runpod
    keys[TOKEN] = existing.get(TOKEN) or secrets.token_hex(32)
    for value in keys.values():
        if not value or any(character.isspace() or character in '\\"\x00' for character in value):
            raise ValueError('API keys must be non-empty single-line tokens')
    document = tomlkit.parse((config_path if config_path.exists() else root / 'config.example.toml').read_text())
    document.setdefault('azuracast', tomlkit.table())['base_url'] = url.rstrip('/')
    if not config_path.exists():
        document.setdefault('scan', tomlkit.table())['auto_process'] = True
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


def ensure_token(root):
    """Add the management API token to an existing .env once; True when it was created."""
    env_path = Path(root) / '.env'
    if env_path.is_symlink():
        raise ValueError('Configuration files must not be symlinks')
    if dotenv_values(env_path, interpolate=False).get(TOKEN):
        return False
    set_key(env_path, TOKEN, secrets.token_hex(32), quote_mode='always')
    return True


def link(root, output):
    """Connection file for the Sweepers plugin: API address and token, nothing else."""
    token = dotenv_values(Path(root) / '.env', interpolate=False).get(TOKEN)
    if not token:
        raise ValueError(f'{TOKEN} is missing; run setup first')
    output = Path(output)
    if output.is_symlink():
        raise ValueError('The plugin connection file must not be a symlink')
    fd, temporary = tempfile.mkstemp(dir=output.parent, prefix='.detector-')
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump({'url': API_URL, 'token': token}, stream)
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('inspect', 'prepare', 'worker-image', 'token', 'link'))
    parser.add_argument('--root', type=Path, default=Path('/setup'))
    parser.add_argument('--output', type=Path, default=Path('/plugin/.detector.json'))
    args = parser.parse_args()
    if args.action in ('inspect', 'worker-image'):
        values = inspect(args.root) if args.action == 'inspect' else worker_image(args.root)
        for value in values:
            print(int(value) if isinstance(value, bool) else value)
    elif args.action == 'link':
        link(args.root, args.output)
    elif args.action == 'token':
        ensure_token(args.root)
    else:
        fields = sys.stdin.buffer.read().split(b'\0')
        if len(fields) != 4 or fields[-1] != b'':
            raise ValueError('Expected three setup input fields')
        prepare(args.root, *(field.decode('utf-8') for field in fields[:-1]))


if __name__ == '__main__':
    main()