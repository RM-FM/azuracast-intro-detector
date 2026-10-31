"""Fetch exact Debian source packages listed in a worker image inventory."""
import argparse
import csv
from email.parser import Parser
import hashlib
import os
from pathlib import Path
import re
import subprocess
import tempfile
from urllib.error import HTTPError
from urllib.parse import urljoin
from urllib.request import urlopen

from build_source_bundle import dsc_records
from verify_debian_sources import verify


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def required_sources(inventory):
    with Path(inventory).open(newline='') as stream:
        rows = csv.DictReader(stream, delimiter='\t')
        sources = set()
        for row in rows:
            source = row.get('source')
            version = row.get('source_version')
            if not source or not version:
                raise ValueError('Debian inventory contains a missing source name or version')
            sources.add((source, version))
    if not sources:
        raise ValueError('Debian inventory contains no source packages')
    return sorted(sources)


def apt_options(source_list):
    return [
        '-o', f'Dir::Etc::sourcelist={source_list}',
        '-o', 'Dir::Etc::sourceparts=-',
        '-o', 'APT::Get::List-Cleanup=0',
    ]


def source_pool_candidates(source, version, snapshot):
    filename = f'{source}_{version}.dsc'
    directory = source[:4] if source.startswith('lib') else source[0]
    sections = ('main', 'contrib', 'non-free', 'non-free-firmware')
    for archive, pool_prefix in (('debian', 'pool'), ('debian-security', 'pool/updates')):
        for section in sections:
            yield (f'https://snapshot.debian.org/archive/{archive}/{snapshot}/'
                   f'{pool_prefix}/{section}/{directory}/{source}/{filename}')


def fetch_from_snapshot_pool(source, version, snapshot, output):
    descriptor_name = f'{source}_{version}.dsc'
    output = Path(output)
    for descriptor_url in source_pool_candidates(source, version, snapshot):
        try:
            with urlopen(descriptor_url, timeout=60) as response:
                if not response.geturl().startswith('https://'):
                    raise ValueError(f'Insecure Debian source redirect: {response.geturl()}')
                descriptor = response.read()
        except HTTPError as error:
            if error.code == 404:
                continue
            raise

        marker = descriptor.find(b'Format:')
        if marker < 0:
            raise ValueError(f'Invalid Debian source descriptor: {descriptor_name}')
        text = descriptor[marker:].split(b'-----BEGIN PGP SIGNATURE-----', 1)[0].decode()
        document = Parser().parsestr(text)
        if document.get('Source') != source or document.get('Version') != version:
            raise ValueError(f'Debian source descriptor identity mismatch: {descriptor_name}')
        checksums = document.get('Checksums-Sha256')
        if not checksums:
            raise ValueError(f'Missing Debian source checksums: {descriptor_name}')

        base_url = descriptor_url.rsplit('/', 1)[0] + '/'
        for line in checksums.splitlines():
            if not line.strip():
                continue
            expected, size, name = line.split()
            if Path(name).name != name:
                raise ValueError(f'Unsafe Debian source filename: {name}')
            destination = output / name
            if destination.is_file() and destination.stat().st_size == int(size):
                if sha256(destination) == expected:
                    continue
            with urlopen(urljoin(base_url, name), timeout=60) as response:
                if not response.geturl().startswith('https://'):
                    raise ValueError(f'Insecure Debian source redirect: {response.geturl()}')
                data = response.read()
            if len(data) != int(size) or hashlib.sha256(data).hexdigest() != expected:
                raise ValueError(f'Debian source checksum mismatch: {name}')
            destination.write_bytes(data)
        (output / descriptor_name).write_bytes(descriptor)
        return
    raise FileNotFoundError(f'Exact Debian source not found in snapshot: {source} {version}')


def fetch(inventory, output, snapshot):
    if not re.fullmatch(r'\d{8}T\d{6}Z', snapshot):
        raise ValueError(f'Invalid Debian snapshot: {snapshot}')
    sources = required_sources(inventory)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    existing = dsc_records(output)
    with tempfile.TemporaryDirectory(prefix='intro-debian-sources-') as temporary_root:
        source_list = Path(temporary_root) / 'debian.sources.list'
        source_list.write_text(
            f'deb-src [signed-by=/usr/share/keyrings/debian-archive-keyring.gpg check-valid-until=no] '
            f'https://snapshot.debian.org/archive/debian/{snapshot}/ bookworm main\n'
            f'deb-src [signed-by=/usr/share/keyrings/debian-archive-keyring.gpg check-valid-until=no] '
            f'https://snapshot.debian.org/archive/debian/{snapshot}/ bookworm-updates main\n'
            f'deb-src [signed-by=/usr/share/keyrings/debian-archive-keyring.gpg check-valid-until=no] '
            f'https://snapshot.debian.org/archive/debian-security/{snapshot}/ bookworm-security main\n')
        options = apt_options(source_list)
        privilege = [] if os.geteuid() == 0 else ['sudo']
        subprocess.run([*privilege, 'apt-get', *options, 'update'], check=True)
        for source, version in sources:
            if (source, version) in existing:
                continue
            try:
                subprocess.run(
                    ['apt-get', *options, 'source', '--download-only', f'{source}={version}'],
                    cwd=output, capture_output=True, text=True, check=True)
            except subprocess.CalledProcessError:
                fetch_from_snapshot_pool(source, version, snapshot, output)
            existing = dsc_records(output)
    return verify(output, inventory)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--snapshot', required=True)
    arguments = parser.parse_args()
    count = fetch(arguments.inventory, arguments.output, arguments.snapshot)
    print(f'Verified Debian source packages: {count}')