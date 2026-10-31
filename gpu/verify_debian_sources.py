import argparse
import csv
from email.parser import Parser
import hashlib
from pathlib import Path


def verify(directory, inventory):
    directory = Path(directory).resolve()
    sources = set()
    for descriptor in sorted(directory.glob('*.dsc')):
        text = descriptor.read_text()
        start = text.find('Format:')
        if start < 0:
            raise ValueError(f'Missing Debian source format: {descriptor.name}')
        document = Parser().parsestr(text[start:].split('-----BEGIN PGP SIGNATURE-----', 1)[0])
        identity = (document['Source'], document['Version'])
        if not all(identity) or not document['Checksums-Sha256']:
            raise ValueError(f'Incomplete source descriptor: {descriptor.name}')
        for line in document['Checksums-Sha256'].splitlines():
            if not line.strip():
                continue
            expected, size, name = line.split()
            path = (directory / name).resolve()
            if not path.is_relative_to(directory) or not path.is_file() or path.stat().st_size != int(size):
                raise ValueError(f'Missing or invalid source archive: {name}')
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(chunk)
            if digest.hexdigest() != expected:
                raise ValueError(f'Source archive checksum mismatch: {name}')
        sources.add(identity)
    with Path(inventory).open() as stream:
        required = {(row['source'], row['source_version']) for row in csv.DictReader(stream, delimiter='\t')}
    if not required or required - sources:
        raise ValueError(f'Missing exact Debian sources: {sorted(required - sources)}')
    return len(required)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    parser.add_argument('inventory', type=Path)
    args = parser.parse_args()
    print(f'Verified Debian source packages: {verify(args.directory, args.inventory)}')