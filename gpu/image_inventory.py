"""Export exact installed package versions, metadata and shipped license files.

Run once with each Python interpreter. --debian also records the binary/source
package versions and installed Debian copyright files. This is not a compliance
certification and does not download corresponding source archives.
"""
import argparse
import base64
import csv
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
from pathlib import PurePosixPath

UNLISTED_NVIDIA_LIBRARIES = {
    'nvidia-cuda-cupti-cu12': (
        ('nvidia/cuda_cupti/lib/libcheckpoint.so',
         'nvidia/cuda_cupti/lib/libnvperf_host.so',
         'nvidia/cuda_cupti/lib/libnvperf_target.so',
         'nvidia/cuda_cupti/lib/libpcsamplingutil.so'),
        'nvidia/cuda_cupti/lib/libcupti.so.12'),
    'nvidia-cusolver-cu12': (
        ('nvidia/cusolver/lib/libcusolverMg.so.11',),
        'nvidia/cusolver/lib/libcusolver.so.11'),
}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify_notices(directory):
    directory = Path(directory).resolve()
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('schema') != 1 or not manifest.get('files'):
        raise ValueError('Missing third-party license manifest')
    for entry in manifest['files']:
        path = (directory / entry['file']).resolve()
        if not path.is_relative_to(directory) or not path.is_file():
            raise ValueError(f'Invalid third-party notice path: {entry["file"]}')
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
            raise ValueError(f'Third-party notice checksum mismatch: {entry["file"]}')
    return manifest


def inventory(output, debian=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for dist in sorted(metadata.distributions(), key=lambda d: (d.metadata['Name'].lower(), str(d.locate_file('')))):
        name = dist.metadata['Name']
        row = dict(name=name, version=dist.version, location=str(dist.locate_file('')),
                   license=dist.metadata.get('License'),
                   license_expression=dist.metadata.get('License-Expression'),
                   classifiers=dist.metadata.get_all('Classifier', []),
                   project_urls=dist.metadata.get_all('Project-URL', []),
                   requires=dist.requires or [], files=[], native_libraries=[])
        for file in dist.files or []:
            if file.name.endswith('.so') or '.so.' in file.name or file.suffix.lower() in ('.dll', '.dylib', '.pyd', '.a'):
                source = Path(dist.locate_file(file))
                if not source.is_file():
                    raise FileNotFoundError(f'Missing installed native library: {source}')
                row['native_libraries'].append(dict(installed=str(file), sha256=file_sha256(source)))
            if not file.name.lower().startswith(('license', 'licence', 'notice', 'copying', 'copyright', 'eula', 'authors')):
                continue
            source = Path(dist.locate_file(file))
            if not source.is_file():
                raise FileNotFoundError(f'Missing installed license file: {source}')
            target = output / 'licenses' / name / f'{len(row["files"]):04}-{source.name}'
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            row['files'].append(dict(installed=str(file), copied=str(target.relative_to(output)),
                                     sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
        rows.append(row)
    (output / 'python.json').write_text(json.dumps(
        dict(python=sys.version, executable=sys.executable, packages=rows), indent=2) + '\n')
    (output / 'requirements.txt').write_text(''.join(f'{r["name"]}=={r["version"]}\n' for r in rows))
    if debian:
        result = subprocess.run(
            ['dpkg-query', '-W', '-f=${binary:Package}\t${Version}\t${source:Package}\t${source:Version}\n'],
            check=True, capture_output=True, text=True)
        (output / 'debian.tsv').write_text('binary\tversion\tsource\tsource_version\n' + result.stdout)
        shutil.copytree('/usr/share/common-licenses', output / 'common-licenses', dirs_exist_ok=True)
        for source in sorted(Path('/usr/share/doc').glob('*/copyright')):
            target = output / 'debian-copyright' / source.parent.name / 'copyright'
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    return rows


def model_inventory(roots, output):
    assets = []
    seen = set()
    denied = {'.env', 'token', 'credentials', 'credentials.json', 'secret', 'secrets',
              'config.toml', 'annotations.json', 'review.json', 'reviews.json'}
    denied_suffixes = {'.mp3', '.flac', '.wav', '.ogg', '.m4a', '.aac', '.db', '.sqlite', '.sqlite3'}
    for label, root in roots:
        root = Path(root)
        if not root.is_dir() or root.is_symlink():
            raise ValueError(f'Invalid model asset root: {label}')
        for path in sorted(root.rglob('*')):
            if path.name.lower() in denied or path.suffix.lower() in denied_suffixes:
                raise ValueError(f'Unexpected private or non-model asset: {path}')
            if not path.is_file():
                continue
            resolved = path.resolve()
            if not resolved.is_relative_to(root.resolve()):
                raise ValueError(f'Model asset symlink escapes inventory root: {path}')
            if resolved in seen:
                continue
            seen.add(resolved)
            record = dict(path=f'{label}/{path.relative_to(root).as_posix()}',
                          sha256=file_sha256(resolved), size=resolved.stat().st_size)
            if path.is_symlink():
                record['symlink_target'] = path.readlink().as_posix()
            assets.append(record)
    if not assets:
        raise ValueError('No model assets were inventoried')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(schema=1, assets=assets), indent=2, sort_keys=True) + '\n')
    return assets


def native_inventory(prefix, name, version, source_url, source_sha256, output):
    prefix = Path(prefix)
    libraries = [dict(installed=str(path.relative_to(prefix)), sha256=file_sha256(path))
                 for path in sorted(prefix.rglob('*')) if path.is_file()
                 and (path.name.endswith('.so') or '.so.' in path.name)]
    if not libraries:
        raise ValueError('Missing native libraries')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    record = dict(name=name, version=version, native_libraries=libraries,
                  source_archive=dict(url=source_url, sha256=source_sha256))
    (output / f'{name}.json').write_text(json.dumps(record, indent=2) + '\n')
    return record


def prune_unlisted_nvidia_libraries(output):
    available = {((dist.metadata.get('Name') or '').lower()): dist
                 for dist in metadata.distributions()}
    reports = []
    for package_name, (unlisted_paths, retained_path) in UNLISTED_NVIDIA_LIBRARIES.items():
        distribution = available.get(package_name)
        if distribution is None:
            raise ValueError(f'Missing {package_name} distribution')
        package_paths = {PurePosixPath(str(file)): Path(distribution.locate_file(file))
                         for file in distribution.files or []}
        selected_paths = {PurePosixPath(path) for path in unlisted_paths}
        selected = {relative: package_paths[relative] for relative in selected_paths if relative in package_paths}
        if set(selected) != selected_paths:
            raise ValueError(f'Unexpected unlisted-library set in {package_name}')
        retained = PurePosixPath(retained_path)
        if retained not in package_paths or not package_paths[retained].is_file():
            raise ValueError(f'Missing retained NVIDIA runtime library: {retained_path}')
        record_path = next((path for relative, path in package_paths.items()
                            if relative.name == 'RECORD' and relative.parent.name.endswith('.dist-info')), None)
        if record_path is None:
            raise ValueError(f'Missing {package_name} RECORD')
        removed = []
        for relative, path in sorted(selected.items()):
            if not path.is_file():
                raise FileNotFoundError(f'Missing unlisted NVIDIA library: {path}')
            removed.append(dict(path=relative.as_posix(), sha256=file_sha256(path)))
            path.unlink()
        with record_path.open(newline='') as stream:
            rows = list(csv.reader(stream))
        removed_names = {relative.as_posix() for relative in selected}
        if not removed_names.issubset({row[0] for row in rows if row}):
            raise ValueError(f'Incomplete {package_name} RECORD')
        with record_path.open('w', newline='') as stream:
            csv.writer(stream).writerows(row for row in rows if not row or row[0] not in removed_names)
        reports.append(dict(package=package_name, version=distribution.version, removed=removed,
                            retained=retained.as_posix()))
    report = dict(schema=1, packages=reports)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')
    return report


def replace_native_runtimes(runtimes, distributions):
    replacements = []
    for dist in distributions:
        changed = {}
        for installed in dist.files or []:
            stem = next((name for name in runtimes if installed.name.startswith(name) and '.so' in installed.name), None)
            if stem is None:
                continue
            target = Path(dist.locate_file(installed))
            source = Path(runtimes[stem]['file'])
            before = file_sha256(target)
            digest = file_sha256(source)
            if before == digest:
                continue
            shutil.copyfile(source, target)
            changed[str(installed)] = ('sha256=' + base64.urlsafe_b64encode(bytes.fromhex(digest)).decode().rstrip('='),
                                       str(target.stat().st_size))
            replacements.append(dict(package=dist.metadata['Name'], installed=str(installed),
                                     original_sha256=before, sha256=digest,
                                     debian=runtimes[stem]['debian']))
        if changed:
            record = next((Path(dist.locate_file(file)) for file in dist.files or []
                           if file.name == 'RECORD' and file.parent.name.endswith('.dist-info')), None)
            if record:
                with record.open(newline='') as stream:
                    rows = list(csv.reader(stream))
                for row in rows:
                    if row[0] in changed:
                        row[1:] = changed[row[0]]
                with record.open('w', newline='') as stream:
                    csv.writer(stream).writerows(rows)
    return replacements


def normalize_native_runtimes(output, names=None):
    runtimes = {}
    packages = (('libgomp', 'libgomp1'), ('libgfortran', 'libgfortran5'),
                ('libquadmath', 'libquadmath0'), ('libsndfile', 'libsndfile1'),
                ('libsoxr', 'libsoxr0'))
    for stem, package in packages:
        if names is not None and stem not in names:
            continue
        files = subprocess.run(['dpkg-query', '-L', package], check=True, capture_output=True, text=True).stdout.splitlines()
        candidates = {Path(file).resolve() for file in files if Path(file).name.startswith(stem + '.so')
                      and Path(file).is_file()}
        if len(candidates) != 1:
            raise ValueError(f'Cannot identify Debian runtime: {package}')
        values = subprocess.run(['dpkg-query', '-W', '-f=${Version}\t${source:Package}\t${source:Version}', package],
                                check=True, capture_output=True, text=True).stdout.split('\t')
        runtimes[stem] = dict(file=str(candidates.pop()), debian=dict(
            binary=package, version=values[0], source=values[1], source_version=values[2]))
    replacements = replace_native_runtimes(runtimes, metadata.distributions())
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(replacements, indent=2) + '\n')
    return replacements


def verify_source_bundle(archive, directory, image_id, revision):
    expected = {}
    for path in sorted(Path(directory).glob('inventory/*/python.json')):
        for package in json.loads(path.read_text())['packages']:
            identifier = f'python/{path.parent.name}/{package["name"]}@{package["version"]}'
            expected[identifier] = dict(version=package['version'],
                                        native_libraries=package.get('native_libraries', []))
    for path in sorted(Path(directory).glob('inventory/*/debian.tsv')):
        with path.open() as stream:
            for package in csv.DictReader(stream, delimiter='\t'):
                identifier = f'debian/{package["binary"]}@{package["version"]}'
                expected[identifier] = dict(version=package['version'], native_libraries=[],
                                            source=package['source'], source_version=package['source_version'])
    for path in sorted(Path(directory).glob('inventory/native/*.json')):
        package = json.loads(path.read_text())
        identifier = f'native/{package["name"]}@{package["version"]}'
        expected[identifier] = dict(version=package['version'], native_libraries=package['native_libraries'],
                                    source_archive=package['source_archive'])
    if not expected:
        raise ValueError('Missing image package inventory')
    with tarfile.open(archive, 'r:*') as bundle:
        members = {}
        for member in bundle.getmembers():
            path = PurePosixPath(member.name)
            if path.is_absolute() or '..' in path.parts or member.name in members:
                raise ValueError('Invalid or duplicate source-bundle path')
            members[member.name] = member

        def read_manifest():
            member = members.get('source-manifest.json')
            if member is None or not member.isfile() or member.size > 10 * 1024 * 1024:
                raise ValueError('Missing or oversized source-manifest.json')
            with bundle.extractfile(member) as stream:
                return json.load(stream)

        def verify_file(entry):
            if not isinstance(entry, dict) or not entry.get('path') or not entry.get('sha256'):
                raise ValueError('Missing source-bundle file record')
            member = members.get(entry['path'])
            if member is None or not member.isfile() or not member.size:
                raise ValueError(f'Missing source-bundle file: {entry["path"]}')
            digest = hashlib.sha256()
            with bundle.extractfile(member) as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(chunk)
            if digest.hexdigest() != entry['sha256']:
                raise ValueError(f'Source-bundle checksum mismatch: {entry["path"]}')

        manifest = read_manifest()
        if manifest.get('schema') != 1 or manifest.get('image_id') != image_id or manifest.get('revision') != revision:
            raise ValueError('Source bundle does not match the reviewed image and revision')
        for entry in manifest.get('additional_sources', []):
            verify_file(entry)
        model_inventory_path = Path(directory) / 'inventory/models.json'
        if model_inventory_path.is_file() and manifest.get('model_assets') != json.loads(
                model_inventory_path.read_text())['assets']:
            raise ValueError('Source-bundle model assets differ from the image inventory')
        components = manifest.get('components', [])
        identifiers = [component['id'] for component in components]
        if len(set(identifiers)) != len(identifiers) or set(identifiers) != set(expected):
            raise ValueError('Source-bundle component coverage differs from the image inventory')
        verify_file(manifest['build_instructions'])
        for component in components:
            actual = expected[component['id']]
            if any(component.get(key) != value for key, value in actual.items()):
                raise ValueError(f'Source-bundle version or native-library mismatch: {component["id"]}')
            license_name = component.get('license', '')
            if not license_name or not component.get('origin', '').startswith('https://'):
                raise ValueError(f'Missing component license or origin: {component["id"]}')
            files = component.get('files', [])
            if component.get('source_required') not in (True, False):
                raise ValueError(f'Missing source obligation review: {component["id"]}')
            if component['source_required'] or 'GPL' in license_name.upper():
                if not files:
                    raise ValueError(f'Missing corresponding source: {component["id"]}')
            elif not component.get('source_not_required_reason'):
                raise ValueError(f'Missing source obligation rationale: {component["id"]}')
            for entry in files:
                verify_file(entry)
            if actual.get('source_archive') and not any(
                    entry['sha256'] == actual['source_archive']['sha256'] for entry in files):
                raise ValueError(f'Missing exact native source archive: {component["id"]}')
            if 'LGPL' in license_name.upper():
                verify_file(component.get('replacement_instructions'))
            if '/nvidia-' in component['id'].lower() or 'NVIDIA' in license_name.upper():
                basis = component.get('redistribution_basis')
                verify_file(basis)
                member = members[basis['path']]
                with bundle.extractfile(member) as stream:
                    decisions = [line.strip() for line in stream.read().decode().splitlines()
                                 if line.startswith('REDISTRIBUTION DECISION:')]
                if decisions != ['REDISTRIBUTION DECISION: APPROVED']:
                    raise ValueError(f'NVIDIA redistribution review is not approved: {component["id"]}')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--debian', action='store_true')
    parser.add_argument('--notices', type=Path)
    parser.add_argument('--source-bundle', type=Path)
    parser.add_argument('--image-id')
    parser.add_argument('--revision')
    parser.add_argument('--native-prefix', type=Path)
    parser.add_argument('--native-name')
    parser.add_argument('--native-version')
    parser.add_argument('--native-source-url')
    parser.add_argument('--native-source-sha256')
    parser.add_argument('--normalize-gnu-runtimes', action='store_true')
    parser.add_argument('--normalize-native-runtimes', action='store_true')
    parser.add_argument('--prune-unlisted-nvidia-libraries', action='store_true')
    parser.add_argument('--model-root', action='append', default=[], metavar='NAME=PATH')
    args = parser.parse_args()
    if args.notices:
        verify_notices(args.notices)
    if args.normalize_gnu_runtimes:
        normalize_native_runtimes(args.output, {'libgomp', 'libgfortran', 'libquadmath'})
    elif args.normalize_native_runtimes:
        normalize_native_runtimes(args.output)
    elif args.prune_unlisted_nvidia_libraries:
        prune_unlisted_nvidia_libraries(args.output)
    elif args.model_root:
        roots = []
        for value in args.model_root:
            name, separator, path = value.partition('=')
            if not separator or not name or not path:
                parser.error('Each --model-root must be NAME=PATH')
            roots.append((name, path))
        model_inventory(roots, args.output)
    elif args.native_prefix:
        if not all((args.native_name, args.native_version, args.native_source_url, args.native_source_sha256)):
            parser.error('--native-prefix requires name, version, source URL and SHA256')
        native_inventory(args.native_prefix, args.native_name, args.native_version,
                         args.native_source_url, args.native_source_sha256, args.output)
    elif args.source_bundle:
        if not args.image_id or not args.revision:
            parser.error('--source-bundle requires --image-id and --revision')
        verify_source_bundle(args.source_bundle, args.output, args.image_id, args.revision)
    else:
        inventory(args.output, args.debian)
