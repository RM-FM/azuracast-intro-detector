"""Assemble a versioned corresponding-source archive for a built worker image."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
import tempfile

from verify_debian_sources import verify as verify_debian_sources


LICENSE_OVERRIDES = {
    'lameenc': 'LGPL-3.0-or-later',
    'soxr': 'LGPL-2.1-or-later',
    'panns-inference': 'MIT',
    'tokenizers': 'Apache-2.0',
    'triton': 'MIT',
}
PYTHON_SOURCES = {
    'lameenc': ('native/lameenc-1.9.0.tar.gz', 'native/lame-4.0.tar.gz'),
    'soxr': ('native/soxr-1.1.0.tar.gz', 'native/libsoxr-a66f3eee.tar.gz'),
}
NATIVE_BUILD_SOURCES = (
    'ffmpeg-8.1.2.tar.xz', 'ffmpeg-8.1.2.tar.xz.asc', 'ffmpeg-devel.asc',
    'lameenc-1.9.0.tar.gz', 'lame-4.0.tar.gz',
    'soxr-1.1.0.tar.gz', 'libsoxr-a66f3eee.tar.gz',
    'libsndfile-1.2.2.tar.xz', 'soundfile-0.13.1.tar.gz',
    'av-18.1.0.tar.gz', 'fribidi-1.0.8.tar.bz2', 'pillow-13.0.0-build-sources.tar.gz',
)
LICENSE_CLASSIFIERS = {
    'Apache Software License': 'Apache-2.0',
    'MIT License': 'MIT',
    'ISC License (ISCL)': 'ISC',
    'Python Software Foundation License': 'PSF-2.0',
    'Mozilla Public License 2.0 (MPL 2.0)': 'MPL-2.0',
    'zlib/libpng License': 'Zlib',
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def python_license(package, inventory_root, environment):
    name = package['name'].lower().replace('_', '-')
    if name in LICENSE_OVERRIDES:
        declared = LICENSE_OVERRIDES[name]
    else:
        declared = package.get('license_expression')
        if not declared:
            value = ' '.join((package.get('license') or '').split())
            if value and len(value) <= 160:
                declared = value
        if not declared:
            for classifier in package.get('classifiers', []):
                for text, license_name in LICENSE_CLASSIFIERS.items():
                    if classifier.endswith(text):
                        declared = license_name
                        break
                if declared:
                    break
        if not declared:
            for classifier in package.get('classifiers', []):
                if classifier.endswith('GNU General Public License v3 (GPLv3)'):
                    declared = 'GPL-3.0-only'
                    break

    license_texts = []
    for item in package.get('files', []):
        path = Path(inventory_root) / f'inventory/{environment}' / item['copied']
        if path.is_file():
            license_texts.append(path.read_text(errors='replace'))
    if not declared and not license_texts:
        raise ValueError(f'No license metadata or license files for {package["name"]} {package["version"]}')
    source_required = bool(declared and ('GPL' in declared.upper() or 'LGPL' in declared.upper()))
    return declared or 'See the exact upstream license files listed in this component record', source_required


def add_file(files, path, source):
    path = PurePosixPath(path).as_posix()
    source = Path(source)
    if path.startswith('/') or '..' in PurePosixPath(path).parts or not source.is_file():
        raise ValueError(f'Invalid source-bundle input: {path}')
    current = files.get(path)
    if current and current != source:
        if sha256(current) != sha256(source):
            raise ValueError(f'Conflicting source-bundle path: {path}')
    else:
        files[path] = source
    return dict(path=path, sha256=sha256(source))


def add_tree(files, root, prefix):
    root = Path(root)
    for path in sorted(root.rglob('*')):
        if path.is_symlink():
            raise ValueError(f'Refusing source-bundle symlink: {path}')
        if path.is_file():
            add_file(files, f'{prefix}/{path.relative_to(root).as_posix()}', path)


def dsc_records(directory):
    records = {}
    for dsc in Path(directory).glob('*.dsc'):
        source_name, source_version = None, None
        entries = []
        in_checksums = False
        for line in dsc.read_text().splitlines():
            if line.startswith('Source: '):
                source_name = line[8:].strip()
            elif line.startswith('Version: '):
                source_version = line[9:].strip()
            elif line == 'Checksums-Sha256:':
                in_checksums = True
            elif in_checksums:
                if not line.startswith(' '):
                    break
                digest, size, name = line.split()
                archive = Path(directory) / name
                if not archive.is_file() or archive.stat().st_size != int(size) or sha256(archive) != digest:
                    raise ValueError(f'Invalid Debian source archive: {name}')
                entries.append(archive)
        if not source_name or not source_version or not entries:
            raise ValueError(f'Incomplete Debian source descriptor: {dsc.name}')
        key = (source_name, source_version)
        if key in records:
            raise ValueError(f'Duplicate Debian source descriptor: {dsc.name}')
        records[key] = (dsc, entries)
    return records


def build(inventory, source_material, repository, context, output, image_id, revision):
    inventory = Path(inventory)
    source_material = Path(source_material)
    repository = Path(repository)
    files = {}
    context_temporary = tempfile.TemporaryDirectory(prefix='worker-source-context-')
    context_root = Path(context_temporary.name)
    with tarfile.open(context, 'r:*') as archive:
        for member in archive.getmembers():
            relative = PurePosixPath(member.name)
            if relative.is_absolute() or '..' in relative.parts or not member.isfile():
                raise ValueError(f'Unsafe worker-context entry: {member.name}')
            target = context_root.joinpath(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as source, target.open('wb') as destination:
                shutil.copyfileobj(source, destination)
    add_tree(files, context_root, 'source/repository')
    for relative in ('gpu/SOURCE_BUILD.md', 'gpu/verify_debian_sources.py',
                     'gpu/build_source_bundle.py', 'docs/NVIDIA_REDISTRIBUTION.md'):
        add_file(files, f'source/repository/{relative}', repository / relative)
    add_tree(files, inventory, 'image-legal')
    additional_sources = [add_file(files, f'source/native/{name}', source_material / 'native' / name)
                         for name in NATIVE_BUILD_SOURCES]
    add_tree(files, source_material / 'debian', 'source/debian')

    build_instructions = add_file(files, 'source/repository/gpu/SOURCE_BUILD.md',
                                  repository / 'gpu/SOURCE_BUILD.md')
    components = []
    for inventory_path in sorted(inventory.glob('inventory/*/python.json')):
        environment = inventory_path.parent.name
        data = json.loads(inventory_path.read_text())
        for package in data['packages']:
            license_name, source_required = python_license(package, inventory, environment)
            lower_license = license_name.lower()
            license_files = []
            for item in package.get('files', []):
                license_files.append(add_file(
                    files, f'image-legal/inventory/{environment}/{item["copied"]}',
                    inventory_path.parent / item['copied']))
            source_files = list(license_files)
            if source_required:
                names = PYTHON_SOURCES.get(package['name'].lower().replace('_', '-'))
                if not names:
                    raise ValueError(f'No corresponding-source mapping for {package["name"]} {package["version"]}')
                for name in names:
                    source_files.append(add_file(files, f'source/{name}', source_material / name))
            component = dict(
                id=f'python/{environment}/{package["name"]}@{package["version"]}',
                version=package['version'], license=license_name,
                origin=next((url.split(',', 1)[1].strip() for url in package.get('project_urls', [])
                             if url.lower().startswith(('source, https://', 'repository, https://', 'homepage, https://'))),
                            f'https://pypi.org/project/{package["name"]}/{package["version"]}/'),
                native_libraries=package.get('native_libraries', []),
                source_required=source_required, files=source_files)
            if source_required:
                component['replacement_instructions'] = add_file(
                    files, 'source/repository/gpu/SOURCE_BUILD.md', repository / 'gpu/SOURCE_BUILD.md')
            else:
                component['source_not_required_reason'] = 'The shipped license terms and notice files are included; no corresponding-source condition is identified for this component.'
            if package['name'].lower().startswith('nvidia-') or 'nvidia' in lower_license:
                if not license_files:
                    raise ValueError(f'Missing NVIDIA package license: {package["name"]}')
                component['redistribution_basis'] = add_file(
                    files, 'source/repository/docs/NVIDIA_REDISTRIBUTION.md',
                    repository / 'docs/NVIDIA_REDISTRIBUTION.md')
            components.append(component)

    debian_records = dsc_records(source_material / 'debian')
    debian_inventory = inventory / 'inventory/base/debian.tsv'
    source_count = verify_debian_sources(source_material / 'debian', debian_inventory)
    with debian_inventory.open() as stream:
        for package in csv.DictReader(stream, delimiter='\t'):
            copyright_name = package['binary'].split(':', 1)[0]
            copyright_path = inventory / 'inventory/base/debian-copyright' / copyright_name / 'copyright'
            if not copyright_path.is_file():
                raise ValueError(f'Missing Debian copyright: {copyright_name}')
            notice = add_file(files, f'image-legal/inventory/base/debian-copyright/{copyright_name}/copyright',
                              copyright_path)
            license_text = copyright_path.read_text(errors='replace')
            source_required = bool(re.search(r'\b(?:AGPL|GPL|LGPL)(?:-[0-9.]+)?\b', license_text, re.I))
            component = dict(
                id=f'debian/{package["binary"]}@{package["version"]}',
                version=package['version'],
                source=package['source'], source_version=package['source_version'],
                license='Declared in the exact Debian copyright file included with this component',
                origin=f'https://snapshot.debian.org/package/{package["source"]}/{package["source_version"]}/',
                native_libraries=[], source_required=source_required, files=[notice])
            if source_required:
                key = (package['source'], package['source_version'])
                if key not in debian_records:
                    raise ValueError(f'Missing exact Debian source: {key}')
                descriptor, source_archives = debian_records[key]
                component['files'].append(add_file(files, f'source/debian/{descriptor.name}', descriptor))
                component['files'].extend(add_file(files, f'source/debian/{source.name}', source)
                                          for source in source_archives)
            else:
                component['source_not_required_reason'] = 'The exact Debian copyright file is included; no GPL/LGPL source condition was detected.'
            if source_required and 'LGPL' in license_text.upper():
                component['replacement_instructions'] = add_file(
                    files, 'source/repository/gpu/SOURCE_BUILD.md', repository / 'gpu/SOURCE_BUILD.md')
            components.append(component)

    native_path = inventory / 'inventory/native/ffmpeg.json'
    native = json.loads(native_path.read_text())
    ffmpeg_source = native['source_archive']
    source_path = source_material / 'native/ffmpeg-8.1.2.tar.xz'
    source_record = add_file(files, 'source/native/ffmpeg-8.1.2.tar.xz', source_path)
    if source_record['sha256'] != ffmpeg_source['sha256']:
        raise ValueError('FFmpeg source archive differs from the native image inventory')
    components.append(dict(
        id=f'native/{native["name"]}@{native["version"]}', version=native['version'],
        license='LGPL-2.1-or-later', origin=ffmpeg_source['url'],
        native_libraries=native['native_libraries'], source_archive=ffmpeg_source,
        source_required=True, files=[source_record],
        replacement_instructions=add_file(files, 'source/repository/gpu/SOURCE_BUILD.md',
                                          repository / 'gpu/SOURCE_BUILD.md')))

    model_inventory = inventory / 'inventory/models.json'
    manifest = dict(schema=1, image_id=image_id, revision=revision,
                    source_packages_verified=source_count,
                    build_instructions=build_instructions, additional_sources=additional_sources,
                    components=components)
    if model_inventory.is_file():
        manifest['model_assets'] = json.loads(model_inventory.read_text())['assets']
    descriptor, manifest_name = tempfile.mkstemp(prefix='source-manifest-', suffix='.json')
    os.close(descriptor)
    manifest_path = Path(manifest_name)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    files['source-manifest.json'] = manifest_path
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('wb') as raw:
        import gzip
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode='w', format=tarfile.PAX_FORMAT) as archive:
                for name in sorted(files):
                    source = files[name]
                    member = tarfile.TarInfo(name)
                    member.size = source.stat().st_size
                    member.mtime = 0
                    member.uid = member.gid = 0
                    member.uname = member.gname = ''
                    member.mode = 0o644
                    with source.open('rb') as stream:
                        archive.addfile(member, stream)
    manifest_path.unlink(missing_ok=True)
    context_temporary.cleanup()
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--source-material', type=Path, required=True)
    parser.add_argument('--repository', type=Path, required=True)
    parser.add_argument('--worker-context', type=Path, required=True)
    parser.add_argument('--image-id', required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = build(args.inventory, args.source_material, args.repository,
                   args.worker_context, args.output, args.image_id, args.revision)
    print(f'Components: {len(result["components"])}; Debian sources: {result["source_packages_verified"]}')