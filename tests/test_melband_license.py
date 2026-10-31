"""Third-party licensing checks without importing PyTorch or downloading models."""
import importlib.util
import ast
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import unittest
from unittest.mock import Mock, patch
from email.message import Message
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('license_melband', ROOT / 'pod/tools/melband_backend.py')
backend = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(backend)
INVENTORY_SPEC = importlib.util.spec_from_file_location('image_inventory', ROOT / 'gpu/image_inventory.py')
image_inventory = importlib.util.module_from_spec(INVENTORY_SPEC)
INVENTORY_SPEC.loader.exec_module(image_inventory)
VERIFY_DEBIAN_SPEC = importlib.util.spec_from_file_location('verify_debian_sources', ROOT / 'gpu/verify_debian_sources.py')
verify_debian_module = importlib.util.module_from_spec(VERIFY_DEBIAN_SPEC)
VERIFY_DEBIAN_SPEC.loader.exec_module(verify_debian_module)
import sys
sys.path.insert(0, str(ROOT / 'gpu'))
SOURCE_BUNDLE_SPEC = importlib.util.spec_from_file_location('build_source_bundle', ROOT / 'gpu/build_source_bundle.py')
source_bundle = importlib.util.module_from_spec(SOURCE_BUNDLE_SPEC)
SOURCE_BUNDLE_SPEC.loader.exec_module(source_bundle)
NATIVE_SOURCE_SPEC = importlib.util.spec_from_file_location(
    'fetch_native_sources', ROOT / 'gpu/fetch_native_sources.py')
native_sources = importlib.util.module_from_spec(NATIVE_SOURCE_SPEC)
NATIVE_SOURCE_SPEC.loader.exec_module(native_sources)
DEBIAN_FETCH_SPEC = importlib.util.spec_from_file_location(
    'fetch_debian_sources', ROOT / 'gpu/fetch_debian_sources.py')
debian_fetch = importlib.util.module_from_spec(DEBIAN_FETCH_SPEC)
DEBIAN_FETCH_SPEC.loader.exec_module(debian_fetch)
PREFETCH_SPEC = importlib.util.spec_from_file_location('image_prefetch', ROOT / 'gpu/prefetch.py')
prefetch = importlib.util.module_from_spec(PREFETCH_SPEC)
PREFETCH_SPEC.loader.exec_module(prefetch)
DEBIAN_SPEC = importlib.util.spec_from_file_location('verify_debian_sources', ROOT / 'gpu/verify_debian_sources.py')
debian_sources = importlib.util.module_from_spec(DEBIAN_SPEC)
DEBIAN_SPEC.loader.exec_module(debian_sources)


class ImageInventory(unittest.TestCase):
    def test_unlisted_nvidia_pruning_keeps_runtimes_and_updates_wheel_records(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package_specs = {
                'nvidia-cuda-cupti-cu12': ('12.1.105', *image_inventory.UNLISTED_NVIDIA_LIBRARIES[
                    'nvidia-cuda-cupti-cu12']),
                'nvidia-cusolver-cu12': ('11.4.5.107', *image_inventory.UNLISTED_NVIDIA_LIBRARIES[
                    'nvidia-cusolver-cu12']),
            }
            distributions = []
            retained_files = []
            record_files = []
            for package_name, (version, unlisted_paths, retained_path) in package_specs.items():
                package_files = []
                rows = []
                for path in (*unlisted_paths, retained_path):
                    relative = Path(path)
                    target = root / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(path.encode())
                    package_files.append(relative)
                    rows.append(f'{relative.as_posix()},sha256=old,16\n')
                record_relative = Path(package_name.replace('-', '_') + f'-{version}.dist-info/RECORD')
                record_path = root / record_relative
                record_path.parent.mkdir(parents=True)
                record_path.write_text(''.join(rows) + f'{record_relative.as_posix()},,\n')
                package_files.append(record_relative)
                metadata = Message()
                metadata['Name'] = package_name
                distributions.append(SimpleNamespace(metadata=metadata, version=version, files=package_files,
                                                     locate_file=lambda file: root / file))
                retained_files.append(root / retained_path)
                record_files.append(record_path)
            with patch.object(image_inventory.metadata, 'distributions', return_value=distributions):
                report = image_inventory.prune_unlisted_nvidia_libraries(
                    root / 'inventory/nvidia-pruned/base.json')

            self.assertEqual({entry['package'] for entry in report['packages']}, set(package_specs))
            self.assertEqual([len(entry['removed']) for entry in report['packages']], [4, 1])
            self.assertTrue(all(path.is_file() for path in retained_files))
            self.assertTrue(all(not (root / path).exists()
                                for _, (_, paths, _) in package_specs.items() for path in paths))
            self.assertTrue(all(not any(path in record.read_text() for _, (_, paths, _) in
                                       package_specs.items() for path in paths) for record in record_files))

    def test_model_inventory_hashes_only_model_assets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            assets = root / 'models'
            assets.mkdir()
            (assets / 'weights.safetensors').write_bytes(b'neutral-model')
            (assets / 'config.json').write_text('{}')
            output = root / 'inventory/models.json'
            records = image_inventory.model_inventory([('qwen', assets)], output)
            manifest = __import__('json').loads(output.read_text())
            self.assertEqual(len(records), 2)
            self.assertEqual(manifest['assets'][1]['path'], 'qwen/weights.safetensors')
            (assets / 'sample.wav').write_bytes(b'private-audio')
            with self.assertRaisesRegex(ValueError, 'Unexpected private'):
                image_inventory.model_inventory([('qwen', assets)], output)

    def test_model_inventory_rejects_symlink_outside_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            assets = root / 'models'
            assets.mkdir()
            outside = root / 'private.json'
            outside.write_text('{"secret":"neutral"}')
            (assets / 'config.json').symlink_to(outside)
            with self.assertRaisesRegex(ValueError, 'escapes inventory root'):
                image_inventory.model_inventory([('model', assets)], root / 'inventory.json')

    def test_runtime_replacement_records_source_and_updates_wheel_record(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'libgomp-example.so.1').write_bytes(b'upstream-runtime')
            (root / 'libsndfile_x86_64.so').write_bytes(b'upstream-sndfile')
            (root / 'debian.so').write_bytes(b'debian-runtime')
            record = root / 'example.dist-info/RECORD'
            record.parent.mkdir()
            record.write_text('libgomp-example.so.1,sha256=old,16\n'
                              'libsndfile_x86_64.so,sha256=old,16\n'
                              'soxr_ext.cpython-311-x86_64-linux-gnu.so,sha256=old,16\n'
                              'example.dist-info/RECORD,,\n')
            info = Message()
            info['Name'] = 'example'
            dist = SimpleNamespace(metadata=info, files=[Path('libgomp-example.so.1'),
                                   Path('libsndfile_x86_64.so'), Path('soxr_ext.cpython-311-x86_64-linux-gnu.so'),
                                   Path('example.dist-info/RECORD')],
                                   locate_file=lambda file: root / file)
            source = dict(binary='libgomp1', version='1.0', source='gcc', source_version='1.0')
            replacements = image_inventory.replace_native_runtimes(
                {'libgomp': dict(file=str(root / 'debian.so'), debian=source),
                 'libsndfile': dict(file=str(root / 'debian.so'), debian=source),
                 'libsoxr': dict(file=str(root / 'debian.so'), debian=source)}, [dist])
            self.assertEqual((root / 'libgomp-example.so.1').read_bytes(), b'debian-runtime')
            self.assertEqual((root / 'libsndfile_x86_64.so').read_bytes(), b'debian-runtime')
            self.assertFalse((root / 'soxr_ext.cpython-311-x86_64-linux-gnu.so').exists())
            self.assertEqual(replacements[0]['debian'], source)
            self.assertEqual(len(replacements), 2)
            self.assertTrue(all(item['sha256'] == hashlib.sha256(b'debian-runtime').hexdigest()
                                for item in replacements))
            self.assertIn('soxr_ext.cpython-311-x86_64-linux-gnu.so,sha256=old,16', record.read_text())
            self.assertNotIn('libsndfile_x86_64.so,sha256=old,16', record.read_text())
            self.assertEqual(image_inventory.replace_native_runtimes(
                {'libgomp': dict(file=str(root / 'debian.so'), debian=source),
                 'libsndfile': dict(file=str(root / 'debian.so'), debian=source),
                 'libsoxr': dict(file=str(root / 'debian.so'), debian=source)}, [dist]), [])

    def test_native_build_records_its_exact_source_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prefix = root / 'library'
            prefix.mkdir()
            (prefix / 'libexample.so.1').write_bytes(b'neutral-library')
            record = image_inventory.native_inventory(prefix, 'example', '1.0',
                'https://example.invalid/source.tar', 'a' * 64, root / 'inventory')
            self.assertEqual(record['source_archive'], dict(
                url='https://example.invalid/source.tar', sha256='a' * 64))
            self.assertEqual(record['native_libraries'], [dict(
                installed='libexample.so.1', sha256=hashlib.sha256(b'neutral-library').hexdigest())])

    def test_upstream_notice_bundle_is_complete_and_unmodified(self):
        manifest = image_inventory.verify_notices(ROOT / 'licenses')
        files = {entry['file']: entry for entry in manifest['files']}
        self.assertIn('Apache-2.0.txt', files)
        self.assertIn('GPL-3.0.txt', files)
        self.assertIn('CC-BY-4.0.txt', files)
        self.assertIn(backend.WEIGHT_REVISION, files['model-cards/melbandroformer.md']['url'])
        self.assertIn(backend.REVISION, files['upstream/msst-LICENSE.txt']['url'])

    def test_modified_notice_fails_verification(self):
        import json

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'LICENSE').write_text('changed')
            (root / 'manifest.json').write_text(json.dumps(dict(schema=1, files=[
                dict(file='LICENSE', sha256='0' * 64)])))
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                image_inventory.verify_notices(root)

    def test_preserves_eula_notice_and_author_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            names = ('EULA.txt', 'NOTICE', 'AUTHORS', 'LICENSE.txt')
            for name in names:
                (root / name).write_text(name)
            metadata = Message()
            metadata['Name'] = 'example-package'
            dist = SimpleNamespace(metadata=metadata, version='1.0', requires=[],
                                   files=[Path(name) for name in names], locate_file=lambda name: root / name)
            with patch.object(image_inventory.metadata, 'distributions', return_value=[dist]):
                rows = image_inventory.inventory(root / 'output')
            self.assertEqual(len(rows[0]['files']), len(names))
            for entry in rows[0]['files']:
                self.assertEqual((root / 'output' / entry['copied']).read_text(), entry['installed'])

    def test_native_library_bytes_are_inventoried(self):
        import hashlib

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'libexample.so.1').write_bytes(b'neutral-library')
            metadata = Message()
            metadata['Name'] = 'example-package'
            dist = SimpleNamespace(metadata=metadata, version='1.0', requires=[],
                                   files=[Path('libexample.so.1')], locate_file=lambda name: root / name)
            with patch.object(image_inventory.metadata, 'distributions', return_value=[dist]):
                rows = image_inventory.inventory(root / 'output')
            self.assertEqual(rows[0]['native_libraries'], [dict(
                installed='libexample.so.1', sha256=hashlib.sha256(b'neutral-library').hexdigest())])


@unittest.skipUnless(os.name == 'posix' and shutil.which('git'), 'Build context checks require Linux and git')
class ImageContext(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / 'bin').mkdir()
        shutil.copyfile(ROOT / 'bin/image-context.sh', self.root / 'bin/image-context.sh')
        self.files = ('LICENSE', 'THIRD_PARTY_NOTICES.md', 'docs/LICENSES.md', 'docs/MODELS.md',
                      'licenses/manifest.json', 'gpu/image_inventory.py', 'pod/experiments/runpod-start.sh',
                      'pod/tools/melband_backend.py', 'pod/tools/intro_review.html',
                      'Dockerfile', 'pyproject.toml', 'MANIFEST.in', 'docker-entrypoint.sh',
                      'src/intro_detector/__init__.py', 'gpu/Dockerfile', 'gpu/prefetch.py',
                      'gpu/requirements-build.lock', 'gpu/requirements-base.lock', 'gpu/requirements-stem.lock',
                      'gpu/requirements-qwen.lock', 'gpu/requirements-qwen-asr.lock',
                      'gpu/requirements-qwen-build.lock', 'gpu/requirements-soxr-build.lock',
                      'gpu/requirements-soxr.lock',
                      'pod/experiments/requirements-stem-comparison.txt',
                      'pod/experiments/requirements-qwen.txt')
        for name in self.files:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('neutral-source\n')
        subprocess.run(['git', 'init', '-q', str(self.root)], check=True)
        subprocess.run(['git', '-C', str(self.root), 'add', '.'], check=True)

    def tearDown(self):
        self.tmp.cleanup()

    def build(self, kind, output):
        return subprocess.run(['bash', str(self.root / 'bin/image-context.sh'), kind, str(output)],
                              capture_output=True, text=True)

    def test_private_and_unversioned_files_are_excluded(self):
        for name in ('pod/tools/.env', 'pod/tools/audio.flac', 'pod/tools/state.sqlite3',
                     'pod/tools/report.json', 'pod/experiments/vocal-references.json'):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('private-marker')
        subprocess.run(['git', '-C', str(self.root), 'add', '.'], check=True)
        (self.root / 'pod/tools/unversioned.py').write_text('private-marker')
        (self.root / 'pod/tools/apply_listening_labels.py').write_text('LISTENED = {"neutral": 1}\n')
        subprocess.run(['git', '-C', str(self.root), 'add', 'pod/tools/apply_listening_labels.py'], check=True)
        for kind in ('worker', 'service'):
            output = self.root / f'{kind}.tar'
            result = self.build(kind, output)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertNotIn(b'private-marker', output.read_bytes())
            with tarfile.open(output) as archive:
                self.assertIn('pod/experiments/runpod-start.sh', archive.getnames())
                self.assertTrue(all(member.mtime == 0 and member.uid == 0 for member in archive))
                self.assertNotIn('pod/tools/apply_listening_labels.py', archive.getnames())
                self.assertNotIn('pod/experiments/vocal-references.json', archive.getnames())
                self.assertEqual('pod/tools/intro_review.html' in archive.getnames(), kind == 'service')
            repeated = self.root / f'{kind}-repeat.tar'
            self.assertEqual(self.build(kind, repeated).returncode, 0)
            self.assertEqual(output.read_bytes(), repeated.read_bytes())

    def test_symlinked_source_is_rejected(self):
        path = self.root / 'gpu/prefetch.py'
        path.unlink()
        path.symlink_to(self.root / 'LICENSE')
        result = self.build('worker', self.root / 'worker.tar')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('symlinked', result.stderr)

    def test_missing_versioned_input_is_rejected(self):
        subprocess.run(['git', '-C', str(self.root), 'rm', '-q', '-f', 'gpu/prefetch.py'], check=True)
        result = self.build('worker', self.root / 'worker.tar')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Missing versioned build input', result.stderr)


class PrefetchedAssets(unittest.TestCase):
    def test_model_coordinates_match_the_runtime_sources(self):
        def constants(path):
            return {node.targets[0].id: ast.literal_eval(node.value)
                    for node in ast.parse(path.read_text()).body
                    if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                    and isinstance(node.value, ast.Constant)}

        smad = constants(ROOT / 'pod/tools/smad_experiment.py')
        qwen = constants(ROOT / 'pod/tools/qwen_experiment.py')
        self.assertEqual((prefetch.SMAD_MODEL, prefetch.SMAD_REVISION), (smad['MODEL_ID'], smad['MODEL_REVISION']))
        self.assertEqual(set(prefetch.QWEN_REVISIONS), {qwen['ASR_MODEL'], qwen['ALIGN_MODEL']})
        runtime = ast.parse((ROOT / 'pod/tools/ultimate_inference.py').read_text())
        self.assertIn(prefetch.PANNS_URL, [node.value for node in ast.walk(runtime)
                                        if isinstance(node, ast.Constant) and isinstance(node.value, str)])

    def test_cached_panns_fingerprint_and_lock_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'panns/Cnn14_DecisionLevelMax.pth'
            path.parent.mkdir()
            path.write_bytes(b'neutral-checkpoint')
            with patch.object(prefetch.urllib.request, 'urlopen', side_effect=AssertionError('Network forbidden')):
                with patch.object(prefetch, 'PANNS_SHA256', hashlib.sha256(path.read_bytes()).hexdigest()):
                    actual, digest = prefetch.panns_checkpoint(root)
                    self.assertEqual(actual, path)
                    self.assertEqual(digest, hashlib.sha256(path.read_bytes()).hexdigest())
                    self.assertEqual(json.loads((path.parent / 'weights.json').read_text()), dict(sha256=digest))
                    prefetch.panns_checkpoint(root)
                    path.write_bytes(b'changed-checkpoint')
                    with patch.object(prefetch, 'PANNS_SHA256', hashlib.sha256(path.read_bytes()).hexdigest()):
                        with self.assertRaisesRegex(ValueError, 'changed from recorded hash'):
                            prefetch.panns_checkpoint(root)

    def test_incomplete_panns_download_is_removed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(prefetch.urllib.request, 'urlopen', return_value=io.BytesIO(b'incomplete')):
                with self.assertRaisesRegex(ValueError, 'Incomplete'):
                    prefetch.panns_checkpoint(root)
            self.assertEqual(list((root / 'panns').iterdir()), [])

    def test_panns_release_hash_is_pinned(self):
        self.assertEqual(prefetch.PANNS_SHA256,
                         'dd3b4043a87d4ec13df8082c0fcfee3fb5084151808e47e060987a95eabdd142')
        self.assertIn('Cnn14_DecisionLevelMax_mAP%3D0.385.pth', prefetch.PANNS_URL)

    def test_qwen_prefetch_does_not_import_private_evaluation_modules(self):
        import sys

        with tempfile.TemporaryDirectory() as directory:
            hub = SimpleNamespace(hf_hub_download=Mock(return_value='cached-file'),
                                  snapshot_download=Mock(return_value='cached-model'))
            with patch.dict(sys.modules, {'huggingface_hub': hub, 'qwen_experiment': None,
                                         'smad_experiment': None, 'ultimate_inference': None}), \
                    patch.object(prefetch, 'EVIDENCE', Path(directory)), patch('builtins.print'):
                prefetch.qwen()
            self.assertEqual(hub.snapshot_download.call_count, len(prefetch.QWEN_REVISIONS))
            for repo, revision in prefetch.QWEN_REVISIONS.items():
                hub.snapshot_download.assert_any_call(repo, revision=revision,
                                                      allow_patterns=prefetch.MODEL_FILES)
            self.assertEqual(hub.hf_hub_download.call_count, 2)
            for name in ('config.json', 'model.safetensors'):
                hub.hf_hub_download.assert_any_call(prefetch.SMAD_MODEL, name,
                                                   revision=prefetch.SMAD_REVISION,
                                                   cache_dir=str(Path(directory) / 'model-cache'))


class SourceBundle(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        inventory = self.root / 'inventory/base'
        inventory.mkdir(parents=True)
        (inventory / 'python.json').write_text(json.dumps(dict(packages=[dict(
            name='example', version='1.0', native_libraries=[])])))
        self.payloads = {'BUILD.md': b'neutral build instructions', 'source.tar': b'neutral source'}
        self.file = lambda name: dict(path=name, sha256=hashlib.sha256(self.payloads[name]).hexdigest())
        self.manifest = dict(schema=1, image_id='sha256:' + 'a' * 64, revision='b' * 40,
                             build_instructions=self.file('BUILD.md'), components=[dict(
                                 id='python/base/example@1.0', version='1.0', native_libraries=[],
                                 license='GPL-3.0-only', origin='https://example.invalid/source',
                                 source_required=True, files=[self.file('source.tar')])])

    def tearDown(self):
        self.tmp.cleanup()

    def verify(self):
        archive = self.root / 'bundle.tar'
        with tarfile.open(archive, 'w') as bundle:
            contents = dict(self.payloads, **{'source-manifest.json': json.dumps(self.manifest).encode()})
            for name, content in contents.items():
                member = tarfile.TarInfo(name)
                member.size = len(content)
                bundle.addfile(member, io.BytesIO(content))
        return image_inventory.verify_source_bundle(archive, self.root, 'sha256:' + 'a' * 64, 'b' * 40)

    def test_exact_image_and_source_material_are_accepted(self):
        self.assertEqual(self.verify(), self.manifest)

    def test_different_image_is_rejected(self):
        self.manifest['image_id'] = 'sha256:' + 'c' * 64
        with self.assertRaisesRegex(ValueError, 'reviewed image'):
            self.verify()

    def test_incomplete_component_coverage_is_rejected(self):
        self.manifest['components'] = []
        with self.assertRaisesRegex(ValueError, 'coverage'):
            self.verify()

    def test_source_file_checksum_is_verified(self):
        self.payloads['source.tar'] = b'changed source'
        with self.assertRaisesRegex(ValueError, 'checksum'):
            self.verify()

    def test_gpl_sources_cannot_be_omitted(self):
        self.manifest['components'][0].update(source_required=False, files=[])
        with self.assertRaisesRegex(ValueError, 'corresponding source'):
            self.verify()

    def test_lgpl_requires_replacement_instructions(self):
        self.manifest['components'][0]['license'] = 'LGPL-2.1-or-later'
        with self.assertRaisesRegex(ValueError, 'file record'):
            self.verify()
        self.manifest['components'][0]['replacement_instructions'] = self.file('BUILD.md')
        self.verify()

    def test_nvidia_requires_redistribution_basis(self):
        self.manifest['components'][0]['license'] = 'NVIDIA EULA'
        with self.assertRaisesRegex(ValueError, 'file record'):
            self.verify()
        self.manifest['components'][0]['redistribution_basis'] = self.file('BUILD.md')
        with self.assertRaisesRegex(ValueError, 'not approved'):
            self.verify()
        self.payloads['BUILD.md'] += b'\nREDISTRIBUTION DECISION: APPROVED\n'
        self.manifest['build_instructions'] = self.file('BUILD.md')
        self.manifest['components'][0]['redistribution_basis'] = self.file('BUILD.md')
        self.manifest['components'][0]['files'] = [self.file('source.tar'), self.file('BUILD.md')]
        self.verify()

    def test_native_library_mismatch_is_rejected(self):
        self.manifest['components'][0]['native_libraries'] = [dict(installed='lib.so', sha256='0' * 64)]
        with self.assertRaisesRegex(ValueError, 'native-library mismatch'):
            self.verify()


class DebianSourceArchives(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        content = b'neutral-source'
        (self.root / 'example.tar').write_bytes(content)
        (self.root / 'example.dsc').write_text(
            'Format: 3.0 (quilt)\nSource: example\nVersion: 1.0\nChecksums-Sha256:\n '
            + hashlib.sha256(content).hexdigest() + f' {len(content)} example.tar\n')
        self.inventory = self.root / 'debian.tsv'
        self.inventory.write_text('binary\tversion\tsource\tsource_version\nexample\t1.0\texample\t1.0\n')

    def tearDown(self):
        self.tmp.cleanup()

    def test_exact_source_version_and_archive_are_verified(self):
        self.assertEqual(debian_sources.verify(self.root, self.inventory), 1)

    def test_modified_archive_is_rejected(self):
        (self.root / 'example.tar').write_bytes(b'changed-source')
        with self.assertRaisesRegex(ValueError, 'checksum|invalid'):
            debian_sources.verify(self.root, self.inventory)

    def test_missing_source_version_is_rejected(self):
        self.inventory.write_text(self.inventory.read_text().replace('1.0', '2.0'))
        with self.assertRaisesRegex(ValueError, 'Missing exact'):
            debian_sources.verify(self.root, self.inventory)

class SourceBundleBuilder(unittest.TestCase):
    def test_native_source_fetch_allowlist_is_complete_and_pinned(self):
        names = set(native_sources.DIRECT_SOURCES) | set(native_sources.GIT_ARCHIVES)
        self.assertEqual(names, set(source_bundle.NATIVE_BUILD_SOURCES))
        hashes = [record[-1] for record in native_sources.GIT_ARCHIVES.values()]
        hashes.extend(record[1] for record in native_sources.DIRECT_SOURCES.values())
        self.assertTrue(all(len(value) == 64 and set(value) <= set('0123456789abcdef') for value in hashes))

    def test_debian_source_fetcher_deduplicates_exact_source_versions(self):
        with tempfile.TemporaryDirectory() as directory:
            inventory = Path(directory) / 'debian.tsv'
            inventory.write_text(
                'binary\tversion\tsource\tsource_version\n'
                'first\t1.0\tshared-source\t2.0\n'
                'second\t1.1\tshared-source\t2.0\n'
                'third\t3.0\tother-source\t4.0\n')
            self.assertEqual(debian_fetch.required_sources(inventory),
                             [('other-source', '4.0'), ('shared-source', '2.0')])

    def test_source_obligation_uses_component_license_not_embedded_references(self):
        package = dict(name='nvidia-cuda-cupti-cu12', version='12.1.105',
                       license='NVIDIA Proprietary Software', license_expression=None,
                       classifiers=[], files=[dict(copied='licenses/nvidia/LICENSE.txt')])
        license_file = self.root / 'inventory/base/licenses/nvidia/LICENSE.txt'
        license_file.parent.mkdir(parents=True)
        license_file.write_text('The SDK includes an optional component under GNU GPLv3.\n')
        license_name, source_required = source_bundle.python_license(package, self.root, 'base')
        self.assertEqual(license_name, 'NVIDIA Proprietary Software')
        self.assertFalse(source_required)

        package['license'] = 'LGPL-2.1-or-later'
        license_name, source_required = source_bundle.python_license(package, self.root, 'base')
        self.assertEqual(license_name, 'LGPL-2.1-or-later')
        self.assertTrue(source_required)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.repository = self.root / 'repository'
        (self.repository / 'gpu').mkdir(parents=True)
        (self.repository / 'docs').mkdir()
        (self.repository / 'docs/NVIDIA_REDISTRIBUTION.md').write_text(
            'REDISTRIBUTION DECISION: APPROVED\n')
        for name in ('SOURCE_BUILD.md', 'verify_debian_sources.py', 'fetch_debian_sources.py',
                 'fetch_native_sources.py', 'build_source_bundle.py'):
            (self.repository / 'gpu' / name).write_text('neutral build material\n')
        self.context = self.root / 'worker-context.tar'
        with tarfile.open(self.context, 'w') as archive:
            content = b'neutral worker source\n'
            member = tarfile.TarInfo('gpu/Dockerfile')
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
        self.inventory = self.root / 'inventory'
        base = self.inventory / 'inventory/base'
        (base / 'licenses/example').mkdir(parents=True)
        (base / 'licenses/example/LICENSE').write_text('MIT License\n')
        (base / 'python.json').write_text(json.dumps(dict(packages=[dict(
            name='example', version='1.0', license='MIT', license_expression=None,
            classifiers=[], project_urls=['Source, https://example.invalid/example'],
            files=[dict(copied='licenses/example/LICENSE')], native_libraries=[])])))
        native = self.inventory / 'inventory/native'
        native.mkdir()
        self.native_source = self.root / 'native-ffmpeg.tar.xz'
        self.native_source.write_bytes(b'neutral ffmpeg source')
        native_record = dict(name='ffmpeg', version='8.1.2',
            native_libraries=[dict(installed='lib/libavcodec.so', sha256='a' * 64)],
            source_archive=dict(url='https://example.invalid/ffmpeg.tar.xz',
                                sha256=hashlib.sha256(self.native_source.read_bytes()).hexdigest()))
        (native / 'ffmpeg.json').write_text(json.dumps(native_record))
        self.source_material = self.root / 'source-material'
        (self.source_material / 'debian').mkdir(parents=True)
        (self.source_material / 'native').mkdir()
        for name in source_bundle.NATIVE_BUILD_SOURCES:
            (self.source_material / 'native' / name).write_bytes(b'neutral native source')
        (self.source_material / 'native' / 'ffmpeg-8.1.2.tar.xz').write_bytes(self.native_source.read_bytes())
        debian_source = b'neutral Debian source'
        (self.source_material / 'debian' / 'example.orig.tar.gz').write_bytes(debian_source)
        dsc_hash = hashlib.sha256(debian_source).hexdigest()
        (self.source_material / 'debian' / 'example_1.0.dsc').write_text(
            'Format: 3.0 (quilt)\nSource: example\nVersion: 1.0\nChecksums-Sha256:\n '
            + f'{dsc_hash} {len(debian_source)} example.orig.tar.gz\n')
        (base / 'debian.tsv').write_text(
            'binary\tversion\tsource\tsource_version\nexample\t1.0\texample\t1.0\n')
        copyright_path = base / 'debian-copyright/example/copyright'
        copyright_path.parent.mkdir(parents=True)
        copyright_path.write_text('License: MIT\n')

    def tearDown(self):
        self.tmp.cleanup()

    def test_bundle_matches_inventory_revision_and_native_sources(self):
        output = self.root / 'corresponding-source.tar.gz'
        manifest = source_bundle.build(self.inventory, self.source_material, self.repository,
                                       self.context, output, 'sha256:' + '1' * 64, '2' * 40)
        self.assertEqual(manifest['source_packages_verified'], 1)
        python_component = next(component for component in manifest['components']
                    if component['id'] == 'python/base/example@1.0')
        self.assertEqual(python_component['license'], 'MIT')
        self.assertTrue(python_component['origin'].startswith('https://'))
        self.assertEqual(image_inventory.verify_source_bundle(
            output, self.inventory, 'sha256:' + '1' * 64, '2' * 40), manifest)
        with self.assertRaisesRegex(ValueError, 'reviewed image'):
            image_inventory.verify_source_bundle(
                output, self.inventory, 'sha256:' + '3' * 64, '2' * 40)


class MelBandAssets(unittest.TestCase):
    def test_only_mit_source_and_original_weights_are_downloaded(self):
        calls = []
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(backend, 'download_verified', side_effect=lambda *a, **k: calls.append((a, k))):
                source, weights = backend.ensure_assets(directory)
            self.assertEqual(source, Path(directory) / ('source-' + backend.REVISION))
            self.assertEqual(weights.name, 'MelBandRoformer.ckpt')
        urls = [args[0] for args, _ in calls]
        self.assertEqual(len(urls), len(backend.SOURCE_FILES) + 1)
        self.assertTrue(all('ZFTurbo/Music-Source-Separation-Training/' in url for url in urls[:-1]))
        self.assertTrue(any(url.endswith('/LICENSE') for url in urls))
        self.assertIn('KimberleyJSN/melbandroformer/resolve/' + backend.WEIGHT_REVISION, urls[-1])
        self.assertEqual(calls[-1][0][2], '87201f4d31afb5bc79993230fc49446918425574db48c01c405e44f365c7559e')
        self.assertNotIn('KimberleyJensen/Mel-Band-Roformer-Vocal-Model', '\n'.join(urls))

    def test_git_blob_digest_and_cached_download(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'LICENSE'
            path.write_bytes(b'hello\n')
            self.assertEqual(backend.file_digest(path, git_blob=True), 'ce013625030ba8dba906f756967f9e9ca394464a')
            with patch.object(backend.urllib.request, 'urlopen') as fetch:
                backend.download_verified('https://unused.invalid', path, backend.file_digest(path))
            fetch.assert_not_called()


if __name__ == '__main__':
    unittest.main()
