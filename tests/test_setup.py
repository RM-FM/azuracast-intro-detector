from pathlib import Path
import json
import os
import shutil
import subprocess
import tempfile
import unittest

import helpers  # noqa: F401
from intro_detector.config import load
from intro_detector.setup_config import inspect, prepare, worker_image


class SetupConfiguration(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / 'config.example.toml').write_text(
            '[azuracast]\nbase_url="https://example.invalid"\nstations="all"\n'
            '[runpod]\nmonthly_budget_usd=50.0\n')

    def tearDown(self):
        self.tmp.cleanup()

    def test_first_setup_is_manual_and_shadow_with_private_secrets(self):
        prepare(self.root, 'https://radio.example/', 'azura-test', 'runpod-test')
        cfg = load(self.root / 'config.toml', environ={'AZURACAST_API_KEY': 'azura-test'})
        self.assertEqual(cfg.base_url, 'https://radio.example')
        self.assertFalse(cfg.auto_process)
        self.assertEqual(cfg.station_defaults.mode, 'shadow')
        self.assertEqual(cfg.runpod.monthly_budget_usd, 50.)
        self.assertEqual((self.root / '.env').stat().st_mode & 0o777, 0o600)
        self.assertNotIn('azura-test', (self.root / 'config.toml').read_text())
        self.assertEqual(inspect(self.root), ('https://radio.example', True, True))

    def test_update_preserves_advanced_settings_budget_and_existing_keys(self):
        prepare(self.root, 'https://radio.example', 'azura-test', 'runpod-test')
        path = self.root / 'config.toml'
        path.write_text(path.read_text().replace('50.0', '75.0') +
                        '\n# Advanced operator setting\n[decision]\nno_intro_below_seconds=3.0\n')
        prepare(self.root, 'https://other.example', '', '')
        self.assertIn('# Advanced operator setting', path.read_text())
        cfg = load(path, environ={'AZURACAST_API_KEY': 'azura-test'})
        self.assertEqual(cfg.runpod.monthly_budget_usd, 75.)
        self.assertEqual(cfg.decision.no_intro_below_seconds, 3.)
        self.assertEqual(inspect(self.root), ('https://other.example', True, True))

    def test_invalid_setup_leaves_existing_files_intact(self):
        prepare(self.root, 'https://radio.example', 'azura-test', 'runpod-test')
        before = [(self.root / name).read_bytes() for name in ('config.toml', '.env')]
        for url, key in (('http://public.example', ''), ('https://radio.example', 'bad\nkey')):
            with self.assertRaises(ValueError):
                prepare(self.root, url, key, '')
        self.assertEqual(before, [(self.root / name).read_bytes() for name in ('config.toml', '.env')])

    def test_worker_reference_uses_repository_default(self):
        self.assertEqual(worker_image(self.root), ('ghcr.io/rm-fm/azuracast-intro-detector-gpu:v1', False))


@unittest.skipUnless(os.name == 'posix' and os.geteuid() == 0, 'Bash setup ownership checks require Linux root')
class BashSetup(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / 'installation with spaces'
        self.root.mkdir()
        (self.root / 'bin').mkdir()
        repo = Path(__file__).resolve().parents[1]
        shutil.copyfile(repo / 'bin/setup.sh', self.root / 'bin/setup.sh')
        (self.root / 'bin/image-context.sh').write_text('#!/usr/bin/env bash\nexit 0\n')
        shutil.copyfile(repo / 'config.example.toml', self.root / 'config.example.toml')
        fake = self.root / 'bin/docker'
        fake.write_text('''#!/usr/bin/env python3
import json, os, pathlib, sys
root = pathlib.Path(os.environ['FAKE_ROOT'])
args = sys.argv[1:]
with (root / 'calls.jsonl').open('a') as stream:
    stream.write(json.dumps(args) + '\\n')
if 'intro_detector.setup_config' in args:
    sys.path.insert(0, os.environ['FAKE_REPO'] + '/src')
    from intro_detector.setup_config import main
    action = args[args.index('intro_detector.setup_config') + 1]
    sys.argv = ['setup_config', action, '--root', str(root)]
    main()
elif 'ssh-keygen' in args:
    (root / 'secrets/runpod-ssh-key').write_text('test-private-key')
elif 'manifest' in args and os.environ.get('FAIL_WORKER_IMAGE'):
    sys.exit(18)
elif args[:2] == ['compose', 'run'] and os.environ.get('FAIL_CHECK'):
    sys.exit(17)
''')
        fake.chmod(0o755)
        self.env = dict(os.environ, PATH=str(self.root / 'bin') + os.pathsep + os.environ['PATH'],
                        FAKE_ROOT=str(self.root), FAKE_REPO=str(repo), TMPDIR=str(self.root))

    def tearDown(self):
        self.tmp.cleanup()

    def run_setup(self, answers, fail_check=False, fail_worker=False):
        env = dict(self.env)
        if fail_check:
            env['FAIL_CHECK'] = '1'
        if fail_worker:
            env['FAIL_WORKER_IMAGE'] = '1'
        return subprocess.run(['bash', str(self.root / 'bin/setup.sh')], input=answers,
                              text=True, capture_output=True, env=env)

    def calls(self):
        return [json.loads(line) for line in (self.root / 'calls.jsonl').read_text().splitlines()]

    def test_first_install_never_exposes_keys_or_starts_gpu_processing(self):
        result = self.run_setup('https://radio.example\nazura-test-key\nrunpod-test-key\n')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        visible = result.stdout + result.stderr + (self.root / 'calls.jsonl').read_text()
        self.assertNotIn('azura-test-key', visible)
        self.assertNotIn('runpod-test-key', visible)
        self.assertNotIn('budget', result.stdout.lower())
        self.assertTrue(any(call[:2] == ['compose', 'up'] for call in self.calls()))
        self.assertTrue(all('--force-recreate' in call for call in self.calls() if call[:2] == ['compose', 'up']))
        self.assertFalse(any('process' in call or 'ab-run' in call for call in self.calls()))
        self.assertEqual((self.root / 'secrets/runpod-ssh-key').stat().st_uid, 10001)

    def test_update_preserves_all_configuration_and_credentials(self):
        result = self.run_setup('https://radio.example\nazura-test-key\nrunpod-test-key\n')
        self.assertEqual(result.returncode, 0, result.stderr)
        config = self.root / 'config.toml'
        config.write_text(config.read_text().replace('monthly_budget_usd = 50.0', 'monthly_budget_usd = 75.0'))
        before = [path.read_bytes() for path in (config, self.root / '.env', self.root / 'secrets/runpod-ssh-key')]
        result = self.run_setup('\n')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(before, [path.read_bytes() for path in (config, self.root / '.env', self.root / 'secrets/runpod-ssh-key')])

    def test_failed_credentials_do_not_restart_service(self):
        result = self.run_setup('https://radio.example\nazura-test-key\nrunpod-test-key\n', fail_check=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(any(call[:2] == ['compose', 'up'] for call in self.calls()))
        self.assertIn('backed up', result.stderr)

    def test_custom_compose_file_is_rejected_before_building(self):
        (self.root / '.env').write_text('COMPOSE_FILE=other-compose.yaml\n')
        result = self.run_setup('')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('COMPOSE_FILE', result.stderr)
        self.assertFalse((self.root / 'calls.jsonl').exists())

    def test_missing_bundled_worker_is_a_release_failure_before_credentials(self):
        result = self.run_setup('', fail_worker=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('repository maintainers', result.stderr)
        self.assertIn('network access to GHCR', result.stderr)
        self.assertNotIn('AzuraCast API key', result.stdout)
        self.assertFalse((self.root / '.env').exists())
        self.assertFalse((self.root / 'config.toml').exists())
        self.assertFalse((self.root / 'secrets').exists())
        self.assertFalse(any(call[:2] == ['compose', 'up'] for call in self.calls()))

    def test_missing_worker_during_update_preserves_files_and_permissions(self):
        result = self.run_setup('https://radio.example\nazura-test-key\nrunpod-test-key\n')
        self.assertEqual(result.returncode, 0, result.stderr)
        paths = [self.root / name for name in ('config.toml', '.env', 'secrets/runpod-ssh-key')]
        before = [(path.read_bytes(), path.stat().st_mode, path.stat().st_uid, path.stat().st_gid)
                  for path in paths]
        previous_calls = len(self.calls())
        result = self.run_setup('', fail_worker=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(before, [(path.read_bytes(), path.stat().st_mode,
                                  path.stat().st_uid, path.stat().st_gid) for path in paths])
        self.assertNotIn('Keep existing', result.stdout)
        self.assertFalse(any(call[:2] == ['compose', 'up'] for call in self.calls()[previous_calls:]))

    def test_custom_private_worker_is_preserved(self):
        result = self.run_setup('https://radio.example\nazura-test-key\nrunpod-test-key\n')
        self.assertEqual(result.returncode, 0, result.stderr)
        config = self.root / 'config.toml'
        config.write_text(config.read_text().replace(
            'image = "ghcr.io/rm-fm/azuracast-intro-detector-gpu:v1"',
            'image = "registry.example/worker:custom"').replace(
                'registry_auth_id = ""', 'registry_auth_id = "registry-auth"'))
        before = config.read_bytes()
        previous_calls = len(self.calls())
        result = self.run_setup('\n', fail_worker=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(config.read_bytes(), before)
        self.assertFalse(any('manifest' in call for call in self.calls()[previous_calls:]))