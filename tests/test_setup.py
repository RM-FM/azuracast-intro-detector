from pathlib import Path
import json
import os
import shutil
import subprocess
import tempfile
import unittest

from dotenv import dotenv_values

import helpers  # noqa: F401
from intro_detector.config import load
from intro_detector.setup_config import TOKEN, ensure_token, inspect, link, prepare, worker_image


class SetupConfiguration(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / 'config.example.toml').write_text(
            '[azuracast]\nbase_url="https://example.invalid"\nstations="all"\n'
            '[runpod]\nmonthly_budget_usd=50.0\n')

    def tearDown(self):
        self.tmp.cleanup()

    def test_first_setup_uses_all_stations_and_automatic_processing(self):
        prepare(self.root, 'https://radio.example/', 'azura-test', 'runpod-test')
        cfg = load(self.root / 'config.toml', environ={'AZURACAST_API_KEY': 'azura-test'})
        self.assertEqual(cfg.base_url, 'https://radio.example')
        self.assertTrue(cfg.all_stations)
        self.assertTrue(cfg.auto_process)
        self.assertEqual(cfg.runpod.monthly_budget_usd, 50.)
        self.assertEqual((self.root / '.env').stat().st_mode & 0o777, 0o600)
        self.assertNotIn('azura-test', (self.root / 'config.toml').read_text())
        self.assertEqual(inspect(self.root), ('https://radio.example', True, True))

    def test_setup_defaults_a_bare_host_to_https(self):
        prepare(self.root, 'radio.example', 'azura-test', 'runpod-test')
        cfg = load(self.root / 'config.toml', environ={'AZURACAST_API_KEY': 'azura-test'})
        self.assertEqual(cfg.base_url, 'https://radio.example')

    def test_setup_allows_http_only_with_private_network_opt_in(self):
        example = self.root / 'config.example.toml'
        example.write_text(example.read_text().replace(
            '[azuracast]', '[azuracast]\nallow_http = true', 1))
        prepare(self.root, 'http://radio.example', 'azura-test', 'runpod-test')
        cfg = load(self.root / 'config.toml', environ={'AZURACAST_API_KEY': 'azura-test'})
        self.assertEqual(cfg.base_url, 'http://radio.example')

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

    def test_runpod_key_is_optional_and_the_api_token_stays(self):
        prepare(self.root, 'https://radio.example', 'azura-test')
        env = dotenv_values(self.root / '.env')
        self.assertNotIn('RUNPOD_API_KEY', env)
        self.assertEqual(len(env[TOKEN]), 64)
        prepare(self.root, 'https://radio.example', '')
        self.assertFalse(ensure_token(self.root))
        self.assertEqual(dotenv_values(self.root / '.env')[TOKEN], env[TOKEN])

    def test_token_is_added_to_older_installations(self):
        (self.root / '.env').write_text('AZURACAST_API_KEY="azura-test"\n')
        self.assertTrue(ensure_token(self.root))
        self.assertEqual(dotenv_values(self.root / '.env')['AZURACAST_API_KEY'], 'azura-test')

    def test_plugin_link_holds_address_and_token_only(self):
        prepare(self.root, 'https://radio.example', 'azura-test', 'runpod-test')
        link(self.root, self.root / '.detector.json')
        data = json.loads((self.root / '.detector.json').read_text())
        self.assertEqual(data, {'url': 'http://intro-detector:8080', 'token': dotenv_values(self.root / '.env')[TOKEN]})


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
    plugin = [a.split(':')[0] for a in args if a.endswith(':/plugin')]
    if plugin:
        sys.argv += ['--output', plugin[0] + '/.detector.json']
    main()
elif args[:2] == ['compose', '--project-directory'] and 'ps' in args:
    print('' if os.environ.get('NO_WEB') else 'web-container')
elif args[:1] == ['inspect'] and 'web-container' in args:
    print('azuracast_default')
elif args[:2] == ['exec', 'web-container']:
    print('1000')
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

    def run_setup(self, answers, fail_check=False, fail_worker=False, args=(), **extra):
        env = dict(self.env, **extra)
        if fail_check:
            env['FAIL_CHECK'] = '1'
        if fail_worker:
            env['FAIL_WORKER_IMAGE'] = '1'
        return subprocess.run(['bash', str(self.root / 'bin/setup.sh'), *args], input=answers,
                              text=True, capture_output=True, env=env)

    def azuracast(self):
        plugin = Path(self.tmp.name) / 'azuracast/plugins/azuracast-sweepers'
        plugin.mkdir(parents=True)
        return plugin

    def test_plugin_setup_runs_unattended_and_links_the_plugin(self):
        plugin = self.azuracast()
        args = ('--azuracast-dir', str(plugin.parents[1]), '--unattended')
        result = self.run_setup('https://radio.example\nazura-test-key\n', args=args)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn('?', result.stdout)
        self.assertNotIn('azura-test-key', result.stdout + result.stderr + (self.root / 'calls.jsonl').read_text())
        override = (self.root / 'compose.override.yaml').read_text()
        self.assertIn('aliases: [intro-detector]', override)
        self.assertIn('name: azuracast_default', override)
        link = plugin / '.detector.json'
        self.assertEqual((link.stat().st_mode & 0o777, link.stat().st_gid), (0o640, 1000))
        token = dotenv_values(self.root / '.env')[TOKEN]
        self.assertEqual(json.loads(link.read_text())['token'], token)
        before = [path.read_bytes() for path in (self.root / 'config.toml', self.root / '.env', link)]
        result = self.run_setup('', args=args)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(before, [path.read_bytes() for path in (self.root / 'config.toml', self.root / '.env', link)])

    def test_plugin_setup_needs_the_plugin_and_running_azuracast(self):
        args = ('--azuracast-dir', str(Path(self.tmp.name)), '--unattended')
        result = self.run_setup('https://radio.example\nazura-test-key\n', args=args)
        self.assertIn('Sweepers plugin first', result.stderr)
        plugin = self.azuracast()
        result = self.run_setup('https://radio.example\nazura-test-key\n', NO_WEB='1',
                                args=('--azuracast-dir', str(plugin.parents[1]), '--unattended'))
        self.assertIn('AzuraCast is not running', result.stderr)
        self.assertFalse((plugin / '.detector.json').exists())

    def test_unattended_first_install_requires_the_azuracast_key(self):
        result = self.run_setup('https://radio.example\n\n', args=('--unattended',))
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(any(call[:2] == ['compose', 'up'] for call in self.calls()))

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


def git(*args, cwd):
    subprocess.run(['git', '-c', 'user.name=t', '-c', 'user.email=t@example.invalid', *args],
                   cwd=cwd, check=True, capture_output=True)


@unittest.skipUnless(os.name == 'posix' and os.geteuid() == 0 and shutil.which('git'),
                     'Update checks need Linux root and git')
class UpdateScript(unittest.TestCase):
    """bin/update.sh against a local origin with two releases and a scripted setup."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        source, self.root = base / 'source', base / 'install'
        (source / 'bin').mkdir(parents=True)
        shutil.copyfile(Path(__file__).resolve().parents[1] / 'bin/update.sh', source / 'bin/update.sh')
        (source / 'bin/setup.sh').write_text(
            '#!/usr/bin/env bash\n'
            'echo "$(git -C "$(dirname "$0")/.." describe --tags) $*" >> "$FAKE_ROOT/setup.log"\n'
            '[[ -z "${FAIL_SETUP:-}" ]]\n')
        (source / 'release').write_text('0\n')
        git('init', '-q', '-b', 'main', cwd=source)
        git('add', '-A', cwd=source)
        git('commit', '-qm', 'first', cwd=source)
        git('tag', 'v0.1.0', cwd=source)
        (source / 'release').write_text('1\n')
        git('commit', '-qam', 'second', cwd=source)
        git('tag', 'v1.0.0', cwd=source)
        git('clone', '-q', str(source), str(self.root), cwd=base)
        git('checkout', '-q', 'v0.1.0', cwd=self.root)
        self.azuracast = base / 'azuracast'
        (self.azuracast / 'plugins/azuracast-sweepers').mkdir(parents=True)
        fake = base / 'fakebin/docker'
        fake.parent.mkdir()
        fake.write_text('#!/usr/bin/env bash\n'
                        'echo "$*" >> "$FAKE_ROOT/docker.log"\n'
                        'if [[ "$1 $2" == "compose ps" && -n "${RUNNING:-}" ]]; then echo detector-id; fi\n')
        fake.chmod(0o755)
        self.env = dict(os.environ, PATH=f'{fake.parent}{os.pathsep}{os.environ["PATH"]}', FAKE_ROOT=str(base))

    def tearDown(self):
        self.tmp.cleanup()

    def update(self, version='v1.0.0', **extra):
        return subprocess.run(['bash', str(self.root / 'bin/update.sh'), '--version', version,
                               '--azuracast-dir', str(self.azuracast)],
                              text=True, capture_output=True, env=dict(self.env, **extra))

    def head(self):
        return subprocess.run(['git', 'describe', '--tags'], cwd=self.root, text=True,
                              capture_output=True).stdout.strip()

    def log(self, name):
        path = Path(self.tmp.name) / name
        return path.read_text() if path.exists() else ''

    def test_update_checks_out_the_release_and_runs_setup_unattended(self):
        result = self.update()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.head(), 'v1.0.0')
        self.assertIn(f'v1.0.0 --azuracast-dir {self.azuracast} --unattended', self.log('setup.log'))
        self.assertIn('image tag azuracast-intro-detector:local azuracast-intro-detector:previous', self.log('docker.log'))

    def test_failed_setup_restores_the_previous_release(self):
        result = self.update(FAIL_SETUP='1')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('previous version was restored', result.stderr)
        self.assertEqual(self.head(), 'v0.1.0')
        self.assertIn('image tag azuracast-intro-detector:previous azuracast-intro-detector:local',
                      self.log('docker.log'))

    def test_unknown_release_and_unchanged_installation(self):
        self.assertIn('was not found', self.update('v9.9.9').stderr)
        self.assertIn('was not found', self.update('v1.0.0-rc.1').stderr)
        self.assertEqual(self.head(), 'v0.1.0')
        self.update()
        (self.azuracast / 'plugins/azuracast-sweepers/.detector.json').write_text('{}')
        result = self.update(RUNNING='1')
        self.assertIn('already installed', result.stdout)
        self.assertEqual(self.log('setup.log').count('--unattended'), 1)
        self.assertNotEqual(self.update('1.0.0').returncode, 0)