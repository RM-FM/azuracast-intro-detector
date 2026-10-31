from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from helpers import database, make_config, tempdir
from intro_detector.cli import _owner, cmd_health


class Owner(unittest.TestCase):
    def test_second_queue_worker_is_refused(self):
        with tempdir() as tmp:
            cfg = SimpleNamespace(data_dir=Path(tmp) / 'data')
            with _owner(cfg):
                with self.assertRaises(SystemExit):
                    with _owner(cfg):
                        pass
            with _owner(cfg):
                pass


class Health(unittest.TestCase):
    def test_running_owner_and_initialized_database_are_healthy(self):
        with tempdir() as tmp:
            cfg = make_config(tmp)
            database(tmp).close()
            before = cfg.db_path.read_bytes()
            with _owner(cfg), patch('intro_detector.cli.config_module.load', return_value=cfg), \
                    patch('intro_detector.cli._service', side_effect=AssertionError('Service construction is forbidden')):
                cmd_health(SimpleNamespace(config=None))
            self.assertEqual(cfg.db_path.read_bytes(), before)

    def test_released_owner_is_not_healthy(self):
        with tempdir() as tmp:
            cfg = make_config(tmp)
            database(tmp).close()
            with _owner(cfg):
                pass
            with patch('intro_detector.cli.config_module.load', return_value=cfg):
                with self.assertRaisesRegex(SystemExit, 'no running service'):
                    cmd_health(SimpleNamespace(config=None))

    def test_missing_state_is_not_created(self):
        with tempdir() as tmp:
            cfg = make_config(tmp)
            with patch('intro_detector.cli.config_module.load', return_value=cfg):
                with self.assertRaisesRegex(SystemExit, 'unavailable'):
                    cmd_health(SimpleNamespace(config=None))
            self.assertEqual(list(Path(tmp).iterdir()), [])
            with _owner(cfg), patch('intro_detector.cli.config_module.load', return_value=cfg):
                with self.assertRaisesRegex(SystemExit, 'unavailable'):
                    cmd_health(SimpleNamespace(config=None))
            self.assertFalse(cfg.db_path.exists())

    def test_uninitialized_database_is_not_migrated(self):
        with tempdir() as tmp:
            cfg = make_config(tmp)
            cfg.db_path.touch()
            with _owner(cfg), patch('intro_detector.cli.config_module.load', return_value=cfg):
                with self.assertRaisesRegex(SystemExit, 'unavailable'):
                    cmd_health(SimpleNamespace(config=None))
            self.assertEqual(cfg.db_path.stat().st_size, 0)

    def test_invalid_configuration_is_reported_without_details(self):
        with patch('intro_detector.cli.config_module.load', side_effect=ValueError('private-value')):
            with self.assertRaisesRegex(SystemExit, '^Unhealthy: configuration or service database is unavailable$'):
                cmd_health(SimpleNamespace(config=None))


if __name__ == '__main__':
    unittest.main()
