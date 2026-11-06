import json
import os
from pathlib import Path
import stat
import unittest

from helpers import database, make_config, tempdir
from intro_detector.config import Runpod, Station
from intro_detector.stations import Instance, Secrets, Stations, key_account, key_slot, migrate

KEY = 'rpa_LEGACYKEY123456'


class StationSettingsStore(unittest.TestCase):
    def setUp(self):
        self.tmp = tempdir()
        self.con = database(self.tmp.name)
        self.cfg = make_config(self.tmp.name, station_defaults=Station(0, 'publish', (7,), 90.),
                               runpod=Runpod(monthly_budget_usd=15.))
        self.store = Stations(self.con, self.cfg, lambda: 5.)

    def tearDown(self):
        self.con.close()
        self.tmp.cleanup()

    def test_unknown_stations_use_disabled_defaults(self):
        settings = self.store.get(4)
        self.assertEqual((settings.enabled, settings.mode, settings.min_song_seconds, settings.exclude_playlists,
                          settings.budget_usd), (False, 'publish', 90., (7,), 15.))
        self.assertEqual(settings.station.mode, 'off')
        self.assertEqual(self.store.all(), [])

    def test_station_settings_are_validated(self):
        settings = self.store.update(3, {'enabled': True, 'mode': 'shadow', 'exclude_playlists': [9, 2, 9],
                                         'budget_usd': 12})
        self.assertEqual((settings.station.mode, settings.exclude_playlists, settings.budget_usd),
                         ('shadow', (2, 9), 12.))
        self.assertEqual(self.store.get(3), settings)
        for bad in ({'mode': 'off'}, {'enabled': 1}, {'budget_usd': -1}, {'exclude_playlists': ['x']},
                    {'min_song_seconds': float('nan')}, {'billing': 'shared'}, {'max_song_seconds': 30},
                    {'max_song_seconds': -1}):
            with self.assertRaises(ValueError):
                self.store.update(3, bad)
        self.assertEqual(self.store.get(3).max_song_seconds, 900.)
        self.assertEqual(self.store.update(3, {'max_song_seconds': 0}).station.max_song_seconds, 0.)

    def test_markers(self):
        instance = Instance(self.con)
        self.assertIsNone(instance.marker('migrated'))
        instance.set_marker('migrated', 2)
        self.assertEqual(instance.marker('migrated'), 2)


class SecretFile(unittest.TestCase):
    def test_keys_are_private_and_removable(self):
        with tempdir() as tmp:
            secrets = Secrets(tmp)
            self.assertIsNone(secrets.get(key_slot(1)))
            secrets.set(key_slot(1), 'rpa_ABCDEFGH12345678')
            secrets.set(key_slot(2), 'rpa_ABCDEFGH12345678')
            self.assertEqual((secrets.slots(), secrets.keys()), (['station-1', 'station-2'], ['rpa_ABCDEFGH12345678']))
            self.assertEqual(stat.S_IMODE(os.stat(Path(tmp) / 'secrets.json').st_mode), 0o600)
            secrets.set(key_slot(1), None)
            self.assertEqual(secrets.slots(), ['station-2'])
            for bad in ('short', 'has space 12345678', 'quote"12345678'):
                with self.assertRaises(ValueError):
                    secrets.set(key_slot(1), bad)

    def test_accounts_name_the_key_without_revealing_it(self):
        self.assertEqual(key_account(KEY), key_account(KEY))
        self.assertNotEqual(key_account(KEY), key_account(KEY + 'x'))
        self.assertNotIn(KEY, key_account(KEY))


class Migration(unittest.TestCase):
    def setUp(self):
        self.tmp = tempdir()
        self.con = database(self.tmp.name)

    def tearDown(self):
        self.con.close()
        self.tmp.cleanup()

    def run_migration(self, cfg, environ):
        stores = Stations(self.con, cfg, lambda: 1.), Instance(self.con), Secrets(self.tmp.name)
        return migrate(self.con, cfg, *stores, environ), stores

    def test_legacy_install_keeps_its_stations_on_the_env_key(self):
        with self.con:
            for station_id in (1, 2, 5):
                self.con.execute('INSERT INTO stations VALUES(?, 0, 0)', (station_id,))
            self.con.execute("INSERT INTO analyses VALUES('abc', 'pod', 'r', x'00', x'00', 0)")
            self.con.execute("INSERT INTO pods(pod_id, price, created_at) VALUES('p', 1, 0)")
        cfg = make_config(self.tmp.name, Station(2, 'off'), Station(5, 'publish', (3,), 30.), all_stations=True,
                          station_defaults=Station(0, 'shadow'), auto_process=False)
        migrated, (stations, instance, secrets) = self.run_migration(cfg, {'RUNPOD_API_KEY': KEY})
        self.assertTrue(migrated)
        summary = [(s.id, s.enabled, s.mode, s.auto_process, s.exclude_playlists) for s in stations.all()]
        self.assertEqual(summary, [(1, True, 'shadow', False, ()), (2, False, 'shadow', False, ()),
                                   (5, True, 'publish', False, (3,))])
        self.assertEqual(secrets.slots(), ['station-1', 'station-2', 'station-5'])
        self.assertEqual(secrets.keys(), [KEY])
        self.assertEqual(instance.marker('worker_names'), {key_account(KEY): 'azuracast-intro-detector'})
        self.assertEqual(self.con.execute('SELECT account FROM pods').fetchone()[0], key_account(KEY))
        self.assertFalse(self.run_migration(cfg, {})[0])

    def test_fresh_install_enables_nothing(self):
        cfg = make_config(self.tmp.name, all_stations=True)
        migrated, (stations, instance, secrets) = self.run_migration(cfg, {'RUNPOD_API_KEY': KEY})
        self.assertFalse(migrated)
        self.assertEqual((stations.all(), secrets.slots()), ([], []))
        self.assertEqual(instance.marker('migrated'), 2)

    def test_stations_on_the_shared_key_get_it_as_their_own(self):
        cfg = make_config(self.tmp.name)
        with self.con:
            self.con.execute('DROP TABLE station_settings')
            self.con.execute('CREATE TABLE station_settings(station_id INTEGER PRIMARY KEY, enabled INTEGER NOT NULL '
                             "DEFAULT 0, mode TEXT NOT NULL DEFAULT 'shadow', min_song_seconds REAL NOT NULL, "
                             "exclude_playlists TEXT NOT NULL DEFAULT '[]', billing TEXT NOT NULL DEFAULT 'own', "
                             'budget_usd REAL NOT NULL, shared_allowed INTEGER NOT NULL DEFAULT 0, '
                             'shared_cap_usd REAL NOT NULL DEFAULT 0, auto_process INTEGER NOT NULL DEFAULT 1, '
                             'updated_at REAL NOT NULL, max_song_seconds REAL NOT NULL DEFAULT 900)')
            self.con.executemany('INSERT INTO station_settings(station_id, enabled, min_song_seconds, billing, '
                                 'budget_usd, updated_at) VALUES(?, 1, 60, ?, 10, 0)',
                                 [(1, 'shared'), (2, 'shared'), (3, 'own'), (4, 'own')])
            self.con.execute("INSERT INTO pods(pod_id, price, created_at, account) VALUES('p', 1, 0, 'shared')")
            self.con.execute("INSERT INTO pods(pod_id, price, created_at, account) VALUES('q', 1, 0, 'station-3')")
            self.con.execute('CREATE TABLE analysis_accounts(sha256 TEXT, account TEXT)')
            self.con.executemany('INSERT INTO instance_settings VALUES(?, ?)',
                                 [('_migrated', '1'), ('share_analyses', 'false'), ('_recovery:shared', '"stopped"')])
        secrets = Secrets(self.tmp.name)
        secrets.set('shared', KEY)
        secrets.set(key_slot(3), 'rpa_OWNSTATIONKEY99')
        migrated, (stations, instance, secrets) = self.run_migration(cfg, {})
        self.assertTrue(migrated)
        self.assertEqual([secrets.get(key_slot(i)) for i in (1, 2, 3, 4)], [KEY, KEY, 'rpa_OWNSTATIONKEY99', None])
        self.assertNotIn('shared', secrets.slots())
        self.assertEqual(instance.marker('worker_names'), {
            key_account(KEY): 'azuracast-intro-detector',
            key_account('rpa_OWNSTATIONKEY99'): 'azuracast-intro-detector-station-3'})
        self.assertEqual(dict(self.con.execute('SELECT pod_id, account FROM pods').fetchall()),
                         {'p': key_account(KEY), 'q': key_account('rpa_OWNSTATIONKEY99')})
        self.assertEqual(instance.marker(f'recovery:{key_account(KEY)}'), 'stopped')
        self.assertIsNone(instance.marker('recovery:shared'))
        keys = {r[0] for r in self.con.execute('SELECT key FROM instance_settings')}
        self.assertNotIn('share_analyses', keys)
        self.assertIsNone(self.con.execute("SELECT name FROM sqlite_master WHERE name='analysis_accounts'").fetchone())
        self.assertEqual(stations.get(1).budget_usd, 10.)
        self.assertEqual(json.loads(self.con.execute("SELECT value FROM instance_settings WHERE key='_migrated'")
                                    .fetchone()[0]), 2)
        self.assertFalse(self.run_migration(cfg, {})[0])


if __name__ == '__main__':
    unittest.main()