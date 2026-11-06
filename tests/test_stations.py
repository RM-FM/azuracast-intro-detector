import os
from pathlib import Path
import stat
import unittest

from helpers import database, make_config, tempdir
from intro_detector.config import Station
from intro_detector.stations import SHARED, Instance, Secrets, Stations, account_station, migrate, station_account


class StationSettingsStore(unittest.TestCase):
    def setUp(self):
        self.tmp = tempdir()
        self.con = database(self.tmp.name)
        self.cfg = make_config(self.tmp.name, station_defaults=Station(0, 'publish', (7,), 90.))
        self.store = Stations(self.con, self.cfg, lambda: 5.)

    def tearDown(self):
        self.con.close()
        self.tmp.cleanup()

    def test_unknown_stations_use_disabled_defaults(self):
        settings = self.store.get(4)
        self.assertEqual((settings.enabled, settings.mode, settings.min_song_seconds, settings.exclude_playlists),
                         (False, 'publish', 90., (7,)))
        self.assertEqual(settings.station.mode, 'off')
        self.assertEqual(settings.account, station_account(4))
        self.assertEqual(self.store.all(), [])

    def test_station_managers_change_station_fields_only(self):
        settings = self.store.update(3, {'enabled': True, 'mode': 'shadow', 'exclude_playlists': [9, 2, 9],
                                         'budget_usd': 12})
        self.assertEqual((settings.station.mode, settings.exclude_playlists, settings.budget_usd),
                         ('shadow', (2, 9), 12.))
        self.assertEqual(self.store.get(3), settings)
        with self.assertRaisesRegex(ValueError, 'not allowed'):
            self.store.update(3, {'shared_allowed': True})
        for bad in ({'mode': 'off'}, {'enabled': 1}, {'budget_usd': -1}, {'exclude_playlists': ['x']},
                    {'min_song_seconds': float('nan')}):
            with self.assertRaises(ValueError):
                self.store.update(3, bad)

    def test_instance_key_needs_the_administrators_allowance(self):
        with self.assertRaisesRegex(ValueError, 'not available'):
            self.store.update(3, {'billing': 'shared'})
        self.store.update(3, {'shared_allowed': True, 'shared_cap_usd': 5}, admin=True)
        self.assertEqual(self.store.update(3, {'billing': 'shared'}).account, SHARED)
        withdrawn = self.store.update(3, {'shared_allowed': False}, admin=True)
        self.assertEqual((withdrawn.billing, withdrawn.account), ('own', station_account(3)))

    def test_account_names(self):
        self.assertEqual(account_station(station_account(12)), 12)
        self.assertIsNone(account_station(SHARED))

    def test_instance_settings_default_to_the_configuration(self):
        instance = Instance(self.con, self.cfg)
        self.assertEqual((instance.get().max_price_per_hour, instance.get().shared_budget_usd), (.9, 50.))
        self.assertTrue(instance.update({'share_analyses': False, 'max_price_per_hour': 1.2}).max_price_per_hour == 1.2)
        self.assertFalse(instance.get().share_analyses)
        with self.assertRaises(ValueError):
            instance.update({'auto_process': True})


class SecretFile(unittest.TestCase):
    def test_keys_are_private_and_removable(self):
        with tempdir() as tmp:
            secrets = Secrets(tmp)
            self.assertIsNone(secrets.get(SHARED))
            secrets.set(SHARED, 'rpa_ABCDEFGH12345678')
            secrets.set('station-2', 'rpa_ZYXWVUTS87654321')
            self.assertEqual(secrets.accounts(), [SHARED, 'station-2'])
            self.assertEqual(stat.S_IMODE(os.stat(Path(tmp) / 'secrets.json').st_mode), 0o600)
            secrets.set(SHARED, None)
            self.assertEqual(secrets.accounts(), ['station-2'])
            for bad in ('short', 'has space 12345678', 'quote"12345678'):
                with self.assertRaises(ValueError):
                    secrets.set(SHARED, bad)


class Migration(unittest.TestCase):
    def setUp(self):
        self.tmp = tempdir()
        self.con = database(self.tmp.name)

    def tearDown(self):
        self.con.close()
        self.tmp.cleanup()

    def run_migration(self, cfg, environ):
        stores = Stations(self.con, cfg, lambda: 1.), Instance(self.con, cfg), Secrets(self.tmp.name)
        return migrate(self.con, cfg, *stores, environ), stores

    def test_existing_install_keeps_its_stations_on_the_shared_key(self):
        with self.con:
            for station_id in (1, 2, 5):
                self.con.execute('INSERT INTO stations VALUES(?, 0, 0)', (station_id,))
            self.con.execute("INSERT INTO analyses VALUES('abc', 'pod', 'r', x'00', x'00', 0)")
        cfg = make_config(self.tmp.name, Station(2, 'off'), Station(5, 'publish', (3,), 30.), all_stations=True,
                          station_defaults=Station(0, 'shadow'), auto_process=False)
        migrated, (stations, _, secrets) = self.run_migration(cfg, {'RUNPOD_API_KEY': 'rpa_LEGACYKEY123456'})
        self.assertTrue(migrated)
        self.assertEqual(secrets.get(SHARED), 'rpa_LEGACYKEY123456')
        summary = [(s.id, s.enabled, s.mode, s.account, s.shared_allowed, s.auto_process, s.exclude_playlists)
                   for s in stations.all()]
        self.assertEqual(summary, [(1, True, 'shadow', SHARED, True, False, ()),
                                   (2, False, 'shadow', SHARED, True, False, ()),
                                   (5, True, 'publish', SHARED, True, False, (3,))])
        self.assertEqual(self.con.execute('SELECT account FROM analysis_accounts').fetchall()[0][0], SHARED)
        self.assertFalse(self.run_migration(cfg, {})[0])

    def test_fresh_install_enables_nothing_but_keeps_the_env_key(self):
        cfg = make_config(self.tmp.name, all_stations=True)
        migrated, (stations, instance, secrets) = self.run_migration(cfg, {'RUNPOD_API_KEY': 'rpa_LEGACYKEY123456'})
        self.assertFalse(migrated)
        self.assertEqual((stations.all(), secrets.accounts()), ([], [SHARED]))
        self.assertEqual(instance.marker('migrated'), 1)


if __name__ == '__main__':
    unittest.main()
