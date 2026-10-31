from pathlib import Path
import unittest

from helpers import tempdir
from intro_detector import config

GOOD = '''
data_dir = "/data"
[azuracast]
base_url = "https://radio.example/"
[[stations]]
id = 1
exclude_playlists = [4, 5]
[[stations]]
id = 2
mode = "publish"
min_song_seconds = 90
[runpod]
registry_auth_id = "reg"
min_speed = 0.6
[runpod.gpus]
"NVIDIA GeForce RTX 4090" = 1.0
"NVIDIA RTX A5000" = 0.5
[scan]
batch_max_songs = 50
'''


class Load(unittest.TestCase):
    def load(self, text, env=None):
        with tempdir() as tmp:
            path = Path(tmp) / 'config.toml'
            path.write_text(text)
            return config.load(path, {'AZURACAST_API_KEY': 'SECRET-123'} if env is None else env)

    def test_defaults_are_safe(self):
        cfg = self.load(GOOD)
        self.assertEqual(cfg.base_url, 'https://radio.example')
        self.assertEqual(cfg.station(1).mode, 'shadow')
        self.assertEqual(cfg.station(1).exclude_playlists, (4, 5))
        self.assertEqual((cfg.station(1).min_song_seconds, cfg.station(2).min_song_seconds), (60, 90))
        self.assertEqual((cfg.runpod.idle_seconds, cfg.runpod.max_price_per_hour), (600, .9))
        self.assertEqual(dict(cfg.runpod.gpus), {'NVIDIA GeForce RTX 4090': 1.0, 'NVIDIA RTX A5000': .5})
        self.assertEqual(cfg.runpod.container_disk_gb, 80)
        self.assertEqual(cfg.batch_max_songs, 50)
        self.assertTrue(cfg.auto_process)
        self.assertFalse(self.load(GOOD + 'auto_process = false\n').auto_process)
        self.assertNotIn('SECRET-123', repr(cfg))

    def test_rejects_unsafe_settings(self):
        for text in (GOOD.replace('https://', 'http://'), GOOD.replace('mode = "publish"', 'mode = "yes"'),
                     GOOD.replace('id = 2', 'id = 1'), GOOD.replace('min_speed = 0.6', 'min_speed = 2.0'),
                     GOOD.replace('= 0.5', '= 0'), GOOD.replace('min_song_seconds = 90', 'min_song_seconds = -1'),
                     GOOD + 'auto_process = "no"\n'):
            with self.subTest(text=text[:80]), self.assertRaises(ValueError):
                self.load(text)
        with self.assertRaises(ValueError):
            self.load(GOOD, env={})
        self.assertEqual(self.load(GOOD.replace('https://', 'http://').replace(
            '[azuracast]', '[azuracast]\nallow_http = true')).base_url, 'http://radio.example')

    def test_all_stations_by_default_with_defaults_and_overrides(self):
        cfg = self.load('[azuracast]\nbase_url = "https://radio.example"\n')
        self.assertTrue(cfg.all_stations)
        self.assertEqual((cfg.station(9).id, cfg.station(9).mode), (9, 'shadow'))

        cfg = self.load(GOOD.replace('[azuracast]', '[station_defaults]\nmode = "publish"\nmin_song_seconds = 75\n'
                                                    '[azuracast]\nstations = "all"'))
        self.assertTrue(cfg.all_stations)
        self.assertEqual((cfg.station(9).mode, cfg.station(9).min_song_seconds), ('publish', 75))
        self.assertEqual((cfg.station(1).mode, cfg.station(1).min_song_seconds), ('publish', 75))
        self.assertEqual(cfg.station(2).min_song_seconds, 90)

    def test_explicit_station_list(self):
        cfg = self.load(GOOD.replace('[azuracast]', '[azuracast]\nstations = [2, 3, 4]'))
        self.assertFalse(cfg.all_stations)
        self.assertEqual(sorted(s.id for s in cfg.stations), [1, 2, 3, 4])
        self.assertEqual(cfg.station(2).mode, 'publish')
        with self.assertRaises(KeyError):
            cfg.station(9)
        self.assertFalse(self.load(GOOD).all_stations)
        for bad in ('stations = "some"', 'stations = [0]'):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.load(GOOD.replace('[azuracast]', f'[azuracast]\n{bad}'))

    def test_decision_settings(self):
        cfg = self.load(GOOD)
        self.assertEqual(cfg.decision.no_intro_below_seconds, 2.)
        self.assertEqual(cfg.decision.tag, '')
        tuned = self.load(GOOD + '[decision]\nno_intro_below_seconds = 3\n[decision.vocals]\nstrong_burst_level = 0.4\n')
        self.assertEqual((tuned.decision.no_intro_below_seconds, tuned.decision.vocals.strong_burst_level), (3., .4))
        self.assertRegex(tuned.decision.tag, r'^\+cfg-[0-9a-f]{8}$')
        for bad in ('[decision]\nno_intro_below_seconds = 40\n', '[decision]\nmusic_min_seconds = 28\n',
                    '[decision]\nunknown = 1\n', '[decision.vocals]\nvocal_level = 2\n',
                    '[decision.vocals]\nadlib_words = [""]\n'):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.load(GOOD + bad)

    def test_example_config_loads_with_defaults(self):
        example = Path(__file__).resolve().parents[1] / 'config.example.toml'
        cfg = self.load(example.read_text())
        self.assertTrue(cfg.all_stations)
        self.assertEqual(cfg.decision, config.Decision())


if __name__ == '__main__':
    unittest.main()
