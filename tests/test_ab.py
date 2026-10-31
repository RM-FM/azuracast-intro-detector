import hashlib
import json
from pathlib import Path
import unittest

from helpers import FakeClient, database, make_config, media, music_row, report_row, tempdir, write_reports
from intro_detector import ab
from intro_detector.config import Station
from intro_detector.db import pack
from intro_detector.service import Service


def sha(media_id):
    return hashlib.sha256(f'audio-{media_id}'.encode()).hexdigest()


class Selector:
    def __init__(self, cues):
        self.cues = cues

    def predict(self, row):
        cue = self.cues.get((row['sha256'], row['duration']))
        return dict(cue_s=cue, decision_s=cue, conflict=False, rank_score=.9,
                    provenance='development_fit', policy='test')


class Gpu:
    def __init__(self):
        self.batches = []

    def run(self, songs, output):
        shas = sorted(p.name.split('.')[0] for p in songs.iterdir())
        self.batches.append(shas)
        # Duration 29 marks B evidence, so the fake selector can return a different cue.
        write_reports(output, [report_row(s, s, 29.) for s in shas], [music_row(s, s) for s in shas])
        return output


LANGUAGES = {1: 'de', 2: 'de', 3: 'fr', 4: 'ko', 5: 'en', 6: 'en', 7: 'en', 8: None}


class AB(unittest.TestCase):
    def setUp(self):
        self.tmp = tempdir()
        self.root = Path(self.tmp.name)
        self.con = database(self.root)
        self.client = FakeClient([media(i, path=f'{i}.mp3') for i in LANGUAGES])
        now = 1.
        with self.con:
            for media_id, language in LANGUAGES.items():
                row = report_row(f'{media_id}.mp3', sha(media_id))
                row['asr']['melband/whisper_auto']['language'] = language
                self.con.execute("INSERT INTO media(station_id, media_id, unique_id, path, mtime, sha256, status, "
                                 "discovered_at, updated_at) VALUES(1,?,?,?,1,?,'done',?,?)",
                                 (media_id, f'u{media_id}', f'{media_id}.mp3', sha(media_id), now, now))
                self.con.execute("INSERT INTO analyses VALUES(?,'pod','run',?,?,?)",
                                 (sha(media_id), pack(row), pack(music_row(row['song'], sha(media_id))), now))
        self.cfg = make_config(self.root, Station(1, 'publish'))
        self.gpu = Gpu()
        cues = {(sha(1), 30.): 10., (sha(1), 29.): 14., (sha(3), 30.): 5., (sha(3), 29.): 5.2}
        self.service = Service(self.cfg, self.con, self.client, self.gpu, lambda: Selector(cues))

    def tearDown(self):
        self.con.close()
        self.tmp.cleanup()

    def test_stored_language_is_read_without_full_parse(self):
        blob = self.con.execute('SELECT report FROM analyses WHERE sha256=?', (sha(4),)).fetchone()[0]
        self.assertEqual(ab.stored_language(blob), 'ko')

    def test_select_round_robin_languages_and_controls(self):
        entries = ab.select(self.con, self.root, 't', count=3, controls=2, log=lambda *_: None)
        picked = [e['language'] for e in entries if not e['control']]
        self.assertEqual(sorted(picked), ['de', 'fr', 'ko'])
        controls = [e for e in entries if e['control']]
        self.assertEqual(len(controls), 2)
        self.assertTrue(all(e['language'] == 'en' for e in controls))
        with self.assertRaisesRegex(ValueError, 'already exists'):
            ab.select(self.con, self.root, 't', log=lambda *_: None)
        with self.assertRaises(ValueError):
            ab.select(self.con, self.root, '../x', log=lambda *_: None)

    def test_quotas_and_cached_language_index(self):
        entries = ab.select(self.con, self.root, 'q', controls=1, quotas={'de': 1, 'other': 5},
                            log=lambda *_: None)
        picked = sorted(e['language'] for e in entries if not e['control'])
        self.assertEqual(picked, ['de', 'fr', 'ko'])
        self.assertTrue((self.root / 'ab/languages.json').exists())
        with self.con:
            self.con.execute('UPDATE analyses SET report=? WHERE sha256=?', (b'not zlib', sha(1)))
        ab.select(self.con, self.root, 'q2', controls=0, quotas={'de': 2}, log=lambda *_: None)

    def test_run_stores_b_separately_never_publishes_and_resumes(self):
        ab.select(self.con, self.root, 't', count=10, controls=0, log=lambda *_: None)
        before = self.con.execute('SELECT COUNT(*) FROM analyses').fetchone()[0]
        result = ab.run(self.service, 't', log=lambda *_: None)
        self.assertEqual(result['stored'], 4)
        self.assertEqual(len(self.gpu.batches), 1)
        self.assertEqual(self.client.writes, [])
        self.assertEqual(self.con.execute('SELECT COUNT(*) FROM analyses').fetchone()[0], before)
        self.assertEqual(self.con.execute('SELECT COUNT(*) FROM publications').fetchone()[0], 0)
        self.assertEqual(ab.run(self.service, 't', log=lambda *_: None)['already'], 4)
        self.assertEqual(len(self.gpu.batches), 1)
        self.assertFalse(any((self.root / 'work').glob('ab-*')))

    def test_cost_cap_limits_budget_and_run_time(self):
        from intro_detector.config import Runpod
        settings = ab.cost_limits(Runpod(max_price_per_hour=.9, startup_minutes=5, idle_seconds=600), 104., 1.5)
        self.assertEqual(settings.monthly_budget_usd, 105.5)
        self.assertEqual(settings.max_run_minutes, 85)
        with self.assertRaisesRegex(ValueError, 'too little'):
            ab.cost_limits(Runpod(max_price_per_hour=.9), 0., .2)

    def test_changed_audio_is_skipped(self):
        ab.select(self.con, self.root, 't', count=10, controls=0, log=lambda *_: None)
        self.client.audio[1] = b'replaced'
        result = ab.run(self.service, 't', log=lambda *_: None)
        self.assertEqual(result['changed_audio'], ['1.mp3'])
        self.assertEqual(result['stored'], 3)

    def test_report_compares_decisions(self):
        ab.select(self.con, self.root, 't', count=10, controls=0, log=lambda *_: None)
        ab.run(self.service, 't', log=lambda *_: None)
        summary = ab.report(self.service, 't', log=lambda *_: None)
        self.assertEqual(summary['non-English']['songs'], 4)
        self.assertEqual(summary['non-English']['changed'], 1)
        lines = (self.root / 'ab/t-report.tsv').read_text().splitlines()
        self.assertEqual(len(lines), 5)
        first = lines[1].split('\t')
        self.assertEqual((first[6], first[12]), ('1.mp3', '1'))
        self.assertEqual(self.con.execute('SELECT COUNT(*) FROM decisions').fetchone()[0], 0)


if __name__ == '__main__':
    unittest.main()
