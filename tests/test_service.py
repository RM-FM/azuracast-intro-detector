import unittest

from helpers import Clock, FakeClient, database, make_config, media, music_row, report_row, tempdir, write_reports
from intro_detector.config import Station
from intro_detector.gpu import GpuError, NoCapacity
from intro_detector.library import sync_station
from intro_detector.service import Service


class FakeSelector:
    def __init__(self, cues):
        self.cues = cues

    def predict(self, row):
        cue = self.cues[row['sha256']]
        return dict(cue_s=cue, decision_s=cue, conflict=False, rank_score=.9,
                    provenance='development_fit', policy='test')


class FakeGpu:
    def __init__(self, drop=(), error=None):
        self.drop, self.error, self.batches = set(drop), error, []

    def run(self, songs, output):
        files = sorted(p.name for p in songs.iterdir())
        self.batches.append(files)
        if self.error:
            raise self.error
        shas = [name.split('.')[0] for name in files if name.split('.')[0] not in self.drop]
        write_reports(output, [report_row(s, s) for s in shas], [music_row(s, s) for s in shas])
        return output

    def idle_tick(self):
        return None


class StationSelection(unittest.TestCase):
    def test_all_stations_follow_azuracast_and_defaults(self):
        with tempdir() as tmp:
            con = database(tmp)
            client = FakeClient([media(1)])
            client.stations = {1: 'a', 3: 'b'}
            cfg = make_config(tmp, Station(2, 'off'), all_stations=True, station_defaults=Station(0, 'publish'))
            service = Service(cfg, con, client, FakeGpu(), clock=Clock())
            self.assertEqual([(s.id, s.mode) for s in service.stations(discover=True)],
                             [(1, 'publish'), (2, 'off'), (3, 'publish')])
            service.sync()
            self.assertEqual({r[0] for r in con.execute('SELECT DISTINCT station_id FROM media')}, {1, 3})
            client.stations = {}
            self.assertEqual([s.id for s in service.stations()], [1, 2, 3])
            con.close()


class Processing(unittest.TestCase):
    def setUp(self):
        self.tmp = tempdir()
        self.con = database(self.tmp.name)
        self.clock = Clock()
        self.client = FakeClient([media(1)])
        self.cfg = make_config(self.tmp.name, Station(1, 'publish'), batch_quiet_seconds=60, max_attempts=2)
        sync_station(self.con, self.client, self.cfg.stations[0], now=self.clock())

    def tearDown(self):
        self.con.close()
        self.tmp.cleanup()

    def service(self, gpu, cues):
        return Service(self.cfg, self.con, self.client, gpu, lambda: FakeSelector(cues),
                       clock=self.clock, sleep=self.clock.sleep)

    def upload(self, *ids):
        for media_id in ids:
            self.client.files[media_id] = media(media_id, path=f'{media_id}.mp3')
        sync_station(self.con, self.client, self.cfg.stations[0], now=self.clock())

    def sha(self, media_id):
        import hashlib
        return hashlib.sha256(f'audio-{media_id}'.encode()).hexdigest()

    def test_batches_new_uploads_after_quiet_period_and_publishes(self):
        gpu = FakeGpu()
        service = self.service(gpu, {self.sha(2): 14.2, self.sha(3): .5})
        self.upload(2, 3)
        self.assertEqual(service.process(), 'idle')
        self.clock.now += 61
        service.process()
        self.assertEqual(len(gpu.batches), 1)
        self.assertEqual(sorted(self.client.writes), [(1, 2, 14.2), (1, 3, 0.)])
        self.assertEqual(self.con.execute("SELECT COUNT(*) FROM media WHERE status='done'").fetchone()[0], 2)

    def test_known_audio_skips_gpu(self):
        gpu = FakeGpu()
        service = self.service(gpu, {self.sha(2): 10.})
        self.upload(2)
        self.clock.now += 61
        service.process()
        self.client.audio[4] = b'audio-2'
        self.upload(4)
        self.clock.now += 61
        service.process()
        self.assertEqual(len(gpu.batches), 1)
        self.assertIn((1, 4, 10.), self.client.writes)

    def test_incomplete_evidence_retries_then_fails(self):
        gpu = FakeGpu(drop={self.sha(2)})
        service = self.service(gpu, {})
        self.upload(2)
        for _ in range(2):
            self.clock.now += 1000
            service.process()
        row = self.con.execute('SELECT status, attempts, error FROM media WHERE media_id=2').fetchone()
        self.assertEqual((row['status'], row['attempts']), ('failed', 2))
        self.assertIn('no report row', row['error'])

    def test_capacity_errors_do_not_count_against_songs(self):
        service = self.service(FakeGpu(error=NoCapacity('none')), {})
        self.upload(2)
        self.clock.now += 61
        service.process()
        row = self.con.execute('SELECT status, attempts FROM media WHERE media_id=2').fetchone()
        self.assertEqual(tuple(row), ('pending', 0))
        self.assertEqual(service.process(), 'waiting for GPU retry')

    def test_batch_failure_counts_attempt(self):
        service = self.service(FakeGpu(error=GpuError('exit 1')), {})
        self.upload(2)
        self.clock.now += 61
        service.process()
        self.assertEqual(self.con.execute('SELECT attempts FROM media WHERE media_id=2').fetchone()[0], 1)

    def test_manual_mode_queues_until_requested_then_drains_queue(self):
        self.cfg = make_config(self.tmp.name, Station(1, 'publish'), batch_quiet_seconds=60,
                               batch_max_songs=1, auto_process=False)
        gpu = FakeGpu()
        service = self.service(gpu, {self.sha(2): 5., self.sha(3): 6.})
        self.upload(2, 3)
        self.clock.now += 61
        self.assertEqual(service.tick(), 'manual processing; nothing requested')
        self.assertEqual(gpu.batches, [])
        service.request_path.write_text('now')
        service.tick()
        service.tick()
        self.assertEqual(len(gpu.batches), 2)
        self.assertEqual(service.tick(), 'idle')
        self.assertFalse(service.request_path.exists())
        self.assertEqual(sorted(self.client.writes), [(1, 2, 5.), (1, 3, 6.)])

    def test_request_skips_quiet_period(self):
        self.cfg = make_config(self.tmp.name, Station(1, 'publish'), batch_quiet_seconds=600, auto_process=False)
        gpu = FakeGpu()
        service = self.service(gpu, {self.sha(2): 5.})
        self.upload(2)
        service.request_path.write_text('now')
        service.tick()
        self.assertEqual(len(gpu.batches), 1)


if __name__ == '__main__':
    unittest.main()
