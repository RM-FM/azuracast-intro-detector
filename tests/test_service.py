import hashlib
import unittest

from helpers import (Clock, FakeClient, FakeWorkers, database, enable, make_config, media, music_row, report_row,
                     tempdir, write_reports)
from intro_detector.config import Station
from intro_detector.gpu import GpuError, NoCapacity, WorkerNeedsRecovery
from intro_detector.library import sync_station
from intro_detector import status
from intro_detector.service import Service


class FakeSelector:
    def __init__(self, cues):
        self.cues = cues

    def predict(self, row):
        cue = self.cues[row['sha256']]
        return dict(cue_s=cue, decision_s=cue, conflict=False, rank_score=.9,
                    provenance='development_fit', policy='test')


class FakeGpu(FakeWorkers):
    def __init__(self, drop=(), error=None, blocked=None, keys=None):
        self.drop, self.error, self.batches = set(drop), error, []
        self.blocked_stations, self.keys = dict(blocked or {}), dict(keys or {})
        self.during = None

    def run(self, songs, output, progress=None):
        files = sorted(p.name for p in songs.iterdir())
        self.batches.append(files)
        if progress:
            progress('worker')
            progress('analyze', 1, 2, detail={'phase': 'run', 'stages': ['qwen']})
            self.during = self.observe() if hasattr(self, 'observe') else None
        if self.error:
            raise self.error
        shas = [name.split('.')[0] for name in files if name.split('.')[0] not in self.drop]
        write_reports(output, [report_row(s, s) for s in shas], [music_row(s, s) for s in shas])
        return output


def sha(media_id):
    return hashlib.sha256(f'audio-{media_id}'.encode()).hexdigest()


class StationSelection(unittest.TestCase):
    def test_only_enabled_stations_are_synced_and_processed(self):
        with tempdir() as tmp:
            con = database(tmp)
            client = FakeClient([media(1)])
            cfg = make_config(tmp)
            enable(con, cfg, Station(1, 'publish'), Station(2, 'off'), Station(3, 'shadow'))
            service = Service(cfg, con, client, FakeGpu(), clock=Clock())
            self.assertEqual([(s.id, s.mode) for s in service.stations()], [(1, 'publish'), (3, 'publish')])
            self.assertEqual(service.station(2).mode, 'off')
            self.assertEqual(service.station(9).mode, 'off')
            service.sync()
            self.assertEqual({r[0] for r in con.execute('SELECT DISTINCT station_id FROM media')}, {1, 3})
            con.close()


class QueueCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempdir()
        self.con = database(self.tmp.name)
        self.clock = Clock()
        self.client = FakeClient([media(1)])
        self.configure(batch_quiet_seconds=60, max_attempts=2)

    def configure(self, *stations, **extra):
        self.cfg = make_config(self.tmp.name, *(stations or (Station(1, 'publish'),)), **extra)
        enable(self.con, self.cfg)
        for station in self.cfg.stations:
            sync_station(self.con, self.client, station, now=self.clock())

    def tearDown(self):
        self.con.close()
        self.tmp.cleanup()

    def service(self, gpu, cues):
        return Service(self.cfg, self.con, self.client, gpu, lambda: FakeSelector(cues),
                       clock=self.clock, sleep=self.clock.sleep)

    def upload(self, *ids, audio=None):
        for media_id in ids:
            self.client.files[media_id] = media(media_id, path=f'{media_id}.mp3')
            if audio is not None:
                self.client.audio[media_id] = audio
        for station in self.cfg.stations:
            sync_station(self.con, self.client, station, now=self.clock())

    def status(self, station_id, media_id):
        row = self.con.execute('SELECT status, attempts, sha256 FROM media WHERE station_id=? AND media_id=?',
                               (station_id, media_id)).fetchone()
        return tuple(row)


class Processing(QueueCase):
    def test_batches_new_uploads_after_quiet_period_and_publishes(self):
        gpu = FakeGpu()
        service = self.service(gpu, {sha(2): 14.2, sha(3): .5})
        self.upload(2, 3)
        self.assertEqual(service.process(), 'idle')
        self.clock.now += 61
        service.process()
        self.assertEqual(len(gpu.batches), 1)
        self.assertEqual(gpu.accounts, ['runpod-test'])
        self.assertEqual(sorted(self.client.writes), [(1, 2, 14.2), (1, 3, 0.)])
        self.assertEqual(self.con.execute("SELECT COUNT(*) FROM media WHERE status='done'").fetchone()[0], 2)
        run = self.con.execute('SELECT account FROM runs').fetchone()
        self.assertEqual(run['account'], 'runpod-test')
        self.assertEqual(tuple(self.con.execute('SELECT station_id, songs FROM run_stations').fetchone()), (1, 2))

    def test_pending_intro_is_skipped_unless_force_is_requested(self):
        self.configure(auto_process=False)
        gpu = FakeGpu()
        service = self.service(gpu, {sha(2): 5.})
        self.upload(2)
        self.client.files[2]['custom_fields']['intro_end'] = '8'
        with self.con:
            self.con.execute("UPDATE media SET intro_end='8' WHERE media_id=2")
        service.request(1)
        self.assertEqual(service.tick(), 'idle')
        self.assertEqual(gpu.batches, [])
        self.assertEqual(service.requests(), {})
        service.request(1, force=True)
        self.assertEqual(service.tick(), 'processed 1')
        self.assertEqual(self.client.writes, [(1, 2, 5.)])

    def test_republish_force_can_replace_existing_intro(self):
        gpu = FakeGpu()
        service = self.service(gpu, {sha(2): 5.})
        self.upload(2)
        self.clock.now += 61
        service.process()
        self.client.files[2]['custom_fields']['intro_end'] = '8'
        self.assertEqual(service.republish(1, 2), 1)
        self.assertEqual(self.client.writes, [(1, 2, 5.)])
        self.assertEqual(service.republish(1, 2, force=True), 1)
        self.assertEqual(self.client.writes, [(1, 2, 5.), (1, 2, 5.)])

    def test_intros_recorded_by_shadow_mode_are_published_once(self):
        service = self.service(FakeGpu(), {sha(2): 5., sha(3): 6.})
        self.upload(2, 3)
        self.clock.now += 61
        service.process()
        writes = list(self.client.writes)
        with self.con:
            self.con.execute("UPDATE publications SET state='planned' WHERE media_id=3")
            self.client.files[3]['custom_fields']['intro_end'] = ''
        service.queue_job(1, 'publish')
        service.run_jobs()
        self.assertEqual(self.client.writes, writes + [(1, 3, 6.)], 'Only the recorded intro is written')
        self.assertEqual(self.con.execute('SELECT state FROM publications WHERE media_id=3').fetchone()[0], 'written')

    def test_known_audio_skips_gpu(self):
        gpu = FakeGpu()
        service = self.service(gpu, {sha(2): 10.})
        self.upload(2)
        self.clock.now += 61
        service.process()
        self.upload(4, audio=b'audio-2')
        self.clock.now += 61
        service.process()
        self.assertEqual(len(gpu.batches), 1)
        self.assertIn((1, 4, 10.), self.client.writes)

    def test_incomplete_evidence_retries_then_fails(self):
        gpu = FakeGpu(drop={sha(2)})
        service = self.service(gpu, {})
        self.upload(2)
        for _ in range(2):
            self.clock.now += 1000
            service.process()
        row = self.con.execute('SELECT status, attempts, error FROM media WHERE media_id=2').fetchone()
        self.assertEqual((row['status'], row['attempts']), ('failed', 2))
        self.assertIn('no report row', row['error'])

    def test_capacity_errors_do_not_count_against_songs_and_pause_the_account(self):
        gpu = FakeGpu(error=NoCapacity('none'))
        service = self.service(gpu, {sha(2): 1.})
        self.upload(2)
        self.clock.now += 61
        service.process()
        self.assertEqual(self.status(1, 2)[:2], ('pending', 0))
        self.assertEqual(service.blocked(service.settings.get(1)), 'waiting for GPU retry')
        self.assertEqual(service.process(), 'idle')
        self.assertEqual(len(gpu.batches), 1)
        self.clock.now += 901
        gpu.error = None
        service.process()
        self.assertEqual(len(gpu.batches), 2)

    def test_batch_failure_counts_attempt(self):
        service = self.service(FakeGpu(error=GpuError('exit 1')), {})
        self.upload(2)
        self.clock.now += 61
        service.process()
        self.assertEqual(self.status(1, 2)[1], 1)

    def test_manual_mode_queues_until_requested_then_drains_queue(self):
        self.configure(batch_quiet_seconds=60, batch_max_songs=1, auto_process=False)
        gpu = FakeGpu()
        service = self.service(gpu, {sha(2): 5., sha(3): 6.})
        self.upload(2, 3)
        self.clock.now += 61
        self.assertEqual(service.tick(), 'idle')
        self.assertEqual(gpu.batches, [])
        service.request(1)
        service.tick()
        self.assertEqual(service.requests(), {1: False})
        service.tick()
        self.assertEqual(len(gpu.batches), 2)
        self.assertEqual(service.requests(), {})
        self.assertEqual(sorted(self.client.writes), [(1, 2, 5.), (1, 3, 6.)])

    def test_request_skips_quiet_period(self):
        self.configure(batch_quiet_seconds=600, auto_process=False)
        gpu = FakeGpu()
        service = self.service(gpu, {sha(2): 5.})
        self.upload(2)
        service.request(1)
        service.tick()
        self.assertEqual(len(gpu.batches), 1)

    def test_worker_needing_recovery_pauses_without_attempts(self):
        service = self.service(FakeGpu(error=WorkerNeedsRecovery('stopped with unconfirmed evidence')), {})
        self.upload(2)
        self.clock.now += 61
        service.process()
        self.assertEqual(self.status(1, 2)[:2], ('pending', 0))
        self.assertEqual(service.recovery('runpod-test'), 'stopped with unconfirmed evidence')

    def test_jobs_run_once_for_enabled_stations(self):
        service = self.service(FakeGpu(), {})
        calls = []
        service.republish = lambda station_id: calls.append(('republish', station_id)) or 0
        service.sync = lambda station_ids: calls.append(('sync', station_ids))
        service.queue_job(1, 'republish')
        service.queue_job(1, 'sync')
        service.queue_job(9, 'sync')
        with self.assertRaises(ValueError):
            service.queue_job(1, 'delete')
        service.run_jobs()
        self.assertEqual(calls, [('republish', 1), ('sync', {1})])
        self.assertEqual(service.jobs(), [])


class Accounts(QueueCase):
    """Two stations with their own Runpod keys or with the same key."""

    def setUp(self):
        self.tmp = tempdir()
        self.con = database(self.tmp.name)
        self.clock = Clock()
        self.client = FakeClient([media(1)])
        self.configure(Station(1, 'publish'), Station(2, 'publish'), batch_quiet_seconds=0)

    def test_one_key_per_batch_and_identical_files_are_reused(self):
        gpu = FakeGpu(keys={1: 'runpod-a', 2: 'runpod-b'})
        service = self.service(gpu, {sha(5): 3.})
        self.upload(5)
        service.process()
        self.assertEqual(self.status(2, 5), ('pending', 0, sha(5)))
        service.process()
        self.assertEqual(gpu.accounts, ['runpod-a'])
        self.assertEqual(self.status(2, 5)[0], 'done')
        self.assertEqual(self.con.execute('SELECT state FROM publications WHERE station_id=2').fetchone()[0],
                         'matching')
        self.assertEqual(len(gpu.batches), 1)

    def test_stations_with_the_same_key_share_a_batch(self):
        gpu = FakeGpu()
        service = self.service(gpu, {sha(5): 3., sha(6): 4.})
        self.upload(5, 6)
        service.process()
        self.assertEqual((gpu.accounts, len(gpu.batches), len(gpu.batches[0])), (['runpod-test'], 1, 2))
        self.assertEqual(sorted(tuple(r) for r in self.con.execute('SELECT station_id, songs FROM run_stations')),
                         [(1, 2), (2, 2)])
        self.assertEqual(self.con.execute("SELECT COUNT(*) FROM media WHERE status='done'").fetchone()[0], 4)

    def test_a_batch_is_eu_only_when_one_of_its_stations_requires_it(self):
        gpu = FakeGpu()
        service = self.service(gpu, {sha(5): 3., sha(6): 4.})
        service.settings.update(2, {'eu_only': True})
        self.upload(5)
        service.process()
        self.assertTrue(gpu.eu_only)
        service.settings.update(2, {'eu_only': False})
        self.upload(6)
        service.process()
        self.assertFalse(gpu.eu_only)
        self.assertEqual(len(gpu.batches), 2)

    def test_station_without_key_downloads_once_then_waits(self):
        gpu = FakeGpu(keys={1: None, 2: 'runpod-b'})
        service = self.service(gpu, {sha(5): 3., sha(6): 4.})
        self.assertEqual(service.blocked(service.settings.get(1)), 'no Runpod key')
        self.upload(5)
        service.process()
        service.process()
        self.assertEqual(gpu.accounts, ['runpod-b'])
        self.assertEqual(self.status(1, 5)[0], 'done')
        gpu.accounts = []
        self.client.files[6] = media(6, path='6.mp3')
        sync_station(self.con, self.client, self.cfg.stations[0], now=self.clock())
        service.process()
        self.assertEqual(self.status(1, 6), ('pending', 0, sha(6)))
        downloads = []
        self.client.download = lambda *args: downloads.append(args)
        self.assertEqual(service.process(), 'idle')
        self.assertEqual(downloads, [])

    def test_failed_batch_pauses_only_its_key(self):
        gpu = FakeGpu(error=NoCapacity('none'), keys={1: 'runpod-a', 2: 'runpod-b'})
        service = self.service(gpu, {})
        self.upload(5)
        service.process()
        self.assertEqual(service.blocked(service.settings.get(1)), 'waiting for GPU retry')
        self.assertIsNone(service.blocked(service.settings.get(2)))

class Progress(QueueCase):
    def setUp(self):
        self.tmp = tempdir()
        self.con = database(self.tmp.name)
        self.clock = Clock()
        self.client = FakeClient([media(1)])
        self.configure(Station(1, 'publish'), Station(2, 'publish'), batch_quiet_seconds=0)

    def test_running_batch_reports_its_step_to_its_own_stations_only(self):
        gpu = FakeGpu(keys={1: 'runpod-a', 2: 'runpod-b'})
        service = self.service(gpu, {sha(5): 3.})
        gpu.observe = lambda: (status.activity(service, 1), status.activity(service, 2))
        self.upload(5)
        service.process()
        own, other = gpu.during
        self.assertEqual((own['own'], own['step'], own['done'], own['total'], own['station_songs']),
                         (True, 'analyze', 1, 2, 1))
        self.assertEqual(other, {'step': 'analyze', 'done': 1, 'total': 2, 'started_at': self.clock(),
                                 'step_started_at': self.clock(), 'detail': {'phase': 'run', 'stages': ['qwen']}, 'own': False})
        self.assertIsNone(service.activity(), 'Progress ends with the batch')
        self.assertIsNone(status.activity(service, 1))

    def test_failed_batch_keeps_the_step_it_stopped_at(self):
        service = self.service(FakeGpu(error=GpuError('lost')), {})
        self.upload(5)
        service.process()
        run = self.con.execute('SELECT status, phase FROM runs').fetchone()
        self.assertEqual(tuple(run), ('failed', 'analyze'))
        self.assertIsNone(service.activity())

if __name__ == '__main__':
    unittest.main()
