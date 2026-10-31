import json
from pathlib import Path
from types import SimpleNamespace
import unittest

from helpers import Clock, database, tempdir
from intro_detector.config import Runpod as Settings
from intro_detector.gpu import GpuError, NoCapacity, Runpod, lane_plan, merge_chunks, month_spend, no_stock, rank

MIXED = (('NVIDIA GeForce RTX 4090', 1.0), ('NVIDIA RTX A5000', .5), ('NVIDIA RTX A4000', .35), ('NVIDIA L40S', .9))
CATALOG = [{'gpuId': 'NVIDIA GeForce RTX 4090', 'secureCloud': True, 'securePricePerHr': .89, 'memoryInGb': 24,
            'dataCenterAvailability': [{'dataCenterId': 'US-TX-3', 'stockStatus': 'High'},
                                       {'dataCenterId': 'EU-RO-1', 'stockStatus': 'Medium'},
                                       {'dataCenterId': 'EU-SE-1', 'stockStatus': 'High'},
                                       {'dataCenterId': 'US-CA-2', 'stockStatus': 'None'}]},
           {'gpuId': 'NVIDIA RTX A5000', 'secureCloud': True, 'securePricePerHr': .27, 'memoryInGb': 24,
            'dataCenterAvailability': [{'dataCenterId': 'US-KS-2', 'stockStatus': 'Low'}]},
           {'gpuId': 'NVIDIA RTX A4000', 'secureCloud': True, 'securePricePerHr': .25,
            'dataCenterAvailability': [{'dataCenterId': 'CA-MTL-1', 'stockStatus': 'High'}]},
           {'gpuId': 'NVIDIA L40S', 'secureCloud': True, 'securePricePerHr': 1.09,
            'dataCenterAvailability': [{'dataCenterId': 'US-GA-1', 'stockStatus': 'High'}]},
           {'gpuId': 'NVIDIA GeForce RTX 5090', 'secureCloud': True, 'securePricePerHr': .69,
            'dataCenterAvailability': [{'dataCenterId': 'EU-CZ-1', 'stockStatus': 'High'}]},
           {'gpuId': 'NVIDIA RTX A6000', 'secureCloud': True, 'securePricePerHr': .59}]


class Fake:
    """Scripted runpodctl/ssh/rsync responses keyed by the command prefix."""

    def __init__(self, pods=()):
        self.pods = list(pods)
        self.commands = []

    def __call__(self, command, check=True):
        self.commands.append(command)
        out, code = '', 0
        if command[:3] == ['runpodctl', 'pod', 'list']:
            out = json.dumps(self.pods)
        elif command[:3] == ['runpodctl', 'pod', 'get']:
            out = json.dumps({'id': command[3], 'ssh': {'ip': '1.2.3.4', 'port': 2222}})
        elif command[:3] == ['runpodctl', 'gpu', 'list']:
            out = json.dumps(CATALOG)
        elif command[:3] == ['runpodctl', 'pod', 'create']:
            out = json.dumps({'id': 'newpod1'})
            self.pods = [{'id': 'newpod1', 'name': 'azuracast-intro-detector', 'desiredStatus': 'RUNNING'}]
        elif command[:1] == ['ssh-keygen']:
            out = 'ssh-ed25519 AAAA test'
        elif command[:3] == ['runpodctl', 'pod', 'delete'] or command[:3] == ['runpodctl', 'pod', 'stop']:
            self.pods = []
        return SimpleNamespace(returncode=code, stdout=out, stderr='')


class Ranking(unittest.TestCase):
    def test_default_is_the_rtx_4090_only(self):
        self.assertEqual({gpu for gpu, *_ in rank(CATALOG, Settings(), songs=100)}, {'NVIDIA GeForce RTX 4090'})

    def test_cheapest_expected_batch_cost_first_region_irrelevant(self):
        order = [(gpu, dc) for gpu, dc, *_ in rank(CATALOG, Settings(gpus=MIXED), songs=100)]
        self.assertEqual(order, [('NVIDIA RTX A5000', 'US-KS-2'), ('NVIDIA GeForce RTX 4090', 'EU-SE-1'),
                                 ('NVIDIA GeForce RTX 4090', 'US-TX-3'), ('NVIDIA GeForce RTX 4090', 'EU-RO-1')])

    def test_minimum_speed_price_cap_and_unlisted_gpus(self):
        order = {gpu for gpu, *_ in rank(CATALOG, Settings(gpus=MIXED, min_speed=.6), songs=100)}
        self.assertEqual(order, {'NVIDIA GeForce RTX 4090'})
        with self.assertRaises(NoCapacity):
            rank(CATALOG, Settings(gpus=MIXED, min_speed=.6, max_price_per_hour=.5), songs=1)

    def test_lane_plan_respects_vram_and_batch_size(self):
        self.assertEqual(lane_plan(Settings(), 100, 24), (3, 10))
        self.assertEqual(lane_plan(Settings(), 20, 24), (3, 7))
        self.assertEqual(lane_plan(Settings(), 2, 24), (2, 1))
        self.assertEqual(lane_plan(Settings(), 20, 16), (2, 10))
        self.assertEqual(lane_plan(Settings(), 20, None), (3, 7))

    def test_merge_chunks_keeps_successful_chunks_only(self):
        with tempdir() as tmp:
            root = Path(tmp)
            for name, code, sha in (('0000', 0, 'a'), ('0001', 0, 'b'), ('0002', 1, 'c')):
                chunk = root / 'chunks' / name
                chunk.mkdir(parents=True)
                (chunk / 'exit').write_text(f'lane=0 exit={code}')
                (chunk / 'run.json').write_text(json.dumps({'stages': [{'stage': 'qwen', 'seconds': 10}]}))
                (chunk / 'report.json').write_text(json.dumps({'policy': 'p', 'rows': [{'sha256': sha}], 'errors': []}))
                (chunk / 'music-report.json').write_text(json.dumps({'profile': 'm', 'rows': [{'sha256': sha}], 'errors': []}))
            self.assertEqual(merge_chunks(root / 'chunks', root), (2, 3))
            report = json.loads((root / 'report.json').read_text())
            self.assertEqual((report['policy'], [r['sha256'] for r in report['rows']]), ('p', ['a', 'b']))
            self.assertEqual(json.loads((root / 'run.json').read_text())['stages'], [{'stage': 'qwen', 'seconds': 30}])

    def test_no_stock_messages(self):
        self.assertTrue(no_stock('{"error": "There are not enough free GPUs on the host machine"}'))
        self.assertFalse(no_stock('invalid registry'))

    def test_month_spend_counts_only_this_month(self):
        now = 1767225600. + 3600 * 24  # 2026-01-02 UTC
        rows = [dict(price=1., created_at=now - 7200, ended_at=now - 3600),
                dict(price=2., created_at=now - 1800, ended_at=None),
                dict(price=5., created_at=now - 40 * 86400, ended_at=now - 39 * 86400)]
        self.assertAlmostEqual(month_spend(rows, now), 2.)


class Lifecycle(unittest.TestCase):
    def setUp(self):
        self.tmp = tempdir()
        self.con = database(self.tmp.name)
        self.clock = Clock(1767312000.)

    def tearDown(self):
        self.con.close()
        self.tmp.cleanup()

    def runpod(self, fake, **settings):
        return Runpod(Settings(registry_auth_id='reg', **settings), self.con, self.tmp.name,
                      call=fake, sleep=self.clock.sleep, clock=self.clock)

    def test_creates_cheapest_option_without_workspace_volume(self):
        fake = Fake()
        self.assertEqual(self.runpod(fake, gpus=MIXED).ensure_worker(songs=100), 'newpod1')
        create = next(c for c in fake.commands if c[:3] == ['runpodctl', 'pod', 'create'])
        self.assertEqual(create[create.index('--gpu-id') + 1], 'NVIDIA RTX A5000')
        self.assertEqual(create[create.index('--container-disk-in-gb') + 1], '80')
        self.assertNotIn('--volume-in-gb', create)
        self.assertIn('--ssh', create)
        row = self.con.execute('SELECT gpu_id, gpu_memory_gb, price FROM pods').fetchone()
        self.assertEqual(tuple(row), ('NVIDIA RTX A5000', 24, .27))

    def test_public_worker_image_does_not_require_registry_credentials(self):
        fake = Fake()
        worker = Runpod(Settings(registry_auth_id=''), self.con, self.tmp.name,
                        call=fake, sleep=self.clock.sleep, clock=self.clock)
        self.assertEqual(worker.ensure_worker(), 'newpod1')
        create = next(command for command in fake.commands if command[:3] == ['runpodctl', 'pod', 'create'])
        self.assertNotIn('--registry-auth-id', create)

    def test_budget_blocks_creation(self):
        with self.con:
            self.con.execute('INSERT INTO pods(pod_id, price, created_at, ended_at) VALUES(?,?,?,?)',
                             ('old', 10., self.clock() - 7200, self.clock() - 1))
        with self.assertRaises(GpuError):
            self.runpod(Fake(), monthly_budget_usd=5.).ensure_worker()

    def test_stopped_worker_without_copy_needs_manual_recovery(self):
        fake = Fake([{'id': 'abc', 'name': 'azuracast-intro-detector', 'desiredStatus': 'EXITED'}])
        with self.assertRaises(GpuError):
            self.runpod(fake).ensure_worker()
        self.assertFalse(any(c[:3] == ['runpodctl', 'pod', 'delete'] for c in fake.commands))

    def test_idle_guard_deletes_copied_and_stops_uncopied(self):
        for copied, action in ((1, 'delete'), (0, 'stop')):
            with self.subTest(copied=copied):
                fake = Fake([{'id': 'abc', 'name': 'azuracast-intro-detector', 'desiredStatus': 'RUNNING'}])
                with self.con:
                    self.con.execute('DELETE FROM pods')
                    self.con.execute('INSERT INTO pods(pod_id, price, created_at, copied, idle_since) VALUES(?,?,?,?,?)',
                                     ('abc', .69, self.clock(), copied, self.clock()))
                runpod = self.runpod(fake, idle_seconds=600)
                self.assertEqual(runpod.idle_tick(), 'idle')
                self.clock.now += 601
                self.assertEqual(runpod.idle_tick(), 'retired')
                self.assertIn(['runpodctl', 'pod', action, 'abc'], fake.commands)


if __name__ == '__main__':
    unittest.main()
