import json
import os
from pathlib import Path
import sqlite3
import unittest
from unittest import mock
import zlib

from helpers import database, report_row, tempdir
from intro_detector import housekeeping
from intro_detector.db import MAGIC, connect, pack, unpack
from intro_detector.housekeeping import Housekeeping
from intro_detector.stations import Instance
from test_service import FakeGpu, QueueCase, sha


def legacy(value):
    return zlib.compress(json.dumps(value).encode())


def evidence(seed=0.):
    curve = [seed + i / 3000 for i in range(3000)]
    level = [-200.00001525878906 + i * .1 for i in range(3000)]
    return dict(song='a.mp3', duration=30., tags=['x', 1, 2.5], short=[.1, .2],
                acoustic={'original/4': dict(rms_db=level, curves={'vocal': curve}, frame_seconds=.01),
                          'original/8': dict(rms_db=list(level), curves={'vocal': curve})})


class Codec(unittest.TestCase):
    def test_round_trip_is_exact_and_stores_identical_arrays_once(self):
        value = evidence(1e-300)
        blob = pack(value)
        self.assertTrue(blob.startswith(MAGIC))
        self.assertEqual(unpack(blob), value)
        self.assertLess(len(blob), len(legacy(value)) / 3)
        distinct = pack(dict(value, acoustic={'original/4': value['acoustic']['original/4']}))
        self.assertLess(len(blob) - len(distinct), 200)

    def test_reads_the_format_of_earlier_releases(self):
        value = evidence()
        self.assertEqual(unpack(legacy(value)), value)


class Store(unittest.TestCase):
    def setUp(self):
        self.tmp = tempdir()
        self.root = Path(self.tmp.name)
        self.con = database(self.root)
        self.now = 1000.
        self.keeper = Housekeeping(self.con, self.root, Instance(self.con), lambda: self.now)

    def tearDown(self):
        self.con.close()
        self.tmp.cleanup()

    def analysis(self, sha256, blob=None):
        blob = blob or pack(report_row('a', sha256))
        with self.con:
            self.con.execute("INSERT INTO analyses(sha256, source, report, music, created_at) VALUES(?, 'pod', ?, ?, 0)",
                             (sha256, blob, blob))
            self.con.execute("INSERT INTO decisions(sha256, policy, raw, marker, value, created_at) "
                             "VALUES(?, 'p', '{}', '{}', 1, 0)", (sha256,))

    def song(self, media_id, sha256, status='done'):
        with self.con:
            self.con.execute("INSERT OR REPLACE INTO media(station_id, media_id, unique_id, path, mtime, sha256, status, "
                             "discovered_at, updated_at) VALUES(1, ?, 'u', 'a.mp3', 1, ?, ?, 0, 0)",
                             (media_id, sha256, status))

    def stored(self):
        return {r[0] for r in self.con.execute('SELECT sha256 FROM analyses')}

    def test_evidence_of_deleted_songs_is_removed_after_a_day(self):
        for name in 'abc':
            self.analysis(name)
        self.song(1, 'a')
        self.song(2, 'b', 'deleted')
        self.song(3, None, 'known')
        with self.con:
            self.con.execute("INSERT INTO publications(station_id, media_id, sha256, state, updated_at) "
                             "VALUES(1, 3, 'c', 'written', 0)")
        self.keeper.forget_unused()
        self.now += housekeeping.UNUSED_SECONDS - 1
        self.assertEqual(self.keeper.forget_unused(), 0)
        self.now += 2
        self.assertEqual(self.keeper.forget_unused(), 1)
        self.assertEqual(self.stored(), {'a', 'c'})
        self.assertEqual({r[0] for r in self.con.execute('SELECT sha256 FROM decisions')}, {'a', 'c'})

    def test_a_song_that_returns_keeps_its_evidence(self):
        self.analysis('a')
        self.song(1, 'a', 'deleted')
        self.keeper.forget_unused()
        self.song(1, 'a', 'known')
        self.keeper.forget_unused()
        self.now += housekeeping.UNUSED_SECONDS * 2
        self.song(1, 'a', 'deleted')
        self.assertEqual(self.keeper.forget_unused(), 0)
        self.assertEqual(self.stored(), {'a'})

    def test_converts_evidence_of_earlier_releases_once(self):
        values = {name: report_row(name, name) for name in ('a', 'b')}
        for name, value in values.items():
            self.analysis(name, legacy(value))
            self.song(ord(name), name)
        with mock.patch.object(housekeeping, 'CONVERT_ROWS', 1), mock.patch.object(housekeeping, 'CONVERT_SECONDS', 0):
            self.assertTrue(self.keeper.convert())
            self.assertEqual(sum(r[0].startswith(MAGIC) for r in self.con.execute('SELECT report FROM analyses')), 1)
            self.assertTrue(self.keeper.convert())
            self.assertFalse(self.keeper.convert())
        for row in self.con.execute('SELECT sha256, report, music FROM analyses'):
            self.assertTrue(row['report'].startswith(MAGIC) and row['music'].startswith(MAGIC))
            self.assertEqual(unpack(row['report']), values[row['sha256']])
        self.assertFalse(self.keeper.convert())

    def test_releases_free_pages_of_a_new_database(self):
        self.fill_and_delete(self.con)
        before = os.path.getsize(self.root / 'state.sqlite3')
        with mock.patch.object(housekeeping, 'SHRINK_BYTES', 1 << 20):
            self.assertTrue(self.keeper.shrink())
        self.assertEqual(self.con.execute('PRAGMA freelist_count').fetchone()[0], 0)
        self.assertLess(os.path.getsize(self.root / 'state.sqlite3'), before / 2)

    def test_rebuilds_a_database_without_incremental_vacuum_once(self):
        path = self.root / 'old.sqlite3'
        sqlite3.connect(path).close()
        con = connect(path)
        con.execute('PRAGMA auto_vacuum=NONE')
        con.execute('VACUUM')
        self.fill_and_delete(con)
        keeper = Housekeeping(con, self.root, Instance(con))
        with mock.patch.object(housekeeping, 'SHRINK_BYTES', 1 << 20):
            self.assertTrue(keeper.shrink())
        self.assertEqual(con.execute('PRAGMA auto_vacuum').fetchone()[0], housekeeping.INCREMENTAL)
        self.assertEqual(con.execute('PRAGMA freelist_count').fetchone()[0], 0)
        con.close()

    def fill_and_delete(self, con):
        with con:
            con.executemany("INSERT INTO analyses(sha256, source, report, music, created_at) VALUES(?, 'pod', ?, x'', 0)",
                            [(str(i), os.urandom(100_000)) for i in range(60)])
            con.execute('DELETE FROM analyses')

    def test_removes_downloads_of_interrupted_batches_only(self):
        batch, other = self.root / 'work' / ('a' * 32), self.root / 'work' / 'ab-test'
        (batch / 'songs').mkdir(parents=True)
        other.mkdir()
        self.keeper.clean_work()
        self.assertFalse(batch.exists())
        self.assertTrue(other.exists())


class DiskFull(QueueCase):
    def test_no_song_is_downloaded_while_the_disk_is_almost_full(self):
        gpu = FakeGpu()
        service = self.service(gpu, {sha(2): 5.})
        self.upload(2)
        self.clock.now += 61
        with mock.patch.object(housekeeping, 'free_bytes', return_value=housekeeping.RESERVE_BYTES - 1):
            self.assertEqual(service.process(), 'disk almost full')
            self.assertEqual(service.blocked(service.settings.get(1)), 'disk almost full')
        self.assertEqual(self.status(1, 2), ('pending', 0, None))
        self.assertEqual(service.process(), 'processed 1')
        self.assertEqual(len(gpu.batches), 1)


if __name__ == '__main__':
    unittest.main()
