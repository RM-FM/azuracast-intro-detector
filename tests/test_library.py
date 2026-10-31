import unittest

from helpers import FakeClient, database, media, tempdir
from intro_detector.config import Station
from intro_detector.library import backfill, sync_station


class Library(unittest.TestCase):
    def setUp(self):
        self.tmp = tempdir()
        self.con = database(self.tmp.name)
        self.station = Station(1, 'publish', exclude_playlists=(9,))

    def tearDown(self):
        self.con.close()
        self.tmp.cleanup()

    def status(self, media_id):
        return self.con.execute('SELECT status FROM media WHERE media_id=?', (media_id,)).fetchone()['status']

    def test_first_sync_records_library_without_queueing(self):
        client = FakeClient([media(1), media(2, intro='12.5'), media(3, playlists=[9]), media(4, path='x.txt')])
        sync_station(self.con, client, self.station, now=10)
        self.assertEqual([self.status(i) for i in (1, 2, 3, 4)], ['known', 'known', 'excluded', 'excluded'])

    def test_jingle_mode_playlists_are_excluded(self):
        client = FakeClient([media(1, playlists=[4]), media(2, playlists=[4, 5]), media(3, playlists=[5])],
                            playlists=[{'id': 4, 'is_jingle': False}, {'id': 5, 'is_jingle': True}])
        sync_station(self.con, client, self.station, now=10)
        self.assertEqual([self.status(i) for i in (1, 2, 3)], ['known', 'excluded', 'excluded'])
        client.playlists[1]['is_jingle'] = False
        sync_station(self.con, client, self.station, now=20)
        self.assertEqual([self.status(i) for i in (2, 3)], ['pending', 'pending'])

    def test_short_or_unknown_length_is_excluded(self):
        client = FakeClient([media(1, length=59.9), media(2, length=60), media(3, length=0)])
        sync_station(self.con, client, self.station, now=10)
        self.assertEqual([self.status(i) for i in (1, 2, 3)], ['excluded', 'known', 'excluded'])
        client.files[3] = media(3, length=180)
        sync_station(self.con, client, self.station, now=20)
        self.assertEqual(self.status(3), 'pending')

    def test_later_uploads_and_replacements_queue(self):
        client = FakeClient([media(1)])
        sync_station(self.con, client, self.station, now=10)
        client.files[2] = media(2)
        sync_station(self.con, client, self.station, now=20)
        self.assertEqual(self.status(2), 'pending')
        with self.con:
            self.con.execute("UPDATE media SET status='done', sha256='abc' WHERE media_id=2")
        client.files[2] = media(2, mtime=5)
        sync_station(self.con, client, self.station, now=30)
        row = self.con.execute('SELECT * FROM media WHERE media_id=2').fetchone()
        self.assertEqual((row['status'], row['sha256'], row['discovered_at']), ('pending', None, 30))

    def test_removed_and_excluded_files(self):
        client = FakeClient([media(1), media(2)])
        sync_station(self.con, client, self.station, now=10)
        del client.files[1]
        client.files[2] = media(2, playlists=[9])
        sync_station(self.con, client, self.station, now=20)
        self.assertEqual((self.status(1), self.status(2)), ('deleted', 'excluded'))

    def test_backfill_only_missing_by_default(self):
        client = FakeClient([media(1), media(2, intro='0'), media(3)])
        sync_station(self.con, client, self.station, now=10)
        self.assertEqual(backfill(self.con, 1, dry_run=True), 2)
        self.assertEqual(backfill(self.con, 1, limit=1), 1)
        row = self.con.execute('SELECT status, priority FROM media WHERE media_id=1').fetchone()
        self.assertEqual(tuple(row), ('pending', 1))
        self.assertEqual(backfill(self.con, 1, include_set=True), 2)


if __name__ == '__main__':
    unittest.main()
