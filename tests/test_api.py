import http.client
import json
import urllib.request
import unittest

from helpers import Clock, FakeClient, database, make_config, media, tempdir
from intro_detector.api import Api, serve
from intro_detector.gpu import Workers
from intro_detector.library import sync_station
from intro_detector.service import Service
from intro_detector.stations import SHARED, Instance, Secrets, Stations, station_account
from test_gpu import Fake

TOKEN = 't' * 40
KEY = 'rpa_VALIDKEY12345678'


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempdir()
        self.cfg = make_config(self.tmp.name)
        self.clock = Clock(1767312000.)
        self.client = FakeClient([media(1), media(2)])
        self.fake = Fake()
        self.accepted = {KEY}
        self.api = Api(TOKEN, self.open, key_works=lambda key: key in self.accepted)

    def tearDown(self):
        self.tmp.cleanup()

    def open(self):
        con = database(self.tmp.name)
        workers = Workers(self.cfg, con, Stations(con, self.cfg, self.clock), Instance(con, self.cfg),
                          Secrets(self.tmp.name), call=self.fake, sleep=self.clock.sleep, clock=self.clock)
        return Service(self.cfg, con, self.client, workers, clock=self.clock)

    def call(self, method, path, body=None, token=TOKEN):
        return self.api.handle(method, path, f'Bearer {token}', json.dumps(body).encode() if body else b'')

    def test_token_is_required(self):
        self.assertEqual(self.call('GET', '/api/v1/health', token='wrong')[0], 401)
        self.assertEqual(self.api.handle('GET', '/api/v1/health', None)[0], 401)
        self.assertEqual(self.call('GET', '/api/v1/health')[1]['api'], 1)
        with self.assertRaises(ValueError):
            Api('short', self.open)

    def test_unknown_paths_and_methods(self):
        self.assertEqual(self.call('GET', '/api/v1/nothing')[0], 404)
        self.assertEqual(self.call('DELETE', '/api/v1/stations/1')[0], 405)
        self.assertEqual(self.api.handle('PUT', '/api/v1/stations/1', f'Bearer {TOKEN}', b'[1]')[0], 400)

    def test_station_managers_enable_and_configure_their_station(self):
        code, body = self.call('PUT', '/api/v1/stations/3', {'enabled': True, 'mode': 'publish', 'budget_usd': 5})
        self.assertEqual(code, 200)
        self.assertEqual((body['settings']['enabled'], body['settings']['mode'], body['limit_usd']), (True, 'publish', 5.))
        self.assertEqual(body['jobs'], ['sync'])
        self.assertEqual(body['blocked'], 'no Runpod key')
        self.assertEqual(self.call('PUT', '/api/v1/stations/3', {'shared_allowed': True})[0], 400)
        self.assertEqual(self.call('PUT', '/api/v1/stations/3', {'billing': 'shared'})[0], 400)

    def test_instance_key_with_a_station_allowance(self):
        self.assertEqual(self.call('PUT', '/api/v1/instance/key', {'key': KEY})[0], 200)
        self.call('PUT', '/api/v1/instance/stations/3', {'shared_allowed': True, 'shared_cap_usd': 2})
        code, body = self.call('PUT', '/api/v1/stations/3', {'enabled': True, 'billing': 'shared'})
        self.assertEqual((code, body['settings']['account'], body['key'], body['limit_usd']), (200, SHARED, True, 2.))
        instance = self.call('GET', '/api/v1/instance')[1]
        self.assertEqual((instance['shared_key'], instance['stations'][0]['shared_allowed']), (True, True))

    def test_keys_are_validated_write_only_and_kept_while_a_worker_runs(self):
        code, body = self.call('PUT', '/api/v1/stations/3/key', {'key': 'rpa_REJECTED1234567'})
        self.assertEqual(code, 422)
        code, body = self.call('PUT', '/api/v1/stations/3/key', {'key': KEY})
        self.assertEqual((code, body['key']), (200, True))
        self.assertNotIn(KEY, json.dumps(body))
        self.assertNotIn(KEY, json.dumps(self.call('GET', '/api/v1/instance')[1]))
        con = database(self.tmp.name)
        with con:
            con.execute('INSERT INTO pods(pod_id, price, created_at, account) VALUES(?,?,?,?)',
                        ('p', .89, self.clock(), station_account(3)))
        self.assertEqual(self.call('DELETE', '/api/v1/stations/3/key')[0], 409)
        with con:
            con.execute('UPDATE pods SET ended_at=?', (self.clock(),))
        con.close()
        self.assertFalse(self.call('DELETE', '/api/v1/stations/3/key')[1]['key'])

    def test_process_publish_and_backfill_need_an_enabled_station(self):
        for path in ('process', 'publish', 'backfill'):
            self.assertEqual(self.call('POST', f'/api/v1/stations/1/{path}')[0], 409)
        self.call('PUT', '/api/v1/stations/1', {'enabled': True})
        self.assertTrue(self.call('POST', '/api/v1/stations/1/process', {'force': True})[1]['process_requested'])
        self.assertEqual(self.call('POST', '/api/v1/stations/1/publish')[0], 409)
        self.call('PUT', '/api/v1/stations/1', {'mode': 'publish'})
        self.assertIn('republish', self.call('POST', '/api/v1/stations/1/publish')[1]['jobs'])
        con = database(self.tmp.name)
        sync_station(con, self.client, self.open().settings.get(1).station, now=self.clock())
        con.close()
        code, body = self.call('POST', '/api/v1/stations/1/backfill', {'dry_run': True})
        self.assertEqual((code, body['songs'], body['station']['queue']), (200, 2, {'known': 2}))
        self.assertEqual(self.call('POST', '/api/v1/stations/1/backfill')[1]['station']['queue'], {'pending': 2})

    def test_retiring_unconfirmed_workers_needs_force(self):
        self.call('PUT', '/api/v1/instance/key', {'key': KEY})
        self.assertEqual(self.call('POST', '/api/v1/instance/worker/retire')[1], {'worker': None})
        self.fake.pods = [{'id': 'abc', 'name': 'azuracast-intro-detector', 'desiredStatus': 'EXITED'}]
        service = self.open()
        service.instance.set_marker(f'recovery:{SHARED}', 'stopped with unconfirmed evidence')
        service.con.close()
        self.assertEqual(self.call('POST', '/api/v1/instance/worker/retire')[0], 409)
        code, body = self.call('POST', '/api/v1/instance/worker/retire', {'force': True})
        self.assertEqual((code, body), (200, {'worker': 'abc', 'action': 'deleted'}))
        self.assertIsNone(self.call('GET', '/api/v1/instance')[1]['shared_recovery'])

    def test_http_server_round_trip(self):
        server = serve(self.api, host='127.0.0.1', port=0)
        base = f'http://127.0.0.1:{server.server_address[1]}'
        try:
            request = urllib.request.Request(base + '/api/v1/health', headers={'Authorization': f'Bearer {TOKEN}'})
            with urllib.request.urlopen(request, timeout=5) as response:
                self.assertEqual(json.loads(response.read())['status'], 'ok')
            connection = http.client.HTTPConnection('127.0.0.1', server.server_address[1], timeout=5)
            connection.putrequest('PUT', '/api/v1/instance')
            connection.putheader('Authorization', f'Bearer {TOKEN}')
            connection.putheader('Content-Length', '70000')
            connection.endheaders()
            self.assertEqual(connection.getresponse().status, 413)
            connection.close()
        finally:
            server.shutdown()
            server.server_close()


if __name__ == '__main__':
    unittest.main()
