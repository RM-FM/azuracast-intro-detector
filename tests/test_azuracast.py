import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import unittest

from helpers import tempdir
from intro_detector.azuracast import AzuraCastError, Client


class Handler(BaseHTTPRequestHandler):
    seen = []

    def log_message(self, *args):
        pass

    def reply(self, code, body, kind='application/json'):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header('Content-Type', kind)
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self.seen.append((self.path, self.headers.get('X-API-Key')))
        if self.headers.get('X-API-Key') != 'k' and not self.path.startswith('/storage'):
            return self.reply(403, {'message': 'denied'})
        if self.path.startswith('/api/station/1/files?'):
            page = int(self.path.rsplit('page=', 1)[1])
            return self.reply(200, {'page': page, 'total_pages': 2, 'rows': [{'id': page}]})
        if self.path.startswith('/api/frontend/dashboard/stations?'):
            return self.reply(200, {'page': 1, 'total_pages': 1, 'rows': [
                {'station': {'id': 3, 'name': 'Rock', 'shortcode': 'rock'}}, {'station': {'id': 'x'}}]})
        if self.path == '/api/station/1/file/7/play':
            return self.reply(200, b'ID3-audio', 'audio/mpeg')
        if self.path == '/api/station/1/file/8/play':
            self.send_response(302)
            self.send_header('Location', f'http://localhost:{self.server.server_port}/storage/8')
            self.end_headers()
            return None
        if self.path == '/storage/8':
            return self.reply(200, b'redirected')
        if self.path == '/api/station/1/file/7':
            return self.reply(200, {'id': 7, 'custom_fields': {'intro_end': '3'}})
        return self.reply(404, {})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        if self.path == '/api/station/1/sweepers/intro/7' and set(body) == {'intro_end'}:
            return self.reply(200, {'success': True, 'intro_end': body['intro_end']})
        return self.reply(400, {'success': False})


class ClientTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.client = Client(f'http://127.0.0.1:{cls.server.server_port}', 'k', timeout=5)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def test_paginated_listing(self):
        self.assertEqual(self.client.list_files(1), [{'id': 1}, {'id': 2}])

    def test_station_list_uses_authenticated_dashboard(self):
        self.assertEqual(self.client.station_names(), {3: 'Rock'})
        self.assertNotIn('/api/stations', [path for path, _ in Handler.seen])

    def test_download_names_file_by_hash(self):
        with tempdir() as tmp:
            path, sha = self.client.download(1, 7, tmp, '.mp3')
            self.assertEqual(sha, hashlib.sha256(b'ID3-audio').hexdigest())
            self.assertEqual(path.name, sha + '.mp3')

    def test_redirect_to_other_host_drops_api_key(self):
        with tempdir() as tmp:
            _, sha = self.client.download(1, 8, tmp, '.mp3')
        self.assertEqual(sha, hashlib.sha256(b'redirected').hexdigest())
        self.assertIn(('/storage/8', None), Handler.seen)

    def test_single_field_write_and_errors(self):
        self.assertEqual(self.client.set_intro(1, 7, 14.5), 14.5)
        self.assertEqual(self.client.get_file(1, 7)['custom_fields']['intro_end'], '3')
        with self.assertRaises(AzuraCastError):
            Client(self.client.base_url, 'wrong', timeout=5).get_file(1, 7)


if __name__ == '__main__':
    unittest.main()
