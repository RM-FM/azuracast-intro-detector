"""Minimal AzuraCast API client (stdlib only)."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import urllib.error
import urllib.parse
import urllib.request

MAX_DOWNLOAD = 1024 ** 3


class AzuraCastError(RuntimeError):
    pass


class _Redirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        # Storage backends may redirect downloads; never forward the key elsewhere.
        if new is not None and urllib.parse.urlsplit(newurl).netloc != urllib.parse.urlsplit(req.full_url).netloc:
            new.remove_header('X-api-key')
        return new


class Client:
    def __init__(self, base_url, api_key, timeout=60, opener=None):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        self.opener = opener or urllib.request.build_opener(_Redirects)

    def _open(self, method, path, body=None):
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(self.base_url + path, data=data, method=method, headers={
            'X-API-Key': self.api_key, 'Accept': 'application/json',
            'User-Agent': 'azuracast-intro-detector/0.1',
            **({'Content-Type': 'application/json'} if data is not None else {})})
        try:
            return self.opener.open(request, timeout=self.timeout)
        except urllib.error.HTTPError as error:
            detail = error.read(400).decode('utf-8', 'replace').strip()
            raise AzuraCastError(f'{method} {path}: HTTP {error.code} {detail}') from None
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise AzuraCastError(f'{method} {path}: {error}') from None

    def _json(self, method, path, body=None):
        with self._open(method, path, body) as response:
            try:
                return json.loads(response.read())
            except ValueError:
                raise AzuraCastError(f'{method} {path}: invalid JSON response') from None

    def _list(self, path, per_page=500):
        page, rows = 1, []
        while True:
            data = self._json('GET', f'{path}?per_page={per_page}&page={page}')
            if isinstance(data, list):
                return data
            if not isinstance(data, dict) or not isinstance(data.get('rows'), list):
                raise AzuraCastError(f'Unexpected list format from {path}')
            rows.extend(data['rows'])
            if page >= int(data.get('total_pages') or 1):
                return rows
            page += 1

    def list_files(self, station_id):
        return self._list(f'/api/station/{int(station_id)}/files')

    def can_read_media(self, station_id):
        """Raises AzuraCastError unless the API key may read the station's media library."""
        self._json('GET', f'/api/station/{int(station_id)}/files?per_page=1&page=1')

    def station_names(self):
        """{station_id: name} of the enabled stations the API key's user may view."""
        names = {}
        for item in self._list('/api/frontend/dashboard/stations'):
            station = item.get('station') if isinstance(item, dict) else None
            if isinstance(station, dict) and str(station.get('id', '')).isdigit():
                names[int(station['id'])] = station.get('name') or station.get('shortcode') or str(station['id'])
        return names

    def list_playlists(self, station_id):
        return self._list(f'/api/station/{int(station_id)}/playlists')

    def get_file(self, station_id, media_id):
        data = self._json('GET', f'/api/station/{int(station_id)}/file/{int(media_id)}')
        if not isinstance(data, dict) or data.get('id') != int(media_id):
            raise AzuraCastError(f'Unexpected media record for {media_id}')
        return data

    def download(self, station_id, media_id, directory, suffix):
        """Stream the original file, returning (path, sha256). Name is <sha256><suffix>."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        digest, size = hashlib.sha256(), 0
        fd, temporary = tempfile.mkstemp(dir=directory, suffix='.part')
        try:
            with os.fdopen(fd, 'wb') as out, self._open('GET', f'/api/station/{int(station_id)}/file/{int(media_id)}/play') as response:
                while block := response.read(1024 * 1024):
                    size += len(block)
                    if size > MAX_DOWNLOAD:
                        raise AzuraCastError(f'Media {media_id} exceeds {MAX_DOWNLOAD} bytes')
                    digest.update(block)
                    out.write(block)
            if size == 0:
                raise AzuraCastError(f'Media {media_id} download is empty')
            target = directory / (digest.hexdigest() + suffix)
            os.replace(temporary, target)
            return target, digest.hexdigest()
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def set_intro(self, station_id, media_id, value):
        data = self._json('POST', f'/api/station/{int(station_id)}/sweepers/intro/{int(media_id)}',
                          {'intro_end': value})
        if not isinstance(data, dict) or data.get('success') is not True:
            raise AzuraCastError(f'Sweepers plugin rejected intro_end for media {media_id}: {data}')
        return data.get('intro_end')
