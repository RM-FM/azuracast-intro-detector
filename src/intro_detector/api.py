"""Management API for the Sweepers plugin; reachable only on the internal Docker network.

Every request needs the bearer token from INTRO_DETECTOR_API_TOKEN. The plugin enforces
AzuraCast permissions; this API trusts it and never returns Runpod keys.
"""
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import os
import re
import threading

from . import status
from .gpu import GpuError, run_command
from .library import backfill
from .stations import SHARED, station_account, valid_key

log = logging.getLogger(__name__)
API_VERSION = 1
MAX_BODY = 64 * 1024
TOKEN_ENV = 'INTRO_DETECTOR_API_TOKEN'


class ApiError(Exception):
    def __init__(self, status_code, message):
        super().__init__(message)
        self.status_code = status_code


def runpod_key_works(key):
    """True when Runpod accepts the key; the key travels only in the environment."""
    reply = run_command(['runpodctl', 'user'], check=False, env={**os.environ, 'RUNPOD_API_KEY': key})
    return reply.returncode == 0


class Api:
    def __init__(self, token, open_service, key_works=runpod_key_works):
        if not token or len(token) < 32:
            raise ValueError(f'{TOKEN_ENV} must be at least 32 characters')
        self.token, self.open_service, self.key_works = token, open_service, key_works
        self.routes = [
            ('GET', r'/api/v1/health', self.health),
            ('GET', r'/api/v1/instance', self.get_instance),
            ('PUT', r'/api/v1/instance', self.put_instance),
            ('PUT', r'/api/v1/instance/key', self.put_shared_key),
            ('DELETE', r'/api/v1/instance/key', self.delete_shared_key),
            ('PUT', r'/api/v1/instance/stations/(\d+)', self.put_station_allowance),
            ('POST', r'/api/v1/instance/worker/retire', self.retire_shared_worker),
            ('GET', r'/api/v1/stations/(\d+)', self.get_station),
            ('PUT', r'/api/v1/stations/(\d+)', self.put_station),
            ('PUT', r'/api/v1/stations/(\d+)/key', self.put_station_key),
            ('DELETE', r'/api/v1/stations/(\d+)/key', self.delete_station_key),
            ('POST', r'/api/v1/stations/(\d+)/process', self.process),
            ('POST', r'/api/v1/stations/(\d+)/retry', self.retry),
            ('POST', r'/api/v1/stations/(\d+)/backfill', self.backfill),
            ('GET', r'/api/v1/stations/(\d+)/preview', self.preview),
            ('POST', r'/api/v1/stations/(\d+)/publish', self.publish),
            ('POST', r'/api/v1/stations/(\d+)/worker/retire', self.retire_station_worker),
        ]

    def authorized(self, header):
        scheme, _, token = (header or '').partition(' ')
        return scheme == 'Bearer' and hmac.compare_digest(token.encode(), self.token.encode())

    def handle(self, method, path, authorization, body=b''):
        """(HTTP status, JSON payload) for one request."""
        if not self.authorized(authorization):
            return 401, {'error': 'Unauthorized'}
        try:
            data = json.loads(body or b'{}')
        except ValueError:
            return 400, {'error': 'Invalid JSON'}
        if not isinstance(data, dict):
            return 400, {'error': 'Expected a JSON object'}
        allowed = False
        for route_method, pattern, handler in self.routes:
            match = re.fullmatch(pattern, path)
            if not match:
                continue
            allowed = True
            if route_method != method:
                continue
            service = self.open_service()
            try:
                return 200, handler(service, data, *(int(g) for g in match.groups()))
            except ApiError as error:
                return error.status_code, {'error': str(error)}
            except ValueError as error:
                return 400, {'error': str(error)}
            except GpuError as error:
                return 502, {'error': str(error)}
            finally:
                service.con.close()
        return (405, {'error': 'Method not allowed'}) if allowed else (404, {'error': 'Not found'})

    # -- instance ----------------------------------------------------------
    def health(self, service, data):
        return dict(status='ok', version=status.version(), api=API_VERSION)

    def get_instance(self, service, data):
        return status.instance(service)

    def put_instance(self, service, data):
        service.instance.update(data)
        return status.instance(service)

    def put_shared_key(self, service, data):
        self._store_key(service, SHARED, data)
        return status.instance(service)

    def delete_shared_key(self, service, data):
        self._remove_key(service, SHARED)
        return status.instance(service)

    def put_station_allowance(self, service, data, station_id):
        service.settings.update(station_id, data, admin=True)
        return status.instance(service)

    def retire_shared_worker(self, service, data):
        return self._retire(service, SHARED, data)

    # -- stations ----------------------------------------------------------
    def get_station(self, service, data, station_id):
        return status.station(service, station_id)

    def put_station(self, service, data, station_id):
        before = service.settings.get(station_id)
        after = service.settings.update(station_id, data)
        if after.enabled and not before.enabled:
            service.queue_job(station_id, 'sync')
        return status.station(service, station_id)

    def put_station_key(self, service, data, station_id):
        self._store_key(service, station_account(station_id), data)
        return status.station(service, station_id)

    def delete_station_key(self, service, data, station_id):
        self._remove_key(service, station_account(station_id))
        return status.station(service, station_id)

    def _enabled(self, service, station_id):
        settings = service.settings.get(station_id)
        if not settings.enabled:
            raise ApiError(409, 'The Intro Detector is not enabled for this station')
        return settings

    def process(self, service, data, station_id):
        self._enabled(service, station_id)
        force = data.get('force', False)
        if not isinstance(force, bool):
            raise ValueError('force must be true or false')
        service.request(station_id, force)
        return status.station(service, station_id)

    def retry(self, service, data, station_id):
        media_id = data.get('media_id')
        if media_id is not None and (not isinstance(media_id, int) or isinstance(media_id, bool) or media_id < 1):
            raise ValueError('media_id must be a positive integer')
        return dict(requeued=service.retry(station_id, media_id), station=status.station(service, station_id))

    def backfill(self, service, data, station_id):
        """Queue the existing library; dry_run reports the count without queuing."""
        self._enabled(service, station_id)
        options = {key: data.get(key, False) for key in ('include_set', 'dry_run')}
        if not all(isinstance(value, bool) for value in options.values()):
            raise ValueError('include_set and dry_run must be true or false')
        count = backfill(service.con, station_id, None, options['include_set'], options['dry_run'],
                         now=service.clock())
        return dict(songs=count, estimated_usd=_estimate(service, count),
                    station=status.station(service, station_id))

    def preview(self, service, data, station_id):
        """What publishing would write: stored decisions not yet in AzuraCast."""
        counts = dict(service.con.execute('SELECT state, COUNT(*) FROM publications WHERE station_id=? '
                                          'GROUP BY state', (station_id,)).fetchall())
        rows = [dict(r) for r in service.con.execute(
            "SELECT p.media_id, m.path, p.value, m.intro_end AS current FROM publications p "
            "JOIN media m ON m.station_id = p.station_id AND m.media_id = p.media_id "
            "WHERE p.station_id=? AND p.state='planned' ORDER BY m.path LIMIT 100", (station_id,))]
        return dict(counts=counts, planned=rows)

    def publish(self, service, data, station_id):
        settings = self._enabled(service, station_id)
        if settings.mode != 'publish':
            raise ApiError(409, 'Switch the station to publish mode first')
        service.queue_job(station_id, 'republish')
        return status.station(service, station_id)

    def retire_station_worker(self, service, data, station_id):
        return self._retire(service, station_account(station_id), data)

    # -- shared helpers ----------------------------------------------------
    def _store_key(self, service, account, data):
        key = valid_key(data.get('key'))
        if not self.key_works(key):
            raise ApiError(422, 'Runpod rejected this API key; check that it may manage Pods')
        service.gpus.secrets.set(account, key)
        log.info('Runpod key stored for %s', account)

    def _remove_key(self, service, account):
        if status.worker(service.con, account):
            raise ApiError(409, 'A GPU worker of this account is still active; remove the key after it is retired')
        service.gpus.secrets.set(account, None)
        log.info('Runpod key removed for %s', account)

    def _retire(self, service, account, data):
        """Stop or delete the account's worker; deleting unconfirmed evidence needs force."""
        delete, force = data.get('delete', True), data.get('force', False)
        if not isinstance(delete, bool) or not isinstance(force, bool):
            raise ValueError('delete and force must be true or false')
        gpu = service.gpus.get(account)
        pod = gpu.worker()
        if pod is None:
            service.instance.set_marker(f'recovery:{account}', None)
            return dict(worker=None)
        row = gpu._row(pod['id'])
        if delete and not (row and row['copied']) and not force:
            raise ApiError(409, 'Reports of this worker were not copied; deleting it loses them')
        gpu.retire(pod['id'], delete=delete)
        if delete:
            service.instance.set_marker(f'recovery:{account}', None)
        return dict(worker=pod['id'], action='deleted' if delete else 'stopped')


def _estimate(service, songs):
    per_thousand = status.cost_per_thousand(service.con)
    return None if per_thousand is None else round(songs * per_thousand / 1000, 2)


class _Handler(BaseHTTPRequestHandler):
    api = None

    def _respond(self):
        length = int(self.headers.get('Content-Length') or 0)
        if length > MAX_BODY:
            code, payload = 413, {'error': 'Request too large'}
        else:
            code, payload = self.api.handle(self.command, self.path.split('?', 1)[0],
                                            self.headers.get('Authorization'), self.rfile.read(length))
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    do_GET = do_PUT = do_POST = do_DELETE = _respond

    def log_message(self, format, *args):
        log.debug('API %s', format % args)


def serve(api, host='0.0.0.0', port=8080):
    """Start the API in a daemon thread; returns the server (server_address holds the bound port)."""
    handler = type('Handler', (_Handler,), {'api': api})
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, name='api', daemon=True).start()
    log.info('Management API listening on port %s', server.server_address[1])
    return server
