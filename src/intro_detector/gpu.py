"""Runpod GPU worker lifecycle.

The worker runs the pipeline (pod/tools/run_ultimate.py) from the prepared image; pod/ is
synced before every batch, so pipeline changes need no new GPU image. gpu/run-lanes.sh runs
several staggered lanes that pull song chunks from
a shared queue, so CPU-only stages of one lane overlap GPU stages of another.
"""
import dataclasses
import datetime
import fcntl
import functools
import json
import logging
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import time
import uuid

from .stations import key_account, key_slot, runpod_account

log = logging.getLogger(__name__)

POD_DIR = '/workspace/whisper-test'
QUEUE = '/workspace/queue/songs'
LANES_RUN = '/workspace/lanes-run'
EVIDENCE = '/workspace/experiments/whisper-evidence'
# Pinned model revisions; the GPU image bakes the same ones (gpu/prefetch.py).
QWEN_REVISIONS = {'Qwen/Qwen3-ASR-1.7B': '7278e1e70fe206f11671096ffdd38061171dd6e5',
                  'Qwen/Qwen3-ForcedAligner-0.6B': 'c7cbfc2048c462b0d63a45797104fc9db3ad62b7'}
POD_SOURCE = Path(os.environ.get('INTRO_DETECTOR_POD_SOURCE', Path(__file__).resolve().parents[2] / 'pod'))
LANES_SOURCE = POD_SOURCE.parent / 'gpu'
EU_COUNTRIES = {'AT', 'BE', 'BG', 'CY', 'CZ', 'DE', 'DK', 'EE', 'ES', 'FI', 'FR', 'GR', 'HR', 'HU', 'IE', 'IT',
                'LT', 'LU', 'LV', 'MT', 'NL', 'PL', 'PT', 'RO', 'SE', 'SI', 'SK'}


class GpuError(RuntimeError):
    """The batch did not produce reports; queued songs stay pending."""


class NoCapacity(GpuError):
    pass


class BudgetExceeded(GpuError):
    pass


class PodBusy(GpuError):
    pass


class KeepPodRunning(GpuError):
    """Evidence could not be copied; the worker stays reachable for recovery."""


class MissingKey(GpuError):
    """No station holds a Runpod API key for this account."""


class WorkerNeedsRecovery(GpuError):
    """A stopped worker may hold reports that were never copied; an operator must decide."""


def run_command(command, check=True, env=None):
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, env=env)
    if check and result.returncode:
        raise GpuError(f'{shlex.join(command)} failed ({result.returncode}): '
                       f'{result.stderr.strip() or result.stdout.strip()}')
    return result


def no_stock(output):
    try:
        message = json.loads(output).get('error', output)
    except (ValueError, AttributeError):
        message = output
    return bool(re.search(r'no longer any instances available|not enough free GPUs|'
                          r'no available (?:gpus|machines|instances)', str(message), re.I))


def in_eu(data_center):
    """Runpod data center IDs name the country second, e.g. EU-RO-1; EUR-IS-1 is Iceland, not EU."""
    region, _, rest = str(data_center).partition('-')
    return region in ('EU', 'EUR') and rest.split('-')[0] in EU_COUNTRIES


def rank(catalog, settings, songs):
    """Secure Cloud options ordered by expected batch cost; any region unless EU only."""
    stock_tier = {'high': 3, 'medium': 2, 'low': 1}
    speeds = dict(settings.gpus)
    options = []
    for row in catalog:
        speed, price = speeds.get(row.get('gpuId')), row.get('securePricePerHr')
        if (speed is None or speed < settings.min_speed or not row.get('secureCloud')
                or price is None or float(price) > settings.max_price_per_hour):
            continue
        hours = settings.startup_minutes / 60 + songs * settings.song_seconds / 3600 / speed
        for site in row.get('dataCenterAvailability') or []:
            dc, stock = site.get('dataCenterId'), str(site.get('stockStatus', '')).lower()
            if dc and stock in stock_tier and (not settings.eu_only or in_eu(dc)):
                # Equal cost prefers the faster GPU, then better stock.
                options.append((round(float(price) * hours, 4), -speed, -stock_tier[stock], dc,
                                row['gpuId'], float(price), row.get('memoryInGb')))
    if not options:
        where = ' in the EU' if settings.eu_only else ''
        raise NoCapacity(f'No configured GPU is in stock{where} within the price limit and minimum speed')
    return [(gpu_id, dc, price, cost, memory) for cost, _, _, dc, gpu_id, price, memory in sorted(options)]


def lane_plan(settings, songs, memory_gb):
    """(lanes, songs per chunk): bounded by VRAM and by having at least one chunk per lane."""
    by_memory = int(memory_gb // settings.lane_vram_gb) if memory_gb else settings.lanes
    lanes = max(1, min(settings.lanes, by_memory, songs))
    return lanes, max(1, min(settings.chunk_songs, math.ceil(songs / lanes)))


def merge_chunks(results, output_dir):
    """Combine per-chunk Pod reports into one report.json / music-report.json / run.json."""
    report, music, chunks = None, None, {}
    for chunk in sorted(p for p in Path(results).iterdir() if p.is_dir() and not p.name.endswith('.tmp')):
        status = (chunk / 'exit').read_text().strip() if (chunk / 'exit').exists() else 'exit=?'
        run = json.loads((chunk / 'run.json').read_text()) if (chunk / 'run.json').exists() else {}
        chunks[chunk.name] = dict(status=status, stages=run.get('stages', []))
        if 'exit=0' not in status or not (chunk / 'report.json').exists() or not (chunk / 'music-report.json').exists():
            continue
        part, tune = (json.loads((chunk / name).read_text()) for name in ('report.json', 'music-report.json'))
        if report is None:
            report = {**part, 'rows': [], 'errors': []}
            music = {**tune, 'rows': [], 'errors': []}
        report['rows'] += part['rows']
        report['errors'] += part.get('errors', [])
        music['rows'] += tune['rows']
        music['errors'] += tune.get('errors', [])
    output_dir = Path(output_dir)
    totals = {}
    for chunk in chunks.values():
        for stage in chunk['stages']:
            totals[stage['stage']] = totals.get(stage['stage'], 0) + (stage.get('seconds') or 0)
    (output_dir / 'run.json').write_text(json.dumps(dict(
        chunks=chunks, stages=[dict(stage=k, seconds=v) for k, v in totals.items()]), indent=2))
    if report is not None:
        (output_dir / 'report.json').write_text(json.dumps(report))
        (output_dir / 'music-report.json').write_text(json.dumps(music))
    return sum('exit=0' in c['status'] for c in chunks.values()), len(chunks)


def month_spend(rows, now):
    start = datetime.datetime.fromtimestamp(now, datetime.timezone.utc).replace(
        day=1, hour=0, minute=0, second=0, microsecond=0).timestamp()
    total = 0.
    for row in rows:
        begin, end = max(row['created_at'], start), min(row['ended_at'] or now, now)
        if end > begin:
            total += row['price'] * (end - begin) / 3600
    return total


def account_spend(con, account, now):
    return month_spend(con.execute('SELECT * FROM pods WHERE account=?', (account,)).fetchall(), now)


def station_spend(con, station_id, now):
    """Estimated GPU spend of a station this month, each Pod split by the songs it analyzed per station."""
    total = 0.
    for pod in con.execute('SELECT * FROM pods').fetchall():
        cost = month_spend([pod], now)
        if not cost:
            continue
        songs = dict(con.execute('SELECT s.station_id, SUM(s.songs) FROM run_stations s JOIN runs r ON r.id = s.run_id '
                                 'WHERE r.pod_id=? GROUP BY s.station_id', (pod['pod_id'],)).fetchall())
        if songs.get(station_id):
            total += cost * songs[station_id] / sum(songs.values())
    return total


def _no_progress(step, done=0, total=0, **values):
    return None


class Runpod:
    def __init__(self, settings, con, data_dir, call=None, sleep=time.sleep, clock=time.time,
                 account='default', api_key=None):
        env = None if api_key is None else {**os.environ, 'RUNPOD_API_KEY': api_key}
        self.call = call or functools.partial(run_command, env=env)
        self.cfg, self.con, self.sleep, self.clock = settings, con, sleep, clock
        self.account, self.api_key = account, api_key
        self.data_dir = Path(data_dir)
        self.key = Path(settings.ssh_key)
        self.last_pod_id = None
        self.progress = _no_progress

    @property
    def lock_path(self):
        return self.data_dir / f'.gpu-{self.account}.lock'

    # -- runpodctl -------------------------------------------------------
    def _json(self, command):
        result = self.call(command)
        try:
            data = json.loads(result.stdout)
        except ValueError:
            raise GpuError(f'Invalid JSON from {shlex.join(command)}') from None
        if isinstance(data, dict) and data.get('error'):
            raise GpuError(f'{shlex.join(command)}: {data["error"]}')
        return data

    def worker(self):
        pods = self._json(['runpodctl', 'pod', 'list', '--all'])
        matches = [p for p in pods if p.get('name') == self.cfg.worker_name]
        if len(matches) > 1:
            raise GpuError(f'Multiple {self.cfg.worker_name} Pods exist; inspect them before retrying')
        if matches and not re.fullmatch(r'[a-z0-9]+', str(matches[0].get('id', ''))):
            raise GpuError('Unexpected Pod ID; refusing to use it')
        return matches[0] if matches else None

    def _row(self, pod_id):
        return self.con.execute('SELECT * FROM pods WHERE pod_id=?', (pod_id,)).fetchone()

    def _update(self, pod_id, **values):
        with self.con:
            self.con.execute(f'UPDATE pods SET {", ".join(k + "=?" for k in values)} WHERE pod_id=?',
                             (*values.values(), pod_id))

    def spend(self):
        return account_spend(self.con, self.account, self.clock())

    def retire(self, pod_id, delete):
        action = 'delete' if delete else 'stop'
        self.call(['runpodctl', 'pod', action, pod_id])
        row = self._row(pod_id)
        if row is None:
            with self.con:
                self.con.execute('INSERT INTO pods(pod_id, price, created_at, account) VALUES(?,?,?,?)',
                                 (pod_id, 0., self.clock(), self.account))
        # A stopped Pod no longer bills GPU time; deleting it later must not extend its spend.
        ended = row['ended_at'] if row is not None and row['ended_at'] else self.clock()
        self._update(pod_id, ended_at=ended, idle_since=None)
        log.info('%s Pod %s', 'Deleted' if delete else 'Stopped', pod_id)

    def ensure_worker(self, songs=1):
        pod = self.worker()
        if pod:
            details = self._json(['runpodctl', 'pod', 'get', pod['id']])
            if details.get('networkVolumeId'):
                raise GpuError('Existing worker has a network volume; refusing to use it')
            row = self._row(pod['id'])
            if pod.get('desiredStatus') == 'RUNNING':
                if row is None:
                    with self.con:
                        self.con.execute('INSERT INTO pods(pod_id, price, created_at, account) VALUES(?,?,?,?)',
                                         (pod['id'], self.cfg.max_price_per_hour, self.clock(), self.account))
                if self.cfg.eu_only and not in_eu(row['location'] if row is not None else None):
                    if row is None or not row['copied']:
                        raise PodBusy(f'Worker {pod["id"]} runs outside the EU and holds uncopied reports')
                    log.info('Worker %s runs outside the EU; replacing it for EU-only songs', pod['id'])
                    self.retire(pod['id'], delete=True)
                else:
                    log.info('Reusing running worker %s', pod['id'])
                    self.progress('worker', detail=dict(phase='reuse'))
                    return pod['id']
            elif pod.get('desiredStatus') == 'EXITED' and row is not None and row['copied']:
                self.retire(pod['id'], delete=True)
            else:
                raise WorkerNeedsRecovery(f'Worker {pod["id"]} is stopped with unconfirmed evidence. Recover or '
                                          f'delete it in Sweepers > Intro Detector before retrying.')
        budget = self.cfg.monthly_budget_usd
        if budget and self.spend() >= budget:
            raise BudgetExceeded(f'Estimated GPU spend this month reached {budget:.2f} USD; no Pod created')
        deadline = self.clock() + self.cfg.wait_minutes * 60
        attempt = 1
        while True:
            try:
                catalog = self._json(['runpodctl', 'gpu', 'list', '--include-unavailable'])
                if not isinstance(catalog, list):
                    raise GpuError('Unexpected Runpod GPU catalog')
                return self._create(rank(catalog, self.cfg, songs))
            except NoCapacity:
                if self.clock() + self.cfg.retry_seconds > deadline:
                    raise
                log.info('No GPU at the capped price; retrying in %ss', self.cfg.retry_seconds)
                attempt += 1
                self.progress('worker', detail=dict(phase='no_stock', attempt=attempt,
                                                    max_price=self.cfg.max_price_per_hour))
                self.sleep(self.cfg.retry_seconds)

    def _public_key(self):
        result = self.call(['ssh-keygen', '-y', '-f', str(self.key)], check=False)
        key = result.stdout.strip()
        if result.returncode or not key.startswith(('ssh-rsa ', 'ssh-ed25519 ', 'ecdsa-sha2-')):
            raise GpuError(f'Could not derive a public SSH key from {self.key}')
        return key

    def _create(self, options):
        env = json.dumps({'PUBLIC_KEY': self._public_key()})
        registry = ['--registry-auth-id', self.cfg.registry_auth_id] if self.cfg.registry_auth_id else []
        for gpu_id, dc, price, cost, memory in options:
            # No volume: /workspace must come from the image, where environments and models are baked.
            command = ['runpodctl', 'pod', 'create', '--name', self.cfg.worker_name,
                       '--image', self.cfg.image, *registry,
                       '--gpu-id', gpu_id, '--gpu-count', '1', '--cloud-type', 'SECURE',
                       '--min-cuda-version', '12.1',
                       '--container-disk-in-gb', str(self.cfg.container_disk_gb),
                       '--data-center-ids', dc, '--ports', '22/tcp', '--env', env, '--ssh']
            log.info('Trying %s in %s at %.2f USD/hour (estimated batch cost %.2f USD)', gpu_id, dc, price, cost)
            self.progress('worker', detail=dict(phase='search', gpu=gpu_id, location=dc, price=price))
            reply = self.call(command, check=False)
            if reply.returncode:
                message = reply.stderr.strip() or reply.stdout.strip()
                if no_stock(message):
                    continue
                raise GpuError(f'Pod creation failed: {message}')
            try:
                pod_id = json.loads(reply.stdout)['id']
            except (ValueError, KeyError, TypeError):
                raise GpuError('Pod create returned no ID; check "runpodctl pod list --all", '
                               'a Pod may already be billing') from None
            if not isinstance(pod_id, str) or not re.fullmatch(r'[a-z0-9]+', pod_id):
                raise GpuError('Invalid Pod ID; check "runpodctl pod list --all"')
            with self.con:
                self.con.execute('INSERT OR REPLACE INTO pods(pod_id, gpu_id, gpu_memory_gb, price, created_at, account, '
                                 'location) VALUES(?,?,?,?,?,?,?)',
                                 (pod_id, gpu_id, memory, price, self.clock(), self.account, dc))
            log.info('Created worker %s: %s in %s', pod_id, gpu_id, dc)
            self.progress('worker', detail=dict(phase='boot', gpu=gpu_id, location=dc, price=price))
            return pod_id
        raise NoCapacity('No GPU location accepted a Pod within the price limit; no Pod created')

    # -- SSH -------------------------------------------------------------
    def endpoint(self, pod_id):
        """Wait until the Pod's public SSH port accepts our key."""
        deadline = self.clock() + self.cfg.ssh_minutes * 60
        while True:
            # runpodctl 3.0.1 "ssh info" can keep reporting "no container" while "pod get" already has the mapping.
            result = self.call(['runpodctl', 'pod', 'get', pod_id], check=False)
            try:
                data = json.loads(result.stdout).get('ssh') or {}
            except (ValueError, AttributeError):
                data = {}
            host, port = data.get('ip'), data.get('port')
            if (result.returncode == 0 and isinstance(port, int) and 0 < port < 65536 and isinstance(host, str) and host
                    and self._ssh(pod_id, host, port, 'true', check=False).returncode == 0):
                return host, port
            if self.clock() + 15 > deadline:
                raise GpuError(f'Pod {pod_id} SSH endpoint did not become ready')
            self.sleep(15)

    def _ssh_options(self, pod_id, port):
        known = self.data_dir / 'ssh' / f'known_hosts-{pod_id}'
        known.parent.mkdir(parents=True, exist_ok=True)
        return ['-i', str(self.key), '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=accept-new',
                '-o', f'UserKnownHostsFile={known}', '-o', 'ConnectTimeout=15', '-p', str(port)]

    def _ssh(self, pod_id, host, port, command, check=True):
        return self.call(['ssh', *self._ssh_options(pod_id, port), f'root@{host}', command], check=check)

    def _rsync(self, pod_id, port, *args):
        transport = shlex.join(['ssh', *self._ssh_options(pod_id, port)])
        return self.call(['rsync', '-a', '-e', transport, *args])

    def remote_busy(self, pod_id, host, port):
        return self._ssh(pod_id, host, port,
                         'mkdir -p /workspace/experiments && flock -n /workspace/experiments/.ultimate-run.lock -c true',
                         check=False).returncode != 0

    # -- batch -----------------------------------------------------------
    def run(self, songs_dir, output_dir, progress=None):
        """Analyze every file in songs_dir; returns the local directory with pulled reports.

        progress(step, done, total) is called when the batch reaches a new step or analyzes more chunks.
        """
        progress = self.progress = progress or _no_progress
        output_dir = Path(output_dir)
        lock_path = self.lock_path
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            songs = sum(1 for p in Path(songs_dir).iterdir() if p.is_file())
            progress('worker')
            pod_id = self.last_pod_id = self.ensure_worker(songs)
            self._update(pod_id, idle_since=None, copied=0)
            try:
                return self._run(pod_id, Path(songs_dir), output_dir, progress)
            finally:
                self._update(pod_id, idle_since=self.clock())

    def _run(self, pod_id, songs_dir, output_dir, progress):
        host, port = self.endpoint(pod_id)
        self._ssh(pod_id, host, port, 'nvidia-smi -L && test -d /workspace')
        if self.remote_busy(pod_id, host, port):
            raise PodBusy(f'Pod {pod_id} is still running an earlier batch')
        songs = sum(1 for p in songs_dir.iterdir() if p.is_file())
        row = self._row(pod_id)
        lanes, chunk = lane_plan(self.cfg, songs, row['gpu_memory_gb'] if row else None)
        destination = f'root@{host}:'
        size = sum(p.stat().st_size for p in songs_dir.iterdir() if p.is_file())
        progress('upload', detail=dict(songs=songs, megabytes=round(size / 1e6)))
        self._ssh(pod_id, host, port, f'mkdir -p {POD_DIR} {QUEUE} {EVIDENCE}/qwen')
        self._rsync(pod_id, port, '--exclude=__pycache__', str(POD_SOURCE / 'tools'), str(POD_SOURCE / 'experiments'),
                    str(LANES_SOURCE / 'run-lanes.sh'), str(LANES_SOURCE / 'lanes_status.py'), destination + POD_DIR + '/')
        self._rsync(pod_id, port, '--delete', str(songs_dir) + '/', destination + QUEUE + '/')
        lock = shlex.quote(json.dumps(QWEN_REVISIONS, indent=2) + '\n')
        self._ssh(pod_id, host, port, f'test -e {EVIDENCE}/qwen/models.json || printf %s {lock} > {EVIDENCE}/qwen/models.json')
        run_id = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ') + uuid.uuid4().hex[:8]
        exit_file = f'{POD_DIR}/ultimate-{run_id}.exit'
        self._ssh(pod_id, host, port, f'nohup bash {POD_DIR}/run-lanes.sh {run_id} {lanes} {chunk} '
                                      f'{self.cfg.stagger_seconds} </dev/null >/dev/null 2>&1 &')
        log.info('Batch %s running on %s: %d songs, %d lanes, chunks of %d', run_id, pod_id, songs, lanes, chunk)
        parts = math.ceil(songs / chunk)
        progress('analyze', 0, parts, detail=dict(phase='prepare', lanes=lanes))
        status, end, last = None, self.clock() + self.cfg.max_run_minutes * 60, None
        while self.clock() < end:
            self.sleep(self.cfg.poll_seconds)
            try:
                status = json.loads(self._ssh(pod_id, host, port,
                                              shlex.join(['python3', f'{POD_DIR}/lanes_status.py', exit_file])).stdout)
            except (GpuError, ValueError) as error:
                log.warning('Status temporarily unavailable: %s', error)
                continue
            summary = f'{status["done"]}/{status["total"]} chunks done, {status["failed"]} failed'
            if summary != (last or {}).get('summary'):
                log.info('Pod: %s', summary)
            stages = sorted({lane['stage'] for lane in status.get('lanes') or [] if lane.get('stage')})
            detail = dict(phase='run' if status.get('total') else 'prepare', lanes=lanes, stages=stages,
                          failed=status['failed'])
            update = dict(summary=summary, done=status.get('progress', status['done']), detail=detail)
            if update != last:
                progress('analyze', update['done'], status['total'] or parts, detail=detail)
                last = update
            if status['exit'] is not None:
                break
        progress('copy')
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            self._rsync(pod_id, port, '--ignore-missing-args', destination + f'{LANES_RUN}/results/',
                        destination + f'{LANES_RUN}/preflight.log', str(output_dir / 'chunks') + '/')
        except GpuError as error:
            raise KeepPodRunning(f'Could not copy reports from Pod {pod_id}; it stays running for recovery: {error}') from error
        self._update(pod_id, copied=1)
        if status is None or status['exit'] is None:
            raise GpuError(f'Batch {run_id} exceeded {self.cfg.max_run_minutes} minutes; its songs are retried')
        succeeded, total = merge_chunks(output_dir / 'chunks', output_dir)
        if not succeeded:
            raise GpuError(f'Batch {run_id} failed (exit {status["exit"]}, {total} chunks): '
                           f'{status.get("preflight") or "see chunk logs"}')
        if succeeded < total:
            log.warning('Batch %s: %d of %d chunks failed; their songs are retried', run_id, total - succeeded, total)
        return output_dir

    # -- idle guard ------------------------------------------------------
    def idle_tick(self):
        """Retire the worker after idle_seconds without work: delete if copied, else stop."""
        pod = self.worker()
        if pod is None or pod.get('desiredStatus') != 'RUNNING':
            with self.con:
                self.con.execute('UPDATE pods SET ended_at=?, idle_since=NULL '
                                 'WHERE ended_at IS NULL AND pod_id!=? AND account=?',
                                 (self.clock(), pod['id'] if pod else '', self.account))
            return None
        lock_path = self.lock_path
        with lock_path.open('a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return 'busy'
            row = self._row(pod['id'])
            if row is None:
                # A worker from a previous process: start a fresh idle clock, never orphan it.
                with self.con:
                    self.con.execute('INSERT INTO pods(pod_id, price, created_at, idle_since, account) VALUES(?,?,?,?,?)',
                                     (pod['id'], self.cfg.max_price_per_hour, self.clock(), self.clock(), self.account))
                return 'idle'
            if row['idle_since'] is None:
                self._update(pod['id'], idle_since=self.clock())
                return 'idle'
            if self.clock() - row['idle_since'] < self.cfg.idle_seconds:
                return 'idle'
            try:
                host, port = self.endpoint(pod['id'])
                if self.remote_busy(pod['id'], host, port):
                    self._update(pod['id'], idle_since=self.clock())
                    return 'busy'
            except GpuError as error:
                log.warning('Could not check remote lock on %s: %s', pod['id'], error)
            self.retire(pod['id'], delete=bool(row['copied']))
            return 'retired'


BALANCE_SECONDS = 600
# A key whose Runpod account could not be determined is asked again after this long.
RESOLVE_SECONDS = 600


class Workers:
    """One Runpod worker per Runpod account; stations whose keys belong to the same account share it.

    A key counts as its own account until Runpod has reported which account it belongs to.
    """

    def __init__(self, cfg, con, stations, instance, secrets, call=None, sleep=time.sleep, clock=time.time):
        self.cfg, self.con, self.stations, self.instance, self.secrets = cfg, con, stations, instance, secrets
        self.call, self.sleep, self.clock = call, sleep, clock
        self._workers, self._checked, self._resolved = {}, {}, {}

    def account_of_key(self, key):
        return (self.instance.marker('key_accounts') or {}).get(key_account(key), key_account(key))

    def account(self, station_id):
        """The GPU account a station's songs run on, or None without a Runpod key."""
        key = self.secrets.get(key_slot(station_id))
        return self.account_of_key(key) if key else None

    def accounts(self):
        return {self.account_of_key(key) for key in self.secrets.keys()}

    def key_stations(self, account):
        """Number of stations whose key belongs to this account."""
        return sum(1 for slot in self.secrets.slots() if self.account_of_key(self.secrets.get(slot)) == account)

    def resolve(self, key):
        """Group the key with other keys of the same Runpod account; True once it is grouped."""
        mapping = self.instance.marker('key_accounts') or {}
        if key_account(key) in mapping:
            return True
        self._resolved[key_account(key)] = self.clock()
        call = self.call or functools.partial(run_command, env={**os.environ, 'RUNPOD_API_KEY': key})
        try:
            user = json.loads(call(['runpodctl', 'user']).stdout)
            account = runpod_account(user['id'])
        except (GpuError, ValueError, KeyError, TypeError) as error:
            log.warning('Runpod account of a key unavailable: %s', error)
            return False
        old = key_account(key)
        # Records of a running worker keep their account until it is retired, so it is never orphaned.
        if self.con.execute('SELECT 1 FROM pods WHERE account=? AND ended_at IS NULL', (old,)).fetchone():
            return False
        names = self.instance.marker('worker_names') or {}
        if old in names:
            names.setdefault(account, names.pop(old))
            self.instance.set_marker('worker_names', names)
        for name in ('recovery', 'balance'):
            value = self.instance.marker(f'{name}:{old}')
            if value is not None:
                self.instance.set_marker(f'{name}:{account}', value)
                self.instance.set_marker(f'{name}:{old}', None)
        retry_at = self.instance.marker(f'retry_at:{old}')
        if retry_at is not None:
            current = self.instance.marker(f'retry_at:{account}')
            self.instance.set_marker(f'retry_at:{account}', max(retry_at, current or 0))
            self.instance.set_marker(f'retry_at:{old}', None)
        with self.con:
            for table in ('pods', 'runs'):
                self.con.execute(f'UPDATE {table} SET account=? WHERE account=?', (account, old))
        self.instance.set_marker('key_accounts', {**mapping, old: account})
        self._workers.pop(old, None)
        return True

    def resolve_keys(self):
        """Ask Runpod for the account of keys not grouped yet, at most every RESOLVE_SECONDS per key."""
        mapping = self.instance.marker('key_accounts') or {}
        for key in self.secrets.keys():
            asked = self._resolved.get(key_account(key), -math.inf)
            if key_account(key) not in mapping and self.clock() - asked >= RESOLVE_SECONDS:
                self.resolve(key)

    def balance(self, account, refresh=False):
        """Prepaid Runpod balance of the key's account, checked at most every BALANCE_SECONDS."""
        cached = self.instance.marker(f'balance:{account}')
        if not refresh and cached and self.clock() - cached['checked_at'] < BALANCE_SECONDS:
            return cached
        try:
            value = float(self.get(account)._json(['runpodctl', 'user'])['clientBalance'])
        except (GpuError, KeyError, TypeError, ValueError) as error:
            log.warning('Runpod balance of %s unavailable: %s', account, error)
            return cached
        cached = dict(balance_usd=round(value, 2), checked_at=self.clock())
        self.instance.set_marker(f'balance:{account}', cached)
        return cached

    def worker_name(self, account):
        names = self.instance.marker('worker_names') or {}
        return names.get(account) or f'{self.cfg.runpod.worker_name}-{account.removeprefix("runpod-")[:8]}'

    def idle_seconds(self, account):
        """Idle timeout of the account's worker: the longest of its stations, so none cuts another short."""
        minutes = [self.stations.get(int(slot.removeprefix('station-'))).idle_timeout_minutes
                   for slot in self.secrets.slots() if self.account_of_key(self.secrets.get(slot)) == account]
        return int(max(minutes) * 60) if minutes else self.cfg.runpod.idle_seconds

    def settings(self, account, eu_only=False):
        # Budgets belong to stations and are checked before a batch; the account itself has no limit.
        return dataclasses.replace(self.cfg.runpod, monthly_budget_usd=0., worker_name=self.worker_name(account),
                                   idle_seconds=self.idle_seconds(account), eu_only=eu_only)

    def get(self, account, eu_only=False):
        """The account's worker for one batch; eu_only when any song in it belongs to an EU-only station."""
        key = next((k for k in self.secrets.keys() if self.account_of_key(k) == account), None)
        if key is None:
            raise MissingKey('No station holds the Runpod key of this worker any more')
        worker = self._workers.get(account)
        if worker is None:
            worker = self._workers[account] = Runpod(
                self.settings(account, eu_only), self.con, self.cfg.data_dir, call=self.call, sleep=self.sleep,
                clock=self.clock, account=account, api_key=key)
        else:
            worker.cfg = self.settings(account, eu_only)
        return worker

    def spend(self, account):
        return account_spend(self.con, account, self.clock())

    def blocked(self, station_id):
        """Why the station cannot start a GPU batch now, or None."""
        if self.account(station_id) is None:
            return 'no Runpod key'
        budget = self.stations.get(station_id).budget_usd
        if budget and station_spend(self.con, station_id, self.clock()) >= budget:
            return 'monthly budget reached'
        return None

    def open_accounts(self):
        return {row[0] for row in self.con.execute('SELECT DISTINCT account FROM pods WHERE ended_at IS NULL')}

    def idle_tick(self):
        """Retire idle workers; accounts without an open worker are checked once per poll interval."""
        self.resolve_keys()
        now, open_accounts = self.clock(), self.open_accounts()
        for account in sorted(self.accounts() | open_accounts):
            if account not in open_accounts and now - self._checked.get(account, -math.inf) < self.cfg.poll_seconds:
                continue
            self._checked[account] = now
            try:
                self.get(account).idle_tick()
            except GpuError as error:
                log.warning('Idle guard for %s: %s', account, error)
            if account in self.accounts():
                self.balance(account)
