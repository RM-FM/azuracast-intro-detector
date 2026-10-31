"""Runpod GPU worker lifecycle.

The worker runs the pipeline (pod/tools/run_ultimate.py) from the prepared image; pod/ is
synced before every batch, so pipeline changes need no new GPU image. gpu/run-lanes.sh runs
several staggered lanes that pull song chunks from
a shared queue, so CPU-only stages of one lane overlap GPU stages of another.
"""
import datetime
import fcntl
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


def run_command(command, check=True):
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
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


def rank(catalog, settings, songs):
    """Secure Cloud options ordered by expected batch cost; region does not matter."""
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
            if dc and stock in stock_tier:
                # Equal cost prefers the faster GPU, then better stock.
                options.append((round(float(price) * hours, 4), -speed, -stock_tier[stock], dc,
                                row['gpuId'], float(price), row.get('memoryInGb')))
    if not options:
        raise NoCapacity('No configured GPU is in stock within the price limit and minimum speed')
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


class Runpod:
    def __init__(self, settings, con, data_dir, call=run_command, sleep=time.sleep, clock=time.time):
        self.cfg, self.con, self.call, self.sleep, self.clock = settings, con, call, sleep, clock
        self.data_dir = Path(data_dir)
        self.key = Path(settings.ssh_key)
        self.last_pod_id = None

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
        return month_spend(self.con.execute('SELECT * FROM pods').fetchall(), self.clock())

    def retire(self, pod_id, delete):
        action = 'delete' if delete else 'stop'
        self.call(['runpodctl', 'pod', action, pod_id])
        if self._row(pod_id) is None:
            with self.con:
                self.con.execute('INSERT INTO pods(pod_id, price, created_at) VALUES(?,?,?)',
                                 (pod_id, 0., self.clock()))
        self._update(pod_id, ended_at=self.clock(), idle_since=None)
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
                        self.con.execute('INSERT INTO pods(pod_id, price, created_at) VALUES(?,?,?)',
                                         (pod['id'], self.cfg.max_price_per_hour, self.clock()))
                log.info('Reusing running worker %s', pod['id'])
                return pod['id']
            if pod.get('desiredStatus') == 'EXITED' and row is not None and row['copied']:
                self.retire(pod['id'], delete=True)
            else:
                raise GpuError(f'Worker {pod["id"]} is stopped with unconfirmed evidence. Inspect it, then '
                               f'run "intro-detector pod delete --force" before retrying.')
        budget = self.cfg.monthly_budget_usd
        if budget and self.spend() >= budget:
            raise BudgetExceeded(f'Estimated GPU spend this month reached {budget:.2f} USD; no Pod created')
        deadline = self.clock() + self.cfg.wait_minutes * 60
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
                self.con.execute('INSERT OR REPLACE INTO pods(pod_id, gpu_id, gpu_memory_gb, price, created_at) '
                                 'VALUES(?,?,?,?,?)', (pod_id, gpu_id, memory, price, self.clock()))
            log.info('Created worker %s: %s in %s', pod_id, gpu_id, dc)
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
    def run(self, songs_dir, output_dir):
        """Analyze every file in songs_dir; returns the local directory with pulled reports."""
        output_dir = Path(output_dir)
        lock_path = self.data_dir / '.gpu.lock'
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            songs = sum(1 for p in Path(songs_dir).iterdir() if p.is_file())
            pod_id = self.last_pod_id = self.ensure_worker(songs)
            self._update(pod_id, idle_since=None, copied=0)
            try:
                return self._run(pod_id, Path(songs_dir), output_dir)
            finally:
                self._update(pod_id, idle_since=self.clock())

    def _run(self, pod_id, songs_dir, output_dir):
        host, port = self.endpoint(pod_id)
        self._ssh(pod_id, host, port, 'nvidia-smi -L && test -d /workspace')
        if self.remote_busy(pod_id, host, port):
            raise PodBusy(f'Pod {pod_id} is still running an earlier batch')
        songs = sum(1 for p in songs_dir.iterdir() if p.is_file())
        row = self._row(pod_id)
        lanes, chunk = lane_plan(self.cfg, songs, row['gpu_memory_gb'] if row else None)
        destination = f'root@{host}:'
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
        status, end, last = None, self.clock() + self.cfg.max_run_minutes * 60, None
        while self.clock() < end:
            self.sleep(self.cfg.poll_seconds)
            try:
                status = json.loads(self._ssh(pod_id, host, port,
                                              shlex.join(['python3', f'{POD_DIR}/lanes_status.py', exit_file])).stdout)
            except (GpuError, ValueError) as error:
                log.warning('Status temporarily unavailable: %s', error)
                continue
            progress = f'{status["done"]}/{status["total"]} chunks done, {status["failed"]} failed'
            if progress != last:
                log.info('Pod: %s', progress)
                last = progress
            if status['exit'] is not None:
                break
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
                self.con.execute('UPDATE pods SET ended_at=?, idle_since=NULL WHERE ended_at IS NULL AND pod_id!=?',
                                 (self.clock(), pod['id'] if pod else ''))
            return None
        lock_path = self.data_dir / '.gpu.lock'
        with lock_path.open('a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return 'busy'
            row = self._row(pod['id'])
            if row is None:
                # A worker from a previous process: start a fresh idle clock, never orphan it.
                with self.con:
                    self.con.execute('INSERT INTO pods(pod_id, price, created_at, idle_since) VALUES(?,?,?,?)',
                                     (pod['id'], self.cfg.max_price_per_hour, self.clock(), self.clock()))
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
