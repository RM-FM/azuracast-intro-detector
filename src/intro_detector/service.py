"""Queue processing: download, analyze on the GPU worker of the station's Runpod key, decide and publish."""
from collections import Counter
import json
import logging
from pathlib import PurePosixPath
import shutil
import time
import uuid

from . import evidence, publisher
from .azuracast import AzuraCastError
from .db import pack, unpack
from .gpu import BudgetExceeded, GpuError, KeepPodRunning, MissingKey, NoCapacity, PodBusy, WorkerNeedsRecovery
from .library import sync_station
from .stations import Instance, Stations

log = logging.getLogger(__name__)
TRANSIENT = (NoCapacity, BudgetExceeded, PodBusy, KeepPodRunning, MissingKey, WorkerNeedsRecovery, BlockingIOError)
RETRY_SECONDS = 900
# publish: write intros recorded as planned by an earlier shadow mode.
JOBS = ('sync', 'republish', 'publish')
ACTIVITY = 'activity'


class Service:
    def __init__(self, cfg, con, client, gpus, selector_factory=None, clock=time.time, sleep=time.sleep):
        self.cfg, self.con, self.client, self.gpus = cfg, con, client, gpus
        self.clock, self.sleep = clock, sleep
        self.settings = Stations(con, cfg, clock)
        self.instance = Instance(con)
        self._selector, self._selector_factory = None, selector_factory
        # Per GPU account: no batch before this time after a failed one.
        self.retry_at = {}

    @property
    def selector(self):
        if self._selector is None:
            if self._selector_factory is None:
                from .selector import Selector
                self._selector = Selector(vocals=self.cfg.decision.vocals)
            else:
                self._selector = self._selector_factory()
        return self._selector

    def stations(self):
        return [s.station for s in self.settings.enabled()]

    def station(self, station_id):
        return self.settings.get(station_id).station

    # -- decisions -------------------------------------------------------
    def reviews(self, sha256):
        return {r['kind'] for r in self.con.execute('SELECT kind FROM reviews WHERE sha256=?', (sha256,))}

    def evaluate(self, report, music, sha256):
        """Selector and resolver on stored evidence without invented subtitle credits; no side effects."""
        report = evidence.clean(report)
        raw = self.selector.predict(report)
        marker = publisher.resolve(raw, report, music, self.reviews(sha256), self.cfg.decision)
        return raw, marker, publisher.target_value(marker)

    def decide(self, sha256):
        row = self.con.execute('SELECT * FROM analyses WHERE sha256=?', (sha256,)).fetchone()
        if row is None:
            return None
        raw, marker, value = self.evaluate(unpack(row['report']), unpack(row['music']), sha256)
        policy = f'{raw["policy"]}+{publisher.POLICY}{self.cfg.decision.tag}'
        with self.con:
            self.con.execute('INSERT OR REPLACE INTO decisions(sha256, policy, raw, marker, value, created_at) '
                             'VALUES(?,?,?,?,?,?)', (sha256, policy, json.dumps(raw), json.dumps(marker),
                                                     value, self.clock()))
        return marker, value

    def finalize(self, media, sha256, force_publish=False):
        """Decide from stored evidence and publish to AzuraCast; True when the item is done."""
        decided = self.decide(sha256)
        if decided is None:
            return False
        marker, value = decided
        state = publisher.publish(
            self.con, self.client, self.station(media['station_id']), media['media_id'], sha256, value,
            self.clock(), force=force_publish)
        with self.con:
            self.con.execute("UPDATE media SET status='done', error=NULL, updated_at=? WHERE station_id=? AND media_id=?",
                             (self.clock(), media['station_id'], media['media_id']))
        log.info('Station %s media %s (%s): %s %s -> %s', media['station_id'], media['media_id'], media['path'],
                 marker.get('intro_type') or marker.get('intro_reason'), value, state)
        return True

    def republish(self, station_id, media_id=None, force=False):
        query = "SELECT * FROM media WHERE station_id=? AND status='done' AND sha256 IS NOT NULL"
        args = [station_id]
        if media_id is not None:
            query += ' AND media_id=?'
            args.append(media_id)
        return sum(
            self.finalize(m, m['sha256'], force_publish=force)
            for m in self.con.execute(query, args).fetchall())

    def publish_planned(self, station_id):
        """Write the intros that were only recorded while the station did not publish."""
        rows = self.con.execute(
            "SELECT m.* FROM media m JOIN publications p ON p.station_id = m.station_id AND p.media_id = m.media_id "
            "WHERE m.station_id=? AND p.state='planned' AND m.status='done' AND m.sha256 IS NOT NULL",
            (station_id,)).fetchall()
        return sum(self.finalize(m, m['sha256']) for m in rows)

    def _failed(self, media, error):
        attempts = media['attempts'] + 1
        status = 'failed' if attempts >= self.cfg.max_attempts else 'pending'
        with self.con:
            self.con.execute('UPDATE media SET status=?, attempts=?, error=?, updated_at=? WHERE station_id=? AND media_id=?',
                             (status, attempts, str(error)[:2000], self.clock(), media['station_id'], media['media_id']))
        log.warning('Station %s media %s: %s (%s)', media['station_id'], media['media_id'], error, status)

    # -- queue -----------------------------------------------------------
    def sync(self, station_ids=None):
        for station in self.stations():
            if station_ids is not None and station.id not in station_ids:
                continue
            try:
                counts = sync_station(self.con, self.client, station, self.clock())
                log.info('Station %s synced: %s', station.id, counts)
            except AzuraCastError as error:
                log.error('Station %s sync failed: %s', station.id, error)

    def request(self, station_id=None, force=False):
        """Process now: skip the quiet period, also without automatic processing."""
        ids = [station_id] if station_id is not None else [s.id for s in self.settings.enabled()]
        with self.con:
            self.con.executemany('INSERT OR REPLACE INTO process_requests(station_id, force, requested_at) '
                                 'VALUES(?,?,?)', [(i, int(force), self.clock()) for i in ids])
        return len(ids)

    def requests(self):
        return {r['station_id']: bool(r['force']) for r in self.con.execute('SELECT * FROM process_requests')}

    def activity(self):
        """The running batch: step, counts, start times and songs per station (keys are station IDs as text)."""
        return self.instance.marker(ACTIVITY)

    def _begin_activity(self, account, rows):
        now = self.clock()
        self.instance.set_marker(ACTIVITY, dict(
            account=account, step='download', done=0, total=len(rows), started_at=now, step_started_at=now,
            songs=len(rows), stations={str(k): v for k, v in Counter(r['station_id'] for r in rows).items()}))

    def _progress(self, step, done=0, total=0, **values):
        current = self.activity()
        if current is None:
            return
        if current.get('step') != step:
            current['step_started_at'] = self.clock()
        # What the step is doing right now, e.g. the GPU tried or the pipeline stage; replaced by every update.
        current.update(values, step=step, done=done, total=total, detail=values.get('detail'))
        self.instance.set_marker(ACTIVITY, current)

    def queue_job(self, station_id, kind):
        """Sync or republish a station from the service loop instead of the caller."""
        if kind not in JOBS:
            raise ValueError(f'Unknown job {kind!r}')
        with self.con:
            self.con.execute('INSERT OR REPLACE INTO station_jobs(station_id, kind, requested_at) VALUES(?,?,?)',
                             (int(station_id), kind, self.clock()))

    def jobs(self):
        return [(r['station_id'], r['kind']) for r in
                self.con.execute('SELECT * FROM station_jobs ORDER BY requested_at').fetchall()]

    def run_jobs(self):
        for station_id, kind in self.jobs():
            with self.con:
                self.con.execute('DELETE FROM station_jobs WHERE station_id=? AND kind=?', (station_id, kind))
            if not self.settings.get(station_id).enabled:
                continue
            try:
                if kind == 'sync':
                    self.sync({station_id})
                elif kind == 'publish':
                    log.info('Station %s: published %d recorded intros', station_id, self.publish_planned(station_id))
                else:
                    log.info('Station %s: re-published %d media', station_id, self.republish(station_id))
            except Exception:
                log.exception('Station %s: %s failed', station_id, kind)

    def retry(self, station_id, media_id=None):
        """Requeue failed songs, or one song in any state, with fresh attempts."""
        query, params = "UPDATE media SET status='pending', attempts=0, error=NULL WHERE station_id=?", [station_id]
        if media_id is not None:
            query += ' AND media_id=?'
            params.append(media_id)
        else:
            query += " AND status='failed'"
        with self.con:
            return self.con.execute(query, params).rowcount

    def recovery(self, account):
        """Message of a stopped worker that waits for an operator decision, or None."""
        return self.instance.marker(f'recovery:{account}')

    def reusable(self, sha256):
        """An identical file was analyzed before, for any station."""
        return self.con.execute('SELECT 1 FROM analyses WHERE sha256=?', (sha256,)).fetchone() is not None

    def blocked(self, settings):
        """Why the station's songs cannot start a GPU batch now, or None."""
        blocked = self.gpus.blocked(settings.id)
        if blocked is None and self.clock() < self.retry_at.get(self.gpus.account(settings.id), 0.):
            return 'waiting for GPU retry'
        return blocked

    def _candidates(self, stations, forced):
        """Pending rows in queue order with whether their station may use the GPU now.

        Songs of a station that cannot use the GPU are downloaded once to learn their SHA256;
        afterwards they wait unless a stored analysis can be reused.
        """
        if not stations:
            return
        gpu = {i: self.blocked(s) is None for i, s in stations.items()}
        accounts = {i: self.gpus.account(i) for i in stations}
        query = (f"SELECT * FROM media WHERE status='pending' AND station_id IN ({','.join('?' * len(stations))}) "
                 f"AND (intro_end IS NULL OR station_id IN ({','.join('?' * len(forced)) or 'NULL'})) "
                 'ORDER BY priority, discovered_at, station_id, media_id')
        for row in self.con.execute(query, (*stations, *forced)).fetchall():
            station_id = row['station_id']
            if gpu[station_id] or row['sha256'] is None or self.reusable(row['sha256']):
                yield row, accounts[station_id], gpu[station_id]

    def _eligible(self, force=False, force_publish=False):
        requests = self.requests()
        stations = {s.id: s for s in self.settings.enabled() if force or s.auto_process or s.id in requests}
        forced = [i for i in stations if force_publish or requests.get(i)]
        return stations, forced, requests

    def ready(self, force=False, force_publish=False):
        """(rows, account): songs for one GPU account, plus songs of others that need no GPU yet."""
        stations, forced, requests = self._eligible(force, force_publish)
        account, rows = None, []
        for row, station_account, gpu in self._candidates(stations, forced):
            if gpu and account is None:
                account = station_account
            if not gpu or station_account == account or row['sha256'] is None or self.reusable(row['sha256']):
                rows.append(row)
                if len(rows) >= self.cfg.batch_max_songs:
                    break
        quiet = not force and not any(r['station_id'] in requests for r in rows)
        if (quiet and rows and len(rows) < self.cfg.batch_max_songs and all(r['priority'] == 0 for r in rows)
                and self.clock() - max(r['discovered_at'] for r in rows) < self.cfg.batch_quiet_seconds):
            return [], account
        return rows, account

    def process(self, force=False, force_publish=False):
        rows, account = self.ready(force, force_publish)
        if not rows:
            return 'idle'
        forced = {i for i, f in self.requests().items() if f}
        batch = self.cfg.data_dir / 'work' / uuid.uuid4().hex
        songs = batch / 'songs'
        waiting = {}
        self._begin_activity(account, rows)
        try:
            for index, media in enumerate(rows, 1):
                publish_forced = force_publish or media['station_id'] in forced
                suffix = PurePosixPath(media['path']).suffix.lower()
                try:
                    path, sha256 = self.client.download(media['station_id'], media['media_id'], songs, suffix)
                except AzuraCastError as error:
                    self._failed(media, error)
                    continue
                finally:
                    self._progress('download', index, len(rows))
                with self.con:
                    self.con.execute('UPDATE media SET sha256=?, updated_at=? WHERE station_id=? AND media_id=?',
                                     (sha256, self.clock(), media['station_id'], media['media_id']))
                if self.reusable(sha256):
                    self._finish(media, sha256, publish_forced)
                elif account is not None and self.gpus.account(media['station_id']) == account:
                    waiting.setdefault(sha256, []).append((media, publish_forced))
            for path in songs.glob('*'):
                if path.name.split('.')[0] not in waiting:
                    path.unlink()
            if waiting:
                self._analyze(account, waiting, songs, batch / 'reports')
            return f'processed {len(rows)}'
        finally:
            self.instance.set_marker(ACTIVITY, None)
            shutil.rmtree(batch, ignore_errors=True)

    def _finish(self, media, sha256, force_publish=False):
        try:
            self.finalize(media, sha256, force_publish=force_publish)
        except AzuraCastError as error:
            self._failed(media, f'publish: {error}')
        except Exception as error:
            log.exception('Decision failed for %s', sha256)
            self._failed(media, f'decision: {error!r}')

    def _record_run(self, run_id, account, waiting):
        per_station = Counter(media['station_id'] for items in waiting.values() for media, _ in items)
        with self.con:
            self.con.execute('INSERT INTO runs(id, songs, status, started_at, account) VALUES(?,?,?,?,?)',
                             (run_id, len(waiting), 'running', self.clock(), account))
            self.con.executemany('INSERT INTO run_stations(run_id, station_id, songs) VALUES(?,?,?)',
                                 [(run_id, station_id, count) for station_id, count in per_station.items()])

    def _analyze(self, account, waiting, songs, reports):
        run_id = reports.parent.name
        self._record_run(run_id, account, waiting)
        per_station = Counter(media['station_id'] for items in waiting.values() for media, _ in items)
        self._progress('worker', songs=sum(per_station.values()),
                       stations={str(k): v for k, v in per_station.items()})
        worker = None
        try:
            worker = self.gpus.get(account)
            worker.run(songs, reports, self._progress)
            report, music = evidence.load_reports(reports / 'report.json', reports / 'music-report.json')
        except (GpuError, OSError, ValueError, BlockingIOError) as error:
            with self.con:
                self.con.execute('UPDATE runs SET status=?, error=?, finished_at=?, pod_id=?, phase=? WHERE id=?',
                                 ('failed', str(error)[:2000], self.clock(),
                                  getattr(worker, 'last_pod_id', None), (self.activity() or {}).get('step'), run_id))
            log.error('GPU batch %s (%s) failed: %s', run_id, account, error)
            self.retry_at[account] = self.clock() + RETRY_SECONDS
            if isinstance(error, WorkerNeedsRecovery):
                self.instance.set_marker(f'recovery:{account}', str(error))
            # Capacity, budget, a busy worker or a failed copy are not the songs' fault.
            if isinstance(error, TRANSIENT):
                return
            for items in waiting.values():
                for media, _ in items:
                    self._failed(media, f'GPU batch: {error}')
            return
        complete, failed = evidence.split(report, music, set(waiting))
        self.instance.set_marker(f'recovery:{account}', None)
        with self.con:
            for sha256, (row, tune) in complete.items():
                self.con.execute('INSERT OR REPLACE INTO analyses(sha256, source, run_id, report, music, created_at) '
                                 'VALUES(?,?,?,?,?,?)', (sha256, 'pod', run_id, pack(row), pack(tune), self.clock()))

            self.con.execute('UPDATE runs SET status=?, finished_at=?, error=?, pod_id=? WHERE id=?',
                             ('complete' if not failed else 'partial', self.clock(),
                              json.dumps(failed)[:2000] if failed else None,
                              getattr(worker, 'last_pod_id', None), run_id))
        saved, total = 0, sum(len(items) for items in waiting.values())
        self._progress('save', 0, total)
        for sha256, items in waiting.items():
            for media, publish_forced in items:
                if sha256 in complete:
                    self._finish(media, sha256, publish_forced)
                else:
                    self._failed(media, 'incomplete evidence: ' + '; '.join(failed[sha256])[:1500])
                saved += 1
                self._progress('save', saved, total)

    def tick(self):
        """One queue step; a station's request ends once none of its songs can make progress."""
        result = self.process()
        stations, forced, requests = self._eligible()
        done = [i for i in requests
                if i not in stations or next(self._candidates({i: stations[i]}, [i] if i in forced else []), None) is None]
        if done:
            with self.con:
                self.con.executemany('DELETE FROM process_requests WHERE station_id=?', [(i,) for i in done])
            log.info('Requested processing finished for stations %s', done)
        return result

    def run_forever(self, tick=15):
        last_sync = 0.
        # A batch interrupted by a restart left its progress behind.
        self.instance.set_marker(ACTIVITY, None)
        with self.con:
            self.con.execute("UPDATE runs SET status='interrupted', finished_at=? WHERE status='running'", (self.clock(),))
        while True:
            if self.clock() - last_sync >= self.cfg.poll_seconds:
                self.sync()
                last_sync = self.clock()
            try:
                self.run_jobs()
                self.tick()
            except Exception:
                log.exception('Processing failed')
            try:
                self.gpus.idle_tick()
            except Exception:
                log.exception('Idle guard failed')
            self.sleep(tick)
