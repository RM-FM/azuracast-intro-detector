"""Queue processing: download, analyze on the GPU worker, decide and publish."""
import json
import logging
from pathlib import PurePosixPath
import shutil
import time
import uuid

from . import evidence, publisher
from .azuracast import AzuraCastError
from .db import pack, unpack
from .gpu import BudgetExceeded, GpuError, KeepPodRunning, NoCapacity, PodBusy
from .library import sync_station

log = logging.getLogger(__name__)
TRANSIENT = (NoCapacity, BudgetExceeded, PodBusy, KeepPodRunning, BlockingIOError)
RETRY_SECONDS = 900
# Created by `intro-detector process` while the service runs; removed once the queue is empty.
PROCESS_REQUEST = '.process-request'


class Service:
    def __init__(self, cfg, con, client, gpu, selector_factory=None, clock=time.time, sleep=time.sleep):
        self.cfg, self.con, self.client, self.gpu = cfg, con, client, gpu
        self.clock, self.sleep = clock, sleep
        self._selector, self._selector_factory = None, selector_factory
        self.retry_at = 0.

    @property
    def selector(self):
        if self._selector is None:
            if self._selector_factory is None:
                from .selector import Selector
                self._selector = Selector(vocals=self.cfg.decision.vocals)
            else:
                self._selector = self._selector_factory()
        return self._selector

    def stations(self, discover=False):
        """Configured stations; with all_stations also those AzuraCast lists or that were synced before."""
        if not self.cfg.all_stations:
            return self.cfg.stations
        ids = {s.id for s in self.cfg.stations}
        ids |= {row[0] for row in self.con.execute('SELECT station_id FROM stations')}
        if discover:
            try:
                ids |= set(self.client.station_names())
            except AzuraCastError as error:
                log.error('Station discovery failed: %s', error)
        return tuple(self.cfg.station(station_id) for station_id in sorted(ids))

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

    def finalize(self, media, sha256):
        """Decide from stored evidence and publish to AzuraCast; True when the item is done."""
        decided = self.decide(sha256)
        if decided is None:
            return False
        marker, value = decided
        station = self.cfg.station(media['station_id'])
        state = publisher.publish(self.con, self.client, station, media['media_id'], sha256, value, self.clock())
        with self.con:
            self.con.execute("UPDATE media SET status='done', error=NULL, updated_at=? WHERE station_id=? AND media_id=?",
                             (self.clock(), media['station_id'], media['media_id']))
        log.info('Station %s media %s (%s): %s %s -> %s', media['station_id'], media['media_id'], media['path'],
                 marker.get('intro_type') or marker.get('intro_reason'), value, state)
        return True

    def republish(self, station_id, media_id=None):
        query = "SELECT * FROM media WHERE station_id=? AND status='done' AND sha256 IS NOT NULL"
        args = [station_id]
        if media_id is not None:
            query += ' AND media_id=?'
            args.append(media_id)
        return sum(self.finalize(m, m['sha256']) for m in self.con.execute(query, args).fetchall())

    def _failed(self, media, error):
        attempts = media['attempts'] + 1
        status = 'failed' if attempts >= self.cfg.max_attempts else 'pending'
        with self.con:
            self.con.execute('UPDATE media SET status=?, attempts=?, error=?, updated_at=? WHERE station_id=? AND media_id=?',
                             (status, attempts, str(error)[:2000], self.clock(), media['station_id'], media['media_id']))
        log.warning('Station %s media %s: %s (%s)', media['station_id'], media['media_id'], error, status)

    # -- queue -----------------------------------------------------------
    def sync(self):
        for station in self.stations(discover=True):
            if station.mode == 'off':
                continue
            try:
                counts = sync_station(self.con, self.client, station, self.clock())
                log.info('Station %s synced: %s', station.id, counts)
            except AzuraCastError as error:
                log.error('Station %s sync failed: %s', station.id, error)

    def ready(self, force=False):
        """Pending items, or [] while new uploads are still arriving (batch them) unless forced."""
        active = tuple(s.id for s in self.stations() if s.mode != 'off')
        if not active:
            return []
        rows = self.con.execute(
            f"SELECT * FROM media WHERE status='pending' AND station_id IN ({','.join('?' * len(active))}) "
            'ORDER BY priority, discovered_at, station_id, media_id LIMIT ?',
            (*active, self.cfg.batch_max_songs)).fetchall()
        if (not force and rows and len(rows) < self.cfg.batch_max_songs and all(r['priority'] == 0 for r in rows)
                and self.clock() - max(r['discovered_at'] for r in rows) < self.cfg.batch_quiet_seconds):
            return []
        return rows

    @property
    def request_path(self):
        return self.cfg.data_dir / PROCESS_REQUEST

    def process(self, force=False):
        if self.clock() < self.retry_at:
            return 'waiting for GPU retry'
        rows = self.ready(force)
        if not rows:
            return 'idle'
        batch = self.cfg.data_dir / 'work' / uuid.uuid4().hex
        songs = batch / 'songs'
        waiting = {}
        try:
            for media in rows:
                suffix = PurePosixPath(media['path']).suffix.lower()
                try:
                    path, sha256 = self.client.download(media['station_id'], media['media_id'], songs, suffix)
                except AzuraCastError as error:
                    self._failed(media, error)
                    continue
                with self.con:
                    self.con.execute('UPDATE media SET sha256=?, updated_at=? WHERE station_id=? AND media_id=?',
                                     (sha256, self.clock(), media['station_id'], media['media_id']))
                if self.con.execute('SELECT 1 FROM analyses WHERE sha256=?', (sha256,)).fetchone():
                    path.unlink(missing_ok=True)
                    self._finish(media, sha256)
                else:
                    waiting.setdefault(sha256, []).append(media)
            if waiting:
                self._analyze(waiting, songs, batch / 'reports')
            return f'processed {len(rows)}'
        finally:
            shutil.rmtree(batch, ignore_errors=True)

    def _finish(self, media, sha256):
        try:
            self.finalize(media, sha256)
        except AzuraCastError as error:
            self._failed(media, f'publish: {error}')
        except Exception as error:
            log.exception('Decision failed for %s', sha256)
            self._failed(media, f'decision: {error!r}')

    def _analyze(self, waiting, songs, reports):
        run_id = reports.parent.name
        started = self.clock()
        with self.con:
            self.con.execute('INSERT INTO runs(id, songs, status, started_at) VALUES(?,?,?,?)',
                             (run_id, len(waiting), 'running', started))
        try:
            self.gpu.run(songs, reports)
            report, music = evidence.load_reports(reports / 'report.json', reports / 'music-report.json')
        except (GpuError, OSError, ValueError, BlockingIOError) as error:
            with self.con:
                self.con.execute('UPDATE runs SET status=?, error=?, finished_at=?, pod_id=? WHERE id=?',
                                 ('failed', str(error)[:2000], self.clock(),
                                  getattr(self.gpu, 'last_pod_id', None), run_id))
            log.error('GPU batch %s failed: %s', run_id, error)
            self.retry_at = self.clock() + RETRY_SECONDS
            # Capacity, budget, a busy worker or a failed copy are not the songs' fault.
            if isinstance(error, TRANSIENT):
                return
            for items in waiting.values():
                for media in items:
                    self._failed(media, f'GPU batch: {error}')
            return
        complete, failed = evidence.split(report, music, set(waiting))
        with self.con:
            for sha256, (row, tune) in complete.items():
                self.con.execute('INSERT OR REPLACE INTO analyses(sha256, source, run_id, report, music, created_at) '
                                 'VALUES(?,?,?,?,?,?)', (sha256, 'pod', run_id, pack(row), pack(tune), self.clock()))
            self.con.execute('UPDATE runs SET status=?, finished_at=?, error=?, pod_id=? WHERE id=?',
                             ('complete' if not failed else 'partial', self.clock(),
                              json.dumps(failed)[:2000] if failed else None,
                              getattr(self.gpu, 'last_pod_id', None), run_id))
        for sha256, items in waiting.items():
            for media in items:
                if sha256 in complete:
                    self._finish(media, sha256)
                else:
                    self._failed(media, 'incomplete evidence: ' + '; '.join(failed[sha256])[:1500])

    def tick(self):
        """One queue step. Without auto_process, only a requested run works the queue, until empty."""
        requested = self.request_path.exists()
        if not (self.cfg.auto_process or requested):
            return 'manual processing; nothing requested'
        result = self.process(force=requested)
        if requested and result == 'idle':
            self.request_path.unlink(missing_ok=True)
            log.info('Requested processing finished: queue is empty')
        return result

    def run_forever(self, tick=15):
        last_sync = 0.
        if not self.cfg.auto_process:
            log.info('auto_process is off: new songs are queued; run "intro-detector process" to analyze them')
        while True:
            if self.clock() - last_sync >= self.cfg.poll_seconds:
                self.sync()
                last_sync = self.clock()
            try:
                self.tick()
            except Exception:
                log.exception('Processing failed')
            try:
                self.gpu.idle_tick()
            except Exception:
                log.exception('Idle guard failed')
            self.sleep(tick)
