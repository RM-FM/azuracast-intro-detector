"""Command line entry point."""
import argparse
import contextlib
import csv
import fcntl
import json
import logging
import os
import signal
import sqlite3
import sys
import time

from . import config as config_module
from .azuracast import AzuraCastError, Client
from .db import audit, connect
from .gpu import GpuError, Workers, account_spend
from .library import backfill
from .service import Service
from .stations import SHARED, Instance, Secrets, Stations, migrate

REVIEW_KINDS = {'instrumental': 'instrumental', 'clear-opening': 'clear_opening'}
log = logging.getLogger(__name__)


def build(cfg, environ=None):
    """Service with its stores; migrates 0.x settings on first use."""
    con = connect(cfg.db_path)
    client = Client(cfg.base_url, cfg.api_key, cfg.timeout_seconds)
    stations, instance, secrets = Stations(con, cfg, time.time), Instance(con, cfg), Secrets(cfg.data_dir)
    if migrate(con, cfg, stations, instance, secrets, os.environ if environ is None else environ):
        log.info('Migrated configured stations and the Runpod key to managed station settings')
    return Service(cfg, con, client, Workers(cfg, con, stations, instance, secrets))


def _service(args):
    return build(config_module.load(args.config))


@contextlib.contextmanager
def _owner(cfg):
    """Only one process may work the queue of a data directory."""
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    with (cfg.data_dir / '.service.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('Another intro-detector run/process is already working this queue '
                             '(use "status" to inspect it)') from None
        yield


def _sha_for(con, args):
    if args.sha256:
        return args.sha256.lower()
    row = con.execute('SELECT sha256 FROM media WHERE station_id=? AND media_id=?', (args.station, args.media)).fetchone()
    if row is None or not row['sha256']:
        raise SystemExit('Media is unknown or not downloaded yet; pass --sha256')
    return row['sha256']


def _redecide(service, sha256):
    media = service.con.execute("SELECT * FROM media WHERE sha256=? AND status='done'", (sha256,)).fetchall()
    return sum(service.finalize(m, sha256) for m in media)


def cmd_run(args):
    service = _service(args)
    # PID 1 in the container ignores SIGTERM by default; exit at once so "docker stop" is clean.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    with _owner(service.cfg):
        service.run_forever()


def cmd_health(args):
    try:
        cfg = config_module.load(args.config)
        with (cfg.data_dir / '.service.lock').open('rb') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                pass
            else:
                raise SystemExit('Unhealthy: no running service owns the queue')
            with contextlib.closing(sqlite3.connect(
                    cfg.db_path.resolve().as_uri() + '?mode=ro', uri=True, timeout=1)) as database:
                for table in ('stations', 'media', 'analyses', 'runs'):
                    database.execute(f'SELECT 1 FROM {table} LIMIT 0')
    except (OSError, ValueError, sqlite3.Error):
        raise SystemExit('Unhealthy: configuration or service database is unavailable') from None


def cmd_sync(args):
    _service(args).sync()


def cmd_process(args):
    service = _service(args)
    if args.station is not None and not service.settings.get(args.station).enabled:
        raise SystemExit(f'Station {args.station} is not enabled')
    service.cfg.data_dir.mkdir(parents=True, exist_ok=True)
    with (service.cfg.data_dir / '.service.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            # The running service owns the queue: hand it the request instead.
            count = service.request(args.station, args.force)
            print(f'Requested for {count} station(s); the running service processes them '
                  '(watch "docker compose logs -f detector")')
            return
        service.sync(None if args.station is None else {args.station})
        service.request(args.station, args.force)
        print(service.process())


def cmd_status(args):
    service = _service(args)
    con = service.con
    requests = service.requests()
    for s in service.settings.all():
        state = 'enabled' if s.enabled else 'disabled'
        print(f'station {s.id:>3}  {state:<8} {s.mode:<7} {"auto" if s.auto_process else "manual":<6} '
              f'pays: {s.account}' + (' (process requested)' if s.id in requests else '')
              + (f'  blocked: {service.blocked(s)}' if s.enabled and service.blocked(s) else ''))
    for row in con.execute('SELECT station_id, status, COUNT(*) n FROM media GROUP BY station_id, status ORDER BY 1, 2'):
        print(f'station {row["station_id"]:>3}  {row["status"]:<9} {row["n"]}')
    for row in con.execute('SELECT station_id, state, COUNT(*) n FROM publications GROUP BY 1, 2 ORDER BY 1, 2'):
        print(f'station {row["station_id"]:>3}  publication {row["state"]:<10} {row["n"]}')
    print(f'analyses: {con.execute("SELECT COUNT(*) FROM analyses").fetchone()[0]}')
    for row in con.execute('SELECT * FROM runs ORDER BY started_at DESC LIMIT 5'):
        print(f'run {row["id"]} {row["account"]} {row["status"]} songs={row["songs"]} '
              f'{time.strftime("%Y-%m-%d %H:%M", time.gmtime(row["started_at"]))} {row["error"] or ""}'[:300])
    # Observed wall time including Pod start; use it to calibrate runpod.gpus speeds.
    for row in con.execute("SELECT p.gpu_id, COUNT(*) runs, SUM(r.songs) songs, SUM(r.finished_at - r.started_at) seconds "
                           "FROM runs r JOIN pods p ON p.pod_id = r.pod_id WHERE r.status IN ('complete', 'partial') "
                           'GROUP BY p.gpu_id'):
        print(f'{row["gpu_id"] or "unknown GPU"}: {row["runs"]} batches, {row["songs"]} songs, '
              f'{row["seconds"] / max(row["songs"], 1):.1f} s/song')
    accounts = set(service.gpus.secrets.accounts()) | {r[0] for r in con.execute('SELECT DISTINCT account FROM pods')}
    for account in sorted(accounts):
        budget = service.gpus.settings(account).monthly_budget_usd
        print(f'account {account}: estimated GPU spend this month {account_spend(con, account, time.time()):.2f} USD '
              f'(limit {budget or "none"}){"" if service.gpus.secrets.has(account) else ", no key"}')


def cmd_ab_select(args):
    from . import ab
    service = _service(args)
    quotas = None
    if args.quota:
        quotas = {}
        for item in args.quota:
            code, _, n = item.partition('=')
            if not code or not n.isdigit():
                raise ValueError(f'--quota expects language=count, got {item!r}')
            quotas[code] = int(n)
    ab.select(service.con, service.cfg.data_dir, args.name, args.count, args.controls, args.seed, quotas)


def cmd_ab_run(args):
    from . import ab
    print(json.dumps(ab.run(_service(args), args.name, args.max_usd), indent=2))


def cmd_ab_report(args):
    from . import ab
    ab.report(_service(args), args.name)


def cmd_decision_diff(args):
    """Dry run: published values that the current decision code and settings would change."""
    from .db import unpack
    service = _service(args)
    con = service.con
    query, params = "SELECT * FROM media WHERE status='done' AND sha256 IS NOT NULL", []
    if args.station:
        query, params = query + ' AND station_id=?', [args.station]
    songs = {}
    for row in con.execute(query + ' ORDER BY station_id, media_id', params):
        songs.setdefault(row['sha256'], []).append(row)
    scanned = affected = 0
    output = service.cfg.data_dir / 'reports' / f'decision-diff-{time.strftime("%Y%m%dT%H%M%S")}.tsv'
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream, delimiter='\t')
        writer.writerow(['station', 'media', 'path', 'stored_value', 'new_value', 'new_type'])
        for sha, media in songs.items():
            analysis = con.execute('SELECT report, music FROM analyses WHERE sha256=?', (sha,)).fetchone()
            if analysis is None:
                continue
            scanned += 1
            stored = con.execute('SELECT value FROM decisions WHERE sha256=? ORDER BY created_at DESC LIMIT 1',
                                 (sha,)).fetchone()
            _, marker, value = service.evaluate(unpack(analysis['report']), unpack(analysis['music']), sha)
            old = stored['value'] if stored else None
            if (old is None) == (value is None) and (old is None or abs(old - value) < .0005):
                continue
            affected += 1
            for item in media:
                writer.writerow([item['station_id'], item['media_id'], item['path'], old, value,
                                 marker.get('intro_type') or marker.get('intro_reason')])
    print(f'{scanned} analyzed songs checked, {affected} decisions would change -> {output}')


def cmd_backfill(args):
    service = _service(args)
    if not service.settings.get(args.station).enabled:
        raise SystemExit(f'Station {args.station} is not enabled')
    count = backfill(service.con, args.station, args.limit, args.include_set, args.dry_run)
    print(f'{"Would queue" if args.dry_run else "Queued"} {count} songs for station {args.station}')


def cmd_retry(args):
    con = _service(args).con
    query, params = "UPDATE media SET status='pending', attempts=0, error=NULL WHERE station_id=?", [args.station]
    if args.media:
        query += ' AND media_id=?'
        params.append(args.media)
    else:
        query += " AND status='failed'"
    with con:
        print(f'Requeued {con.execute(query, params).rowcount} songs')


def cmd_decisions(args):
    con = _service(args).con
    writer = csv.writer(sys.stdout, delimiter='\t', lineterminator='\n')
    writer.writerow(['media_id', 'path', 'status', 'type', 'reason', 'value', 'azuracast', 'publication', 'error'])
    for row in con.execute(
            'SELECT m.*, d.marker, d.value, p.state FROM media m '
            'LEFT JOIN decisions d ON d.sha256=m.sha256 '
            'LEFT JOIN publications p ON p.station_id=m.station_id AND p.media_id=m.media_id '
            'WHERE m.station_id=? ORDER BY m.path', (args.station,)):
        marker = json.loads(row['marker']) if row['marker'] else {}
        writer.writerow([row['media_id'], row['path'], row['status'], marker.get('intro_type') or '',
                         marker.get('intro_reason') or '', '' if row['value'] is None else row['value'],
                         row['intro_end'] or '', row['state'] or '', row['error'] or ''])


def cmd_review(args):
    service = _service(args)
    con = service.con
    if args.action == 'list':
        for row in con.execute('SELECT * FROM reviews ORDER BY created_at'):
            print(f'{row["sha256"]} {row["kind"]} {row["note"] or ""}')
        return
    sha256, kind = _sha_for(con, args), REVIEW_KINDS[args.kind]
    with con:
        if args.action == 'add':
            con.execute('INSERT OR REPLACE INTO reviews(sha256, kind, note, created_at) VALUES(?,?,?,?)',
                        (sha256, kind, args.note, time.time()))
        else:
            con.execute('DELETE FROM reviews WHERE sha256=? AND kind=?', (sha256, kind))
        audit(con, f'review_{args.action}', sha256=sha256, after=kind, detail=args.note)
    print(f'Review {args.action}: {kind} for {sha256}; re-published {_redecide(service, sha256)} media')


def cmd_unlock(args):
    con = _service(args).con
    with con:
        changed = con.execute("DELETE FROM publications WHERE station_id=? AND media_id=? AND state IN ('manual', 'matching')",
                              (args.station, args.media)).rowcount
        audit(con, 'unlock', args.station, args.media)
    print('Unlocked; the next decision may overwrite AzuraCast' if changed else 'Nothing to unlock')


def cmd_republish(args):
    print(f'Re-published {_service(args).republish(args.station, args.media, force=args.force)} media')


def cmd_pod(args):
    service = _service(args)
    gpu = service.gpus.get(args.account)
    pod = gpu.worker()
    if args.action == 'status' or pod is None:
        print(json.dumps(pod, indent=2) if pod else 'No worker Pod')
        return
    row = gpu._row(pod['id'])
    if args.action == 'delete' and not (row and row['copied']) and not args.force:
        raise SystemExit('Reports from this Pod were not confirmed as copied; use --force after recovery')
    gpu.retire(pod['id'], delete=args.action == 'delete')


def _assignments(items):
    values = {}
    for item in items:
        name, separator, raw = item.partition('=')
        if not separator:
            raise ValueError(f'Expected name=value, got {item!r}')
        try:
            values[name] = json.loads(raw)
        except ValueError:
            values[name] = raw
    return values


def cmd_station(args):
    service = _service(args)
    if args.action == 'set':
        if args.id is None:
            raise SystemExit('station set needs a station ID')
        service.settings.update(args.id, _assignments(args.values), admin=True)
    for settings in ([service.settings.get(args.id)] if args.id is not None else service.settings.all()):
        print(json.dumps(settings.public()))


def cmd_instance(args):
    instance = _service(args).instance
    if args.values:
        instance.update(_assignments(args.values))
    print(json.dumps(instance.get().__dict__))


def cmd_key(args):
    secrets = Secrets(config_module.load(args.config).data_dir)
    if args.action == 'set':
        secrets.set(args.account, sys.stdin.readline().strip())
        print(f'Runpod key stored for {args.account}')
    elif args.action == 'clear':
        secrets.set(args.account, None)
        print(f'Runpod key removed for {args.account}')
    else:
        print('\n'.join(secrets.accounts()) or 'No Runpod keys')


def cmd_check(args):
    """Verify configuration, AzuraCast access per enabled station and each Runpod key."""
    service = _service(args)
    cfg = service.cfg
    try:
        names = service.client.station_names()
    except AzuraCastError as error:
        raise SystemExit(f'AzuraCast: connection or API key failed: {error}') from None
    print(f'AzuraCast: connected to {cfg.base_url}; the API key sees {len(names)} stations')
    problems = []
    enabled = service.settings.enabled()
    for settings in enabled:
        if settings.id not in names:
            problems.append(f'Station {settings.id} is not visible: give the API user View permission on it')
            continue
        try:
            service.client.can_read_media(settings.id)
        except AzuraCastError:
            problems.append(f'Station {settings.id} ({names[settings.id]}): give the API user Media permission')
            continue
        print(f'  station {settings.id:>3}  {settings.mode:<8} {names[settings.id]}')
    if not enabled:
        print('No station enabled yet; enable stations in Sweepers > Intro Detector')
    secrets = service.gpus.secrets
    for account in secrets.accounts():
        reply = service.gpus.get(account).call(['runpodctl', 'pod', 'list', '--all'], check=False)
        if reply.returncode:
            problems.append(f'Runpod rejected the {account} key; check the key and its permissions')
        else:
            print(f'Runpod: {account} key accepted')
    if not secrets.accounts():
        print('No Runpod key yet; add one in Sweepers > Intro Detector')
    if not os.access(cfg.runpod.ssh_key, os.R_OK):
        problems.append(f'SSH key {cfg.runpod.ssh_key} is missing or unreadable')
    for problem in problems:
        print(f'PROBLEM: {problem}')
    if problems:
        raise SystemExit(1)
    print('All checks passed.')


def main(argv=None):
    parser = argparse.ArgumentParser(prog='intro-detector', description=__doc__)
    parser.add_argument('--config', help='TOML file (default $INTRO_DETECTOR_CONFIG or /config/config.toml)')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('run', help='Run the service loop').set_defaults(func=cmd_run)
    sub.add_parser('health', help='Check local service ownership and database readiness'
                   ).set_defaults(func=cmd_health)
    sub.add_parser('sync', help='Sync AzuraCast libraries once').set_defaults(func=cmd_sync)
    p = sub.add_parser('process', help='Process pending songs now (asks the running service if there is one)')
    p.add_argument('--station', type=int, help='Only this station')
    p.add_argument('--force', action='store_true',
                   help='Include songs with an intro and allow replacing it (only when station mode is publish)')
    p.set_defaults(func=cmd_process)
    sub.add_parser('status', help='Stations, queue, runs and GPU spend').set_defaults(func=cmd_status)
    sub.add_parser('check', help='Check configuration, AzuraCast permissions and Runpod keys'
                   ).set_defaults(func=cmd_check)
    p = sub.add_parser('station', help='Show or change station settings (also those only administrators may set)')
    p.add_argument('action', choices=('show', 'set'))
    p.add_argument('id', type=int, nargs='?')
    p.add_argument('values', nargs='*', metavar='NAME=VALUE', help='e.g. enabled=true mode=publish billing=shared')
    p.set_defaults(func=cmd_station)
    p = sub.add_parser('instance', help='Show or change instance settings')
    p.add_argument('values', nargs='*', metavar='NAME=VALUE', help='e.g. max_price_per_hour=0.9')
    p.set_defaults(func=cmd_instance)
    p = sub.add_parser('key', help='Runpod API keys per paying account (set reads the key from stdin)')
    p.add_argument('action', choices=('list', 'set', 'clear'))
    p.add_argument('account', nargs='?', default=SHARED, help='shared (default) or station-ID')
    p.set_defaults(func=cmd_key)
    p = sub.add_parser('ab-select', help='Choose non-English recordings plus English controls for an A/B test')
    p.add_argument('--name', required=True)
    p.add_argument('--count', type=int, default=80, help='Non-English recordings (default 80)')
    p.add_argument('--controls', type=int, default=10, help='English control recordings (default 10)')
    p.add_argument('--seed', type=int, default=27)
    p.add_argument('--quota', action='append', metavar='LANG=N',
                   help="Fixed mix instead of round-robin, e.g. --quota de=30 --quota other=5")
    p.set_defaults(func=cmd_ab_select)
    p = sub.add_parser('ab-run', help='Analyze an A/B set with the current pod/ (GPU); never publishes')
    p.add_argument('--name', required=True)
    p.add_argument('--max-usd', type=float,
                   help='Hard cost cap for this run (allows it beyond the monthly budget and limits Pod time)')
    p.set_defaults(func=cmd_ab_run)
    p = sub.add_parser('ab-report', help='Compare stored (A) and new (B) decisions of an A/B set')
    p.add_argument('--name', required=True)
    p.set_defaults(func=cmd_ab_report)
    p = sub.add_parser('decision-diff', help='Dry run: published values the current decision settings would change')
    p.add_argument('--station', type=int, help='Only this station')
    p.set_defaults(func=cmd_decision_diff)
    p = sub.add_parser('backfill', help='Queue existing library songs')
    p.add_argument('--station', type=int, required=True)
    p.add_argument('--limit', type=int)
    p.add_argument('--include-set', action='store_true', help='Also songs whose intro_end is already set')
    p.add_argument('--dry-run', action='store_true')
    p.set_defaults(func=cmd_backfill)
    p = sub.add_parser('retry', help='Requeue failed songs (or one media)')
    p.add_argument('--station', type=int, required=True)
    p.add_argument('--media', type=int)
    p.set_defaults(func=cmd_retry)
    p = sub.add_parser('decisions', help='TSV of decisions and publication state')
    p.add_argument('--station', type=int, required=True)
    p.set_defaults(func=cmd_decisions)
    p = sub.add_parser('review', help='Listener reviews bound to the audio SHA256')
    p.add_argument('action', choices=('add', 'remove', 'list'))
    p.add_argument('--kind', choices=sorted(REVIEW_KINDS))
    p.add_argument('--sha256')
    p.add_argument('--station', type=int)
    p.add_argument('--media', type=int)
    p.add_argument('--note')
    p.set_defaults(func=cmd_review)
    p = sub.add_parser('unlock', help='Allow overwriting a manually set intro_end')
    p.add_argument('--station', type=int, required=True)
    p.add_argument('--media', type=int, required=True)
    p.set_defaults(func=cmd_unlock)
    p = sub.add_parser('republish', help='Re-decide and publish analyzed songs (e.g. after shadow -> publish)')
    p.add_argument('--station', type=int, required=True)
    p.add_argument('--media', type=int)
    p.add_argument('--force', action='store_true',
                   help='Allow replacing an existing intro_end (only when station mode is publish)')
    p.set_defaults(func=cmd_republish)
    p = sub.add_parser('pod', help='Inspect or retire the GPU worker of a paying account')
    p.add_argument('action', choices=('status', 'stop', 'delete'))
    p.add_argument('--account', default=SHARED, help='shared (default) or station-ID')
    p.add_argument('--force', action='store_true')
    p.set_defaults(func=cmd_pod)
    args = parser.parse_args(argv)
    if args.command == 'review' and args.action != 'list' and (
            not args.kind or not (args.sha256 or (args.station and args.media))):
        parser.error('review add/remove needs --kind and --sha256 or --station/--media')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    try:
        args.func(args)
    except (GpuError, ValueError, OSError) as error:
        raise SystemExit(f'ERROR: {error}')
