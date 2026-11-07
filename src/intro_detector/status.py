"""Station summaries for the management API; read-only."""
from importlib import metadata

from .gpu import account_spend, station_spend

RECENT_RUNS = 10
RECENT_FAILURES = 50


def version():
    try:
        return metadata.version('azuracast-intro-detector')
    except metadata.PackageNotFoundError:
        return 'unknown'


def _counts(con, query, args):
    return {row[0]: row[1] for row in con.execute(query, args)}


def total_spend(con):
    """Estimated spend over all time, from Pod lifetimes at their hourly price."""
    return sum(r['price'] * max(0., r['ended_at'] - r['created_at']) / 3600
               for r in con.execute('SELECT price, created_at, ended_at FROM pods WHERE ended_at IS NOT NULL'))


def cost_per_thousand(con):
    """Estimated spend per 1,000 analyzed songs, or None before the first completed batch."""
    songs = con.execute("SELECT COALESCE(SUM(songs), 0) FROM runs WHERE status IN ('complete', 'partial')").fetchone()[0]
    return round(total_spend(con) / songs * 1000, 2) if songs else None


def estimate(con, songs):
    """Estimated GPU cost of analyzing songs, or None before the first completed batch."""
    per_thousand = cost_per_thousand(con)
    return None if per_thousand is None else round(songs * per_thousand / 1000, 2)


def worker(con, account, idle_seconds=None):
    """The account's running worker; shutdown_at is when an idle worker is retired, None while it works."""
    if account is None:
        return None
    row = con.execute('SELECT pod_id, gpu_id, price, created_at, idle_since, copied FROM pods '
                      'WHERE account=? AND ended_at IS NULL ORDER BY created_at DESC LIMIT 1', (account,)).fetchone()
    if row is None:
        return None
    shutdown = row['idle_since'] + idle_seconds if row['idle_since'] is not None and idle_seconds else None
    return dict(row, shutdown_at=shutdown)


def key_summary(service, account, now):
    """The station's Runpod key as all stations using it see it; never the key itself."""
    balance = service.gpus.balance(account) or {}
    return dict(stations=service.gpus.key_stations(account),
                spend_month_usd=round(account_spend(service.con, account, now), 4),
                balance_usd=balance.get('balance_usd'), balance_checked_at=balance.get('checked_at'))

def activity(service, station_id):
    """The running batch as this station may see it; another station's batch shows only its progress."""
    current = service.activity()
    if not current:
        return None
    view = {key: current.get(key) for key in ('step', 'done', 'total', 'started_at', 'step_started_at', 'detail')}
    songs = (current.get('stations') or {}).get(str(station_id))
    if songs:
        return dict(view, own=True, songs=current.get('songs'), station_songs=songs)
    return dict(view, own=False)


def station(service, station_id):
    con, now = service.con, service.clock()
    settings = service.settings.get(station_id)
    account = service.gpus.account(station_id)
    library = con.execute("SELECT COUNT(*) FROM media WHERE station_id=? AND status='known' AND intro_end IS NULL",
                          (station_id,)).fetchone()[0]
    runs = [dict(r) for r in con.execute(
        'SELECT r.id, r.status, s.songs, r.started_at, r.finished_at, r.phase, SUBSTR(r.error, 1, 300) AS error '
        'FROM runs r JOIN run_stations s ON s.run_id = r.id WHERE s.station_id=? '
        'ORDER BY r.started_at DESC LIMIT ?', (station_id, RECENT_RUNS))]
    failures = [dict(r) for r in con.execute(
        "SELECT media_id, path, attempts, SUBSTR(error, 1, 300) AS error, updated_at FROM media "
        "WHERE station_id=? AND status='failed' ORDER BY updated_at DESC LIMIT ?", (station_id, RECENT_FAILURES))]
    return dict(
        settings=settings.public(),
        queue=_counts(con, 'SELECT status, COUNT(*) FROM media WHERE station_id=? GROUP BY status', (station_id,)),
        publications=_counts(con, 'SELECT state, COUNT(*) FROM publications WHERE station_id=? GROUP BY state',
                             (station_id,)),
        library_without_intro=library,
        library_estimate_usd=estimate(con, library),
        process_requested=station_id in service.requests(),
        jobs=[kind for sid, kind in service.jobs() if sid == station_id],
        blocked=service.blocked(settings) if settings.enabled else None,
        key=None if account is None else key_summary(service, account, now),
        spend_month_usd=round(station_spend(con, station_id, now), 4),
        limit_usd=settings.budget_usd,
        worker=worker(con, account, service.cfg.runpod.idle_seconds),
        recovery=service.recovery(account) if account else None,
        activity=activity(service, station_id),
        now=now,
        runs=runs,
        failures=failures)

