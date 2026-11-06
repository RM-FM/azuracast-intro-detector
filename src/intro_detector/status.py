"""Station and instance summaries for the management API; read-only."""
from importlib import metadata

from .gpu import account_spend, station_spend
from .stations import SHARED

RECENT_RUNS = 10
RECENT_FAILURES = 50


def version():
    try:
        return metadata.version('azuracast-intro-detector')
    except metadata.PackageNotFoundError:
        return 'unknown'


def _counts(con, query, args):
    return {row[0]: row[1] for row in con.execute(query, args)}


def total_spend(con, account=None):
    """Estimated spend over all time, from Pod lifetimes at their hourly price."""
    query, args = 'SELECT price, created_at, ended_at FROM pods WHERE ended_at IS NOT NULL', ()
    if account is not None:
        query, args = query + ' AND account=?', (account,)
    return sum(r['price'] * max(0., r['ended_at'] - r['created_at']) / 3600 for r in con.execute(query, args))


def cost_per_thousand(con):
    """Estimated spend per 1,000 analyzed songs, or None before the first completed batch."""
    songs = con.execute("SELECT COALESCE(SUM(songs), 0) FROM runs WHERE status IN ('complete', 'partial')").fetchone()[0]
    return round(total_spend(con) / songs * 1000, 2) if songs else None


def worker(con, account):
    row = con.execute('SELECT pod_id, gpu_id, price, created_at, idle_since, copied FROM pods '
                      'WHERE account=? AND ended_at IS NULL ORDER BY created_at DESC LIMIT 1', (account,)).fetchone()
    return dict(row) if row else None


def station(service, station_id):
    con, now = service.con, service.clock()
    settings = service.settings.get(station_id)
    account = settings.account
    if account == SHARED:
        spend, limit = station_spend(con, SHARED, station_id, now), settings.shared_cap_usd
    else:
        spend, limit = account_spend(con, account, now), settings.budget_usd
    runs = [dict(r) for r in con.execute(
        'SELECT r.id, r.status, s.songs, r.started_at, r.finished_at, SUBSTR(r.error, 1, 300) AS error '
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
        library_without_intro=con.execute("SELECT COUNT(*) FROM media WHERE station_id=? AND status='known' "
                                          'AND intro_end IS NULL', (station_id,)).fetchone()[0],
        process_requested=station_id in service.requests(),
        jobs=[kind for sid, kind in service.jobs() if sid == station_id],
        blocked=service.blocked(settings) if settings.enabled else None,
        key=service.gpus.secrets.has(account),
        shared_key_available=settings.shared_allowed and service.gpus.secrets.has(SHARED),
        spend_month_usd=round(spend, 4),
        limit_usd=limit,
        worker=worker(con, account),
        recovery=service.recovery(account),
        runs=runs,
        failures=failures)


def instance(service):
    con, now = service.con, service.clock()
    settings = service.instance.get()
    stations = []
    for item in service.settings.all():
        stations.append(dict(id=item.id, enabled=item.enabled, billing=item.billing,
                             shared_allowed=item.shared_allowed, shared_cap_usd=item.shared_cap_usd,
                             shared_spend_month_usd=round(station_spend(con, SHARED, item.id, now), 4)))
    return dict(
        version=version(),
        settings=settings.__dict__,
        shared_key=service.gpus.secrets.has(SHARED),
        shared_spend_month_usd=round(account_spend(con, SHARED, now), 4),
        shared_worker=worker(con, SHARED),
        shared_recovery=service.recovery(SHARED),
        stations=stations,
        analyses=con.execute('SELECT COUNT(*) FROM analyses').fetchone()[0],
        cost_per_1000_songs_usd=cost_per_thousand(con))
