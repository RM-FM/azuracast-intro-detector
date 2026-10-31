"""Map resolved intro markers to the Sweepers plugin's intro_end and publish safely."""
import math
import time

from .config import Decision
from .core import resolver
from .db import audit

POLICY = resolver.POLICY
TOLERANCE = 0.0005


def resolve(raw, report_row, music_row, reviews=(), decision=Decision()):
    return resolver.resolve(raw, report_row, music_row, reviewed='clear_opening' in reviews,
                       instrumental_review='instrumental' in reviews,
                       no_intro_below=decision.no_intro_below_seconds,
                       music_window=(decision.music_min_seconds, decision.music_max_seconds))


def target_value(marker):
    """Seconds to publish, 0 for vocals at the start, or None to leave AzuraCast untouched."""
    end = marker.get('intro_end')
    if marker.get('intro_status') == 'confirmed' and isinstance(end, (int, float)) and math.isfinite(end):
        return round(float(end), 3)
    if marker.get('intro_reason') == 'no_intro':
        return 0.0
    return None


def parse(raw):
    if raw is None or str(raw).strip() == '':
        return None
    try:
        value = float(raw)
    except ValueError:
        return math.nan
    return value if math.isfinite(value) else math.nan


def same(left, right):
    return left is not None and right is not None and abs(left - right) <= TOLERANCE


def _record(con, station_id, media_id, sha256, state, value, now):
    con.execute('INSERT INTO publications(station_id, media_id, sha256, policy, value, state, updated_at) '
                'VALUES(?,?,?,?,?,?,?) ON CONFLICT(station_id, media_id) DO UPDATE SET '
                'sha256=excluded.sha256, policy=excluded.policy, value=excluded.value, '
                'state=excluded.state, updated_at=excluded.updated_at',
                (station_id, media_id, sha256, POLICY, value, state, now))


def publish(con, client, station, media_id, sha256, value, now=None):
    """Never overwrite a value that this service did not write itself."""
    now = time.time() if now is None else now
    last = con.execute('SELECT * FROM publications WHERE station_id=? AND media_id=?',
                       (station.id, media_id)).fetchone()
    if station.mode == 'off':
        return 'off'
    if last is not None and last['state'] == 'manual':
        return 'manual'
    if value is None:
        with con:
            _record(con, station.id, media_id, sha256, 'unresolved', last['value'] if last else None, now)
        return 'unresolved'
    if station.mode == 'shadow':
        with con:
            _record(con, station.id, media_id, sha256, 'planned', value, now)
        return 'planned'
    current_raw = (client.get_file(station.id, media_id).get('custom_fields') or {}).get('intro_end')
    current = parse(current_raw)
    ours = last is not None and last['state'] == 'written'
    with con:
        # Someone else set, changed or cleared the field: their decision wins.
        if (ours and not same(current, last['value'])) or (not ours and current is not None):
            state = 'matching' if not ours and same(current, value) else 'manual'
            _record(con, station.id, media_id, sha256, state, current if state == 'manual' else value, now)
            if state == 'manual':
                audit(con, 'manual_value_kept', station.id, media_id, sha256, current_raw, value)
            return state
        if same(current, value):
            _record(con, station.id, media_id, sha256, 'written', value, now)
            return 'unchanged'
    client.set_intro(station.id, media_id, value)
    with con:
        _record(con, station.id, media_id, sha256, 'written', value, now)
        audit(con, 'write', station.id, media_id, sha256, current_raw, value)
    return 'written'
