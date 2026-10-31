"""Library discovery: mirror AzuraCast media metadata into the local queue."""
from collections import Counter
import json
from pathlib import PurePosixPath
import time

# Formats accepted by the vendored Pod pipeline (whisper_experiment.EXTENSIONS).
EXTENSIONS = {'.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac', '.opus'}


def intro_value(item):
    raw = (item.get('custom_fields') or {}).get('intro_end')
    return None if raw is None or str(raw).strip() == '' else str(raw).strip()


def playlist_ids(item):
    return sorted({int(p['id']) for p in item.get('playlists') or []
                   if isinstance(p, dict) and str(p.get('id', '')).isdigit()})


def jingle_playlists(client, station_id):
    return {int(p['id']) for p in client.list_playlists(station_id)
            if isinstance(p, dict) and p.get('is_jingle') and str(p.get('id', '')).isdigit()}


def sync_station(con, client, station, now=None):
    """New files after the first sync queue automatically; the first sync only records the library."""
    now = time.time() if now is None else now
    skip = set(station.exclude_playlists) | jingle_playlists(client, station.id)
    items = client.list_files(station.id)
    initialized = con.execute('SELECT 1 FROM stations WHERE station_id=?', (station.id,)).fetchone() is not None
    counts, seen = Counter(), set()
    with con:
        for item in items:
            media_id = int(item['id'])
            seen.add(media_id)
            path = str(item.get('path') or '')
            playlists = playlist_ids(item)
            length = float(item.get('length') or 0)
            # Unknown length (0) counts as short; the file queues once AzuraCast reports it.
            excluded = (PurePosixPath(path).suffix.lower() not in EXTENSIONS
                        or length < station.min_song_seconds
                        or bool(set(playlists) & skip))
            fields = dict(unique_id=str(item.get('unique_id') or ''), path=path,
                          mtime=int(item.get('mtime') or 0), length=length,
                          playlists=json.dumps(playlists), intro_end=intro_value(item))
            old = con.execute('SELECT * FROM media WHERE station_id=? AND media_id=?',
                              (station.id, media_id)).fetchone()
            if old is None:
                status = 'excluded' if excluded else 'pending' if initialized else 'known'
                con.execute('INSERT INTO media(station_id, media_id, unique_id, path, mtime, length, playlists, '
                            'intro_end, status, discovered_at, updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)',
                            (station.id, media_id, *fields.values(), status, now, now))
                counts[status] += 1
                continue
            changed = old['unique_id'] != fields['unique_id'] or old['mtime'] != fields['mtime']
            status, extra = old['status'], {}
            if excluded:
                status = 'excluded'
            elif old['status'] in ('excluded', 'deleted') or (changed and old['status'] != 'known'):
                # A replaced or newly eligible file is analyzed again; its hash decides reuse.
                status, extra = 'pending', dict(sha256=None, attempts=0, priority=0, error=None, discovered_at=now)
            if status != old['status']:
                counts[status] += 1
            values = {**fields, **extra, 'status': status, 'updated_at': now}
            con.execute(f'UPDATE media SET {", ".join(k + "=?" for k in values)} WHERE station_id=? AND media_id=?',
                        (*values.values(), station.id, media_id))
        for row in con.execute("SELECT media_id FROM media WHERE station_id=? AND status!='deleted'",
                               (station.id,)).fetchall():
            if row['media_id'] not in seen:
                con.execute("UPDATE media SET status='deleted', updated_at=? WHERE station_id=? AND media_id=?",
                            (now, station.id, row['media_id']))
                counts['deleted'] += 1
        con.execute('INSERT INTO stations(station_id, initialized_at, last_sync) VALUES(?,?,?) '
                    'ON CONFLICT(station_id) DO UPDATE SET last_sync=excluded.last_sync',
                    (station.id, now, now))
    return dict(counts, total=len(items))


def backfill(con, station_id, limit=None, include_set=False, dry_run=False, now=None):
    """Queue existing library songs at lower priority than new uploads."""
    now = time.time() if now is None else now
    query = "SELECT media_id FROM media WHERE station_id=? AND status='known'"
    if not include_set:
        query += ' AND intro_end IS NULL'
    query += ' ORDER BY media_id'
    rows = [r['media_id'] for r in con.execute(query, (station_id,)).fetchall()]
    rows = rows[:limit] if limit else rows
    if not dry_run:
        with con:
            con.executemany("UPDATE media SET status='pending', priority=1, attempts=0, error=NULL, "
                            'discovered_at=?, updated_at=? WHERE station_id=? AND media_id=?',
                            [(now, now, station_id, media_id) for media_id in rows])
    return len(rows)
