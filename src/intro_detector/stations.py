"""Station settings managed through the API, Runpod keys per station and settings migrations."""
from dataclasses import asdict, dataclass, fields, replace
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

from .config import Station

MAX_USD = 100000.
# Version of the stored settings model; version 1 had a shared Runpod key, version 2 a shadow mode.
SETTINGS_VERSION = 3


def key_slot(station_id):
    """Name under which a station's Runpod key is stored."""
    return f'station-{int(station_id)}'


def key_account(key):
    """Account of a key until Runpod reports its account; named after the key's hash."""
    return 'runpod-' + hashlib.sha256(key.encode()).hexdigest()[:12]


def runpod_account(user_id):
    """Account of every key of one Runpod user; named after the hashed user ID."""
    return 'runpod-' + hashlib.sha256(f'user:{user_id}'.encode()).hexdigest()[:12]


def _bool(name, value):
    if not isinstance(value, bool):
        raise ValueError(f'{name} must be true or false')
    return value


def _number(name, value, low, high):
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) \
            or not low <= value <= high:
        raise ValueError(f'{name} must be a number from {low:g} to {high:g}')
    return float(value)


def _playlists(value):
    if not isinstance(value, (list, tuple)) or len(value) > 1000 or not all(
            isinstance(p, int) and not isinstance(p, bool) and p > 0 for p in value):
        raise ValueError('exclude_playlists must be a list of playlist IDs')
    return tuple(sorted(set(value)))


def valid_key(key):
    if not isinstance(key, str) or not 8 <= len(key) <= 256 or any(
            c.isspace() or c in '\\"\'\x00' for c in key):
        raise ValueError('The Runpod API key must be a single token without spaces or quotes')
    return key


class Secrets:
    """Runpod API keys per station in a 0600 file of the data volume; never returned by the API."""

    def __init__(self, data_dir):
        self.path = Path(data_dir) / 'secrets.json'

    def _read(self):
        try:
            data = json.loads(self.path.read_text())
        except FileNotFoundError:
            return {}
        return data if isinstance(data, dict) else {}

    def get(self, slot):
        return self._read().get(slot) or None

    def has(self, slot):
        return self.get(slot) is not None

    def slots(self):
        return sorted(slot for slot, key in self._read().items() if key)

    def keys(self):
        return sorted({key for key in self._read().values() if key})

    def set(self, slot, key):
        data = self._read()
        if key is None:
            data.pop(slot, None)
        else:
            data[slot] = valid_key(key)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(dir=self.path.parent, prefix='.secrets-')
        try:
            with os.fdopen(fd, 'w') as stream:
                json.dump(data, stream)
            os.chmod(temporary, 0o600)
            os.replace(temporary, self.path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)


@dataclass(frozen=True)
class StationSettings:
    id: int
    # An enabled station writes its intros to AzuraCast.
    enabled: bool = False
    # False: new songs wait for "Process now".
    auto_process: bool = True
    min_song_seconds: float = 60.0
    # Longer files, such as DJ mixes, are skipped; 0 means no maximum.
    max_song_seconds: float = 900.0
    exclude_playlists: tuple = ()
    # Estimated GPU spend of this station per UTC month; 0 disables the limit.
    budget_usd: float = 10.0
    # Minutes a GPU worker stays after its last batch; a shared worker uses the longest of its stations.
    idle_timeout_minutes: float = 5.0

    @property
    def station(self):
        return Station(self.id, 'publish' if self.enabled else 'off', self.exclude_playlists, self.min_song_seconds,
                       self.max_song_seconds)

    def public(self):
        return {**asdict(self), 'exclude_playlists': list(self.exclude_playlists)}


STATION_FIELDS = tuple(f.name for f in fields(StationSettings) if f.name != 'id')


def _validated(settings, changes):
    unknown = set(changes) - set(STATION_FIELDS)
    if unknown:
        raise ValueError(f'Unknown station settings: {sorted(unknown)}')
    values = dict(changes)
    for name in ('enabled', 'auto_process'):
        if name in values:
            _bool(name, values[name])

    if 'min_song_seconds' in values:
        values['min_song_seconds'] = _number('min_song_seconds', values['min_song_seconds'], 0, 3600)
    if 'max_song_seconds' in values:
        values['max_song_seconds'] = _number('max_song_seconds', values['max_song_seconds'], 0, 86400)
    if 'exclude_playlists' in values:
        values['exclude_playlists'] = _playlists(values['exclude_playlists'])
    if 'budget_usd' in values:
        values['budget_usd'] = _number('budget_usd', values['budget_usd'], 0, MAX_USD)
    if 'idle_timeout_minutes' in values:
        values['idle_timeout_minutes'] = _number('idle_timeout_minutes', values['idle_timeout_minutes'], 1, 60)
    updated = replace(settings, **values)
    if updated.max_song_seconds and updated.max_song_seconds <= updated.min_song_seconds:
        raise ValueError('max_song_seconds must be above min_song_seconds, or 0 for no maximum')
    return updated


class Stations:
    def __init__(self, con, cfg, clock):
        self.con, self.cfg, self.clock = con, cfg, clock

    def defaults(self, station_id):
        base = self.cfg.station_defaults
        return StationSettings(int(station_id), auto_process=self.cfg.auto_process, min_song_seconds=float(base.min_song_seconds),
                               exclude_playlists=tuple(base.exclude_playlists),
                               budget_usd=float(self.cfg.runpod.monthly_budget_usd))

    @staticmethod
    def _from_row(row):
        values = {f.name: row['station_id' if f.name == 'id' else f.name] for f in fields(StationSettings)}
        for name in ('enabled', 'auto_process'):
            values[name] = bool(values[name])
        values['exclude_playlists'] = tuple(json.loads(values['exclude_playlists']))
        return StationSettings(**values)

    def get(self, station_id):
        row = self.con.execute('SELECT * FROM station_settings WHERE station_id=?', (int(station_id),)).fetchone()
        return self._from_row(row) if row else self.defaults(station_id)

    def all(self):
        return [self._from_row(r) for r in self.con.execute('SELECT * FROM station_settings ORDER BY station_id')]

    def enabled(self):
        return [s for s in self.all() if s.enabled]

    def save(self, settings):
        values = asdict(settings)
        values['exclude_playlists'] = json.dumps(list(settings.exclude_playlists))
        station_id = values.pop('id')
        columns = ['station_id', *values, 'updated_at']
        with self.con:
            self.con.execute(
                f'INSERT INTO station_settings({", ".join(columns)}) VALUES({", ".join("?" * len(columns))}) '
                'ON CONFLICT(station_id) DO UPDATE SET '
                + ', '.join(f'{c}=excluded.{c}' for c in columns[1:]),
                (station_id, *values.values(), self.clock()))
        return settings

    def update(self, station_id, changes):
        return self.save(_validated(self.get(station_id), changes))


class Instance:
    """Internal markers of the service; all settings belong to stations."""

    def __init__(self, con):
        self.con = con

    def _stored(self):
        return {r['key']: json.loads(r['value']) for r in self.con.execute('SELECT * FROM instance_settings')}

    def marker(self, name):
        return self._stored().get(f'_{name}')

    def set_marker(self, name, value):
        with self.con:
            self.con.execute('INSERT OR REPLACE INTO instance_settings(key, value) VALUES(?,?)',
                             (f'_{name}', json.dumps(value)))


def _keep_worker_name(con, instance, old_account, key, name):
    """Records of an earlier account move to the key's account; its worker keeps its Runpod name."""
    account = key_account(key)
    names = instance.marker('worker_names') or {}
    names.setdefault(account, name)
    instance.set_marker('worker_names', names)
    recovery = instance.marker(f'recovery:{old_account}')
    if recovery:
        instance.set_marker(f'recovery:{account}', recovery)
    with con:
        con.execute('DELETE FROM instance_settings WHERE key=?', (f'_recovery:{old_account}',))
        for table in ('pods', 'runs'):
            con.execute(f'UPDATE {table} SET account=? WHERE account=?', (account, old_account))


def _from_legacy_install(con, cfg, stations, instance, secrets, environ):
    """0.x: configured stations stay enabled and get the Runpod key from .env."""
    existing = con.execute('SELECT EXISTS(SELECT 1 FROM stations) OR EXISTS(SELECT 1 FROM analyses)').fetchone()[0]
    if not existing:
        return False
    key = environ.get('RUNPOD_API_KEY')
    ids = {s.id for s in cfg.stations}
    if cfg.all_stations:
        ids |= {row[0] for row in con.execute('SELECT station_id FROM stations')}
    for station_id in sorted(ids):
        legacy = cfg.station(station_id)
        stations.save(replace(stations.defaults(station_id), enabled=legacy.mode != 'off',
                              min_song_seconds=float(legacy.min_song_seconds),
                              exclude_playlists=tuple(legacy.exclude_playlists)))
        if key:
            secrets.set(key_slot(station_id), key)
    if key:
        _keep_worker_name(con, instance, 'shared', key, cfg.runpod.worker_name)
    return True


def _from_shared_key(con, cfg, instance, secrets):
    """Version 1: stations billed to the shared key now hold that key themselves."""
    shared = secrets.get('shared')
    columns = {row['name'] for row in con.execute('PRAGMA table_info(station_settings)')}
    for row in con.execute('SELECT * FROM station_settings').fetchall():
        slot = key_slot(row['station_id'])
        if shared and 'billing' in columns and row['billing'] == 'shared':
            secrets.set(slot, shared)
        elif secrets.has(slot):
            _keep_worker_name(con, instance, slot, secrets.get(slot), f'{cfg.runpod.worker_name}-{slot}')
    if shared:
        _keep_worker_name(con, instance, 'shared', shared, cfg.runpod.worker_name)
        secrets.set('shared', None)
    with con:
        for key in [k for k in instance._stored() if not k.startswith('_')]:
            con.execute('DELETE FROM instance_settings WHERE key=?', (key,))
        con.execute('DROP TABLE IF EXISTS analysis_accounts')


def migrate(con, cfg, stations, instance, secrets, environ):
    """Bring stored settings to the current model once; True when an existing installation was migrated."""
    version = instance.marker('migrated') or 0
    if version >= SETTINGS_VERSION:
        return False
    if version == 0:
        migrated = _from_legacy_install(con, cfg, stations, instance, secrets, environ)
    else:
        if version == 1:
            _from_shared_key(con, cfg, instance, secrets)
        migrated = True
    _publish_proposals(con, stations)
    instance.set_marker('migrated', SETTINGS_VERSION)
    return migrated


def _publish_proposals(con, stations):
    """Intros proposed while a station only recorded them are written once it is enabled."""
    with con:
        con.executemany("INSERT OR REPLACE INTO station_jobs(station_id, kind, requested_at) VALUES(?, 'publish', 0)",
                        [(s.id,) for s in stations.enabled()])
