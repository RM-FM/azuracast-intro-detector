"""Station and instance settings managed through the API, Runpod keys and the 0.x migration."""
from dataclasses import asdict, dataclass, fields, replace
import json
import math
import os
from pathlib import Path
import tempfile

from .config import Station

SHARED = 'shared'
STATION_MODES = ('shadow', 'publish')
BILLING = ('own', 'shared')
MAX_USD = 100000.


def station_account(station_id):
    return f'station-{int(station_id)}'


def account_station(account):
    """Station ID of a station account, or None for the shared account."""
    prefix, _, number = account.partition('-')
    return int(number) if prefix == 'station' and number.isdigit() else None


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
    """Runpod API keys per account in a 0600 file of the data volume; never returned by the API."""

    def __init__(self, data_dir):
        self.path = Path(data_dir) / 'secrets.json'

    def _read(self):
        try:
            data = json.loads(self.path.read_text())
        except FileNotFoundError:
            return {}
        return data if isinstance(data, dict) else {}

    def get(self, account):
        return self._read().get(account) or None

    def has(self, account):
        return self.get(account) is not None

    def accounts(self):
        return sorted(account for account, key in self._read().items() if key)

    def set(self, account, key):
        data = self._read()
        if key is None:
            data.pop(account, None)
        else:
            data[account] = valid_key(key)
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
    enabled: bool = False
    mode: str = 'shadow'
    # False: new songs wait for "Process now".
    auto_process: bool = True
    min_song_seconds: float = 60.0
    exclude_playlists: tuple = ()
    # 'own': the station's Runpod key and budget_usd; 'shared': the instance key within shared_cap_usd.
    billing: str = 'own'
    budget_usd: float = 10.0
    shared_allowed: bool = False
    # Monthly limit of this station on the shared key; 0 means only the shared budget applies.
    shared_cap_usd: float = 0.0

    @property
    def account(self):
        return SHARED if self.billing == 'shared' else station_account(self.id)

    @property
    def station(self):
        return Station(self.id, self.mode if self.enabled else 'off', self.exclude_playlists, self.min_song_seconds)

    def public(self):
        return {**asdict(self), 'exclude_playlists': list(self.exclude_playlists), 'account': self.account}


STATION_FIELDS = ('enabled', 'mode', 'auto_process', 'min_song_seconds', 'exclude_playlists', 'billing', 'budget_usd')
ADMIN_FIELDS = ('shared_allowed', 'shared_cap_usd')


def _validated(settings, changes, admin):
    allowed = STATION_FIELDS + (ADMIN_FIELDS if admin else ())
    unknown = set(changes) - set(allowed)
    if unknown:
        raise ValueError(f'Settings not allowed here: {sorted(unknown)}')
    values = dict(changes)
    for name in ('enabled', 'auto_process', 'shared_allowed'):
        if name in values:
            _bool(name, values[name])
    if 'mode' in values and values['mode'] not in STATION_MODES:
        raise ValueError(f'mode must be one of {STATION_MODES}')
    if 'billing' in values and values['billing'] not in BILLING:
        raise ValueError(f'billing must be one of {BILLING}')
    if 'min_song_seconds' in values:
        values['min_song_seconds'] = _number('min_song_seconds', values['min_song_seconds'], 0, 3600)
    if 'exclude_playlists' in values:
        values['exclude_playlists'] = _playlists(values['exclude_playlists'])
    for name in ('budget_usd', 'shared_cap_usd'):
        if name in values:
            values[name] = _number(name, values[name], 0, MAX_USD)
    updated = replace(settings, **values)
    if updated.billing == 'shared' and not updated.shared_allowed:
        if 'billing' in values:
            raise ValueError('The instance key is not available to this station')
        updated = replace(updated, billing='own')
    return updated


class Stations:
    def __init__(self, con, cfg, clock):
        self.con, self.cfg, self.clock = con, cfg, clock

    def defaults(self, station_id):
        base = self.cfg.station_defaults
        return StationSettings(int(station_id), mode=base.mode if base.mode in STATION_MODES else 'shadow',
                               auto_process=self.cfg.auto_process, min_song_seconds=float(base.min_song_seconds),
                               exclude_playlists=tuple(base.exclude_playlists))

    @staticmethod
    def _from_row(row):
        values = {f.name: row['station_id' if f.name == 'id' else f.name] for f in fields(StationSettings)}
        for name in ('enabled', 'auto_process', 'shared_allowed'):
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

    def update(self, station_id, changes, admin=False):
        return self.save(_validated(self.get(station_id), changes, admin))


@dataclass(frozen=True)
class InstanceSettings:
    max_price_per_hour: float
    shared_budget_usd: float
    share_analyses: bool = True


class Instance:
    def __init__(self, con, cfg):
        self.con, self.cfg = con, cfg

    def _stored(self):
        return {r['key']: json.loads(r['value']) for r in self.con.execute('SELECT * FROM instance_settings')}

    def get(self):
        stored = self._stored()
        defaults = InstanceSettings(self.cfg.runpod.max_price_per_hour, self.cfg.runpod.monthly_budget_usd)
        return replace(defaults, **{f.name: stored[f.name] for f in fields(InstanceSettings) if f.name in stored})

    def update(self, changes):
        unknown = set(changes) - {f.name for f in fields(InstanceSettings)}
        if unknown:
            raise ValueError(f'Unknown instance settings: {sorted(unknown)}')
        values = dict(changes)
        if 'share_analyses' in values:
            _bool('share_analyses', values['share_analyses'])
        if 'max_price_per_hour' in values:
            values['max_price_per_hour'] = _number('max_price_per_hour', values['max_price_per_hour'], .05, 20)
        if 'shared_budget_usd' in values:
            values['shared_budget_usd'] = _number('shared_budget_usd', values['shared_budget_usd'], 0, MAX_USD)
        self._put(values)
        return self.get()

    def _put(self, values):
        with self.con:
            self.con.executemany('INSERT OR REPLACE INTO instance_settings(key, value) VALUES(?,?)',
                                 [(k, json.dumps(v)) for k, v in values.items()])

    def marker(self, name):
        return self._stored().get(f'_{name}')

    def set_marker(self, name, value):
        self._put({f'_{name}': value})


def migrate(con, cfg, stations, instance, secrets, environ):
    """Once: a Runpod key from .env becomes the shared key; stations of a 0.x install stay enabled on it."""
    if instance.marker('migrated'):
        return False
    key = environ.get('RUNPOD_API_KEY')
    if key and not secrets.has(SHARED):
        secrets.set(SHARED, key)
    existing = con.execute('SELECT EXISTS(SELECT 1 FROM stations) OR EXISTS(SELECT 1 FROM analyses)').fetchone()[0]
    if existing:
        ids = {s.id for s in cfg.stations}
        if cfg.all_stations:
            ids |= {row[0] for row in con.execute('SELECT station_id FROM stations')}
        for station_id in sorted(ids):
            legacy = cfg.station(station_id)
            stations.save(replace(stations.defaults(station_id), enabled=legacy.mode != 'off',
                                  mode=legacy.mode if legacy.mode in STATION_MODES else 'shadow',
                                  min_song_seconds=float(legacy.min_song_seconds),
                                  exclude_playlists=tuple(legacy.exclude_playlists),
                                  billing='shared', shared_allowed=True))
        with con:
            con.execute('INSERT OR IGNORE INTO analysis_accounts(sha256, account) SELECT sha256, ? FROM analyses',
                        (SHARED,))
    instance.set_marker('migrated', 1)
    return bool(existing)
