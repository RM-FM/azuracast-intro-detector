"""Service configuration: TOML file for settings, environment for secrets."""
from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
import os
from pathlib import Path
import tomllib
from urllib.parse import urlsplit

MODES = ('off', 'shadow', 'publish')
# Relative throughput vs. the RTX 4090. The 4090 is the default because it reproduced the PoC
# (also a 4090) 20/20; A5000, A40 and L40S were measured at ~0.5 and changed 1 of 20 cues.
DEFAULT_GPUS = (('NVIDIA GeForce RTX 4090', 1.0),)


@dataclass(frozen=True)
class Station:
    id: int
    mode: str = 'shadow'
    exclude_playlists: tuple = ()
    min_song_seconds: float = 60.0


@dataclass(frozen=True)
class VocalCalibration:
    """Selector thresholds tuned against listener-checked songs."""
    vocal_level: float = .1
    min_burst_seconds: float = .5
    strong_burst_seconds: float = 2.
    strong_burst_level: float = .3
    adlib_words: tuple = ('ah', 'ba', 'be', 'boom', 'da', 'eh', 'hmm', 'mm', 'na', 'ooh', 'uh')


@dataclass(frozen=True)
class Decision:
    no_intro_below_seconds: float = 2.
    music_min_seconds: float = 3.
    music_max_seconds: float = 27.
    vocals: VocalCalibration = VocalCalibration()

    @property
    def tag(self):
        """Empty for the tested defaults, else a short settings hash recorded with each decision."""
        if self == Decision():
            return ''
        digest = hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()
        return f'+cfg-{digest[:8]}'


@dataclass(frozen=True)
class Runpod:
    image: str = 'ghcr.io/rm-fm/azuracast-intro-detector-gpu:v1'
    registry_auth_id: str = ''
    gpus: tuple = DEFAULT_GPUS
    # GPUs slower than this fraction of an RTX 4090 are never chosen.
    min_speed: float = 0.5
    max_price_per_hour: float = 0.90
    # Fixed billed time per batch (start, image pull, model loads) and RTX 4090 seconds per
    # song in one lane (measured 15 s); used only to rank GPUs by expected batch cost.
    startup_minutes: float = 5.0
    song_seconds: float = 15.0
    container_disk_gb: int = 80
    # Parallel pipeline lanes per Pod, limited by VRAM (measured peak 6.6 GB per lane).
    lanes: int = 3
    lane_vram_gb: float = 7.0
    chunk_songs: int = 10
    # Staggered lane starts so CPU-only stages overlap other lanes' GPU stages.
    stagger_seconds: int = 60
    worker_name: str = 'azuracast-intro-detector'
    ssh_key: str = '/secrets/runpod-ssh-key'
    idle_seconds: int = 600
    wait_minutes: int = 20
    retry_seconds: int = 30
    ssh_minutes: int = 20
    max_run_minutes: int = 240
    poll_seconds: int = 30
    # Estimated GPU spend per calendar month (UTC); 0 disables the limit.
    monthly_budget_usd: float = 50.0


@dataclass(frozen=True)
class Config:
    base_url: str
    api_key: str = field(repr=False)
    stations: tuple
    runpod: Runpod
    data_dir: Path = Path('/data')
    poll_seconds: int = 300
    batch_quiet_seconds: int = 120
    batch_max_songs: int = 100
    max_attempts: int = 2
    timeout_seconds: int = 60
    # False: discover and queue, but start GPU batches only on `intro-detector process`.
    auto_process: bool = True
    # True: every station AzuraCast lists, plus `stations`; False: only `stations`.
    all_stations: bool = False
    station_defaults: Station = Station(0)
    decision: Decision = Decision()

    @property
    def db_path(self):
        return self.data_dir / 'state.sqlite3'

    def station(self, station_id):
        for station in self.stations:
            if station.id == station_id:
                return station
        if self.all_stations:
            return replace(self.station_defaults, id=station_id)
        raise KeyError(f'Station {station_id} is not configured')


def _positive(name, value, kind=int, allow_zero=False):
    if not isinstance(value, kind) or isinstance(value, bool) or value < 0 or (value == 0 and not allow_zero):
        raise ValueError(f'{name} must be a {"non-negative" if allow_zero else "positive"} {kind.__name__}')
    return value


def _seconds(name, value, low=0., high=30.):
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not low <= value <= high:
        raise ValueError(f'{name} must be a number from {low:g} to {high:g}')
    return float(value)


def _station(item, defaults, name):
    station = Station(id=item.get('id', defaults.id), mode=item.get('mode', defaults.mode),
                      exclude_playlists=tuple(_positive('exclude_playlists', p)
                                              for p in item.get('exclude_playlists', defaults.exclude_playlists)),
                      min_song_seconds=item.get('min_song_seconds', defaults.min_song_seconds))
    if (not isinstance(station.min_song_seconds, (int, float)) or isinstance(station.min_song_seconds, bool)
            or station.min_song_seconds < 0):
        raise ValueError(f'{name}: min_song_seconds must be zero or positive')
    if station.mode not in MODES:
        raise ValueError(f'{name}: mode must be one of {MODES}')
    return station


def _decision(raw):
    vocals_raw = dict(raw.get('vocals', {}))
    unknown = set(vocals_raw) - set(VocalCalibration.__dataclass_fields__)
    if unknown:
        raise ValueError(f'decision.vocals: unknown settings {sorted(unknown)}')
    defaults = VocalCalibration()
    words = vocals_raw.get('adlib_words', defaults.adlib_words)
    if not isinstance(words, (list, tuple)) or not all(isinstance(w, str) and w.strip() for w in words):
        raise ValueError('decision.vocals.adlib_words must be a list of words')
    vocals = VocalCalibration(
        vocal_level=_seconds('decision.vocals.vocal_level', vocals_raw.get('vocal_level', defaults.vocal_level), .01, 1.),
        min_burst_seconds=_seconds('decision.vocals.min_burst_seconds',
                                   vocals_raw.get('min_burst_seconds', defaults.min_burst_seconds), 0., 10.),
        strong_burst_seconds=_seconds('decision.vocals.strong_burst_seconds',
                                      vocals_raw.get('strong_burst_seconds', defaults.strong_burst_seconds), .1, 30.),
        strong_burst_level=_seconds('decision.vocals.strong_burst_level',
                                    vocals_raw.get('strong_burst_level', defaults.strong_burst_level), .01, 1.),
        adlib_words=tuple(sorted({w.strip().casefold() for w in words})))
    unknown = set(raw) - set(Decision.__dataclass_fields__)
    if unknown:
        raise ValueError(f'decision: unknown settings {sorted(unknown)}')
    decision = Decision(
        no_intro_below_seconds=_seconds('decision.no_intro_below_seconds',
                                        raw.get('no_intro_below_seconds', Decision.no_intro_below_seconds)),
        music_min_seconds=_seconds('decision.music_min_seconds', raw.get('music_min_seconds', Decision.music_min_seconds)),
        music_max_seconds=_seconds('decision.music_max_seconds', raw.get('music_max_seconds', Decision.music_max_seconds)),
        vocals=vocals)
    if decision.music_min_seconds >= decision.music_max_seconds:
        raise ValueError('decision.music_min_seconds must be below music_max_seconds')
    return decision


def load(path=None, environ=None):
    env = os.environ if environ is None else environ
    path = Path(path or env.get('INTRO_DETECTOR_CONFIG', '/config/config.toml'))
    with path.open('rb') as stream:
        raw = tomllib.load(stream)
    azura = raw.get('azuracast', {})
    base_url = str(azura.get('base_url', '')).rstrip('/')
    parts = urlsplit(base_url)
    if parts.scheme not in ('https', 'http') or not parts.netloc:
        raise ValueError('azuracast.base_url must be an absolute URL')
    # Plain HTTP only for a private Docker network to AzuraCast's web container.
    if parts.scheme == 'http' and not azura.get('allow_http', False):
        raise ValueError('azuracast.base_url must use HTTPS (set allow_http only on a private network)')
    api_key = env.get('AZURACAST_API_KEY', '')
    if not api_key:
        raise ValueError('AZURACAST_API_KEY is required')
    defaults = _station(dict(raw.get('station_defaults', {}), id=0), Station(0), 'station_defaults')
    stations = []
    for item in raw.get('stations', []):
        _positive('stations.id', item.get('id'))
        stations.append(_station(item, defaults, f'Station {item.get("id")}'))
    selection = azura.get('stations', 'all' if not stations else None)
    if selection == 'all':
        all_stations = True
    elif selection is None or isinstance(selection, list):
        all_stations = False
        listed = {s.id for s in stations}
        stations += [replace(defaults, id=_positive('azuracast.stations', i))
                     for i in selection or () if i not in listed]
    else:
        raise ValueError('azuracast.stations must be "all" or a list of station IDs')
    if not all_stations and not stations or len({s.id for s in stations}) != len(stations):
        raise ValueError('Configure at least one station, each only once')
    pod = dict(raw.get('runpod', {}))
    if 'gpus' in pod:
        if not isinstance(pod['gpus'], dict) or not pod['gpus']:
            raise ValueError('runpod.gpus must map Runpod GPU type names to relative speed')
        pod['gpus'] = tuple(pod['gpus'].items())
    runpod = Runpod(**{key: tuple(value) if isinstance(value, list) else value for key, value in pod.items()})
    for name, speed in runpod.gpus:
        if not isinstance(name, str) or not name.strip() or not isinstance(speed, (int, float)) or not 0 < speed <= 10:
            raise ValueError(f'runpod.gpus: invalid entry {name!r} = {speed!r}')
    if not any(speed >= runpod.min_speed for _, speed in runpod.gpus):
        raise ValueError('runpod.min_speed excludes every configured GPU')
    for name in ('max_price_per_hour', 'startup_minutes', 'song_seconds', 'min_speed', 'lane_vram_gb'):
        value = getattr(runpod, name)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
            raise ValueError(f'runpod.{name} must be positive')
    if not isinstance(runpod.lanes, int) or not 1 <= runpod.lanes <= 9:
        raise ValueError('runpod.lanes must be 1-9')
    _positive('runpod.chunk_songs', runpod.chunk_songs)
    _positive('runpod.stagger_seconds', runpod.stagger_seconds, allow_zero=True)
    if not isinstance(runpod.monthly_budget_usd, (int, float)) or runpod.monthly_budget_usd < 0:
        raise ValueError('runpod.monthly_budget_usd must be zero or positive')
    if not 40 <= runpod.container_disk_gb <= 1000:
        raise ValueError('runpod.container_disk_gb must be 40-1000 (the image alone is ~20 GB)')
    for name in ('idle_seconds', 'wait_minutes', 'retry_seconds', 'ssh_minutes', 'max_run_minutes', 'poll_seconds'):
        _positive(f'runpod.{name}', getattr(runpod, name), allow_zero=name == 'wait_minutes')
    scan = raw.get('scan', {})
    auto_process = scan.get('auto_process', True)
    if not isinstance(auto_process, bool):
        raise ValueError('scan.auto_process must be true or false')
    config = Config(base_url=base_url, api_key=api_key, stations=tuple(stations), runpod=runpod,
                    data_dir=Path(raw.get('data_dir', '/data')), auto_process=auto_process,
                    poll_seconds=_positive('scan.poll_seconds', scan.get('poll_seconds', 300)),
                    batch_quiet_seconds=_positive('scan.batch_quiet_seconds', scan.get('batch_quiet_seconds', 120), allow_zero=True),
                    batch_max_songs=_positive('scan.batch_max_songs', scan.get('batch_max_songs', 100)),
                    max_attempts=_positive('scan.max_attempts', scan.get('max_attempts', 2)),
                    timeout_seconds=_positive('azuracast.timeout_seconds', azura.get('timeout_seconds', 60)),
                    all_stations=all_stations, station_defaults=defaults,
                    decision=_decision(raw.get('decision', {})))
    return config
