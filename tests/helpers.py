"""Shared fakes; no network, GPU or AzuraCast access."""
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from intro_detector.config import Config, Runpod, Station  # noqa: E402
from intro_detector.db import connect  # noqa: E402
from intro_detector.evidence import INPUTS  # noqa: E402


def make_config(root, *stations, **extra):
    return Config(base_url='https://radio.example', api_key='secret', runpod=Runpod(),
                  stations=stations or (Station(1, 'publish'),), data_dir=Path(root), **extra)


def database(root):
    return connect(Path(root) / 'state.sqlite3')


def report_row(song, sha256, duration=30.):
    return dict(song=song, sha256=sha256, duration=duration, assets={'original': {'audio': '/x'}},
                candidates=[], smad={'class_order': ['none', 'singing_music'], 'window_starts': [0.],
                                     'probabilities': [[.5, .5]], 'duration': duration},
                asr={f'{v}/{m}': {'text': '', 'first_word_start': None}
                     for v in INPUTS for m in ('turbo', 'qwen', 'whisper_auto')},
                acoustic={f'{v}/{n}': {'curves': {}} for v in INPUTS for n in (4, 8)})


def music_row(song, sha256, candidate=None):
    return dict(song=song, sha256=sha256, duration=30., candidate=candidate, reason='x')


def write_reports(directory, rows, music, errors=()):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'report.json').write_text(json.dumps(dict(
        policy='intro-benchmark-v2-smad-auto-language', rows=rows, errors=list(errors))))
    (directory / 'music-report.json').write_text(json.dumps(dict(
        profile='percussion-energy-30s-v1', rows=music, errors=[])))


class FakeClient:
    def __init__(self, files=(), playlists=()):
        self.files = {f['id']: f for f in files}
        self.playlists = list(playlists)
        self.stations = {1: 'Station 1'}
        self.audio = {}
        self.writes = []

    def station_names(self):
        return dict(self.stations)

    def list_files(self, station_id):
        return list(self.files.values())

    def list_playlists(self, station_id):
        return self.playlists

    def get_file(self, station_id, media_id):
        return self.files[media_id]

    def download(self, station_id, media_id, directory, suffix):
        import hashlib
        data = self.audio.get(media_id, f'audio-{media_id}'.encode())
        sha = hashlib.sha256(data).hexdigest()
        Path(directory).mkdir(parents=True, exist_ok=True)
        path = Path(directory) / (sha + suffix)
        path.write_bytes(data)
        return path, sha

    def set_intro(self, station_id, media_id, value):
        self.writes.append((station_id, media_id, value))
        self.files[media_id].setdefault('custom_fields', {})['intro_end'] = '' if value is None else str(value)
        return value


def media(media_id, path='a.mp3', mtime=1, playlists=(), intro=None, length=200):
    return dict(id=media_id, unique_id=f'u{media_id}', path=path, mtime=mtime, length=length,
                playlists=[{'id': p} for p in playlists], custom_fields={'intro_end': intro})


class Clock:
    def __init__(self, now=1000.):
        self.now = now

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def tempdir():
    return tempfile.TemporaryDirectory()
