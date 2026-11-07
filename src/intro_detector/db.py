"""SQLite state. One service instance owns the database."""
import json
import lzma
import sqlite3
import time
import zlib

import numpy

# Lossless binary evidence; blobs without this prefix are the zlib JSON of earlier releases.
MAGIC = b'IDE1'
ARRAY_REF = '$a'
MIN_ARRAY = 32
LZMA_PRESET = 6

SCHEMA = '''
CREATE TABLE IF NOT EXISTS stations(
  station_id INTEGER PRIMARY KEY,
  initialized_at REAL NOT NULL,
  last_sync REAL);
CREATE TABLE IF NOT EXISTS media(
  station_id INTEGER NOT NULL,
  media_id INTEGER NOT NULL,
  unique_id TEXT NOT NULL,
  path TEXT NOT NULL,
  mtime INTEGER NOT NULL,
  length REAL,
  playlists TEXT NOT NULL DEFAULT '[]',
  intro_end TEXT,
  sha256 TEXT,
  status TEXT NOT NULL,
  priority INTEGER NOT NULL DEFAULT 0,
  attempts INTEGER NOT NULL DEFAULT 0,
  error TEXT,
  discovered_at REAL NOT NULL,
  updated_at REAL NOT NULL,
  PRIMARY KEY(station_id, media_id));
CREATE INDEX IF NOT EXISTS media_status ON media(status, priority, discovered_at);
CREATE INDEX IF NOT EXISTS media_sha ON media(sha256);
CREATE TABLE IF NOT EXISTS analyses(
  sha256 TEXT PRIMARY KEY,
  source TEXT NOT NULL,
  run_id TEXT,
  report BLOB NOT NULL,
  music BLOB NOT NULL,
  created_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS decisions(
  sha256 TEXT NOT NULL,
  policy TEXT NOT NULL,
  raw TEXT NOT NULL,
  marker TEXT NOT NULL,
  value REAL,
  created_at REAL NOT NULL,
  PRIMARY KEY(sha256, policy));
CREATE TABLE IF NOT EXISTS publications(
  station_id INTEGER NOT NULL,
  media_id INTEGER NOT NULL,
  sha256 TEXT,
  policy TEXT,
  value REAL,
  state TEXT NOT NULL,
  updated_at REAL NOT NULL,
  PRIMARY KEY(station_id, media_id));
CREATE TABLE IF NOT EXISTS audit(
  id INTEGER PRIMARY KEY,
  at REAL NOT NULL,
  station_id INTEGER,
  media_id INTEGER,
  sha256 TEXT,
  action TEXT NOT NULL,
  before TEXT,
  after TEXT,
  detail TEXT);
CREATE TABLE IF NOT EXISTS reviews(
  sha256 TEXT NOT NULL,
  kind TEXT NOT NULL,
  note TEXT,
  created_at REAL NOT NULL,
  PRIMARY KEY(sha256, kind));
CREATE TABLE IF NOT EXISTS runs(
  id TEXT PRIMARY KEY,
  pod_id TEXT,
  songs INTEGER NOT NULL,
  status TEXT NOT NULL,
  error TEXT,
  started_at REAL NOT NULL,
  finished_at REAL);
CREATE TABLE IF NOT EXISTS ab_analyses(
  set_name TEXT NOT NULL,
  sha256 TEXT NOT NULL,
  report BLOB NOT NULL,
  music BLOB NOT NULL,
  created_at REAL NOT NULL,
  PRIMARY KEY(set_name, sha256));
CREATE TABLE IF NOT EXISTS pods(
  pod_id TEXT PRIMARY KEY,
  gpu_id TEXT,
  gpu_memory_gb REAL,
  price REAL NOT NULL,
  created_at REAL NOT NULL,
  ended_at REAL,
  copied INTEGER NOT NULL DEFAULT 0,
  idle_since REAL);
CREATE TABLE IF NOT EXISTS station_settings(
  station_id INTEGER PRIMARY KEY,
  enabled INTEGER NOT NULL DEFAULT 0,
  min_song_seconds REAL NOT NULL,
  max_song_seconds REAL NOT NULL DEFAULT 900,
  exclude_playlists TEXT NOT NULL DEFAULT '[]',
  budget_usd REAL NOT NULL,
  idle_timeout_minutes REAL NOT NULL DEFAULT 5,
  eu_only INTEGER NOT NULL DEFAULT 0,
  auto_process INTEGER NOT NULL DEFAULT 1,
  updated_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS instance_settings(
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS run_stations(
  run_id TEXT NOT NULL,
  station_id INTEGER NOT NULL,
  songs INTEGER NOT NULL,
  PRIMARY KEY(run_id, station_id));
CREATE TABLE IF NOT EXISTS process_requests(
  station_id INTEGER PRIMARY KEY,
  force INTEGER NOT NULL DEFAULT 0,
  requested_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS station_jobs(
  station_id INTEGER NOT NULL,
  kind TEXT NOT NULL,
  requested_at REAL NOT NULL,
  PRIMARY KEY(station_id, kind));
CREATE TABLE IF NOT EXISTS unused(
  sha256 TEXT PRIMARY KEY,
  since REAL NOT NULL);
'''
# Columns added after the first release; the migration assigns existing rows to the key's account.
COLUMNS = (('pods', 'account', "TEXT NOT NULL DEFAULT 'shared'"),
           ('runs', 'account', "TEXT NOT NULL DEFAULT 'shared'"),
           ('runs', 'phase', 'TEXT'),
           ('station_settings', 'max_song_seconds', 'REAL NOT NULL DEFAULT 900'),
           ('station_settings', 'idle_timeout_minutes', 'REAL NOT NULL DEFAULT 5'),
           ('station_settings', 'eu_only', 'INTEGER NOT NULL DEFAULT 0'),
           ('pods', 'location', 'TEXT'))


def connect(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path, timeout=30)
    con.row_factory = sqlite3.Row
    # Takes effect for a new database; an existing one switches with its first housekeeping rebuild.
    con.execute('PRAGMA auto_vacuum=INCREMENTAL')
    con.execute('PRAGMA journal_mode=WAL')
    con.execute('PRAGMA busy_timeout=30000')
    con.executescript(SCHEMA)
    for table, column, declaration in COLUMNS:
        if column not in {row['name'] for row in con.execute(f'PRAGMA table_info({table})')}:
            con.execute(f'ALTER TABLE {table} ADD COLUMN {column} {declaration}')
    con.commit()
    return con


def pack(value):
    """Lossless binary evidence: number arrays stored once as typed data, the rest as JSON."""
    arrays, index = [], {}

    def walk(node):
        if isinstance(node, dict):
            return {k: walk(v) for k, v in node.items()}
        if isinstance(node, list):
            if len(node) >= MIN_ARRAY and all(type(x) is float for x in node):
                data = _encode_array(node)
                ref = index.setdefault((data.dtype.str, data.tobytes()), len(arrays))
                if ref == len(arrays):
                    arrays.append(data)
                return {ARRAY_REF: ref}
            return [walk(v) for v in node]
        return node

    header = json.dumps({'v': walk(value), 'a': [[data.dtype.str, len(data)] for data in arrays]},
                        ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode()
    # Grouping the n-th bytes of all numbers together lets the compressor find the repetition.
    body = b''.join(data.view('u1').reshape(-1, data.itemsize).T.tobytes() for data in arrays)
    payload = len(header).to_bytes(4, 'little') + header + body
    return MAGIC + lzma.compress(payload, preset=LZMA_PRESET)


def unpack(blob):
    if not blob.startswith(MAGIC):
        return json.loads(zlib.decompress(blob))
    payload = lzma.decompress(blob[len(MAGIC):])
    size = int.from_bytes(payload[:4], 'little')
    header = json.loads(payload[4:4 + size])
    arrays, offset = [], 4 + size
    for dtype, count in header['a']:
        dtype = numpy.dtype(dtype)
        width = count * dtype.itemsize
        data = numpy.frombuffer(payload, 'u1', width, offset).reshape(dtype.itemsize, count).T.copy().view(dtype)
        arrays.append(data.ravel().astype(numpy.float64).tolist())
        offset += width

    def fill(node):
        if isinstance(node, dict):
            return arrays[node[ARRAY_REF]] if node.keys() == {ARRAY_REF} else {k: fill(v) for k, v in node.items()}
        if isinstance(node, list):
            return [fill(v) for v in node]
        return node
    return fill(header['v'])


def _encode_array(values):
    data = numpy.asarray(values, dtype=numpy.float64)
    single = data.astype('<f4')
    return single if numpy.array_equal(single.astype(numpy.float64), data) else data.astype('<f8')


def audit(con, action, station_id=None, media_id=None, sha256=None, before=None, after=None, detail=None):
    con.execute('INSERT INTO audit(at, station_id, media_id, sha256, action, before, after, detail) '
                'VALUES(?,?,?,?,?,?,?,?)',
                (time.time(), station_id, media_id, sha256, action,
                 None if before is None else str(before), None if after is None else str(after), detail))
