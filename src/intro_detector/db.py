"""SQLite state. One service instance owns the database."""
import json
import sqlite3
import time
import zlib

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
'''


def connect(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path, timeout=30)
    con.row_factory = sqlite3.Row
    con.execute('PRAGMA journal_mode=WAL')
    con.execute('PRAGMA busy_timeout=30000')
    con.executescript(SCHEMA)
    return con


def pack(value):
    return zlib.compress(json.dumps(value, ensure_ascii=False, allow_nan=False,
                                    separators=(',', ':')).encode())


def unpack(blob):
    return json.loads(zlib.decompress(blob))


def audit(con, action, station_id=None, media_id=None, sha256=None, before=None, after=None, detail=None):
    con.execute('INSERT INTO audit(at, station_id, media_id, sha256, action, before, after, detail) '
                'VALUES(?,?,?,?,?,?,?,?)',
                (time.time(), station_id, media_id, sha256, action,
                 None if before is None else str(before), None if after is None else str(after), detail))
