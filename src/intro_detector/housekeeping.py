"""Keep the data volume small without operator work: drop unused evidence, store it compactly, shrink the file."""
import logging
import shutil
import time

from .db import MAGIC, pack, unpack

log = logging.getLogger(__name__)
# Below this much free space no song is downloaded or analyzed.
RESERVE_BYTES = 500 << 20
# Evidence of a song that left every library is kept this long, so a short AzuraCast hiccup costs nothing.
UNUSED_SECONDS = 24 * 3600
CONVERT_SECONDS = 20
CONVERT_ROWS = 25
# Free pages beyond this are returned to the file system.
SHRINK_BYTES = 64 << 20
INCREMENTAL = 2
REFERENCED = ("SELECT sha256 FROM media WHERE sha256 IS NOT NULL AND status!='deleted' "
              'UNION SELECT p.sha256 FROM publications p JOIN media m '
              'ON m.station_id=p.station_id AND m.media_id=p.media_id '
              "WHERE p.sha256 IS NOT NULL AND m.status!='deleted'")
CONVERTED = 'storage:converted'
TABLES = ('analyses', 'ab_analyses')


def free_bytes(path):
    return shutil.disk_usage(path).free


def disk_full(path):
    return free_bytes(path) < RESERVE_BYTES


class Housekeeping:
    def __init__(self, con, data_dir, instance, clock=time.time):
        self.con, self.data_dir, self.instance, self.clock = con, data_dir, instance, clock

    def tick(self):
        """One idle step; each part is cheap when there is nothing to do."""
        self.forget_unused()
        if self.convert():
            return
        self.shrink()

    def forget_unused(self):
        """Evidence no current or published song uses; removed after a grace period."""
        now = self.clock()
        with self.con:
            self.con.execute(f'DELETE FROM unused WHERE sha256 IN ({REFERENCED})')
            self.con.execute(f'INSERT OR IGNORE INTO unused(sha256, since) SELECT sha256, ? FROM analyses '
                             f'WHERE sha256 NOT IN ({REFERENCED})', (now,))
            expired = [r[0] for r in self.con.execute('SELECT sha256 FROM unused WHERE since<?', (now - UNUSED_SECONDS,))]
            for sha256 in expired:
                for table in ('analyses', 'decisions', 'unused'):
                    self.con.execute(f'DELETE FROM {table} WHERE sha256=?', (sha256,))
        if expired:
            log.info('Removed the stored analyses of %d songs that left every library', len(expired))
        return len(expired)

    def convert(self):
        """Rewrite evidence of earlier releases in the compact format; True while rows remain."""
        progress = self.instance.marker(CONVERTED) or {}
        deadline = self.clock() + CONVERT_SECONDS
        for table in TABLES:
            last = progress.get(table, 0)
            while last is not None:
                rows = self.con.execute(f'SELECT rowid, report, music FROM {table} WHERE rowid>? ORDER BY rowid LIMIT ?',
                                        (last, CONVERT_ROWS)).fetchall()
                for row in rows:
                    updates = {}
                    for column in ('report', 'music'):
                        if row[column].startswith(MAGIC):
                            continue
                        value = unpack(row[column])
                        blob = pack(value)
                        if unpack(blob) == value:
                            updates[column] = blob
                        else:
                            log.error('Kept %s %s of rowid %s in the earlier format: conversion was not exact',
                                      table, column, row['rowid'])
                    if updates:
                        with self.con:
                            self.con.execute(f'UPDATE {table} SET {", ".join(k + "=?" for k in updates)} '
                                             'WHERE rowid=?', (*updates.values(), row['rowid']))
                    last = row['rowid']
                if len(rows) < CONVERT_ROWS:
                    if last:
                        log.info('All stored analyses in %s use the compact format', table)
                    last = None
                elif self.clock() >= deadline:
                    break
            progress[table] = last
            if last is not None:
                break
        self.instance.set_marker(CONVERTED, progress)
        return any(progress.get(table, 0) is not None for table in TABLES)

    def shrink(self):
        """Return free pages to the disk; the first time this rebuilds the file once."""
        page = self.con.execute('PRAGMA page_size').fetchone()[0]
        pages, free = (self.con.execute(f'PRAGMA {name}').fetchone()[0] for name in ('page_count', 'freelist_count'))
        if free * page < SHRINK_BYTES:
            return False
        if self.con.execute('PRAGMA auto_vacuum').fetchone()[0] == INCREMENTAL:
            # executescript steps the pragma to completion; a plain execute stops after one page on Python 3.11.
            self.con.executescript('PRAGMA incremental_vacuum;')
            self.con.execute('PRAGMA wal_checkpoint(TRUNCATE)')
            return True
        # The rebuild writes the used pages twice (copy and WAL) before the old file is released.
        needed = 2 * (pages - free) * page + RESERVE_BYTES
        if free_bytes(self.data_dir) < needed:
            return False
        log.info('Rebuilding the database to release %.1f GB', free * page / 1e9)
        self.con.execute(f'PRAGMA auto_vacuum={INCREMENTAL}')
        self.con.execute('VACUUM')
        self.con.execute('PRAGMA wal_checkpoint(TRUNCATE)')
        return True

    def clean_work(self):
        """Downloads of a batch interrupted by a restart; only the service writes there."""
        work = self.data_dir / 'work'
        for path in work.glob('*') if work.is_dir() else ():
            if path.is_dir() and len(path.name) == 32 and path.name.isalnum():
                shutil.rmtree(path, ignore_errors=True)
