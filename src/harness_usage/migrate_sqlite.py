"""One-time legacy import. SQLite is never consulted after DuckDB publication."""
from pathlib import Path
from contextlib import closing
import hashlib
import json
import os
import re
import sqlite3
from tempfile import TemporaryDirectory

from .database import Connection, open_database


def _upgrade(db: sqlite3.Connection) -> None:
    version = db.execute('SELECT schema_version FROM ledger_meta').fetchone()[0]
    if version not in (1, 2, 3, 4):
        raise ValueError('unsupported_legacy_schema')
    if version == 4:
        return
    schema = Path(__file__).with_name('legacy_schema.sql').read_text()
    db.execute('PRAGMA foreign_keys=OFF')
    db.execute('BEGIN')
    try:
        if version < 3:
            db.execute('ALTER TABLE session ADD COLUMN title_excerpt TEXT CHECK(title_excerpt IS NULL OR length(title_excerpt) BETWEEN 1 AND 160)')
        session_sql = schema.split('CREATE TABLE session (', 1)[1].split(';', 1)[0]
        db.execute('CREATE TABLE session_v4 (' + session_sql)
        columns = ','.join(row[1] for row in db.execute('PRAGMA table_info(session)'))
        db.execute(f'INSERT INTO session_v4({columns}) SELECT {columns} FROM session')
        db.execute('DROP TABLE session')
        db.execute('ALTER TABLE session_v4 RENAME TO session')
        db.execute('CREATE TABLE meta_v4(singleton INTEGER PRIMARY KEY CHECK(singleton=1),schema_version INTEGER NOT NULL CHECK(schema_version=4),revision INTEGER NOT NULL CHECK(revision>=0)) STRICT')
        db.execute('INSERT INTO meta_v4 SELECT singleton,4,revision FROM ledger_meta')
        db.execute('DROP TABLE ledger_meta')
        db.execute('ALTER TABLE meta_v4 RENAME TO ledger_meta')
        for statement in schema.split('-- Delegation ownership metadata, independent of accounting observations.', 1)[1].split(';'):
            if statement.strip():
                db.execute(statement.replace('CREATE TABLE ', 'CREATE TABLE IF NOT EXISTS ').replace('CREATE INDEX ', 'CREATE INDEX IF NOT EXISTS '))
        db.commit()
    except BaseException:
        db.rollback()
        raise


def _fingerprint(cursor: object) -> tuple[int, str]:
    # Both native cursors support fetchmany; keep the full-ledger check bounded.
    count = 0
    digest = hashlib.sha256()
    while rows := cursor.fetchmany(16384):  # type: ignore[attr-defined]
        for row in rows:
            digest.update(json.dumps(tuple(row), ensure_ascii=True, separators=(',', ':')).encode())
            digest.update(b'\n')
        count += len(rows)
    return count, digest.hexdigest()


def _copy(source: sqlite3.Connection, target: Path) -> None:
    schema = Path(__file__).with_name('schema.sql').read_text()
    maximum = max(source.execute(f'SELECT COALESCE(MAX(rowid),0) FROM {table}').fetchone()[0]
                  for table in ('diagnostic', 'import_run', 'appearance', 'codex_evidence'))
    raw = open_database(target)
    db = Connection(raw)
    try:
        db.execute('BEGIN')
        raw.execute(schema.replace('START 1;', f'START {maximum + 1};'))
        db.execute('DELETE FROM ledger_meta')
        source_tables = {row[0] for row in source.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        for table in re.findall(r'CREATE TABLE (\w+) \(', schema):
            if table not in source_tables and table != 'session_attribution':
                continue
            names = [row[0] for row in raw.execute(f'DESCRIBE {table}').fetchall()]
            expressions = [('6 AS schema_version' if table == 'ledger_meta' and name == 'schema_version' else
                            'rowid AS ordinal' if name == 'ordinal' else
                            'id AS session_id' if table == 'session_attribution' and name == 'session_id' else name)
                           for name in names]
            origin = 'session' if table == 'session_attribution' else table
            select = f'SELECT {",".join(expressions)} FROM {origin}'
            # Explicit ordinals preserve legacy ordering even across checkpoint/restart.
            cursor = source.execute(select + (' ORDER BY rowid' if 'ordinal' in names else ''))
            while rows := cursor.fetchmany(16384):
                db.executemany(f'INSERT INTO {table}({",".join(names)}) VALUES({",".join("?" for _ in names)})', rows)
            expected = _fingerprint(source.execute(select + ' ORDER BY ' + ','.join(names)))
            actual = _fingerprint(raw.execute(f'SELECT {",".join(names)} FROM {table} ORDER BY ALL'))
            if actual != expected:
                raise ValueError('migration_row_mismatch:' + table)
        if db.execute('SELECT count(*) FROM session s LEFT JOIN session_attribution a ON a.session_id=s.id WHERE a.session_id IS NULL').one()[0]:
            raise ValueError('migration_missing_attribution')
        db.commit()
        raw.execute('CHECKPOINT')
    finally:
        raw.close()


def migrate_sqlite(source: Path, target: Path) -> None:
    """Publish one verified native ledger; preserve the original for recovery."""
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.ledger-migration-', dir=target.parent) as folder:
        temporary = Path(folder)
        snapshot = temporary / 'legacy.sqlite3'
        with closing(sqlite3.connect(source.resolve().as_uri() + '?mode=ro', uri=True)) as original, closing(sqlite3.connect(snapshot)) as copy:
            original.backup(copy)
            if copy.execute('PRAGMA integrity_check').fetchall() != [('ok',)]:
                raise ValueError('invalid_legacy_database')
            _upgrade(copy)
            if copy.execute('PRAGMA foreign_key_check').fetchone() is not None:
                raise ValueError('invalid_legacy_references')
            pending = temporary / 'ledger.duckdb'
            _copy(copy, pending)
        with pending.open('rb') as file:
            os.fsync(file.fileno())
        # A same-filesystem hard link publishes atomically without overwriting an
        # existing ledger if another migration won the race.
        os.link(pending, target)
        descriptor = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
