"""Atomic native DuckDB schema-5 to schema-6 migration."""
from collections.abc import Sequence
import hashlib
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import duckdb

from .database import open_database
from .migrate_sqlite import _fingerprint


V5_TABLES: dict[str, tuple[str, ...]] = {
    'ledger_meta': ('singleton', 'schema_version', 'revision'),
    'import_run': ('id', 'state', 'started_us', 'finished_us', 'files_processed', 'revision', 'error_code', 'ordinal'),
    'project': ('id', 'identity', 'kind', 'label'),
    'session': ('id', 'harness', 'native_id', 'display_name', 'started_us', 'last_seen_us', 'title_excerpt', 'cwd', 'parent_locator'),
    'session_attribution': ('session_id', 'project_id', 'worktree', 'attribution_reason'),
    'source_generation': ('id', 'locator', 'generation', 'sha256', 'session_id', 'profile', 'complete_bytes', 'pending_tail', 'availability'),
    'observation': ('id', 'session_id', 'native_entry_id', 'fingerprint', 'kind', 'time_kind', 'at_us', 'start_us', 'end_us', 'time_reason', 'provider', 'model', 'stop_reason', 'tool_call_id', 'safe_facts_json'),
    'appearance': ('source_id', 'line', 'observation_id', 'ordinal'),
    'entry_edge': ('source_id', 'line', 'native_entry_id', 'parent_entry_id'),
    'token_value': ('observation_id', 'measure', 'state', 'amount', 'reason'),
    'recorded_estimate': ('observation_id', 'state', 'amount_decimal', 'currency', 'components_json', 'source_ref', 'reason'),
    'decision': ('observation_id', 'measure', 'state', 'owner_session', 'canonical', 'reason', 'rule_version'),
    'diagnostic': ('id', 'source_id', 'observation_id', 'line', 'code', 'measure'),
    'source_metadata': ('source_id', 'complete_sha256'),
    'delegation_ref': ('source_id', 'entry_id', 'kind', 'value', 'owner_path'),
    'delegation_scan': ('source_id', 'profile', 'path_hash'),
    'codex_source': ('source_id', 'parent_thread_id', 'forked_from_id', 'root_session_id'),
    'codex_evidence': ('source_id', 'line', 'observation_id', 'source_kind', 'response_id', 'thread_id', 'turn_id', 'mirror_response_id', 'cumulative_json', 'last_json', 'state', 'reason', 'ordinal'),
    'codex_title': ('session_id', 'title'),
}


def schema_version(path: Path) -> int | None:
    raw = open_database(path)
    try:
        exists = raw.execute("SELECT 1 FROM information_schema.tables WHERE table_name='ledger_meta'").fetchone()
        if not exists:
            return None
        row = raw.execute('SELECT schema_version FROM ledger_meta').fetchone()
        if row is None:
            raise ValueError('unsupported_schema')
        return int(row[0])
    finally:
        raw.close()


def _copy_table(source: duckdb.DuckDBPyConnection, target: duckdb.DuckDBPyConnection,
                table: str, columns: Sequence[str]) -> None:
    names = ','.join(columns)
    expression = 'singleton,6 AS schema_version,revision' if table == 'ledger_meta' else names
    cursor = source.execute(f'SELECT {expression} FROM {table} ORDER BY ALL')
    placeholders = ','.join('?' for _ in columns)
    while rows := cursor.fetchmany(16384):
        target.executemany(f'INSERT INTO {table}({names}) VALUES({placeholders})', rows)
    expected = _fingerprint(source.execute(f'SELECT {expression} FROM {table} ORDER BY ALL'))
    actual = _fingerprint(target.execute(f'SELECT {names} FROM {table} ORDER BY ALL'))
    if actual != expected:
        raise ValueError('migration_row_mismatch:' + table)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def migrate_duckdb_v5(path: Path) -> None:
    """Replace a closed schema-5 ledger only after a verified schema-6 copy."""
    path = path.resolve()
    if schema_version(path) != 5:
        raise ValueError('unsupported_schema')
    maximum = 0
    source = duckdb.connect(str(path), read_only=True, config={
        'autoinstall_known_extensions': False, 'autoload_known_extensions': False,
    })
    try:
        for table, column in (('diagnostic', 'id'), ('import_run', 'ordinal'),
                              ('appearance', 'ordinal'), ('codex_evidence', 'ordinal')):
            row = source.execute(f'SELECT COALESCE(MAX({column}),0) FROM {table}').fetchone()
            if row is None:
                raise ValueError('migration_row_mismatch:' + table)
            maximum = max(maximum, int(row[0]))
        with TemporaryDirectory(prefix='.ledger-v6-', dir=path.parent) as folder:
            pending = Path(folder) / path.name
            target = duckdb.connect(str(pending), config={
                'autoinstall_known_extensions': False, 'autoload_known_extensions': False,
            })
            try:
                schema = Path(__file__).with_name('schema.sql').read_text().replace('START 1;', f'START {maximum + 1};')
                target.execute('BEGIN')
                target.execute(schema)
                target.execute('DELETE FROM ledger_meta')
                for table, columns in V5_TABLES.items():
                    _copy_table(source, target, table, columns)
                target.commit()
                target.execute('CHECKPOINT')
            except BaseException:
                try:
                    target.rollback()
                except duckdb.TransactionException:
                    pass
                raise
            finally:
                target.close()
            source.close()
            with pending.open('rb') as file:
                os.fsync(file.fileno())
            digest = _sha256(path)
            backup = path.with_name(f'{path.stem}.v5-backup-{digest}.duckdb')
            if backup.exists():
                if _sha256(backup) != digest:
                    raise FileExistsError(backup)
            else:
                os.link(path, backup)
            os.replace(pending, path)
            _fsync_directory(path.parent)
    finally:
        try:
            source.close()
        except duckdb.ConnectionException:
            pass
