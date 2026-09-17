"""Build genuine legacy SQLite fixtures without a runtime SQLite backend."""
from pathlib import Path
import sqlite3


def export_legacy(store, path):
    with sqlite3.connect(path) as old, store.connect() as new:
        old.executescript(Path('src/harness_usage/legacy_schema.sql').read_text())
        old.execute('DELETE FROM ledger_meta')
        tables = [row[0] for row in old.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY rowid")]
        for table in tables:
            names = [row[1] for row in old.execute(f'PRAGMA table_info({table})')]
            columns = ','.join(names)
            origin = 'session_view' if table == 'session' else table
            ordered = ' ORDER BY ordinal' if table in ('import_run','appearance','codex_evidence') else ''
            rows = [tuple(row) for row in new.execute(f'SELECT {columns} FROM {origin}' + ordered)]
            if table == 'ledger_meta':
                rows = [(singleton,4,revision) for singleton,_,revision in rows]
            old.executemany(f'INSERT INTO {table}({columns}) VALUES({",".join("?" for _ in names)})', rows)
