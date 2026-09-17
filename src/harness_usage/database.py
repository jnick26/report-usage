"""Native DuckDB transactions, named rows and typed batch transfer."""
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
import re
from threading import RLock
from typing import Any, SupportsIndex, overload

import duckdb
import pyarrow as pa  # type: ignore[import-untyped]


_PARAMETER_INSERT = re.compile(
    r'INSERT[ \t\r\n]+(?:(?P<ignore>OR[ \t\r\n]+IGNORE)[ \t\r\n]+)?'
    r'INTO[ \t\r\n]+(?P<table>\w+)[ \t\r\n]*'
    r'(?:\([ \t\r\n]*(?P<columns>\w+(?:[ \t\r\n]*,[ \t\r\n]*\w+)*)[ \t\r\n]*\))?'
    r'[ \t\r\n]+VALUES[ \t\r\n]*\([ \t\r\n]*\?(?:[ \t\r\n]*,[ \t\r\n]*\?)*[ \t\r\n]*\)'
)


def _parameter_insert(sql: str) -> tuple[str, tuple[str, ...] | None, bool] | None:
    match = _PARAMETER_INSERT.fullmatch(sql)
    if match is None:
        return None
    column_list = match.group('columns')
    columns = tuple(name.strip() for name in column_list.split(',')) if column_list else None
    return match.group('table'), columns, match.group('ignore') is not None


@lru_cache(maxsize=None)
def writer_lock(path: Path) -> RLock:
    # ponytail: one writer per local ledger; no multi-process writer protocol.
    return RLock()


def open_database(path: Path) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(path), config={'threads': 4, 'memory_limit': '2GB',
        'max_temp_directory_size': '2GB', 'temp_directory': str(path.with_suffix('.spill')),
        'autoinstall_known_extensions': False, 'autoload_known_extensions': False})


class Row(tuple[Any, ...]):
    names: list[str]

    def __new__(cls, values: Sequence[Any], names: list[str]) -> 'Row':
        row = super().__new__(cls, values)
        row.names = names
        return row

    @overload
    def __getitem__(self, key: SupportsIndex | str) -> Any: ...
    @overload
    def __getitem__(self, key: slice) -> tuple[Any, ...]: ...
    def __getitem__(self, key: SupportsIndex | str | slice) -> Any:
        return super().__getitem__(self.names.index(key) if isinstance(key, str) else key)


class Cursor:
    def __init__(self, rows: Sequence[Sequence[Any]] = (), names: list[str] | None = None, rowcount: int = -1):
        self.rows = [Row(row, names or []) for row in rows]
        self.position = 0
        self.rowcount = rowcount

    def fetchone(self) -> Row | None:
        if self.position == len(self.rows):
            return None
        row = self.rows[self.position]
        self.position += 1
        return row

    def fetchall(self) -> list[Row]:
        rows = self.rows[self.position:]
        self.position = len(self.rows)
        return rows

    def one(self) -> Row:
        row = self.fetchone()
        if row is None:
            raise ValueError('expected_database_row')
        return row

    def __iter__(self) -> Iterator[Row]:
        while (row := self.fetchone()) is not None:
            yield row


class Connection:
    def __init__(self, raw: duckdb.DuckDBPyConnection):
        self.raw = raw
        self.in_transaction = False
        self.total_changes = 0
        self.trace: Callable[[str], None] | None = None
        self.pending: dict[str, list[Sequence[Any]]] | None = None

    def set_trace_callback(self, callback: Callable[[str], None]) -> None:
        self.trace = callback

    def execute(self, sql: str, parameters: Sequence[Any] = ()) -> Cursor:
        if self.trace:
            self.trace(sql)
        if sql in ('BEGIN', 'BEGIN IMMEDIATE'):
            if not self.in_transaction:
                self.raw.execute('BEGIN')
                self.in_transaction = True
            return Cursor()
        insert = _parameter_insert(sql)
        if self.pending is not None and insert is not None:
            self.pending.setdefault(sql, []).append(parameters)
            return Cursor()
        if insert is not None:
            table, columns, _ = insert
            schema = {row[0]: row[1] for row in self.raw.execute(f'DESCRIBE {table}').fetchall()}
            insert_names = columns or tuple(schema)
            if any(schema[name] == 'BIGINT' and value is not None and type(value) is not int
                   for name, value in zip(insert_names, parameters, strict=True)):
                raise ValueError('invalid_database_integer')
        result = self.raw.execute(sql, parameters)
        names = [column[0] for column in result.description or ()]
        rows = result.fetchall()
        count = int(rows[0][0]) if names == ['Count'] and rows else -1
        if count >= 0 and sql.lstrip().upper().startswith(('INSERT', 'UPDATE', 'DELETE')):
            self.total_changes += count
        return Cursor(rows, names, count)

    def executemany(self, sql: str, parameters: Iterable[Sequence[Any]]) -> None:
        rows = list(parameters)
        if not rows:
            return
        insert = _parameter_insert(sql)
        if insert is None:
            for row in rows:
                self.execute(sql, row)
            return
        table, columns, ignore = insert
        schema = {row[0]: row[1] for row in self.raw.execute(f'DESCRIBE {table}').fetchall()}
        names = columns or tuple(schema)
        for start in range(0, len(rows), 16384):
            chunk = rows[start:start + 16384]
            for i, name in enumerate(names):
                if schema[name] == 'BIGINT' and any(row[i] is not None and type(row[i]) is not int for row in chunk):
                    raise ValueError('invalid_database_integer')
            arrays = [pa.array([row[i] for row in chunk], type=pa.int64() if schema[name] == 'BIGINT' else pa.string()) for i, name in enumerate(names)]
            self.raw.register('_ledger_batch', pa.Table.from_arrays(arrays, names=names))
            try:
                self.execute(f'INSERT INTO {table} ({",".join(names)}) SELECT * FROM _ledger_batch' + (' ON CONFLICT DO NOTHING' if ignore else ''))
            finally:
                self.raw.unregister('_ledger_batch')

    @contextmanager
    def batch(self) -> Iterator[None]:
        self.pending = {}
        try:
            yield
            pending, self.pending = self.pending, None
            for sql, rows in pending.items():
                self.executemany(sql, rows)
        finally:
            self.pending = None

    def commit(self) -> None:
        self.raw.commit()
        self.in_transaction = False

    def rollback(self) -> None:
        self.raw.rollback()
        self.in_transaction = False
