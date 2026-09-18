"""Read-only, schema-qualified snapshots of Copilot CLI's session store."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, fields
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
import json
from hashlib import sha256
import os
from pathlib import Path
import sqlite3
import stat
import tempfile
from typing import Iterator
from uuid import UUID


PROFILE = "copilot-cli-session-store/schema-8-shape-1"
_LIMIT = 2**63
_MAX_TEXT = 256
_MAX_CWD = 4096
_MAX_TIMESTAMP = 64
_MAX_DECIMAL_DIGITS = 128
_MAX_DECIMAL_EXPONENT = 128
_MAX_DECIMAL_TEXT = 256
_MAX_TOKEN_DETAILS_BYTES = 64 * 1024
_MAX_TOKEN_DETAILS_ITEMS = 64
_TOKEN_TYPES = frozenset(("input", "output", "cache_read", "cache_write"))


@dataclass(frozen=True, slots=True)
class CopilotStoreSession:
    session_id: str
    cwd: str | None
    created_at: datetime | None
    updated_at: datetime | None
    host_type: str | None


@dataclass(frozen=True, slots=True)
class CopilotStoreCall:
    row_id: int
    session_id: str
    turn_index: int | None
    created_at: datetime | None
    model: str | None
    agent_id: str | None
    parent_tool_call_id: str | None
    input_tokens: int | None
    output_tokens: int | None
    cache_read_tokens: int | None
    cache_write_tokens: int | None
    reasoning_tokens: int | None
    total_nano_aiu: Decimal | None
    request_multiplier: Decimal | None
    token_details_json: str | None
    invalid_fields: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CopilotStoreSnapshot:
    locator: str
    fingerprint: str
    sessions: tuple[CopilotStoreSession, ...]
    calls: tuple[CopilotStoreCall, ...]
    profile: str = PROFILE
    schema_version: int = 8


class _InvalidStore(ValueError):
    """Internal marker for a content-free qualification failure."""


def _reject(reason: str = "unqualified_session_store") -> _InvalidStore:
    return _InvalidStore(reason)


def _as_uri(path: Path, query: str) -> str:
    return f"{path.resolve(strict=True).as_uri()}?{query}"


def _open_source(path: Path) -> sqlite3.Connection:
    try:
        # A private page cache lets SQLite read the staged WAL without sharing
        # read marks with the source database.
        uri = _as_uri(path, "mode=ro&cache=private")
        connection = sqlite3.connect(uri, uri=True, isolation_level=None)
    except (OSError, sqlite3.Error):
        raise OSError("session_store_unreadable") from None
    try:
        connection.execute("PRAGMA query_only=ON")
    except sqlite3.Error:
        connection.close()
        raise ValueError("session_store_unreadable") from None
    return connection


def _sqlite_affinity(declared: str) -> str:
    value = declared.upper()
    if "INT" in value:
        return "INTEGER"
    if any(token in value for token in ("CHAR", "CLOB", "TEXT")):
        return "TEXT"
    if "BLOB" in value or not value:
        return "BLOB"
    if any(token in value for token in ("REAL", "FLOA", "DOUB")):
        return "REAL"
    return "NUMERIC"


def _table_info(connection: sqlite3.Connection, table: str) -> dict[str, tuple[str, int, int]]:
    try:
        rows = tuple(connection.execute(f"PRAGMA table_info({table})"))
    except sqlite3.Error:
        raise _reject("invalid_session_store_schema") from None
    if not rows:
        raise _reject("invalid_session_store_schema")
    result: dict[str, tuple[str, int, int]] = {}
    for row in rows:
        if len(row) != 6 or not isinstance(row[1], str):
            raise _reject("invalid_session_store_schema")
        name = row[1]
        if name in result:
            raise _reject("invalid_session_store_schema")
        declared = row[2] if isinstance(row[2], str) else ""
        not_null = row[3] if type(row[3]) is int else -1
        primary_key = row[5] if type(row[5]) is int else -1
        result[name] = (_sqlite_affinity(declared), not_null, primary_key)
    return result


def _require_columns(
    connection: sqlite3.Connection,
    table: str,
    required: dict[str, tuple[str, bool, bool]],
) -> None:
    columns = _table_info(connection, table)
    for name, (affinity, not_null, primary_key) in required.items():
        actual = columns.get(name)
        if actual is None or actual[0] != affinity:
            raise _reject("invalid_session_store_schema")
        if not_null and actual[1] != 1:
            raise _reject("invalid_session_store_schema")
        if primary_key and actual[2] != 1:
            raise _reject("invalid_session_store_schema")


def _qualify_schema(connection: sqlite3.Connection) -> None:
    _require_columns(connection, "schema_version", {"version": ("INTEGER", True, False)})
    _require_columns(
        connection,
        "sessions",
        {
            "id": ("TEXT", False, True),
            "cwd": ("TEXT", False, False),
            "created_at": ("TEXT", False, False),
            "updated_at": ("TEXT", False, False),
            "host_type": ("TEXT", False, False),
        },
    )
    _require_columns(
        connection,
        "assistant_usage_events",
        {
            "id": ("INTEGER", False, True),
            "session_id": ("TEXT", True, False),
            "model": ("TEXT", True, False),
            "turn_index": ("INTEGER", False, False),
            "input_tokens": ("INTEGER", False, False),
            "output_tokens": ("INTEGER", False, False),
            "cache_read_tokens": ("INTEGER", False, False),
            "cache_write_tokens": ("INTEGER", False, False),
            "reasoning_tokens": ("INTEGER", False, False),
            "total_nano_aiu": ("INTEGER", False, False),
            "request_multiplier": ("REAL", False, False),
            "agent_id": ("TEXT", False, False),
            "parent_tool_call_id": ("TEXT", False, False),
            "initiator": ("TEXT", False, False),
            "token_details_json": ("TEXT", False, False),
            "created_at": ("TEXT", False, False),
        },
    )
    try:
        versions = tuple(connection.execute("SELECT version FROM schema_version"))
    except sqlite3.Error:
        raise _reject("invalid_session_store_schema") from None
    if len(versions) != 1 or type(versions[0][0]) is not int or versions[0][0] != 8:
        raise _reject("unsupported_session_store_schema")


def _quick_check(connection: sqlite3.Connection) -> None:
    try:
        result = tuple(connection.execute("PRAGMA quick_check"))
    except sqlite3.Error:
        raise _reject("invalid_session_store") from None
    if result != (("ok",),):
        raise _reject("invalid_session_store")


def _source_files(path: Path, max_bytes: int | None) -> tuple[Path, tuple[Path, ...]]:
    if max_bytes is not None and (type(max_bytes) is not int or max_bytes < 0):
        raise ValueError("invalid_session_store_limit")
    try:
        for component in (path, *path.parents):
            if component.is_symlink():
                raise _reject("symlink_session_store")
        resolved = path.resolve(strict=True)
    except OSError:
        raise OSError("session_store_unreadable") from None
    files = [resolved]
    for suffix in ("-wal", "-shm"):
        sidecar = resolved.with_name(resolved.name + suffix)
        try:
            if sidecar.is_symlink():
                raise _reject("symlink_session_store")
            if sidecar.exists():
                files.append(sidecar)
        except OSError:
            raise OSError("session_store_unreadable") from None
    for file in files:
        try:
            if not stat.S_ISREG(os.stat(file, follow_symlinks=False).st_mode):
                raise _reject("invalid_session_store_file")
        except OSError:
            raise OSError("session_store_unreadable") from None
    if max_bytes is not None:
        try:
            if sum(file.stat().st_size for file in files) > max_bytes:
                raise _reject("session_store_too_large")
        except OSError:
            raise OSError("session_store_unreadable") from None
    return resolved, tuple(files)


def _copy_regular(source: Path, destination: Path, remaining: int | None) -> int:
    total = 0
    try:
        descriptor = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(descriptor, "rb") as source_file, destination.open("wb") as destination_file:
            while True:
                chunk = source_file.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if remaining is not None and total > remaining:
                    raise _reject("session_store_too_large")
                destination_file.write(chunk)
    except _InvalidStore:
        raise
    except OSError:
        raise OSError("session_store_unreadable") from None
    return total


def _file_signature(files: tuple[Path, ...]) -> tuple[tuple[str, int, int], ...]:
    try:
        return tuple((str(file), file.stat().st_size, file.stat().st_mtime_ns) for file in files)
    except OSError:
        raise OSError("session_store_unreadable") from None


def _open_snapshot(path: Path, max_bytes: int | None) -> tuple[tempfile.TemporaryDirectory[str], sqlite3.Connection]:
    temp_dir = tempfile.TemporaryDirectory(prefix="harness-usage-copilot-store-")
    source: sqlite3.Connection | None = None
    destination: sqlite3.Connection | None = None
    connection: sqlite3.Connection | None = None
    try:
        # SQLite's backup API updates WAL shared-memory read marks.  Copy the
        # complete main/WAL/SHM set before any SQLite read, then run backup on
        # that private copy so the user's files are never opened or modified.
        resolved, _ = _source_files(path, max_bytes)
        source_path = Path(temp_dir.name) / "source.db"
        for _attempt in range(3):
            before_files = _source_files(path, max_bytes)[1]
            before_signature = _file_signature(before_files)
            for suffix in ("-wal", "-shm"):
                (source_path.with_name(source_path.name + suffix)).unlink(missing_ok=True)
            copied = 0
            for index, file in enumerate(before_files):
                target = source_path if index == 0 else source_path.with_name(
                    source_path.name + file.name.removeprefix(resolved.name)
                )
                copied += _copy_regular(file, target, None if max_bytes is None else max_bytes - copied)
            after_files = _source_files(path, max_bytes)[1]
            if before_files == after_files and before_signature == _file_signature(after_files):
                break
        else:
            raise _reject("source_changed_during_snapshot")
        source = _open_source(source_path)
        backup_path = Path(temp_dir.name) / "snapshot.db"
        destination = sqlite3.connect(backup_path, isolation_level=None)
        source.backup(destination)
        destination.close()
        destination = None
        source.close()
        source = None
        connection = sqlite3.connect(
            _as_uri(backup_path, "mode=ro&immutable=1"),
            uri=True,
            isolation_level=None,
        )
        connection.execute("PRAGMA query_only=ON")
        _quick_check(connection)
        _qualify_schema(connection)
        return temp_dir, connection
    except _InvalidStore:
        if destination is not None:
            destination.close()
        if source is not None:
            source.close()
        if connection is not None:
            connection.close()
        temp_dir.cleanup()
        raise
    except (OSError, sqlite3.Error):
        if destination is not None:
            destination.close()
        if source is not None:
            source.close()
        if connection is not None:
            connection.close()
        temp_dir.cleanup()
        raise ValueError("unreadable_session_store") from None


@contextmanager
def session_store_connection(path: Path, *, max_bytes: int | None = None) -> Iterator[sqlite3.Connection]:
    """Yield a validated immutable snapshot connection, never the source DB."""
    temp_dir, connection = _open_snapshot(path, max_bytes)
    try:
        yield connection
    finally:
        connection.close()
        temp_dir.cleanup()


def _uuid4(value: object) -> str | None:
    if not isinstance(value, str) or len(value) != 36:
        return None
    try:
        parsed = UUID(value)
    except (ValueError, AttributeError):
        return None
    return value if parsed.version == 4 and str(parsed) == value else None


def _text(value: object, *, limit: int = _MAX_TEXT) -> str | None:
    return value if isinstance(value, str) and value.strip() and len(value) <= limit else None


def _cwd(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip() or len(value) > _MAX_CWD or not Path(value).is_absolute():
        return None
    return value


def _timestamp(value: object, *, reject_invalid: bool) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, str) or len(value) > _MAX_TIMESTAMP:
        if reject_invalid:
            raise _reject("invalid_call_timestamp")
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError
        return parsed.astimezone(UTC)
    except (TypeError, ValueError, OverflowError, OSError):
        if reject_invalid:
            raise _reject("invalid_call_timestamp") from None
        return None


def _bounded_int(value: object) -> tuple[int | None, bool]:
    if value is None:
        return None, False
    if type(value) is int and 0 <= value < _LIMIT:
        return value, False
    return None, True


def _bounded_decimal(value: object) -> Decimal | None:
    if value is None:
        return None
    try:
        result = Decimal(value if isinstance(value, str) else str(value))
    except (InvalidOperation, ValueError, TypeError):
        return None
    if not result.is_finite() or not 0 <= result < _LIMIT:
        return None
    _, digits, exponent = result.as_tuple()
    if len(digits) > _MAX_DECIMAL_DIGITS or abs(int(exponent)) > _MAX_DECIMAL_EXPONENT:
        return None
    try:
        if len(format(result, "f")) > _MAX_DECIMAL_TEXT:
            return None
    except (ValueError, OverflowError):
        return None
    return result


def _json_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_json_key")
        result[key] = value
    return result


def _token_details(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    if len(value.encode("utf-8", "strict")) > _MAX_TOKEN_DETAILS_BYTES:
        return None
    try:
        parsed = json.loads(
            value,
            parse_int=int,
            parse_float=Decimal,
            parse_constant=lambda _value: (_ for _ in ()).throw(ValueError("constant")),
            object_pairs_hook=_json_pairs,
        )
    except (ValueError, TypeError, UnicodeError, RecursionError):
        return None
    if not isinstance(parsed, list) or len(parsed) > _MAX_TOKEN_DETAILS_ITEMS:
        return None
    canonical: list[dict[str, object]] = []
    for raw in parsed:
        if not isinstance(raw, dict):
            return None
        if set(raw) - {"tokenType", "tokenCount", "batchSize", "costPerBatch", "model"}:
            return None
        if set(raw) != {"tokenType", "tokenCount", "batchSize", "costPerBatch"} and set(raw) != {
            "tokenType", "tokenCount", "batchSize", "costPerBatch", "model"
        }:
            return None
        token_type = raw.get("tokenType")
        token_count, bad_count = _bounded_int(raw.get("tokenCount"))
        batch_size, bad_batch = _bounded_int(raw.get("batchSize"))
        cost_per_batch, bad_cost = _bounded_int(raw.get("costPerBatch"))
        if not isinstance(token_type, str) or token_type not in _TOKEN_TYPES or bad_count or bad_batch or bad_cost:
            return None
        if batch_size is None or batch_size < 1 or token_count is None or cost_per_batch is None:
            return None
        model: str | None = None
        if "model" in raw:
            model = _text(raw["model"])
            if model is None:
                return None
        item: dict[str, object] = {
            "batchSize": batch_size,
            "costPerBatch": cost_per_batch,
            "tokenCount": token_count,
            "tokenType": token_type,
        }
        if "model" in raw:
            item["model"] = model
        canonical.append(item)
    return json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _session_records(connection: sqlite3.Connection) -> tuple[CopilotStoreSession, ...]:
    try:
        rows = tuple(connection.execute(
            "SELECT id,cwd,created_at,updated_at,host_type FROM sessions ORDER BY id"
        ))
    except sqlite3.Error:
        raise _reject("invalid_session_store") from None
    sessions: list[CopilotStoreSession] = []
    seen: set[str] = set()
    for row in rows:
        if len(row) != 5:
            raise _reject("invalid_session_store")
        session_id = _uuid4(row[0])
        if session_id is None or session_id in seen:
            raise _reject("invalid_session_identity")
        created = _timestamp(row[2], reject_invalid=False)
        updated = _timestamp(row[3], reject_invalid=False)
        if row[2] is not None and created is None or row[3] is not None and updated is None:
            raise _reject("invalid_session_metadata")
        cwd = _cwd(row[1]) if row[1] is not None else None
        host_type = _text(row[4]) if row[4] is not None else None
        sessions.append(CopilotStoreSession(session_id, cwd, created, updated, host_type))
        seen.add(session_id)
    return tuple(sessions)


def _identity_field(value: object) -> str | None:
    if value is None:
        return None
    value_text = _text(value)
    if value_text is None:
        raise _reject("invalid_call_identity")
    return value_text


def _call_records(
    connection: sqlite3.Connection,
    session_ids: set[str],
) -> tuple[CopilotStoreCall, ...]:
    try:
        rows = tuple(connection.execute(
            "SELECT id,session_id,model,turn_index,input_tokens,output_tokens,"
            "cache_read_tokens,cache_write_tokens,reasoning_tokens,total_nano_aiu,"
            "CAST(request_multiplier AS TEXT),agent_id,parent_tool_call_id,initiator,"
            "token_details_json,created_at FROM assistant_usage_events ORDER BY id"
        ))
    except sqlite3.Error:
        raise _reject("invalid_session_store") from None
    calls: list[CopilotStoreCall] = []
    for row in rows:
        if len(row) != 16 or type(row[0]) is not int or row[0] < 0:
            raise _reject("invalid_call_identity")
        session_id = _uuid4(row[1])
        if session_id is None or session_id not in session_ids:
            raise _reject("dangling_session")
        model = _text(row[2])
        if model is None:
            raise _reject("invalid_model_identity")
        turn_index, invalid_turn = _bounded_int(row[3])
        created_at = _timestamp(row[15], reject_invalid=True)
        agent_id = _identity_field(row[11])
        parent_tool_call_id = _identity_field(row[12])
        _identity_field(row[13])
        invalid: list[str] = ["turn_index"] if invalid_turn else []
        values: list[int | None] = []
        for name, raw in zip(
            ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens", "reasoning_tokens"),
            row[4:9],
            strict=True,
        ):
            value, bad = _bounded_int(raw)
            values.append(value)
            if bad:
                invalid.append(name)
        input_tokens, output_tokens, cache_read_tokens, cache_write_tokens, reasoning_tokens = values
        if reasoning_tokens is not None and output_tokens is not None and reasoning_tokens > output_tokens:
            reasoning_tokens = None
            invalid.append("reasoning_tokens")
        total_nano_aiu, bad_total = _bounded_int(row[9])
        if bad_total:
            invalid.append("total_nano_aiu")
        multiplier = _bounded_decimal(row[10])
        if row[10] is not None and multiplier is None:
            invalid.append("request_multiplier")
        token_details = _token_details(row[14])
        if row[14] is not None and token_details is None:
            invalid.append("token_details")
        calls.append(CopilotStoreCall(
            row[0], session_id, turn_index, created_at, model, agent_id, parent_tool_call_id,
            input_tokens, output_tokens, cache_read_tokens, cache_write_tokens, reasoning_tokens,
            Decimal(total_nano_aiu) if total_nano_aiu is not None else None,
            multiplier, token_details, tuple(dict.fromkeys(invalid)),
        ))
    return tuple(calls)


def _fingerprint(
    sessions: tuple[CopilotStoreSession, ...],
    calls: tuple[CopilotStoreCall, ...],
) -> str:
    def encode(value: object) -> object:
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, Decimal):
            return str(value)
        if isinstance(value, tuple):
            return [encode(item) for item in value]
        if isinstance(value, (CopilotStoreSession, CopilotStoreCall)):
            return {
                field.name: encode(getattr(value, field.name))
                for field in fields(value)
            }
        return value

    payload = {
        "profile": PROFILE,
        "schema_version": 8,
        "sessions": [encode(item) for item in sessions],
        "calls": [encode(item) for item in calls],
    }
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_session_store(path: Path) -> CopilotStoreSnapshot:
    """Read one complete schema-8 snapshot without modifying ``path``."""
    with session_store_connection(path) as connection:
        sessions = _session_records(connection)
        calls = _call_records(connection, {session.session_id for session in sessions})
    return CopilotStoreSnapshot(str(path), _fingerprint(sessions, calls), sessions, calls)


__all__ = [
    "PROFILE",
    "CopilotStoreCall",
    "CopilotStoreSession",
    "CopilotStoreSnapshot",
    "read_session_store",
    "session_store_connection",
]
