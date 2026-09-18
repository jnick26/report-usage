from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
import hashlib
import sqlite3
from pathlib import Path

import pytest

from harness_usage.copilot_store_reader import (
    CopilotStoreCall,
    CopilotStoreSession,
    CopilotStoreSnapshot,
    read_session_store,
    session_store_connection,
)


SESSION = "11111111-1111-4111-8111-111111111111"
START = "2026-09-16T00:00:00Z"


def make_store(path: Path, *, wal: bool = False) -> sqlite3.Connection:
    db = sqlite3.connect(path)
    if wal:
        assert db.execute("PRAGMA journal_mode=WAL").fetchone() == ("wal",)
    db.executescript(
        """
        CREATE TABLE schema_version (version INTEGER NOT NULL);
        INSERT INTO schema_version VALUES (8);
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY,
            cwd TEXT,
            repository TEXT,
            branch TEXT,
            summary TEXT,
            created_at TEXT,
            updated_at TEXT,
            host_type TEXT
        );
        CREATE TABLE assistant_usage_events (
            id INTEGER PRIMARY KEY,
            session_id TEXT NOT NULL,
            model TEXT NOT NULL,
            turn_index INTEGER,
            input_tokens INTEGER,
            output_tokens INTEGER,
            cache_read_tokens INTEGER,
            cache_write_tokens INTEGER,
            reasoning_tokens INTEGER,
            total_nano_aiu INTEGER,
            request_multiplier REAL,
            agent_id TEXT,
            parent_tool_call_id TEXT,
            initiator TEXT,
            token_details_json TEXT,
            created_at TEXT
        );
        CREATE TABLE turns (
            id INTEGER PRIMARY KEY,
            session_id TEXT,
            turn_index INTEGER,
            user_text TEXT,
            assistant_text TEXT
        );
        """
    )
    db.execute(
        "INSERT INTO sessions VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (SESSION, "/workspace", "/workspace", "main", "private", START, START, "cli"),
    )
    db.commit()
    return db


def add_call(db: sqlite3.Connection, row_id: int, **overrides: object) -> None:
    values: dict[str, object] = {
        "id": row_id,
        "session_id": SESSION,
        "model": "claude-ignored",
        "turn_index": 1,
        "input_tokens": 10,
        "output_tokens": 20,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "reasoning_tokens": 5,
        "total_nano_aiu": 30,
        "request_multiplier": "1.25",
        "agent_id": "agent-a",
        "parent_tool_call_id": "tool-a",
        "initiator": "user",
        "token_details_json": '[{"tokenType":"input","tokenCount":10,"batchSize":1,"costPerBatch":2}]',
        "created_at": "2026-09-16T00:00:01.000Z",
    }
    values.update(overrides)
    names = tuple(values)
    db.execute(
        f"INSERT INTO assistant_usage_events ({','.join(names)}) VALUES ({','.join('?' for _ in names)})",
        tuple(values[name] for name in names),
    )
    db.commit()


def test_wal_snapshot_includes_committed_row_without_mutating_source(tmp_path: Path) -> None:
    path = tmp_path / "session-store.db"
    db = make_store(path, wal=True)
    add_call(db, 1)
    add_call(db, 2, output_tokens=0)
    source_files = tuple(path.parent / name for name in (path.name, path.name + "-wal", path.name + "-shm"))
    before = {file: (file.read_bytes(), file.stat().st_mtime_ns) for file in source_files if file.exists()}

    snapshot = read_session_store(path)

    assert {call.row_id for call in snapshot.calls} == {1, 2}
    assert {file: (file.read_bytes(), file.stat().st_mtime_ns) for file in before} == before
    db.close()


def test_reader_projects_bounded_fields_and_distinguishes_invalid_from_absent(tmp_path: Path) -> None:
    path = tmp_path / "session-store.db"
    db = make_store(path)
    add_call(db, 1)
    add_call(
        db,
        2,
        turn_index=None,
        input_tokens=-1,
        output_tokens=10,
        cache_read_tokens=0,
        cache_write_tokens=0,
        reasoning_tokens=99,
        total_nano_aiu=-1,
        request_multiplier="not-a-number",
        token_details_json='[{"tokenType":[],"tokenCount":1,"batchSize":1,"costPerBatch":1}]',
    )
    db.close()

    snapshot = read_session_store(path)
    first, second = snapshot.calls
    assert first.input_tokens == 10
    assert first.output_tokens == 20
    assert first.total_nano_aiu == Decimal(30)
    assert first.request_multiplier == Decimal("1.25")
    assert first.token_details_json == '[{"batchSize":1,"costPerBatch":2,"tokenCount":10,"tokenType":"input"}]'
    assert second.input_tokens is None
    assert second.output_tokens == 10
    assert second.cache_read_tokens == 0
    assert second.turn_index is None
    assert second.total_nano_aiu is None
    assert second.request_multiplier is None
    assert second.token_details_json is None
    assert set(second.invalid_fields) >= {
        "input_tokens",
        "reasoning_tokens",
        "total_nano_aiu",
        "request_multiplier",
        "token_details",
    }


def test_reader_rejects_schema_identity_and_dangling_rows(tmp_path: Path) -> None:
    path = tmp_path / "session-store.db"
    db = make_store(path)
    db.execute("UPDATE schema_version SET version=7")
    db.commit()
    db.close()
    with pytest.raises(ValueError):
        read_session_store(path)

    path = tmp_path / "dangling.db"
    db = make_store(path)
    add_call(db, 1, session_id="22222222-2222-4222-8222-222222222222")
    db.close()
    with pytest.raises(ValueError):
        read_session_store(path)

    path = tmp_path / "bad-session.db"
    db = make_store(path)
    db.execute("UPDATE sessions SET id='not-a-uuid'")
    db.commit()
    db.close()
    with pytest.raises(ValueError):
        read_session_store(path)


def test_fingerprint_is_repeatable_and_changes_with_call_content(tmp_path: Path) -> None:
    path = tmp_path / "session-store.db"
    db = make_store(path)
    add_call(db, 1)
    db.close()
    first = read_session_store(path)
    second = read_session_store(path)
    assert first == second
    assert first.fingerprint == second.fingerprint

    db = sqlite3.connect(path)
    db.execute("UPDATE assistant_usage_events SET output_tokens=21 WHERE id=1")
    db.commit()
    db.close()
    changed = read_session_store(path)
    assert changed.fingerprint != first.fingerprint

    db = sqlite3.connect(path)
    db.execute("DELETE FROM assistant_usage_events")
    db.commit()
    db.close()
    deleted = read_session_store(path)
    assert deleted.calls == ()
    assert deleted.fingerprint != changed.fingerprint


def test_context_helper_yields_read_only_validated_snapshot(tmp_path: Path) -> None:
    path = tmp_path / "session-store.db"
    db = make_store(path)
    add_call(db, 1)
    db.close()

    with session_store_connection(path) as snapshot_db:
        assert snapshot_db.execute("SELECT count(*) FROM assistant_usage_events").fetchone() == (1,)
        assert snapshot_db.execute("PRAGMA query_only").fetchone() == (1,)
        with pytest.raises(sqlite3.OperationalError):
            snapshot_db.execute("DELETE FROM assistant_usage_events")


def test_snapshot_dataclasses_are_immutable_and_constructor_matches_shared_contract() -> None:
    session = CopilotStoreSession(SESSION, None, None, None, None)
    call = CopilotStoreCall(1, SESSION, None, None, "model", None, None, None, None, None, None, None, None, None, None)
    snapshot = CopilotStoreSnapshot("session-store.db", hashlib.sha256(b"x").hexdigest(), (session,), (call,))
    assert snapshot.schema_version == 8
    with pytest.raises(AttributeError):
        snapshot.calls = ()  # type: ignore[misc]
