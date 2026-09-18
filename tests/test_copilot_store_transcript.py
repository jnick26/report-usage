import sqlite3

import pytest

from harness_usage.storage import Storage
from harness_usage.transcript import Message, TranscriptUnavailable
from harness_usage.transcript_access import read_transcript_page
from harness_usage.transcript_rendering import render_transcript

NATIVE = '11111111-1111-4111-8111-111111111111'
SESSION = 'copilot-cli:' + NATIVE


def store_database(path, *, turns=True):
    with sqlite3.connect(path) as db:
        db.executescript('''
            CREATE TABLE schema_version(version INTEGER NOT NULL);
            INSERT INTO schema_version VALUES(8);
            CREATE TABLE sessions(id TEXT PRIMARY KEY,cwd TEXT,created_at TEXT,updated_at TEXT,host_type TEXT);
            CREATE TABLE assistant_usage_events(id INTEGER PRIMARY KEY,session_id TEXT NOT NULL,turn_index INTEGER,
                created_at TEXT,model TEXT NOT NULL,agent_id TEXT,parent_tool_call_id TEXT,initiator TEXT,input_tokens INTEGER,
                output_tokens INTEGER,cache_read_tokens INTEGER,cache_write_tokens INTEGER,
                reasoning_tokens INTEGER,total_nano_aiu INTEGER,request_multiplier REAL,token_details_json TEXT);
        ''')
        db.execute('INSERT INTO sessions VALUES(?,?,?,?,?)', (NATIVE, '/fixture', None, None, 'cli'))
        if turns:
            db.executescript('''CREATE TABLE turns(id INTEGER PRIMARY KEY,session_id TEXT,turn_index INTEGER,
                user_message TEXT,assistant_response TEXT,timestamp TEXT);''')
            db.execute('INSERT INTO turns VALUES(1,?,0,?,?,?)',
                       (NATIVE, '<script>turn-canary</script>', 'Retained answer', '2026-09-17T12:00:00Z'))


def register_store(storage, path):
    with storage.connect() as db:
        db.execute("INSERT INTO session(id,harness,native_id) VALUES(?,'copilot-cli',?)", (SESSION, NATIVE))
        db.execute("INSERT INTO session_attribution VALUES(?,NULL,NULL,'unknown')", (SESSION,))
        db.execute("INSERT INTO source_generation VALUES('store',?,0,?,NULL,?,0,0,'available')",
                   (str(path), '0' * 64, 'copilot-cli-session-store/schema-8-shape-1'))
        db.execute('INSERT INTO copilot_store_session VALUES(?,?,NULL,NULL,NULL)', ('store', SESSION))


def test_db_only_turns_are_on_demand_lossy_and_escaped(tmp_path):
    source = tmp_path / 'session-store.db'
    store_database(source)
    storage = Storage(tmp_path / 'ledger.duckdb')
    register_store(storage, source)
    page = read_transcript_page(storage, (str(tmp_path),), SESSION)
    assert [entry.role for entry in page.transcript.entries if isinstance(entry, Message)] == ['user', 'assistant']
    assert page.transcript.entries[1].blocks[0].text == '<script>turn-canary</script>'
    assert page.transcript.entries[0].label == 'Lossy turn-summary view'
    assert page.transcript.branches == ()
    for standalone in (False, True):
        html = render_transcript(page, standalone=standalone)
        assert '&lt;script&gt;turn-canary&lt;/script&gt;' in html
        assert '<script>turn-canary</script>' not in html
        assert '<summary>Lossy turn-summary view</summary>' in html
    with sqlite3.connect(source) as db:
        db.execute("UPDATE turns SET assistant_response='Changed on demand'")
    assert read_transcript_page(storage, (str(tmp_path),), SESSION).transcript.entries[-1].blocks[0].text == 'Changed on demand'
    assert b'turn-canary' not in (tmp_path / 'ledger.duckdb').read_bytes()


@pytest.mark.parametrize('case', ['absent', 'empty', 'branch', 'bound', 'turn_bound', 'outside', 'symlink', 'directory_symlink', 'wal_symlink'])
def test_store_transcript_rejects_unavailable_or_unsafe_input(tmp_path, monkeypatch, case):
    from harness_usage import copilot_store_transcript

    root = tmp_path / 'root'
    root.mkdir()
    source = root / 'session-store.db'
    store_database(source, turns=case != 'absent')
    if case == 'empty':
        with sqlite3.connect(source) as db:
            db.execute('DELETE FROM turns')
    if case == 'bound':
        monkeypatch.setattr(copilot_store_transcript, 'MAX_TRANSCRIPT_BYTES', 4)
    if case == 'turn_bound':
        monkeypatch.setattr(copilot_store_transcript, 'MAX_TURNS', 0)
    if case == 'symlink':
        actual = tmp_path / 'actual.db'
        source.rename(actual)
        source.symlink_to(actual)
    if case == 'directory_symlink':
        linked = tmp_path / 'linked'
        linked.symlink_to(root, target_is_directory=True)
        source = linked / 'session-store.db'
    if case == 'wal_symlink':
        outside = tmp_path / 'private-wal'
        outside.write_bytes(b'not authorized')
        source.with_name(source.name + '-wal').symlink_to(outside)
    storage = Storage(tmp_path / 'ledger.duckdb')
    register_store(storage, source)
    roots = (str(tmp_path / 'other'),) if case == 'outside' else (str(tmp_path),)
    with pytest.raises(TranscriptUnavailable) as error:
        read_transcript_page(storage, roots, SESSION, 'invented' if case == 'branch' else None)
    assert error.value.kind == ('invalid_branch' if case == 'branch' else 'too_large' if case in ('bound', 'turn_bound')
                                else 'invalid_source' if case in ('outside', 'symlink', 'directory_symlink', 'wal_symlink') else 'unsupported')


def test_native_jsonl_transcript_wins_over_store_summary(tmp_path):
    from pathlib import Path

    source = tmp_path / 'session-store.db'
    store_database(source)
    storage = Storage(tmp_path / 'ledger.duckdb')
    register_store(storage, source)
    native = tmp_path / 'events.jsonl'
    native.write_bytes((Path(__file__).parent / 'fixtures/copilot_cli/current/events.jsonl').read_bytes())
    with storage.connect() as db:
        db.execute("INSERT INTO source_generation VALUES('native',?,0,?,?,?,0,0,'available')",
                   (str(native), '0' * 64, SESSION, 'copilot-cli-events/e60d903-shape-1'))
    page = read_transcript_page(storage, (str(tmp_path),), SESSION)
    assert all('lossy' not in notice.lower() for notice in page.transcript.warnings)
    assert 'turn-canary' not in render_transcript(page)
    native.unlink()
    assert 'Lossy turn-summary view' in render_transcript(read_transcript_page(storage, (str(tmp_path),), SESSION))


def test_retired_store_membership_cannot_open_old_turns(tmp_path):
    source = tmp_path / 'session-store.db'
    store_database(source)
    storage = Storage(tmp_path / 'ledger.duckdb')
    register_store(storage, source)
    with storage.connect() as db:
        db.execute("INSERT INTO source_generation VALUES('replacement',?,1,?,NULL,?,0,0,'available')",
                   (str(source), '1' * 64, 'copilot-cli-session-store/schema-8-shape-1'))
    with pytest.raises(TranscriptUnavailable) as error:
        read_transcript_page(storage, (str(tmp_path),), SESSION)
    assert error.value.kind == 'missing'
