from pathlib import Path

import pytest

from harness_usage.application import Application, SourceFailure


def test_scan_routes_database_without_reading_as_json_and_counts_once(tmp_path, monkeypatch):
    from harness_usage import copilot_store_reader
    from test_copilot_store_reader import make_store

    root = tmp_path / 'copilot'
    child = root / 'session-state'
    child.mkdir(parents=True)
    database = root / 'session-store.db'
    make_store(database).close()
    snapshot = copilot_store_reader.read_session_store(database)
    calls = []
    read_store = copilot_store_reader.read_session_store

    def read_snapshot(path):
        calls.append(path)
        return read_store(path)

    monkeypatch.setattr(copilot_store_reader, 'read_session_store', read_snapshot)
    original = Path.read_bytes

    def read_json_only(path):
        assert path != database, 'SQLite must use its snapshot reader'
        return original(path)

    monkeypatch.setattr(Path, 'read_bytes', read_json_only)
    app = Application(tmp_path / 'data')
    progress = []
    monkeypatch.setattr(app, '_update_progress', lambda *args: progress.append(args))
    try:
        assert tuple(app._scan((str(root), str(child)))) == (snapshot,)
        assert calls == [database]
        assert progress[-1][-2:] == (1, 1)
        calls.clear()
        assert tuple(app._scan((str(child),))) == ()
        assert calls == []
        database.unlink()
        assert tuple(app._scan((str(root),))) == ()
    finally:
        app.close()


def test_invalid_database_uses_existing_content_free_source_failure(tmp_path):
    root = tmp_path / 'copilot'
    root.mkdir()
    (root / 'session-store.db').write_bytes(b'PRIVATE_INVALID_DATABASE')
    app = Application(tmp_path / 'data')
    try:
        with pytest.raises(SourceFailure) as error:
            tuple(app._scan((str(root),)))
        assert error.value.code == 'source_1_read_failed'
        assert 'PRIVATE' not in str(error.value)
    finally:
        app.close()


def test_scan_denies_symlink_database_and_directory(tmp_path):
    root = tmp_path / 'authorized'
    outside = tmp_path / 'outside'
    root.mkdir()
    outside.mkdir()
    (outside / 'session-store.db').write_bytes(b'not authorized')
    (root / 'session-store.db').symlink_to(outside / 'session-store.db')
    (root / 'linked').symlink_to(outside, target_is_directory=True)
    app = Application(tmp_path / 'data')
    try:
        assert tuple(app._scan((str(root),))) == ()
    finally:
        app.close()


def test_database_import_progress_saved_roots_and_missing_history(tmp_path):
    from harness_usage.reporting import AllTime, ReportQuery
    from test_copilot_store_reader import make_store, add_call, SESSION

    root = tmp_path / 'copilot'
    legacy = root / 'session-state'
    legacy.mkdir(parents=True)
    database = root / 'session-store.db'
    db = make_store(database)
    add_call(db, 1)
    db.close()
    app = Application(tmp_path / 'data', timezone='UTC')
    try:
        app.set_roots((str(legacy),))
        app.start_import(); app.close()
        assert app.status().files_processed == 0
        assert app.get_roots() == (str(legacy),)
        app.set_roots((str(root), str(legacy)))
        app.start_import(); app.close()
        assert app.status().state == 'succeeded'
        assert (app.status().files_processed, app.status().files_checked, app.status().files_total) == (1, 1, 1)
        assert [str(session.id) for session in app.report(ReportQuery(None, AllTime())).sessions] == ['copilot-cli:' + SESSION]
        app.start_import(); app.close()
        assert app.status().files_processed == 1
        database.unlink()
        app.start_import(); app.close()
        assert app.status().state == 'succeeded'
        assert app.status().files_processed == 0
        assert any(entry.code == 'saved_history' for entry in app.report(ReportQuery(None, AllTime())).coverage)
    finally:
        app.close()
