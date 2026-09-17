"""Native migration against genuine legacy SQLite, including failed publication."""
import importlib.util
import json
from pathlib import Path
import sqlite3
import sys

import duckdb
import pytest

from harness_usage.application import Application
from harness_usage.domain import Interval, Point, RecordedEstimate
from harness_usage.migrate_sqlite import migrate_sqlite
from harness_usage.pi_reader import ReadBatch, read_pi
from harness_usage.storage import Storage, digest, micros, token_data


def native_v5(path: Path) -> None:
    raw = duckdb.connect(str(path))
    try:
        raw.execute((Path(__file__).parent / 'fixtures/schema-v5.sql').read_text())
        raw.execute('UPDATE ledger_meta SET revision=7')
        raw.execute("INSERT INTO session(id,harness,native_id,display_name,started_us,last_seen_us,cwd) VALUES('pi:fixture','pi','fixture','Fixture',1,2,'/fixture')")
        raw.execute("INSERT INTO session_attribution VALUES('pi:fixture',NULL,NULL,'unknown')")
        raw.execute("INSERT INTO source_generation VALUES('source','/fixture',0,?, 'pi:fixture','pi-v3/0.85.1-shape-1',1,0,'available')", ('a' * 64,))
        raw.execute("INSERT INTO source_metadata VALUES('source',?)", ('b' * 64,))
        raw.execute("INSERT INTO observation VALUES('observation','pi:fixture','entry','fingerprint','assistant','point',1,NULL,NULL,'response_recorded_at','test','test','stop',NULL,'{}')")
        raw.execute("INSERT INTO appearance(source_id,line,observation_id,ordinal) VALUES('source',1,'observation',41)")
        raw.execute("INSERT INTO entry_edge VALUES('source',1,'entry',NULL)")
        for measure in ('input', 'output', 'cache_read', 'cache_write', 'reported_total', 'reasoning', 'cache_write_1h'):
            raw.execute("INSERT INTO token_value VALUES('observation',?,'known',0,NULL)", (measure,))
        raw.execute("INSERT INTO recorded_estimate VALUES('observation','missing',NULL,NULL,NULL,NULL,'not_recorded')")
        for measure in ('input', 'output', 'cache_read', 'cache_write', 'total', 'recorded_usd'):
            raw.execute("INSERT INTO decision VALUES('observation',?,'selected','pi:fixture',NULL,'fixture','pi-1')", (measure,))
        raw.execute('CHECKPOINT')
    finally:
        raw.close()


def native_v5_pi(path: Path, data: bytes) -> None:
    batch = read_pi(data, locator='/fixture')
    assert isinstance(batch, ReadBatch) and len(batch.usage) == 1
    record = batch.usage[0]
    fingerprint = '6dabc396420964449cc11327a621608511634d1945a3a011fe45392d3e9f9644'
    observation_id = 'eeb87e4cec6c0a4cbb85c4ccbc9f919008596371d93208062a1428b914444d1e'
    raw = duckdb.connect(str(path))
    try:
        raw.execute((Path(__file__).parent / 'fixtures/schema-v5.sql').read_text())
        session = batch.session
        raw.execute('INSERT INTO session VALUES(?,?,?,?,?,?,?,?,?)',
                    (session.id, 'pi', session.native_id, session.display_name, micros(session.started), micros(session.last_observed), session.title_excerpt, session.cwd, session.parent_locator))
        raw.execute("INSERT INTO session_attribution VALUES(?,NULL,NULL,'not_resolved')", (session.id,))
        source_id = digest('/fixture:0')
        raw.execute('INSERT INTO source_generation VALUES(?,?,?,?,?,?,?,?,?)',
                    (source_id, '/fixture', 0, digest(data), session.id, 'pi-v3/0.85.1-shape-1', batch.complete_bytes, int(batch.pending_tail), 'available'))
        raw.execute('INSERT INTO source_metadata VALUES(?,?)', (source_id, digest(data[:batch.complete_bytes])))
        for edge in batch.entries:
            raw.execute('INSERT INTO entry_edge VALUES(?,?,?,?)', (source_id, edge.line, edge.native_id, edge.parent_id))
        if isinstance(record.time, Point):
            time = ('point', micros(record.time.at), None, None, 'response_recorded_at')
        elif isinstance(record.time, Interval):
            time = ('interval', None, micros(record.time.start), micros(record.time.end), None)
        else:
            time = ('undated', None, None, None, record.time.reason)
        raw.execute('INSERT INTO observation VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                    (observation_id, session.id, record.entry.native_id, fingerprint, record.kind, *time,
                     record.model.provider, record.model.model, record.stop_reason, record.tool_call_id,
                     json.dumps(dict(record.safe_facts), default=str)))
        raw.execute('INSERT INTO appearance(source_id,line,observation_id) VALUES(?,?,?)',
                    (source_id, record.entry.line, observation_id))
        values = (*record.tokens.buckets.values, record.tokens.reported_total, record.tokens.reasoning, record.tokens.cache_write_1h)
        for name, value in zip(('input', 'output', 'cache_read', 'cache_write', 'reported_total', 'reasoning', 'cache_write_1h'), values):
            raw.execute('INSERT INTO token_value VALUES(?,?,?,?,?)', (observation_id, name, *token_data(value)))
        assert isinstance(record.money, RecordedEstimate)
        raw.execute('INSERT INTO recorded_estimate VALUES(?,?,?,?,?,?,?)',
                    (observation_id, 'known', str(record.money.amount), 'USD', json.dumps(dict(record.money.components), default=str), record.money.source_ref, None))
        for measure in ('input', 'output', 'cache_read', 'cache_write', 'total', 'recorded_usd'):
            raw.execute('INSERT INTO decision VALUES(?,?,?,?,?,?,?)',
                        (observation_id, measure, 'selected', session.id, None, 'fixture', 'pi-1'))
        raw.execute('CHECKPOINT')
    finally:
        raw.close()


def test_native_v5_migrates_without_changing_revision_or_evidence(tmp_path):
    path = tmp_path / 'ledger.duckdb'
    native_v5(path)
    store = Storage(path)
    with store.connect() as db:
        assert db.execute('SELECT schema_version FROM ledger_meta').one()[0] == 6
        assert db.execute('SELECT count(*) FROM quantity_value').one()[0] == 0
        assert db.execute('SELECT ordinal FROM appearance').one()[0] == 41
    assert store.snapshot().revision == 7
    assert len(store.snapshot().observations) == 1
    backups = list(tmp_path.glob('ledger.v5-backup-*.duckdb'))
    assert len(backups) == 1 and backups[0].exists()


def test_native_v5_migration_then_append_reuses_existing_observation_identity(tmp_path):
    path = tmp_path / 'ledger.duckdb'
    data = (Path(__file__).parent / 'fixtures/pi/source/ordinary.jsonl').read_bytes()
    native_v5_pi(path, data)
    store = Storage(path)
    header, first = data.decode().splitlines()
    second = json.loads(first)
    second['id'] = '00000002'
    second['timestamp'] = '2026-09-12T10:00:00Z'
    appended = ('\n'.join((header, first, json.dumps(second))) + '\n').encode()
    store.import_source('/fixture', appended)
    observations = store.snapshot().observations
    assert len(observations) == 2
    retained = [row for row in observations if row.record.entry.native_id == '00000001']
    assert len(retained) == 1
    assert retained[0].id == 'eeb87e4cec6c0a4cbb85c4ccbc9f919008596371d93208062a1428b914444d1e'
    assert 'identity_conflict' not in retained[0].reasons


def test_failed_native_v5_migration_leaves_original_unchanged(tmp_path, monkeypatch):
    import harness_usage.migrate_duckdb as migration
    path = tmp_path / 'ledger.duckdb'
    native_v5(path)
    before = path.read_bytes()

    def fail(*args, **kwargs):
        raise RuntimeError('injected row copy failure')

    monkeypatch.setattr(migration, '_copy_table', fail)
    with pytest.raises(RuntimeError, match='injected'):
        Storage(path)
    assert path.read_bytes() == before
    with duckdb.connect(str(path), read_only=True) as db:
        assert db.execute('SELECT schema_version FROM ledger_meta').fetchone()[0] == 5
    assert not list(tmp_path.glob('ledger.v5-backup-*.duckdb'))
    assert not list(tmp_path.glob('.ledger-v6-*'))


def legacy_storage(path):
    # Seed checked domain fixtures, then export the actual old schema representation.
    from legacy_fixture import export_legacy
    seed = Storage(path.with_name('seed.duckdb'))
    data = (Path(__file__).parent / 'fixtures/pi/source/ordinary.jsonl').read_bytes()
    seed.import_source('/fixture', data)
    seed.begin_import('unfinished')
    export_legacy(seed, path)
    return seed


def test_migration_preserves_evidence_and_uses_only_duckdb_afterward(tmp_path, monkeypatch):
    source = tmp_path / 'ledger.sqlite3'
    old = legacy_storage(source)
    data = (Path(__file__).parent / 'fixtures/pi/source/ordinary.jsonl').read_bytes()
    before = old.snapshot()
    target = tmp_path / 'ledger.duckdb'
    migrate_sqlite(source, target)
    new = Storage(target)
    actual = new.snapshot()
    assert (actual.revision, actual.sessions, actual.attributions, actual.diagnostics) == (before.revision, before.sessions, before.attributions, before.diagnostics)
    assert repr(actual.observations) == repr(before.observations)
    assert source.exists()
    new.close()
    def forbidden(*args, **kwargs):
        raise AssertionError('SQLite accessed after migration')
    monkeypatch.setattr(sqlite3, 'connect', forbidden)
    app = Application(tmp_path)
    assert app.status().state == 'interrupted'
    assert app.storage.path == target
    assert not (tmp_path / 'reports.duckdb').exists()
    assert app.storage.import_source('/fixture', data) == before.revision
    app.close()


def test_failed_migration_does_not_publish_or_change_source(tmp_path, monkeypatch):
    import harness_usage.migrate_sqlite as migration
    source, target = tmp_path / 'ledger.sqlite3', tmp_path / 'ledger.duckdb'
    legacy_storage(source)
    before = source.read_bytes()
    def fail(*args):
        raise RuntimeError('injected copy failure')
    monkeypatch.setattr(migration, '_copy', fail)
    with pytest.raises(RuntimeError, match='injected'):
        migrate_sqlite(source, target)
    assert not target.exists()
    assert source.read_bytes() == before
    assert not list(tmp_path.glob('.ledger-migration-*'))


@pytest.mark.parametrize('version', [1, 2, 3])
def test_supported_legacy_schema_versions(tmp_path, version):
    source = tmp_path / 'ledger.sqlite3'
    old = legacy_storage(source)
    before = old.snapshot()
    with sqlite3.connect(source) as db:
        if version < 3:
            db.execute('ALTER TABLE session DROP COLUMN title_excerpt')
        if version == 1:
            db.execute('DROP TABLE delegation_ref')
            db.execute('DROP TABLE delegation_scan')
        db.execute('ALTER TABLE ledger_meta RENAME TO old_meta')
        db.execute(f'CREATE TABLE ledger_meta(singleton INTEGER PRIMARY KEY,schema_version INTEGER CHECK(schema_version={version}),revision INTEGER)')
        db.execute(f'INSERT INTO ledger_meta SELECT singleton,{version},revision FROM old_meta')
        db.execute('DROP TABLE old_meta')
    store = Storage(tmp_path / 'ledger.duckdb')
    assert repr(store.snapshot().observations) == repr(before.observations)
    assert store.snapshot().revision == before.revision
    store.close()


def test_abrupt_exit_before_publication_is_safe_to_retry(tmp_path):
    import os
    import subprocess
    source, target = tmp_path / 'ledger.sqlite3', tmp_path / 'ledger.duckdb'
    legacy_storage(source).close()
    before = source.read_bytes()
    code = '''
import os,sys
from pathlib import Path
import harness_usage.migrate_sqlite as migration
original = migration._copy
def crash(*args):
    original(*args)
    os._exit(18)
migration._copy = crash
migration.migrate_sqlite(Path(sys.argv[1]), Path(sys.argv[2]))
'''
    result = subprocess.run([sys.executable,'-c',code,str(source),str(target)],
                            env={**os.environ, 'PYTHONPATH': str(Path.cwd()/'src')}, timeout=30)
    assert result.returncode == 18
    assert not target.exists()
    assert source.read_bytes() == before
    migrate_sqlite(source,target)
    assert Storage(target).snapshot().revision == 1


def test_existing_target_is_never_overwritten(tmp_path):
    source, target = tmp_path/'ledger.sqlite3', tmp_path/'ledger.duckdb'
    legacy_storage(source).close()
    Storage(target).close()
    before = target.read_bytes()
    with pytest.raises(FileExistsError):
        migrate_sqlite(source,target)
    assert target.read_bytes() == before
