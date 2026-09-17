"""Only the approved, bounded first user excerpt crosses the transcript boundary."""
import json
import sqlite3

from harness_usage.pi_reader import ReadBatch, read_metadata, read_pi
from harness_usage.reporting import AllTime, ReportQuery, build_report
from harness_usage.storage import Storage


def source(identity='main', *, name=None, prompt='  First\n  request  ', usage=True):
    rows = [{'type': 'session', 'version': 3, 'id': identity, 'cwd': '/project'}]
    rows += [{'type': 'message', 'id': 'empty', 'message': {'role': 'user', 'content': [{'type': 'image', 'data': 'IMAGE PRIVATE'}, {'type': 'text', 'text': '  '}]}},
             {'type': 'message', 'id': 'prompt', 'message': {'role': 'user', 'content': prompt}},
             {'type': 'message', 'id': 'later', 'message': {'role': 'user', 'content': 'LATER PRIVATE'}}]
    if name is not None:
        rows.append({'type': 'session_info', 'id': 'name', 'name': name})
    if usage:
        rows.append({'type': 'message', 'id': 'usage', 'message': {'role': 'assistant', 'content': 'ASSISTANT PRIVATE', 'usage': {'input': 10, 'output': 0, 'cacheRead': 0, 'cacheWrite': 0}}})
    return ('\n'.join(json.dumps(row) for row in rows) + '\n').encode()


def test_first_nonempty_user_excerpt_and_explicit_name_precedence():
    result = read_pi(source(prompt=[{'type': 'text', 'text': ' First\n'}, {'type': 'image', 'data': 'PRIVATE'}, {'type': 'text', 'text': ' request '}]), locator='/main')
    assert isinstance(result, ReadBatch)
    assert result.session.title_excerpt == 'First request'
    assert result.session.display_name is None
    assert 'PRIVATE' not in repr(result)
    named = read_pi(source(name='Native name'), locator='/main')
    assert isinstance(named, ReadBatch)
    assert named.session.display_name == 'Native name'
    assert named.session.title_excerpt is None
    assert 'First request' not in repr(named)


def test_excerpt_is_bounded_and_excludes_unretained_text():
    result = read_pi(source(prompt='word ' * 50 + 'TAIL PRIVATE'), locator='/main')
    assert isinstance(result, ReadBatch)
    assert len(result.session.title_excerpt) == 160
    assert result.session.title_excerpt.endswith('…')
    assert 'PRIVATE' not in repr(result)


def test_metadata_scanner_matches_reader_when_native_name_is_cleared():
    data = source(name='Native name') + b'{"type":"session_info","id":"clear"}\n'
    result = read_pi(data, locator='/main')
    assert isinstance(result, ReadBatch)
    assert result.session.display_name is None
    assert result.session.title_excerpt == read_metadata(data)[1] == 'First request'
    assert read_metadata(b'{"type":"session","version":2,"id":"old"}\n' + data)[1] is None


def test_excerpt_cannot_prove_delegation_or_supply_an_untitled_roots_label(tmp_path):
    storage = Storage(tmp_path / 'ledger.sqlite3')
    root = b'{"type":"session","version":3,"id":"main","cwd":"/project"}\n'
    storage.import_sources([('/main', root), ('/child', source('child', prompt='Exact child name'))])
    with storage.connect() as db:
        source_id = db.execute("SELECT id FROM source_generation WHERE locator='/main'").fetchone()[0]
        db.execute('INSERT INTO delegation_ref VALUES(?,?,?,?,?)', (source_id, 'spawn', 'child_name', 'Exact child name', ''))
    query = ReportQuery(None, AllTime())
    data = storage.report_input(query)
    assert data.contributions[0].family is None
    with storage.connect() as db:
        db.execute('UPDATE delegation_ref SET kind=?,value=?', ('child_path', '/child'))
    data = storage.report_input(query)
    assert data.contributions[0].family.name is None
    result = build_report(data.contributions, revision=data.revision, query=query)
    assert result.sessions[0].name == 'Session main'


def test_family_uses_roots_own_excerpt_even_without_root_usage(tmp_path):
    storage = Storage(tmp_path / 'ledger.sqlite3')
    root = source(usage=False) + json.dumps({'type': 'message', 'id': 'spawn', 'message': {'role': 'toolResult', 'toolName': 'subagent', 'details': {'results': [{'sessionFile': '/child'}]}}}).encode() + b'\n'
    storage.import_sources([('/main', root), ('/child', source('child', prompt='Child request'))])
    query = ReportQuery(None, AllTime())
    data = storage.report_input(query)
    result = build_report(data.contributions, revision=data.revision, query=query)
    assert len(result.sessions) == 1
    assert result.sessions[0].name == 'First request'
    assert data.contributions[0].name == 'Child request'
    assert data.contributions[0].family.name == 'First request'
    named = Storage(tmp_path / 'named.sqlite3')
    named.import_source('/main', source(name='Native name'))
    assert named.report_input(query).contributions[0].name == 'Native name'


def test_v2_title_backfill_once_without_accounting_replay(tmp_path, monkeypatch):
    import harness_usage.pi_reader as reader
    import harness_usage.storage as module
    path = tmp_path / 'ledger.sqlite3'
    storage = Storage(tmp_path/'seed.duckdb')
    data = source()
    storage.import_source('/main', data)
    before = storage.snapshot()
    with storage.connect() as db:
        hashes = list(map(tuple, db.execute('SELECT * FROM source_generation')))
        decisions = list(map(tuple, db.execute('SELECT * FROM decision')))
    from legacy_fixture import export_legacy
    export_legacy(storage, path)
    with sqlite3.connect(path) as db:
        db.executescript('''
            ALTER TABLE session DROP COLUMN title_excerpt;
            CREATE TABLE old_meta(singleton INTEGER PRIMARY KEY CHECK(singleton=1),schema_version INTEGER NOT NULL CHECK(schema_version=2),revision INTEGER NOT NULL CHECK(revision>=0)) STRICT;
            INSERT INTO old_meta SELECT singleton,2,revision FROM ledger_meta;
            DROP TABLE ledger_meta;
            ALTER TABLE old_meta RENAME TO ledger_meta;
            UPDATE delegation_scan SET profile='pi-delegation-2';
        ''')
    def forbidden(*args, **kwargs):
        raise AssertionError('Metadata backfill must not replay accounting')
    monkeypatch.setattr(module, 'read_pi', forbidden)
    monkeypatch.setattr(reader, 'read_pi', forbidden)
    monkeypatch.setattr(Storage, '_reconcile', forbidden)
    migrated = Storage(path)
    backups = [path]
    assert len(backups) == 1
    with sqlite3.connect(backups[0]) as backup:
        assert backup.execute('SELECT schema_version FROM ledger_meta').fetchone()[0] == 2
    revision = migrated.import_source('/main', data)
    assert revision == before.revision + 1
    assert migrated.snapshot().sessions[0].title_excerpt == 'First request'
    assert migrated.snapshot().observations == before.observations
    with migrated.connect() as db:
        assert list(map(tuple, db.execute('SELECT * FROM source_generation'))) == hashes
        assert list(map(tuple, db.execute('SELECT * FROM decision'))) == decisions
        assert db.execute('SELECT schema_version FROM ledger_meta').fetchone()[0] == 6
    monkeypatch.setattr(module, 'read_metadata', forbidden)
    assert migrated.import_source('/main', data) == revision
