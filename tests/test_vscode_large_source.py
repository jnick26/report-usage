"""Oversized chat candidates still require full reader qualification."""
import json
from decimal import Decimal
from pathlib import Path

import pytest

from harness_usage.application import Application
from harness_usage.copilot_vscode_reader import CopilotVscodeReadBatch, read_copilot_vscode
from harness_usage.reporting import AllTime, ReportQuery, build_report
from harness_usage.source_input import MAX_PROBE_BYTES, SourcePayload, detect_source
from harness_usage.storage import Storage
from test_source_input import CLAUDE_RECORD, CODEX_HEADER, PI_HEADER


def chat_bytes(suffix, *, large=True, extension='github.copilot-chat', version=3):
    state = {
        'version': version, 'sessionId': 'large-chat', 'requests': [{
            'requestId': 'request', 'agent': {'extensionId': {'value': extension}},
            'promptTokens': 12, 'completionTokens': 5, 'copilotCredits': 0.25,
            'response': [{'kind': 'markdownContent', 'content': {
                'value': 'x' * (MAX_PROBE_BYTES + 1 if large else 1)}}],
        }],
    }
    value = {'kind': 0, 'v': state} if suffix == '.jsonl' else state
    return json.dumps(value).encode() + b'\n'


def chat_path(root, suffix):
    return root / f'User/workspaceStorage/key/chatSessions/session{suffix}'


@pytest.mark.parametrize('suffix', ('.json', '.jsonl'))
def test_large_chat_detection_and_direct_import_retain_qualified_usage(tmp_path, suffix):
    source = SourcePayload(str(chat_path(tmp_path, suffix)), chat_bytes(suffix))
    assert len(source.data.split(b'\n', 1)[0]) > MAX_PROBE_BYTES
    assert isinstance(read_copilot_vscode(source.data, locator=source.locator), CopilotVscodeReadBatch)
    assert detect_source(source) == 'copilot-vscode'
    store = Storage(tmp_path / 'ledger.duckdb')
    try:
        revision = store.import_sources((source,))
        with store.connect() as db:
            assert tuple(db.execute('SELECT raw_input,raw_output FROM copilot_vscode_evidence')) == ((12, 5),)
            assert db.execute('SELECT count(*) FROM observation').one()[0] == 1
        assert store.import_sources((source,)) == revision
    finally:
        store.close()
    reopened = Storage(tmp_path / 'ledger.duckdb')
    try:
        assert reopened.import_sources((source,)) == revision
        with reopened.connect() as db:
            assert db.execute('SELECT count(*) FROM source_generation').one()[0] == 1
    finally:
        reopened.close()


@pytest.mark.parametrize('suffix', ('.json', '.jsonl'))
def test_scanner_keeps_growing_chat_available_and_accounting_stable(tmp_path, suffix):
    root = tmp_path / 'history'
    path = chat_path(root, suffix)
    path.parent.mkdir(parents=True)
    app = Application(tmp_path / 'data', timezone='UTC')
    app.set_roots((str(root),))
    query = ReportQuery(None, AllTime())
    for large in (False, True):
        path.write_bytes(chat_bytes(suffix, large=large))
        app.start_import()
        app.close()
        assert app.status().state == 'succeeded'
        assert app.status().files_processed == 1
        report = app.report(query)
        assert (report.tokens.input.known, report.tokens.output.known) == (12, 5)
        assert [(row.measure, row.known) for row in report.quantities] == [('ai_credits', Decimal('0.25'))]
        assert 'saved_history' not in {row.code for row in report.coverage}
    with app.storage.connect() as db:
        assert tuple(db.execute('SELECT generation,availability FROM source_generation ORDER BY generation')) == (
            (0, 'available'), (1, 'available'))
    app.start_import()
    app.close()
    assert app.report(query) == report
    reopened = Application(tmp_path / 'data', timezone='UTC')
    try:
        assert reopened.report(query) == report
    finally:
        reopened.close()


@pytest.mark.parametrize('suffix', ('.json', '.jsonl'))
@pytest.mark.parametrize('invalid', ('foreign', 'schema', 'malformed'))
def test_large_candidates_cannot_bypass_reader_accounting_qualification(tmp_path, suffix, invalid):
    data = (chat_bytes(suffix, extension='other.extension') if invalid == 'foreign'
            else chat_bytes(suffix, version=3.0) if invalid == 'schema'
            else b'{' + b'x' * MAX_PROBE_BYTES + b'\n')
    path = chat_path(tmp_path / 'history', suffix)
    path.parent.mkdir(parents=True)
    path.write_bytes(data)
    app = Application(tmp_path / 'data', timezone='UTC')
    try:
        candidates = tuple(app._scan((str(tmp_path / 'history'),)))
        assert len(candidates) == 1
        app.storage.import_sources(candidates)
        with app.storage.connect() as db:
            assert db.execute('SELECT count(*) FROM observation').one()[0] == 0
            assert db.execute('SELECT count(*) FROM copilot_vscode_evidence').one()[0] == 0
            if invalid != 'foreign':
                assert db.execute('SELECT count(*) FROM session').one()[0] == 0
                assert db.execute('SELECT count(*) FROM diagnostic').one()[0] == 1
    finally:
        app.close()


@pytest.mark.parametrize('locator', (
    '/chatSessions/session.json', '/User/other/key/chatSessions/session.jsonl',
    '/User/workspaceStorage/key/other/session.json',
    '/User/workspaceStorage/key/chatSessions/session.txt',
    'User/workspaceStorage/key/chatSessions/session.json',
))
def test_large_fallback_requires_approved_layout_and_suffix(locator):
    assert detect_source(SourcePayload(locator, chat_bytes(Path(locator).suffix))) is None


@pytest.mark.parametrize('suffix', ('.json', '.jsonl'))
@pytest.mark.parametrize('data', (b'{broken', b'{"event":"foreign"}', b'[]'))
def test_small_invalid_candidates_do_not_gain_path_only_detection(tmp_path, suffix, data):
    assert detect_source(SourcePayload(str(chat_path(tmp_path, suffix)), data)) is None


def test_jsonl_large_body_does_not_override_small_foreign_first_line(tmp_path):
    data = b'{"event":"foreign"}\n' + b'x' * (MAX_PROBE_BYTES + 1)
    assert detect_source(SourcePayload(str(chat_path(tmp_path, '.jsonl')), data)) is None


@pytest.mark.parametrize(('header', 'kind'), ((PI_HEADER, 'pi'), (CODEX_HEADER, 'codex'), (CLAUDE_RECORD, 'claude')))
@pytest.mark.parametrize('suffix', ('.json', '.jsonl'))
def test_existing_source_headers_precede_large_vscode_candidate_routing(tmp_path, header, kind, suffix):
    data = header + b'x' * (MAX_PROBE_BYTES + 1)
    assert detect_source(SourcePayload(str(chat_path(tmp_path, suffix)), data)) == kind


@pytest.mark.parametrize('suffix', ('.json', '.jsonl'))
def test_malformed_probe_at_limit_stays_unclassified(tmp_path, suffix):
    data = b'{' + b'x' * (MAX_PROBE_BYTES - 1)
    assert detect_source(SourcePayload(str(chat_path(tmp_path, suffix)), data)) is None


def malformed_chat(suffix):
    return b'{' + b'x' * MAX_PROBE_BYTES + (b'\n' if suffix == '.jsonl' else b'')


def vscode_decisions(store):
    with store.connect() as db:
        return tuple(tuple(db.execute(
            f"SELECT d.* FROM {table} d JOIN observation o ON o.id=d.observation_id "
            "JOIN session s ON s.id=o.session_id WHERE s.harness='copilot-vscode' "
            "ORDER BY d.observation_id,d.measure")) for table in ('decision', 'quantity_decision'))


@pytest.mark.parametrize('suffix', ('.json', '.jsonl'))
def test_present_rejected_replacement_is_unresolved_until_repaired_after_reopen(tmp_path, suffix):
    app = Application(tmp_path / 'data', timezone='UTC')
    locator = str(chat_path(tmp_path, suffix))
    query = ReportQuery(None, AllTime())
    app.storage.import_source(locator, chat_bytes(suffix, large=False))
    expected = vscode_decisions(app.storage)
    assert expected[0] and expected[1]
    before = app.report(query)
    assert (before.tokens.input.known, before.tokens.output.known) == (12, 5)
    assert 'unresolved_evidence' not in {item.code for item in before.coverage}
    revision = app.storage.import_source(locator, malformed_chat(suffix))
    with app.storage.connect() as db:
        assert db.execute('SELECT session_id,availability FROM source_generation ORDER BY generation DESC LIMIT 1').one() == (None, 'available')
        assert db.execute('SELECT d.code FROM diagnostic d JOIN source_generation g ON g.id=d.source_id ORDER BY g.generation DESC LIMIT 1').one()[0] == (
            'malformed_chat_operation' if suffix == '.jsonl' else 'malformed_chat_state')
    for reopened in (False, True):
        if reopened:
            app.close()
            app = Application(tmp_path / 'data', timezone='UTC')
        with app.storage.connect(write=True) as db:
            app.storage._reconcile(db)
        decisions = vscode_decisions(app.storage)
        assert {row[2] for group in decisions for row in group} == {'unresolved'}
        assert app.storage.import_source(locator, malformed_chat(suffix)) == revision
        data = app.storage.report_input(query)
        assert data.contributions and not any(item.saved_history for item in data.contributions)
        report = app.report(query)
        assert report == build_report(data.contributions, revision=data.revision, query=query, catalog=app.catalog)
        assert (report.tokens.input.known, report.tokens.output.known) == (0, 0)
        assert not any(row.known for row in report.quantities)
        assert 'unresolved_evidence' in {item.code for item in report.coverage}
    revision = app.storage.import_source(locator, chat_bytes(suffix))
    assert vscode_decisions(app.storage) == expected
    repaired = app.report(query)
    assert (repaired.tokens.input.known, repaired.tokens.output.known) == (12, 5)
    assert 'unresolved_evidence' not in {item.code for item in repaired.coverage}
    app.close()
    app = Application(tmp_path / 'data', timezone='UTC')
    assert app.storage.import_source(locator, chat_bytes(suffix)) == revision
    assert app.report(query) == repaired
    app.close()


@pytest.mark.parametrize('rejected_suffix', ('.json', '.jsonl'))
def test_current_valid_alternate_remains_selected_when_other_candidate_is_rejected(tmp_path, rejected_suffix):
    store = Storage(tmp_path / 'ledger.duckdb')
    for suffix in ('.json', '.jsonl'):
        store.import_source(str(chat_path(tmp_path, suffix)), chat_bytes(suffix, large=False))
    expected = vscode_decisions(store)
    store.import_source(str(chat_path(tmp_path, rejected_suffix)), malformed_chat(rejected_suffix))
    assert vscode_decisions(store) == expected
    with store.connect(write=True) as db:
        store._reconcile(db)
    store.close()
    store = Storage(store.path)
    assert vscode_decisions(store) == expected
    store.close()


@pytest.mark.parametrize('suffix', ('.json', '.jsonl'))
def test_all_missing_valid_vscode_usage_remains_saved_history(tmp_path, suffix):
    store = Storage(tmp_path / 'ledger.duckdb')
    locator = str(chat_path(tmp_path, suffix))
    store.import_source(locator, chat_bytes(suffix))
    expected = vscode_decisions(store)
    store.mark_missing((locator,))
    with store.connect(write=True) as db:
        store._reconcile(db)
    store.close()
    store = Storage(store.path)
    assert vscode_decisions(store) == expected
    data = store.report_input(ReportQuery(None, AllTime()))
    assert data.contributions and all(item.saved_history for item in data.contributions)
    store.close()


def test_vscode_rejected_replacement_and_repair_preserve_other_harness_decisions(tmp_path):
    fixtures = Path(__file__).parent / 'fixtures'
    store = Storage(tmp_path / 'ledger.duckdb')
    store.import_sources((
        (str(tmp_path / 'pi.jsonl'), (fixtures / 'pi/source/ordinary.jsonl').read_bytes()),
        (str(tmp_path / 'codex.jsonl'), (fixtures / 'codex/mixed.jsonl').read_bytes()),
        (str(tmp_path / '11111111-1111-4111-8111-111111111111.jsonl'), (fixtures / 'claude/versioned.jsonl').read_bytes()),
        (str(tmp_path / 'session-state/11111111-1111-4111-8111-111111111111/events.jsonl'),
         (fixtures / 'copilot_cli/current/events.jsonl').read_bytes()),
    ))
    with store.connect() as db:
        assert {row[0] for row in db.execute('SELECT DISTINCT harness FROM session')} == {'pi', 'codex', 'claude', 'copilot-cli'}
        before = tuple(tuple(db.execute(
            f'SELECT d.rowid,d.* FROM {table} d ORDER BY d.rowid'))
            for table in ('decision', 'quantity_decision'))
    locator = str(chat_path(tmp_path, '.json'))
    for data in (chat_bytes('.json'), malformed_chat('.json'), chat_bytes('.json')):
        store.import_source(locator, data)
        with store.connect() as db:
            after = tuple(tuple(db.execute(
                f'SELECT d.rowid,d.* FROM {table} d JOIN observation o ON o.id=d.observation_id '
                "JOIN session s ON s.id=o.session_id WHERE s.harness<>'copilot-vscode' ORDER BY d.rowid"))
                for table in ('decision', 'quantity_decision'))
        assert after == before
    store.close()
