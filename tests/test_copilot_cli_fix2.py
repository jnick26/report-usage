"""Pinned synthetic regressions for Task 5 fix2; no real-source qualification."""
import json

import pytest
from fastapi.testclient import TestClient

from harness_usage.application import Application
from harness_usage.copilot_cli_reader import CopilotCLIReadBatch, read_copilot_cli
from harness_usage.copilot_cli_transcript import parse_copilot_cli_transcript
from harness_usage.domain import Unknown
from harness_usage.pi_reader import RejectedSource
from harness_usage.reporting import AllTime, ReportQuery
from harness_usage.source_input import SourcePayload, detect_source
from harness_usage.storage import Storage
from harness_usage.transcript import TranscriptUnavailable
from harness_usage.web import create_app
from test_copilot_cli_source import EVENTS, SESSION, cli_locator, event_bytes, event_rows


def read(tmp_path, rows):
    return read_copilot_cli(event_bytes(rows), locator=cli_locator(tmp_path))


def states(store, model='model-alpha'):
    with store.connect() as db:
        return tuple(db.execute(
            "SELECT d.state,v.amount,d.reason FROM decision d "
            "JOIN observation o ON o.id=d.observation_id "
            "JOIN token_value v ON v.observation_id=o.id AND v.measure=d.measure "
            "WHERE o.model=? AND d.measure='output' ORDER BY d.state,v.amount", (model,)))


@pytest.mark.parametrize('context', ['absent', {}, {'gitRoot': '/fixture/repo'}, {'cwd': None},
                                     {'cwd': ''}, {'cwd': 'relative'}, {'cwd': 7},
                                     {'cwd': '/fixture/work-a'}])
def test_f2_context_presence_is_shared(tmp_path, context):
    rows = event_rows()[:5]
    if context == 'absent':
        rows[0]['data'].pop('context')
    else:
        rows[0]['data']['context'] = context
    valid = context in ('absent', {'cwd': '/fixture/work-a'})
    result = read(tmp_path, rows)
    assert isinstance(result, CopilotCLIReadBatch) == valid
    if valid:
        assert 'SYNTHETIC_USER' in repr(parse_copilot_cli_transcript(event_bytes(rows), SESSION, SESSION))
    else:
        with pytest.raises(TranscriptUnavailable):
            parse_copilot_cli_transcript(event_bytes(rows), SESSION, SESSION)


@pytest.mark.parametrize('field', ['requests', 'usage'])
@pytest.mark.parametrize('bad', ['absent', None, []])
def test_f2_required_model_containers_never_fall_back_or_violate_state(tmp_path, field, bad):
    rows = event_rows()
    metric = rows[-1]['data']['modelMetrics']['model-alpha']
    if bad == 'absent':
        metric.pop(field)
    else:
        metric[field] = bad
    batch = read(tmp_path, rows)
    record, proof = next((record, proof) for record, proof in zip(batch.usage, batch.evidence)
                         if proof.line == 8 and record.model.model == 'model-alpha')
    assert (proof.state, proof.reason) == ('unresolved', 'invalid_cli_counter')
    assert isinstance(record.tokens.buckets.output, Unknown)
    store = Storage(tmp_path / 'ledger.duckdb')
    store.import_source(cli_locator(tmp_path), event_bytes(rows))
    store.import_source(cli_locator(tmp_path), event_bytes(rows))
    assert {row[0] for row in states(store)} == {'unresolved'}
    with store.connect() as db:
        assert db.execute("SELECT count(*) FROM copilot_cli_evidence WHERE state='usable' AND reason IS NOT NULL").one()[0] == 0
        assert db.execute("SELECT count(*) FROM quantity_decision d JOIN observation o ON o.id=d.observation_id WHERE o.model='model-alpha' AND d.state='selected'").one()[0] == 0
    store.close()


def test_f2_optional_count_absence_supersedes_older_count(tmp_path):
    rows = event_rows()
    rows[-1]['data']['modelMetrics']['model-alpha']['requests'] = {}
    store = Storage(tmp_path / 'ledger.duckdb')
    store.import_source(cli_locator(tmp_path), event_bytes(rows))
    assert any(row[:2] == ('selected', 25) for row in states(store))
    with store.connect() as db:
        assert db.execute("SELECT count(*) FROM quantity_decision d JOIN observation o ON o.id=d.observation_id WHERE o.model='model-alpha' AND d.state='selected'").one()[0] == 0
    store.close()


@pytest.mark.parametrize('field', ['codeChanges', 'totalApiDurationMs', 'totalNanoAiu', 'modelMetrics'])
def test_f2_required_shutdown_and_agent_cores(tmp_path, field):
    rows = event_rows()
    if field == 'codeChanges':
        rows[-1]['data'][field] = {}
        scope = 'session'
    else:
        rows[-1]['data']['agentMetrics']['main'].pop(field)
        scope = 'agent'
    batch = read(tmp_path, rows)
    proofs = [proof for proof in batch.evidence if proof.line == 8 and proof.scope == scope]
    assert any(proof.state == 'unresolved' and proof.reason for proof in proofs)
    store = Storage(tmp_path / 'ledger.duckdb')
    store.import_source(cli_locator(tmp_path), event_bytes(rows))
    if field == 'codeChanges':
        assert {row[0] for row in states(store)} == {'unresolved'}
    store.close()


@pytest.mark.parametrize('where,field', [('model', 'totalNanoAiu'), ('agent', 'totalNanoAiu'),
                                         ('agent', 'totalApiDurationMs')])
def test_f2_native_scalar_canaries_never_persist(tmp_path, caplog, where, field):
    rows = event_rows()
    target = (rows[-1]['data']['modelMetrics']['model-alpha'] if where == 'model'
              else rows[-1]['data']['agentMetrics']['main'])
    target[field] = {'SCALAR_SHAPE_CANARY': 7}
    batch = read(tmp_path, rows)
    assert 'SCALAR_SHAPE_CANARY' not in repr(batch)
    assert any(proof.state == 'unresolved' for proof in batch.evidence if proof.line == 8)
    path = tmp_path / 'ledger.duckdb'
    store = Storage(path)
    store.import_source(cli_locator(tmp_path), event_bytes(rows))
    with store.connect() as db:
        assert 'SCALAR_SHAPE_CANARY' not in repr(tuple(db.execute('SELECT * FROM copilot_cli_evidence')))
    store.close()
    assert b'SCALAR_SHAPE_CANARY' not in path.read_bytes()
    assert 'SCALAR_SHAPE_CANARY' not in caplog.text


@pytest.mark.parametrize('reverse', [False, True])
@pytest.mark.parametrize('mutation', ['currentModel', 'agent', 'duration'])
def test_f2_all_safe_event_facts_conflict_and_availability_recovers(tmp_path, reverse, mutation):
    one, two = event_rows(), event_rows()
    if mutation == 'agent':
        two[-1]['data']['agentMetrics']['main']['totalNanoAiu'] = 999
    elif mutation == 'duration':
        two[-1]['data']['totalApiDurationMs'] = 999
    else:
        two[-1]['data']['currentModel'] = 'other-model'
    sources = [SourcePayload(cli_locator(tmp_path / 'one'), event_bytes(one)),
               SourcePayload(cli_locator(tmp_path / 'two'), event_bytes(two))]
    path = tmp_path / 'ledger.duckdb'
    store = Storage(path)
    store.import_sources(tuple(reversed(sources)) if reverse else tuple(sources))
    assert {row[0] for row in states(store)} == {'unresolved'}
    store.close()
    store = Storage(path)
    assert {row[0] for row in states(store)} == {'unresolved'}
    store.mark_missing((sources[1].locator,))
    assert [row[1] for row in states(store) if row[0] == 'selected'] == [25]
    store.close()
    store = Storage(path)
    assert [row[1] for row in states(store) if row[0] == 'selected'] == [25]
    store.import_source(sources[1].locator, sources[1].data)
    assert {row[0] for row in states(store)} == {'unresolved'}
    store.close()
    store = Storage(path)
    assert {row[0] for row in states(store)} == {'unresolved'}
    store.close()


@pytest.mark.parametrize('reverse', [False, True])
@pytest.mark.parametrize('remove_model', [False, True])
def test_f2_common_event_extension_uses_appearances_not_local_lines(tmp_path, reverse, remove_model):
    rows = event_rows()
    rows[4]['data']['modelMetrics']['model-alpha']['requests'].pop('count')
    # The extra older conversation rows deliberately do not change common controls.
    extras = [{'id': f'00000000-0000-4000-8000-{100+i:012}', 'parentId': rows[2]['id'],
               'timestamp': rows[2]['timestamp'], 'type': 'user.message', 'data': {'content': 'inserted'}}
              for i in range(10)]
    old = rows[:3] + extras + rows[3:5]
    new = json.loads(json.dumps(rows))
    if remove_model:
        new[-1]['data']['modelMetrics'].pop('model-beta')
    sources = (SourcePayload(cli_locator(tmp_path / 'old'), event_bytes(old)),
               SourcePayload(cli_locator(tmp_path / 'new'), event_bytes(new)))
    path = tmp_path / 'ledger.duckdb'
    store = Storage(path)
    store.import_sources(tuple(reversed(sources)) if reverse else sources)
    for _ in range(2):
        assert [row[1] for row in states(store) if row[0] == 'selected'] == [25]
        if remove_model:
            assert {row[0] for row in states(store, 'model-beta')} == {'unresolved'}
        store.close()
        store = Storage(path)
    store.close()


@pytest.mark.parametrize('reverse', [False, True])
def test_f2_optional_count_does_not_reverse_sparse_common_event_extension(tmp_path, reverse):
    rows = event_rows()
    rows[4]['data']['modelMetrics']['model-alpha']['requests'].pop('count')
    sparse = [rows[0], rows[3], rows[4]]
    sources = (SourcePayload(cli_locator(tmp_path / 'sparse'), event_bytes(sparse)),
               SourcePayload(cli_locator(tmp_path / 'full'), event_bytes(rows)))
    path = tmp_path / 'ledger.duckdb'
    store = Storage(path)
    store.import_sources(tuple(reversed(sources)) if reverse else sources)
    assert [row[1] for row in states(store) if row[0] == 'selected'] == [25]
    store.close()
    store = Storage(path)
    assert [row[1] for row in states(store) if row[0] == 'selected'] == [25]
    store.close()


def test_f2_conflicting_physical_orders_cannot_form_a_proven_dag(tmp_path):
    rows = event_rows()
    a, b, final = rows[3], rows[5], rows[-1]
    b['parentId'] = a['parentId']
    b['data'] = json.loads(json.dumps(a['data']))
    final['parentId'] = b['id']
    one = rows[:3] + [a, b, final]
    two = rows[:3] + [b, a, final]
    store = Storage(tmp_path / 'ledger.duckdb')
    store.import_sources((SourcePayload(cli_locator(tmp_path / 'one'), event_bytes(one)),
                          SourcePayload(cli_locator(tmp_path / 'two'), event_bytes(two))))
    with store.connect() as db:
        assert db.execute("SELECT count(*) FROM quantity_decision WHERE measure='nano_aiu' AND state='selected'").one()[0] == 0
    store.close()
@pytest.mark.parametrize('reverse', [False, True])
def test_f2_workspace_conflict_clears_parent_and_proven_children(tmp_path, reverse):
    a = [row for row in event_rows() if row['type'] != 'session.context_changed']
    b = json.loads(json.dumps(a))
    b[0]['data']['context'] = {'cwd': '/fixture/work-b', 'gitRoot': '/fixture/work-b'}
    sources = [SourcePayload(cli_locator(tmp_path / 'a'), event_bytes(a)),
               SourcePayload(cli_locator(tmp_path / 'b'), event_bytes(b))]
    app = Application(tmp_path / 'app')
    app.storage.import_sources(tuple(reversed(sources)) if reverse else tuple(sources))
    app.storage.import_source(cli_locator(tmp_path / 'again'), event_bytes(a))
    app._assign_projects()
    with app.storage.connect() as db:
        assert tuple(db.execute("SELECT cwd,attribution_reason FROM session_view WHERE harness='copilot-cli' ORDER BY id")) == (
            (None, 'workspace_changed_cumulative_scope'), (None, 'workspace_changed_cumulative_scope'))
    app.close()
    reopened = Storage(tmp_path / 'app' / 'ledger.duckdb')
    with reopened.connect() as db:
        assert tuple(db.execute("SELECT cwd,attribution_reason FROM session_view WHERE harness='copilot-cli' ORDER BY id")) == (
            (None, 'workspace_changed_cumulative_scope'), (None, 'workspace_changed_cumulative_scope'))
    reopened.close()


@pytest.mark.parametrize('payload', [b'', b'{bad', ('id: ' + SESSION + '\n').encode()])
def test_f2_workspace_only_dispatch_scan_and_unavailable_coverage(tmp_path, payload):
    source = tmp_path / 'root' / 'session-state' / SESSION / 'workspace.yaml'
    source.parent.mkdir(parents=True)
    source.write_bytes(payload)
    app = Application(tmp_path / 'app')
    sources = tuple(app._scan((str(tmp_path / 'root'),)))
    assert len(sources) == 1
    assert detect_source(sources[0]) == 'copilot-cli'
    app.storage.import_sources(sources)
    with app.storage.connect() as db:
        assert db.execute("SELECT count(*) FROM decision WHERE state='selected'").one()[0] == 6
        assert db.execute("SELECT count(*) FROM quantity_decision WHERE state='selected'").one()[0] == 3
        assert db.execute("SELECT count(*) FROM quantity_value WHERE amount_decimal IS NOT NULL").one()[0] == 0
    assert 'events_unavailable' in app.storage.report_input(ReportQuery(None, AllTime())).diagnostics
    app.storage.mark_missing((str(source),))
    app.storage.import_sources(sources)
    app.close()
    reopened = Storage(tmp_path / 'app' / 'ledger.duckdb')
    with reopened.connect() as db:
        assert db.execute("SELECT count(*) FROM decision WHERE state='selected'").one()[0] == 6
        assert db.execute("SELECT count(*) FROM quantity_decision WHERE state='selected'").one()[0] == 3
    reopened.close()


@pytest.mark.parametrize('tail', [b'{bad', b'{"id":'])
def test_f2_partial_tail_keeps_shutdown_exact_at_event(tmp_path, tail):
    payload = event_bytes(event_rows()[:5]) + tail
    batch = read_copilot_cli(payload, locator=cli_locator(tmp_path))
    assert 'cli_partial_lifetime' in {item.code for item in batch.diagnostics}
    assert batch.pending_tail == (tail == b'{"id":')
    path = tmp_path / 'ledger.duckdb'
    store = Storage(path)
    store.import_source(cli_locator(tmp_path), payload)
    for _ in range(2):
        with store.connect() as db:
            assert tuple(db.execute("SELECT q.amount_decimal,q.lower_bound FROM quantity_value q JOIN quantity_decision d ON d.observation_id=q.observation_id AND d.measure=q.measure WHERE d.state='selected' ORDER BY q.measure,q.amount_decimal")) == (
                ('120', 0), ('2', 0), ('1', 0), ('2', 0))
        store.mark_missing((cli_locator(tmp_path),))
        store.import_source(cli_locator(tmp_path), payload)
        store.close()
        store = Storage(path)
    store.close()


@pytest.mark.parametrize('reverse', [False, True])
def test_f2_incomparable_removal_cannot_leave_one_models_stale_snapshot(tmp_path, reverse):
    one, two = event_rows(), event_rows()
    one[-1]['id'] = '00000000-0000-4000-8000-000000000091'
    two[-1]['id'] = '00000000-0000-4000-8000-000000000092'
    two[-1]['data']['modelMetrics'].pop('model-beta')
    sources = (SourcePayload(cli_locator(tmp_path / 'a'), event_bytes(one)),
               SourcePayload(cli_locator(tmp_path / 'b'), event_bytes(two)))
    store = Storage(tmp_path / 'ledger.duckdb')
    store.import_sources(tuple(reversed(sources)) if reverse else sources)
    assert {row[0] for row in states(store, 'model-beta')} == {'unresolved'}
    with store.connect() as db:
        assert db.execute("SELECT count(*) FROM quantity_decision d JOIN observation o ON o.id=d.observation_id WHERE o.model='model-beta' AND d.state='selected'").one()[0] == 0
    store.close()


def test_f2_unavailable_metadata_survives_multi_session_full_reconciliation(tmp_path):
    store = Storage(tmp_path / 'ledger.duckdb')
    other = '22222222-2222-4222-8222-222222222222'
    sidecar = cli_locator(tmp_path / 'other', other, 'workspace.yaml')
    store.import_sources((SourcePayload(cli_locator(tmp_path), EVENTS), SourcePayload(sidecar, b'')))
    with store.connect(write=True) as db:
        store._reconcile(db)
    with store.connect() as db:
        assert db.execute("SELECT count(*) FROM decision d JOIN observation o ON o.id=d.observation_id WHERE o.session_id=? AND d.state='selected'", ('copilot-cli:' + other,)).one()[0] == 6
        assert db.execute("SELECT count(*) FROM quantity_decision d JOIN observation o ON o.id=d.observation_id WHERE o.session_id=? AND d.state='selected'", ('copilot-cli:' + other,)).one()[0] == 3
    store.close()


def test_f2_start_only_supersedes_workspace_placeholder_then_real_events(tmp_path):
    store = Storage(tmp_path / 'ledger.duckdb')
    store.import_source(cli_locator(tmp_path, name='workspace.yaml'), b'')
    store.import_source(cli_locator(tmp_path), event_bytes(event_rows()[:1]))
    with store.connect() as db:
        assert db.execute("SELECT count(*) FROM decision WHERE state='selected'").one()[0] == 6
        assert db.execute("SELECT count(*) FROM quantity_decision WHERE state='selected'").one()[0] == 3
    store.import_source(cli_locator(tmp_path), EVENTS)
    assert [row[1] for row in states(store) if row[0] == 'selected'] == [25]
    store.close()


def test_f2_injected_quantity_write_failure_rolls_back_then_retries(tmp_path, monkeypatch):
    from harness_usage.database import Connection
    original = Connection.executemany
    def fail(self, sql, rows):
        if sql.startswith('INSERT INTO quantity_decision'):
            raise RuntimeError('synthetic interruption')
        return original(self, sql, rows)
    store = Storage(tmp_path / 'ledger.duckdb')
    with monkeypatch.context() as patch:
        patch.setattr(Connection, 'executemany', fail)
        with pytest.raises(RuntimeError, match='synthetic interruption'):
            store.import_source(cli_locator(tmp_path), EVENTS)
    with store.connect() as db:
        assert db.execute('SELECT count(*) FROM copilot_cli_evidence').one()[0] == 0
        assert db.execute('SELECT count(*) FROM source_generation').one()[0] == 0
        assert db.execute('SELECT revision FROM ledger_meta').one()[0] == 0
    store.import_source(cli_locator(tmp_path), EVENTS)
    assert [row[1] for row in states(store) if row[0] == 'selected'] == [25]
    store.close()


def tool_rows(completion):
    rows = event_rows()
    rows.extend([
        {'id': '00000000-0000-4000-8000-000000000009', 'parentId': rows[-1]['id'],
         'timestamp': '2026-09-16T00:00:00.080Z', 'type': 'tool.execution_start',
         'data': {'toolCallId': 'tool', 'toolName': 'fixture', 'arguments': {}}},
        {'id': '00000000-0000-4000-8000-000000000010',
         'parentId': '00000000-0000-4000-8000-000000000009',
         'timestamp': '2026-09-16T00:00:00.090Z', 'type': 'tool.execution_complete',
         'data': {'toolCallId': 'tool', **completion}}])
    return rows


@pytest.mark.parametrize('success,result,error,valid', [(True, True, False, True), (False, False, True, True),
    (False, True, False, False), (True, False, True, False), (True, True, True, False), (False, True, True, False)])
def test_f2_completion_status_agrees_with_payload(success, result, error, valid):
    completion = {'success': success}
    if result:
        completion['result'] = {'content': 'RESULT_CANARY'}
    if error:
        completion['error'] = {'message': 'ERROR_CANARY'}
    transcript = parse_copilot_cli_transcript(event_bytes(tool_rows(completion)), SESSION, SESSION)
    rendered = repr(transcript.entries)
    assert ('CANARY' in rendered) == valid
    if valid:
        assert ("status='succeeded'" if success else "status='failed'") in rendered
    else:
        assert 'Unsupported Copilot CLI entry' in rendered


@pytest.mark.parametrize('mutation,status', [('prestart', 409), ('nonfinite', 200), ('duplicate', 200),
                                             ('chain', 200), ('envelope', 200), ('completion', 200)])
def test_f2_protected_http_and_direct_parser_privacy(tmp_path, mutation, status):
    rows = event_rows()
    hostile = {'id': '00000000-0000-4000-8000-000000000099', 'parentId': rows[-1]['id'],
               'timestamp': '2026-09-16T00:00:00.090Z', 'type': 'assistant.message',
               'data': {'messageId': 'message', 'content': 'HTTP_BODY_CANARY', 'toolRequests': [
                   {'toolCallId': 'tool', 'name': 'fixture', 'arguments': {'value': 'ARG_CANARY'}}]}}
    if mutation == 'prestart':
        hostile['timestamp'] = rows[0]['data']['startTime']
        hostile['parentId'] = '00000000-0000-4000-8000-000000000098'
        rows.insert(0, hostile)
    elif mutation == 'completion':
        rows = tool_rows({'success': False, 'result': {'content': 'HTTP_BODY_CANARY'}})
    else:
        if mutation == 'nonfinite':
            hostile['data']['toolRequests'][0]['arguments']['value'] = float('nan')
        elif mutation == 'chain':
            hostile['parentId'] = hostile['id']
        elif mutation == 'envelope':
            hostile.pop('parentId')
        rows.append(hostile)
    payload = event_bytes(rows)
    if mutation == 'duplicate':
        payload = payload.replace(b'"messageId":"message"', b'"messageId":"message","messageId":"other"')
    if status == 409:
        assert isinstance(read_copilot_cli(payload, locator=cli_locator(tmp_path)), RejectedSource)
        with pytest.raises(TranscriptUnavailable) as error:
            parse_copilot_cli_transcript(payload, SESSION, SESSION)
        assert error.value.kind == 'changed'
    else:
        projected = parse_copilot_cli_transcript(payload, SESSION, SESSION)
        assert 'HTTP_BODY_CANARY' not in repr(projected)
        assert 'ARG_CANARY' not in repr(projected)
    source = tmp_path / 'root' / 'session-state' / SESSION / 'events.jsonl'
    source.parent.mkdir(parents=True)
    source.write_bytes(EVENTS)
    app = Application(tmp_path / 'app')
    app.set_roots((str(tmp_path / 'root'),))
    app.storage.import_source(str(source), EVENTS)
    source.write_bytes(payload)
    with TestClient(create_app(app), base_url='http://127.0.0.1:8765') as client:
        url = f'/sessions/copilot-cli%3A{SESSION}/transcript'
        for suffix in ('', '.html'):
            response = client.get(url + suffix)
            assert response.status_code == status
            assert 'HTTP_BODY_CANARY' not in response.text and 'ARG_CANARY' not in response.text
        assert client.get(url + '?branch=invalid').status_code == 422
        source.unlink()
        assert client.get(url).status_code == 404
    app.close()
