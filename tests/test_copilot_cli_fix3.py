"""Storage lifecycle regressions for the pinned synthetic CLI profile."""
import json

import pytest
from fastapi.testclient import TestClient

from harness_usage.application import Application
from harness_usage.copilot_cli_reader import CopilotCLIReadBatch, read_copilot_cli
from harness_usage.copilot_cli_transcript import parse_copilot_cli_transcript
from harness_usage.pi_reader import RejectedSource
from harness_usage.reporting import AllTime, ReportQuery
from harness_usage.source_input import SourcePayload
from harness_usage.storage import Storage
from harness_usage.transcript import TranscriptUnavailable
from harness_usage.web import create_app
from test_copilot_cli_source import EVENTS, SESSION, cli_locator, event_bytes, event_rows


def selected_rows(store):
    with store.connect() as db:
        tokens = tuple(db.execute(
            "SELECT d.measure,v.amount FROM decision d JOIN token_value v "
            "ON v.observation_id=d.observation_id AND v.measure=d.measure "
            "WHERE d.state='selected' ORDER BY d.measure,v.amount"))
        quantities = tuple(db.execute(
            "SELECT d.measure,q.amount_decimal FROM quantity_decision d JOIN quantity_value q "
            "ON q.observation_id=d.observation_id AND q.measure=d.measure "
            "WHERE d.state='selected' ORDER BY d.measure,q.amount_decimal"))
        return tokens, quantities


def test_f3_rejected_available_replacement_retires_superseded_decisions(tmp_path):
    path = tmp_path / 'ledger.duckdb'
    locator = cli_locator(tmp_path)
    store = Storage(path)
    store.import_source(locator, EVENTS)
    expected = selected_rows(store)
    assert expected[0] and expected[1]
    replacement = event_rows()
    replacement[0]['data']['sessionId'] = '22222222-2222-4222-8222-222222222222'
    store.import_source(locator, event_bytes(replacement))
    with store.connect() as db:
        assert db.execute('SELECT session_id,availability FROM source_generation ORDER BY generation DESC LIMIT 1').one() == (None, 'available')
        assert db.execute("SELECT count(*) FROM diagnostic WHERE code='conflicting_session_identity'").one()[0] == 1
    assert selected_rows(store) == ((), ())
    with store.connect(write=True) as db:
        store._reconcile(db)
    assert selected_rows(store) == ((), ())
    assert not store.report_input(ReportQuery(None, AllTime())).contributions
    store.close()
    store = Storage(path)
    assert selected_rows(store) == ((), ())
    store.import_source(locator, EVENTS)
    assert selected_rows(store) == expected
    store.close()
    store = Storage(path)
    assert selected_rows(store) == expected
    store.close()


def test_f3_missing_valid_source_keeps_saved_only_decisions(tmp_path):
    path = tmp_path / 'ledger.duckdb'
    locator = cli_locator(tmp_path)
    store = Storage(path)
    store.import_source(locator, EVENTS)
    expected = selected_rows(store)
    store.mark_missing((locator,))
    assert selected_rows(store) == expected
    assert 'saved_history' in store.report_input(ReportQuery(None, AllTime())).diagnostics
    with store.connect(write=True) as db:
        store._reconcile(db)
    assert selected_rows(store) == expected
    store.close()
    store = Storage(path)
    assert selected_rows(store) == expected
    store.import_source(locator, EVENTS)
    assert selected_rows(store) == expected
    store.close()


def family_rows(store):
    with store.connect() as db:
        return tuple(db.execute(
            "SELECT cwd,project_id,worktree,attribution_reason FROM session_view "
            "WHERE harness='copilot-cli' ORDER BY id"))


@pytest.mark.parametrize('late', [False, True])
def test_f3_new_child_gets_sticky_workspace_conflict_before_assignment(tmp_path, late):
    app = Application(tmp_path / 'app')
    if late:
        a = [row for row in event_rows() if row['type'] != 'session.context_changed']
        a[2].pop('agentId')
        b = json.loads(json.dumps(a))
        b[0]['data']['context'] = {'cwd': '/fixture/work-b', 'gitRoot': '/fixture/work-b'}
        app.storage.import_source(cli_locator(tmp_path / 'a'), event_bytes(a))
        app.storage.import_source(cli_locator(tmp_path / 'b'), event_bytes(b))
        assert len(family_rows(app.storage)) == 1
        a[2]['agentId'] = 'agent-child'
        app.storage.import_source(cli_locator(tmp_path / 'a'), event_bytes(a))
    else:
        app.storage.import_source(cli_locator(tmp_path), EVENTS)
    expected = ((None, None, None, 'workspace_changed_cumulative_scope'),) * 2
    assert family_rows(app.storage) == expected
    app._assign_projects()
    assert family_rows(app.storage) == expected
    path = app.storage.path
    app.close()
    store = Storage(path)
    assert family_rows(store) == expected
    store.close()


def cycle_rows(control=False):
    rows = event_rows()
    first, second = '00000000-0000-4000-8000-000000000091', '00000000-0000-4000-8000-000000000092'
    rows.extend([
        {'id': first, 'parentId': second, 'timestamp': '2026-09-16T00:00:00.080Z',
         'type': 'user.message', 'data': {'content': 'CYCLE_FIRST_CANARY'}},
        {'id': second, 'parentId': first, 'timestamp': '2026-09-16T00:00:00.090Z',
         'type': 'user.message', 'data': {'content': 'CYCLE_SECOND_CANARY'}},
        {'id': '00000000-0000-4000-8000-000000000093', 'parentId': second,
         'timestamp': '2026-09-16T00:00:00.100Z',
         'type': 'user.message', 'data': {'content': 'CYCLE_DESCENDANT_CANARY'}},
    ])
    if control:
        rows[-3]['type'] = 'session.usage_checkpoint'
        rows[-3]['data'] = {'totalNanoAiu': 998, 'totalPremiumRequests': 8}
        rows[-2]['type'] = 'session.shutdown'
        rows[-2]['data'] = json.loads(json.dumps(rows[7]['data']))
        body = rows[-2]['data']
        body['totalNanoAiu'], body['totalPremiumRequests'] = 999, 9
        body['modelMetrics']['model-alpha']['usage'].update(inputTokens=999, outputTokens=999)
        body['modelMetrics']['model-alpha']['requests']['count'] = 9
    return rows


def test_f3_known_cycle_and_descendants_cannot_project_direct_or_http_content(tmp_path):
    payload = event_bytes(cycle_rows())
    batch = read_copilot_cli(payload, locator=cli_locator(tmp_path))
    assert isinstance(batch, CopilotCLIReadBatch)
    assert {item.line for item in batch.entries}.isdisjoint({9, 10, 11})
    assert {item.line for item in batch.diagnostics if item.code == 'invalid_event_chain'} >= {9, 10, 11}
    assert 'CYCLE_' not in repr(parse_copilot_cli_transcript(payload, SESSION, SESSION))
    source = tmp_path / 'sources' / 'session-state' / SESSION / 'events.jsonl'
    source.parent.mkdir(parents=True)
    source.write_bytes(payload)
    app = Application(tmp_path / 'app')
    app.set_roots((str(tmp_path / 'sources'),))
    app.storage.import_source(str(source), payload)
    with TestClient(create_app(app), base_url='http://127.0.0.1:8765') as client:
        for suffix in ('transcript', 'transcript.html'):
            response = client.get(f'/sessions/copilot-cli%3A{SESSION}/{suffix}')
            assert response.status_code == 200
            assert 'CYCLE_' not in response.text
            assert 'SYNTHETIC_USER' in response.text
    app.close()


def test_f3_acyclic_forward_parent_keeps_descendants_and_shutdown_transcript(tmp_path):
    rows = event_rows()
    reordered = rows[:3] + [rows[4], rows[3], *rows[5:]]
    payload = event_bytes(reordered)
    batch = read_copilot_cli(payload, locator=cli_locator(tmp_path))
    assert isinstance(batch, CopilotCLIReadBatch)
    assert {item.line for item in batch.entries} == set(range(1, 9))
    assert 'invalid_event_chain' not in {item.code for item in batch.diagnostics}
    assert sum(item.source_kind == 'shutdown' for item in batch.evidence) >= 2
    transcript = parse_copilot_cli_transcript(payload, SESSION, SESSION)
    assert 'SYNTHETIC_USER' in repr(transcript)


def test_f3_graph_profile_upgrade_reprojects_legacy_cli_and_third_import_is_noop(tmp_path):
    rows = event_rows()
    payload = event_bytes(rows[:3] + [rows[4], rows[3], *rows[5:]])
    locator = cli_locator(tmp_path)
    store = Storage(tmp_path / 'ledger.duckdb')
    first_revision = store.import_source(locator, payload)
    with store.connect(write=True) as db:
        db.execute(
            "UPDATE source_generation SET profile='copilot-cli-events/e60d903-shape-1' WHERE locator=?",
            (locator,))
    before = selected_rows(store)
    second_revision = store.import_source(locator, payload)
    assert second_revision > first_revision
    assert selected_rows(store) == before
    with store.connect() as db:
        assert tuple(db.execute(
            'SELECT generation,profile FROM source_generation ORDER BY generation')) == (
                (0, 'copilot-cli-events/e60d903-shape-1'),
                (1, 'copilot-cli-events/e60d903-shape-2'),
            )
    third_revision = store.import_source(locator, payload)
    assert third_revision == second_revision
    with store.connect() as db:
        assert db.execute('SELECT count(*) FROM source_generation').one()[0] == 2
    store.close()


def test_f3_known_cycle_control_cannot_authorize_counters(tmp_path):
    payload = event_bytes(cycle_rows(control=True))
    batch = read_copilot_cli(payload, locator=cli_locator(tmp_path))
    assert isinstance(batch, CopilotCLIReadBatch)
    assert all(item.line < 9 for item in batch.evidence)
    store = Storage(tmp_path / 'ledger.duckdb')
    store.import_source(cli_locator(tmp_path), payload)
    tokens, quantities = selected_rows(store)
    assert ('output', 25) in tokens and ('output', 999) not in tokens
    assert ('nano_aiu', '200') in quantities and ('nano_aiu', '999') not in quantities
    with store.connect() as db:
        assert db.execute('SELECT count(*) FROM copilot_cli_evidence WHERE line>=9').one()[0] == 0
    store.close()


@pytest.mark.parametrize('field,bad', [('pendingGitContext', 1), ('pendingGitContext', 'yes'),
                                     ('gitRoot', 7), ('gitRoot', {'CONTEXT_CANARY': 1})])
@pytest.mark.parametrize('start', [True, False])
def test_f3_present_invalid_context_members_never_settle(tmp_path, field, bad, start):
    rows = event_rows()[:5] if start else event_rows()
    context = rows[0]['data']['context'] if start else rows[6]['data']
    context[field] = bad
    batch = read_copilot_cli(event_bytes(rows), locator=cli_locator(tmp_path))
    if start:
        assert isinstance(batch, RejectedSource)
        with pytest.raises(TranscriptUnavailable):
            parse_copilot_cli_transcript(event_bytes(rows), SESSION, SESSION)
    else:
        assert isinstance(batch, CopilotCLIReadBatch)
        assert batch.session.cwd == '/fixture/work-a'
        assert any(item.code == 'invalid_cli_event' and item.line == 7 for item in batch.diagnostics)
        assert 'workspace_changed_cumulative_scope' not in {item.code for item in batch.diagnostics}
    assert 'CONTEXT_CANARY' not in repr(batch)


@pytest.mark.parametrize('reverse', [False, True])
@pytest.mark.parametrize('mutation', ['agent_list', 'agent_null', 'count', 'cost'])
def test_f3_optional_native_invalidity_differs_from_valid_absence(tmp_path, caplog, reverse, mutation):
    one, two = event_rows(), event_rows()
    if mutation.startswith('agent_'):
        one[-1]['data']['agentMetrics'] = {}
        two[-1]['data']['agentMetrics'] = [] if mutation == 'agent_list' else None
    else:
        one[-1]['data']['agentMetrics']['main']['modelMetrics']['model-alpha']['requests'].pop(mutation)
        two[-1]['data']['agentMetrics']['main']['modelMetrics']['model-alpha']['requests'][mutation] = (
            True if mutation == 'count' else {'NATIVE_MEMBER_CANARY': 3})
    batch = read_copilot_cli(event_bytes(two), locator=cli_locator(tmp_path))
    assert isinstance(batch, CopilotCLIReadBatch)
    assert any(item.line == 8 and item.state == 'unresolved' for item in batch.evidence)
    assert any(item.line == 8 and item.code == 'invalid_cli_counter' for item in batch.diagnostics)
    assert 'NATIVE_MEMBER_CANARY' not in repr(batch.evidence)
    sources = (SourcePayload(cli_locator(tmp_path / 'one'), event_bytes(one)),
               SourcePayload(cli_locator(tmp_path / 'two'), event_bytes(two)))
    path = tmp_path / 'ledger.duckdb'
    store = Storage(path)
    store.import_sources(tuple(reversed(sources)) if reverse else sources)
    with store.connect() as db:
        assert {row[0] for row in db.execute("SELECT d.state FROM decision d JOIN observation o ON o.id=d.observation_id WHERE d.measure='output' AND o.model='model-alpha'")} == {'unresolved'}
        assert 'NATIVE_MEMBER_CANARY' not in repr(tuple(db.execute('SELECT * FROM copilot_cli_evidence')))
    store.close()
    assert b'NATIVE_MEMBER_CANARY' not in path.read_bytes()
    assert 'NATIVE_MEMBER_CANARY' not in caplog.text
    store = Storage(path)
    assert ('output', 25) not in selected_rows(store)[0]
    store.close()
