import json
from decimal import Decimal
from pathlib import Path

import pytest

from harness_usage.copilot_cli_reader import (
    CLI_PROFILE, CopilotCLIReadBatch, read_copilot_cli, read_workspace_metadata,
)
from harness_usage.copilot_cli_transcript import parse_copilot_cli_transcript
from harness_usage.application import Application
from harness_usage.domain import Known, Unknown, Unassigned
from harness_usage.pi_reader import RejectedSource
from harness_usage.reporting import AllTime, ReportQuery
from harness_usage.source_input import SourcePayload, detect_source
from harness_usage.storage import Storage
from harness_usage.transcript import TranscriptUnavailable


FIXTURES = Path(__file__).parent / 'fixtures/copilot_cli'
EVENTS = (FIXTURES / 'current/events.jsonl').read_bytes()
WORKSPACE = (FIXTURES / 'current/workspace.yaml').read_bytes()
SESSION = '11111111-1111-4111-8111-111111111111'


def cli_locator(root, session=SESSION, name='events.jsonl'):
    return str(root / 'session-state' / session / name)


def event_rows(data=EVENTS):
    return [json.loads(line) for line in data.splitlines()]


def event_bytes(rows):
    return b''.join(json.dumps(row, separators=(',', ':')).encode() + b'\n' for row in rows)


def read_fixture(tmp_path, data=EVENTS, *, context=(('workspace.yaml', WORKSPACE),)):
    return read_copilot_cli(data, locator=cli_locator(tmp_path), context=context)


def test_reader_uses_pinned_shape_profile_and_does_not_project_start_writer_version(tmp_path):
    result = read_fixture(tmp_path)
    assert isinstance(result, CopilotCLIReadBatch)
    assert {item.profile for item in result.evidence} == {CLI_PROFILE}
    start = next(item for item in result.evidence if item.source_kind == 'metadata')
    assert (start.event_schema_version, start.writer_version) == ('1', '1.0.0-synthetic')
    assert all(item.writer_version is None for item in result.evidence if item.source_kind != 'metadata')


def test_reader_and_detector_accept_copilot_agent_producer(tmp_path):
    rows = event_rows()
    rows[0]['data']['producer'] = 'copilot-agent'
    payload = event_bytes(rows)
    locator = cli_locator(tmp_path)
    assert detect_source(SourcePayload(locator, payload)) == 'copilot-cli'
    assert isinstance(read_copilot_cli(payload, locator=locator), CopilotCLIReadBatch)


def test_reader_accepts_only_session_level_durable_controls_and_ignores_assistant_usage(tmp_path):
    rows = event_rows()
    rows.append({'id': '00000000-0000-4000-8000-000000000009',
                 'parentId': rows[-1]['id'], 'timestamp': '2026-09-16T00:00:00.080Z',
                 'type': 'assistant.usage', 'ephemeral': False,
                 'data': {'inputTokens': 999999}})
    checkpoint = json.loads(json.dumps(rows[3]))
    checkpoint['id'] = '00000000-0000-4000-8000-000000000010'
    checkpoint['ephemeral'] = True
    rows.append(checkpoint)
    result = read_fixture(tmp_path, event_bytes(rows))
    assert isinstance(result, CopilotCLIReadBatch)
    assert 'ephemeral_event_in_history' in {item.code for item in result.diagnostics}
    assert {item.event_id for item in result.evidence}.isdisjoint({rows[-1]['id'], rows[-2]['id']})


@pytest.mark.parametrize('agent_value', [None, '', 'main', 1, True, [], {}, 'x' * 257])
def test_cli_controls_reject_every_present_agent_id_shape(tmp_path, agent_value):
    rows = event_rows()[:5]
    rows[3]['agentId'] = agent_value
    result = read_fixture(tmp_path, event_bytes(rows))
    assert isinstance(result, CopilotCLIReadBatch)
    assert rows[3]['id'] not in {item.event_id for item in result.evidence}
    assert any(item.code == 'invalid_cli_event' and item.line == 4 for item in result.diagnostics)


def test_cli_checkpoint_only_import_is_lower_bound_and_transaction_safe(tmp_path):
    locator = cli_locator(tmp_path)
    store = Storage(tmp_path / 'checkpoint-only.duckdb')
    store.import_source(locator, event_bytes(event_rows()[:4]))
    with store.connect() as db:
        quantities = tuple(db.execute(
            "SELECT q.measure,q.amount_decimal,q.lower_bound,d.state FROM quantity_value q "
            "JOIN quantity_decision d ON d.observation_id=q.observation_id AND d.measure=q.measure "
            "WHERE d.state='selected' ORDER BY q.measure"))
        assert quantities == (('nano_aiu', '100', 1, 'selected'),
                             ('premium_requests', '2', 1, 'selected'))
        assert db.execute("SELECT count(*) FROM diagnostic WHERE line=0").one()[0] == 0
    store.close()
    reopened = Storage(tmp_path / 'checkpoint-only.duckdb')
    with reopened.connect() as db:
        assert db.execute("SELECT count(*) FROM quantity_decision WHERE state='selected'").one()[0] == 2
    reopened.close()


def test_reader_requires_one_start_matching_directory_and_shutdown_milliseconds(tmp_path):
    assert isinstance(read_fixture(tmp_path), CopilotCLIReadBatch)
    assert isinstance(read_copilot_cli(EVENTS, locator=cli_locator(tmp_path, '22222222-2222-4222-8222-222222222222')), RejectedSource)
    rows = event_rows()
    rows[-1]['data']['sessionStartTime'] += 1
    result = read_fixture(tmp_path, event_bytes(rows))
    assert isinstance(result, CopilotCLIReadBatch)
    assert 'incompatible_session_epoch' in {item.code for item in result.diagnostics}


def test_reader_bounds_safe_vectors_and_rejects_inexact_or_hostile_numbers(tmp_path):
    rows = event_rows()
    rows[-1]['data']['modelMetrics']['model-alpha']['usage']['inputTokens'] = True
    rows[-1]['data']['totalNanoAiu'] = -1
    result = read_fixture(tmp_path, event_bytes(rows))
    assert isinstance(result, CopilotCLIReadBatch)
    assert 'invalid_cli_counter' in {item.code for item in result.diagnostics}
    final = [item for item in result.evidence if item.event_id == rows[-1]['id']]
    assert all('true' not in item.counters_json and '-1' not in item.counters_json for item in final)


def test_cli_optional_request_fields_do_not_discard_model_tokens(tmp_path):
    rows = event_rows()
    for metric in rows[-1]['data']['modelMetrics'].values():
        metric['requests'].pop('count')
        metric['requests'].pop('cost')
    result = read_fixture(tmp_path, event_bytes(rows))
    assert isinstance(result, CopilotCLIReadBatch)
    models = {record.model.model: record for record, item in zip(result.usage, result.evidence, strict=True)
              if item.event_id == rows[-1]['id'] and item.scope == 'model'}
    assert set(models) == {'model-alpha', 'model-beta'}
    assert models['model-alpha'].tokens.buckets.input == Known(150)
    assert models['model-beta'].tokens.buckets.output == Known(40)
    assert all(value.state == 'unknown' and value.amount is None and value.reason == 'not_reported'
               for record in models.values() for value in record.quantities if value.measure == 'request_count')


@pytest.mark.parametrize('lexeme', ['0.' + '1' * 129, '1e-129', str(2**63)])
def test_cli_decimal_safety_bound_rejects_hostile_exact_lexemes(tmp_path, lexeme):
    payload = event_bytes(event_rows()).replace(b'"totalNanoAiu":200',
                                                 ('"totalNanoAiu":' + lexeme).encode(), 1)
    result = read_fixture(tmp_path, payload)
    assert isinstance(result, CopilotCLIReadBatch)
    assert 'invalid_cli_counter' in {item.code for item in result.diagnostics}
    assert lexeme not in repr(result.evidence)


def test_cli_conflicting_valid_epochs_are_unresolved_not_two_selected_histories(tmp_path):
    one = event_rows()
    two = event_rows()
    for row in two:
        if row['type'] == 'session.start':
            row['data']['startTime'] = '2026-09-16T00:00:00.001Z'
        elif row['type'] == 'session.shutdown':
            row['data']['sessionStartTime'] += 1
    store = Storage(tmp_path / 'epochs.duckdb')
    store.import_sources((SourcePayload(cli_locator(tmp_path / 'one'), event_bytes(one)),
                          SourcePayload(cli_locator(tmp_path / 'two'), event_bytes(two))))
    with store.connect() as db:
        assert {row[0] for row in db.execute(
            "SELECT d.state FROM decision d JOIN observation o ON o.id=d.observation_id "
            "WHERE o.model='model-alpha' AND d.measure='output'")} == {'unresolved'}
        assert db.execute("SELECT count(DISTINCT counter_epoch) FROM copilot_cli_evidence").one()[0] == 2
    store.close()


def test_cli_absent_optional_shutdown_quantity_keeps_checkpoint_lower_bound(tmp_path):
    rows = event_rows()[:5]
    rows[-1]['data'].pop('totalPremiumRequests')
    store = Storage(tmp_path / 'optional-quantity.duckdb')
    store.import_source(cli_locator(tmp_path), event_bytes(rows))
    with store.connect() as db:
        selected = tuple(db.execute(
            "SELECT q.amount_decimal,q.lower_bound,d.state FROM quantity_value q "
            "JOIN quantity_decision d ON d.observation_id=q.observation_id AND d.measure=q.measure "
            "WHERE q.measure='premium_requests' AND d.state='selected'"))
    assert selected == (('2', 1, 'selected'),)
    store.close()


def test_cli_missing_optional_shutdown_quantities_are_visible_unknown_not_zero(tmp_path):
    all_rows = event_rows()
    rows = all_rows[:3] + [all_rows[-1]]
    rows[-1]['parentId'] = rows[-2]['id']
    rows[-1]['data'].pop('totalNanoAiu')
    rows[-1]['data'].pop('totalPremiumRequests')
    store = Storage(tmp_path / 'missing-quantities.duckdb')
    store.import_source(cli_locator(tmp_path), event_bytes(rows))
    with store.connect() as db:
        missing = tuple(db.execute(
            "SELECT q.measure,q.state,q.amount_decimal,d.state,d.reason "
            "FROM quantity_value q JOIN quantity_decision d "
            "ON d.observation_id=q.observation_id AND d.measure=q.measure "
            "WHERE q.measure IN ('nano_aiu','premium_requests') "
            "ORDER BY q.measure"))
    assert missing == (('nano_aiu', 'unknown', None, 'unresolved', 'copilot_cli_incomparable_history'),
                      ('premium_requests', 'unknown', None, 'unresolved', 'copilot_cli_incomparable_history'))
    store.close()


def test_reader_preserves_input_cache_presence_and_reasoning_subset(tmp_path):
    result = read_fixture(tmp_path)
    assert isinstance(result, CopilotCLIReadBatch)
    final = [(record, item) for record, item in zip(result.usage, result.evidence, strict=True)
             if item.event_id.endswith('008') and item.scope == 'model']
    values = {record.model.model: record.tokens for record, _ in final}
    alpha = values['model-alpha']
    beta = values['model-beta']
    assert alpha.buckets.input == Known(150) and alpha.buckets.output == Known(25)
    assert alpha.reasoning == Known(5) and alpha.total == Known(175)
    assert beta.buckets.input == Unknown('cache_inclusion_unknown')
    assert beta.buckets.output == Known(40) and beta.buckets.cache_read == Known(60)
    assert beta.buckets.cache_write == Known(10) and beta.reasoning == Known(0)


def test_reader_keeps_session_model_and_agent_representations_non_additive(tmp_path):
    result = read_fixture(tmp_path)
    assert isinstance(result, CopilotCLIReadBatch)
    final = [item for item in result.evidence if item.event_id.endswith('008')]
    assert [item.scope for item in final].count('session') == 1
    assert [item.scope for item in final].count('model') == 2
    assert [item.scope for item in final].count('agent') == 2


def test_cli_retains_bounded_native_breakdown_facts_without_adding_them(tmp_path):
    rows = event_rows()
    rows[-1]['data']['modelMetrics']['model-alpha']['totalNanoAiu'] = 123
    rows[-1]['data']['modelMetrics']['model-alpha']['tokenDetails'] = {
        'input': {'tokenCount': 7}, 'output': {'tokenCount': 2}}
    rows[-1]['data']['tokenDetails'] = {'input': {'tokenCount': 12}}
    rows[-1]['data']['agentMetrics']['main']['agentName'] = 'AGENT_LABEL_CANARY'
    rows[-1]['data']['agentMetrics']['main']['modelMetrics']['model-alpha']['rogue'] = {'count': 99}
    result = read_fixture(tmp_path, event_bytes(rows))
    assert isinstance(result, CopilotCLIReadBatch)
    facts = [item.counters_json for item in result.evidence
             if item.event_id == rows[-1]['id'] and item.scope == 'model' and item.agent_id is None]
    assert any('"total_nano_aiu":123' in value and '"token_details"' in value for value in facts)
    session_facts = [item.counters_json for item in result.evidence
                     if item.event_id == rows[-1]['id'] and item.scope == 'session']
    assert any('"token_details":{"input":{"tokenCount":12}}' in value for value in session_facts)
    assert all('AGENT_LABEL_CANARY' not in value for value in session_facts)
    agent_facts = [item.counters_json for item in result.evidence
                   if item.event_id == rows[-1]['id'] and item.scope == 'agent']
    assert all('"rogue"' not in value for value in agent_facts)


def test_reader_keeps_native_units_and_cost_multiplier_separate(tmp_path):
    result = read_fixture(tmp_path)
    assert isinstance(result, CopilotCLIReadBatch)
    final = [(record, item) for record, item in zip(result.usage, result.evidence, strict=True)
             if item.event_id.endswith('008') and item.scope != 'agent']
    quantities = {(record.model.model, value.measure): value.amount
                  for record, _ in final for value in record.quantities}
    assert quantities == {
        (None, 'nano_aiu'): Decimal(200), (None, 'premium_requests'): Decimal(4),
        ('model-alpha', 'request_count'): Decimal(3),
        ('model-beta', 'request_count'): Decimal(2),
    }
    assert all(record.money.reason == 'not_recorded' for record, _ in final)
    assert any('"cost":"2.5"' in item.counters_json for _, item in final)


def test_reconciliation_covers_checkpoint_fields_independently(tmp_path):
    store = Storage(tmp_path / 'ledger.duckdb')
    store.import_source(cli_locator(tmp_path), EVENTS)
    with store.connect() as db:
        selected = {(row['measure'], row['amount_decimal']) for row in db.execute(
            "SELECT q.measure,q.amount_decimal FROM quantity_value q JOIN quantity_decision d "
            "ON d.observation_id=q.observation_id AND d.measure=q.measure WHERE d.state='selected'")}
    assert selected == {('nano_aiu', '200'), ('premium_requests', '4'),
                        ('request_count', '3'), ('request_count', '2')}
    store.close()


def test_reconciliation_marks_error_resume_and_nonterminal_shutdowns_partial(tmp_path):
    rows = event_rows()
    rows[-1]['data']['shutdownType'] = 'error'
    result = read_fixture(tmp_path, event_bytes(rows))
    assert isinstance(result, CopilotCLIReadBatch)
    assert 'cli_partial_lifetime' in {item.code for item in result.diagnostics}


def test_reconciliation_uses_physical_order_and_quarantines_only_regressed_measures(tmp_path):
    rows = event_rows()
    rows[-1]['data']['modelMetrics']['model-alpha']['usage']['outputTokens'] = 19
    store = Storage(tmp_path / 'regression.duckdb')
    store.import_source(cli_locator(tmp_path), event_bytes(rows))
    with store.connect() as db:
        output = {row['state'] for row in db.execute(
            "SELECT d.state FROM decision d JOIN observation o ON o.id=d.observation_id "
            "WHERE o.model='model-alpha' AND d.measure='output'")}
        cache = {row['state'] for row in db.execute(
            "SELECT d.state FROM decision d JOIN observation o ON o.id=d.observation_id "
            "WHERE o.model='model-alpha' AND d.measure='cache_read'")}
    assert output == {'unresolved'}
    assert 'selected' in cache
    store.close()


def test_cli_removed_latest_model_does_not_leave_stale_selected_usage(tmp_path):
    rows = event_rows()
    rows[-1]['data']['modelMetrics'].pop('model-beta')
    store = Storage(tmp_path / 'removed-model.duckdb')
    store.import_source(cli_locator(tmp_path), event_bytes(rows))
    with store.connect() as db:
        states = {row[0] for row in db.execute(
            "SELECT d.state FROM decision d JOIN observation o ON o.id=d.observation_id "
            "WHERE o.model='model-beta' AND d.measure='output'")}
    assert states == {'unresolved'}
    store.close()


def test_cli_divergent_sibling_shutdown_copies_are_unresolved(tmp_path):
    first = event_rows()
    second = event_rows()
    first[-1]['id'] = '00000000-0000-4000-8000-000000000009'
    second[-1]['id'] = '00000000-0000-4000-8000-000000000010'
    store = Storage(tmp_path / 'sibling-shutdowns.duckdb')
    store.import_sources((SourcePayload(cli_locator(tmp_path / 'one'), event_bytes(first)),
                          SourcePayload(cli_locator(tmp_path / 'two'), event_bytes(second))))
    with store.connect() as db:
        states = {row[0] for row in db.execute(
            "SELECT d.state FROM decision d JOIN observation o ON o.id=d.observation_id "
            "WHERE o.model='model-alpha' AND d.measure='output'")}
    assert states == {'unresolved'}
    store.close()


def test_reconciliation_deduplicates_exact_same_session_events_and_quarantines_cross_session_ids(tmp_path):
    store = Storage(tmp_path / 'copies.duckdb')
    one = cli_locator(tmp_path / 'one')
    two = cli_locator(tmp_path / 'two')
    store.import_sources((SourcePayload(one, EVENTS), SourcePayload(two, EVENTS)))
    with store.connect() as db:
        assert db.execute("SELECT count(*) FROM decision WHERE state='selected' AND measure='output'").one()[0] == 2
        assert db.execute("SELECT count(*) FROM quantity_decision WHERE state='selected' AND measure='nano_aiu'").one()[0] == 1
    rows = event_rows()
    other = '22222222-2222-4222-8222-222222222222'
    rows[0]['data']['sessionId'] = other
    store.import_source(cli_locator(tmp_path / 'other', other), event_bytes(rows))
    with store.connect() as db:
        assert {row[0] for row in db.execute(
            "SELECT state FROM decision WHERE measure='output'")} == {'unresolved'}
    store.close()


def test_cli_same_event_top_level_presence_mutation_quarantines_model_coordinates(tmp_path):
    first = event_rows()
    second = event_rows()
    second[-1]['data'].pop('totalPremiumRequests')
    store = Storage(tmp_path / 'signature-conflict.duckdb')
    store.import_sources((SourcePayload(cli_locator(tmp_path / 'one'), event_bytes(first)),
                          SourcePayload(cli_locator(tmp_path / 'two'), event_bytes(second))))
    with store.connect() as db:
        states = {row[0] for row in db.execute(
            "SELECT d.state FROM decision d JOIN observation o ON o.id=d.observation_id "
            "WHERE o.model='model-alpha' AND d.measure='output'")}
    assert states == {'unresolved'}
    store.close()


def test_cli_same_event_kind_mutation_quarantines_original_model_coordinates(tmp_path):
    first = event_rows()
    second = event_rows()
    second[-1]['type'] = 'session.usage_checkpoint'
    store = Storage(tmp_path / 'kind-conflict.duckdb')
    store.import_sources((SourcePayload(cli_locator(tmp_path / 'one'), event_bytes(first)),
                          SourcePayload(cli_locator(tmp_path / 'two'), event_bytes(second))))
    with store.connect() as db:
        states = {row[0] for row in db.execute(
            "SELECT d.state FROM decision d JOIN observation o ON o.id=d.observation_id "
            "WHERE o.model='model-alpha' AND d.measure='output' ")}
    assert states == {'unresolved'}
    store.close()


def test_reader_keys_accounting_by_model_metrics_and_leaves_provider_unavailable(tmp_path):
    result = read_fixture(tmp_path)
    assert isinstance(result, CopilotCLIReadBatch)
    models = {record.model for record, item in zip(result.usage, result.evidence, strict=True)
              if item.scope == 'model'}
    assert {model.model for model in models} == {'model-alpha', 'model-beta'}
    assert {model.provider for model in models} == {None}
    assert 'selected-only' not in {model.model for model in models}


def test_agent_children_require_supported_conversation_events_not_metric_labels(tmp_path):
    store = Storage(tmp_path / 'children.duckdb')
    rows = event_rows()
    rows[2].pop('agentId')
    store.import_source(cli_locator(tmp_path), event_bytes(rows))
    with store.connect() as db:
        assert db.execute(
            "SELECT count(*) FROM session WHERE harness='copilot-cli' AND parent_locator IS NOT NULL").one()[0] == 0
    store.import_source(cli_locator(tmp_path), EVENTS)
    with store.connect() as db:
        children = tuple(db.execute(
            "SELECT native_id,parent_locator FROM session WHERE harness='copilot-cli' "
            "AND parent_locator IS NOT NULL"))
    assert children == ((SESSION + ':agent:agent-child', cli_locator(tmp_path)),)
    store.close()


def test_cli_append_restart_and_removal_keep_one_cumulative_control_and_saved_usage(tmp_path):
    locator = cli_locator(tmp_path)
    store = Storage(tmp_path / 'restart.duckdb')
    store.import_source(locator, event_bytes(event_rows()[:5]))
    store.close()
    store = Storage(tmp_path / 'restart.duckdb')
    store.import_source(locator, EVENTS)
    store.mark_missing((locator,))
    with store.connect() as db:
        assert db.execute(
            "SELECT count(*) FROM quantity_decision WHERE measure='nano_aiu' AND state='selected'").one()[0] == 1
        assert db.execute(
            "SELECT amount_decimal FROM quantity_value q JOIN quantity_decision d "
            "ON d.observation_id=q.observation_id AND d.measure=q.measure "
            "WHERE q.measure='nano_aiu' AND d.state='selected'").one()[0] == '200'
    store.import_source(locator, EVENTS)
    with store.connect() as db:
        assert db.execute(
            "SELECT count(*) FROM quantity_decision WHERE measure='nano_aiu' AND state='selected'").one()[0] == 1
    store.close()


def test_cli_pending_or_malformed_retained_tail_keeps_lifetime_partial(tmp_path):
    for suffix in (b'{bad', b'{"id":'):
        result = read_fixture(tmp_path, EVENTS + suffix)
        assert isinstance(result, CopilotCLIReadBatch)
        assert 'cli_partial_lifetime' in {item.code for item in result.diagnostics}


def test_cli_pending_context_is_ignored_and_settled_git_root_wins(tmp_path):
    rows = event_rows()
    rows[6]['data']['pendingGitContext'] = True
    result = read_fixture(tmp_path, event_bytes(rows))
    assert isinstance(result, CopilotCLIReadBatch)
    assert result.session.cwd == '/fixture/work-a'


def test_cli_workspace_only_without_optional_id_is_retained_unavailable(tmp_path):
    locator = cli_locator(tmp_path, SESSION, 'workspace.yaml')
    data = b'cwd: /fixture/metadata-only\n'
    result = read_copilot_cli(data, locator=locator)
    assert isinstance(result, CopilotCLIReadBatch)
    assert result.session.native_id == SESSION
    assert all(value.amount is None for value in result.usage[0].quantities)


def test_cli_workspace_conflict_stays_unassigned_after_a_b_a_reappearance(tmp_path):
    base = event_rows()
    a = [row for row in base if row['type'] != 'session.context_changed']
    b = json.loads(json.dumps(a))
    b[0]['data']['context']['cwd'] = '/fixture/work-b'
    b[0]['data']['context']['gitRoot'] = '/fixture/work-b'
    a_again = json.loads(json.dumps(a))
    a_again[0]['data']['branch'] = 'main'
    store = Storage(tmp_path / 'workspace-copies.duckdb')
    store.import_source(cli_locator(tmp_path / 'a'), event_bytes(a))
    store.import_source(cli_locator(tmp_path / 'b'), event_bytes(b))
    store.import_source(cli_locator(tmp_path / 'a'), event_bytes(a_again))
    with store.connect() as db:
        assert db.execute("SELECT cwd FROM session WHERE id=?", ('copilot-cli:' + SESSION,)).one()[0] is None
    store.close()


def test_cli_parent_locator_is_unique_ambiguous_and_cross_harness_isolated(tmp_path):
    locator = cli_locator(tmp_path)
    store = Storage(tmp_path / 'families.duckdb')
    store.import_source(locator, EVENTS)
    assert 'ambiguous_subagent_owner' not in store.report_input(ReportQuery(None, AllTime())).diagnostics
    with store.connect(write=True) as db:
        db.execute("INSERT INTO session(id,harness,native_id) VALUES('copilot-cli:other','copilot-cli','other')")
        db.execute("INSERT INTO session_attribution VALUES('copilot-cli:other',NULL,NULL,'not_resolved')")
        db.execute("INSERT INTO source_generation VALUES('ambiguous',?,99,?,'copilot-cli:other',?,0,0,'available')",
                   (locator, '0' * 64, CLI_PROFILE))
    assert 'ambiguous_subagent_owner' in store.report_input(ReportQuery(None, AllTime())).diagnostics
    with store.connect(write=True) as db:
        db.execute("UPDATE session SET parent_locator='/foreign/pi.jsonl' WHERE parent_locator IS NOT NULL")
        db.execute("INSERT INTO session(id,harness,native_id) VALUES('pi:foreign','pi','foreign')")
        db.execute("INSERT INTO session_attribution VALUES('pi:foreign',NULL,NULL,'not_resolved')")
        db.execute("INSERT INTO source_generation VALUES('foreign','/foreign/pi.jsonl',0,?,'pi:foreign','pi-v3',0,0,'available')",
                   ('1' * 64,))
    diagnostics = store.report_input(ReportQuery(None, AllTime())).diagnostics
    assert 'missing_subagent_owner' in diagnostics
    store.close()


def test_cli_content_canaries_never_enter_reader_evidence_logs_errors_or_duckdb(tmp_path, caplog):
    rows = event_rows()
    rows[1]['data']['content'] = 'PROMPT_CANARY https://invalid.example/secret'
    rows[2]['data']['content'] = 'ASSISTANT_CANARY <script>active()</script>'
    data = event_bytes(rows)
    result = read_fixture(tmp_path, data)
    assert isinstance(result, CopilotCLIReadBatch)
    assert 'PROMPT_CANARY' not in repr(result.evidence)
    assert 'ASSISTANT_CANARY' not in repr(result.evidence)
    store = Storage(tmp_path / 'privacy.duckdb')
    store.import_source(cli_locator(tmp_path), data)
    with store.connect() as db:
        persisted = repr(tuple(db.execute(
            'SELECT safe_facts_json FROM observation UNION ALL SELECT counters_json FROM copilot_cli_evidence')))
    assert 'PROMPT_CANARY' not in persisted and 'ASSISTANT_CANARY' not in persisted
    assert 'PROMPT_CANARY' not in caplog.text and 'ASSISTANT_CANARY' not in caplog.text
    store.close()


def test_attribution_uses_one_settled_context_or_workspace_changed_unassigned(tmp_path):
    result = read_fixture(tmp_path)
    assert isinstance(result, CopilotCLIReadBatch)
    assert result.session.cwd is None
    assert 'workspace_changed_cumulative_scope' in {item.code for item in result.diagnostics}
    app = Application(tmp_path / 'application')
    app.storage.import_source(cli_locator(tmp_path), EVENTS)
    app._assign_projects()
    attribution = app.storage.attributions()['copilot-cli:' + SESSION]
    assert isinstance(attribution, Unassigned)
    assert attribution.reason == 'workspace_changed_cumulative_scope'
    app.close()


def test_workspace_yaml_is_optional_unversioned_metadata_and_never_zero_usage(tmp_path):
    metadata = read_workspace_metadata(WORKSPACE)
    assert (metadata.session_id, metadata.cwd) == (SESSION, '/fixture/yaml-fallback')
    result = read_copilot_cli((FIXTURES / 'legacy/workspace.yaml').read_bytes(),
                              locator=cli_locator(tmp_path, 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa', 'workspace.yaml'))
    assert isinstance(result, CopilotCLIReadBatch)
    assert result.diagnostics[0].code == 'events_unavailable'
    assert all(isinstance(value, Unknown) for value in result.usage[0].tokens.buckets.values)
    assert all(value.amount is None for value in result.usage[0].quantities)


def test_cli_transcript_projects_only_durable_physical_order_with_safe_tool_pairing(tmp_path):
    main = parse_copilot_cli_transcript(EVENTS, SESSION, 'copilot-cli:' + SESSION)
    child = parse_copilot_cli_transcript(
        EVENTS, SESSION, 'copilot-cli:' + SESSION + ':agent:agent-child', agent_id='agent-child')
    assert [entry.role for entry in main.entries if hasattr(entry, 'role')] == ['user']
    assert [entry.role for entry in child.entries if hasattr(entry, 'role')] == ['assistant']
    assert 'SYNTHETIC_USER' in repr(main.entries) and 'SYNTHETIC_CHILD' not in repr(main.entries)
    assert 'SYNTHETIC_CHILD' in repr(child.entries)
    rows = event_rows()
    rows.extend([
        {'id': '00000000-0000-4000-8000-000000000009', 'parentId': rows[-1]['id'],
         'timestamp': '2026-09-16T00:00:00.080Z', 'type': 'assistant.reasoning',
         'data': {'reasoningId': 'reasoning-1', 'content': 'REASONING_CANARY'}},
        {'id': '00000000-0000-4000-8000-000000000010',
         'parentId': '00000000-0000-4000-8000-000000000009',
         'timestamp': '2026-09-16T00:00:00.090Z', 'type': 'assistant.message',
         'data': {'messageId': 'message-1', 'content': 'TOOL_MESSAGE', 'toolRequests': [
             {'toolCallId': 'tool-1', 'name': 'fixture_tool', 'arguments': {'value': 'ARG_CANARY'}}]}},
        {'id': '00000000-0000-4000-8000-000000000011',
         'parentId': '00000000-0000-4000-8000-000000000010',
         'timestamp': '2026-09-16T00:00:00.100Z', 'type': 'tool.execution_start',
         'data': {'toolCallId': 'tool-1', 'toolName': 'fixture_tool', 'arguments': 'DUPLICATE_START'}},
        {'id': '00000000-0000-4000-8000-000000000012',
         'parentId': '00000000-0000-4000-8000-000000000011',
         'timestamp': '2026-09-16T00:00:00.110Z', 'type': 'tool.execution_complete',
         'data': {'toolCallId': 'tool-1', 'success': True,
                  'result': {'content': 'RESULT_CANARY'}}},
    ])
    projected = parse_copilot_cli_transcript(event_bytes(rows), SESSION, 'copilot-cli:' + SESSION)
    rendered = repr(projected.entries)
    assert rendered.count("call_id='tool-1'") == 1
    assert 'ARG_CANARY' in rendered and 'RESULT_CANARY' in rendered and 'DUPLICATE_START' not in rendered
    incomplete = parse_copilot_cli_transcript(EVENTS + b'{"id":', SESSION, 'copilot-cli:' + SESSION)
    malformed = parse_copilot_cli_transcript(EVENTS + b'{bad', SESSION, 'copilot-cli:' + SESSION)
    assert any('incomplete' in warning for warning in incomplete.warnings)
    assert all('incomplete' not in warning for warning in malformed.warnings)
    with pytest.raises(TranscriptUnavailable):
        parse_copilot_cli_transcript(WORKSPACE, SESSION, 'copilot-cli:' + SESSION)


@pytest.mark.parametrize('mutation', (
    'message_id', 'reasoning_id', 'scalar_result', 'missing_success', 'alias_start', 'alias_request'))
def test_cli_transcript_requires_pinned_body_fields_before_projecting_content(tmp_path, mutation):
    rows = event_rows()
    rows.extend([
        {'id': '00000000-0000-4000-8000-000000000009', 'parentId': rows[-1]['id'],
         'timestamp': '2026-09-16T00:00:00.080Z', 'type': 'assistant.message',
         'data': {'messageId': 'message-1', 'content': 'VALID_ASSISTANT', 'toolRequests': []}},
        {'id': '00000000-0000-4000-8000-000000000010',
         'parentId': '00000000-0000-4000-8000-000000000009',
         'timestamp': '2026-09-16T00:00:00.090Z', 'type': 'assistant.reasoning',
         'data': {'reasoningId': 'reasoning-1', 'content': 'VALID_REASONING'}},
        {'id': '00000000-0000-4000-8000-000000000011',
         'parentId': '00000000-0000-4000-8000-000000000010',
         'timestamp': '2026-09-16T00:00:00.100Z', 'type': 'tool.execution_start',
         'data': {'toolCallId': 'tool-1', 'toolName': 'fixture_tool', 'arguments': {'safe': True}}},
        {'id': '00000000-0000-4000-8000-000000000012',
         'parentId': '00000000-0000-4000-8000-000000000011',
         'timestamp': '2026-09-16T00:00:00.110Z', 'type': 'tool.execution_complete',
         'data': {'toolCallId': 'tool-1', 'success': True,
                  'result': {'content': 'VALID_RESULT'}}},
    ])
    if mutation == 'message_id':
        rows[8]['data'].pop('messageId')
        rows[8]['data']['content'] = 'CANARY_MISSING_MESSAGE_ID'
    elif mutation == 'reasoning_id':
        rows[9]['data'].pop('reasoningId')
        rows[9]['data']['content'] = 'CANARY_MISSING_REASONING_ID'
    elif mutation == 'scalar_result':
        rows[11]['data'].pop('result')
        rows[11]['data']['result'] = 'CANARY_SCALAR_RESULT'
    elif mutation == 'missing_success':
        rows[11]['data'].pop('success')
        rows[11]['data']['result'] = {'content': 'CANARY_MISSING_SUCCESS'}
    elif mutation == 'alias_start':
        rows[10]['data'].pop('toolName')
        rows[10]['data']['name'] = 'fixture_tool'
        rows[10]['data']['arguments'] = 'CANARY_ALIAS_START'
    else:
        rows[8]['data']['toolRequests'] = [
            {'id': 'alias-tool', 'name': 'fixture_tool', 'arguments': 'CANARY_ALIAS_REQUEST'}]
    transcript = parse_copilot_cli_transcript(event_bytes(rows), SESSION, 'copilot-cli:' + SESSION)
    rendered = repr(transcript.entries)
    assert 'CANARY_' not in rendered


def test_cli_public_transcript_rejects_invalid_agent_and_missing_event_identity(tmp_path):
    rows = event_rows()
    rows.append({'id': '00000000-0000-4000-8000-000000000009',
                 'parentId': rows[-1]['id'], 'timestamp': '2026-09-16T00:00:00.080Z',
                 'type': 'user.message', 'agentId': 123,
                 'data': {'content': 'INVALID_AGENT_CANARY'}})
    rows.append({'parentId': rows[-1]['id'], 'timestamp': '2026-09-16T00:00:00.090Z',
                 'type': 'user.message', 'data': {'content': 'MISSING_ID_CANARY'}})
    transcript = parse_copilot_cli_transcript(event_bytes(rows), SESSION, 'copilot-cli:' + SESSION)
    rendered = repr(transcript.entries)
    assert 'INVALID_AGENT_CANARY' not in rendered
    assert 'MISSING_ID_CANARY' not in rendered
    assert 'copilot-cli-line-' not in rendered


def test_cli_public_transcript_rejects_start_time_after_retained_events_as_changed(tmp_path):
    rows = event_rows()
    rows[0]['data']['startTime'] = '2026-09-16T00:00:00.080Z'
    with pytest.raises(TranscriptUnavailable) as error:
        parse_copilot_cli_transcript(event_bytes(rows), SESSION, 'copilot-cli:' + SESSION)
    assert error.value.kind == 'changed'


def test_cli_public_transcript_keeps_tool_user_requested_as_bounded_notice(tmp_path):
    rows = event_rows()
    rows.append({
        'id': '00000000-0000-4000-8000-000000000013', 'parentId': rows[-1]['id'],
        'timestamp': '2026-09-16T00:00:00.120Z', 'type': 'tool.user_requested',
        'data': {'toolCallId': 'tool-user-requested', 'toolName': 'fixture_tool',
                 'arguments': 'CANARY_USER_REQUESTED'},
    })
    transcript = parse_copilot_cli_transcript(event_bytes(rows), SESSION, 'copilot-cli:' + SESSION)
    rendered = repr(transcript.entries)
    assert 'CANARY_USER_REQUESTED' not in rendered
    assert 'Unsupported Copilot CLI entry' in rendered


def test_cli_full_and_scoped_reconciliation_match_and_leave_unrelated_rows_unchanged(tmp_path):
    store = Storage(tmp_path / 'scoped.duckdb')
    pi = Path(__file__).parent / 'fixtures/pi/source/ordinary.jsonl'
    store.import_source('/fixture/pi.jsonl', pi.read_bytes())
    with store.connect() as db:
        pi_before = tuple(db.execute(
            "SELECT d.* FROM decision d JOIN observation o ON o.id=d.observation_id "
            "WHERE o.session_id LIKE 'pi:%' ORDER BY d.observation_id,d.measure"))
    store.import_source(cli_locator(tmp_path), EVENTS)
    with store.connect() as db:
        cli_scoped = tuple(db.execute(
            "SELECT d.* FROM decision d JOIN observation o ON o.id=d.observation_id "
            "WHERE o.session_id LIKE 'copilot-cli:%' ORDER BY d.observation_id,d.measure"))
    with store.connect(write=True) as db:
        store._reconcile(db)
    with store.connect() as db:
        cli_full = tuple(db.execute(
            "SELECT d.* FROM decision d JOIN observation o ON o.id=d.observation_id "
            "WHERE o.session_id LIKE 'copilot-cli:%' ORDER BY d.observation_id,d.measure"))
        pi_after = tuple(db.execute(
            "SELECT d.* FROM decision d JOIN observation o ON o.id=d.observation_id "
            "WHERE o.session_id LIKE 'pi:%' ORDER BY d.observation_id,d.measure"))
    assert cli_scoped == cli_full
    assert pi_before == pi_after
    store.close()


def test_cli_import_reopen_missing_and_reappearance_leave_all_other_harness_rows_unchanged(tmp_path):
    fixture_root = Path(__file__).parent / 'fixtures'
    cli_path = cli_locator(tmp_path)
    sources = (
        SourcePayload('/fixture/pi.jsonl', (fixture_root / 'pi/source/ordinary.jsonl').read_bytes()),
        SourcePayload('/fixture/codex.jsonl', (fixture_root / 'codex/mixed.jsonl').read_bytes()),
        SourcePayload('/fixture/claude.jsonl', (fixture_root / 'claude/main.jsonl').read_bytes()),
        SourcePayload('/fixture/workspaceStorage/key/chatSessions/session-v3.jsonl',
                      (fixture_root / 'copilot_vscode/session-v3.jsonl').read_bytes()),
    )
    store = Storage(tmp_path / 'all-harnesses.duckdb')
    store.import_sources(sources)

    def unrelated_rows(owner: Storage):
        with owner.connect() as db:
            decisions = tuple(db.execute(
                "SELECT d.rowid,d.* FROM decision d JOIN observation o ON o.id=d.observation_id "
                "WHERE o.session_id NOT LIKE 'copilot-cli:%' ORDER BY d.observation_id,d.measure"))
            quantities = tuple(db.execute(
                "SELECT q.rowid,q.* FROM quantity_decision q JOIN observation o ON o.id=q.observation_id "
                "WHERE o.session_id NOT LIKE 'copilot-cli:%' ORDER BY q.observation_id,q.measure"))
            return decisions, quantities

    before = unrelated_rows(store)
    store.import_source(cli_path, EVENTS)
    store.import_source(cli_path, EVENTS)
    store.mark_missing((cli_path,))
    store.import_source(cli_path, EVENTS)
    store.close()

    reopened = Storage(tmp_path / 'all-harnesses.duckdb')
    assert unrelated_rows(reopened) == before
    reopened.close()
