import json
from decimal import Decimal
from pathlib import Path

import pytest

from harness_usage.copilot_vscode_reader import (
    CopilotVscodeReadBatch, copilot_vscode_scope, read_copilot_vscode,
    replay_chat_v3,
)
from harness_usage.copilot_vscode_transcript import parse_copilot_vscode_transcript
from harness_usage.domain import Known, Unknown
from harness_usage.pi_reader import RejectedSource
from harness_usage.source_input import SourcePayload
from harness_usage.storage import Storage
from harness_usage.transcript import AttachmentBlock, Message, Notice, ReasoningBlock, TextBlock, ToolBlock
from harness_usage.transcript_access import read_transcript_page
from harness_usage.transcript import TranscriptUnavailable


FIXTURES = Path(__file__).parent / 'fixtures/copilot_vscode'
FLAT = (FIXTURES / 'session-v3.json').read_bytes()
LOG = (FIXTURES / 'session-v3.jsonl').read_bytes()


def test_scope_collapses_workspace_copies_but_separates_installations(tmp_path):
    stable_a = tmp_path / 'stable/User/workspaceStorage/one/chatSessions/s.json'
    stable_b = tmp_path / 'stable/User/workspaceStorage/two/chatSessions/s.jsonl'
    insiders = tmp_path / 'insiders/User/workspaceStorage/one/chatSessions/s.json'
    assert copilot_vscode_scope(str(stable_a)) == copilot_vscode_scope(str(stable_b))
    assert copilot_vscode_scope(str(stable_a)) != copilot_vscode_scope(str(insiders))
    assert copilot_vscode_scope(str(tmp_path / 'chatSessions/s.json')) is None


def test_replay_flat_and_operation_log_have_equal_final_state_and_provenance():
    flat = replay_chat_v3(FLAT, representation='flat')
    log = replay_chat_v3(LOG, representation='operation_log')
    assert log.value == flat.value
    assert flat.line_for(('requests', 1, 'sessionCopilotCredits')) == 1
    assert log.line_for(('requests', 0)) == 2
    assert log.line_for(('requests', 1)) == 3
    assert log.line_for(('requests', 1, 'sessionCopilotCredits')) == 4
    assert not flat.pending_tail and not log.pending_tail
    assert flat.complete_bytes == len(FLAT) and log.complete_bytes == len(LOG)


@pytest.mark.parametrize(('lines', 'code'), (
    ([{'kind': 1, 'k': ['x'], 'v': 1}], 'missing_initial_chat_state'),
    ([{'kind': 0, 'v': {}}, {'kind': 0, 'v': {}}], 'duplicate_initial_chat_state'),
    ([{'kind': 0, 'v': {'items': [{}]}}, {'kind': 3, 'k': ['items', 0]}], 'invalid_delete_path'),
    ([{'kind': 0, 'v': {'items': []}}, {'kind': 2, 'k': ['items'], 'i': True}], 'invalid_push_index'),
    ([{'kind': 0, 'v': {'items': []}}, {'kind': 2, 'k': ['items'], 'i': 1}], 'invalid_push_index'),
    ([{'kind': 0, 'v': {'items': []}}, {'kind': 1, 'k': ['missing', 'x'], 'v': 1}], 'invalid_operation_path'),
    ([{'kind': 0, 'v': {'items': [1]}}, {'kind': 1, 'k': ['items', 0], 'v': 2}], 'invalid_operation_path'),
), ids=('initial-first', 'one-initial', 'no-list-delete', 'bool-index', 'push-bound', 'no-autocreate',
        'no-numeric-terminal-set'))
def test_replay_rejects_unsafe_operation_boundaries(lines, code):
    data = b''.join(json.dumps(line).encode() + b'\n' for line in lines)
    with pytest.raises(ValueError, match=code):
        replay_chat_v3(data, representation='operation_log')


def test_replay_set_requires_an_explicit_value():
    data = b'{"kind":0,"v":{"value":1}}\n{"kind":1,"k":["value"]}\n'
    with pytest.raises(ValueError, match='missing_set_value'):
        replay_chat_v3(data, representation='operation_log')


def test_replay_applies_nested_set_delete_push_and_truncate_without_aliasing():
    lines = (
        {'kind': 0, 'v': {'items': [{'value': 1, 'drop': True}], 'copy': []}},
        {'kind': 1, 'k': ['items', 0, 'value'], 'v': 2},
        {'kind': 3, 'k': ['items', 0, 'drop']},
        {'kind': 2, 'k': ['copy'], 'v': [{'nested': []}]},
        {'kind': 2, 'k': ['copy', 0, 'nested'], 'v': [1, 2]},
        {'kind': 2, 'k': ['copy', 0, 'nested'], 'i': 1},
    )
    snapshot = replay_chat_v3(b''.join(json.dumps(line).encode() + b'\n' for line in lines), representation='operation_log')
    assert snapshot.value == {'items': [{'value': 2}], 'copy': [{'nested': [1]}]}
    assert ('items', 0, 'drop') not in snapshot.provenance
    assert ('copy', 0, 'nested', 1) not in snapshot.provenance


@pytest.mark.parametrize(('updates', 'expected_line'), (
    (({'kind': 1, 'k': ['requests', 0, 'modelTotals', 0, 'outputTokens'], 'v': 7},
      {'kind': 1, 'k': ['requests', 0, 'modelTotals', 0, 'cachedTokens'], 'v': 3}), 3),
    (({'kind': 1, 'k': ['requests', 0, 'modelTotals'],
       'v': [{'model': 'actual', 'inputTokens': 10, 'cachedTokens': 3, 'outputTokens': 7}]},), 2),
    (({'kind': 3, 'k': ['requests', 0, 'modelTotals', 0, 'outputTokens']},), 1),
    (({'kind': 2, 'k': ['requests', 0, 'modelTotals'], 'i': 0},
      {'kind': 2, 'k': ['requests', 0, 'modelTotals'],
       'v': [{'model': 'actual', 'inputTokens': 10, 'cachedTokens': 3, 'outputTokens': 7}]}), 3),
))
def test_model_total_provenance_uses_last_present_field_operation(tmp_path, updates, expected_line):
    initial = {
        'version': 3, 'sessionId': 'provenance', 'requests': [{
            'requestId': 'request', 'agent': {'extensionId': {'value': 'github.copilot-chat'}},
            'modelTotals': [{'model': 'actual', 'inputTokens': 10, 'cachedTokens': 2, 'outputTokens': 5}],
        }],
    }
    operations = ({'kind': 0, 'v': initial}, *updates)
    data = b''.join(json.dumps(operation).encode() + b'\n' for operation in operations)
    result = read_copilot_vscode(data, locator=canonical_locator(tmp_path, suffix='.jsonl'))
    assert isinstance(result, CopilotVscodeReadBatch)
    evidence = next(item for item in result.evidence if item.evidence_kind == 'model_totals')
    assert evidence.line == expected_line


@pytest.mark.parametrize(('identity', 'updates', 'expected_line'), (
    ({'requestId': 'request-one', 'responseId': 'response-one'},
     ({'kind': 1, 'k': ['requests', 0, 'requestId'], 'v': 'request-two'},), 2),
    ({'requestId': 'request-one', 'responseId': 'response-one'},
     ({'kind': 1, 'k': ['requests', 0, 'responseId'], 'v': 'response-two'},), 2),
    ({}, ({'kind': 1, 'k': ['requests', 0, 'requestId'], 'v': 'request'},), 2),
    ({'requestId': 'request'},
     ({'kind': 1, 'k': ['requests', 0, 'responseId'], 'v': 'response'},), 2),
    ({'requestId': 'request-one', 'responseId': 'response-one'}, (
        {'kind': 1, 'k': ['requests', 0, 'requestId'], 'v': 'request-two'},
        {'kind': 1, 'k': ['requests', 0, 'responseId'], 'v': 'response-two'},
    ), 3),
    ({'requestId': 'request', 'responseId': 'response'},
     ({'kind': 3, 'k': ['requests', 0, 'responseId']},), 1),
    ({'requestId': 'request', 'responseId': 'response'},
     ({'kind': 3, 'k': ['requests', 0, 'requestId']},), 1),
))
def test_model_total_provenance_includes_only_present_identity_field_operations(
        tmp_path, identity, updates, expected_line):
    request = {
        **identity,
        'agent': {'extensionId': {'value': 'github.copilot-chat'}},
        'modelTotals': [{'model': 'actual', 'inputTokens': 10,
                         'cachedTokens': 2, 'outputTokens': 5}],
    }
    operations = ({'kind': 0, 'v': {
        'version': 3, 'sessionId': 'identity-provenance', 'requests': [request],
    }}, *updates)
    data = b''.join(json.dumps(operation).encode() + b'\n' for operation in operations)
    result = read_copilot_vscode(data, locator=canonical_locator(tmp_path, suffix='.jsonl'))
    assert isinstance(result, CopilotVscodeReadBatch)
    evidence = next(item for item in result.evidence if item.evidence_kind == 'model_totals')
    assert evidence.line == expected_line


def test_replay_accepts_complete_unterminated_tail_but_marks_only_truncation_pending():
    complete = LOG.rstrip(b'\n')
    assert replay_chat_v3(complete, representation='operation_log').complete_bytes == len(complete)
    pending = LOG + b'{"kind":1,"k":["requests",0'
    snapshot = replay_chat_v3(pending, representation='operation_log')
    assert snapshot.pending_tail
    assert snapshot.complete_bytes == len(LOG)
    malformed_middle = LOG + b'{bad}\n' + b'{"kind":1,"k":["requests",0,"promptTokens"],"v":9}\n'
    with pytest.raises(ValueError, match='malformed_chat_operation'):
        replay_chat_v3(malformed_middle, representation='operation_log')


def canonical_locator(tmp_path, workspace='one', suffix='.json'):
    return str(tmp_path / f'User/workspaceStorage/{workspace}/chatSessions/session{suffix}')


def operation_log(value):
    return json.dumps({'kind': 0, 'v': value}, separators=(',', ':')).encode() + b'\n'


def test_reader_normalizes_multi_model_and_lossy_summary_without_inventing_input_or_price(tmp_path):
    result = read_copilot_vscode(FLAT, locator=canonical_locator(tmp_path))
    assert isinstance(result, CopilotVscodeReadBatch)
    assert result.raw_session_id == 'vs-session'
    assert result.session.cwd == '/work/example'
    assert result.session.display_name == 'Synthetic Copilot session'
    model_rows = [(record.model.model, record.tokens.buckets.cache_read, record.tokens.buckets.output,
                   record.tokens.buckets.input, record.money.reason)
                  for record, evidence in zip(result.usage, result.evidence, strict=True)
                  if evidence.evidence_kind == 'model_totals']
    assert model_rows == [
        ('model-a', Known(2), Known(5), Unknown('cache_inclusion_unknown'), 'partial_tokens'),
        ('model-b', Known(3), Known(7), Unknown('cache_inclusion_unknown'), 'partial_tokens'),
    ]
    summary = next(record for record, evidence in zip(result.usage, result.evidence, strict=True)
                   if evidence.request_id == 'request-b' and evidence.evidence_kind == 'turn_summary')
    assert summary.model.provider == 'github-copilot' and summary.model.model is None
    assert summary.tokens.buckets.input == Known(8)
    assert summary.tokens.buckets.output == Known(3)
    assert summary.tokens.buckets.cache_read == Unknown('not_reported')
    assert summary.money.reason == 'selected_model_unpriced'
    assert 'lossy_turn_summary' in {item.code for item in result.diagnostics}


def test_reader_preserves_credit_zero_unknown_and_session_control_as_distinct_quantities(tmp_path):
    result = read_copilot_vscode(LOG, locator=canonical_locator(tmp_path, suffix='.jsonl'))
    assert isinstance(result, CopilotVscodeReadBatch)
    quantities = [(e.request_id, e.evidence_kind, q.state, q.amount, q.lower_bound)
                  for record, e in zip(result.usage, result.evidence, strict=True)
                  for q in record.quantities]
    assert ('request-a', 'turn_summary', 'known', 0, True) in quantities
    assert ('request-b', 'turn_summary', 'known', pytest.approx(0.25), True) in quantities
    assert ('request-b', 'session_control', 'known', pytest.approx(0.5), False) in quantities
    control = next((record, evidence) for record, evidence in zip(result.usage, result.evidence, strict=True)
                   if evidence.evidence_kind == 'session_control')
    assert control[0].entry.line == 4 and control[1].line == 4


def test_reader_filters_non_copilot_participants_and_emits_unavailable_for_qualified_request(tmp_path):
    value = {
        'version': 3, 'sessionId': 'vs-session', 'requests': [
            {'requestId': 'other', 'agent': {'extensionId': {'value': 'other.extension'}}, 'promptTokens': 99},
            {'requestId': 'copilot', 'agent': {'extensionId': {'value': 'GITHUB.COPILOT-CHAT'}}},
        ],
    }
    result = read_copilot_vscode(json.dumps(value).encode(), locator=canonical_locator(tmp_path))
    assert isinstance(result, CopilotVscodeReadBatch)
    assert [(e.request_id, e.evidence_kind) for e in result.evidence] == [('copilot', 'unavailable')]
    assert all(isinstance(value, Unknown) for value in result.usage[0].tokens.buckets.values)
    assert result.usage[0].quantities[0].state == 'unknown'


@pytest.mark.parametrize('version', (3.0, True, '3'))
def test_reader_and_public_transcript_require_exact_integer_schema_version(tmp_path, version):
    value = json.loads(FLAT)
    value['version'] = version
    data = json.dumps(value).encode()
    result = read_copilot_vscode(data, locator=canonical_locator(tmp_path))
    assert isinstance(result, RejectedSource)
    assert result.diagnostics[0].code == 'invalid_vscode_chat_state'

    source = Path(canonical_locator(tmp_path / 'access'))
    source.parent.mkdir(parents=True)
    source.write_bytes(FLAT)
    store = Storage(tmp_path / f'access-{version!s}.duckdb')
    store.import_source(str(source), FLAT)
    with store.connect() as db:
        session_id = db.execute("SELECT id FROM session WHERE harness='copilot-vscode'").one()[0]
    source.write_bytes(data)
    with pytest.raises(TranscriptUnavailable) as unavailable:
        read_transcript_page(store, (str(tmp_path),), session_id)
    assert unavailable.value.kind == 'changed'
    store.close()


def test_reader_does_not_accept_copilot_display_spoofs(tmp_path):
    spoof = {
        'version': 3, 'sessionId': 'spoof', 'requests': [{
            'requestId': 'spoof', 'promptTokens': 99, 'copilotCredits': 99,
            'agent': {'id': 'github.copilot-chat', 'extensionDisplayName': 'GitHub Copilot Chat'},
        }],
    }
    result = read_copilot_vscode(json.dumps(spoof).encode(), locator=canonical_locator(tmp_path))
    assert isinstance(result, CopilotVscodeReadBatch)
    assert result.usage == () and result.evidence == ()


def test_reader_keeps_credit_decimal_text_exact_and_rejects_nonfinite_numbers(tmp_path):
    exact = (b'{"version":3,"sessionId":"decimal","requests":[{"requestId":"r",'
             b'"agent":{"extensionId":{"value":"github.copilot-chat"}},'
             b'"copilotCredits":0.12345678901234567890123456789}]}')
    result = read_copilot_vscode(exact, locator=canonical_locator(tmp_path))
    assert isinstance(result, CopilotVscodeReadBatch)
    assert result.usage[0].quantities[0].amount == Decimal('0.12345678901234567890123456789')
    for literal in (b'NaN', b'Infinity', b'-Infinity'):
        invalid = exact.replace(b'0.12345678901234567890123456789', literal)
        result = read_copilot_vscode(invalid, locator=canonical_locator(tmp_path))
        assert isinstance(result, CopilotVscodeReadBatch)
        assert result.evidence[0].state == 'unresolved'
        assert result.evidence[0].reason == 'invalid_numeric_value'


@pytest.mark.parametrize(('field', 'value'), (
    ('promptTokens', True), ('promptTokens', 1.5), ('promptTokens', '1'),
    ('promptTokens', -1), ('promptTokens', 2**63),
    ('copilotCredits', True), ('copilotCredits', '0.5'), ('copilotCredits', -1),
    ('copilotCredits', 2**63),
))
def test_reader_rejects_invalid_numeric_wire_values(tmp_path, field, value):
    source = {'version': 3, 'sessionId': 'vs-session', 'requests': [{
        'requestId': 'request', 'agent': {'extensionId': {'value': 'github.copilot-chat'}}, field: value,
    }]}
    result = read_copilot_vscode(json.dumps(source).encode(), locator=canonical_locator(tmp_path))
    assert isinstance(result, CopilotVscodeReadBatch)
    assert result.evidence[0].state == 'unresolved'
    assert result.evidence[0].reason == 'invalid_numeric_value'


@pytest.mark.parametrize('field', ('promptTokens', 'completionTokens',
                                   'inputTokens', 'cachedTokens', 'outputTokens'))
@pytest.mark.parametrize('value', (True, 1.5, '1', None, [], {}, -1, 2**63),
                         ids=('bool', 'fraction', 'string', 'null', 'list', 'object', 'negative', 'overflow'))
def test_public_import_total_numeric_gate_quarantines_every_token_field(tmp_path, field, value):
    request = {
        'requestId': 'request', 'agent': {'extensionId': {'value': 'github.copilot-chat'}},
    }
    evidence_kind = 'turn_summary'
    raw_column = {'promptTokens': 'raw_input', 'completionTokens': 'raw_output',
                  'inputTokens': 'raw_input', 'cachedTokens': 'raw_cache_read',
                  'outputTokens': 'raw_output'}[field]
    if field in ('promptTokens', 'completionTokens'):
        request[field] = value
    else:
        request['modelTotals'] = [{
            'model': 'actual', 'inputTokens': 3, 'cachedTokens': 2, 'outputTokens': 1,
            field: value,
        }]
        evidence_kind = 'model_totals'
    source = {'version': 3, 'sessionId': f'invalid-{field}', 'requests': [request]}
    store = Storage(tmp_path / 'numeric.duckdb')
    store.import_source(canonical_locator(tmp_path), json.dumps(source).encode())
    with store.connect() as db:
        row = db.execute(
            f"SELECT {raw_column},state,reason FROM copilot_vscode_evidence WHERE evidence_kind=?",
            (evidence_kind,),
        ).one()
        assert row == (None, 'unresolved', 'invalid_numeric_value')
        assert {item[0] for item in db.execute('SELECT DISTINCT state FROM decision')} == {'unresolved'}
    store.close()


def test_reader_rejects_duplicate_actual_model_rows_and_missing_scope(tmp_path):
    source = {'version': 3, 'sessionId': 'vs-session', 'requests': [{
        'requestId': 'request', 'agent': {'extensionId': {'value': 'github.copilot-chat'}},
        'modelTotals': [
            {'model': 'same', 'inputTokens': 1, 'cachedTokens': 0, 'outputTokens': 1},
            {'model': 'same', 'inputTokens': 1, 'cachedTokens': 0, 'outputTokens': 1},
        ],
    }]}
    result = read_copilot_vscode(json.dumps(source).encode(), locator=canonical_locator(tmp_path))
    assert isinstance(result, CopilotVscodeReadBatch)
    assert all(e.state == 'unresolved' and e.reason == 'duplicate_actual_model' for e in result.evidence)
    rejected = read_copilot_vscode(FLAT, locator=str(tmp_path / 'chatSessions/session.json'))
    assert isinstance(rejected, RejectedSource)
    assert rejected.diagnostics[0].code == 'missing_copilot_vscode_scope'


def test_reversing_distinct_model_totals_preserves_logical_identity(tmp_path):
    value = json.loads(FLAT)
    first = read_copilot_vscode(json.dumps(value).encode(), locator=canonical_locator(tmp_path))
    value['requests'][0]['modelTotals'].reverse()
    second = read_copilot_vscode(json.dumps(value).encode(), locator=canonical_locator(tmp_path))
    assert isinstance(first, CopilotVscodeReadBatch) and isinstance(second, CopilotVscodeReadBatch)
    first_models = {(record.entry.native_id, evidence.request_id, record.model.model,
                     record.tokens.buckets.cache_read, record.tokens.buckets.output)
                    for record, evidence in zip(first.usage, first.evidence, strict=True)
                    if evidence.evidence_kind == 'model_totals'}
    second_models = {(record.entry.native_id, evidence.request_id, record.model.model,
                      record.tokens.buckets.cache_read, record.tokens.buckets.output)
                     for record, evidence in zip(second.usage, second.evidence, strict=True)
                     if evidence.evidence_kind == 'model_totals'}
    assert first_models == second_models


def test_storage_collapses_exact_duplicate_request_identity_but_quarantines_conflict(tmp_path):
    request = {
        'requestId': 'duplicate', 'responseId': 'response',
        'agent': {'extensionId': {'value': 'github.copilot-chat'}},
        'promptTokens': 4, 'completionTokens': 2, 'copilotCredits': 0.1,
    }
    exact = {'version': 3, 'sessionId': 'duplicate-session', 'requests': [request, dict(request)]}
    store = Storage(tmp_path / 'exact.duckdb')
    store.import_source(canonical_locator(tmp_path / 'exact'), json.dumps(exact).encode())
    with store.connect() as db:
        assert db.execute('SELECT count(*) FROM observation').one()[0] == 1
        assert db.execute("SELECT DISTINCT state FROM decision").one()[0] == 'selected'
    store.close()

    changed = dict(request)
    changed['completionTokens'] = 3
    conflict = {'version': 3, 'sessionId': 'duplicate-session', 'requests': [request, changed]}
    store = Storage(tmp_path / 'conflict.duckdb')
    store.import_source(canonical_locator(tmp_path / 'conflict'), json.dumps(conflict).encode())
    with store.connect() as db:
        assert db.execute('SELECT count(*) FROM observation').one()[0] == 2
        assert {row[0] for row in db.execute('SELECT DISTINCT state FROM decision')} == {'unresolved'}
        assert {row[0] for row in db.execute('SELECT DISTINCT state FROM quantity_decision')} == {'unresolved'}
    store.close()


def duplicate_compatibility_requests(path, changed, reverse):
    request = {
        'requestId': 'duplicate', 'responseId': 'response',
        'agent': {'extensionId': {'value': 'github.copilot-chat'}},
        'promptTokens': 12, 'completionTokens': 4, 'copilotCredits': 0.25,
        'modelTotals': [{
            'model': 'actual', 'inputTokens': 10, 'cachedTokens': 2, 'outputTokens': 4,
        }],
    }
    variant = json.loads(json.dumps(request))
    target = variant
    for key in path[:-1]:
        target = target[key]
    if changed is None:
        del target[path[-1]]
    else:
        target[path[-1]] = changed
    requests = [request, variant]
    return request, list(reversed(requests)) if reverse else requests


@pytest.mark.parametrize(('path', 'changed', 'evidence_kind'), (
    (('modelTotals', 0, 'inputTokens'), 99, 'model_totals'),
    (('modelTotals', 0, 'inputTokens'), None, 'model_totals'),
    (('modelTotals', 0, 'inputTokens'), True, 'model_totals'),
    (('copilotCredits',), 0.5, 'turn_summary'),
), ids=('raw-value', 'absent-present', 'invalidity', 'credit'))
def test_reader_native_identity_excludes_accounting_compatibility(
        tmp_path, path, changed, evidence_kind):
    _, requests = duplicate_compatibility_requests(path, changed, False)
    ids = []
    for request in requests:
        result = read_copilot_vscode(json.dumps({
            'version': 3, 'sessionId': 'stable-native-id', 'requests': [request],
        }).encode(), locator=canonical_locator(tmp_path))
        assert isinstance(result, CopilotVscodeReadBatch)
        ids.append(next(record.entry.native_id for record, evidence in
                        zip(result.usage, result.evidence, strict=True)
                        if evidence.evidence_kind == evidence_kind))
    assert ids[0] == ids[1]


def test_public_storage_keeps_compatibility_beside_stable_identity_and_private_content(tmp_path):
    request, requests = duplicate_compatibility_requests(
        ('modelTotals', 0, 'inputTokens'), 99, False)
    request['message'] = {'text': 'PRIVATE_PROMPT_CANARY_FIX3'}
    requests[1]['response'] = [{'value': 'PRIVATE_RESPONSE_CANARY_FIX3'}]
    locator = canonical_locator(tmp_path)
    database = tmp_path / 'stable-identity.duckdb'
    source = {'version': 3, 'sessionId': 'stable-native-id', 'requests': requests}
    store = Storage(database)
    store.import_source(locator, json.dumps(source).encode())

    def model_rows(storage):
        with storage.connect() as db:
            return tuple(db.execute(
                "SELECT o.native_entry_id,o.safe_facts_json,d.state,d.reason "
                "FROM observation o JOIN copilot_vscode_evidence e ON e.observation_id=o.id "
                "JOIN decision d ON d.observation_id=o.id AND d.measure='output' "
                "WHERE e.request_id='duplicate' AND e.evidence_kind='model_totals' "
                "ORDER BY o.safe_facts_json"
            ))

    rows = model_rows(store)
    assert len(rows) == 2
    assert len({row['native_entry_id'] for row in rows}) == 1
    assert {row['state'] for row in rows} == {'unresolved'}
    assert {row['reason'] for row in rows} == {'copilot_vscode_identity_conflict'}
    metadata = [json.loads(row['safe_facts_json']) for row in rows]
    assert metadata == [
        {'copilot_vscode_presence_v1': '1110'},
        {'copilot_vscode_presence_v1': '1110'},
    ]
    store.close()

    reopened = Storage(database)
    assert model_rows(reopened) == rows
    repaired = {
        'version': 3, 'sessionId': 'stable-native-id',
        'requests': [request, json.loads(json.dumps(request))],
    }
    repaired_data = json.dumps(repaired).encode()
    reopened.import_source(locator, repaired_data)
    repaired_rows = model_rows(reopened)
    assert {row['state'] for row in repaired_rows} == {'selected', 'excluded'}
    assert all(row['state'] != 'unresolved' for row in repaired_rows)
    with reopened.connect() as db:
        before = (reopened.snapshot().revision,
                  db.execute('SELECT count(*) FROM observation').one()[0],
                  db.execute('SELECT count(*) FROM copilot_vscode_evidence').one()[0])
        stored = '\n'.join(str(tuple(row)) for table in (
            'session_view', 'source_generation', 'observation', 'copilot_vscode_evidence',
            'diagnostic', 'token_value', 'quantity_value', 'decision', 'quantity_decision',
        ) for row in db.execute(f'SELECT * FROM {table}'))
    reopened.import_source(locator, repaired_data)
    with reopened.connect() as db:
        after = (reopened.snapshot().revision,
                 db.execute('SELECT count(*) FROM observation').one()[0],
                 db.execute('SELECT count(*) FROM copilot_vscode_evidence').one()[0])
    assert after == before
    assert 'PRIVATE_PROMPT_CANARY_FIX3' not in stored
    assert 'PRIVATE_RESPONSE_CANARY_FIX3' not in stored
    reopened.close()
    database_bytes = database.read_bytes()
    assert b'PRIVATE_PROMPT_CANARY_FIX3' not in database_bytes
    assert b'PRIVATE_RESPONSE_CANARY_FIX3' not in database_bytes


def vscode_evidence_decisions(store, evidence_kind):
    with store.connect() as db:
        return tuple(db.execute(
            "SELECT e.raw_input,e.raw_output,d.state,d.reason FROM copilot_vscode_evidence e "
            "JOIN decision d ON d.observation_id=e.observation_id AND d.measure='output' "
            "WHERE e.request_id='duplicate' AND e.evidence_kind=? "
            "ORDER BY e.raw_input NULLS FIRST,e.raw_output NULLS FIRST,d.state",
            (evidence_kind,),
        ))


@pytest.mark.parametrize(('path', 'changed', 'evidence_kind'), (
    (('modelTotals', 0, 'inputTokens'), 99, 'model_totals'),
    (('modelTotals', 0, 'inputTokens'), None, 'model_totals'),
    (('promptTokens',), 99, 'turn_summary'),
    (('completionTokens',), 99, 'turn_summary'),
), ids=('model-input-value', 'model-input-presence', 'covered-prompt', 'covered-completion'))
@pytest.mark.parametrize('reverse', [False, True], ids=('original-order', 'reversed-order'))
def test_same_file_duplicate_compatibility_conflicts_reopen_and_repair(
        tmp_path, path, changed, evidence_kind, reverse):
    request, requests = duplicate_compatibility_requests(path, changed, reverse)
    locator = canonical_locator(tmp_path)
    source = {'version': 3, 'sessionId': 'duplicate-compatibility', 'requests': requests}
    database = tmp_path / f'same-file-{evidence_kind}-{changed}-{reverse}.duckdb'
    store = Storage(database)
    store.import_source(locator, json.dumps(source).encode())
    rows = vscode_evidence_decisions(store, evidence_kind)
    assert len(rows) == 2
    assert {row[2:] for row in rows} == {('unresolved', 'copilot_vscode_identity_conflict')}
    store.close()

    reopened = Storage(database)
    assert vscode_evidence_decisions(reopened, evidence_kind) == rows
    repaired = {
        'version': 3, 'sessionId': 'duplicate-compatibility',
        'requests': [request, json.loads(json.dumps(request))],
    }
    repaired_data = json.dumps(repaired).encode()
    reopened.import_source(locator, repaired_data)
    repaired_rows = vscode_evidence_decisions(reopened, evidence_kind)
    expected_states = ({'selected', 'excluded'} if evidence_kind == 'model_totals'
                       else {'excluded'})
    assert {row[2] for row in repaired_rows} == expected_states
    assert all(row[2] != 'unresolved' for row in repaired_rows)
    with reopened.connect() as db:
        before = (reopened.snapshot().revision,
                  db.execute('SELECT count(*) FROM observation').one()[0],
                  db.execute('SELECT count(*) FROM copilot_vscode_evidence').one()[0])
    reopened.import_source(locator, repaired_data)
    with reopened.connect() as db:
        after = (reopened.snapshot().revision,
                 db.execute('SELECT count(*) FROM observation').one()[0],
                 db.execute('SELECT count(*) FROM copilot_vscode_evidence').one()[0])
    assert after == before
    reopened.close()


@pytest.mark.parametrize(('path', 'changed'), (
    (('modelTotals', 0, 'inputTokens'), 99),
    (('modelTotals', 0, 'inputTokens'), None),
    (('promptTokens',), 99),
    (('completionTokens',), 99),
), ids=('model-input-value', 'model-input-presence', 'covered-prompt', 'covered-completion'))
@pytest.mark.parametrize('reverse_requests', [False, True], ids=('original-requests', 'reversed-requests'))
@pytest.mark.parametrize('reverse_imports', [False, True], ids=('flat-first', 'log-first'))
def test_flat_log_duplicate_compatibility_conflicts_in_every_order_and_repairs(
        tmp_path, path, changed, reverse_requests, reverse_imports):
    request, requests = duplicate_compatibility_requests(path, changed, reverse_requests)
    flat = SourcePayload(canonical_locator(tmp_path, 'flat', '.json'), json.dumps({
        'version': 3, 'sessionId': 'duplicate-compatibility', 'requests': [request],
    }).encode())
    log_locator = canonical_locator(tmp_path, 'log', '.jsonl')
    log = SourcePayload(log_locator, operation_log({
        'version': 3, 'sessionId': 'duplicate-compatibility', 'requests': requests,
    }))
    database = tmp_path / f'alternates-{changed}-{reverse_requests}-{reverse_imports}.duckdb'
    store = Storage(database)
    store.import_sources((log, flat) if reverse_imports else (flat, log))
    with store.connect() as db:
        assert {row[0] for row in db.execute(
            "SELECT DISTINCT d.state FROM decision d JOIN observation o ON o.id=d.observation_id "
            "WHERE o.session_id IN (SELECT id FROM session WHERE harness='copilot-vscode')"
        )} == {'unresolved'}
        assert {row[0] for row in db.execute(
            "SELECT DISTINCT d.reason FROM decision d JOIN observation o ON o.id=d.observation_id "
            "WHERE o.session_id IN (SELECT id FROM session WHERE harness='copilot-vscode')"
        )} == {'copilot_vscode_alternate_conflict'}
    store.close()

    reopened = Storage(database)
    with reopened.connect() as db:
        assert {row[0] for row in db.execute(
            "SELECT DISTINCT d.state FROM decision d JOIN observation o ON o.id=d.observation_id "
            "WHERE o.session_id IN (SELECT id FROM session WHERE harness='copilot-vscode')"
        )} == {'unresolved'}
    reopened.import_source(log_locator, operation_log({
        'version': 3, 'sessionId': 'duplicate-compatibility', 'requests': [request],
    }))
    with reopened.connect() as db:
        assert 'unresolved' not in {row[0] for row in db.execute(
            "SELECT DISTINCT d.state FROM decision d JOIN observation o ON o.id=d.observation_id "
            "WHERE o.session_id IN (SELECT id FROM session WHERE harness='copilot-vscode')"
        )}
    reopened.close()


@pytest.mark.parametrize('requests', (
    (
        {'requestId': 'a:b', 'responseId': 'c'},
        {'requestId': 'a', 'responseId': 'b:c'},
    ),
    (
        {'requestId': 'same'},
        {'requestId': 'same', 'responseId': 'no-response'},
    ),
))
def test_observation_identity_is_typed_and_delimiter_safe(tmp_path, requests):
    rows = []
    for identity in requests:
        rows.append({
            **identity,
            'agent': {'extensionId': {'value': 'github.copilot-chat'}},
            'promptTokens': 4, 'completionTokens': 2,
        })
    source = {'version': 3, 'sessionId': 'identity-tuples', 'requests': rows}
    store = Storage(tmp_path / 'identity.duckdb')
    store.import_source(canonical_locator(tmp_path), json.dumps(source).encode())
    with store.connect() as db:
        assert db.execute('SELECT count(*) FROM observation').one()[0] == 2
        assert db.execute('SELECT count(DISTINCT native_entry_id) FROM observation').one()[0] == 2
        assert {row[0] for row in db.execute('SELECT DISTINCT state FROM decision')} == {'selected'}
    store.close()

def test_reader_workspace_uri_precedence_and_rejection(tmp_path):
    value = json.loads(FLAT)
    value['workingDirectory'] = 'https://example.invalid/work'
    context = (('workspace.json', b'{"folder":"file:///sidecar/work"}'),)
    result = read_copilot_vscode(json.dumps(value).encode(), locator=canonical_locator(tmp_path), context=context)
    assert isinstance(result, CopilotVscodeReadBatch) and result.session.cwd == '/sidecar/work'
    value['workingDirectory'] = 'file:///primary/work'
    result = read_copilot_vscode(json.dumps(value).encode(), locator=canonical_locator(tmp_path), context=context)
    assert isinstance(result, CopilotVscodeReadBatch) and result.session.cwd == '/primary/work'
    for uri in ('file://user:secret@host/work', 'file:///work?query=1', 'file:///work#fragment',
                'file:///work%00bad', 'file:///work%2', 'file:///work%GG', 'file:///work%FF'):
        value['workingDirectory'] = uri
        result = read_copilot_vscode(json.dumps(value).encode(), locator=canonical_locator(tmp_path), context=())
        assert isinstance(result, CopilotVscodeReadBatch) and result.session.cwd is None


@pytest.mark.parametrize('uri', (
    'file://[broken/path', 'file://[v.bad]/path',
    'file:///work\nPRIVATE_URI_CANARY', 'file:///work%0APRIVATE_URI_CANARY',
    'file://localhost:80/work', 'file://user:secret@localhost/work',
    'file:////server/share', 'file://localhost//server/share', 'file:///C:/work',
    'file:///work%2', 'file:///work%GG', 'file:///work%00PRIVATE_URI_CANARY',
    'file:relative', 'https://example.invalid/work',
))
@pytest.mark.parametrize('sidecar', [False, True])
def test_workspace_file_uri_parser_is_total_and_local_only(tmp_path, uri, sidecar):
    value = json.loads(FLAT)
    context = ()
    if sidecar:
        value.pop('workingDirectory', None)
        context = (('workspace.json', json.dumps({'folder': uri}).encode()),)
    else:
        value['workingDirectory'] = uri
    result = read_copilot_vscode(json.dumps(value).encode(), locator=canonical_locator(tmp_path), context=context)
    assert isinstance(result, CopilotVscodeReadBatch)
    assert result.session.cwd is None
    assert 'PRIVATE_URI_CANARY' not in repr(result.diagnostics)


def test_storage_persists_vscode_evidence_and_reconciles_models_and_credit_control(tmp_path):
    store = Storage(tmp_path / 'ledger.duckdb')
    flat = canonical_locator(tmp_path, 'one', '.json')
    log = canonical_locator(tmp_path, 'two', '.jsonl')
    store.import_sources((SourcePayload(flat, FLAT), SourcePayload(log, LOG)))
    with store.connect() as db:
        assert db.execute("SELECT count(*) FROM session WHERE harness='copilot-vscode'").one()[0] == 1
        assert db.execute('SELECT count(DISTINCT session_native_id) FROM copilot_vscode_evidence').one()[0] == 1
        token = list(db.execute(
            "SELECT e.request_id,e.evidence_kind,o.model,d.state FROM copilot_vscode_evidence e "
            "JOIN observation o ON o.id=e.observation_id JOIN decision d ON d.observation_id=o.id "
            "WHERE d.measure='output' ORDER BY e.request_id,e.evidence_kind,o.model,d.state"))
        assert ('request-a', 'model_totals', 'model-a', 'selected') in token
        assert ('request-a', 'model_totals', 'model-b', 'selected') in token
        assert ('request-a', 'turn_summary', None, 'excluded') in token
        assert ('request-b', 'turn_summary', None, 'selected') in token
        quantities = list(db.execute(
            "SELECT e.request_id,e.evidence_kind,q.amount_decimal,d.state FROM copilot_vscode_evidence e "
            "JOIN quantity_value q ON q.observation_id=e.observation_id "
            "JOIN quantity_decision d ON d.observation_id=q.observation_id ORDER BY e.evidence_kind,e.request_id"))
        assert ('request-b', 'session_control', '0.5', 'selected') in quantities
        assert ('request-a', 'turn_summary', '0', 'excluded') in quantities
        assert ('request-b', 'turn_summary', '0.25', 'excluded') in quantities
    store.close()


def test_session_credit_control_covers_only_turns_recorded_through_its_request(tmp_path):
    value = {
        'version': 3,
        'sessionId': 'credit-coverage',
        'requests': [
            {'requestId': 'first', 'agent': {'extensionId': {'value': 'github.copilot-chat'}},
             'copilotCredits': 0.1},
            {'requestId': 'controlled-through-here',
             'agent': {'extensionId': {'value': 'github.copilot-chat'}},
             'copilotCredits': 0.2, 'sessionCopilotCredits': 0.5},
            {'requestId': 'after-control', 'agent': {'extensionId': {'value': 'github.copilot-chat'}},
             'copilotCredits': 0.3},
        ],
    }
    store = Storage(tmp_path / 'ledger.duckdb')
    store.import_source(canonical_locator(tmp_path), json.dumps(value).encode())
    with store.connect() as db:
        decisions = {(row['request_id'], row['evidence_kind']): row['state'] for row in db.execute(
            "SELECT e.request_id,e.evidence_kind,d.state FROM copilot_vscode_evidence e "
            "JOIN quantity_decision d ON d.observation_id=e.observation_id")}
    assert decisions == {
        ('first', 'turn_summary'): 'excluded',
        ('controlled-through-here', 'turn_summary'): 'excluded',
        ('controlled-through-here', 'session_control'): 'selected',
        ('after-control', 'turn_summary'): 'selected',
    }
    store.close()


def test_credit_controls_select_latest_equal_maximum_and_leave_later_turn_uncovered(tmp_path):
    controls = (('first', 0.1, 0.3), ('second', 0.2, 0.5), ('third', None, 0.5),
                ('after', 0.1, None))
    requests = []
    for request_id, turn, control in controls:
        request = {'requestId': request_id,
                   'agent': {'extensionId': {'value': 'github.copilot-chat'}}}
        if turn is not None:
            request['copilotCredits'] = turn
        if control is not None:
            request['sessionCopilotCredits'] = control
        requests.append(request)
    store = Storage(tmp_path / 'ledger.duckdb')
    source = {'version': 3, 'sessionId': 'controls', 'requests': requests}
    store.import_source(canonical_locator(tmp_path), json.dumps(source).encode())
    with store.connect() as db:
        selected = list(db.execute(
            "SELECT e.request_id,e.evidence_kind FROM copilot_vscode_evidence e "
            "JOIN quantity_decision d ON d.observation_id=e.observation_id WHERE d.state='selected' "
            "ORDER BY e.request_id,e.evidence_kind"))
    assert selected == [('after', 'turn_summary'), ('third', 'session_control')]
    store.close()


def credit_order_state(order):
    rows = {
        'a': {'requestId': 'a', 'agent': {'extensionId': {'value': 'github.copilot-chat'}},
              'copilotCredits': 0.1},
        'b': {'requestId': 'b', 'agent': {'extensionId': {'value': 'github.copilot-chat'}},
              'copilotCredits': 0.2},
        'control': {'requestId': 'control', 'agent': {'extensionId': {'value': 'github.copilot-chat'}},
                    'copilotCredits': 0, 'sessionCopilotCredits': 0.5},
    }
    return {'version': 3, 'sessionId': 'canonical-credit-order',
            'requests': [rows[name] for name in order]}


def selected_credit_total(store):
    with store.connect() as db:
        values = [Decimal(row[0]) for row in db.execute(
            "SELECT q.amount_decimal FROM quantity_value q JOIN quantity_decision d "
            "ON d.observation_id=q.observation_id AND d.measure=q.measure "
            "WHERE q.measure='ai_credits' AND d.state='selected'"
        )]
    return sum(values, Decimal(0))


def test_same_locator_credit_coverage_uses_only_current_generation_order(tmp_path):
    locator = canonical_locator(tmp_path)
    store = Storage(tmp_path / 'current-order.duckdb')
    store.import_source(locator, json.dumps(credit_order_state(('a', 'b', 'control'))).encode())
    store.import_source(locator, json.dumps(credit_order_state(('control', 'a', 'b'))).encode())
    assert selected_credit_total(store) == Decimal('0.8')
    store.close()
    reopened = Storage(store.path)
    assert selected_credit_total(reopened) == Decimal('0.8')
    reopened.close()


@pytest.mark.parametrize('reverse', [False, True])
def test_credit_coverage_uses_canonical_log_order_not_matching_flat_order(tmp_path, reverse):
    flat = SourcePayload(canonical_locator(tmp_path, 'flat', '.json'),
                         json.dumps(credit_order_state(('a', 'b', 'control'))).encode())
    log_state = credit_order_state(('control', 'a', 'b'))
    log = SourcePayload(canonical_locator(tmp_path, 'log', '.jsonl'), operation_log(log_state))
    store = Storage(tmp_path / 'alternate-order.duckdb')
    store.import_sources((log, flat) if reverse else (flat, log))
    assert selected_credit_total(store) == Decimal('0.8')
    store.close()


def test_storage_scope_separates_installations_and_reimport_restart_is_idempotent(tmp_path):
    path = tmp_path / 'ledger.duckdb'
    store = Storage(path)
    stable = canonical_locator(tmp_path / 'stable')
    insiders = canonical_locator(tmp_path / 'insiders')
    first_revision = store.import_sources((SourcePayload(stable, FLAT), SourcePayload(insiders, FLAT)))
    with store.connect() as db:
        assert db.execute("SELECT count(*) FROM session WHERE harness='copilot-vscode'").one()[0] == 2
        before = tuple(db.execute("SELECT id FROM observation ORDER BY id"))
    assert store.import_source(stable, FLAT) == first_revision
    store.close()
    reopened = Storage(path)
    with reopened.connect() as db:
        assert tuple(db.execute("SELECT id FROM observation ORDER BY id")) == before
        assert db.execute('SELECT schema_version FROM ledger_meta').one()[0] == 6
    reopened.close()


def test_scoped_reconciliation_does_not_rewrite_colliding_other_installation(tmp_path):
    store = Storage(tmp_path / 'scoped.duckdb')
    stable = canonical_locator(tmp_path / 'stable')
    insiders = canonical_locator(tmp_path / 'insiders')
    store.import_sources((SourcePayload(stable, FLAT), SourcePayload(insiders, FLAT)))
    insiders_id = f"copilot-vscode:{copilot_vscode_scope(insiders)}:vs-session"

    def decision_rows(with_rowid):
        prefix = 'd.rowid,' if with_rowid else ''
        with store.connect() as db:
            return tuple(db.execute(
                f"SELECT {prefix}d.observation_id,d.measure,d.state,d.owner_session,d.canonical,d.reason,d.rule_version "
                "FROM decision d JOIN observation o ON o.id=d.observation_id "
                "WHERE o.session_id=? ORDER BY d.observation_id,d.measure",
                (insiders_id,),
            ))

    before_rowids = decision_rows(True)
    store.import_source(stable, FLAT + b'\n')
    assert decision_rows(True) == before_rowids
    scoped = decision_rows(False)
    with store.connect(write=True) as db:
        store._reconcile(db)
    assert decision_rows(False) == scoped
    store.close()


def test_storage_conflicting_alternates_follow_missing_and_reappearance_lifecycle(tmp_path):
    store = Storage(tmp_path / 'ledger.duckdb')
    flat = canonical_locator(tmp_path, 'one', '.json')
    log = canonical_locator(tmp_path, 'two', '.jsonl')
    changed = json.loads(FLAT)
    changed['requests'][1]['promptTokens'] = 9
    changed_flat = json.dumps(changed, separators=(',', ':')).encode()
    store.import_sources((SourcePayload(flat, changed_flat), SourcePayload(log, LOG)))

    def state():
        with store.connect() as db:
            return {row[0] for row in db.execute(
                "SELECT DISTINCT d.state FROM decision d JOIN observation o ON o.id=d.observation_id "
                "JOIN copilot_vscode_evidence e ON e.observation_id=o.id "
                "WHERE e.request_id='request-b' AND e.evidence_kind='turn_summary'")}

    assert state() == {'unresolved'}
    store.mark_missing((log,))
    assert state() == {'selected', 'excluded'}
    store.import_source(log, LOG)
    assert state() == {'unresolved'}
    store.close()


@pytest.mark.parametrize('reverse', [False, True])
def test_storage_treats_flat_and_log_as_whole_session_alternates_not_per_request_mix(tmp_path, reverse):
    store = Storage(tmp_path / 'ledger.duckdb')
    flat = canonical_locator(tmp_path, 'one', '.json')
    log = canonical_locator(tmp_path, 'two', '.jsonl')
    superset = json.loads(FLAT)
    superset['requests'].append({
        'requestId': 'flat-only', 'agent': {'extensionId': {'value': 'github.copilot-chat'}},
        'promptTokens': 4, 'completionTokens': 1,
    })
    inputs = [SourcePayload(flat, json.dumps(superset).encode()), SourcePayload(log, LOG)]
    store.import_sources(tuple(reversed(inputs)) if reverse else tuple(inputs))
    with store.connect() as db:
        assert {row[0] for row in db.execute(
            "SELECT DISTINCT d.state FROM decision d JOIN observation o ON o.id=d.observation_id "
            "WHERE o.session_id IN (SELECT id FROM session WHERE harness='copilot-vscode')")} == {'unresolved'}
        assert {row[0] for row in db.execute(
            "SELECT DISTINCT d.state FROM quantity_decision d JOIN observation o ON o.id=d.observation_id "
            "WHERE o.session_id IN (SELECT id FROM session WHERE harness='copilot-vscode')")} == {'unresolved'}
        session_id = db.execute("SELECT id FROM session WHERE harness='copilot-vscode'").one()[0]
    with pytest.raises(TranscriptUnavailable, match='unambiguous'):
        read_transcript_page(store, (str(tmp_path),), session_id)
    store.close()


@pytest.mark.parametrize(('field', 'changed'), (
    ('inputTokens', 99),
    ('inputTokens', None),
    ('promptTokens', 99),
    ('promptTokens', None),
    ('completionTokens', 99),
    ('completionTokens', None),
))
@pytest.mark.parametrize('reverse', [False, True])
def test_current_alternates_compare_raw_accounting_values_and_presence(tmp_path, field, changed, reverse):
    request = {
        'requestId': 'request', 'responseId': 'response',
        'agent': {'extensionId': {'value': 'github.copilot-chat'}},
        'promptTokens': 12, 'completionTokens': 4,
        'modelTotals': [{'model': 'actual', 'inputTokens': 10, 'cachedTokens': 2, 'outputTokens': 4}],
    }
    flat_state = {'version': 3, 'sessionId': 'raw-signature', 'requests': [request]}
    log_state = json.loads(json.dumps(flat_state))
    target = log_state['requests'][0]['modelTotals'][0] if field == 'inputTokens' else log_state['requests'][0]
    if changed is None:
        del target[field]
    else:
        target[field] = changed
    sources = [
        SourcePayload(canonical_locator(tmp_path, 'flat', '.json'), json.dumps(flat_state).encode()),
        SourcePayload(canonical_locator(tmp_path, 'log', '.jsonl'), operation_log(log_state)),
    ]
    store = Storage(tmp_path / f'{field}-{changed}-{reverse}.duckdb')
    store.import_sources(tuple(reversed(sources)) if reverse else tuple(sources))
    store.close()
    reopened = Storage(store.path)
    with reopened.connect() as db:
        assert {row[0] for row in db.execute(
            "SELECT DISTINCT d.state FROM decision d JOIN observation o ON o.id=d.observation_id "
            "WHERE o.session_id IN (SELECT id FROM session WHERE harness='copilot-vscode')"
        )} == {'unresolved'}
        assert {row[0] for row in db.execute(
            "SELECT DISTINCT d.reason FROM decision d JOIN observation o ON o.id=d.observation_id "
            "WHERE o.session_id IN (SELECT id FROM session WHERE harness='copilot-vscode')"
        )} == {'copilot_vscode_alternate_conflict'}
    reopened.close()


def test_same_locator_replacement_rechecks_current_alternate_compatibility(tmp_path):
    flat = canonical_locator(tmp_path, 'flat', '.json')
    log = canonical_locator(tmp_path, 'log', '.jsonl')
    state = {
        'version': 3, 'sessionId': 'raw-replacement', 'requests': [{
            'requestId': 'request', 'agent': {'extensionId': {'value': 'github.copilot-chat'}},
            'modelTotals': [{'model': 'actual', 'inputTokens': 10, 'cachedTokens': 2, 'outputTokens': 4}],
        }],
    }
    store = Storage(tmp_path / 'replacement.duckdb')
    store.import_sources((SourcePayload(flat, json.dumps(state).encode()),
                          SourcePayload(log, operation_log(state))))
    state['requests'][0]['modelTotals'][0]['inputTokens'] = 99
    store.import_source(log, operation_log(state))
    with store.connect() as db:
        assert {row[0] for row in db.execute(
            "SELECT DISTINCT d.state FROM decision d JOIN observation o ON o.id=d.observation_id "
            "WHERE o.session_id IN (SELECT id FROM session WHERE harness='copilot-vscode')"
        )} == {'unresolved'}
    store.close()
    reopened = Storage(store.path)
    with reopened.connect() as db:
        assert {row[0] for row in db.execute(
            "SELECT DISTINCT d.state FROM decision d JOIN observation o ON o.id=d.observation_id "
            "WHERE o.session_id IN (SELECT id FROM session WHERE harness='copilot-vscode')"
        )} == {'unresolved'}
    reopened.close()


def test_empty_current_alternate_conflicts_with_nonempty_session_source(tmp_path):
    store = Storage(tmp_path / 'ledger.duckdb')
    empty = {'version': 3, 'sessionId': 'vs-session', 'requests': []}
    store.import_sources((
        SourcePayload(canonical_locator(tmp_path, 'one', '.json'), json.dumps(empty).encode()),
        SourcePayload(canonical_locator(tmp_path, 'two', '.jsonl'), LOG),
    ))
    with store.connect() as db:
        assert {row[0] for row in db.execute(
            "SELECT DISTINCT d.state FROM decision d JOIN observation o ON o.id=d.observation_id "
            "WHERE o.session_id IN (SELECT id FROM session WHERE harness='copilot-vscode')")} == {'unresolved'}
        assert {row[0] for row in db.execute(
            "SELECT DISTINCT d.state FROM quantity_decision d JOIN observation o ON o.id=d.observation_id "
            "WHERE o.session_id IN (SELECT id FROM session WHERE harness='copilot-vscode')")} == {'unresolved'}
    store.close()


def test_latest_rejected_operation_log_falls_back_to_current_flat_source(tmp_path):
    store = Storage(tmp_path / 'ledger.duckdb')
    flat = Path(canonical_locator(tmp_path, 'one', '.json'))
    log = Path(canonical_locator(tmp_path, 'two', '.jsonl'))
    flat.parent.mkdir(parents=True)
    log.parent.mkdir(parents=True)
    flat.write_bytes(FLAT)
    log.write_bytes(LOG)
    store.import_sources((SourcePayload(str(flat), FLAT), SourcePayload(str(log), LOG)))
    malformed = LOG + b'{complete but invalid}\n'
    log.write_bytes(malformed)
    store.import_source(str(log), malformed)
    with store.connect() as db:
        session_id = db.execute("SELECT id FROM session WHERE harness='copilot-vscode'").one()[0]
        selected = {row[0] for row in db.execute(
            "SELECT DISTINCT e.representation FROM copilot_vscode_evidence e "
            "JOIN decision d ON d.observation_id=e.observation_id "
            "JOIN source_generation g ON g.id=e.source_id WHERE d.state='selected' "
            "AND NOT EXISTS (SELECT 1 FROM source_generation newer "
            "WHERE newer.locator=g.locator AND newer.generation>g.generation)")}
        assert selected == {'flat'}
        assert 'malformed_chat_operation' in {row[0] for row in db.execute('SELECT code FROM diagnostic')}
    page = read_transcript_page(store, (str(tmp_path),), session_id)
    assert page.transcript.title == 'Synthetic Copilot session'
    store.close()


def test_all_current_vscode_sources_marked_missing_return_missing_after_reopen(tmp_path):
    flat = Path(canonical_locator(tmp_path, 'flat', '.json'))
    log = Path(canonical_locator(tmp_path, 'log', '.jsonl'))
    flat.parent.mkdir(parents=True)
    log.parent.mkdir(parents=True)
    flat.write_bytes(FLAT)
    log.write_bytes(LOG)
    path = tmp_path / 'missing.duckdb'
    store = Storage(path)
    store.import_sources((SourcePayload(str(flat), FLAT), SourcePayload(str(log), LOG)))
    with store.connect() as db:
        session_id = db.execute("SELECT id FROM session WHERE harness='copilot-vscode'").one()[0]
        selected_before = db.execute("SELECT count(*) FROM decision WHERE state='selected'").one()[0]
    flat.unlink()
    log.unlink()
    store.mark_missing((str(flat), str(log)))
    store.close()
    reopened = Storage(path)
    with reopened.connect() as db:
        assert db.execute("SELECT count(*) FROM decision WHERE state='selected'").one()[0] == selected_before
    with pytest.raises(TranscriptUnavailable) as unavailable:
        read_transcript_page(reopened, (str(tmp_path),), session_id)
    assert unavailable.value.kind == 'missing'
    reopened.close()


def test_available_rejected_latest_vscode_generation_returns_changed(tmp_path):
    source = Path(canonical_locator(tmp_path))
    source.parent.mkdir(parents=True)
    source.write_bytes(FLAT)
    store = Storage(tmp_path / 'changed.duckdb')
    store.import_source(str(source), FLAT)
    with store.connect() as db:
        session_id = db.execute("SELECT id FROM session WHERE harness='copilot-vscode'").one()[0]
    malformed = b'{complete but malformed}\n'
    source.write_bytes(malformed)
    store.import_source(str(source), malformed)
    with pytest.raises(TranscriptUnavailable) as unavailable:
        read_transcript_page(store, (str(tmp_path),), session_id)
    assert unavailable.value.kind == 'changed'
    store.close()


def test_storage_sidecar_change_preserves_accounting_identity_but_conflicting_cwd_is_unavailable(tmp_path):
    store = Storage(tmp_path / 'ledger.duckdb')
    locator = canonical_locator(tmp_path)
    value = json.loads(FLAT)
    del value['workingDirectory']
    source = json.dumps(value, separators=(',', ':')).encode()
    first = SourcePayload(locator, source, (('workspace.json', b'{"folder":"file:///first"}'),))
    second = SourcePayload(locator, source, (('workspace.json', b'{"folder":"file:///second"}'),))
    store.import_sources((first,))
    with store.connect() as db:
        ids = tuple(db.execute("SELECT id FROM observation ORDER BY id"))
    store.import_sources((second,))
    with store.connect() as db:
        assert tuple(db.execute("SELECT id FROM observation ORDER BY id")) == ids
        assert db.execute("SELECT cwd FROM session WHERE harness='copilot-vscode'").one()[0] is None
        assert db.execute('SELECT count(*) FROM source_generation WHERE locator=?', (locator,)).one()[0] == 2
    store.close()


def test_flat_and_operation_log_project_the_same_bounded_inert_transcript(tmp_path):
    native = 'scope:vs-session'
    flat = parse_copilot_vscode_transcript(FLAT, 'vs-session', 'copilot-vscode:' + native,
                                           native, representation='flat')
    log = parse_copilot_vscode_transcript(LOG, 'vs-session', 'copilot-vscode:' + native,
                                          native, representation='operation_log')
    assert flat.title == log.title == 'Synthetic Copilot session'
    messages = [entry for entry in flat.entries if isinstance(entry, Message)]
    log_messages = [entry for entry in log.entries if isinstance(entry, Message)]
    assert [(message.role, message.id) for message in messages] == [
        ('user', 'request-a'), ('assistant', 'response-a'),
        ('user', 'request-b'), ('assistant', 'response-b'),
    ]
    assert [(message.role, message.id) for message in log_messages] == [(message.role, message.id) for message in messages]
    assert messages[0].blocks == (TextBlock('Explain <b>safe</b>'),)
    assert isinstance(messages[1].blocks[0], ReasoningBlock)
    assert messages[1].blocks[1] == TextBlock('Use [link](javascript:alert(1)).')
    assert messages[1].blocks[2] == AttachmentBlock('Attachment: reference')
    tool = messages[3].blocks[0]
    assert isinstance(tool, ToolBlock) and tool.name == 'search' and tool.source_line == 1
    assert isinstance(log_messages[3].blocks[0], ToolBlock) and log_messages[3].blocks[0].source_line == 3
    assert any(isinstance(entry, Notice) and entry.label == 'Unsupported Copilot response part'
               for entry in flat.entries)
    rendered = repr(flat)
    assert 'user:secret' not in rendered and 'PRIVATE_UNSUPPORTED_CANARY' not in rendered


def test_transcript_uses_response_origin_and_accounting_timestamp_fallback():
    initial = {
        'version': 3, 'sessionId': 'origin', 'creationDate': 1789540000000,
        'requests': [{
            'requestId': 'request', 'timestamp': 1789540001000,
            'message': {'text': 'hello'},
            'agent': {'extensionId': {'value': 'github.copilot-chat'}},
        }],
    }
    response = [{'kind': 'toolInvocation', 'toolName': 'search', 'arguments': '{}', 'result': 'done'}]
    operations = (
        {'kind': 0, 'v': initial},
        {'kind': 1, 'k': ['requests', 0, 'response'], 'v': response},
        {'kind': 1, 'k': ['requests', 0, 'responseTimestamp'], 'v': 'invalid'},
    )
    data = b''.join(json.dumps(operation).encode() + b'\n' for operation in operations)
    transcript = parse_copilot_vscode_transcript(
        data, 'origin', 'copilot-vscode:scope:origin', 'scope:origin', representation='operation_log')
    messages = [entry for entry in transcript.entries if isinstance(entry, Message)]
    assistant = messages[1]
    assert assistant.timestamp == '2026-09-16T06:26:41+00:00'
    tool = assistant.blocks[0]
    assert isinstance(tool, ToolBlock) and tool.source_line == 2
