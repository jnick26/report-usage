from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
import json

import pytest

from harness_usage.domain import Interval, Known, RecordedEstimate, Unknown, Undated
from harness_usage.pi_reader import ReadBatch, RejectedSource, read_pi

FIXTURES = Path(__file__).parent / 'fixtures/pi/source'
HEADER = {'type': 'session', 'version': 3, 'id': 'native', 'timestamp': '2026-09-12T08:00:00Z', 'cwd': '/project'}


def source(*entries: dict[str, object], header: dict[str, object] = HEADER) -> bytes:
    return ('\n'.join(json.dumps(e) for e in (header, *entries)) + '\n').encode()


def assistant(usage: object, **fields: object) -> dict[str, object]:
    return {'type': 'message', 'id': 'entry', 'parentId': None, 'timestamp': '2026-09-12T09:00:00Z', 'message': {'role': 'assistant', 'stopReason': 'stop', 'usage': usage, **fields}}


def read(data: bytes) -> ReadBatch:
    result = read_pi(data, locator='/sessions/native.jsonl')
    assert isinstance(result, ReadBatch)
    return result


def test_all_four_kinds_preserve_time_identity_money_and_source_lines() -> None:
    batch = read((FIXTURES / 'auxiliary.jsonl').read_bytes())
    assert [record.kind for record in batch.usage] == ['assistant', 'compaction', 'branch_summary']
    assert [record.tokens.total for record in batch.usage] == [Known(970), Known(35), Known(10)]
    assert batch.usage[0].entry.line == 2
    assert batch.usage[0].time.at == datetime(2026, 9, 12, 9, tzinfo=UTC)
    assert batch.usage[1].time == Interval(None, datetime(2026, 9, 12, 9, tzinfo=UTC))
    assert batch.usage[1].model.provider is None and batch.usage[1].model.model is None
    assert batch.usage[0].money.amount == Decimal('0.01')
    tool = read((FIXTURES / 'tool-parent.jsonl').read_bytes()).usage[0]
    assert tool.kind == 'tool_result' and isinstance(tool.time, Interval)
    assert tool.tool_call_id == 'synthetic-child-call'


def test_native_names_and_nonusage_edges_and_lifetime_without_transcripts() -> None:
    batch = read(source({'type': 'session_info', 'id': 'a', 'parentId': None, 'name': 'A <name>', 'timestamp': '2026-09-12T10:00:00Z'}, {'type': 'message', 'id': 'b', 'parentId': 'a', 'timestamp': '2026-09-12T11:00:00Z', 'message': {'role': 'user', 'content': 'PRIVATE SENTINEL'}}, assistant({'input': 0}, content='PRIVATE SENTINEL', errorMessage='PRIVATE SENTINEL', details={'text': 'PRIVATE SENTINEL'})))
    assert batch.session.id == 'pi:native'
    assert batch.session.display_name == 'A <name>'
    assert batch.session.last_observed == datetime(2026, 9, 12, 11, tzinfo=UTC)
    assert batch.entries[1].parent_id == 'a'
    assert 'PRIVATE SENTINEL' not in repr(batch)
    assert read(source({'type': 'session_info', 'id': 'a', 'name': 'old'}, {'type': 'session_info', 'id': 'b'})).session.display_name is None


@pytest.mark.parametrize('header', [{}, {**HEADER, 'version': True}, {**HEADER, 'version': 2}, {**HEADER, 'id': ''}, {**HEADER, 'cwd': []}, {**HEADER, 'timestamp': 'bad'}])
def test_invalid_header_rejects_source(header: dict[str, object]) -> None:
    assert isinstance(read_pi(source(header=header), locator='x'), RejectedSource)


def test_valid_final_record_without_newline_and_pending_tail_checkpoint() -> None:
    complete = source(assistant({'input': 1}))
    assert read(complete[:-1]).complete_bytes == len(complete) - 1
    partial = complete + b'{"type":"message","id":"next"'
    batch = read(partial)
    assert batch.pending_tail and batch.complete_bytes == len(complete)
    assert len(batch.usage) == 1
    repaired = read(partial + b',"message":{"role":"assistant","usage":{"input":2}}}\n')
    assert len(repaired.usage) == 2 and not repaired.pending_tail


def test_malformed_middle_and_newline_terminated_tail_diagnose_safely() -> None:
    batch = read(source() + b'{PRIVATE SENTINEL}\n' + source(assistant({'input': 1})).split(b'\n', 1)[1] + b'{"broken":\n')
    assert len(batch.usage) == 1 and not batch.pending_tail
    assert [d.line for d in batch.diagnostics if d.code == 'malformed_line'] == [2, 4]
    assert 'PRIVATE SENTINEL' not in repr(batch)


def test_partial_zero_and_exact_decimals() -> None:
    record = read((FIXTURES / 'partial-zero.jsonl').read_bytes()).usage[0]
    assert record.tokens.total == Known(120)
    assert isinstance(record.tokens.buckets.cache_write, Unknown)
    assert isinstance(record.money, RecordedEstimate) and record.money.amount == Decimal(0)
    record = read(source(assistant({'cost': {'total': 0.1234567890123456}}))).usage[0]
    assert record.money.amount == Decimal('0.1234567890123456')


@pytest.mark.parametrize('bad', [-1, True, 1.5, '5', 2**63, float('nan'), float('inf')])
def test_invalid_count_preserves_independent_measures(bad: object) -> None:
    batch = read(source(assistant({'input': bad, 'output': 2, 'cacheRead': 0, 'cacheWrite': 0, 'totalTokens': 2})))
    assert batch.usage[0].tokens.buckets.input == Unknown('invalid_count')
    assert batch.usage[0].tokens.buckets.output == Known(2)
    assert any(d.code == 'invalid_count' and d.measure == 'input' for d in batch.diagnostics)


def test_subset_and_total_conflicts_are_per_measure() -> None:
    batch = read(source(assistant({'input': 10, 'output': 2, 'cacheRead': 0, 'cacheWrite': 1, 'totalTokens': 99, 'reasoning': 3, 'cacheWrite1h': 2})))
    record = batch.usage[0]
    assert record.tokens.buckets.input == Known(10)
    assert record.tokens.total == Unknown('total_conflict')
    assert record.tokens.reasoning == Unknown('invalid_subset')
    assert record.tokens.cache_write_1h == Unknown('invalid_subset')
    assert {d.measure for d in batch.diagnostics if d.code == 'invalid_subset'} == {'reasoning', 'cache_write_1h'}
    assert any(d.code == 'total_conflict' for d in batch.diagnostics)


def test_cost_component_mismatch_preserves_aggregate_and_invalid_component_is_omitted() -> None:
    batch = read(source(assistant({'cost': {'total': 0.4, 'input': 0.1, 'output': 0.1, 'cacheRead': 0, 'cacheWrite': 0}})))
    assert batch.usage[0].money.amount == Decimal('0.4')
    assert any(d.code == 'cost_component_mismatch' for d in batch.diagnostics)
    batch = read(source(assistant({'cost': {'total': 0, 'input': -1}})))
    assert batch.usage[0].money.amount == Decimal(0)
    assert batch.usage[0].money.components == ()


def test_unknown_shapes_missing_usage_duplicate_ids_and_bad_time_are_diagnosed() -> None:
    batch = read(source(assistant(None), {'type': 'custom', 'id': 'entry', 'usage': {'input': 22}}, {'type': 'compaction', 'id': 'aux', 'timestamp': 'bad', 'usage': {'input': 3}}))
    assert len(batch.usage) == 2
    assert isinstance(batch.usage[1].time, Undated)
    assert {'missing_usage', 'duplicate_entry_id', 'unsupported_usage_shape', 'invalid_timestamp'} <= {d.code for d in batch.diagnostics}


def test_duplicate_keys_and_unsupported_profile_reject() -> None:
    assert isinstance(read_pi(b'{"type":"session","version":3,"id":"a","id":"b"}\n', locator='x'), RejectedSource)
    assert isinstance(read_pi(source(), locator='x', profile='unknown'), RejectedSource)


@pytest.mark.parametrize('tail', [b'{"type":"message","message":{"usage":{"input":1e', b'{"name":"\xe2\x82'])
def test_incomplete_number_or_utf8_tail_is_retryable(tail: bytes) -> None:
    prefix = source()
    result = read(prefix + tail)
    assert result.pending_tail
    assert result.complete_bytes == len(prefix)


def test_invalid_counter_strings_never_retain_arbitrary_payloads() -> None:
    result = read(source(assistant({'input': 'PRIVATE SENTINEL', 'cost': {'total': 'PRIVATE SENTINEL'}})))
    assert 'PRIVATE SENTINEL' not in repr(result)
    assert result.usage[0].tokens.buckets.input == Unknown('invalid_count')


def test_second_header_is_reported_as_header_conflict() -> None:
    result = read(source({**HEADER, 'cwd': '/different'}))
    assert any(d.code == 'conflicting_header' for d in result.diagnostics)
    assert not result.entries


def test_every_partial_write_of_a_valid_record_is_retryable() -> None:
    prefix = source()
    record = b'{"type":"message","id":"next","message":{"role":"assistant","usage":{"input":-1,"cost":{"total":1.2e-3}},"content":"\\u20ac"}}'
    for end in range(1, len(record)):
        batch = read(prefix + record[:end])
        assert batch.pending_tail, record[:end]
        assert batch.complete_bytes == len(prefix)


def test_ordinary_tool_result_without_accounting_keeps_edge_and_lifetime_only() -> None:
    result = read(source({'type': 'message', 'id': 'tool', 'parentId': 'previous', 'timestamp': '2026-09-12T12:00:00Z', 'message': {'role': 'toolResult', 'toolCallId': 'call', 'content': 'PRIVATE SENTINEL'}}))
    assert result.usage == ()
    assert result.entries[0].parent_id == 'previous'
    assert result.session.last_observed == datetime(2026, 9, 12, 12, tzinfo=UTC)
    assert not result.diagnostics


def test_retained_record_constructors_reject_invalid_identity_time_and_variants() -> None:
    from dataclasses import replace
    from harness_usage.pi_reader import EntryIdentity, SessionMetadata
    from harness_usage.domain import SessionId
    with pytest.raises(ValueError):
        EntryIdentity(' ', None)
    with pytest.raises(ValueError):
        EntryIdentity('entry', None, True)
    with pytest.raises(ValueError):
        SessionMetadata(SessionId(''), 'native', None, None, None, None)
    with pytest.raises(ValueError):
        SessionMetadata(SessionId('pi:native'), 'native', datetime(2026, 9, 12), None, None, None)
    with pytest.raises(ValueError):
        SessionMetadata(SessionId('pi:native'), 'native', datetime(2026, 9, 12, 9, tzinfo=UTC), None, None, None, datetime(2026, 9, 12, 8, tzinfo=UTC))
    record = read(source(assistant({'input': 1}))).usage[0]
    with pytest.raises(ValueError):
        replace(record, kind='unknown')
    with pytest.raises(ValueError):
        replace(record, safe_facts=[('usage.input', 1)])
    with pytest.raises(ValueError):
        replace(record, safe_facts=(('content', 'PRIVATE SENTINEL'),))
def test_delegation_metadata_keeps_only_explicit_references():
    import json
    from harness_usage.pi_reader import read_delegations, read_pi
    header = {'type': 'session', 'version': 3, 'id': 'parent', 'cwd': '/repo'}
    entries = [
        {'type': 'message', 'id': 'spawn', 'message': {'role': 'toolResult', 'toolName': 'subagent', 'content': ['PRIVATE'], 'details': {'results': [{'sessionFile': '/child.jsonl', 'output': 'PRIVATE'}], 'workflow': {'value': {'results': [{'sessionFile': '/nested.jsonl'}]}}}}},
        {'type': 'custom', 'id': 'saved', 'customType': 'subagents:record', 'data': {'id': 'agent-uuid', 'sessionFile': '/saved-child.jsonl', 'originParentSessionFile': '/original-parent.jsonl', 'result': 'PRIVATE'}},
        {'type': 'message', 'id': 'agent', 'message': {'role': 'toolResult', 'toolName': 'Agent', 'details': {'agentId': 'other-agent'}}},
        {'type': 'message', 'id': 'unrelated', 'message': {'role': 'toolResult', 'toolName': 'read', 'details': {'results': [{'sessionFile': '/not-a-child'}]}}},
    ]
    data = ('\n'.join(json.dumps(row) for row in [header, *entries]) + '\n').encode()
    refs = read_delegations(data)
    assert {(r.entry_id, r.kind, r.value, r.owner_path) for r in refs} == {
        ('spawn', 'child_path', '/child.jsonl', None),
        ('spawn', 'child_path', '/nested.jsonl', None),
        ('saved', 'child_path', '/saved-child.jsonl', '/original-parent.jsonl'),
        ('saved', 'agent_id', 'agent-uuid', '/original-parent.jsonl'),
        ('agent', 'agent_id', 'other-agent', None),
    }
    assert 'PRIVATE' not in repr(refs)
    assert read_pi(data, locator='/parent.jsonl').delegations == refs
    assert read_delegations(b'bad header\n' + data) == ()


def test_delegation_metadata_rejects_invalid_owner_and_tolerates_partial_tail():
    from harness_usage.pi_reader import read_delegations
    invalid = {'type': 'custom', 'id': 'bad', 'customType': 'subagents:record',
               'data': {'id': 'agent', 'sessionFile': '/child', 'originParentSessionFile': []}}
    valid = {'type': 'custom', 'id': 'good', 'customType': 'subagents:record',
             'data': {'id': 'agent'}}
    data = source(invalid, valid) + b'{"type":"custom"'
    refs = read_delegations(data)
    assert len(refs) == 1 and refs[0].entry_id == 'good'
    assert read(data).delegations == refs


def test_canonical_subagent_path_hash_is_typed_and_format_scoped():
    from harness_usage.pi_reader import DelegationRef, read_delegations
    from harness_usage.domain import ContractViolation
    result = {'type': 'message', 'id': 'finish', 'message': {
        'role': 'toolResult', 'toolName': 'subagent', 'details': {'lifecycleStatus': {
            'processTerminal': {'canonicalSession': {'canonicalSessionId': 'a' * 64}}}}}}
    refs = read_delegations(source(result))
    assert refs == (DelegationRef('finish', 'child_path_hash', 'a' * 64),)
    result['message']['toolName'] = 'read'
    assert read_delegations(source(result)) == ()
    with pytest.raises(ContractViolation):
        DelegationRef('finish', 'child_path_hash', 'not-a-path-hash')


def test_control_notice_retains_exact_child_name_but_no_notice_content():
    from harness_usage.pi_reader import DelegationRef, read_delegations
    notice = {'type': 'custom_message', 'id': 'notice', 'customType': 'subagent_control_notice',
              'details': {'childIntercomTarget': 'subagent-worker-run', 'reason': 'PRIVATE'},
              'content': 'PRIVATE'}
    refs = read_delegations(source(notice))
    assert refs == (DelegationRef('notice', 'child_name', 'subagent-worker-run'),)
    assert read(source(notice)).delegations == refs
    notice['customType'] = 'unrelated'
    assert read_delegations(source(notice)) == ()


def test_workflow_graph_derives_only_run_bound_typed_child_names():
    from harness_usage.pi_reader import read_delegations
    result = {'type': 'message', 'id': 'launch', 'message': {'role': 'toolResult', 'toolName': 'subagent',
        'details': {'runId': 'Run-123', 'workflowGraph': {'runId': 'Run-123', 'nodes': [
            {'kind': 'parallel', 'children': [
                {'kind': 'agent', 'agent': ' Code Reviewer ', 'flatIndex': 0, 'task': 'PRIVATE'},
                {'kind': 'step', 'agent': 'Writer', 'flatIndex': 2},
                {'kind': 'agent', 'agent': 'Invalid', 'flatIndex': True}]},
            {'kind': 'other', 'agent': 'Invalid', 'flatIndex': 3}]}}}}
    refs = read_delegations(source(result))
    assert {r.value for r in refs} == {'subagent-code-reviewer-run-123-1', 'subagent-writer-run-123-3'}
    assert 'PRIVATE' not in repr(refs)
    assert read(source(result)).delegations == refs
    result['message']['details']['workflowGraph']['runId'] = 'different-run'
    assert read_delegations(source(result)) == ()


def test_result_child_name_uses_explicit_index_never_array_position():
    from harness_usage.pi_reader import read_delegations
    result = {'type': 'message', 'id': 'result', 'message': {'role': 'toolResult', 'toolName': 'subagent',
        'details': {'runId': 'run', 'results': [
            {'agent': 'Worker', 'index': 4}, {'agent': 'No-index'},
            {'agent': 'Invalid', 'index': -1}, {'agent': 'Invalid', 'index': True}]}}}
    assert [r.value for r in read_delegations(source(result))] == ['subagent-worker-run-5']


def test_async_single_launch_links_matching_tool_call_without_retaining_arguments():
    from harness_usage.pi_reader import read_delegations
    call = {'type': 'message', 'id': 'call', 'message': {'role': 'assistant', 'content': [
        {'type': 'toolCall', 'id': 'spawn-call', 'name': 'subagent',
         'arguments': {'agent': 'Reviewer', 'task': 'PRIVATE TASK'}}]}}
    result = {'type': 'message', 'id': 'result', 'message': {'role': 'toolResult', 'toolName': 'subagent',
        'toolCallId': 'spawn-call', 'details': {'mode': 'single', 'runId': 'run-123', 'asyncId': 'run-123', 'results': []}}}
    refs = read_delegations(source(call, result))
    assert [r.value for r in refs] == ['subagent-reviewer-run-123-1']
    assert read(source(call, result)).delegations == refs
    assert 'PRIVATE TASK' not in repr(refs)
    result['message']['toolCallId'] = 'different-call'
    assert read_delegations(source(call, result)) == ()


def test_conflicting_call_identity_after_result_does_not_guess_child_agent():
    from harness_usage.pi_reader import read_delegations
    def call(agent):
        return {'type': 'message', 'id': agent, 'message': {'role': 'assistant', 'content': [
            {'type': 'toolCall', 'id': 'same-call', 'name': 'subagent', 'arguments': {'agent': agent}}]}}
    result = {'type': 'message', 'id': 'result', 'message': {'role': 'toolResult', 'toolName': 'subagent',
        'toolCallId': 'same-call', 'details': {'mode': 'single', 'runId': 'run', 'asyncId': 'run'}}}
    assert read_delegations(source(call('one'), result, call('two'))) == ()
