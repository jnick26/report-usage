from decimal import Decimal
import json

import pytest

from harness_usage.application import Application
from harness_usage.pi_transcript import parse_pi_transcript
from harness_usage.pricing import Catalog
from harness_usage.transcript import Message, TranscriptUnavailable


def assistant(identity, parent, input_count=100, output_count=50, *, model='model-a'):
    return {'type': 'message', 'id': identity, 'parentId': parent, 'timestamp': '2026-09-17T00:00:01Z',
            'message': {'role': 'assistant', 'provider': 'test', 'model': model, 'stopReason': 'stop',
                        'content': [{'type': 'text', 'text': identity}],
                        'usage': {'input': input_count, 'output': output_count, 'cacheRead': 0, 'cacheWrite': 0}}}


def encoded(*entries):
    header = {'type': 'session', 'version': 3, 'id': 'cost-session', 'cwd': '/fixture', 'timestamp': '2026-09-17T00:00:00Z'}
    user = {'type': 'message', 'id': 'user', 'parentId': None, 'message': {'role': 'user', 'content': 'Synthetic request'}}
    return b''.join(json.dumps(row).encode() + b'\n' for row in (header, user, *entries))


def application(tmp_path, payload, *, costs=None):
    root = tmp_path / 'sessions'
    root.mkdir()
    path = root / 'session.jsonl'
    path.write_bytes(payload)
    app = Application(tmp_path / 'data')
    app.set_roots((str(root),))
    app.catalog = Catalog.from_bytes(json.dumps({'test': {'models': {'model-a': {'cost': costs or {
        'input': 2, 'output': 4, 'cache_read': 1, 'cache_write': 3}}}}}).encode(), snapshot_date='test', sha256='test')
    app.storage.import_source(str(path), payload)
    return app, path


def test_original_assistant_identity_is_distinct_from_checkpoint_replay():
    first = assistant('a', 'user')
    payload = encoded(first, {'type': 'compaction', 'id': 'compact', 'parentId': 'a',
        'timestamp': '2026-09-17T00:00:02Z', 'summary': 'Synthetic checkpoint', 'retainedTail': [first['message']]})
    entries = [entry for entry in parse_pi_transcript(payload, 'pi:cost-session').entries if isinstance(entry, Message)]
    assert [getattr(entry, 'usage_id', None) for entry in entries] == [None, 'a', None]


def test_two_original_requests_use_exact_prices_and_running_amount(tmp_path):
    app, _ = application(tmp_path, encoded(assistant('a', 'user'), assistant('b', 'a', 50, 100)))
    try:
        page = app.transcript('pi:cost-session')
        assert [(point.amount, point.cumulative, point.incomplete, point.cumulative_incomplete) for point in page.costs] == [
            (Decimal('0.0004'), Decimal('0.0004'), False, False),
            (Decimal('0.0005'), Decimal('0.0009'), False, False)]
        assert [point.message_id for point in page.costs] == [entry.id for entry in page.transcript.entries if isinstance(entry, Message) and entry.role == 'assistant']
        assert [(line.category, line.tokens, line.rate) for line in page.costs[0].calculations] == [('input', 100, Decimal(2)), ('output', 50, Decimal(4))]
    finally:
        app.close()


def test_branch_excludes_sibling_and_unavailable_cost_makes_total_sticky(tmp_path):
    payload = encoded(assistant('a', 'user'), assistant('b', 'a', 50, 100),
                      assistant('unknown', 'a', model='missing-model'), assistant('c', 'unknown', 50, 100))
    app, _ = application(tmp_path, payload)
    try:
        selected = app.transcript('pi:cost-session', 'b')
        assert [point.amount for point in selected.costs] == [Decimal('0.0004'), Decimal('0.0005')]
        latest = app.transcript('pi:cost-session')
        assert [point.amount for point in latest.costs] == [Decimal('0.0004'), None, Decimal('0.0005')]
        assert [point.cumulative for point in latest.costs] == [Decimal('0.0004'), Decimal('0.0004'), Decimal('0.0009')]
        assert [point.cumulative_incomplete for point in latest.costs] == [False, True, True]
        assert {line.reason for line in latest.costs[1].calculations} == {'model_not_in_catalog'}
    finally:
        app.close()


def test_changed_bytes_are_unavailable_until_the_exact_snapshot_is_imported(tmp_path):
    payload = encoded(assistant('a', 'user'))
    app, path = application(tmp_path, payload)
    try:
        changed = payload.replace(b'"text": "a"', b'"text": "changed"')
        assert changed != payload
        path.write_bytes(changed)
        point = app.transcript('pi:cost-session').costs[0]
        assert (point.amount, point.cumulative, point.incomplete) == (None, Decimal(0), True)
        app.storage.import_source(str(path), changed)
        assert app.transcript('pi:cost-session').costs[0].amount == Decimal('0.0004')
        # A rejected current generation cannot fall back to old priced bytes.
        app.storage.import_source(str(path), b'not a session')
        with pytest.raises(TranscriptUnavailable):
            app.transcript('pi:cost-session')
    finally:
        app.close()


def test_unimported_source_cannot_join_another_files_native_entry(tmp_path):
    from harness_usage.transcript_costs import pi_transcript_costs
    payload = encoded(assistant('a', 'user'))
    app, path = application(tmp_path, payload)
    try:
        points = pi_transcript_costs(app.storage, parse_pi_transcript(payload, 'pi:cost-session'),
                                     str(path.with_name('unimported.jsonl')), payload, app.catalog)
        assert len(points) == 1
        assert points[0].amount is None and points[0].cumulative_incomplete
    finally:
        app.close()


def test_identical_copies_count_once_but_conflicting_selected_evidence_does_not_price(tmp_path):
    payload = encoded(assistant('a', 'user'))
    app, path = application(tmp_path, payload)
    try:
        copied = path.with_name('copy.jsonl')
        copied.write_bytes(payload)
        app.storage.import_source(str(copied), payload)
        page = app.transcript('pi:cost-session')
        assert len(page.costs) == 1 and page.costs[0].amount == Decimal('0.0004')
        conflict = encoded(assistant('a', 'user', 101, 51))
        copied.write_bytes(conflict)
        app.storage.import_source(str(copied), conflict)
        point = app.transcript('pi:cost-session').costs[0]
        assert point.amount is None and point.incomplete
        assert {line.reason for line in point.calculations} == {'unresolved_tokens'}
    finally:
        app.close()


def test_tool_only_assistant_has_one_cost_and_checkpoint_replay_has_none(tmp_path):
    first = assistant('a', 'user')
    first['message']['content'] = [{'type': 'toolCall', 'id': 'call', 'name': 'read', 'arguments': {}}]
    result = {'type': 'message', 'id': 'tool', 'parentId': 'a', 'message': {
        'role': 'toolResult', 'toolCallId': 'call', 'toolName': 'read', 'content': 'Synthetic output', 'isError': False}}
    checkpoint = {'type': 'compaction', 'id': 'compact', 'parentId': 'tool',
        'timestamp': '2026-09-17T00:00:02Z', 'summary': 'Synthetic checkpoint', 'retainedTail': [first['message']]}
    app, _ = application(tmp_path, encoded(first, result, checkpoint))
    try:
        page = app.transcript('pi:cost-session')
        assert len(page.costs) == 1 and page.costs[0].amount == Decimal('0.0004')
        request = next(entry for entry in page.transcript.entries if isinstance(entry, Message) and entry.usage_id)
        assert page.costs[0].message_id == request.id
        assert not any(entry.usage_id for entry in page.transcript.entries if isinstance(entry, Message) and entry.phase == 'retained checkpoint')
    finally:
        app.close()


@pytest.mark.parametrize('shape,amount', [('missing', None), ('invalid-output', Decimal('0.0002')), ('zero', Decimal(0))])
def test_missing_invalid_and_zero_usage_are_distinct(tmp_path, shape, amount):
    first = assistant('a', 'user')
    if shape == 'missing':
        first['message'].pop('usage')
    elif shape == 'invalid-output':
        first['message']['usage']['output'] = True
    else:
        first = assistant('a', 'user', 0, 0)
    app, _ = application(tmp_path, encoded(first))
    try:
        point = app.transcript('pi:cost-session').costs[0]
        assert point.amount == amount
        assert point.incomplete == (shape != 'zero')
        assert point.cumulative_incomplete == point.incomplete
    finally:
        app.close()


def test_partial_cache_price_keeps_known_subtotal_and_calculation_reason(tmp_path):
    first = assistant('a', 'user')
    first['message']['usage']['cacheRead'] = 20
    app, _ = application(tmp_path, encoded(first), costs={'input': 2, 'output': 4})
    try:
        point = app.transcript('pi:cost-session').costs[0]
        assert point.amount == Decimal('0.0004')
        assert point.incomplete and point.cumulative_incomplete
        cache = next(line for line in point.calculations if line.category == 'cache_read')
        assert (cache.tokens, cache.subtotal, cache.reason) == (20, None, 'missing_category_rate')
    finally:
        app.close()


def test_cost_projection_never_reconstructs_the_full_ledger(tmp_path, monkeypatch):
    app, _ = application(tmp_path, encoded(assistant('a', 'user')))
    def forbidden(*args, **kwargs):
        raise AssertionError('whole-ledger report/snapshot is forbidden for one transcript')
    monkeypatch.setattr(app.storage, 'snapshot', forbidden)
    monkeypatch.setattr(app.storage, 'report_input', forbidden)
    try:
        assert app.transcript('pi:cost-session').costs[0].amount == Decimal('0.0004')
    finally:
        app.close()
