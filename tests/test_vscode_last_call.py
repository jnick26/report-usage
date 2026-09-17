"""Exact retained last-successful-call profile; no recursive token extraction."""
from copy import deepcopy
from datetime import UTC, datetime, timedelta
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from harness_usage.application import Application
from harness_usage.copilot_vscode_reader import CopilotVscodeReadBatch, PROFILE, read_copilot_vscode
from harness_usage.domain import Known, ModelIdentity, Unknown
from harness_usage.reporting import AllTime, DateRange, ReportQuery, build_report
from harness_usage.transcript_access import read_transcript_page
from harness_usage.web import bucket_tooltip, create_app
from test_claude_source import _frozen_import
from test_copilot_vscode_source import canonical_locator, operation_log


def state(output=7, prompt=31, *, profile='metadata'):
    return {'version': 3, 'sessionId': 'last-call', 'creationDate': 1789540000000,
            'requests': [{'requestId': 'request', 'responseId': 'response',
                          'timestamp': 1789540001000, 'responseTimestamp': 1789540002000,
                          'agent': {'extensionId': {'value': 'github.copilot-chat'}},
                          'message': {'text': 'Synthetic request'},
                          'response': [{'kind': 'markdownContent', 'content': {'value': 'Synthetic answer'}}],
                          'modelId': 'selected-is-not-actual',
                          'result': ({'metadata': {'promptTokens': prompt, 'outputTokens': output}}
                                     if profile == 'metadata' else
                                     {'usage': {'promptTokens': prompt, 'completionTokens': output}})}]}


def payload(value, suffix='.json'):
    return operation_log(value) if suffix == '.jsonl' else json.dumps(value).encode()


def exact_report(app, query=None):
    query = query or ReportQuery(None, AllTime())
    data = app.storage.report_input(query)
    report = app.report(query)
    assert report == build_report(data.contributions, revision=data.revision, query=query, catalog=app.catalog)
    return report


@pytest.mark.parametrize('suffix', ('.json', '.jsonl'))
@pytest.mark.parametrize('output', (0, 7, 2**63 - 1))
@pytest.mark.parametrize('profile', ('metadata', 'usage'))
def test_last_call_reader_retains_raw_prompt_and_bounded_output_without_route_or_total(tmp_path, suffix, output, profile):
    result = read_copilot_vscode(payload(state(output, profile=profile), suffix), locator=canonical_locator(tmp_path, suffix=suffix))
    assert isinstance(result, CopilotVscodeReadBatch)
    record, = result.usage
    evidence, = result.evidence
    assert record.tokens.buckets.output == Known(output)
    assert isinstance(record.tokens.buckets.input, Unknown)
    assert isinstance(record.tokens.buckets.cache_read, Unknown)
    assert isinstance(record.tokens.buckets.cache_write, Unknown)
    assert isinstance(record.tokens.total, Unknown)
    assert record.model == ModelIdentity(None, None)
    assert record.money.reason == 'selected_model_unpriced'
    assert (evidence.raw_input, evidence.raw_output, evidence.output_lower_bound) == (31, output, True)
    assert record.quantities[0].state == 'unknown'


@pytest.mark.parametrize('metadata', (
    {'promptTokens': 3}, {'outputTokens': 4}, {'promptTokens': True, 'outputTokens': 4},
    {'promptTokens': 3, 'outputTokens': -1}, {'promptTokens': 3, 'outputTokens': 2**63},
    {'promptTokens': '3', 'outputTokens': 4}, {'promptTokens': 3, 'outputTokens': 4.0},
    {'promptTokens': 3, 'completionTokens': 4}, {'nested': {'promptTokens': 3, 'outputTokens': 4}},
))
def test_invalid_or_unqualified_metadata_remains_unknown(tmp_path, metadata):
    value = state()
    value['requests'][0]['result']['metadata'] = metadata
    result = read_copilot_vscode(payload(value), locator=canonical_locator(tmp_path))
    assert isinstance(result, CopilotVscodeReadBatch)
    assert all(isinstance(row.tokens.buckets.output, Unknown) for row in result.usage)


def test_foreign_participant_does_not_gain_last_call_usage(tmp_path):
    value = state()
    value['requests'][0]['agent']['extensionId']['value'] = 'another.extension'
    result = read_copilot_vscode(payload(value), locator=canonical_locator(tmp_path))
    assert isinstance(result, CopilotVscodeReadBatch) and result.usage == ()


@pytest.mark.parametrize('with_metadata', (False, True))
@pytest.mark.parametrize('output', (0, 7))
def test_exact_legacy_usage_is_one_last_call_lower_bound(tmp_path, with_metadata, output):
    value = state()
    value['requests'][0]['result'] = {'usage': {'promptTokens': 31, 'completionTokens': output}}
    if with_metadata:
        value['requests'][0]['result']['metadata'] = {'promptTokens': 31, 'outputTokens': output}
    result = read_copilot_vscode(payload(value), locator=canonical_locator(tmp_path))
    assert isinstance(result, CopilotVscodeReadBatch)
    record, = result.usage
    assert record.tokens.buckets.output == Known(output)
    assert isinstance(record.tokens.buckets.input, Unknown)
    assert record.model == ModelIdentity(None, None)
    assert result.evidence[0].raw_input == 31
    assert result.evidence[0].output_lower_bound


@pytest.mark.parametrize('legacy', (None, {}, {'promptTokens': 31}, {'completionTokens': 7},
    {'promptTokens': True, 'completionTokens': 7}, {'promptTokens': 31, 'completionTokens': 7.0},
    {'promptTokens': -1, 'completionTokens': 7}, {'promptTokens': 31, 'completionTokens': 2**63}))
def test_explicit_invalid_legacy_usage_blocks_metadata_fallback(tmp_path, legacy):
    value = state()
    value['requests'][0]['result']['usage'] = legacy
    result = read_copilot_vscode(payload(value), locator=canonical_locator(tmp_path))
    assert isinstance(result, CopilotVscodeReadBatch)
    assert all(isinstance(record.tokens.buckets.output, Unknown) for record in result.usage)
    assert 'invalid_retained_usage' in {diagnostic.code for diagnostic in result.diagnostics}
    assert all(item.state == 'unresolved' for item in result.evidence)


def test_valid_legacy_usage_survives_invalid_lower_priority_metadata(tmp_path):
    value = state(prompt=True)
    value['requests'][0]['result']['usage'] = {'promptTokens': 31, 'completionTokens': 7}
    result = read_copilot_vscode(payload(value), locator=canonical_locator(tmp_path))
    assert isinstance(result, CopilotVscodeReadBatch)
    assert result.usage[0].tokens.buckets.output == Known(7)
    assert result.evidence[0].state == 'usable'


def test_unequal_valid_nested_pairs_retain_conflict_and_never_add_or_select_maximum(tmp_path):
    app = Application(tmp_path / 'data', timezone='UTC')
    value = state(output=9, prompt=30)
    value['requests'][0]['result']['usage'] = {'promptTokens': 31, 'completionTokens': 7}
    app.storage.import_source(canonical_locator(tmp_path), payload(value))
    report = exact_report(app)
    assert (report.tokens.output.known, report.tokens.output.lower_bound_observations) == (0, 0)
    assert 'unresolved_evidence' in {entry.code for entry in report.coverage}
    observation, = app.storage.snapshot().observations
    assert observation.record.tokens.buckets.output == Known(7)
    assert 'conflicting_retained_usage' in observation.reasons
    # Conflicting lower-priority counters stay distinct without changing native identity.
    duplicate = deepcopy(value['requests'][0])
    duplicate['result']['metadata']['outputTokens'] = 10
    value['requests'].append(duplicate)
    app.storage.import_source(canonical_locator(tmp_path), payload(value))
    with app.storage.connect() as db:
        assert db.execute('SELECT count(*),count(DISTINCT native_entry_id) FROM observation').one() == (2, 1)
    assert exact_report(app).tokens.output.lower_bound_observations == 0
    app.close()


@pytest.mark.parametrize('core', (
    {'promptTokens': 15, 'completionTokens': 20}, {'completionTokens': 0},
    {'promptTokens': -1}, {'completionTokens': None}, {'modelTotals': []}, {'modelTotals': None},
    {'modelTotals': [{'model': 'actual', 'inputTokens': 15, 'cachedTokens': 2, 'outputTokens': 20}]},
))
@pytest.mark.parametrize('profile', ('metadata', 'usage'))
def test_core_fields_block_nested_fallback_including_invalid_core(tmp_path, core, profile):
    value = state(profile=profile)
    value['requests'][0].update(core)
    result = read_copilot_vscode(payload(value), locator=canonical_locator(tmp_path))
    without = deepcopy(value)
    without['requests'][0].pop('result')
    expected = read_copilot_vscode(payload(without), locator=canonical_locator(tmp_path))
    assert isinstance(result, CopilotVscodeReadBatch) and isinstance(expected, CopilotVscodeReadBatch)
    assert result.usage == expected.usage
    assert result.evidence == expected.evidence


@pytest.mark.parametrize('output', (0, 7, 10001))
@pytest.mark.parametrize('profile', ('metadata', 'usage'))
def test_lower_bound_report_ui_and_bucket_tooltips_preserve_exact_zero_and_small_counts(tmp_path, output, profile):
    app = Application(tmp_path / 'data', timezone='UTC')
    app.set_roots((str(tmp_path),))
    app.storage.import_source(canonical_locator(tmp_path), payload(state(output, profile=profile)))
    report = exact_report(app)
    assert (report.tokens.output.known, report.tokens.output.lower_bound_observations) == (output, 1)
    assert report.tokens.input.known == 0 and report.tokens.total.known == 0
    for row in (*report.projects, *report.sessions, *report.models, *report.buckets):
        assert (row.tokens.output.known, row.tokens.output.lower_bound_observations) == (output, 1)
    assert report.models[0].model == ModelIdentity(None, None)
    assert report.money.known == 0 and report.money.missing_observations == 1
    assert 'lower_bound' in {entry.code for entry in report.coverage}
    number = f'{output // 1_000_000}.{output % 1_000_000:06d}'.rstrip('0').rstrip('.')
    assert f'Output ≥ {number} Mtok' in bucket_tooltip(report.buckets[0])
    client = TestClient(create_app(app), base_url='http://localhost')
    for url in ('/', '/?project=unassigned'):
        for headers in ({}, {'X-Up-Target': '#main'}):
            response = client.get(url, headers=headers)
            assert response.status_code == 200
            assert f'title="At least {output:,} tokens"' in response.text
            assert f'≥ {number}</span>' in response.text
            assert '≥ &lt;' not in response.text and '≥ <' not in response.text
    # Zero lower bounds survive mixing with an entirely unknown output row.
    value = state()
    value['sessionId'] = 'unknown-output'
    value['requests'][0].pop('result')
    app.storage.import_source(canonical_locator(tmp_path, workspace='unknown'), payload(value))
    mixed = exact_report(app)
    assert mixed.tokens.output.unknown_observations == 1
    assert mixed.tokens.output.lower_bound_observations == 1
    assert f'title="At least {output:,} tokens"' in client.get('/').text
    app.close()


@pytest.mark.parametrize('suffix', ('.json', '.jsonl'))
@pytest.mark.parametrize('profile', ('metadata', 'usage'))
def test_last_call_reprojects_real_retained_shape1_and_preserves_native_identity(tmp_path, suffix, profile):
    path = tmp_path / 'data/ledger.duckdb'
    locator = Path(canonical_locator(tmp_path, suffix=suffix))
    data = payload(state(profile=profile), suffix)
    _frozen_import('final-accepted', path, ((locator, data),))
    app = Application(path.parent, timezone='UTC')
    before = exact_report(app)
    assert before.tokens.output.known == 0
    app.storage.import_source(str(locator), data)
    current = exact_report(app)
    assert (current.tokens.output.known, current.tokens.output.lower_bound_observations) == (7, 1)
    with app.storage.connect() as db:
        assert tuple(db.execute('SELECT generation,profile FROM source_generation ORDER BY generation')) == (
            (0, 'vscode-chat-v3/copilot-shape-1'), (1, PROFILE))
        assert db.execute('SELECT schema_version FROM ledger_meta').one()[0] == 6
    revision = app.storage.import_source(str(locator), data)
    assert revision == current.revision
    assert read_transcript_page(app.storage, (str(tmp_path),), current.sessions[0].id)
    app.close()
    app = Application(path.parent, timezone='UTC')
    assert exact_report(app) == current
    # Changing counters keeps the same native evidence identity.
    with app.storage.connect() as db:
        native = db.execute('SELECT native_entry_id FROM observation WHERE model IS NULL ORDER BY rowid DESC LIMIT 1').one()[0]
    app.storage.import_source(str(locator), payload(state(9, profile=profile), suffix))
    with app.storage.connect() as db:
        assert db.execute('SELECT native_entry_id FROM observation ORDER BY rowid DESC LIMIT 1').one()[0] == native
    assert exact_report(app).tokens.output.known == 9
    app.close()


@pytest.mark.parametrize('profile', ('metadata', 'usage'))
def test_last_call_alternates_conflicts_generations_missing_rejection_and_repair(tmp_path, profile):
    app = Application(tmp_path / 'data', timezone='UTC')
    flat, log = (canonical_locator(tmp_path, suffix=suffix) for suffix in ('.json', '.jsonl'))
    initial = state(profile=profile)
    app.storage.import_sources(((flat, payload(initial)), (log, payload(initial, '.jsonl'))))
    first = exact_report(app)
    assert (first.tokens.output.known, first.tokens.output.lower_bound_observations) == (7, 1)
    app.storage.import_source(log, payload(state(8, profile=profile), '.jsonl'))
    conflict = exact_report(app)
    assert (conflict.tokens.output.known, conflict.tokens.output.lower_bound_observations) == (0, 0)
    assert 'unresolved_evidence' in {entry.code for entry in conflict.coverage}
    # Differing physical copies stay quarantined even if one has core usage.
    core = state(profile=profile)
    core['requests'][0]['completionTokens'] = 10
    app.storage.import_source(log, payload(core, '.jsonl'))
    assert exact_report(app).tokens.output.known == 0
    app.storage.import_sources(((flat, payload(core)), (log, payload(core, '.jsonl'))))
    exact = exact_report(app)
    assert (exact.tokens.output.known, exact.tokens.output.lower_bound_observations) == (10, 0)
    app.storage.import_sources(((flat, payload(initial)), (log, payload(initial, '.jsonl'))))
    app.storage.mark_missing((flat, log))
    missing = exact_report(app)
    assert (missing.tokens.output.known, missing.tokens.output.lower_bound_observations) == (7, 1)
    assert 'saved_history' in {entry.code for entry in missing.coverage}
    for locator in (flat, log):
        app.storage.import_source(locator, b'{malformed}\n')
    rejected = exact_report(app)
    assert (rejected.tokens.output.known, rejected.tokens.output.lower_bound_observations) == (0, 0)
    app.close()
    app = Application(tmp_path / 'data', timezone='UTC')
    assert exact_report(app) == rejected
    app.storage.import_source(flat, payload(initial))
    repaired = exact_report(app)
    assert (repaired.tokens.output.known, repaired.tokens.output.lower_bound_observations) == (7, 1)
    assert 'unresolved_evidence' not in {entry.code for entry in repaired.coverage}
    app.close()


@pytest.mark.parametrize('profile', ('metadata', 'usage'))
def test_last_call_operation_provenance_and_scoped_report_parity(tmp_path, profile):
    app = Application(tmp_path / 'data', timezone='UTC')
    value = state(0, profile=profile)
    field = 'outputTokens' if profile == 'metadata' else 'completionTokens'
    data = operation_log(value) + json.dumps({'kind': 1, 'k': ['requests', 0, 'result', profile, field], 'v': 7}).encode() + b'\n'
    locator = canonical_locator(tmp_path, suffix='.jsonl')
    result = read_copilot_vscode(data, locator=locator)
    assert isinstance(result, CopilotVscodeReadBatch) and result.evidence[0].line == 2
    app.storage.import_source(locator, data)
    instant = datetime.fromtimestamp(1789540002, UTC)
    for query in (ReportQuery('unassigned', AllTime()),
                  ReportQuery(None, DateRange(instant - timedelta(hours=1), instant + timedelta(hours=1), 'UTC'))):
        assert exact_report(app, query).tokens.output.lower_bound_observations == 1
    app.close()


def test_paired_shape1_copies_reproject_once_and_raw_prompt_conflicts_stay_unresolved(tmp_path):
    ledger = tmp_path / 'data/ledger.duckdb'
    flat, log = (Path(canonical_locator(tmp_path, suffix=suffix)) for suffix in ('.json', '.jsonl'))
    sources = ((flat, payload(state())), (log, payload(state(), '.jsonl')))
    _frozen_import('final-accepted', ledger, sources)
    app = Application(ledger.parent, timezone='UTC')
    app.storage.import_sources(tuple((str(locator), data) for locator, data in sources))
    report = exact_report(app)
    assert (report.tokens.output.known, report.tokens.output.lower_bound_observations) == (7, 1)
    app.storage.import_sources(tuple((str(locator), data) for locator, data in sources))
    assert exact_report(app) == report
    # Ambiguous prompt is not counted as fresh input, but still participates in compatibility.
    app.storage.import_source(str(log), payload(state(prompt=32), '.jsonl'))
    conflict = exact_report(app)
    assert (conflict.tokens.output.known, conflict.tokens.output.lower_bound_observations) == (0, 0)
    app.close()


def test_last_call_incompatible_duplicates_keep_stable_identity_and_unresolved_bounds(tmp_path):
    app = Application(tmp_path / 'data', timezone='UTC')
    value = state()
    value['requests'].extend(state(prompt=32)['requests'])
    app.storage.import_source(canonical_locator(tmp_path), payload(value))
    report = exact_report(app)
    assert (report.tokens.output.known, report.tokens.output.lower_bound_observations) == (0, 0)
    with app.storage.connect() as db:
        assert db.execute('SELECT count(*),count(DISTINCT native_entry_id) FROM observation').one() == (2, 1)
    app.close()


def test_last_call_saved_snapshot_retains_explicit_output_finality(tmp_path):
    app = Application(tmp_path / 'data', timezone='UTC')
    app.storage.import_source(canonical_locator(tmp_path), payload(state()))
    observation, = app.storage.snapshot().observations
    assert observation.record.tokens.buckets.output == Known(7)
    assert dict(observation.record.safe_facts)['usage.outputFinality'] == 'last_successful_call'
    app.close()
