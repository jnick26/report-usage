"""Aggregate reads preserve report values without transferring recorded estimates."""
from contextlib import contextmanager
from dataclasses import replace
from decimal import Decimal
import json
from pathlib import Path
import re

import pytest

from harness_usage.aggregate_reporting import build_aggregate_report
from harness_usage.application import Application
from harness_usage.domain import Assigned, ModelIdentity, ProjectId, SessionId
from harness_usage.pricing import Catalog
from harness_usage.reporting import AllTime, Coverage, ReportQuery, build_report, parse_range
from harness_usage.storage import Storage
from harness_usage.source_input import SourcePayload
from test_codex_storage import codex_source, legacy, modern, usage
from test_delegation_storage import reference as delegation, source
from test_report_storage import guard_read_connection, import_intervals

FIXTURES = Path(__file__).parent / 'fixtures'
PI = FIXTURES / 'pi/source'
CASES = json.loads((FIXTURES / 'pi/cases.json').read_text())['cases']


def catalog():
    cost = {'input': 1, 'output': 2, 'cache_read': 0.1, 'cache_write': 0.2,
            'tiers': [{'tier': {'type': 'context', 'size': 950},
                       'input': 3, 'output': 4, 'cache_read': 0.3, 'cache_write': 0.6}]}
    models = {'fixture-provider': ['fixture-model'], 'test': ['test'],
              'openai': ['gpt-5', 'gpt-5.6-sol']}
    return Catalog.from_bytes(json.dumps({provider: {'models': {model: {'cost': cost} for model in names}}
                                         for provider, names in models.items()}).encode(),
                              snapshot_date='2026-09-12', sha256='fixture')


def visible(report):
    # Dashboard values exclude coverage; detailed evidence remains in the ledger.
    return replace(report, coverage=(),
                   sessions=tuple(replace(session, coverage=()) for session in report.sessions))


@pytest.mark.parametrize(('provider', 'name'), [
    ('claude-bridge', 'claude-sonnet-4-6'),
    (None, 'claude-sonnet-4-6'), ('github-copilot', 'claude-sonnet-4-6'),
    (None, 'claude-sonnet-4.6'), ('github-copilot', 'claude-sonnet-4.6'),
])
@pytest.mark.parametrize('kind', ['point', 'aggregate', 'unknown_context'])
def test_official_pricing_fallback_has_identical_sql_response_tiers_and_identity(tmp_path, provider, kind, name):
    cost = {'input': 1, 'output': 2, 'tiers': [
        {'tier': {'type': 'context', 'size': 200}, 'input': 3, 'output': 4}]}
    rates = Catalog.from_bytes(json.dumps({'anthropic': {'models': {'claude-sonnet-4-6': {'cost': cost}}}}).encode(),
                               snapshot_date='fixture', sha256='fixture')
    rows = [{'type': 'session', 'version': 3, 'id': 'pricing-fallback', 'timestamp': '2026-09-12T08:00:00Z'}]
    for index, input_count in enumerate((150, 201)):
        values = {'input': input_count, 'output': 20, 'cacheRead': 0, 'cacheWrite': 0}
        if kind == 'unknown_context':
            values.pop('input')
        body = {'role': 'assistant', 'provider': provider, 'model': name, 'usage': values, 'stopReason': 'stop'}
        entry = {'id': str(index), 'timestamp': '2026-09-12T09:00:00Z'}
        rows.append(dict(entry, type='compaction', **body) if kind == 'aggregate'
                    else dict(entry, type='message', message=body))
    storage = Storage(tmp_path / 'pricing.duckdb')
    storage.import_source('/synthetic-pricing.jsonl', ('\n'.join(map(json.dumps, rows)) + '\n').encode())
    query = ReportQuery(None, AllTime())
    for store in (storage, Storage(storage.path)):
        data = store.report_input(query)
        ordinary = build_report(data.contributions, revision=data.revision, query=query, catalog=rates)
        actual = build_aggregate_report(store, query, rates)
        assert actual == ordinary
        assert actual.money.known == Decimal('0.000873' if kind == 'point' else '0')
        assert actual.money.missing_observations == (0 if kind == 'point' else 2)
        assert {line.model for line in actual.money.calculations} == {ModelIdentity(provider, name)}
        assert {line.source_provider for line in actual.money.calculations} == {'anthropic'}
        if kind == 'point':
            assert {line.context_threshold for line in actual.money.calculations} == {None, 200}
        if provider is None:
            assert 'unknown_model_identity' in {entry.code for entry in actual.coverage}
        assert build_aggregate_report(store, query, rates, include_sessions=False).money == actual.money
        assert build_aggregate_report(store, query, rates, session_page=1).money == actual.money


def test_repaired_same_locator_diagnostics_do_not_survive_reopen(tmp_path):
    locator = str(tmp_path / '11111111-1111-4111-8111-111111111111.jsonl')
    fixture = (FIXTURES / 'claude/versioned.jsonl').read_bytes()
    records = [json.loads(line) for line in fixture.splitlines()]
    for record in records:
        record.pop('cwd', None)
    storage = Storage(tmp_path / 'repaired.duckdb')
    query = ReportQuery(None, AllTime())
    def exact(store):
        data = store.report_input(query)
        actual = build_aggregate_report(store, query, catalog())
        assert actual == build_report(data.contributions, revision=data.revision, query=query, catalog=catalog())
        return actual
    storage.import_source(locator, ('\n'.join(map(json.dumps, records)) + '\n').encode())
    assert 'cwd_attribution_unavailable' in {entry.code for entry in exact(storage).coverage}
    storage.import_source(locator, fixture)
    repaired = exact(storage)
    assert 'cwd_attribution_unavailable' not in {entry.code for entry in repaired.coverage}
    assert exact(Storage(storage.path)) == repaired
    clean = Storage(tmp_path / 'clean.duckdb')
    clean.import_source(locator, fixture)
    assert replace(repaired, revision=exact(clean).revision) == exact(clean)


@pytest.mark.parametrize('injection', ['diagnostic', 'reason'])
def test_saved_history_is_derived_from_availability_not_raw_codes(tmp_path, injection):
    storage = Storage(tmp_path / 'raw-code.duckdb')
    storage.import_source('/ordinary', (PI / 'ordinary.jsonl').read_bytes())
    with storage.connect(write=True) as db:
        if injection == 'diagnostic':
            db.execute("INSERT INTO diagnostic(source_id,code) SELECT id,'saved_history' FROM source_generation")
        else:
            db.execute("UPDATE decision SET state='unresolved',owner_session=NULL,reason='saved_history' WHERE measure='output'")
    query = ReportQuery(None, AllTime())
    for store in (storage, Storage(storage.path)):
        data = store.report_input(query)
        ordinary = build_report(data.contributions, revision=data.revision, query=query, catalog=catalog())
        aggregate = build_aggregate_report(store, query, catalog())
        assert aggregate == ordinary
        assert 'saved_history' not in data.diagnostics
        assert 'saved_history' not in {entry.code for entry in ordinary.coverage}
    storage.mark_missing(('/ordinary',))
    data = storage.report_input(query)
    saved = build_report(data.contributions, revision=data.revision, query=query, catalog=catalog())
    assert saved == build_aggregate_report(storage, query, catalog())
    assert 'saved_history' in {entry.code for entry in saved.coverage}


def test_repaired_model_diagnostic_uses_same_current_generation_in_both_paths(tmp_path):
    storage = Storage(tmp_path / 'model-repair.duckdb')
    fixture = (PI / 'ordinary.jsonl').read_bytes()
    storage.import_source('/ordinary', fixture)
    with storage.connect(write=True) as db:
        db.execute("INSERT INTO diagnostic(source_id,code) SELECT id,'codex_model_conflict' FROM source_generation")
    query = ReportQuery(None, AllTime())
    assert build_aggregate_report(storage, query, catalog()).models[0].model == ModelIdentity(None, None)
    storage.import_source('/ordinary', fixture + b'\n')
    data = storage.report_input(query)
    expected = build_report(data.contributions, revision=data.revision, query=query, catalog=catalog())
    assert expected.models[0].model == ModelIdentity('fixture-provider', 'fixture-model')
    assert build_aggregate_report(storage, query, catalog()) == expected


def test_scoped_mixed_source_coverage_matches_without_erasing_coverage(tmp_path):
    storage = Storage(tmp_path / 'coverage.duckdb')
    storage.import_source(str(tmp_path / 'User/workspaceStorage/one/chatSessions/session.jsonl'),
                          (FIXTURES / 'copilot_vscode/session-v3.jsonl').read_bytes())
    storage.import_source(str(tmp_path / 'session-state/11111111-1111-4111-8111-111111111111/events.jsonl'),
                          (FIXTURES / 'copilot_cli/current/events.jsonl').read_bytes())
    query = ReportQuery(None, AllTime())
    data = storage.report_input(query)
    ordinary = build_report(data.contributions, revision=data.revision, query=query, catalog=catalog())
    actual = build_aggregate_report(storage, query, catalog())
    assert actual == ordinary
    codes = [entry.code for entry in actual.coverage]
    assert codes.count('copilot_cross_source_join_unavailable') == 1
    assert 'lossy_turn_summary' in codes
    assert build_aggregate_report(storage, query, catalog(), session_page=2, page_size=1).coverage == actual.coverage
    assert build_aggregate_report(storage, query, catalog(), include_sessions=False).coverage == actual.coverage
    for scoped in (ReportQuery(ProjectId('absent'), AllTime()),
                   ReportQuery(None, parse_range('2026-09-16T06:00Z', '2026-09-16T07:00Z', 'UTC'))):
        data = storage.report_input(scoped)
        limited = build_aggregate_report(storage, scoped, catalog())
        assert limited == build_report(data.contributions, revision=data.revision, query=scoped, catalog=catalog())
        assert 'copilot_cross_source_join_unavailable' not in {entry.code for entry in limited.coverage}


@pytest.mark.parametrize(('path', 'changed'), [(('modelTotals', 0, 'inputTokens'), 99), (('copilotCredits',), 0.5)])
def test_vscode_conflict_copies_repair_and_private_metadata_preserve_reporting(tmp_path, path, changed):
    from test_copilot_vscode_source import duplicate_compatibility_requests, canonical_locator
    request, requests = duplicate_compatibility_requests(path, changed, False)
    storage = Storage(tmp_path / 'conflict.duckdb')
    locator = canonical_locator(tmp_path)
    payload = {'version': 3, 'sessionId': 'same-native', 'requests': requests}
    storage.import_source(locator, json.dumps(payload).encode())
    query = ReportQuery(None, AllTime())
    def exact():
        data = storage.report_input(query)
        report = build_aggregate_report(storage, query, catalog())
        assert report == build_report(data.contributions, revision=data.revision, query=query, catalog=catalog())
        assert 'copilot_vscode_presence_v1' not in repr(report)
        return report
    conflicted = exact()
    assert 'unresolved_evidence' in {entry.code for entry in conflicted.coverage}
    log = str(Path(locator).with_suffix('.jsonl'))
    storage.import_source(log, json.dumps({'kind': 0, 'v': payload}).encode() + b'\n')
    assert replace(exact(), revision=conflicted.revision) == conflicted
    repaired = dict(payload, requests=[request])
    storage.import_sources(((locator, json.dumps(repaired).encode()),
                            (log, json.dumps({'kind': 0, 'v': repaired}).encode() + b'\n')))
    repaired_report = exact()
    assert 'unresolved_evidence' not in {entry.code for entry in repaired_report.coverage}
    assert repaired_report.quantities[0].known == Decimal('0.25')
    assert repaired_report.quantities[0].known_observations == 1
    storage = Storage(storage.path)
    storage.import_sources(((locator, json.dumps(repaired).encode()),
                            (log, json.dumps({'kind': 0, 'v': repaired}).encode() + b'\n')))
    assert exact() == repaired_report


def oracle(storage, query, rates, **options):
    data = storage.report_input(query)
    report = build_report(data.contributions, revision=data.revision, query=query, catalog=rates, **options)
    covered = {entry.code for entry in report.coverage}
    return visible(replace(report, coverage=report.coverage + tuple(
        Coverage(code, ()) for code in data.diagnostics if code not in covered)))


def assert_matches(storage, query, rates, **options):
    expected = oracle(storage, query, rates, **options)
    actual = build_aggregate_report(storage, query, rates, **options)
    assert visible(actual) == expected
    return actual


@pytest.mark.parametrize('case', CASES, ids=lambda case: case['name'])
def test_aggregate_preserves_pi_fixture_values(tmp_path, case):
    storage = Storage(tmp_path / 'ledger.duckdb')
    for name in case['files']:
        storage.import_source('/fixtures/pi/source/' + name, (PI / name).read_bytes())
    rates = catalog()
    ranges = [AllTime(), parse_range(*case.get('range_utc', ['2026-09-12T00:00Z', '2026-09-13T00:00Z']), 'UTC')]
    for time_range in ranges:
        query = ReportQuery(None, time_range, 'hour')
        assert_matches(storage, query, rates)
        assert_matches(storage, query, rates, include_sessions=False)


def test_aggregate_preserves_mixed_harness_models_titles_and_saved_history(tmp_path):
    storage = Storage(tmp_path / 'ledger.duckdb')
    storage.import_source('/pi', (PI / 'explicit-name.jsonl').read_bytes())
    storage.import_source('/codex', (FIXTURES / 'codex/mixed.jsonl').read_bytes())
    storage.save_codex_titles({'codex:fixture-child': 'Native Codex title'})
    storage.mark_missing(('/pi',))
    query = ReportQuery(None, AllTime(), 'five_minutes')
    report = assert_matches(storage, query, catalog())
    assert {row.harness for row in report.models} == {'pi', 'codex'}
    assert {row.name for row in report.sessions} == {'Synthetic session name', 'Native Codex title'}
    assert 'saved_history' in {entry.code for entry in report.coverage}
    assert 'saved_history' in storage.report_input(query).diagnostics
    # The legacy prefix has an unavailable ancestor; only the modern 120 is selected.
    assert report.tokens.total.known == 1090 and report.tokens.total.unknown_observations == 1


def test_aggregate_report_includes_claude_partial_unknown_usage_without_pricing(tmp_path):
    identity = '11111111-1111-4111-8111-111111111111'
    data = (FIXTURES / 'claude/main.jsonl').read_bytes()
    storage = Storage(tmp_path / 'ledger.duckdb')
    storage.import_source(str(tmp_path / f'{identity}.jsonl'), data)
    report = assert_matches(storage, ReportQuery(None, AllTime()), catalog())
    assert report.tokens.input.known == 10
    assert report.tokens.output.unknown_observations == 2
    assert report.money.missing_observations == 2
    assert {row.harness for row in report.models} == {'claude'}


def test_claude_output_correction_aggregate_matches_ordinary_known_output(tmp_path):
    identity = '11111111-1111-4111-8111-111111111111'
    data = (FIXTURES / 'claude/versioned.jsonl').read_bytes()
    storage = Storage(tmp_path / 'ledger.duckdb')
    storage.import_source(str(tmp_path / f'{identity}.jsonl'), data)
    report = assert_matches(storage, ReportQuery(None, AllTime()), catalog())
    assert report.tokens.input.known == 15
    assert report.tokens.output.known == 37
    assert report.tokens.output.unknown_observations == 1
    assert report.tokens.total.known == 54
    assert report.tokens.total.unknown_observations == 2
    assert report.money.missing_observations == 3
    assert {row.harness for row in report.models} == {'claude'}


def test_aggregate_report_keeps_vscode_partial_tokens_and_ai_credits_separate(tmp_path):
    fixture = FIXTURES / 'copilot_vscode/session-v3.jsonl'
    locator = tmp_path / 'User/workspaceStorage/one/chatSessions/session.jsonl'
    storage = Storage(tmp_path / 'ledger.duckdb')
    storage.import_sources((SourcePayload(str(locator), fixture.read_bytes()),))
    report = assert_matches(storage, ReportQuery(None, AllTime()), catalog())
    assert report.tokens.input.known == 8
    assert report.tokens.input.unknown_observations == 2
    assert report.tokens.output.known == 15
    assert report.tokens.cache_read.known == 5
    assert report.money.known == 0 and report.money.missing_observations == 3
    assert report.quantities[0].harness == 'copilot-vscode'
    assert report.quantities[0].measure == 'ai_credits'
    assert report.quantities[0].known == Decimal('0.5')
    assert report.quantities[0].known_observations == 1
    assert report.quantities[0].lower_bound_observations == 0
    assert {row.harness for row in report.models} == {'copilot-vscode'}


def test_cli_report_keeps_tokens_quantities_unknowns_and_provider_pricing_separate(tmp_path):
    fixture = FIXTURES / 'copilot_cli/current/events.jsonl'
    locator = tmp_path / 'session-state/11111111-1111-4111-8111-111111111111/events.jsonl'
    storage = Storage(tmp_path / 'ledger.duckdb')
    storage.import_source(str(locator), fixture.read_bytes())
    report = assert_matches(storage, ReportQuery(None, AllTime()), catalog())
    assert report.tokens.input.known == 150 and report.tokens.input.unknown_observations == 1
    assert report.tokens.output.known == 65
    assert report.tokens.cache_read.known == 60 and report.tokens.cache_write.known == 10
    assert report.money.known == 0 and report.money.missing_observations == 2
    quantities = {row.measure: row for row in report.quantities}
    assert quantities['nano_aiu'].known == Decimal('200')
    assert quantities['premium_requests'].known == Decimal('4')
    assert quantities['request_count'].known == Decimal('5')
    assert {row.harness for row in report.models} == {'copilot-cli'}


def test_family_project_filters_cost_pages_and_detail_keep_the_same_population(tmp_path):
    storage = Storage(tmp_path / 'ledger.duckdb')
    storage.import_sources([('/main', source('main', 100, name='Main task')),
                            ('/child', source('child', 400)), ('/other', source('other', 300)),
                            ('/unpriced', source('unpriced').replace(b'"model": "test"', b'"model": "missing"'))])
    delegation(storage, '/main', '/child')
    owner = Assigned(ProjectId('directory:/main-project'), '/main-project', 'recorded_directory')
    child_owner = Assigned(ProjectId('directory:/child-project'), '/child-project', 'recorded_directory')
    storage.save_attributions({'pi:main': owner, 'pi:child': child_owner, 'pi:other': owner})
    rates = catalog()
    for project in (None, owner.project_id, child_owner.project_id, ProjectId('unassigned'), ProjectId('directory:/absent')):
        query = ReportQuery(project, AllTime(), 'hour')
        for sort in ('started', 'cost_desc', 'cost_asc'):
            for page in (1, 2, 99):
                assert_matches(storage, query, rates, session_sort=sort, session_page=page, page_size=1)
        for identity in ('pi:main', 'pi:child', 'pi:missing'):
            assert_matches(storage, query, rates, session_id=SessionId(identity))
    report = assert_matches(storage, ReportQuery(None, AllTime()), rates, session_sort='cost_desc', session_page=1, page_size=1)
    assert report.total_session_count == 3
    assert report.sessions[0].id == 'pi:main' and report.sessions[0].subagent_count == 1
    assert report.sessions[0].tokens.total.known == 500


def test_codex_families_replay_conflicts_and_range_only_descendants(tmp_path):
    storage = Storage(tmp_path / 'ledger.duckdb')
    first, second = usage(10), usage(5, 50)
    cumulative = {name: first[name] + second[name] for name in first}
    storage.import_sources([
        ('/root', codex_source('root', legacy(first, first))),
        ('/child', codex_source('child', legacy(first, first), legacy(second, cumulative, '2026-09-13T10:00:00Z'), parent='root')),
        ('/conflict-a', codex_source('conflict', modern(thread='conflict'))),
        ('/conflict-b', codex_source('conflict', modern(thread='conflict', output=11)).replace(b'gpt-5', b'gpt-6')),
    ])
    rates = catalog()
    assert_matches(storage, ReportQuery(None, AllTime()), rates)
    query = ReportQuery(None, parse_range('2026-09-13T00:00Z', '2026-09-14T00:00Z', 'UTC'), 'hour')
    report = assert_matches(storage, query, rates)
    assert report.total_session_count == 1 and report.sessions[0].id == 'codex:root'
    assert report.sessions[0].subagent_count == 1
    assert report.sessions[0].started.day == 12
    assert report.tokens.output.known == 5


def test_context_tiers_are_chosen_per_response_before_summing_tokens(tmp_path):
    storage = Storage(tmp_path / 'ledger.duckdb')
    header, first = [json.loads(line) for line in (PI / 'ordinary.jsonl').read_text().splitlines()]
    second = json.loads(json.dumps(first))
    second['id'] = 'second'
    second['timestamp'] = '2026-09-12T10:00:00Z'
    second['message']['usage'].update(input=101, totalTokens=971)
    storage.import_source('/tiers', ('\n'.join(map(json.dumps, [header, first, second])) + '\n').encode())
    report = assert_matches(storage, ReportQuery(None, AllTime(), 'hour'), catalog())
    assert report.money.known == Decimal('0.000883')
    assert {line.context_threshold for line in report.money.calculations} == {None, 950}
    assert sum(bucket.money.known for bucket in report.buckets) == report.money.known


def test_unknown_and_not_applicable_tokens_and_interval_prices_are_preserved(tmp_path, monkeypatch):
    storage = Storage(tmp_path / 'ledger.duckdb')
    import_intervals(storage, monkeypatch, {'compaction': (1789200000000000, 1789207200000000)},
                     model=ModelIdentity('fixture-provider', 'fixture-model'))
    storage.import_source('/partial', (PI / 'partial-zero.jsonl').read_bytes())
    with storage.connect(write=True) as db:
        db.execute("UPDATE token_value SET state='not_applicable',amount=NULL,reason='not_supported' WHERE measure='input' AND observation_id IN (SELECT id FROM observation WHERE kind='branch_summary')")
    for time_range in (AllTime(), parse_range('2026-09-12T08:00Z', '2026-09-12T10:00Z', 'UTC'),
                       parse_range('2026-09-12T09:00Z', '2026-09-12T11:00Z', 'UTC')):
        assert_matches(storage, ReportQuery(None, time_range, 'hour'), catalog())
    report = assert_matches(storage, ReportQuery(None, AllTime()), catalog())
    assert report.tokens.input.not_applicable_observations == 1
    assert report.tokens.cache_write.unknown_observations == 1
    assert 'aggregate_context' in {line.reason for line in report.money.calculations}


def test_aggregate_application_report_never_reads_recorded_estimates_or_writes_ledger(tmp_path, monkeypatch):
    application = Application(tmp_path / 'app', timezone='UTC')
    application.catalog = catalog()
    storage = application.storage
    storage.import_source('/ordinary', (PI / 'ordinary.jsonl').read_bytes())
    query = ReportQuery(None, AllTime())
    expected = oracle(storage, query, application.catalog, include_sessions=False)
    before = storage.snapshot()
    original = storage.connect

    @contextmanager
    def readonly_without_estimates():
        with original() as db:
            guard_read_connection(db, forbidden=('recorded_estimate',))
            yield db

    with monkeypatch.context() as patch:
        patch.setattr(storage, 'connect', readonly_without_estimates)
        assert visible(build_aggregate_report(storage, query, application.catalog, include_sessions=False)) == expected
        assert visible(application.report(query, include_sessions=False)) == expected
    assert storage.snapshot() == before


def test_family_name_matching_ignores_transcript_excerpts(tmp_path):
    storage = Storage(tmp_path / 'ledger.duckdb')
    storage.import_sources([('/main', source('main')), ('/child', source('child', name='Worker name')),
                            ('/excerpt', source('excerpt'))])
    with storage.connect(write=True) as db:
        db.execute("UPDATE session SET title_excerpt='Worker name' WHERE id='pi:excerpt'")
    delegation(storage, '/main', 'Worker name', kind='child_name')
    query = ReportQuery(None, AllTime())
    expected = oracle(storage, query, catalog())
    assert expected.total_session_count == 2
    assert next(row for row in expected.sessions if row.id == 'pi:main').subagent_count == 1
    assert_matches(storage, query, catalog())


def test_exact_aggregate_tokens_and_prices_can_exceed_sqlite_int64(tmp_path):
    storage = Storage(tmp_path / 'ledger.duckdb')
    header, first = [json.loads(line) for line in source('huge', 2**62 + 1).splitlines()]
    second = dict(first, id='second-usage')
    storage.import_source('/huge', ('\n'.join(map(json.dumps, [header, first, second])) + '\n').encode())
    report = assert_matches(storage, ReportQuery(None, AllTime(), 'hour'), catalog())
    assert report.tokens.input.known == report.tokens.total.known == 9223372036854775810
    assert report.money.known == Decimal('27670116110564.327430')
    assert report.sessions[0].tokens == report.tokens
    assert report.buckets[0].money == report.money


def test_kyiv_day_buckets_preserve_usage_across_daylight_saving_transition(tmp_path):
    storage = Storage(tmp_path / 'ledger.duckdb')
    storage.import_sources([
        ('/before', source('before', 10).replace(b'2026-09-12', b'2026-03-28')),
        ('/transition', source('transition', 20).replace(b'2026-09-12', b'2026-03-29')),
        ('/after', source('after', 30).replace(b'2026-09-12', b'2026-03-30')),
    ])
    query = ReportQuery(None, parse_range('2026-03-28T00:00', '2026-03-31T00:00', 'Europe/Kyiv'), 'day')
    report = assert_matches(storage, query, catalog())
    assert [(bucket.end - bucket.start).total_seconds() / 3600 for bucket in report.buckets] == [24, 23, 24]
    assert [bucket.tokens.total.known for bucket in report.buckets] == [10, 20, 30]
    assert all([(bucket.start, bucket.end) for bucket in session.buckets] ==
               [(bucket.start, bucket.end) for bucket in report.buckets] for session in report.sessions)


@pytest.mark.parametrize('check', ['diagnostics', 'materialization'])
def test_dashboard_skips_unused_diagnostics_and_intermediate_observation_tables(tmp_path, monkeypatch, check):
    application = Application(tmp_path / 'app', timezone='UTC')
    application.catalog = catalog()
    storage = application.storage
    storage.import_source('/ordinary', (PI / 'ordinary.jsonl').read_bytes())
    storage.mark_missing(('/ordinary',))
    original_connect = storage.connect
    statements = []

    @contextmanager
    def guarded_connect():
        with original_connect() as db:
            def trace(sql):
                statements.append(sql)
                if check == 'materialization':
                    assert not re.match(r'CREATE TEMP TABLE (?:values_by_observation|classified)\b', sql, re.I), sql
            db.set_trace_callback(trace)
            yield db

    monkeypatch.setattr(storage, 'connect', guarded_connect)
    report = application.report(ReportQuery(None, AllTime()))
    if check == 'diagnostics':
        diagnostic_reads = [sql for sql in statements if re.search(r'(?:from|join) diagnostic\b', sql, re.I)]
        assert diagnostic_reads
        assert all("d.code='codex_model_conflict'" in sql or ('FROM base r JOIN appearance' in sql and 'd.code IN (' in sql) for sql in diagnostic_reads)
    assert 'saved_history' in {entry.code for entry in report.coverage}
    assert all('saved_history' in {entry.code for entry in session.coverage} for session in report.sessions)
    assert report.tokens.total.known == 970
