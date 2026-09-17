"""Independent fixture arithmetic exercised through durable import and reports."""
from decimal import Decimal
import json
from pathlib import Path
from typing import Any

import pytest

from harness_usage.domain import ObservationId, SessionId, Unassigned
from harness_usage.reporting import AllTime, Report, ReportQuery, SelectedContribution, build_report, parse_range
from harness_usage.storage import Storage

FIXTURES = Path(__file__).parent / 'fixtures' / 'pi'
CASES: list[dict[str, Any]] = json.loads((FIXTURES / 'cases.json').read_text())['cases']


def fixture_bytes(name: str) -> bytes:
    return (FIXTURES / 'source' / name).read_bytes()


def fixture_locator(name: str) -> str:
    # This is the absolute parentSession namespace recorded in the fixtures.
    return '/fixtures/pi/source/' + name


def persisted_report(storage: Storage, case: dict[str, Any]) -> Report:
    snapshot = storage.snapshot()
    sessions = {session.id: session for session in snapshot.sessions}
    contributions = []
    for item in snapshot.observations:
        session = sessions[SessionId(item.session_id)]
        record = item.record
        contributions.append(SelectedContribution(
            ObservationId(item.id), session.id, session.display_name, session.cwd,
            record.model, record.time, record.tokens, record.money,
            Unassigned('synthetic_directory_unavailable'), tuple(item.decisions.items()),
            session.started, session.last_observed, item.reasons,
        ))
    time_range = parse_range(*case['range_utc'], timezone='UTC') if 'range_utc' in case else AllTime()
    return build_report(contributions, revision=snapshot.revision, query=ReportQuery(None, time_range))


def assert_oracle(report: Report, case: dict[str, Any]) -> None:
    for measure, expected in case['expected'].items():
        if measure == 'recorded_usd':
            assert report.money.known == Decimal(expected)
            assert sum((row.money.known for row in report.sessions), Decimal(0)) == Decimal(expected)
            assert sum((row.money.known for row in report.projects), Decimal(0)) == Decimal(expected)
            assert sum((row.money.known for row in report.models), Decimal(0)) == Decimal(expected)
            continue
        metric = getattr(report.tokens, measure)
        if expected is None:
            assert metric.known == 0
            assert metric.unknown_observations > 0
            continue
        assert metric.known == expected, measure
        assert sum(getattr(row.tokens, measure).known for row in report.sessions) == expected
        assert sum(getattr(row.tokens, measure).known for row in report.projects) == expected
        assert sum(getattr(row.tokens, measure).known for row in report.models) == expected
        assert sum(getattr(bucket.tokens, measure).known for bucket in report.buckets) + getattr(report.unbucketed, measure).known == expected
    for measure in case.get('unknown', []):
        if measure in ('provider', 'model'):
            assert any(getattr(row.model, measure) is None for row in report.models)
        else:
            assert getattr(report.tokens, measure).unknown_observations > 0
    for measure in case.get('observed_zero', []):
        metric = getattr(report.tokens, measure)
        assert metric.known == metric.unknown_observations == 0
    if 'unbucketed_total' in case:
        assert report.unbucketed.total.known == case['unbucketed_total']
    if 'display_name' in case:
        assert report.sessions[0].name == case['display_name']
    if case.get('unresolved'):
        assert report.tokens.total.unknown_observations >= len(case['unresolved'])
        assert 'unresolved_evidence' in {item.code for item in report.coverage}
    if case.get('unknown_model_entries'):
        unknown = [row for row in report.models if row.model.model is None]
        assert len(unknown) == 1
        assert unknown[0].tokens.total.known == 45


@pytest.mark.parametrize('case', CASES, ids=lambda case: case['name'])
def test_fixture_oracle_after_import_repeat_and_restart(tmp_path: Path, case: dict[str, Any]) -> None:
    for index, order in enumerate(case.get('import_orders', [case['files']])):
        database = tmp_path / str(index) / 'ledger.sqlite3'
        storage = Storage(database)
        if case['name'] == 'tail_completed':
            locator = fixture_locator('tail-before.jsonl')
            storage.import_source(locator, fixture_bytes('tail-before.jsonl'))
            storage.import_source(locator, fixture_bytes('tail-after.jsonl'))
            stable = storage.snapshot().revision
            assert storage.import_source(locator, fixture_bytes('tail-after.jsonl')) == stable
        else:
            for name in order:
                revision = storage.import_source(fixture_locator(name), fixture_bytes(name))
                assert storage.import_source(fixture_locator(name), fixture_bytes(name)) == revision
        assert_oracle(persisted_report(storage, case), case)
        restarted = Storage(database)
        assert persisted_report(restarted, case) == persisted_report(storage, case)
        if case.get('diagnostic'):
            assert case['diagnostic'] in restarted.snapshot().diagnostics
        if 'pending_tail' in case:
            with restarted.connect() as connection:
                pending = connection.execute('SELECT pending_tail FROM source_generation ORDER BY generation DESC LIMIT 1').fetchone()[0]
            assert bool(pending) == case['pending_tail']
        if case['name'] == 'retention_and_repeat':
            original = fixture_locator('ordinary.jsonl')
            relocated = '/fixtures/pi/archive/ordinary.jsonl'
            restarted.import_source(relocated, fixture_bytes('ordinary.jsonl'))
            restarted.mark_missing((original,))
            assert_oracle(persisted_report(restarted, case), case)
            restarted.mark_missing((relocated,))
            assert_oracle(persisted_report(Storage(database), case), case)
            assert 'saved_history' in restarted.snapshot().diagnostics


def test_reports_remain_one_revision_during_concurrent_commits(tmp_path: Path) -> None:
    from concurrent.futures import ThreadPoolExecutor

    database = tmp_path / 'concurrent.sqlite3'
    storage = Storage(database)
    ordinary = fixture_bytes('ordinary.jsonl')
    native_id = json.loads(ordinary.splitlines()[0])['id']

    def import_distinct_sessions() -> None:
        for index in range(20):
            data = ordinary.replace(native_id.encode(), f'snapshot-{index}'.encode())
            storage.import_source(f'/fixtures/concurrent/{index}.jsonl', data)

    case: dict[str, Any] = {}
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(import_distinct_sessions)
        while not future.done():
            report = persisted_report(storage, case)
            assert report.tokens.total.known == report.revision * 970
            assert report.money.known == Decimal('0.01') * report.revision
        future.result()
    final = persisted_report(storage, case)
    assert final.revision == 20
    assert final.tokens.total.known == 19_400


def test_sanitized_real_session_matches_independent_source_arithmetic(tmp_path: Path) -> None:
    sample = FIXTURES / 'real-sanitized'
    expected = json.loads((sample / 'expected.json').read_text())
    storage = Storage(tmp_path / 'real.sqlite3')
    data = (sample / 'session.jsonl').read_bytes()
    storage.import_source('/fixtures/real-sanitized/session.jsonl', data)
    report = persisted_report(storage, {})
    names = {'input':'input', 'output':'output', 'cache_read':'cacheRead', 'cache_write':'cacheWrite', 'total':'totalTokens'}
    for measure, native in names.items():
        assert getattr(report.tokens, measure).known == expected['assistant'][native] + expected['auxiliary'][native]
    assert report.money.known == Decimal(expected['assistant_recorded_usd']) + Decimal(expected['auxiliary_recorded_usd'])
    assert report.unbucketed.total.known == expected['auxiliary']['totalTokens']
    snapshot = storage.snapshot()
    source_records = [json.loads(line) for line in data.splitlines()]
    assert sum(record.get('message', {}).get('role') == 'toolResult' for record in source_records) == expected['counts']['tool_result_records']
    assert sum(item.record.kind == 'assistant' for item in snapshot.observations) == expected['counts']['assistant']
    assert sum(item.record.kind == 'tool_result' for item in snapshot.observations) == expected['counts']['tool_result_usage_observations']
    assert all(all(state == 'unresolved' for state in item.decisions.values()) for item in snapshot.observations if item.record.kind == 'tool_result')
    assert report.tokens.total.unknown_observations == 0
    assert 'unresolved_evidence' not in {item.code for item in report.coverage}
    revision = storage.import_source('/fixtures/real-sanitized/session.jsonl', data)
    assert revision == report.revision
    assert persisted_report(Storage(storage.path), {}) == report
    # The retained fixture has accounting and lineage metadata only.
    forbidden = {'content', 'summary', 'details', 'arguments', 'errorMessage'}
    def check_keys(value: object) -> None:
        if isinstance(value, dict):
            assert not forbidden.intersection(value)
            for nested in value.values():
                check_keys(nested)
        elif isinstance(value, list):
            for nested in value:
                check_keys(nested)
    for line in data.splitlines():
        check_keys(json.loads(line))
