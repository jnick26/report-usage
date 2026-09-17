"""Report reads preserve the full-snapshot oracle while projecting source evidence."""
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
import json
import re

import pytest

from harness_usage.application import Application
from harness_usage.domain import Assigned, Interval, ObservationId, ProjectId, SessionId, Unassigned
from harness_usage.reporting import AllTime, Coverage, DateRange, ReportQuery, SelectedContribution, build_report, parse_range
from harness_usage.storage import Storage, from_micros

FIXTURES = Path(__file__).parent / 'fixtures' / 'pi'
CASES = json.loads((FIXTURES / 'cases.json').read_text())['cases']


def import_intervals(storage, monkeypatch, intervals, *, model=None):
    # Pi does not record interval starts. Supply these report-domain cases before
    # insertion, so native foreign keys remain enabled throughout the test.
    from harness_usage import storage as storage_module
    original = storage_module.read_pi
    def with_intervals(*args, **kwargs):
        batch = original(*args, **kwargs)
        return replace(batch, usage=tuple(replace(record,
            time=Interval(*(from_micros(value) for value in intervals[record.kind])),
            model=model or record.model) if record.kind in intervals else record for record in batch.usage))
    with monkeypatch.context() as patch:
        patch.setattr(storage_module, 'read_pi', with_intervals)
        storage.import_source('/auxiliary', (FIXTURES / 'source' / 'auxiliary.jsonl').read_bytes())


def guard_read_connection(db, *, forbidden=()):
    execute = db.execute
    temporary = set()
    def guarded(sql, parameters=()):
        assert not any(re.search(r'\b' + table + r'\b', sql, re.I) for table in forbidden), sql
        if match := re.match(r'CREATE TEMP TABLE (\w+)', sql, re.I):
            temporary.add(match[1].lower())
        elif match := re.match(r'(?:INSERT INTO|UPDATE|DELETE FROM) (\w+)', sql, re.I):
            assert match[1].lower() in temporary, sql
        elif match := re.match(r'CREATE (?:UNIQUE )?INDEX \w+ ON (\w+)', sql, re.I):
            assert match[1].lower() in temporary, sql
        else:
            assert sql.lstrip().upper().startswith(('SELECT', 'WITH', 'BEGIN', 'EXPLAIN', 'DESCRIBE')), sql
        return execute(sql, parameters)
    db.execute = guarded


def reference_report(storage, query, **options):
    snapshot = storage.snapshot()
    sessions = {session.id: session for session in snapshot.sessions}
    contributions = []
    for observation in snapshot.observations:
        session = sessions[SessionId(observation.session_id)]
        record = observation.record
        contributions.append(SelectedContribution(ObservationId(observation.id), session.id, session.display_name, session.cwd,
            record.model, record.time, record.tokens, record.money, snapshot.attributions.get(session.id, Unassigned('not_resolved')),
            tuple(observation.decisions.items()), session.started, session.last_observed, observation.reasons + observation.diagnostics))
    report = build_report(contributions, revision=snapshot.revision, query=query, **options)
    # Global snapshot diagnostics are not part of scoped report coverage.
    return report


def projected_report(storage, query, **options):
    data = storage.report_input(query)
    report = build_report(data.contributions, revision=data.revision, query=query, **options)
    covered = {item.code for item in report.coverage}
    return replace(report, coverage=report.coverage + tuple(Coverage(code, ()) for code in data.diagnostics if code not in covered))


@pytest.mark.parametrize('case', CASES, ids=lambda case: case['name'])
def test_lean_report_matches_full_snapshot_for_every_fixture(tmp_path, case):
    storage = Storage(tmp_path / 'ledger.duckdb')
    for name in case['files']:
        storage.import_source('/fixtures/pi/source/' + name, (FIXTURES / 'source' / name).read_bytes())
    time_range = parse_range(*case['range_utc'], timezone='UTC') if 'range_utc' in case else AllTime()
    query = ReportQuery(None, time_range)
    assert projected_report(storage, query) == reference_report(storage, query)
    assert projected_report(storage, query, include_sessions=False) == reference_report(storage, query, include_sessions=False)


def test_project_and_time_projection_keeps_unknown_timing_and_lifetime(tmp_path):
    storage = Storage(tmp_path / 'ledger.duckdb')
    storage.import_source('/auxiliary', (FIXTURES / 'source' / 'auxiliary.jsonl').read_bytes())
    storage.import_source('/ordinary', (FIXTURES / 'source' / 'midnight.jsonl').read_bytes())
    snapshot = storage.snapshot()
    first = snapshot.sessions[0]
    storage.save_attributions({first.id: Assigned(ProjectId('directory:/project'), '/project', 'recorded_directory')})
    for project in (None, ProjectId('unassigned'), ProjectId('directory:/project'), ProjectId('directory:/absent')):
        for time_range in (AllTime(), parse_range('2026-09-12T21:00Z', '2026-09-13T21:00Z', 'UTC')):
            query = ReportQuery(project, time_range, 'hour')
            assert projected_report(storage, query) == reference_report(storage, query)
    query = ReportQuery(None, parse_range('2026-09-12T21:00Z', '2026-09-13T21:00Z', 'UTC'))
    inputs = storage.report_input(query)
    # One midnight point plus two unknown-start auxiliary records are relevant.
    assert len(inputs.contributions) == 3
    assert projected_report(storage, query, session_page=1, page_size=1) == reference_report(storage, query, session_page=1, page_size=1)


def test_known_interval_projection_retains_only_contained_and_overlapping_evidence(tmp_path, monkeypatch):
    storage = Storage(tmp_path / 'ledger.duckdb')
    import_intervals(storage, monkeypatch, {'compaction': (1789200000000000, 1789207200000000),
                                           'branch_summary': (1789113600000000, 1789120800000000)})
    for start, end in [('2026-09-12T07:00Z', '2026-09-12T11:00Z'), ('2026-09-12T09:30Z', '2026-09-12T11:00Z'), ('2026-09-13T00:00Z', '2026-09-14T00:00Z')]:
        query = ReportQuery(None, parse_range(start, end, 'UTC'))
        assert projected_report(storage, query) == reference_report(storage, query)


def test_application_report_uses_projection_and_forwards_paging(tmp_path, monkeypatch):
    application = Application(tmp_path / 'app')
    application.storage.import_source('/ordinary', (FIXTURES / 'source' / 'ordinary.jsonl').read_bytes())
    query = ReportQuery(None, AllTime())
    expected = reference_report(application.storage, query, include_sessions=False, catalog=application.catalog)
    def forbidden_snapshot():
        raise AssertionError('Report must not construct the full ledger snapshot')
    monkeypatch.setattr(application.storage, 'snapshot', forbidden_snapshot)
    assert application.report(query, include_sessions=False) == expected
    page = application.report(query, session_page=1, page_size=1)
    assert page.total_session_count == 1 and len(page.sessions) == 1
    assert application.report(query, session_id=page.sessions[0].id).sessions == page.sessions


def test_report_projection_is_a_single_readonly_ledger_snapshot(tmp_path, monkeypatch):
    from contextlib import contextmanager

    storage = Storage(tmp_path / 'ledger.duckdb')
    storage.import_source('/ordinary', (FIXTURES / 'source' / 'ordinary.jsonl').read_bytes())
    query = ReportQuery(None, AllTime())
    expected = reference_report(storage, query)
    writer = Storage(storage.path)
    committed = False

    original = storage.connect
    @contextmanager
    def read_only():
        with original() as db:
            guard_read_connection(db)
            def interleave(sql):
                nonlocal committed
                if 'FROM session_view s LEFT JOIN codex_title' in sql and not committed:
                    committed = True
                    writer.import_source('/midnight', (FIXTURES / 'source' / 'midnight.jsonl').read_bytes())
            db.set_trace_callback(interleave)
            yield db

    monkeypatch.setattr(storage, 'connect', read_only)
    assert projected_report(storage, query) == expected
    assert committed
    assert projected_report(storage, query).revision > expected.revision


def test_attribution_input_reads_metadata_without_accounting(tmp_path, monkeypatch):
    from contextlib import contextmanager
    storage=Storage(tmp_path/'ledger.duckdb')
    storage.import_source('/ordinary',(FIXTURES/'source'/'ordinary.jsonl').read_bytes())
    expected=storage.snapshot()
    original=storage.connect
    @contextmanager
    def metadata_only():
        with original() as db:
            guard_read_connection(db, forbidden=('observation','token_value','decision','appearance','recorded_estimate','diagnostic'))
            yield db
    monkeypatch.setattr(storage,'connect',metadata_only)
    sessions,attributions=storage.attribution_input()
    assert sessions==expected.sessions and attributions==expected.attributions


def test_empty_project_does_not_scan_accounting_tables(tmp_path, monkeypatch):
    from contextlib import contextmanager
    storage=Storage(tmp_path/'ledger.duckdb')
    rows=[json.loads(line) for line in (FIXTURES/'source'/'ordinary.jsonl').read_text().splitlines()]
    entries=[{**rows[1],'id':f'{i:08x}'} for i in range(500)]
    storage.import_source('/many', ('\n'.join(json.dumps(row) for row in [rows[0],*entries])+'\n').encode())
    original=storage.connect
    rows_scanned = []
    @contextmanager
    def bounded_read():
        with original() as db:
            def trace(sql):
                if sql.startswith(('SELECT v.* FROM relevant','SELECT p.* FROM relevant','SELECT d.observation_id,d.measure','SELECT o.id,o.session_id')):
                    rows_scanned.extend(json.loads(row[1])['cumulative_rows_scanned']
                                        for row in db.execute('EXPLAIN (FORMAT JSON, ANALYZE) ' + sql))
            db.set_trace_callback(trace)
            yield db
    monkeypatch.setattr(storage,'connect',bounded_read)
    assert storage.report_input(ReportQuery(ProjectId('directory:/absent'),AllTime())).contributions==()
    assert sum(rows_scanned) == 0


def test_report_decision_and_observation_reads_need_no_sort(tmp_path, monkeypatch):
    from contextlib import contextmanager
    storage = Storage(tmp_path / 'ledger.duckdb')
    storage.import_source('/ordinary', (FIXTURES / 'source' / 'ordinary.jsonl').read_bytes())
    original = storage.connect
    plans = []
    @contextmanager
    def explain_reads():
        with original() as db:
            def trace(sql):
                if sql.startswith(('SELECT d.observation_id,d.measure', 'SELECT o.id,o.session_id')):
                    plans.extend(row[1] for row in db.execute('EXPLAIN ' + sql))
            db.set_trace_callback(trace)
            yield db
    monkeypatch.setattr(storage, 'connect', explain_reads)
    assert projected_report(storage, ReportQuery(None, AllTime())) == reference_report(storage, ReportQuery(None, AllTime()))
    assert plans and not any('ORDER_BY' in plan for plan in plans)


def test_existing_ledger_gets_eligibility_index_without_changing_evidence(tmp_path):
    path = tmp_path / 'ledger.duckdb'
    storage = Storage(path)
    storage.import_source('/ordinary', (FIXTURES / 'source' / 'ordinary.jsonl').read_bytes())
    before = storage.snapshot()
    with storage.connect(write=True) as db:
        db.execute('DROP INDEX IF EXISTS decision_nonexcluded')
    storage = Storage(path)
    assert storage.snapshot() == before
    with storage.connect() as db:
        indexes = db.execute("SELECT index_name FROM duckdb_indexes() WHERE table_name='decision'").fetchall()
    assert ('decision_nonexcluded',) in indexes
