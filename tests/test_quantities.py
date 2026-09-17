from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from harness_usage.aggregate_reporting import build_aggregate_report
from harness_usage.domain import (
    ContractViolation, Interval, Known, MeasuredQuantity, MissingEstimate, ModelIdentity,
    Point, SessionId, TokenBreakdown, TokenEvidence, Undated, Unknown, NotApplicable,
)
from harness_usage.pi_reader import EntryIdentity, UsageRecord
from harness_usage.pricing import Catalog
from harness_usage.reporting import AllTime, QuantityRow, ReportQuery, build_report
from harness_usage.reporting import quantity_rows
from harness_usage.reporting import DateRange
from harness_usage.storage import Storage


def test_quantity_decisions_preserve_every_state_and_exact_decimal():
    members = [
        (MeasuredQuantity('ai_credits', 'known', Decimal('0'), None, False, 'fixture'), 'selected'),
        (MeasuredQuantity('ai_credits', 'known', Decimal('1.25'), None, True, 'fixture'), 'selected'),
        (MeasuredQuantity('ai_credits', 'unknown', None, 'private', False, None), 'selected'),
        (MeasuredQuantity('ai_credits', 'known', Decimal('9'), None, False, 'fixture'), 'unresolved'),
        (MeasuredQuantity('ai_credits', 'not_applicable', None, 'private', False, None), 'selected'),
        (MeasuredQuantity('ai_credits', 'known', Decimal('100'), None, False, 'fixture'), 'excluded'),
    ]
    result = quantity_rows(('copilot-vscode', value, state) for value, state in members)
    assert result[0].unknown_observations == 1
    assert result[0].unresolved_observations == 1
    assert result[0].not_applicable_observations == 1
    assert (result[0].known, result[0].known_observations, result[0].lower_bound_observations) == (Decimal('1.25'), 2, 1)
    assert quantity_rows(('copilot-vscode', value, state) for value, state in reversed(members)) == result
    assert quantity_rows(('copilot-vscode', value, 'excluded') for value, _ in members) == ()
    exact = quantity_rows(('copilot-vscode', MeasuredQuantity('ai_credits', 'known', Decimal(value), None, False, 'fixture'), 'selected')
                          for value in ('0.12345678901234567890123456781', '0.00000000000000000000000000001'))
    assert exact[0].known == Decimal('0.12345678901234567890123456782')


def test_quantity_scope_states_and_full_coverage_parity(tmp_path):
    store = Storage(tmp_path / 'matrix.duckdb')
    rates = Catalog({}, '2026-09-16', 'fixture')
    def at(hour):
        return datetime(2026, 9, 16, hour, tzinfo=UTC)
    values = [
        (Point(at(12)), 'known', '0', False, 'selected'),
        (Point(at(14)), 'known', '100', False, 'selected'),
        (Interval(at(12), at(14)), 'known', '1.25', True, 'selected'),
        (Interval(at(11), at(13)), 'known', '10', False, 'selected'),
        (Interval(None, at(13)), 'known', '20', False, 'selected'),
        (Undated('private'), 'known', '30', False, 'selected'),
        (Point(at(12)), 'unknown', None, False, 'selected'),
        (Point(at(12)), 'known', '9', False, 'unresolved'),
        (Point(at(12)), 'not_applicable', None, False, 'selected'),
    ]
    tokens = TokenEvidence(TokenBreakdown(*(Unknown('private') for _ in range(4))), Unknown('private'))
    with store.connect(write=True) as db:
        db.execute("INSERT INTO session(id,harness,native_id) VALUES('pi:misleading','copilot-vscode','opaque')")
        db.execute("INSERT INTO session_attribution VALUES('pi:misleading',NULL,NULL,'unknown')")
        db.execute("INSERT INTO source_generation VALUES('source','/PRIVATE-CANARY',0,?,'pi:misleading','fixture',1,0,'available')", ('a' * 64,))
        for line, (timing, state, amount, lower, decision) in enumerate(values, 1):
            quantity = MeasuredQuantity('ai_credits', state, Decimal(amount) if amount is not None else None,
                                        'PRIVATE-CANARY' if amount is None else None, lower, 'PRIVATE-CANARY' if amount is not None else None)
            record = UsageRecord(EntryIdentity(str(line), None, line), 'assistant', timing, ModelIdentity(None, None),
                                 tokens, MissingEstimate('private'), 'stop', None, (), (quantity,))
            store._insert_observation(db, 'source', 'pi:misleading', record)
            if decision == 'unresolved':
                db.execute("UPDATE quantity_decision SET state='unresolved',owner_session=NULL WHERE observation_id IN (SELECT observation_id FROM appearance WHERE line=?)", (line,))
        db.execute('DELETE FROM decision')
        db.execute("INSERT INTO diagnostic(source_id,line,code) VALUES('source',NULL,'PRIVATE-CANARY https://secret <img> credential')")
    for scope, amount, known in [(AllTime(), '161.25', 6), (DateRange(at(12), at(14), 'UTC'), '1.25', 2)]:
        query = ReportQuery(None, scope)
        data = store.report_input(query)
        ordinary = build_report(data.contributions, revision=data.revision, query=query, catalog=rates)
        actual = build_aggregate_report(store, query, rates)
        assert actual == ordinary
        assert actual.quantities == (QuantityRow('copilot-vscode', 'ai_credits', Decimal(amount), known, 1, 1, 1, 1),)
        assert actual.projects[0].session_count == 1 and actual.sessions[0].harness == 'copilot-vscode'
        assert 'PRIVATE-CANARY' not in repr(actual) and 'PRIVATE-CANARY' not in repr(data.diagnostics)
        assert len([entry for entry in actual.coverage if entry.code == 'timing_unavailable']) == 1


def test_total_only_unknown_coverage_matches_aggregate(tmp_path):
    store = Storage(tmp_path / 'total.duckdb')
    store.import_source('/fixture', (Path(__file__).parent / 'fixtures/pi/source/invalid-count.jsonl').read_bytes())
    with store.connect(write=True) as db:
        db.execute("DELETE FROM decision WHERE measure<>'total'")
    query = ReportQuery(None, AllTime())
    rates = Catalog({}, '2026-09-16', 'fixture')
    data = store.report_input(query)
    expected = build_report(data.contributions, revision=data.revision, query=query, catalog=rates)
    assert 'usage_unavailable' in {entry.code for entry in expected.coverage}
    assert build_aggregate_report(store, query, rates) == expected


@pytest.mark.parametrize('state', ['known', 'unknown', 'lower_bound', 'unresolved', 'not_applicable', 'excluded', 'token_not_applicable'])
def test_persisted_overlap_population_requires_usage_not_only_na(tmp_path, state):
    store = Storage(tmp_path / 'overlap.duckdb')
    tokens = TokenEvidence(TokenBreakdown(*(Known(1) for _ in range(4))), Known(4))
    na = NotApplicable('not_applicable')
    na_tokens = TokenEvidence(TokenBreakdown(na, na, na, na), na)
    quantity_state = 'not_applicable' if state in {'not_applicable', 'token_not_applicable'} else 'unknown' if state == 'unknown' else 'known'
    quantity = MeasuredQuantity('request_count', quantity_state, Decimal('0') if quantity_state == 'known' else None,
                                None if quantity_state == 'known' else 'private', state == 'lower_bound', 'fixture' if quantity_state == 'known' else None)
    with store.connect(write=True) as db:
        db.execute("INSERT INTO project VALUES('project','fixture','directory','Fixture')")
        for sid, harness in [('one', 'copilot-vscode'), ('two', 'copilot-cli')]:
            db.execute('INSERT INTO session(id,harness,native_id) VALUES(?,?,?)', (sid, harness, sid))
            db.execute("INSERT INTO session_attribution VALUES(?,'project','/fixture','fixture')", (sid,))
            db.execute("INSERT INTO source_generation VALUES(?,?,0,?,?,'fixture',1,0,'available')", (sid, '/'+sid, 'a'*64, sid))
            record = UsageRecord(EntryIdentity(sid, None, 1), 'assistant', Point(datetime(2026, 9, 16, 12, tzinfo=UTC)),
                                 ModelIdentity('fixture', 'model'), tokens if sid == 'one' else na_tokens,
                                 MissingEstimate('private'), 'stop', None, (), (quantity,) if sid == 'two' and state != 'token_not_applicable' else ())
            store._insert_observation(db, sid, sid, record)
        db.execute("INSERT INTO decision SELECT id,'input','selected',session_id,NULL,'fixture','test' FROM observation WHERE session_id='one'")
        if state == 'token_not_applicable':
            db.execute("DELETE FROM quantity_decision")
            db.execute("INSERT INTO decision SELECT id,'input','selected',session_id,NULL,'fixture','test' FROM observation WHERE session_id='two'")
        else:
            db.execute("DELETE FROM decision WHERE observation_id IN (SELECT id FROM observation WHERE session_id='two')")
        if state in {'unresolved', 'excluded'}:
            canonical = "(SELECT id FROM observation WHERE session_id='one')" if state == 'excluded' else 'NULL'
            db.execute(f"UPDATE quantity_decision SET state=?,owner_session=NULL,canonical={canonical}", (state,))
    query = ReportQuery(None, AllTime())
    rates = Catalog({}, '2026-09-16', 'fixture')
    data = store.report_input(query)
    ordinary = build_report(data.contributions, revision=data.revision, query=query, catalog=rates)
    actual = build_aggregate_report(store, query, rates)
    assert actual == ordinary
    codes = [entry.code for entry in actual.coverage]
    assert codes.count('copilot_cross_source_join_unavailable') == (1 if state in {'known', 'unknown', 'lower_bound', 'unresolved'} else 0)


def test_quantity_keeps_reported_zero_distinct_from_unavailable():
    assert MeasuredQuantity('ai_credits', 'known', Decimal('0'), None, False, 'vscode:session').amount == Decimal('0')
    missing = MeasuredQuantity('ai_credits', 'unknown', None, 'not_recorded', False, None)
    assert missing.amount is None and missing.reason == 'not_recorded'


def test_quantity_rejects_contradictory_or_inexact_states():
    with pytest.raises(ContractViolation):
        MeasuredQuantity('nano_aiu', 'unknown', Decimal('0'), 'not_recorded', False, None)
    with pytest.raises(ContractViolation):
        MeasuredQuantity('premium_requests', 'known', Decimal('-1'), None, False, 'cli:shutdown')


def test_quantity_persistence_and_reports_keep_zero_unknown_and_lower_bounds_distinct(tmp_path):
    store = Storage(tmp_path / 'ledger.duckdb')
    query = ReportQuery(None, AllTime())
    rates = Catalog({}, '2026-09-16', 'fixture')
    before = build_aggregate_report(store, query, rates)
    zero_tokens = TokenEvidence(TokenBreakdown(*(Known(0) for _ in range(4))), Known(0), Known(0), Known(0))
    quantities = (
        MeasuredQuantity('ai_credits', 'known', Decimal('0'), None, False, 'vscode:zero'),
        MeasuredQuantity('ai_credits', 'known', Decimal('1.25'), None, True, 'vscode:lower-bound'),
        MeasuredQuantity('ai_credits', 'unknown', None, 'not_recorded', False, None),
    )
    with store.connect(write=True) as db:
        db.execute("INSERT INTO session(id,harness,native_id) VALUES('copilot-vscode:session','copilot-vscode','session')")
        db.execute("INSERT INTO session_attribution VALUES('copilot-vscode:session',NULL,NULL,'unknown')")
        db.execute("INSERT INTO source_generation VALUES('source','/fixture',0,?,'copilot-vscode:session','fixture',1,0,'available')", ('a' * 64,))
        for line, quantity in enumerate(quantities, 1):
            record = UsageRecord(EntryIdentity(f'entry-{line}', None, line), 'assistant',
                                 Point(datetime(2026, 9, 16, line, tzinfo=UTC)), ModelIdentity(None, None),
                                 zero_tokens, MissingEstimate('not_recorded'), 'stop', None, (), (quantity,))
            store._insert_observation(db, 'source', 'copilot-vscode:session', record)
    expected = (QuantityRow('copilot-vscode', 'ai_credits', Decimal('1.25'), 2, 1, 1),)
    aggregate = build_aggregate_report(store, query, rates)
    report_input = store.report_input(query)
    ordinary = build_report(report_input.contributions, revision=report_input.revision, query=query, catalog=rates)
    assert aggregate.quantities == ordinary.quantities == expected
    assert aggregate.tokens == before.tokens
    assert aggregate.money.known == before.money.known
    assert {(value.state, value.amount) for row in store.snapshot().observations for value in row.record.quantities} == {
        ('known', Decimal('0')), ('known', Decimal('1.25')), ('unknown', None),
    }
