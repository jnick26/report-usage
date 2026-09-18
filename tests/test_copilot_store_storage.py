from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
import hashlib
import json
from pathlib import Path

import duckdb
import pytest

from harness_usage.storage import Storage
from harness_usage.reporting import AllTime, ReportQuery, build_report, DateRange
from harness_usage.aggregate_reporting import build_aggregate_report
from harness_usage.pricing import Catalog
from test_copilot_cli_source import SESSION, cli_locator, event_rows, event_bytes

START = datetime(2026, 9, 16, tzinfo=UTC)


def test_populated_schema6_upgrade_preserves_data_and_restarts(tmp_path):
    import harness_usage.storage as module
    schema = Path(module.__file__).with_name('schema.sql').read_text()
    schema = schema.split('-- Copilot session-store provenance.', 1)[0]
    schema = schema.replace('schema_version=7', 'schema_version=6').replace('VALUES(1,7,0)', 'VALUES(1,6,0)')
    path = tmp_path / 'ledger.duckdb'
    db = duckdb.connect(str(path))
    db.execute(schema)
    db.execute("INSERT INTO session VALUES('pi:old','pi','old',NULL,NULL,NULL,NULL,NULL,NULL)")
    db.execute("INSERT INTO session_attribution VALUES('pi:old',NULL,NULL,'not_resolved')")
    db.execute('UPDATE ledger_meta SET revision=23')
    db.close()
    for _ in range(2):
        store = Storage(path)
        with store.connect() as db:
            assert db.execute('SELECT schema_version,revision FROM ledger_meta').one() == (7, 23)
            assert db.execute('SELECT id FROM session_view').one()[0] == 'pi:old'
        store.close()


def snapshot(tmp_path, outputs=(4,), *, locator='session-store.db', rows=None, sessions=None):
    from harness_usage.copilot_store_reader import CopilotStoreCall, CopilotStoreSession, CopilotStoreSnapshot
    sessions = sessions if sessions is not None else (CopilotStoreSession(SESSION, None, START, START + timedelta(seconds=1), None),)
    calls = rows if rows is not None else tuple(CopilotStoreCall(
        i + 1, SESSION, i, START + timedelta(milliseconds=10 + i), 'model-alpha', None, None,
        0, output, 0, 0, 0, Decimal(0), None, None) for i, output in enumerate(outputs))
    return CopilotStoreSnapshot(str(tmp_path / locator), hashlib.sha256(repr((sessions, calls)).encode()).hexdigest(), sessions, calls)


def report(store, query=None):
    query = query or ReportQuery(None, AllTime())
    data = store.report_input(query)
    catalog = Catalog({}, 'test', 'test')
    ordinary = build_report(data.contributions, revision=data.revision, query=query, catalog=catalog)
    assert build_aggregate_report(store, query, catalog) == ordinary
    return ordinary


def shutdown(tmp_path, output=10, count=2):
    rows = event_rows()[:1]
    row = event_rows()[4]
    row['parentId'] = rows[0]['id']
    row['data']['totalNanoAiu'] = 0
    row['data']['modelMetrics'] = {'model-alpha': {'requests': {'count': count, 'cost': 0},
        'usage': {'inputTokens': 0, 'outputTokens': output, 'cacheReadTokens': 0, 'cacheWriteTokens': 0, 'reasoningTokens': 0}}}
    return (cli_locator(tmp_path), event_bytes(rows + [row]))


@pytest.mark.parametrize('outputs,want,unknown', [((4, 6), 10, False), ((4,), 10, False), ((12,), 0, True)])
@pytest.mark.parametrize('reverse', [False, True])
def test_json_overlap_never_adds_both(tmp_path, outputs, want, unknown, reverse):
    store = Storage(tmp_path / 'ledger.duckdb')
    sources = [shutdown(tmp_path), snapshot(tmp_path, outputs)]
    store.import_sources(reversed(sources) if reverse else sources)
    output = report(store).tokens.output
    assert output.known == want
    assert bool(output.unknown_observations) == unknown
    store.close()


def test_database_only_lower_bounds_zero_unknown_and_reopen(tmp_path):
    store = Storage(tmp_path / 'ledger.duckdb')
    source = snapshot(tmp_path, (0, 4))
    first = store.import_session_store(source)
    assert store.import_session_store(source) == first
    result = report(store)
    assert result.tokens.output.known == 4
    assert result.tokens.output.lower_bound_observations == 2
    assert result.tokens.cache_read.lower_bound_observations == 2
    assert result.tokens.input.lower_bound_observations == 2
    assert report(store, ReportQuery(None, DateRange(START, START + timedelta(days=1), 'UTC'))).tokens.output.known == 4
    store.close()
    store = Storage(tmp_path / 'ledger.duckdb')
    assert report(store).tokens.output.known == 4
    store.close()


def test_copies_rebuild_ids_and_duplicate_call_multiplicity(tmp_path):
    store = Storage(tmp_path / 'ledger.duckdb')
    source = snapshot(tmp_path, (4, 4))
    # Same semantic call twice is two calls, not one matching hash.
    calls = (source.calls[0], replace(source.calls[0], row_id=2))
    source = snapshot(tmp_path, rows=calls)
    store.import_session_store(source)
    assert report(store).tokens.output.known == 8
    copied = snapshot(tmp_path, rows=tuple(replace(call, row_id=call.row_id + 100) for call in calls), locator='copy.db')
    store.import_session_store(copied)
    assert report(store).tokens.output.known == 8
    rebuilt = snapshot(tmp_path, rows=tuple(replace(call, row_id=call.row_id + 1000) for call in calls))
    store.import_session_store(rebuilt)
    assert report(store).tokens.output.known == 8
    store.import_session_store(snapshot(tmp_path, (5,), locator='copy.db'))
    assert report(store).tokens.output.known == 0
    assert report(store).tokens.output.unknown_observations > 0
    store.close()


def test_replacement_deletion_missing_and_reappearance_are_distinct(tmp_path):
    store = Storage(tmp_path / 'ledger.duckdb')
    source = snapshot(tmp_path, (4, 6))
    store.import_session_store(source)
    store.mark_missing((source.locator,))
    assert report(store).tokens.output.known == 10
    assert any(row.saved_history for row in store.report_input(ReportQuery(None, AllTime())).contributions)
    store.import_session_store(snapshot(tmp_path, (4,)))
    assert report(store).tokens.output.known == 4
    store.import_session_store(snapshot(tmp_path, ()))
    assert report(store).tokens.output.known == 0
    assert report(store).tokens.output.unknown_observations > 0
    store.import_session_store(snapshot(tmp_path, (), sessions=()))
    assert report(store).tokens.output.known == 0
    assert report(store).tokens.output.unknown_observations == 0
    store.import_session_store(source)
    assert report(store).tokens.output.known == 10
    store.close()


def test_equal_detail_has_point_dates_and_does_not_use_store_creation_epoch(tmp_path):
    store = Storage(tmp_path / 'ledger.duckdb')
    source = snapshot(tmp_path, (4, 6))
    source = replace(source, sessions=(replace(source.sessions[0], created_at=START - timedelta(days=3)),))
    store.import_sources((shutdown(tmp_path), source))
    result = report(store, ReportQuery(None, DateRange(START + timedelta(milliseconds=10), START + timedelta(milliseconds=11), 'UTC')))
    assert result.tokens.output.known == 4
    assert result.tokens.output.lower_bound_observations == 0
    store.close()


@pytest.mark.parametrize('at', [None, START - timedelta(seconds=1), START + timedelta(days=1)])
def test_calls_outside_proven_epoch_are_never_appended(tmp_path, at):
    store = Storage(tmp_path / 'ledger.duckdb')
    source = snapshot(tmp_path, (10,))
    source = snapshot(tmp_path, rows=(replace(source.calls[0], created_at=at),))
    store.import_sources((shutdown(tmp_path, count=1), source))
    result = report(store)
    assert result.tokens.output.known == 0
    assert result.tokens.output.unknown_observations > 0
    store.close()


def test_flat_cache_vector_preserves_unknown_fresh_input_and_agent_is_not_added_twice(tmp_path):
    store = Storage(tmp_path / 'ledger.duckdb')
    source = snapshot(tmp_path)
    call = replace(source.calls[0], input_tokens=100, cache_read_tokens=20, cache_write_tokens=10,
                   agent_id='child', parent_tool_call_id='tool', token_details_json='[{"type":"cache_write","count":999}]')
    store.import_session_store(snapshot(tmp_path, rows=(call,)))
    result = report(store)
    assert result.tokens.input.known == 0 and result.tokens.input.unknown_observations == 1
    assert result.tokens.cache_read.known == 20 and result.tokens.cache_write.known == 10
    assert result.tokens.output.known == 4
    assert result.tokens.cache_write.lower_bound_observations == 1
    store.close()


def test_invalid_owner_rolls_back_whole_snapshot(tmp_path):
    store = Storage(tmp_path / 'ledger.duckdb')
    source = snapshot(tmp_path)
    source = snapshot(tmp_path, rows=(replace(source.calls[0], session_id='missing-owner'),))
    with pytest.raises(ValueError, match='invalid_store_owner'):
        store.import_session_store(source)
    assert store.locators() == ()
    store.close()


def test_equal_native_copies_and_separate_imports_do_not_remove_detail(tmp_path):
    store = Storage(tmp_path / 'ledger.duckdb')
    store.import_session_store(snapshot(tmp_path, (4, 6)))
    store.import_sources((shutdown(tmp_path / 'one'), shutdown(tmp_path / 'two')))
    result = report(store)
    assert result.tokens.output.known == 10
    assert result.tokens.output.unknown_observations == 0
    assert result.tokens.output.lower_bound_observations == 0
    store.close()


def test_current_model_presence_never_resurrects_stale_database_model(tmp_path):
    store = Storage(tmp_path / 'ledger.duckdb')
    source = snapshot(tmp_path, (10,))
    store.import_sources((shutdown(tmp_path, count=1), source))
    locator, payload = shutdown(tmp_path, count=1)
    rows = [json.loads(line) for line in payload.splitlines()]
    rows[-1]['data']['modelMetrics'] = {}
    store.import_source(locator, event_bytes(rows))
    result = report(store)
    assert result.tokens.output.known == 0
    assert result.tokens.output.unknown_observations > 0
    store.close()


def test_unknown_native_epoch_is_not_inferred_from_matching_counts(tmp_path):
    store = Storage(tmp_path / 'ledger.duckdb')
    locator, payload = shutdown(tmp_path)
    rows = [json.loads(line) for line in payload.splitlines()]
    rows[0]['data']['startTime'] = '2026-09-15T00:00:00.000Z'
    rows[-1]['data']['sessionStartTime'] -= 86400000
    store.import_sources((shutdown(tmp_path / 'one'), (locator, event_bytes(rows)), snapshot(tmp_path, (4, 6))))
    result = report(store)
    assert result.tokens.output.known == 0
    assert result.tokens.output.unknown_observations > 0
    store.close()


def test_invalid_call_does_not_become_zero_or_poison_other_session(tmp_path):
    store = Storage(tmp_path / 'ledger.duckdb')
    source = snapshot(tmp_path, (4, 6))
    other = '22222222-2222-4222-8222-222222222222'
    sessions = source.sessions + (replace(source.sessions[0], session_id=other),)
    calls = (replace(source.calls[0], output_tokens=None, invalid_fields=('output_tokens',)),
             replace(source.calls[1], session_id=other))
    store.import_session_store(snapshot(tmp_path, rows=calls, sessions=sessions))
    result = report(store)
    assert result.tokens.output.known == 6
    assert result.tokens.output.unknown_observations == 1
    assert result.tokens.output.lower_bound_observations == 1
    store.close()


def test_metadata_only_session_reports_unavailable_not_known_zero(tmp_path):
    store = Storage(tmp_path / 'ledger.duckdb')
    store.import_session_store(snapshot(tmp_path, ()))
    result = report(store)
    assert len(result.sessions) == 1
    assert result.tokens.output.known == 0
    assert result.tokens.output.unknown_observations == 1
    assert result.tokens.output.lower_bound_observations == 0
    store.close()


@pytest.mark.parametrize('db_nano,want,unresolved', [(10, Decimal(10), 0), (6, Decimal(10), 0), (12, Decimal(0), 2)])
@pytest.mark.parametrize('reverse,copied', [(False, False), (True, False), (False, True), (True, True)])
def test_nano_aiu_reconciles_independently_of_conflicting_tokens(tmp_path, db_nano, want, unresolved, reverse, copied):
    store = Storage(tmp_path / 'ledger.duckdb')
    locator, payload = shutdown(tmp_path, output=10, count=1)
    native = [json.loads(line) for line in payload.splitlines()]
    native[-1]['data']['totalNanoAiu'] = 10
    source = snapshot(tmp_path, (12,))
    source = snapshot(tmp_path, rows=(replace(source.calls[0], total_nano_aiu=Decimal(db_nano)),))
    sources = [(locator, event_bytes(native)), source]
    if copied:
        sources.append(snapshot(tmp_path, rows=(replace(source.calls[0], row_id=123),), locator='copy.db'))
    store.import_sources(reversed(sources) if reverse else sources)
    result = report(store)
    assert result.tokens.output.known == 0
    assert result.tokens.output.unknown_observations > 0
    nano = next(value for value in result.quantities if value.measure == 'nano_aiu')
    assert nano.known == want
    assert nano.known_observations == (1 if want else 0)
    assert nano.unresolved_observations == unresolved
    assert store.import_sources(sources) == result.revision
    assert report(store) == result
    store.close()


def test_matching_nano_aiu_does_not_bypass_native_epoch_bounds(tmp_path):
    store = Storage(tmp_path / 'ledger.duckdb')
    locator, payload = shutdown(tmp_path, output=10, count=1)
    native = [json.loads(line) for line in payload.splitlines()]
    native[-1]['data']['totalNanoAiu'] = 10
    source = snapshot(tmp_path, (10,))
    source = snapshot(tmp_path, rows=(replace(source.calls[0], total_nano_aiu=Decimal(10), created_at=START + timedelta(days=1)),))
    store.import_sources(((locator, event_bytes(native)), source))
    nano = next(value for value in report(store).quantities if value.measure == 'nano_aiu')
    assert nano.known == 0
    assert nano.unresolved_observations > 0
    store.close()


@pytest.mark.parametrize('field', ['request_multiplier', 'token_details', 'turn_index', 'output_tokens', 'total_nano_aiu'])
@pytest.mark.parametrize('copied', [False, True])
def test_invalid_store_field_preserves_other_database_only_measures(tmp_path, field, copied):
    store = Storage(tmp_path / 'ledger.duckdb')
    source = snapshot(tmp_path, (7,))
    call = replace(source.calls[0], input_tokens=3, total_nano_aiu=Decimal(9), invalid_fields=(field,))
    if field in ('output_tokens', 'total_nano_aiu'):
        call = replace(call, **{field: None})
    source = snapshot(tmp_path, rows=(call,))
    sources = [source]
    if copied:
        sources.append(snapshot(tmp_path, rows=(replace(call, row_id=123),), locator='copy.db'))
    store.import_sources(sources)
    result = report(store)
    assert result.tokens.input.known == 3
    assert result.tokens.input.unknown_observations == 0
    assert result.tokens.input.lower_bound_observations == 1
    assert result.tokens.output.known == (0 if field == 'output_tokens' else 7)
    assert result.tokens.output.unknown_observations == (1 if field == 'output_tokens' else 0)
    assert result.tokens.cache_read.unknown_observations == 0
    assert result.tokens.cache_read.lower_bound_observations == 1
    quantities = {value.measure: value for value in result.quantities}
    assert quantities['request_count'].known == 1
    assert quantities['request_count'].lower_bound_observations == 1
    assert quantities['nano_aiu'].known == (0 if field == 'total_nano_aiu' else 9)
    assert quantities['nano_aiu'].unknown_observations == (1 if field == 'total_nano_aiu' else 0)
    assert quantities['nano_aiu'].unresolved_observations == 0
    store.close()


@pytest.mark.parametrize('field', ['request_multiplier', 'token_details', 'turn_index', 'output_tokens', 'total_nano_aiu'])
def test_invalid_store_fields_do_not_cross_token_and_nano_overlap_scopes(tmp_path, field):
    store = Storage(tmp_path / 'ledger.duckdb')
    locator, payload = shutdown(tmp_path, output=10, count=1)
    native = [json.loads(line) for line in payload.splitlines()]
    native[-1]['data']['totalNanoAiu'] = 9
    source = snapshot(tmp_path, (10,))
    call = replace(source.calls[0], total_nano_aiu=Decimal(9), invalid_fields=(field,))
    if field in ('output_tokens', 'total_nano_aiu'):
        call = replace(call, **{field: None})
    store.import_sources(((locator, event_bytes(native)), snapshot(tmp_path, rows=(call,))))
    result = report(store)
    assert result.tokens.output.known == (0 if field == 'output_tokens' else 10)
    assert bool(result.tokens.output.unknown_observations) == (field == 'output_tokens')
    nano = next(value for value in result.quantities if value.measure == 'nano_aiu')
    assert nano.known == (0 if field == 'total_nano_aiu' else 9)
    assert bool(nano.unresolved_observations) == (field == 'total_nano_aiu')
    store.close()
