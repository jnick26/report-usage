"""A retained CLI quantity can be superseded while its observation stays relevant."""
import json
from pathlib import Path

from harness_usage.aggregate_reporting import build_aggregate_report
from harness_usage.pricing import load_bundled_catalog
from harness_usage.reporting import AllTime, ReportQuery, build_report
from harness_usage.storage import Storage


def test_cli_missing_nano_aiu_does_not_crash_detailed_report(tmp_path):
    rows = [json.loads(line) for line in
            (Path(__file__).parent / 'fixtures/copilot_cli/current/events.jsonl').read_bytes().splitlines()]
    start, shutdown = rows[0], rows[4]
    shutdown['parentId'] = start['id']
    shutdown['data'].pop('totalNanoAiu')
    locator = tmp_path / 'session-state' / start['data']['sessionId'] / 'events.jsonl'
    store = Storage(tmp_path / 'ledger.duckdb')
    store.import_source(str(locator), ('\n'.join(map(json.dumps, (start, shutdown))) + '\n').encode())
    query = ReportQuery(None, AllTime())
    catalog = load_bundled_catalog()
    data = store.report_input(query)
    assert any(value.measure not in dict(row.quantity_decisions)
               for row in data.contributions for value in row.quantities)
    actual = build_report(data.contributions, revision=data.revision, query=query, catalog=catalog)
    expected = build_aggregate_report(store, query, catalog)
    assert actual.quantities == expected.quantities
    assert actual.tokens == expected.tokens
    assert actual.money == expected.money
    assert next(row for row in actual.quantities if row.measure == 'nano_aiu').unresolved_observations == 1
    store.close()
