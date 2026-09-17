"""HTTP trust boundaries and the real no-JavaScript navigation contract."""
import re
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from harness_usage.application import Application
from harness_usage.web import create_app


class Sources:
    def __init__(self):
        self.roots = ()
        self.imports = 0
        self.closed = False

    def close(self):
        self.closed = True

    def get_roots(self):
        return self.roots

    def set_roots(self, roots):
        self.roots = roots

    def start_import(self):
        self.imports += 1
        return self.status()

    def status(self):
        return SimpleNamespace(state='idle', revision=0, run_id=None, files_processed=0, error=None)


def csrf(client):
    response = client.get('/sources')
    assert response.status_code == 200
    return re.search(r'name="csrf" value="([^"]+)"', response.text)[1]


@pytest.mark.parametrize(('known','unknown','lower','unresolved','na','amount','statuses'), [
    (1, 0, 0, 0, 0, '0', ('Recorded total',)),
    (0, 1, 0, 0, 0, '0', ('Unavailable',)),
    (0, 0, 0, 1, 0, '0', ('Unresolved',)),
    (0, 0, 0, 0, 1, '0', ('Not applicable',)),
    (1, 1, 1, 1, 1, '0.12345678901234567890123456782', ('Lower bound', 'Unavailable', 'Unresolved', 'Not applicable')),
])
def test_source_quantity_table_has_visible_exact_states_and_accessible_units(known, unknown, lower, unresolved, na, amount, statuses):
    from decimal import Decimal
    from harness_usage.reporting import QuantityRow, Coverage, build_report
    backend = Sources()
    backend.roots = ('/fixture',)
    backend.timezone = 'UTC'
    backend.report = lambda query, **options: replace(
        build_report([], revision=1, query=query),
        quantities=(QuantityRow('copilot-vscode', 'ai_credits', Decimal(amount), known, unknown, lower, unresolved, na),),
        coverage=(Coverage('usage_unavailable', ()), Coverage('/private/SECRET credential https://secret', ()),))
    client = TestClient(create_app(backend), base_url='http://localhost')
    for headers in ({}, {'X-Up-Target': '#main'}):
        response = client.get('/', headers=headers)
        assert response.status_code == 200
        page = response.text
        section = re.search(r'<section[^>]*aria-label="Source quantities".*?</section>', page, re.S)
        assert section is not None
        visible = re.sub(r'<[^>]+>', '', section[0])
        assert 'Copilot in VS Code' in visible and 'AI credits' in visible and 'Not billed spend' in visible
        assert all(status in visible for status in statuses)
        assert 'Authoritative' not in visible and '$' not in visible
        assert re.findall(r'<th scope="col">(.*?)</th>', section[0]) == ['Source', 'Measure', 'Amount', 'Unit', 'Status']
        assert 'role="region"' in section[0] and 'tabindex="0"' in section[0] and '<caption>' in section[0]
        cells = [re.sub(r'<[^>]+>', '', value).strip() for value in re.findall(r'<td[^>]*>(.*?)</td>', section[0], re.S)]
        assert cells[2] == (('≥ ' if lower else '') + amount if known else '—')
        assert '/private/SECRET' not in page and 'https://secret' not in page


def test_unknown_quantity_labels_never_render_private_source_values():
    from decimal import Decimal
    from harness_usage.reporting import QuantityRow, build_report
    backend = Sources()
    backend.roots = ('/fixture',)
    backend.timezone = 'UTC'
    backend.report = lambda query, **options: replace(build_report([], revision=1, query=query), quantities=(
        QuantityRow('PRIVATE-HARNESS <script>', 'PRIVATE-MEASURE https://secret', Decimal('0'), 1, 0, 0),))
    page = TestClient(create_app(backend), base_url='http://localhost').get('/').text
    assert 'Unknown source' in page and 'Unknown measure' in page and 'Unknown unit' in page
    assert 'PRIVATE-HARNESS' not in page and 'PRIVATE-MEASURE' not in page and 'https://secret' not in page


def test_sources_post_requires_same_origin_and_per_launch_token(tmp_path):
    backend = Sources()
    client = TestClient(create_app(backend), base_url='http://127.0.0.1:8765')
    token = csrf(client)
    data = {'csrf': token, 'roots': str(tmp_path)}
    assert client.post('/sources', data=data, headers={'Origin': 'https://foreign.example'}).status_code == 403
    assert client.post('/sources', data={**data, 'csrf': 'wrong'}).status_code == 403
    assert client.post('/sources', data={**data, 'csrf': 'é'}).status_code == 403
    assert backend.roots == ()
    assert client.post('/sources', data=data, follow_redirects=False).status_code == 303
    assert backend.roots == (str(tmp_path),)
    assert backend.imports == 1
    other = TestClient(create_app(backend), base_url='http://127.0.0.1:8765')
    assert other.post('/sources', data=data).status_code == 403


def test_host_and_static_boundary_and_invalid_source(tmp_path):
    client = TestClient(create_app(Sources()), base_url='http://localhost:8765')
    assert client.get('/sources', headers={'Host': 'attacker.example'}).status_code == 400
    assert client.post('/import', headers={'Origin': 'http://localhost:9999'}).status_code == 403
    token = csrf(client)
    assert client.post('/sources', data={'csrf': token, 'roots': str(tmp_path / 'missing')}).status_code == 422
    outside = tmp_path / 'private.txt'
    outside.write_text('private evidence')
    assert client.get('/static/' + str(outside)).status_code == 404
    for asset in ['app.css', 'app.js', 'vendor/unpoly.min.js', 'vendor/NotoSans.ttf']:
        assert client.get('/static/' + asset).status_code == 200


def test_source_names_escape_html_and_import_redirect_is_local(tmp_path):
    backend = Sources()
    backend.roots = ('/tmp/<script>alert(1)</script>',)
    client = TestClient(create_app(backend), base_url='http://127.0.0.1')
    page = client.get('/sources').text
    assert '&lt;script&gt;' in page
    assert '<script>alert(1)</script>' not in page
    token = csrf(client)
    result = client.post('/import', data={'csrf': token, 'return_to': 'https://foreign.example'}, follow_redirects=False)
    assert result.status_code == 303
    assert result.headers['location'] == '/'


@pytest.mark.parametrize(('amount', 'cost_label'), [('0', 'Unpriced'), ('0.01', '>$0.01</button>')])
def test_real_report_renders_nine_columns_escaping_and_unknowns(amount, cost_label):
    from datetime import UTC, datetime
    from decimal import Decimal
    from harness_usage.domain import Assigned, ProjectId, SessionId, ModelIdentity
    from harness_usage.pricing import CostLine
    from harness_usage.reporting import (AllTime, ReportQuery, Report, MetricSum, TokenSums,
                                        MoneySum, ProjectRow, SessionRow, UnknownElapsed)
    known = MetricSum(5, 0, 0)
    unknown = MetricSum(0, 1, 0)
    tokens = TokenSums(known, known, unknown, known, MetricSum(15, 0, 0))
    money = MoneySum(Decimal(amount), 1, 'models_dev_usd', (CostLine(ModelIdentity('openai', 'example'), 'input', 5000, Decimal('2'), Decimal('0.01'), 'openai'), CostLine(ModelIdentity(None,None), 'input', 100, None, None, None, reason='model_not_in_catalog')), '2026-09-12')
    session = SessionRow(SessionId('pi:abc'), '<img src=x onerror=alert(1)>', Assigned(ProjectId('repo:test'), '/a/tree', 'git'), UnknownElapsed('missing_endpoints'), tokens, money, (), (), started=datetime(2026, 9, 12, tzinfo=UTC), subagent_count=2)
    project = ProjectRow(ProjectId('repo:test'), 'Example', tokens, money, 1)
    backend = Sources()
    backend.timezone = 'UTC'
    backend.roots = ('/saved',)
    backend.report = lambda query, **options: Report(3, query, (project,), (session,), tokens, money, (), tokens, ())
    client = TestClient(create_app(backend), base_url='http://localhost')
    page = client.get('/?project=repo%3Atest').text
    columns = [re.sub(r'<[^>]+>', '', cell) for cell in re.findall(r'<th scope="col"[^>]*>(.*?)</th>', page)]
    assert columns[:9] == ['Started', 'Elapsed', 'Session', 'Activity', 'Input', 'Output', 'Cache read', 'Cache write', 'Est. cost']
    assert '2 subagents' in page
    assert '&lt;img src=x onerror=alert(1)&gt;' in page
    assert '<img src=x' not in page
    assert '&lt;0.01' in page
    assert 'Unavailable' in page
    assert cost_label in page
    if amount == '0':
        assert '<strong>Unpriced</strong>' in page
        assert '<strong>$0.00</strong>' not in page
    assert 'data-revision="3"' in page
    assert 'type="datetime-local"' in page
    assert 'name="offset_from"' not in page
    assert 'Time options' not in page
    assert ' + partial' not in page
    assert 'Usage breakdown in Mtok' in page
    assert 'Cache read' in page
    assert '0.005 Mtok × $2/Mtok = $0.01' in page
    assert 'Unknown model' not in page
    assert '= $—' not in page
    assert 'models.dev' in page and '2026-09-12' in page
    projects = client.get('/').text
    project_columns = re.findall(r'<th scope="col"[^>]*>(.*?)</th>', projects)
    assert project_columns == ['Project', 'Input', 'Output', 'Cache read', 'Cache write', 'Sessions', 'Est. cost']
    assert '+ partial' not in projects


def test_report_renders_stored_claude_harness_without_pi_fallback(tmp_path):
    identity = '11111111-1111-4111-8111-111111111111'
    fixture = Path(__file__).parent / 'fixtures/claude/main.jsonl'
    application = Application(tmp_path / 'data')
    application.set_roots((str(tmp_path),))
    application.storage.import_source(str(tmp_path / f'{identity}.jsonl'), fixture.read_bytes())
    with TestClient(create_app(application), base_url='http://127.0.0.1:8765') as client:
        page = client.get('/?project=unassigned').text
    assert 'Claude Code' in page
    assert 'Claude Code / Unknown provider / claude-sonnet-4-5' in page
    assert '<div class="small muted">Pi' not in page
    assert '&lt;$0.01' in page
    assert 'Subtotal' in page
    assert 'final output may be unavailable' in page


def test_invalid_dates_retain_previous_query_and_display_inline_error():
    from harness_usage.reporting import Report, MetricSum, TokenSums, MoneySum
    from decimal import Decimal
    zero = MetricSum(0, 0, 0)
    tokens = TokenSums(zero, zero, zero, zero, zero)
    backend = Sources()
    backend.timezone = 'Europe/Kyiv'
    backend.roots = ('/saved',)
    queries = []
    def report(query, **options):
        queries.append(query)
        return Report(2, query, (), (), tokens, MoneySum(Decimal(0), 0, 'pi_recorded_usd'), (), tokens, ())
    backend.report = report
    client = TestClient(create_app(backend), base_url='http://localhost')
    response = client.get('/', params={'from': '2026-10-25T03:30', 'to': '2026-10-25T05:00', 'tz': 'Europe/Kyiv', 'previous_from': '2026-09-12T00:00+03:00', 'previous_to': '2026-09-13T00:00+03:00'})
    assert response.status_code == 422
    assert 'role="alert"' in response.text
    assert queries[-1].range.start.isoformat() == '2026-09-11T21:00:00+00:00'
    assert 'No sessions in this range' in response.text
    assert 'class="ruler"' not in response.text


def test_presets_use_local_calendar_days_and_hour_actual_duration():
    from datetime import UTC, datetime
    from harness_usage.web import preset_range
    spring = preset_range('day', 'Europe/Kyiv', datetime(2026, 3, 29, 10, tzinfo=UTC))
    fall = preset_range('day', 'Europe/Kyiv', datetime(2026, 10, 25, 10, tzinfo=UTC))
    assert (spring.end - spring.start).total_seconds() == 23 * 3600
    assert (fall.end - fall.start).total_seconds() == 25 * 3600
    repeated = preset_range('hour', 'Europe/Kyiv', datetime(2026, 10, 25, 1, 30, tzinfo=UTC))
    assert repeated.start.isoformat() == '2026-10-25T01:00:00+00:00'
    assert (repeated.end - repeated.start).total_seconds() == 3600


def test_explicit_offset_url_preserves_exact_ambiguous_hour():
    from decimal import Decimal
    from harness_usage.reporting import Report, MetricSum, TokenSums, MoneySum
    zero = MetricSum()
    tokens = TokenSums(zero, zero, zero, zero, zero)
    backend = Sources()
    backend.timezone = 'Europe/Kyiv'
    backend.roots = ('/saved',)
    queries = []
    def report(query, **options):
        queries.append(query)
        return Report(1, query, (), (), tokens, MoneySum(Decimal(0), 0), (), tokens, ())
    backend.report = report
    client = TestClient(create_app(backend), base_url='http://localhost')
    page = client.get('/', params={'from': '2026-10-25T03:00', 'to': '2026-10-25T04:00', 'offset_from': '2026-10-25T03:00+02:00', 'offset_to': '2026-10-25T04:00+02:00', 'tz': 'Europe/Kyiv'})
    assert page.status_code == 200
    assert queries[-1].range.start.isoformat() == '2026-10-25T01:00:00+00:00'
    assert queries[-1].range.end.isoformat() == '2026-10-25T02:00:00+00:00'
    assert 'value="2026-10-25T03:00"' in page.text


def test_failed_source_is_identified_without_raw_error_details():
    backend = Sources()
    backend.roots = ('/first/root', '/second/root')
    backend.status = lambda: SimpleNamespace(state='failed', revision=2, run_id='run', files_processed=1, error='source_2_unavailable')
    client = TestClient(create_app(backend), base_url='http://localhost')
    page = client.get('/sources').text
    assert 'Source 2' in page
    assert '/second/root' in page
    assert 'source_2_unavailable' not in page


def test_configured_empty_source_shows_setup_guidance_not_an_empty_filter(tmp_path):
    from harness_usage.application import Application
    source = tmp_path / 'empty-sources'
    source.mkdir()
    backend = Application(tmp_path / 'data', timezone='UTC')
    backend.set_roots((str(source),))
    client = TestClient(create_app(backend), base_url='http://localhost')
    page = client.get('/').text
    assert 'No imported history yet' in page
    assert 'No sessions in this range' not in page


def test_empty_project_keeps_name_and_huge_axis_keeps_previous_report(tmp_path):
    from harness_usage.application import Application
    backend = Application(tmp_path / 'data', timezone='UTC')
    backend.set_roots((str(tmp_path),))
    client = TestClient(create_app(backend), base_url='http://localhost', raise_server_exceptions=False)
    empty = client.get('/', params={'project': 'git:/projects/Example/.git', 'from': '2026-09-12T00:00', 'to': '2026-09-13T00:00', 'tz': 'UTC'})
    assert '<h1>Example</h1>' in empty.text
    (tmp_path / 'ordinary.jsonl').write_bytes((Path(__file__).parent / 'fixtures/pi/source/ordinary.jsonl').read_bytes())
    backend.start_import()
    backend.close()
    huge = client.get('/', params={'project': 'unassigned', 'from': '0001-01-01T00:00+00:00', 'to': '9999-01-01T00:00+00:00', 'tz': 'UTC', 'previous_from': '2026-09-12T00:00', 'previous_to': '2026-09-13T00:00'})
    assert huge.status_code == 422
    assert 'role="alert"' in huge.text
    assert '<h1>Unassigned</h1>' in huge.text


def test_range_links_do_not_discard_subminute_precision(tmp_path):
    from harness_usage.application import Application
    backend = Application(tmp_path / 'data', timezone='UTC')
    backend.set_roots((str(tmp_path),))
    client = TestClient(create_app(backend), base_url='http://localhost')
    page = client.get('/', params={'from': '2026-09-12T09:00:30.123456+00:00', 'to': '2026-09-12T09:05:30.654321+00:00'}).text
    assert 'name="previous_from" value="2026-09-12T09:00:30.123456+00:00"' in page
    assert 'name="previous_to" value="2026-09-12T09:05:30.654321+00:00"' in page
    assert '09%3A00%3A30.123456' in page


def test_source_changes_during_import_keep_specific_recovery_message(tmp_path):
    from harness_usage.application import SourceChangeDuringImport
    class BusySources(Sources):
        def set_roots(self, roots):
            raise SourceChangeDuringImport()
    backend = BusySources()
    client = TestClient(create_app(backend), base_url='http://localhost')
    page = client.post('/sources', data={'csrf': csrf(client), 'roots': str(tmp_path)})
    assert page.status_code == 422
    assert 'Wait for the current import to finish' in page.text


def test_http_lifespan_closes_application_before_shutdown():
    backend = Sources()
    with TestClient(create_app(backend), base_url='http://localhost') as client:
        assert client.get('/sources').status_code == 200
        assert not backend.closed
    assert backend.closed


def test_real_pages_keep_usage_sorting_and_pagination(tmp_path):
    import json
    from uuid import UUID
    from harness_usage.application import Application
    from harness_usage.domain import Assigned, ProjectId

    backend = Application(tmp_path / 'data', timezone='UTC')
    backend.set_roots((str(tmp_path),))
    ordinary = (Path(__file__).parent / 'fixtures/pi/source/ordinary.jsonl').read_text().splitlines()
    files = []
    for index in range(52):
        header = json.loads(ordinary[0])
        header['id'] = str(UUID(int=index + 1))
        files.append((f'/synthetic/{index}.jsonl', (json.dumps(header) + '\n' + ordinary[1] + '\n').encode()))
    backend.storage.import_sources(files)
    backend.storage.save_attributions({session.id: Assigned(ProjectId('directory:/projects/example'), '/projects/example/feature-search', 'directory') for session in backend.storage.snapshot().sessions})
    client = TestClient(create_app(backend), base_url='http://localhost')
    params = {'project': 'directory:/projects/example', 'tz': 'UTC', 'sort': 'cost_desc', 'from': '2026-09-12T00:00+00:00', 'to': '2026-09-13T00:00+00:00'}
    first = client.get('/', params=params)
    second = client.get('/', params={**params, 'page': '2'})
    assert first.status_code == second.status_code == 200
    assert first.text.count('class="session-name"') == 50
    assert second.text.count('class="session-name"') == 2
    assert '1–50 of 52' in first.text and '51–52 of 52' in second.text
    assert re.search(r'<div class="usage-summary".*?</div>', first.text, re.S)[0] == re.search(r'<div class="usage-summary".*?</div>', second.text, re.S)[0]
    assert 'Pi · feature-search' not in first.text
    assert '<p class="project-path muted small">/projects/example</p>' in first.text
    assert '/projects/example/feature-search' not in first.text
    assert first.text.count('class="ticks"') == 2
    assert 'class="ticks"' not in re.search(r'<tbody>.*?</tbody>', first.text, re.S)[0]
    assert 'Observed first-to-last' not in first.text
    assert '<details class="value">' not in first.text
    assert 'title="' in first.text
    assert 'Interval amounts' not in first.text
    assert 'Exact interval amounts' not in first.text
    assert 'Partial evidence' not in first.text
    assert 'timing and remain separate' not in first.text
    assert 'known tokens.</p>' not in first.text  # Hidden per-session interval data is not rendered.
    assert 'page=2' in second.text
    assert 'aria-sort="descending"' in first.text
    assert 'name="sort" value="cost_desc"' in first.text
    assert 'sort=cost_desc' in second.text
    assert 'sort=cost_asc' in first.text
    assert client.get('/', params={**params, 'page': '0'}).status_code == 422
    overview = client.get('/', params={'tz': 'UTC'})
    assert 'Usage breakdown in Mtok' in overview.text
    assert 'No sessions in this range' not in overview.text


def test_bucket_tooltip_is_compact_disjoint_usage_and_cost_without_dates():
    from datetime import UTC, datetime
    from decimal import Decimal
    from harness_usage.reporting import Bucket, MetricSum, TokenSums, MoneySum
    from harness_usage.web import bucket_tooltip
    values = TokenSums(*(MetricSum(n) for n in (100000, 200000, 300000, 400000, 1000000)))
    bucket = Bucket(datetime(2026,9,12,tzinfo=UTC), datetime(2026,9,13,tzinfo=UTC), values, MoneySum(Decimal('1.25'),0))
    text = bucket_tooltip(bucket)
    assert text == 'Input 0.10 · Output 0.20 Mtok\nCache read 0.30 · Write 0.40 Mtok\n$1.25'
    assert '2026' not in text and 'known tokens' not in text


def test_calculation_formatters_preserve_exact_quantities_without_float_noise():
    from decimal import Decimal
    from harness_usage.web import exact_mtok, decimal_text
    assert exact_mtok(1) == '0.000001'
    assert exact_mtok(2_000_000) == '2'
    assert decimal_text(Decimal('20')) == '20'
    assert decimal_text(Decimal('0.214500000')) == '0.2145'
