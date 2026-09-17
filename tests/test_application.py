from pathlib import Path
from threading import Event
import sqlite3
import pytest

from harness_usage.reporting import AllTime, ReportQuery
from harness_usage.source_input import MAX_PROBE_BYTES, SourcePayload, detect_source
from test_source_input import CLAUDE_RECORD, CLI_START, CODEX_HEADER, PI_HEADER, VSCODE_V3, VSCODE_V3_LOG

FIXTURE = Path(__file__).parent / 'fixtures/pi/source/ordinary.jsonl'


def test_five_source_reporting_survives_reimport_restart_and_saved_history(tmp_path):
    from dataclasses import replace
    from decimal import Decimal
    from fastapi.testclient import TestClient
    from harness_usage.application import Application
    from harness_usage.web import create_app
    from harness_usage.reporting import build_report
    from test_aggregate_reporting import catalog
    fixtures = FIXTURE.parents[2]
    root = tmp_path / 'history'
    root.mkdir()
    (root / 'pi.jsonl').write_bytes(FIXTURE.read_bytes())
    (root / 'codex.jsonl').write_bytes((fixtures / 'codex/mixed.jsonl').read_bytes())
    app = Application(tmp_path / 'data', timezone='UTC')
    app.catalog = catalog()
    app.set_roots((str(root),))
    app.start_import(); app.close()
    query = ReportQuery(None, AllTime())
    before = app.report(query)
    assert before.tokens.total.known == 1090
    (root / '11111111-1111-4111-8111-111111111111.jsonl').write_bytes((fixtures / 'claude/versioned.jsonl').read_bytes())
    chats = root / 'User/workspaceStorage/key/chatSessions'
    chats.mkdir(parents=True)
    (chats.parent / 'workspace.json').write_text('{"folder":"file:///fixture/repo"}')
    (chats / 'session.jsonl').write_bytes((fixtures / 'copilot_vscode/session-v3.jsonl').read_bytes())
    for kind, identity in [('current', '11111111-1111-4111-8111-111111111111'), ('legacy', 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa')]:
        directory = root / 'session-state' / identity
        directory.mkdir(parents=True)
        (directory / 'workspace.yaml').write_bytes((fixtures / f'copilot_cli/{kind}/workspace.yaml').read_bytes())
        if kind == 'current':
            (directory / 'events.jsonl').write_bytes((fixtures / 'copilot_cli/current/events.jsonl').read_bytes())
    app.start_import(); app.close()
    assert app.status().state == 'succeeded'
    report = app.report(query)
    data = app.storage.report_input(query)
    assert report == build_report(data.contributions, revision=data.revision, query=query, catalog=app.catalog)
    assert {row.harness for row in report.sessions} == {'pi', 'codex', 'claude', 'copilot-vscode', 'copilot-cli'}
    assert tuple(row for row in report.models if row.harness in {'pi', 'codex'}) == before.models
    claude, = (row for row in report.sessions if row.harness == 'claude')
    assert (claude.tokens.output.known, claude.tokens.output.unknown_observations) == (37, 1)
    assert claude.money.known == 0 and claude.money.missing_observations == 3
    assert all(model.provider is None for model in claude.models)
    amounts = {(row.harness, row.measure): row.known for row in report.quantities}
    assert amounts == {('copilot-vscode', 'ai_credits'): Decimal('0.5'), ('copilot-cli', 'nano_aiu'): Decimal('200'),
                       ('copilot-cli', 'premium_requests'): Decimal('4'), ('copilot-cli', 'request_count'): Decimal('5')}
    page = TestClient(create_app(app), base_url='http://localhost').get('/?project=unassigned').text
    for label in ['Pi', 'Codex', 'Claude Code', 'Copilot in VS Code', 'Copilot CLI', 'Not billed spend']:
        assert label in page
    app.start_import(); app.close()
    assert app.report(query) == report
    reopened = Application(tmp_path / 'data', timezone='UTC')
    reopened.catalog = catalog()
    assert reopened.report(query) == report
    path = root / 'pi.jsonl'
    original = path.read_bytes()
    path.unlink()
    reopened.start_import(); reopened.close()
    missing = reopened.report(query)
    assert missing.tokens == report.tokens and missing.quantities == report.quantities
    assert 'saved_history' in {entry.code for entry in missing.coverage}
    path.write_bytes(original)
    reopened.start_import(); reopened.close()
    assert replace(reopened.report(query), revision=report.revision) == report
    legacy = next((row for row in report.sessions if row.id == 'copilot-cli:aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa'), None)
    assert legacy is not None
    assert 'usage_unavailable' in {entry.code for entry in legacy.coverage}


def test_scan_returns_only_structurally_recognized_sources_with_fixed_sidecars(tmp_path):
    from harness_usage.application import Application
    root = tmp_path / 'sources'
    root.mkdir()
    (root / 'pi.jsonl').write_bytes(PI_HEADER)
    (root / 'codex.jsonl').write_bytes(CODEX_HEADER)
    (root / 'claude.jsonl').write_bytes(CLAUDE_RECORD)
    (root / 'notes.jsonl').write_bytes(b'{"event":"other"}\n')
    (root / 'unrelated.json').write_text('{"version":3}')
    (root / 'unrelated.yaml').write_text('id: unrelated\ncwd: /wrong\n')

    workspace = root / 'workspace'
    chats = workspace / 'chatSessions'
    chats.mkdir(parents=True)
    workspace_sidecar = b'{"folder":"file:///work"}'
    (workspace / 'workspace.json').write_bytes(workspace_sidecar)
    (chats / 'session.json').write_bytes(VSCODE_V3)
    (chats / 'session.jsonl').write_bytes(VSCODE_V3_LOG)
    (chats / 'wrong.json').write_bytes(VSCODE_V3.replace(b'GitHub.copilot-chat', b'other.extension'))

    current = root / 'session-state' / 'current'
    current.mkdir(parents=True)
    cli_sidecar = b'id: current\ncwd: /work\n'
    (current / 'events.jsonl').write_bytes(CLI_START)
    (current / 'workspace.yaml').write_bytes(cli_sidecar)
    legacy = root / 'session-state' / 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa'
    legacy.mkdir()
    legacy_workspace = b'id: aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa\ncwd: /legacy\n'
    (legacy / 'workspace.yaml').write_bytes(legacy_workspace)

    outside = tmp_path / 'outside.jsonl'
    outside.write_bytes(PI_HEADER)
    (root / 'escape.jsonl').symlink_to(outside)

    app = Application(tmp_path / 'data')
    try:
        payloads = tuple(app._scan((str(root),)))
        locators = tuple(payload.locator for payload in payloads)
        assert locators == tuple(sorted((
            str(root / 'claude.jsonl'), str(root / 'codex.jsonl'), str(root / 'pi.jsonl'),
            str(chats / 'session.json'), str(chats / 'session.jsonl'),
            str(chats / 'wrong.json'),
            str(current / 'events.jsonl'), str(legacy / 'workspace.yaml'),
        )))
        assert all(isinstance(payload, SourcePayload) for payload in payloads)
        by_locator = {payload.locator: payload for payload in payloads}
        assert by_locator[str(chats / 'session.json')].context == (('workspace.json', workspace_sidecar),)
        assert by_locator[str(chats / 'session.jsonl')].context == (('workspace.json', workspace_sidecar),)
        assert by_locator[str(current / 'events.jsonl')].context == (('workspace.yaml', cli_sidecar),)
        metadata = by_locator[str(legacy / 'workspace.yaml')]
        assert metadata.context == () and detect_source(metadata) == 'copilot-cli'
        before = by_locator[str(chats / 'session.json')].fingerprint()
        (workspace / 'workspace.json').write_bytes(b'{"folder":"file:///changed"}')
        changed = {payload.locator: payload for payload in app._scan((str(root),))}
        assert changed[str(chats / 'session.json')].fingerprint() != before
    finally:
        app.close()


def test_scan_does_not_read_vscode_sidecar_outside_configured_root(tmp_path):
    from harness_usage.application import Application
    workspace = tmp_path / 'workspace'
    chats = workspace / 'chatSessions'
    chats.mkdir(parents=True)
    (workspace / 'workspace.json').write_text('{"folder":"outside-authority"}')
    source = chats / 'session.json'
    source.write_bytes(VSCODE_V3)
    app = Application(tmp_path / 'data')
    try:
        payload, = app._scan((str(chats),))
        assert payload.locator == str(source)
        assert payload.context == ()
    finally:
        app.close()


def test_scan_yields_empty_initial_vscode_log_for_source_specific_reader(tmp_path):
    from harness_usage.application import Application
    chats = tmp_path / 'User/workspaceStorage/key/chatSessions'
    chats.mkdir(parents=True)
    source = chats / 'session.jsonl'
    source.write_bytes(b'{"kind":0,"v":{"version":3,"sessionId":"vs-session","requests":[]}}\n')
    app = Application(tmp_path / 'data')
    try:
        payload, = app._scan((str(tmp_path / 'User'),))
        assert payload.locator == str(source)
        assert detect_source(payload) == 'copilot-vscode'
    finally:
        app.close()


def test_scan_accepts_sidecar_at_probe_limit(tmp_path):
    from harness_usage.application import Application
    workspace = tmp_path / 'workspace'
    chats = workspace / 'chatSessions'
    chats.mkdir(parents=True)
    sidecar = b'x' * MAX_PROBE_BYTES
    (workspace / 'workspace.json').write_bytes(sidecar)
    (chats / 'session.json').write_bytes(VSCODE_V3)
    app = Application(tmp_path / 'data')
    try:
        payload, = app._scan((str(workspace),))
        assert payload.context == (('workspace.json', sidecar),)
    finally:
        app.close()


def test_scan_rejects_sidecar_over_probe_limit_with_fixed_error(tmp_path):
    from harness_usage.application import Application, SourceFailure
    workspace = tmp_path / 'workspace'
    chats = workspace / 'chatSessions'
    chats.mkdir(parents=True)
    (workspace / 'workspace.json').write_bytes(b'x' * (MAX_PROBE_BYTES + 1))
    (chats / 'session.json').write_bytes(VSCODE_V3)
    app = Application(tmp_path / 'data')
    try:
        with pytest.raises(SourceFailure) as failure:
            tuple(app._scan((str(workspace),)))
        assert failure.value.code == 'source_1_sidecar_too_large'
    finally:
        app.close()


def test_import_join_retention_and_restart(tmp_path):
    from harness_usage.application import Application
    root = tmp_path / 'sources'; root.mkdir()
    source = root / 'session.jsonl'; source.write_bytes(FIXTURE.read_bytes())
    app = Application(tmp_path / 'data', timezone='Europe/Kyiv')
    app.set_roots((str(root),))
    entered, proceed = Event(), Event()
    original = app._scan
    def blocked(roots):
        entered.set(); assert proceed.wait(5)
        return original(roots)
    app._scan = blocked
    started = app.start_import()
    assert entered.wait(5)
    assert app.start_import().run_id == started.run_id
    assert app.report(ReportQuery(None, AllTime())).tokens.total.known == 0
    proceed.set(); app.close()
    assert app.status().state == 'succeeded'
    assert app.report(ReportQuery(None, AllTime())).tokens.total.known == 970
    source.unlink()
    app.start_import(); app.close()
    assert app.report(ReportQuery(None, AllTime())).tokens.total.known == 970
    restarted = Application(tmp_path / 'data', timezone='UTC')
    assert restarted.get_roots() == (str(root),)
    assert restarted.status().state == 'succeeded'
    restarted.close()


def test_failure_keeps_prior_commit_and_retry(tmp_path):
    from harness_usage.application import Application
    root = tmp_path / 'sources'; root.mkdir()
    (root / 's.jsonl').write_bytes(FIXTURE.read_bytes())
    app = Application(tmp_path / 'data')
    app.set_roots((str(root),))
    app.start_import(); app.close()
    original = app._scan
    def fail(roots):
        raise OSError('PRIVATE_SENTINEL')
    app._scan = fail
    app.start_import(); app.close()
    assert app.status().state == 'failed'
    assert 'PRIVATE_SENTINEL' not in str(app.status())
    assert app.report(ReportQuery(None, AllTime())).tokens.total.known == 970
    app._scan = original
    app.start_import(); app.close()
    assert app.status().state == 'succeeded'


def test_root_validation_and_symlink_escape(tmp_path):
    from harness_usage.application import Application
    root = tmp_path / 'sources'; root.mkdir()
    outside = tmp_path / 'outside.jsonl'; outside.write_bytes(FIXTURE.read_bytes())
    (root / 'escape.jsonl').symlink_to(outside)
    app = Application(tmp_path / 'data')
    with pytest.raises(ValueError): app.set_roots(('relative',))
    app.set_roots((str(root),))
    app.start_import(); app.close()
    assert app.status().files_processed == 0


def test_interrupted_run_is_recoverable(tmp_path):
    from harness_usage.application import Application
    data = tmp_path / 'data'
    app = Application(data); app.close()
    with app.storage.connect() as db:
        db.execute("INSERT INTO import_run(id,state,started_us,finished_us,files_processed,revision,error_code) VALUES('crashed','running',0,NULL,2,0,NULL)")
    restarted = Application(data)
    assert restarted.status().state == 'interrupted'
    restarted.start_import(); restarted.close()
    assert restarted.status().state == 'succeeded'


def test_rejected_source_diagnostic_survives_without_observations(tmp_path):
    from harness_usage.application import Application
    root = tmp_path / 'sources'; root.mkdir()
    (root / 'invalid.jsonl').write_text('{"type":"session","version":999}\n')
    app = Application(tmp_path / 'data')
    app.set_roots((str(root),)); app.start_import(); app.close()
    report = app.report(ReportQuery(None, AllTime()))
    assert not report.sessions
    assert report.coverage == ()
    assert app.storage.report_input(ReportQuery(None, AllTime())).diagnostics == ()
    assert any('header' in code for code in app.storage.snapshot().diagnostics)


def test_failure_after_batch_assigns_committed_sessions(tmp_path):
    from harness_usage.application import Application
    import json
    root = tmp_path / 'sources'; root.mkdir()
    rows = [json.loads(line) for line in FIXTURE.read_text().splitlines()]
    rows[0]['cwd'] = str(tmp_path)
    for index in range(33):
        (root / f'{index:02}.jsonl').write_text('\n'.join(map(json.dumps, rows)))
    app = Application(tmp_path / 'data')
    app.set_roots((str(root),))
    original = app.storage.import_sources
    calls = 0
    def failing_batch(sources):
        nonlocal calls
        calls += 1
        if calls == 2: raise OSError('unreadable')
        return original(sources)
    app.storage.import_sources = failing_batch
    app.start_import(); app.close()
    report = app.report(ReportQuery(None, AllTime()))
    assert app.status().state == 'failed'
    assert report.tokens.total.known == 970
    assert report.projects[0].id == 'directory:' + str(tmp_path)


def test_multiple_sources_identify_missing_root_and_retry(tmp_path):
    from harness_usage.application import Application
    first = tmp_path / 'first'; first.mkdir()
    (first / 'session.jsonl').write_bytes(FIXTURE.read_bytes())
    second = tmp_path / 'missing'
    app = Application(tmp_path / 'data')
    app.set_roots((str(first),))
    app.start_import(); app.close()
    app.set_roots((str(first), str(second)))
    app.start_import(); app.close()
    assert app.status().state == 'failed'
    assert app.status().error == 'source_2_unavailable'
    assert app.report(ReportQuery(None, AllTime())).tokens.total.known == 970
    second.mkdir()
    app.start_import(); app.close()
    assert app.status().state == 'succeeded'
    assert app.status().error is None


def test_read_failure_names_source_without_exposing_exception(tmp_path, monkeypatch):
    from harness_usage.application import Application
    first = tmp_path / 'first'; first.mkdir()
    second = tmp_path / 'second'; second.mkdir()
    (first / 'session.jsonl').write_bytes(FIXTURE.read_bytes())
    blocked = second / 'blocked.jsonl'; blocked.write_bytes(FIXTURE.read_bytes())
    app = Application(tmp_path / 'data')
    app.set_roots((str(first),))
    app.start_import(); app.close()
    app.set_roots((str(first), str(second)))
    original = Path.read_bytes
    def fail_one(path):
        if path == blocked:
            raise OSError('PRIVATE_TRANSCRIPT_SENTINEL')
        return original(path)
    monkeypatch.setattr(Path, 'read_bytes', fail_one)
    app.start_import(); app.close()
    assert app.status().state == 'failed'
    assert app.status().error == 'source_2_read_failed'
    assert 'PRIVATE_TRANSCRIPT_SENTINEL' not in str(app.status())
    assert app.report(ReportQuery(None, AllTime())).tokens.total.known == 970


def test_sources_cannot_change_until_current_import_finishes(tmp_path):
    from harness_usage.application import Application, SourceChangeDuringImport
    import json
    first = tmp_path / 'first'; first.mkdir()
    second = tmp_path / 'second'; second.mkdir()
    data = FIXTURE.read_bytes()
    (first / 'session.jsonl').write_bytes(data)
    native_id = json.loads(data.splitlines()[0])['id'].encode()
    (second / 'session.jsonl').write_bytes(data.replace(native_id, b'new-source-session'))
    app = Application(tmp_path / 'data')
    app.set_roots((str(first),))
    entered, proceed = Event(), Event()
    original = app._scan
    def blocked(roots):
        entered.set()
        assert proceed.wait(5)
        return original(roots)
    app._scan = blocked
    app.start_import()
    try:
        assert entered.wait(5)
        with pytest.raises(SourceChangeDuringImport):
            app.set_roots((str(second),))
        assert app.get_roots() == (str(first),)
        app.set_roots((str(first),))
    finally:
        proceed.set()
        app.close()
    app._scan = original
    app.set_roots((str(second),))
    app.start_import(); app.close()
    assert app.status().state == 'succeeded'
    assert app.get_roots() == (str(second),)
    assert app.report(ReportQuery(None, AllTime())).tokens.total.known == 1940


def test_codex_titles_read_latest_native_name_from_configured_root_only(tmp_path):
    import json
    from harness_usage.application import codex_titles
    home = tmp_path / 'codex'
    sessions = home / 'sessions'
    archive = home / 'archived_sessions'
    sessions.mkdir(parents=True)
    archive.mkdir()
    entries = [
        {'id': 'one', 'thread_name': 'Old name', 'updated_at': '2026-09-11T12:00:00Z'},
        {'id': 'one', 'thread_name': '  Native\n name  ', 'updated_at': '2026-09-12T12:00:00Z'},
        {'id': 'one', 'thread_name': 'Older copy', 'updated_at': '2026-09-10T12:00:00Z'},
        {'id': 'bad', 'thread_name': ['invalid'], 'updated_at': '2026-09-12T12:00:00Z'},
        {'id': 'long', 'thread_name': 'x'*200, 'updated_at': '2026-09-12T12:00:00Z'},
    ]
    (home / 'session_index.jsonl').write_text('\n'.join(json.dumps(row) for row in entries) + '\n{"incomplete":')
    assert codex_titles((str(sessions), str(archive))) == {'codex:one': 'Native name', 'codex:long': 'x'*159+'…'}
    assert codex_titles((str(tmp_path),)) == {}


def test_mixed_sources_archive_copy_native_title_and_catalog_price(tmp_path):
    from harness_usage.application import Application
    import json
    from decimal import Decimal
    from harness_usage.reporting import ReportQuery, AllTime
    project = tmp_path / 'project'
    project.mkdir()
    pi_root = tmp_path / 'pi'
    codex_root = tmp_path / 'codex' / 'sessions'
    archive = codex_root.parent / 'archived_sessions'
    for directory in (pi_root,codex_root,archive):
        directory.mkdir(parents=True)
    pi_lines = (Path(__file__).parent / 'fixtures/pi/source/ordinary.jsonl').read_text().splitlines()
    header = json.loads(pi_lines[0]); header['cwd'] = str(project)
    (pi_root / 'session.jsonl').write_text(json.dumps(header)+'\n'+pi_lines[1]+'\n')
    raw = [
        {'type':'session_meta','timestamp':'2026-09-12T08:00:00Z','payload':{'id':'codex-main','timestamp':'2026-09-12T08:00:00Z','cwd':str(project),'model_provider':'openai'}},
        {'type':'turn_context','timestamp':'2026-09-12T09:00:00Z','payload':{'turn_id':'turn','model':'gpt-5.6-sol'}},
        {'type':'token_usage_record','timestamp':'2026-09-12T09:01:00Z','payload':{'thread_id':'codex-main','turn_id':'turn','response_id':'response','usage':{'input_tokens':100,'cached_input_tokens':30,'cache_write_input_tokens':10,'output_tokens':20,'reasoning_output_tokens':5,'total_tokens':120}}},
    ]
    data='\n'.join(json.dumps(row) for row in raw)+'\n'
    (codex_root/'rollout.jsonl').write_text(data)
    (archive/'copy.jsonl').write_text(data)
    (codex_root.parent/'session_index.jsonl').write_text(json.dumps({'id':'codex-main','thread_name':'Native Codex name','updated_at':'2026-09-12T09:02:00Z'})+'\n')
    backend=Application(tmp_path/'data',timezone='UTC')
    backend.set_roots(tuple(str(p) for p in (pi_root,codex_root,archive)))
    backend.start_import();backend.close()
    assert backend.status().state=='succeeded'
    report=backend.report(ReportQuery(None,AllTime()))
    assert report.total_session_count==2 and len(report.projects)==1
    assert report.tokens.total.known==1090
    assert report.money.known==Decimal('0.000702')
    assert next(row.name for row in report.sessions if row.id=='codex:codex-main')=='Native Codex name'
    backend.start_import();backend.close()
    again=backend.report(ReportQuery(None,AllTime()))
    assert (again.tokens,again.money,again.sessions)==(report.tokens,report.money,report.sessions)
