"""Process safety for the local runtime, independent of the development cwd."""
import pytest
import subprocess
import re
import sys
from pathlib import Path
import hashlib
import json
import os


def stage_runtime_sources(root):
    """Curated synthetic corpus, also callable via runpy for later preview staging.

    Task 5/6 expectations are provisional until their reviewed contracts freeze.
    Expected amounts below are hand-counted from the named fixture bytes, never
    obtained from production readers. The paired VS Code forms count once.
    """
    root.mkdir(parents=True, exist_ok=False)
    fixtures = Path(__file__).parent / 'fixtures'
    current = '11111111-1111-4111-8111-111111111111'
    legacy = 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa'
    workspace = 'vscode/User/workspaceStorage/fixture-workspace'
    copies = {
        'pi/ordinary.jsonl': 'pi/source/ordinary.jsonl',
        'codex/mixed.jsonl': 'codex/mixed.jsonl',
        f'claude/{current}.jsonl': 'claude/main.jsonl',
        f'{workspace}/chatSessions/session-v3.json': 'copilot_vscode/session-v3.json',
        f'{workspace}/chatSessions/session-v3.jsonl': 'copilot_vscode/session-v3.jsonl',
        f'copilot/session-state/{current}/events.jsonl': 'copilot_cli/current/events.jsonl',
        f'copilot/session-state/{current}/workspace.yaml': 'copilot_cli/current/workspace.yaml',
        f'copilot/session-state/{legacy}/workspace.yaml': 'copilot_cli/legacy/workspace.yaml',
    }
    for target, source in copies.items():
        path = root / target
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((fixtures / source).read_bytes())
    assert json.loads((root / f'claude/{current}.jsonl').read_text().splitlines()[0])['sessionId'] == current
    assert json.loads((root / f'copilot/session-state/{current}/events.jsonl').read_text().splitlines()[0])['data']['sessionId'] == current
    sidecar = root / workspace / 'workspace.json'
    sidecar.write_text('{"folder":"file:///fixture/vscode"}\n')
    conversations = {
        'pi/bundle-pi-transcript.jsonl': [
            {'type': 'session', 'version': 3, 'id': 'bundle-pi-transcript', 'timestamp': '2026-09-13T08:00:00Z', 'cwd': '/bundle'},
            {'type': 'message', 'id': 'pi-user', 'parentId': None, 'message': {'role': 'user', 'content': 'Bundled Pi transcript'}},
        ],
        'codex/bundle-codex-transcript.jsonl': [
            {'type': 'session_meta', 'timestamp': '2026-09-13T08:00:00Z', 'payload': {'id': 'bundle-codex-transcript', 'timestamp': '2026-09-13T08:00:00Z', 'cwd': '/bundle', 'history_mode': 'legacy'}},
            {'type': 'response_item', 'timestamp': '2026-09-13T08:00:01Z', 'payload': {'type': 'message', 'role': 'user', 'content': [{'type': 'input_text', 'text': 'Bundled Codex transcript'}]}},
        ],
    }
    for target, rows in conversations.items():
        (root / target).write_text(''.join(json.dumps(row) + '\n' for row in rows))
    # Independent implementation of the documented scoped identity wire rule.
    scope = hashlib.sha256(b'harness-usage/copilot-vscode-scope-v1\0' + os.fsencode((root / 'vscode/User').resolve())).hexdigest()
    sidecars = (sidecar, root / f'copilot/session-state/{current}/workspace.yaml')
    paths = tuple(root / target for target in (*copies, *conversations)) + (sidecar,)
    return {
        'primaries': tuple(path for path in paths if path not in sidecars),
        'sidecars': sidecars,
        'hashes': {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        'transcripts': {
            'pi:bundle-pi-transcript': 'Bundled Pi transcript',
            'codex:bundle-codex-transcript': 'Bundled Codex transcript',
            f'claude:{current}': 'CANARY_RESPONSE_9d2c',
            f'copilot-vscode:{scope}:vs-session': 'Explain &lt;b&gt;safe&lt;/b&gt;',
            f'copilot-cli:{current}': 'SYNTHETIC_USER',
        },
        'unavailable': f'copilot-cli:{legacy}',
        # CLI has the root, durable agent-child, and metadata-only legacy row.
        # Reporting rolls up the child and omits the two transcript-only rows.
        'sessions': {'pi': 2, 'codex': 2, 'claude': 1, 'copilot-vscode': 1, 'copilot-cli': 3},
        'report_session_count': 6,
        # Disjoint input, output, cache read, cache write, total known amounts.
        # Zero known sums are paired with unknown counts, never called known zero.
        'tokens': {'pi': (100, 20, 800, 50, 970), 'codex': (60, 20, 30, 10, 120),
                   'claude': (10, 0, 4, 3, 0), 'copilot-vscode': (8, 15, 5, 0, 0),
                   'copilot-cli': (150, 65, 60, 10, 175)},
        'unknown_tokens': {'pi': (0, 0, 0, 0, 0), 'codex': (1, 1, 1, 1, 1),
                           'claude': (0, 2, 0, 0, 2), 'copilot-vscode': (2, 0, 1, 3, 3),
                           'copilot-cli': (2, 1, 1, 1, 2)},
        'money': {'pi': ('0', 1), 'codex': ('0.000702', 1), 'claude': ('0', 2),
                  'copilot-vscode': ('0', 3), 'copilot-cli': ('0', 3)},
        'quantities': {('copilot-vscode', 'ai_credits'): ('0.5', 1, 0, 0),
                       ('copilot-cli', 'nano_aiu'): ('200', 1, 1, 0),
                       ('copilot-cli', 'premium_requests'): ('4', 1, 1, 0),
                       ('copilot-cli', 'request_count'): ('5', 2, 1, 0)},
    }


def test_runtime_staging_manifest(tmp_path):
    manifest = stage_runtime_sources(tmp_path / 'sources')
    assert len(manifest['primaries']) == 9
    assert len(manifest['sidecars']) == 2
    assert len(manifest['hashes']) == 11
    assert all(path.is_file() for path in manifest['primaries'])
    assert all(hashlib.sha256(path.read_bytes()).hexdigest() == value
               for path, value in manifest['hashes'].items())
    assert set(manifest['sessions']) == {'pi', 'codex', 'claude', 'copilot-vscode', 'copilot-cli'}


def assert_runtime_accounting(report, manifest, load_count=0, *, require_final_fields=False):
    """Independent fixture oracle; provisional Task 5/6 amounts/counts."""
    from decimal import Decimal
    measures = ('input', 'output', 'cache_read', 'cache_write', 'total')
    for harness, expected in manifest['tokens'].items():
        rows = [row for row in report.sessions if row.harness == harness]
        multiplier = 1 + load_count if harness == 'pi' else 1
        assert tuple(sum(getattr(row.tokens, name).known for row in rows) for name in measures) == tuple(value * multiplier for value in expected), harness
        assert tuple(sum(getattr(row.tokens, name).unknown_observations for row in rows) for name in measures) == manifest['unknown_tokens'][harness], harness
        assert all(getattr(row.tokens, name).not_applicable_observations == 0 for row in rows for name in measures)
        amount, unpriced = manifest['money'][harness]
        assert sum((row.money.known for row in rows), Decimal(0)) == Decimal(amount)
        assert sum(row.money.missing_observations for row in rows) == unpriced * multiplier
    quantities = {(row.harness, row.measure): (row.known, row.known_observations, row.unknown_observations, row.lower_bound_observations) for row in report.quantities}
    assert quantities == {key: (Decimal(value[0]), *value[1:]) for key, value in manifest['quantities'].items()}
    for row in report.quantities:
        # This corpus has selected known/unknown quantities, no unresolved or
        # not-applicable quantities. Task 6's separate state matrix covers those.
        for field in ('unresolved_observations', 'not_applicable_observations'):
            if require_final_fields and field == 'unresolved_observations':
                assert hasattr(row, field), f'Final quantity report lacks {field}'
            if hasattr(row, field):
                assert getattr(row, field) == 0


def assert_runtime_quantity_html(page):
    """Require the final Task 6 table; absent UI must fail artifact acceptance."""
    from html import unescape
    def text(value):
        return ' '.join(unescape(re.sub(r'<[^>]*>', ' ', value)).split())
    assert 'Source quantities' in text(page)
    assert 'Not billed spend' in text(page)
    tables = [table for table in re.findall(r'<table\b[^>]*>.*?</table>', page, re.S)
              if [text(cell) for cell in re.findall(r'<th\b[^>]*>(.*?)</th>', table, re.S)]
              == ['Source', 'Measure', 'Amount', 'Unit', 'Status']]
    assert len(tables) == 1, 'Expected one separate source-quantity table'
    rows = [[text(cell) for cell in re.findall(r'<td\b[^>]*>(.*?)</td>', row, re.S)]
            for row in re.findall(r'<tr\b[^>]*>(.*?)</tr>', tables[0], re.S)]
    rows = [row for row in rows if row]
    expected = {
        ('Copilot in VS Code', 'AI credits'): ('AI credits', '0.5', False),
        ('Copilot CLI', 'nano-AIU'): ('nano-AIU', '200', True),
        ('Copilot CLI', 'premium requests'): ('Premium requests', '4', True),
        ('Copilot CLI', 'requests'): ('Request count', '5', True),
    }
    assert len(rows) == len(expected)
    assert {(row[0], row[3]) for row in rows} == set(expected)
    for row in rows:
        measure, amount, unknown = expected[row[0], row[3]]
        assert row[1] == measure
        assert row[2] == amount
        assert 'Recorded total' in row[4]
        assert ('Unavailable' in row[4]) == unknown
        assert not any(word in row[4] for word in ('Unresolved', 'Not applicable', 'Lower bound'))
        assert not any(value in ' '.join(row) for value in ('$', 'USD', 'Mtok'))


def test_runtime_source_oracle(tmp_path):
    """Exercise synthetic discovery/duplicate handling without a bundle/server."""
    from harness_usage.application import Application
    from harness_usage.aggregate_reporting import build_aggregate_report
    from harness_usage.pricing import load_bundled_catalog
    from harness_usage.reporting import AllTime, ReportQuery
    root = tmp_path / 'sources'
    manifest = stage_runtime_sources(root)
    app = Application(tmp_path / 'data')
    try:
        sources = tuple(app._scan((str(root),)))
        assert len(sources) == len(manifest['primaries'])
        app.storage.import_sources(sources)
        with app.storage.connect() as db:
            assert dict(db.execute('SELECT harness,count(*) FROM session GROUP BY harness')) == manifest['sessions']
        report = build_aggregate_report(app.storage, ReportQuery(None, AllTime()), load_bundled_catalog())
        assert report.total_session_count == manifest['report_session_count']
        assert_runtime_accounting(report, manifest)
    finally:
        app.close()


def test_cli_help_works_from_other_directory(tmp_path):
    result = subprocess.run([sys.executable, '-m', 'harness_usage', '--help'], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert '--data-dir' in result.stdout
    assert '--root' in result.stdout
    assert '--timezone' in result.stdout
    help_text = ' '.join(result.stdout.split())
    for product in ('Pi', 'Codex', 'Claude Code', 'Copilot in VS Code', 'Copilot CLI'):
        assert product in help_text
    assert 'session-storage directories' in help_text


def test_data_directory_lock_excludes_second_process(tmp_path):
    from harness_usage.__main__ import data_lock
    with data_lock(tmp_path):
        code = 'from pathlib import Path; from harness_usage.__main__ import data_lock;\nwith data_lock(Path(' + repr(str(tmp_path)) + ')): pass'
        result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    assert result.returncode != 0
    assert 'already running' in result.stderr
    with data_lock(tmp_path):
        pass


@pytest.mark.parametrize('legacy', [False, True])
def test_offline_bundle_import_restart_and_local_assets(tmp_path, legacy):
    """Opt-in artifact acceptance with outbound network denied and an empty PATH."""
    import json
    import http.client
    import urllib.parse
    import os
    import socket
    import time
    import urllib.request
    import pytest
    executable = os.environ.get('HARNESS_USAGE_BUNDLE')
    if not executable:
        pytest.skip('Set HARNESS_USAGE_BUNDLE to the built executable for offline acceptance')
    root = tmp_path / 'sources'
    manifest = stage_runtime_sources(root)
    source_before = (root / 'pi/ordinary.jsonl').read_bytes()
    if legacy:
        from harness_usage.storage import Storage
        from legacy_fixture import export_legacy
        seed = Storage(tmp_path / 'seed.duckdb')
        seed.import_source('/legacy', source_before)
        (tmp_path / 'data').mkdir()
        export_legacy(seed, tmp_path / 'data' / 'ledger.sqlite3')
        seed.close()

    assert Path(executable).is_absolute() and Path(executable).is_file()
    artifact_hash = hashlib.sha256(Path(executable).read_bytes()).hexdigest()
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        port = probe.getsockname()[1]
    base = f'http://127.0.0.1:{port}'
    environment = {key: os.environ[key] for key in ('TMPDIR', 'LANG', 'LC_ALL', 'LC_CTYPE') if key in os.environ}
    environment.update(PATH='', PYTHONPATH='', PYTHONHOME='', HOME=str(tmp_path))
    profile = '(version 1)(allow default)(deny network-outbound)(allow network-outbound (remote ip "localhost:*"))'
    assert Path('/usr/bin/sandbox-exec').is_file(), 'Offline acceptance requires sandbox-exec'
    control = subprocess.run(['/usr/bin/sandbox-exec', '-p', profile, sys.executable, '-I', '-c',
        'import errno,socket\ns=socket.socket();s.settimeout(2)\ntry:s.connect(("192.0.2.1",9))\nexcept OSError as e: assert e.errno in (errno.EPERM,errno.EACCES),e.errno\nelse:raise AssertionError("external connection allowed")'],
        cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=5)
    assert control.returncode == 0, 'Sandbox negative control did not prove denial: ' + control.stderr
    command = ['/usr/bin/sandbox-exec', '-p', profile, executable, '--no-browser', '--port', str(port), '--data-dir', str(tmp_path / 'data'), '--root', str(root), '--timezone', 'Europe/Kyiv']
    totals = []
    typed_totals = []
    for launch in range(2):
        with (tmp_path / 'runtime.log').open('w+') as log:
            process = subprocess.Popen(command, cwd=tmp_path, env=environment, stdout=log, stderr=log)
            shutdown_stream = None
            try:
                deadline = time.monotonic() + 30
                state = {}
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        log.seek(0)
                        pytest.fail(log.read())
                    try:
                        with urllib.request.urlopen(base + '/events', timeout=1) as response:
                            response.readline()
                            state = json.loads(response.readline().decode().removeprefix('data: '))
                        if state['state'] in ('succeeded', 'failed'):
                            break
                    except (OSError, ValueError):
                        pass
                    time.sleep(.1)
                assert state['state'] == 'succeeded'
                with urllib.request.urlopen(base, timeout=5) as response:
                    page = response.read().decode()
                if launch == 0:
                    # A slow, unread event stream must not obstruct the importer.
                    slow_reader = urllib.request.urlopen(base + '/events', timeout=5)
                    for index in range(192):
                        copied = source_before.replace(b'74811efe-ab6b-5971-88ac-bce0d809fe49', f'load-session-{index}'.encode())
                        load_path = root / f'load-{index}.jsonl'
                        load_path.write_bytes(copied)
                        manifest['hashes'][load_path] = hashlib.sha256(copied).hexdigest()
                    token = re.search(r'name="csrf" value="([^"]+)"', page)[1]
                    connection = http.client.HTTPConnection('127.0.0.1', port, timeout=5)
                    connection.request('POST', '/import', urllib.parse.urlencode({'csrf': token}), {'Content-Type': 'application/x-www-form-urlencoded'})
                    response = connection.getresponse()
                    assert response.status == 303
                    response.read()
                    connection.close()
                    with urllib.request.urlopen(base + '/events', timeout=5) as active:
                        active.readline()
                        running = json.loads(active.readline().decode().removeprefix('data: '))
                    assert running['state'] == 'running'
                    deadline = time.monotonic() + 30
                    while time.monotonic() < deadline:
                        # Reconnect ignores an arbitrary old event id: there is no replay dependency.
                        request = urllib.request.Request(base + '/events', headers={'Last-Event-ID': 'missing-old-event'})
                        with urllib.request.urlopen(request, timeout=5) as fresh:
                            assert fresh.readline() == b'event: status\n'
                            recovered = json.loads(fresh.readline().decode().removeprefix('data: '))
                        if recovered['state'] != 'running':
                            break
                        time.sleep(.1)
                    slow_reader.close()
                    assert recovered['state'] == 'succeeded'
                    assert recovered['revision'] > running['revision']
                    assert recovered['files_processed'] == len(manifest['primaries']) + 192
                    with urllib.request.urlopen(base, timeout=5) as response:
                        page = response.read().decode()
                assert 'PyInstaller' not in page
                assert 'No sessions in this range' not in page
                totals.append(re.findall(r'title="([\d,]+) tokens"', page))
                assert 'Codex / openai / gpt-5.6-sol' in page
                for label in ('Pi', 'Codex', 'Claude Code', 'Copilot in VS Code', 'Copilot CLI'):
                    assert label in page
                assert_runtime_quantity_html(page)
                # The legacy session is undated and may be on the final page
                # after the 192-file stress import. Check its own row, not an
                # unrelated global coverage label or the transcript error page.
                session_pages = []
                for number in range(1, (manifest['report_session_count'] + 192 + 49) // 50 + 1):
                    with urllib.request.urlopen(base + f'/?project=unassigned&page={number}', timeout=5) as response:
                        assert response.status == 200
                        session_pages.append(response.read().decode())
                unavailable_url = '/sessions/' + urllib.parse.quote(manifest['unavailable'], safe='') + '/transcript'
                legacy_rows = [row for content in session_pages
                               for row in re.findall(r'<tr\b[^>]*>.*?</tr>', content, re.S)
                               if unavailable_url in row]
                assert len(legacy_rows) == 1
                assert 'Copilot CLI' in legacy_rows[0]
                assert 'Usage unavailable' in legacy_rows[0]
                for asset in ['app.js', 'app.css', 'vendor/unpoly.min.js', 'vendor/unpoly.min.css', 'vendor/NotoSans.ttf']:
                    with urllib.request.urlopen(base + '/static/' + asset, timeout=5) as response:
                        assert response.status == 200
                        assert response.read()
                for asset in ['transcript.js', 'transcript.css', 'vendor/NotoSansMono.ttf', 'vendor/SymbolsNerdFontMono-subset.woff']:
                    with urllib.request.urlopen(base + '/static/' + asset, timeout=5) as response:
                        assert response.status == 200
                        assert response.read()
                for session_id, expected_text in manifest['transcripts'].items():
                    path = '/sessions/' + urllib.parse.quote(session_id, safe='')
                    with urllib.request.urlopen(base + path + '/transcript', timeout=5) as response:
                        transcript = response.read().decode()
                        assert response.status == 200
                        assert "script-src 'self'" in response.headers['Content-Security-Policy']
                    assert expected_text in transcript
                    assert '<script>alert(1)</script>' not in transcript
                    assert 'href="/static/transcript.css"' in transcript
                    assert 'src="/static/transcript.js"' in transcript
                    with urllib.request.urlopen(base + path + '/transcript.html', timeout=5) as response:
                        download = response.read().decode()
                        assert response.status == 200
                        assert response.headers['Content-Disposition'].startswith('attachment;')
                        assert 'sha256-' in response.headers['Content-Security-Policy']
                    assert 'data:font/ttf;base64,' in download
                    assert 'data:font/woff;base64,' in download
                    assert '/static/transcript.' not in download
                    assert not re.search(r'(?:src|href)=["\']https?://', download)
                with urllib.request.urlopen(base + '/sources', timeout=5) as response:
                    assert response.status == 200
                    assert b'Sources' in response.read()
                from urllib.error import HTTPError
                unavailable_path = '/sessions/' + urllib.parse.quote(manifest['unavailable'], safe='')
                with pytest.raises(HTTPError) as unavailable:
                    urllib.request.urlopen(base + unavailable_path + '/transcript', timeout=5)
                assert unavailable.value.code == 422
                assert b'Usage unavailable for this Copilot CLI source.' in unavailable.value.read()
                with urllib.request.urlopen(base + '/?project=unassigned&preset=day', timeout=5) as response:
                    assert b'Sessions' in response.read()
                shutdown_stream = urllib.request.urlopen(base + '/events', timeout=5)
            finally:
                process.terminate()
                try:
                    process.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
                    raise
                finally:
                    if shutdown_stream is not None:
                        shutdown_stream.close()
        # The bundled app uses one native ledger, with no reporting projection.
        from harness_usage.storage import Storage
        from harness_usage.aggregate_reporting import build_aggregate_report
        from harness_usage.pricing import load_bundled_catalog
        from harness_usage.reporting import AllTime, ReportQuery
        ledger = Storage(tmp_path / 'data' / 'ledger.duckdb')
        query, catalog = ReportQuery(None, AllTime()), load_bundled_catalog()
        assert ledger.path.exists()
        assert not (tmp_path / 'data' / 'reports.duckdb').exists()
        report = build_aggregate_report(ledger, query, catalog)
        assert report.total_session_count == manifest['report_session_count'] + 192
        assert_runtime_accounting(report, manifest, load_count=192, require_final_fields=True)
        assert report.revision == ledger.snapshot().revision
        from dataclasses import asdict
        with ledger.connect() as db:
            assert db.execute('SELECT schema_version FROM ledger_meta').one()[0] == 6
            sessions = dict(db.execute('SELECT harness,count(*) FROM session GROUP BY harness'))
            expected_sessions = dict(manifest['sessions'])
            expected_sessions['pi'] += 192
            assert sessions == expected_sessions
            assert db.execute('SELECT count(*) FROM session WHERE id=?', (manifest['unavailable'],)).one()[0] == 1
        typed_totals.append((report.revision, sessions, asdict(report.tokens), asdict(report.money),
                             tuple(asdict(row) for row in report.quantities),
                             tuple((row.code, len(row.observation_ids)) for row in report.coverage)))
        ledger.close()
    assert totals[0] == totals[1] and totals[0]
    assert typed_totals[0] == typed_totals[1]
    assert hashlib.sha256(Path(executable).read_bytes()).hexdigest() == artifact_hash
    assert all(hashlib.sha256(path.read_bytes()).hexdigest() == before for path, before in manifest['hashes'].items())
