"""Synthetic-only privacy and restart checks for the reviewed research utility."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'docs/research/three-source-local-validation.py'
MARKER = 'PRIVATE_IDENTIFIER_TEXT_URL_cwd_918273'


def validator():
    assert SCRIPT.exists(), 'aggregate-only validator is not implemented'
    spec = importlib.util.spec_from_file_location('local_validation', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    import harness_usage
    assert Path(harness_usage.__file__).resolve().is_relative_to(ROOT / 'src')
    return module


def sources(tmp_path):
    tmp_path = tmp_path / MARKER
    root = tmp_path / 'Code/User/workspaceStorage'
    chat = root / MARKER / 'chatSessions'
    chat.mkdir(parents=True)
    raw = json.loads((ROOT / 'tests/fixtures/copilot_vscode/session-v3.json').read_text())
    raw.update(customTitle=MARKER, workingDirectory=MARKER, sessionId=MARKER)
    raw['requests'][0]['message']['text'] = MARKER
    raw['requests'][0]['requestId'] = MARKER
    (chat / (MARKER + '.json')).write_text(json.dumps(raw))
    (root / MARKER / 'workspace.json').write_text(json.dumps({'folder': MARKER}))
    cli = tmp_path / 'session-state'
    legacy = cli / 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa'
    legacy.mkdir(parents=True)
    (legacy / 'workspace.yaml').write_text('id: "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"\ncwd: "' + MARKER + '"\n')
    return {'vscode-stable': root, 'copilot-cli': cli}


def test_private_counts_restart_and_preservation(tmp_path):
    m = validator()
    roots = sources(tmp_path)
    target = tmp_path / 'output'
    target.mkdir(mode=0o700)
    (target / 'keep').write_text(MARKER)
    result = m.validate(target, roots)
    assert MARKER not in json.dumps(result)
    assert result['error'] is None
    assert result['qualification']['stable'] == 'passed'
    assert result['qualification']['restart'] == 'passed'
    assert result['qualification']['unchanged_revision'] == 'passed'
    assert result['qualification']['metadata_only_cli'] == 'passed'
    assert result['qualification']['file_census'] == 'passed'
    assert result['expected']['vscode-stable']['flat_requests'] == 2
    assert result['expected']['copilot-cli']['metadata_only_files'] == 1
    assert result['observed']['sessions'] == {'copilot-vscode': 1, 'copilot-cli': 1}
    assert result['observed_diagnostics']['status'] == 'complete'
    assert result['observed_diagnostics']['profiles']['vscode-stable']['flat']['complete_files'] == 1
    assert (target / 'keep').read_text() == MARKER
    assert list(target.iterdir()) == [target / 'keep']
    created = list(tmp_path.glob('output-*'))
    assert len(created) == 1
    for file in [created[0], *created[0].rglob('*')]:
        assert file.stat().st_mode & 0o077 == 0
        if file.is_file() and file.name not in ('ledger.duckdb', 'ledger.duckdb.wal'):
            assert MARKER.encode() not in file.read_bytes()
    m.check_output(result)


def test_missing_malformed_symlinks_and_unsafe_output(tmp_path):
    m = validator()
    roots = sources(tmp_path)
    missing = m.validate(tmp_path / 'absent-output', {'vscode-insiders': tmp_path / 'missing'})
    assert missing['roots']['vscode-insiders'] is False
    assert missing['qualification']['comparison'] == 'unavailable'
    assert missing['qualification']['restart'] == 'unavailable'
    assert missing['observed_diagnostics']['status'] == 'unavailable'
    roots['vscode-stable'].joinpath(MARKER, 'chatSessions', 'broken.json').write_text(MARKER)
    malformed = m.validate(tmp_path / 'malformed-output', roots)
    assert malformed['expected']['vscode-stable']['malformed_records'] == 1
    assert MARKER not in json.dumps(malformed)
    link = tmp_path / 'link'
    link.symlink_to(roots['vscode-stable'], target_is_directory=True)
    rejected = m.validate(tmp_path / 'link-output', {'vscode-stable': link})
    assert rejected['error'] == 'unsafe_source'
    roots['vscode-stable'].joinpath(MARKER, 'chatSessions', 'linked.json').symlink_to(tmp_path / MARKER)
    assert m.validate(tmp_path / 'child-link-output', roots)['error'] == 'unsafe_source'
    unsafe = tmp_path / 'unsafe'
    unsafe.mkdir(mode=0o755)
    assert m.validate(unsafe, roots)['error'] == 'unsafe_output'
    assert unsafe.stat().st_mode & 0o777 == 0o755


def test_change_detection_and_fixed_error_output(tmp_path, monkeypatch):
    m = validator()
    roots = sources(tmp_path)
    original = m.observe
    def changed(app):
        result = original(app)
        roots['vscode-stable'].joinpath(MARKER, 'workspace.json').write_text('{}')
        return result
    monkeypatch.setattr(m, 'observe', changed)
    result = m.validate(tmp_path / 'output', roots)
    assert result['qualification']['stable'] == 'inconclusive'
    assert result['qualification']['comparison'] == 'inconclusive'
    assert result['observed_diagnostics']['status'] == 'inconclusive'
    command = subprocess.run([sys.executable, str(SCRIPT), '--' + MARKER], capture_output=True, text=True)
    assert command.returncode == 2
    assert not command.stderr
    assert MARKER not in command.stdout
    assert json.loads(command.stdout)['error'] == 'invalid_arguments'


def test_raw_operation_and_current_cli_denominators(tmp_path):
    m = validator()
    roots = sources(tmp_path)
    current = roots['copilot-cli'] / '11111111-1111-4111-8111-111111111111'
    current.mkdir()
    shutil.copyfile(ROOT / 'tests/fixtures/copilot_cli/current/events.jsonl', current / 'events.jsonl')
    chat = roots['vscode-stable'] / MARKER / 'chatSessions'
    operations = [
        {'kind': 0, 'v': {'version': 3, 'sessionId': MARKER, 'requests': [], 'customTitle': MARKER}},
        {'kind': 2, 'k': ['requests'], 'v': [{'requestId': MARKER, 'promptTokens': 2, 'message': {'text': MARKER}}]},
        {'kind': 1, 'k': ['requests', 0, 'promptTokens'], 'v': 9},
        {'kind': 3, 'k': ['requests', 0, 'promptTokens']},
    ]
    (chat / 'operations.jsonl').write_text('\n'.join(map(json.dumps, operations)) + '\n')
    result = m.validate(tmp_path / 'output', roots)
    assert result['error'] is None
    assert result['expected']['vscode-stable']['operation_records'] == 4
    assert result['expected']['vscode-stable']['operation_fields']['promptTokens'] == 2
    assert result['expected']['copilot-cli']['cli_shutdown_records'] == 2
    assert result['expected']['copilot-cli']['cli_model_rows'] == 4
    assert result['expected']['copilot-cli']['cli_fields']['inputTokens'] == 4
    assert MARKER not in json.dumps(result)
    assert result['qualification']['file_census'] == 'passed'
    assert result['qualification']['comparison'] == 'not_compared'
    assert result['observed']['metadata_sessions'] == 1
    assert result['observed']['metadata_known_tokens'] == 0
    assert result['observed']['metadata_known_quantities'] == 0
    assert result['qualification']['metadata_only_cli'] == 'passed'


def test_output_schema_rejects_arbitrary_keys_and_values(tmp_path):
    import pytest
    m = validator()
    for mutate in (
        lambda r: r.update({MARKER: 1}),
        lambda r: r['versions'].update({'copilot-cli': MARKER}),
        lambda r: r['observed']['diagnostics'].update({MARKER: 1}),
        lambda r: r['qualification'].update({'stable': MARKER}),
    ):
        result = m.empty_result()
        mutate(result)
        with pytest.raises(m.ValidationError):
            m.check_output(result)
    target = tmp_path / 'file-output'
    target.write_text(MARKER)
    assert m.validate(target, {'copilot-cli': tmp_path / 'missing'})['error'] == 'unsafe_output'
    assert target.read_text() == MARKER
    link = tmp_path / 'linked-output'
    link.symlink_to(target)
    assert m.validate(link, {'copilot-cli': tmp_path / 'missing'})['error'] == 'unsafe_output'


def test_additions_deletions_and_unrelated_files(tmp_path, monkeypatch):
    m = validator()
    roots = sources(tmp_path)
    # This valid foreign-format file must never be opened by the bounded scan.
    (roots['copilot-cli'] / 'ignored.jsonl').write_text(MARKER)
    original = m.observe
    changed = False
    def mutate(app):
        nonlocal changed
        observed = original(app)
        if not changed:
            changed = True
            chat = roots['vscode-stable'] / MARKER / 'chatSessions'
            (chat / 'new.json').write_text('{}')
            (roots['vscode-stable'] / MARKER / 'workspace.json').unlink()
        return observed
    monkeypatch.setattr(m, 'observe', mutate)
    result = m.validate(tmp_path / 'output', roots)
    assert result['error'] is None
    assert result['expected']['copilot-cli']['files'] == 1
    assert result['qualification']['stable'] == 'inconclusive'
    assert result['qualification']['file_census'] == 'inconclusive'
    assert result['qualification']['unchanged_revision'] == 'inconclusive'
    assert MARKER not in json.dumps(result)


def test_fixed_exception_wrong_owner_and_nonregular_source(tmp_path, monkeypatch):
    m = validator()
    roots = sources(tmp_path)
    def fail(_app):
        raise ValueError(MARKER)
    monkeypatch.setattr(m, 'observe', fail)
    result = m.validate(tmp_path / 'error-output', roots)
    assert result['error'] == 'validation_failed'
    assert MARKER not in json.dumps(result)
    target = tmp_path / 'wrong-owner'
    target.mkdir(mode=0o700)
    actual_uid = os.getuid()
    monkeypatch.setattr(m.os, 'getuid', lambda: actual_uid + 1)
    assert m.validate(target, roots)['error'] == 'unsafe_output'
    monkeypatch.undo()
    os.mkfifo(roots['vscode-stable'] / MARKER / 'chatSessions/fifo.json')
    assert m.validate(tmp_path / 'fifo-output', roots)['error'] == 'unsafe_source'


def test_bounded_installed_versions_do_not_emit_arbitrary_metadata(tmp_path):
    m = validator()
    package = tmp_path / 'package.json'
    package.write_text(json.dumps({'version': '1.110.3', 'title': MARKER, 'path': MARKER}))
    result = m.validate(tmp_path / 'output', {'copilot-cli': tmp_path / 'missing'},
                        metadata_paths={'vscode-insiders': package})
    assert result['versions']['vscode-insiders'] == {'app': '1.110.3', 'extension': None}
    assert result['versions']['copilot-cli'] == {'cli': None}
    assert MARKER not in json.dumps(result)
    package.write_text(json.dumps({'version': '1.2.3-' + MARKER}))
    assert m.installed_versions({'vscode-insiders': package})['vscode-insiders']['app'] is None
    package.unlink()
    package.symlink_to(tmp_path / MARKER)
    assert m.installed_versions({'vscode-insiders': package})['vscode-insiders']['app'] is None


@pytest.mark.parametrize('size', [1048576, 1048577])
@pytest.mark.parametrize('stage', ['snapshot', 'import'])
def test_sidecar_same_limit_for_snapshot_and_import(tmp_path, size, stage):
    m = validator()
    roots = sources(tmp_path)
    sidecar = roots['vscode-stable'] / MARKER / 'workspace.json'
    sidecar.write_bytes(b'{}' + b' ' * (size - 2))
    census = {profile: m.empty_census() for profile in m.PROFILES}
    output = tmp_path / 'direct-output'
    output.mkdir(mode=0o700)
    app = m.controlled_application(output, roots)
    try:
        operation = (lambda: m.snapshot(roots, census=census)) if stage == 'snapshot' else (lambda: list(app._scan(())))
        if size == 1048576:
            result = operation()
            if stage == 'snapshot':
                assert census['vscode-stable']['sidecars'] == 1
            else:
                assert len(result) == 2
        else:
            with pytest.raises(m.ValidationError, match='^limit_exceeded$'):
                operation()
    finally:
        app.close()


@pytest.mark.parametrize('version, accepted', [
    ('1.137.0-insider', True), ('1.137.0', True), ('0.0.0', True),
    ('1.137.0-' + MARKER, False), ('1.137.0+' + MARKER, False),
    (' 1.137.0-insider', False), ('1.137.0-insider\n', False),
    ('v1.137.0-insider', False), ('01.137.0-insider', False),
    ('1.0137.0-insider', False), ('1.137.00-insider', False),
    ('1.137-insider', False), ('1.137.0.1-insider', False),
    ('1.137.0-INSIDER', False), ('1.137.0-insider+private', False),
])
def test_strict_insiders_app_version(tmp_path, version, accepted):
    m = validator()
    package = tmp_path / 'package.json'
    package.write_text(json.dumps({'version': version, 'description': MARKER}))
    versions = m.installed_versions({'vscode-insiders': package})
    assert versions['vscode-insiders']['app'] == (version if accepted else None)
    assert versions['vscode-insiders']['extension'] is None
    assert versions['copilot-cli']['cli'] is None
    result = m.empty_result()
    result['versions'] = versions
    m.check_output(result)
    assert MARKER not in json.dumps(result)


def test_observed_replay_retained_paths_and_participants(tmp_path):
    m = validator()
    chat = tmp_path / 'workspaceStorage' / MARKER / 'chatSessions'
    chat.mkdir(parents=True)
    recognized = {'extensionId': {'value': 'GitHub.Copilot-Chat'}}
    requests = [
        {'agent': recognized, 'promptTokens': 9, 'completionTokens': 4,
         'payload': {'promptTokens': 3, 'private': MARKER}},
        {'agent': {'extensionId': 'github.copilot-chat'}, 'copilotCredits': 2},
        {'agent': {'extensionId': {'value': MARKER}}, 'inputTokens': 8},
        {'payload': [{'outputTokens': 7, MARKER: MARKER}]},
    ]
    operations = [
        {'kind': 0, 'v': {'version': 3, 'requests': requests}},
        {'kind': 3, 'k': ['requests', 0, 'promptTokens']},
        {'kind': 1, 'k': ['requests', 0, 'completionTokens'], 'v': 12},
    ]
    (chat / 'operations.jsonl').write_text('\n'.join(map(json.dumps, operations)) + '\n')
    (chat / 'flat.json').write_text(json.dumps({'version': 3, 'requests': requests}))
    result = m.empty_result()
    m.snapshot({'vscode-stable': chat.parents[1]}, census=result['expected'],
               observed_diagnostics=result['observed_diagnostics'])
    d = result['observed_diagnostics']
    assert (d['oracle'], d['independence'], d['comparison'], d['status']) == (
        'production_replay', 'not_independent', 'not_compared', 'complete')
    operation = d['profiles']['vscode-stable']['operation_log']
    copilot = operation['participants']['recognized_copilot']
    other = operation['participants']['other_or_missing_participant']
    assert operation['complete_files'] == 1
    assert copilot['final_requests'] == 1
    assert copilot['direct_fields']['promptTokens'] == 0
    assert copilot['direct_fields']['completionTokens'] == 1
    assert copilot['nested_fields']['promptTokens'] == 1
    assert copilot['nested_fields']['completionTokens'] == 0
    assert other['final_requests'] == 3
    assert other['direct_fields']['copilotCredits'] == 1
    assert other['direct_fields']['inputTokens'] == 1
    assert other['direct_fields']['outputTokens'] == 0
    assert other['nested_fields']['outputTokens'] == 1
    flat = d['profiles']['vscode-stable']['flat']['participants']['recognized_copilot']
    assert flat['direct_fields']['promptTokens'] == 1
    assert flat['nested_fields']['promptTokens'] == 1
    assert result['expected']['vscode-stable']['operation_fields']['promptTokens'] == 2
    assert result['expected']['vscode-stable']['operation_fields']['completionTokens'] == 2
    assert MARKER not in json.dumps(result)
    m.check_output(result)


def test_observed_replay_pending_prefix_and_failures(tmp_path):
    m = validator()
    chat = tmp_path / 'workspaceStorage' / MARKER / 'chatSessions'
    chat.mkdir(parents=True)
    initial = {'kind': 0, 'v': {'version': 3, 'requests': [{'promptTokens': 3}]}}
    (chat / 'pending.jsonl').write_text(json.dumps(initial) + '\n{"kind":')
    (chat / 'broken.jsonl').write_text(json.dumps(initial) + '\n' + MARKER + '\n')
    (chat / 'broken.json').write_text(MARKER)
    result = m.empty_result()
    m.snapshot({'vscode-insiders': chat.parents[1]}, observed_diagnostics=result['observed_diagnostics'])
    d = result['observed_diagnostics']
    assert d['status'] == 'partial'
    operations = d['profiles']['vscode-insiders']['operation_log']
    assert (operations['complete_files'], operations['partial_files'], operations['replay_failed_files']) == (0, 1, 1)
    assert operations['participants']['other_or_missing_participant']['final_requests'] == 1
    assert operations['participants']['other_or_missing_participant']['direct_fields']['promptTokens'] == 1
    assert d['profiles']['vscode-insiders']['flat']['replay_failed_files'] == 1
    assert d['comparison'] == 'not_compared'
    assert MARKER not in json.dumps(result)
    m.check_output(result)


@pytest.mark.parametrize('key,value', [
    ('oracle', MARKER), ('independence', 'passed'), ('comparison', 'passed'),
    ('status', MARKER),
])
def test_observed_diagnostics_rejects_unapproved_labels(key, value):
    m = validator()
    result = m.empty_result()
    result['observed_diagnostics'][key] = value
    with pytest.raises(m.ValidationError):
        m.check_output(result)


@pytest.mark.parametrize('value', [True, -1, 1.5, MARKER, None])
def test_observed_diagnostics_rejects_noncounts_and_arbitrary_fields(value):
    m = validator()
    result = m.empty_result()
    counts = result['observed_diagnostics']['profiles']['vscode-stable']['flat']['participants']['recognized_copilot']['direct_fields']
    counts['promptTokens'] = value
    with pytest.raises(m.ValidationError):
        m.check_output(result)
    counts['promptTokens'] = 0
    counts[MARKER] = 1
    with pytest.raises(m.ValidationError):
        m.check_output(result)


def test_observed_diagnostics_schema_version_is_exact():
    m = validator()
    result = m.empty_result()
    assert result['schema'] == 2
    m.check_output(result)
    for version in (0, 1, 3, True, 2.0, '2', None):
        result['schema'] = version
        with pytest.raises(m.ValidationError):
            m.check_output(result)


def test_unexpected_replay_error_is_operational_failure(tmp_path, monkeypatch):
    import harness_usage.copilot_vscode_reader as reader
    m = validator()
    roots = sources(tmp_path)
    original = reader.replay_chat_v3
    calls = 0
    def unexpected(data, *, representation):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError(MARKER)
        return original(data, representation=representation)
    monkeypatch.setattr(reader, 'replay_chat_v3', unexpected)
    result = m.validate(tmp_path / 'output', roots)
    assert result['error'] == 'validation_failed'
    assert result['observed_diagnostics']['status'] == 'inconclusive'
    assert result['qualification']['stable'] != 'passed'
    assert result['observed_diagnostics']['profiles']['vscode-stable']['flat']['replay_failed_files'] == 0
    assert MARKER not in json.dumps(result)
    m.check_output(result)
