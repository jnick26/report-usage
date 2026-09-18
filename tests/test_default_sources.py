import json
import os
from pathlib import Path
import socket
import sys

import pytest
import uvicorn

from harness_usage import __main__ as cli
from harness_usage.application import Application


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    home = tmp_path / 'home'
    home.mkdir()
    monkeypatch.setenv('HOME', str(home))
    for name in ('PI_CODING_AGENT_SESSION_DIR', 'PI_CODING_AGENT_DIR', 'CODEX_HOME', 'CLAUDE_CONFIG_DIR', 'COPILOT_HOME'):
        monkeypatch.delenv(name, raising=False)
    return home


def standard_roots(home):
    return tuple(home / name for name in (
        '.pi/agent/sessions', '.codex/sessions', '.codex/archived_sessions',
        '.claude/projects', '.copilot',
        'Library/Application Support/Code/User/workspaceStorage',
        'Library/Application Support/Code - Insiders/User/workspaceStorage',
    ))


def run_cli(data_dir, monkeypatch, *extra):
    # Exercise CLI/config/import without opening a listener or serving forever.
    monkeypatch.setattr(socket.socket, 'bind', lambda self, address: None)
    monkeypatch.setattr(socket.socket, 'listen', lambda self, backlog: None)
    monkeypatch.setattr(sys, 'argv', ['harness-usage', '--data-dir', str(data_dir),
                                   '--port', '8765', '--timezone', 'UTC', '--no-browser', *extra])
    monkeypatch.setattr(uvicorn.Server, 'run', lambda self, **kwargs: None)
    cli.main()


def test_first_cli_start_saves_existing_defaults_and_imports(isolated_home, tmp_path, monkeypatch):
    roots = standard_roots(isolated_home)
    for root in roots:
        root.mkdir(parents=True)
    fixture = Path(__file__).parent / 'fixtures/pi/source/ordinary.jsonl'
    (roots[0] / 'ordinary.jsonl').write_bytes(fixture.read_bytes())
    data = tmp_path / 'data'
    run_cli(data, monkeypatch)
    expected = [str(root.resolve()) for root in roots]
    assert json.loads((data / 'sources.json').read_text()) == expected
    reopened = Application(data, timezone='UTC')
    try:
        assert list(reopened.get_roots()) == expected
        assert reopened.status().state == 'succeeded'
        assert len(reopened.storage.snapshot().observations) > 0
    finally:
        reopened.close()


@pytest.mark.parametrize('saved', [[], ['custom']])
def test_saved_roots_including_empty_prevent_discovery(isolated_home, tmp_path, monkeypatch, saved):
    standard_roots(isolated_home)[0].mkdir(parents=True)
    data = tmp_path / 'data'
    custom = tmp_path / 'custom'
    custom.mkdir()
    application = Application(data, timezone='UTC')
    application.set_roots(tuple(str(custom) for _ in saved))
    application.close()
    before = (data / 'sources.json').read_bytes()
    run_cli(data, monkeypatch)
    assert (data / 'sources.json').read_bytes() == before


def test_explicit_root_wins_over_saved_and_defaults(isolated_home, tmp_path, monkeypatch):
    standard_roots(isolated_home)[0].mkdir(parents=True)
    data = tmp_path / 'data'
    application = Application(data, timezone='UTC')
    application.set_roots(())
    application.close()
    explicit = tmp_path / 'explicit'
    explicit.mkdir()
    run_cli(data, monkeypatch, '--root', str(explicit))
    assert json.loads((data / 'sources.json').read_text()) == [str(explicit.resolve())]


def test_no_defaults_does_not_save_empty_config_or_discover_from_application(isolated_home, tmp_path, monkeypatch):
    data = tmp_path / 'data'
    run_cli(data, monkeypatch)
    assert not (data / 'sources.json').exists()
    standard_roots(isolated_home)[0].mkdir(parents=True)
    application = Application(data, timezone='UTC')
    try:
        assert application.get_roots() == ()
        assert not (data / 'sources.json').exists()
    finally:
        application.close()


def test_overrides_replace_defaults_and_normalized_paths_deduplicate(isolated_home, monkeypatch):
    for root in standard_roots(isolated_home):
        root.mkdir(parents=True)
    override = isolated_home / 'override'
    for name in ('sessions', 'archived_sessions', 'projects', 'session-state'):
        (override / name).mkdir(parents=True, exist_ok=True)
    for name in ('PI_CODING_AGENT_DIR', 'CODEX_HOME', 'CLAUDE_CONFIG_DIR', 'COPILOT_HOME'):
        monkeypatch.setenv(name, '~/override/../override')
    assert cli.default_source_roots() == tuple(str(path.resolve()) for path in (
        override / 'sessions', override / 'archived_sessions', override / 'projects',
        override, *standard_roots(isolated_home)[-2:]))
    monkeypatch.setenv('CODEX_HOME', str(isolated_home / 'absent'))
    assert str(standard_roots(isolated_home)[1]) not in cli.default_source_roots()


def test_empty_overrides_use_defaults(isolated_home, monkeypatch):
    root = standard_roots(isolated_home)[0]
    root.mkdir(parents=True)
    monkeypatch.setenv('PI_CODING_AGENT_DIR', '')
    assert cli.default_source_roots() == (str(root.resolve()),)


def test_pi_direct_sessions_override_takes_precedence_without_fallback(isolated_home, monkeypatch):
    standard_roots(isolated_home)[0].mkdir(parents=True)
    agent = isolated_home / 'agent'
    (agent / 'sessions').mkdir(parents=True)
    direct = isolated_home / 'direct-sessions'
    direct.mkdir()
    monkeypatch.setenv('PI_CODING_AGENT_DIR', str(agent))
    monkeypatch.setenv('PI_CODING_AGENT_SESSION_DIR', str(direct))
    assert cli.default_source_roots() == (str(direct.resolve()),)
    monkeypatch.setenv('PI_CODING_AGENT_SESSION_DIR', str(isolated_home / 'absent'))
    assert cli.default_source_roots() == ()
    monkeypatch.setenv('PI_CODING_AGENT_SESSION_DIR', '')
    assert cli.default_source_roots() == (str((agent / 'sessions').resolve()),)


def test_missing_inaccessible_file_and_symlink_roots_are_skipped(isolated_home, monkeypatch):
    roots = standard_roots(isolated_home)
    roots[0].mkdir(parents=True)
    roots[1].parent.mkdir(parents=True)
    roots[1].symlink_to(roots[0], target_is_directory=True)
    roots[2].write_text('not a directory')
    roots[3].mkdir(parents=True)
    roots[3].chmod(0)
    try:
        assert cli.default_source_roots() == (str(roots[0].resolve()),)
    finally:
        roots[3].chmod(0o700)


def test_unreadable_directory_probe_is_skipped(isolated_home, monkeypatch):
    root = standard_roots(isolated_home)[0]
    root.mkdir(parents=True)
    def denied(path):
        raise PermissionError('synthetic denied directory')
    monkeypatch.setattr(os, 'scandir', denied)
    assert cli.default_source_roots() == ()
