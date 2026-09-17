from dataclasses import asdict
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import subprocess
from threading import Event

import pytest
from fastapi.testclient import TestClient

from harness_usage.application import Application
from harness_usage.storage import ImportStatus
from harness_usage.web import create_app


FIXTURE = Path(__file__).parent / 'fixtures/pi/source/ordinary.jsonl'


def test_real_work_progress_stays_finalizing_until_commit_and_attribution(tmp_path, monkeypatch):
    root = tmp_path / 'sources'
    root.mkdir()
    (root / 'a.jsonl').write_text('{"unsupported": true}\n')
    for name in ('b.jsonl', 'c.jsonl'):
        (root / name).write_bytes(FIXTURE.read_bytes())
    app = Application(tmp_path / 'data', timezone='UTC')
    app.set_roots((str(root), str(root)))
    stages = {name: (Event(), Event()) for name in ('discover', 'a.jsonl', 'b.jsonl', 'c.jsonl', 'commit', 'attribute')}
    def gate(name):
        entered, proceed = stages[name]
        entered.set()
        assert proceed.wait(10)
    walk, read, commit, attribute = os.walk, Path.read_bytes, app.storage.import_sources, app._assign_projects
    def walking(*args, **kwargs):
        gate('discover')
        yield from walk(*args, **kwargs)
    def reading(path):
        if path.parent == root:
            gate(path.name)
        return read(path)
    def committing(batch):
        gate('commit')
        return commit(batch)
    def attributing():
        gate('attribute')
        attribute()
    monkeypatch.setattr(os, 'walk', walking)
    monkeypatch.setattr(Path, 'read_bytes', reading)
    monkeypatch.setattr(app.storage, 'import_sources', committing)
    monkeypatch.setattr(app, '_assign_projects', attributing)
    try:
        run = app.start_import()
        for name, phase, checked, total in (
            ('discover', 'discovering', 0, None),
            ('a.jsonl', 'checking', 0, 3),
            ('b.jsonl', 'checking', 1, 3),
            ('c.jsonl', 'checking', 2, 3),
            ('commit', 'finalizing', 3, 3),
            ('attribute', 'finalizing', 3, 3),
        ):
            assert stages[name][0].wait(10)
            status = app.status()
            assert (status.run_id, status.state, status.phase, status.files_checked, status.files_total) == (run.run_id, 'running', phase, checked, total)
            assert app.start_import().run_id == run.run_id
            if name == 'discover':
                assert tuple(app._scan(())) == ()
                assert app.status() == status
            if name == 'commit':
                assert status.files_processed == 0
                assert app.storage.snapshot().observations == ()
            stages[name][1].set()
    finally:
        for _, proceed in stages.values():
            proceed.set()
        app.close()
    status = app.status()
    assert (status.state, status.phase, status.files_checked, status.files_total, status.files_processed) == ('succeeded', None, 3, 3, 2)
    assert len(app.storage.snapshot().observations) > 0
    reopened = Application(tmp_path / 'data', timezone='UTC')
    try:
        assert reopened.status().state == 'succeeded'
        assert reopened.status().phase is None
        assert reopened.status().files_total is None
    finally:
        reopened.close()


@pytest.mark.parametrize('failure_stage', ['read', 'commit', 'attribute'])
def test_failure_never_reports_completed_and_retry_resets_progress(tmp_path, monkeypatch, failure_stage):
    root = tmp_path / 'sources'
    root.mkdir()
    (root / 'a.jsonl').write_bytes(FIXTURE.read_bytes())
    app = Application(tmp_path / 'data', timezone='UTC')
    app.set_roots((str(root),))
    with monkeypatch.context() as patch:
        def fail(*args, **kwargs):
            raise OSError('PRIVATE_PROGRESS_SENTINEL')
        if failure_stage == 'read':
            patch.setattr(Path, 'read_bytes', fail)
        elif failure_stage == 'commit':
            patch.setattr(app.storage, 'import_sources', fail)
        else:
            patch.setattr(app, '_assign_projects', fail)
        old = app.start_import().run_id
        app.close()
    status = app.status()
    assert status.state == 'failed'
    assert status.phase is None
    assert status.files_checked == (0 if failure_stage == 'read' else 1)
    assert status.files_total == 1
    assert 'PRIVATE_PROGRESS_SENTINEL' not in json.dumps(asdict(status))
    app.start_import()
    app.close()
    assert app.status().run_id != old
    assert app.status().state == 'succeeded'
    assert (app.status().files_checked, app.status().files_total) == (1, 1)
    app._update_progress(old, 'checking', 999, 999)
    assert (app.status().files_checked, app.status().files_total) == (1, 1)


def test_empty_scan_finalizes_without_division_by_zero_and_restart_has_no_fake_progress(tmp_path, monkeypatch):
    app = Application(tmp_path / 'data', timezone='UTC')
    seen = []
    original = app._assign_projects
    def attribute():
        seen.append(app.status())
        original()
    monkeypatch.setattr(app, '_assign_projects', attribute)
    app.start_import()
    app.close()
    assert (seen[0].phase, seen[0].files_checked, seen[0].files_total) == ('finalizing', 0, 0)
    app.storage.begin_import('interrupted-run')
    reopened = Application(tmp_path / 'data', timezone='UTC')
    try:
        status = reopened.status()
        assert (status.state, status.phase, status.files_total) == ('interrupted', None, None)
    finally:
        reopened.close()


class ProgressParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.progress = None

    def handle_starttag(self, tag, attrs):
        if tag == 'progress':
            self.progress = dict(attrs)


@pytest.mark.parametrize(('state', 'phase', 'checked', 'total', 'value', 'text'), [
    ('running', 'discovering', 0, None, None, 'Discovering files'),
    ('running', 'checking', 1, 3, '1', '1 / 3 files checked · 33%'),
    ('running', 'finalizing', 3, 3, None, 'Finalizing'),
    ('failed', None, 3, 3, None, 'Import failed'),
    ('interrupted', None, 0, None, None, 'interrupted'),
    ('succeeded', None, 3, 3, None, 'Saved local history'),
])
def test_main_page_accessible_progress_matches_lifecycle(tmp_path, monkeypatch, state, phase, checked, total, value, text):
    app = Application(tmp_path / 'data', timezone='UTC')
    root = tmp_path / 'sources'
    root.mkdir()
    app.set_roots((str(root),))
    status = ImportStatus('run', state, 0, 0, 'import_failed' if state in ('failed', 'interrupted') else None,
                          phase=phase, files_checked=checked, files_total=total)
    monkeypatch.setattr(app, 'status', lambda: status)
    try:
        html = TestClient(create_app(app), base_url='http://localhost').get('/').text
        parser = ProgressParser()
        parser.feed(html)
        assert parser.progress is not None
        assert parser.progress['aria-label'] == 'Import progress'
        assert parser.progress.get('value') == value
        assert ('hidden' in parser.progress) == (state != 'running')
        if value is not None:
            assert parser.progress['max'] == '3'
        assert text in html
    finally:
        app.close()


def test_events_update_bar_and_refresh_only_committed_report():
    result = subprocess.run(['node', 'tests/check_frontend_events.cjs'], cwd=Path(__file__).parents[1],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
