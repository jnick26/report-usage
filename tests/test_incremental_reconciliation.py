"""Scoped accounting must match full reconciliation and leave unrelated decisions alone."""
import json
from pathlib import Path

import pytest

from harness_usage.aggregate_reporting import build_aggregate_report
from harness_usage.pricing import load_bundled_catalog
from harness_usage.reporting import AllTime, ReportQuery
from harness_usage.storage import Storage
from test_codex_storage import codex_source, legacy, modern, usage

PI = Path(__file__).parent / 'fixtures/pi/source'


def accounting(db):
    return (tuple(tuple(r) for r in db.execute('SELECT * FROM decision ORDER BY observation_id,measure')),
            tuple(tuple(r) for r in db.execute('SELECT source_id,observation_id,line,code,measure FROM diagnostic ORDER BY source_id,observation_id,line,code,measure')))


def assert_full_equal(store):
    query = ReportQuery(None, AllTime())
    report = build_aggregate_report(store, query, load_bundled_catalog())
    with store.connect() as db:
        before = accounting(db)
        store._reconcile(db)
        assert accounting(db) == before
    assert build_aggregate_report(store, query, load_bundled_catalog()) == report


@pytest.mark.parametrize('harness', ['pi', 'codex'])
def test_changed_source_does_not_rewrite_unrelated_decisions(tmp_path, harness):
    store = Storage(tmp_path / 'ledger.db')
    if harness == 'codex':
        first = codex_source('first', modern('first-response', 'first'))
        other = codex_source('other', modern('other-response', 'other'))
        other_id = 'codex:other'
    else:
        first = (PI / 'ordinary.jsonl').read_bytes()
        rows = [json.loads(line) for line in first.splitlines()]
        rows[0]['id'] = 'unrelated'
        other = ('\n'.join(map(json.dumps, rows)) + '\n').encode()
        other_id = 'pi:unrelated'
    store.import_sources((('/first', first), ('/other', other)))
    def unrelated_rows():
        with store.connect() as db:
            return tuple(db.execute('SELECT d.rowid,d.* FROM decision d JOIN observation o ON o.id=d.observation_id WHERE o.session_id=? ORDER BY d.measure', (other_id,)))
    before = unrelated_rows()
    store.import_source('/first', first + b'\n')
    assert unrelated_rows() == before
    assert_full_equal(store)


@pytest.mark.parametrize('reverse', [False, True])
def test_pi_fixture_generations_match_full_accounting(tmp_path, reverse):
    store = Storage(tmp_path / 'ledger.db')
    files = sorted(PI.glob('*.jsonl'), reverse=reverse)
    for path in files:
        store.import_source('/fixtures/pi/source/' + path.name, path.read_bytes())
        assert_full_equal(store)


def test_codex_owner_identity_lineage_and_source_order_match_full(tmp_path):
    store = Storage(tmp_path / 'ledger.db')
    ten, five = usage(10), usage(5, 50)
    total = {key: ten[key] + five[key] for key in ten}
    scenarios = [
        ('/unrelated', codex_source('unrelated', modern('alone', 'unrelated'))),
        ('/child', codex_source('child', legacy(ten, ten), legacy(five, total), parent='parent')),
        ('/parent', codex_source('parent', legacy(ten, ten))),
        # The source header is NOT the owner of this modern response.
        ('/carrier', codex_source('carrier', modern('shared', 'actual-owner'))),
        ('/conflict', codex_source('conflict', modern('shared', 'another-owner', output=11))),
        ('/child', codex_source('child', legacy(ten, ten), modern('boundary', 'elsewhere'), legacy(five, total), parent='parent')),
        ('/child', codex_source('child', legacy(ten, ten), parent='different-parent')),
        ('/different-parent', codex_source('different-parent', legacy(ten, ten))),
        # Replacement source retains prior generations and dependency edges.
        ('/carrier', codex_source('replacement', modern('new', 'replacement'))),
    ]
    for locator, data in scenarios:
        store.import_source(locator, data)
        assert_full_equal(store)


def test_parent_locator_becoming_ambiguous_updates_existing_children(tmp_path):
    store = Storage(tmp_path / 'ledger.db')
    parent = (PI / 'parent.jsonl').read_bytes()
    store.import_source('/fixtures/pi/source/parent.jsonl', parent)
    store.import_source('/fixtures/pi/source/fork.jsonl', (PI / 'fork.jsonl').read_bytes())
    rows = [json.loads(line) for line in parent.splitlines()]
    rows[0]['id'] = 'different-parent'
    store.import_source('/fixtures/pi/source/parent.jsonl', ('\n'.join(map(json.dumps, rows)) + '\n').encode())
    assert_full_equal(store)


def test_scoped_decision_failure_rolls_back_import_and_revision(tmp_path, monkeypatch):
    store = Storage(tmp_path / 'ledger.db')
    data = codex_source('root', modern())
    store.import_source('/root', data)
    before = store.snapshot()
    from harness_usage.database import Connection
    original = Connection.executemany
    def fail(self, sql, rows):
        if sql.startswith('INSERT INTO decision'):
            raise RuntimeError('interrupted')
        return original(self, sql, rows)
    with monkeypatch.context() as patch:
        patch.setattr(Connection, 'executemany', fail)
        with pytest.raises(RuntimeError, match='interrupted'):
            store.import_source('/root', data + b'\n')
    assert store.snapshot() == before
    assert store.import_source('/root', data + b'\n') == before.revision + 1
    assert_full_equal(store)


def test_unrelated_model_conflict_diagnostics_survive_scoped_import(tmp_path):
    store = Storage(tmp_path / 'ledger.db')
    data = codex_source('conflict', modern('conflicting', 'conflict'))
    store.import_source('/model-one', data)
    store.import_source('/model-two', data.replace(b'gpt-5', b'gpt-5.6-sol'))
    with store.connect() as db:
        before = tuple(tuple(r) for r in db.execute("SELECT * FROM diagnostic WHERE code='codex_model_conflict' ORDER BY id"))
    assert before
    store.import_source('/unrelated', codex_source('other', modern('other', 'other')))
    with store.connect() as db:
        assert tuple(tuple(r) for r in db.execute("SELECT * FROM diagnostic WHERE code='codex_model_conflict' ORDER BY id")) == before
    assert_full_equal(store)
