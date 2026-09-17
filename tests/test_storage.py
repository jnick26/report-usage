from pathlib import Path
import sqlite3
from decimal import Decimal

import pytest

FIXTURES = Path(__file__).parent / 'fixtures/pi/source'


def load(store, name, locator=None):
    store.import_source(locator or '/fixtures/pi/source/' + name, (FIXTURES / name).read_bytes())


def output(store):
    return sum(o.record.tokens.buckets.output.value for o in store.snapshot().observations if o.decisions['output'] == 'selected')


def test_repeat_move_delete_and_tail(tmp_path):
    from harness_usage.storage import Storage
    store = Storage(tmp_path / 'ledger.db')
    load(store, 'ordinary.jsonl')
    revision = store.snapshot().revision
    load(store, 'ordinary.jsonl')
    assert store.snapshot().revision == revision
    load(store, 'ordinary.jsonl', '/archive/ordinary.jsonl')
    assert output(store) == 20
    store.mark_missing(('/fixtures/pi/source/ordinary.jsonl',))
    assert output(store) == 20
    other = Storage(tmp_path / 'tail.db')
    load(other, 'tail-before.jsonl', '/live.jsonl')
    load(other, 'tail-after.jsonl', '/live.jsonl')
    load(other, 'tail-after.jsonl', '/live.jsonl')
    assert output(other) == 50


@pytest.mark.parametrize('order', [('parent.jsonl','fork.jsonl'), ('fork.jsonl','parent.jsonl')])
def test_copied_prefix_counts_once_in_either_order(tmp_path, order):
    from harness_usage.storage import Storage
    store = Storage(tmp_path / 'ledger.db')
    for filename in order:
        load(store, filename)
    assert output(store) == 130
    assert sum(o.decisions['output'] == 'excluded' for o in store.snapshot().observations) == 1


def test_tool_conflict_and_distinct_attempts(tmp_path):
    from harness_usage.storage import Storage
    store = Storage(tmp_path / 'ledger.db')
    load(store, 'tool-parent.jsonl'); load(store, 'tool-child.jsonl')
    assert output(store) == 50
    assert any('tool_overlap_unproven' in o.reasons for o in store.snapshot().observations)
    conflict = Storage(tmp_path / 'conflict.db')
    load(conflict, 'conflict-a.jsonl'); load(conflict, 'conflict-b.jsonl')
    assert output(conflict) == 0
    attempts = Storage(tmp_path / 'attempts.db')
    load(attempts, 'identical-attempts.jsonl')
    assert output(attempts) == 40


def test_atomic_failure_rolls_back_evidence_and_revision(tmp_path, monkeypatch):
    from harness_usage.storage import Storage
    path = tmp_path / 'ledger.db'; store = Storage(path)
    from harness_usage.database import Connection
    original = Connection.executemany
    def fail(self, sql, rows):
        if sql.startswith('INSERT OR IGNORE INTO observation'):
            raise RuntimeError('test interruption')
        return original(self, sql, rows)
    monkeypatch.setattr(Connection, 'executemany', fail)
    with pytest.raises(RuntimeError, match='test interruption'):
        load(store, 'ordinary.jsonl')
    assert store.snapshot().revision == 0
    assert store.snapshot().observations == ()


def test_decimal_round_trip_and_no_transcript(tmp_path):
    from harness_usage.storage import Storage
    store = Storage(tmp_path / 'ledger.db')
    data = (FIXTURES / 'ordinary.jsonl').read_bytes().replace(b'"content":[]', b'"content":[{"type":"text","text":"PRIVATE_SENTINEL"}]')
    store.import_source('/ordinary.jsonl', data)
    assert store.snapshot().observations[0].record.money.amount == Decimal('0.01')
    assert b'PRIVATE_SENTINEL' not in (tmp_path / 'ledger.db').read_bytes()


def test_batch_import_is_atomic_and_repeat_is_idempotent(tmp_path):
    from harness_usage.storage import Storage
    store = Storage(tmp_path / 'ledger.db')
    sources = tuple(('/fixtures/pi/source/' + name, (FIXTURES / name).read_bytes()) for name in ('parent.jsonl', 'fork.jsonl'))
    assert store.import_sources(sources) == 1
    assert output(store) == 130
    assert store.import_sources(sources) == 1
    assert set(store.locators()) == {source[0] for source in sources}


def test_attribution_persists_and_unchanged_save_does_not_advance_revision(tmp_path):
    from harness_usage.storage import Storage
    from harness_usage.domain import Assigned, ProjectId
    store = Storage(tmp_path / 'ledger.db')
    load(store, 'ordinary.jsonl')
    sid = store.snapshot().sessions[0].id
    value = Assigned(ProjectId('directory:/fixtures/atlas'), '/fixtures/atlas', 'recorded_cwd')
    store.save_attributions({sid: value})
    revision = store.snapshot().revision
    assert Storage(store.path).attributions()[sid] == value
    store.save_attributions({sid: value})
    assert store.snapshot().revision == revision


def test_derived_total_overflow_remains_unknown_without_losing_buckets(tmp_path):
    from harness_usage.storage import Storage
    from harness_usage.domain import Known, Unknown
    import json
    lines = [json.loads(line) for line in (FIXTURES / 'ordinary.jsonl').read_text().splitlines()]
    lines[1]['message']['usage'] = {'input': 2**63 - 1, 'output': 1, 'cacheRead': 0, 'cacheWrite': 0}
    store = Storage(tmp_path / 'ledger.db')
    store.import_source('/x', ('\n'.join(map(json.dumps, lines)) + '\n').encode())
    tokens = store.snapshot().observations[0].record.tokens
    assert tokens.buckets.input == Known(2**63 - 1)
    assert tokens.total == Unknown('invalid_count')


def test_conflicting_parent_header_is_not_silently_merged(tmp_path):
    from harness_usage.storage import Storage
    import json
    data = (FIXTURES / 'ordinary.jsonl').read_bytes()
    lines = data.splitlines()
    header = json.loads(lines[0]); header['parentSession'] = '/other'
    store = Storage(tmp_path / 'ledger.db')
    store.import_source('/first', data)
    store.import_source('/second', json.dumps(header).encode() + b'\n' + b'\n'.join(lines[1:]))
    assert output(store) == 0
    assert 'header_conflict' in store.snapshot().diagnostics


def test_copy_requires_matching_accounting_on_entire_parent_path(tmp_path):
    from harness_usage.storage import Storage
    import json
    parent = [json.loads(line) for line in (FIXTURES / 'parent.jsonl').read_text().splitlines()]
    child = [json.loads(line) for line in (FIXTURES / 'fork.jsonl').read_text().splitlines()]
    parent.append(child[-1])
    child[1]['message']['usage']['output'] = 101
    child[1]['message']['usage']['totalTokens'] = 101
    store = Storage(tmp_path / 'ledger.db')
    for name, rows in (('parent.jsonl', parent), ('fork.jsonl', child)):
        store.import_source('/fixtures/pi/source/' + name, ('\n'.join(map(json.dumps, rows)) + '\n').encode())
    child_session = 'pi:' + child[0]['id']
    assert all(o.decisions['output'] == 'unresolved' for o in store.snapshot().observations if o.session_id == child_session)


def test_snapshot_contains_same_revision_attribution_and_scoped_diagnostics(tmp_path):
    from harness_usage.storage import Storage
    from harness_usage.domain import Unassigned
    store = Storage(tmp_path / 'ledger.db')
    load(store, 'ordinary.jsonl'); load(store, 'invalid-count.jsonl')
    sid = store.snapshot().sessions[0].id
    store.save_attributions({sid: Unassigned('missing_cwd')})
    snapshot = store.snapshot()
    assert snapshot.attributions[sid] == Unassigned('missing_cwd')
    ordinary = next(o for o in snapshot.observations if o.record.entry.native_id == '00000001')
    invalid = next(o for o in snapshot.observations if o.record.entry.native_id == '70000001')
    assert 'invalid_count' not in ordinary.diagnostics
    assert 'invalid_count' in invalid.diagnostics


def test_names_update_only_for_proven_append_and_conflicts_keep_counters(tmp_path):
    from harness_usage.storage import Storage
    import json
    store = Storage(tmp_path / 'ledger.db')
    data = (FIXTURES / 'explicit-name.jsonl').read_bytes()
    store.import_source('/a', data)
    appended = data + json.dumps({'type': 'session_info', 'id': 'newname', 'name': 'New name'}).encode() + b'\n'
    store.import_source('/a', appended)
    assert store.snapshot().sessions[0].display_name == 'New name'
    store.import_source('/different', data)
    assert store.snapshot().sessions[0].display_name is None
    assert output(store) == 20
    assert 'name_conflict' in store.snapshot().diagnostics


def test_batch_failure_rolls_back_earlier_files(tmp_path):
    from harness_usage.storage import Storage
    store = Storage(tmp_path / 'ledger.db')
    def broken_sources():
        yield '/ordinary', (FIXTURES / 'ordinary.jsonl').read_bytes()
        raise OSError('read interrupted')
    with pytest.raises(OSError):
        store.import_sources(broken_sources())
    assert store.snapshot().revision == 0
    assert store.snapshot().observations == ()
    assert store.locators() == ()


def test_identity_conflict_quarantines_only_differing_measures(tmp_path):
    from harness_usage.storage import Storage
    from harness_usage.domain import Known
    import json
    rows = [json.loads(line) for line in (FIXTURES / 'ordinary.jsonl').read_text().splitlines()]
    changed = json.loads(json.dumps(rows))
    changed[1]['message']['usage']['output'] = 21
    changed[1]['message']['usage']['totalTokens'] = 971
    store = Storage(tmp_path / 'ledger.db')
    for locator, records in (('/one', rows), ('/two', changed)):
        store.import_source(locator, ('\n'.join(map(json.dumps, records)) + '\n').encode())
    observations = store.snapshot().observations
    assert sum(o.record.tokens.buckets.input.value for o in observations if o.decisions['input'] == 'selected') == 100
    assert all(o.decisions['output'] == 'unresolved' for o in observations)
    assert all(o.decisions['total'] == 'unresolved' for o in observations)
    assert sum(o.record.money.amount for o in observations if o.decisions['recorded_usd'] == 'selected') == Decimal('0.01')


def test_durable_import_run_start_join_progress_and_interruption(tmp_path):
    from harness_usage.storage import Storage
    store = Storage(tmp_path / 'ledger.db')
    assert store.import_status().state == 'idle'
    first = store.begin_import('first')
    assert store.begin_import('second').run_id == first.run_id
    store.advance_import('first', 3)
    assert store.import_status().files_processed == 3
    store.interrupt_runs()
    assert store.import_status().state == 'interrupted'
    second = store.begin_import('second')
    store.finish_import(second.run_id, 1, None)
    assert store.import_status().state == 'succeeded'


@pytest.mark.parametrize('state,run_id,count,error', [('idle','id',0,None), ('running',None,0,None), ('failed','id',0,None), ('succeeded','id',0,'bad'), ('running','id',-1,None), ('running','id',True,None)])
def test_import_status_rejects_impossible_combinations(state, run_id, count, error):
    from harness_usage.storage import ImportStatus
    with pytest.raises(ValueError):
        ImportStatus(run_id, state, count, 0, error)


@pytest.mark.parametrize('state,canonical', [('excluded', None), ('excluded', 'same'), ('selected', 'other'), ('unresolved', 'other')])
def test_decision_constructor_rejects_invalid_canonical_relationships(state, canonical):
    from harness_usage.accounting import Decision
    with pytest.raises(ValueError):
        Decision('same', state, canonical, 'rule')


def test_import_status_uses_insertion_order_when_wall_clock_moves_back(tmp_path):
    from harness_usage.storage import Storage
    store = Storage(tmp_path / 'ledger.db')
    with store.connect() as db:
        db.execute("INSERT INTO import_run(id,state,started_us,finished_us,files_processed,revision,error_code) VALUES('future','succeeded',9999999999999999,9999999999999999,0,0,NULL)")
    status = store.begin_import('current')
    assert store.import_status().run_id == status.run_id == 'current'
    assert store.begin_import('joined').run_id == 'current'


def test_git_project_keeps_repository_storage_kind(tmp_path):
    from harness_usage.storage import Storage
    from harness_usage.domain import Assigned, ProjectId
    store = Storage(tmp_path / 'ledger.db'); load(store, 'ordinary.jsonl')
    sid = store.snapshot().sessions[0].id
    store.save_attributions({sid: Assigned(ProjectId('git:/project/.git'), '/project', 'git_common_dir')})
    with store.connect() as db:
        assert db.execute('SELECT kind,label FROM project').fetchone()[:] == ('repository', 'project')


def test_missing_ancestor_is_qualified_until_parent_arrives(tmp_path):
    from harness_usage.storage import Storage
    store = Storage(tmp_path / 'ledger.db')
    load(store, 'fork.jsonl')
    assert output(store) == 0
    assert all('missing_ancestor' in o.reasons for o in store.snapshot().observations)
    load(store, 'parent.jsonl')
    assert output(store) == 130


@pytest.mark.parametrize('reverse', [False, True])
def test_nested_copies_converge_to_earliest_origin(tmp_path, reverse):
    from harness_usage.storage import Storage
    import json
    grandchild = [json.loads(line) for line in (FIXTURES / 'fork.jsonl').read_text().splitlines()]
    grandchild[0]['id'] = 'grandchild'
    grandchild[0]['parentSession'] = '/fixtures/pi/source/fork.jsonl'
    fresh = json.loads(json.dumps(grandchild[-1]))
    fresh['id'] = 'grandchild-new'; fresh['parentId'] = grandchild[-1]['id']
    fresh['message']['usage']['output'] = fresh['message']['usage']['totalTokens'] = 7
    grandchild.append(fresh)
    sources = [('parent.jsonl', (FIXTURES / 'parent.jsonl').read_bytes()), ('fork.jsonl', (FIXTURES / 'fork.jsonl').read_bytes()), ('grandchild.jsonl', ('\n'.join(map(json.dumps, grandchild)) + '\n').encode())]
    store = Storage(tmp_path / 'ledger.db')
    for name, data in reversed(sources) if reverse else sources:
        store.import_source('/fixtures/pi/source/' + name, data)
    assert output(store) == 137
    with store.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM decision d LEFT JOIN decision c ON d.canonical=c.observation_id AND d.measure=c.measure WHERE d.state='excluded' AND (c.state IS NULL OR c.state<>'selected')").fetchone()[0] == 0


def test_snapshot_remains_available_during_uncommitted_writer_transaction(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from harness_usage.storage import Storage
    store = Storage(tmp_path / 'ledger.db')
    load(store, 'ordinary.jsonl')
    original = store.snapshot()
    with ThreadPoolExecutor(max_workers=1) as pool:
        with store.connect() as writer:
            writer.execute('BEGIN IMMEDIATE')
            source_id = writer.execute('SELECT id FROM source_generation LIMIT 1').fetchone()[0]
            writer.executemany('INSERT INTO diagnostic(source_id,code) VALUES(?,?)', ((source_id, 'spill_' + 'x' * 4096) for _ in range(256)))
            writer.execute('UPDATE ledger_meta SET revision=revision+1')
            pending = pool.submit(store.snapshot)
            try:
                during = pending.result(timeout=2)
                assert during == original
                writer.commit()
            finally:
                if writer.in_transaction:
                    writer.rollback()
    assert store.snapshot().revision == original.revision + 1


def test_snapshot_diagnostics_scale_by_matching_lines_not_source_cross_product(tmp_path, monkeypatch):
    from contextlib import contextmanager
    from harness_usage.storage import Storage
    import json
    rows = [json.loads(line) for line in (FIXTURES / 'ordinary.jsonl').read_text().splitlines()]
    entries = [{**rows[1], 'id': f'{index:08x}'} for index in range(500)]
    store = Storage(tmp_path / 'ledger.db')
    store.import_source('/diagnostic-source', ('\n'.join(json.dumps(row) for row in [rows[0], *entries]) + '\n').encode())
    with store.connect() as db:
        source = db.execute('SELECT id FROM source_generation').fetchone()[0]
        db.executemany('INSERT INTO diagnostic(source_id,line,code) VALUES(?,?,?)', ((source, index + 2, f'entry_{index}') for index in range(500)))
        db.execute("INSERT INTO diagnostic(source_id,code) VALUES(?,'whole_source')", (source,))
    original_connect = store.connect
    @contextmanager
    def bounded_read():
        with original_connect() as db:
            original = db.execute
            def bounded(sql, args=()):
                result = original(sql, args)
                if sql.lstrip().startswith('SELECT'):
                    assert len(result.rows) <= 500 * 8
                return result
            monkeypatch.setattr(db, 'execute', bounded)
            yield db
    monkeypatch.setattr(store, 'connect', bounded_read)
    snapshot = store.snapshot()
    assert len(snapshot.observations) == 500
    for observation in snapshot.observations:
        index = int(observation.record.entry.native_id, 16)
        assert observation.diagnostics == (f'entry_{index}', 'whole_source')


def test_snapshot_reads_accounting_tables_in_batches(tmp_path, monkeypatch):
    from contextlib import contextmanager
    from harness_usage.storage import Storage
    import json
    rows = [json.loads(line) for line in (FIXTURES / 'ordinary.jsonl').read_text().splitlines()]
    entries = [{**rows[1], 'id': f'{index:08x}'} for index in range(10)]
    store = Storage(tmp_path / 'ledger.db')
    store.import_source('/batch-read', ('\n'.join(json.dumps(row) for row in [rows[0], *entries]) + '\n').encode())
    statements = []
    original_connect = store.connect
    @contextmanager
    def observed_read():
        with original_connect() as db:
            db.set_trace_callback(statements.append)
            yield db
    monkeypatch.setattr(store, 'connect', observed_read)
    snapshot = store.snapshot()
    assert len(snapshot.observations) == 10
    assert sum(item.record.tokens.total.value for item in snapshot.observations) == 9700
    assert len([statement for statement in statements if statement.startswith('SELECT')]) <= 16
