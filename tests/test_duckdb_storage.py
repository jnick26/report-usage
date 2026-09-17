from dataclasses import replace
from pathlib import Path
import pytest
import duckdb
from harness_usage.database import Connection
from harness_usage.pi_reader import EntryIdentity
from harness_usage.storage import Storage


INSERT_FORMS = (
    'INSERT INTO probe(a,b) VALUES(?,?)',
    'INSERT INTO probe(a, b) VALUES(?,?)',
    'INSERT INTO probe( a,b ) VALUES(?,?)',
    'INSERT INTO probe (a,b) VALUES(?,?)',
    'INSERT INTO probe(a,b) VALUES (?,?)',
    'INSERT INTO probe(a,b) VALUES(?, ?)',
)
INSERT_PATHS = ('execute', 'executemany', 'batch')
INTEGER_VALUES = ((1, (0, 1)), (1.5, None), (True, None))


def run_parameter_insert(db, path, sql, row):
    if path == 'execute':
        db.execute(sql, row)
    elif path == 'executemany':
        db.executemany(sql, [row])
    else:
        with db.batch():
            db.execute(sql, row)


@pytest.mark.parametrize('sql', INSERT_FORMS)
@pytest.mark.parametrize('path', INSERT_PATHS)
@pytest.mark.parametrize(('value', 'expected'), INTEGER_VALUES,
                         ids=('integer', 'fraction', 'boolean'))
def test_parameter_insert_whitespace_preserves_exact_bigint_guard(sql, path, value, expected):
    raw = duckdb.connect(':memory:')
    db = Connection(raw)
    db.execute('CREATE TABLE probe(a BIGINT,b BIGINT)')
    if expected is None:
        with pytest.raises(ValueError, match='^invalid_database_integer$'):
            run_parameter_insert(db, path, sql, (0, value))
        assert db.execute('SELECT a,b FROM probe').fetchall() == []
    else:
        run_parameter_insert(db, path, sql, (0, value))
        assert db.execute('SELECT a,b FROM probe').fetchall() == [expected]
    raw.close()


@pytest.mark.parametrize('sql', (
    'INSERT INTO probe VALUES ( ?, ? )',
    'INSERT OR IGNORE INTO probe ( a, b ) VALUES ( ?, ? )',
))
@pytest.mark.parametrize('path', INSERT_PATHS)
@pytest.mark.parametrize(('value', 'expected'), INTEGER_VALUES,
                         ids=('integer', 'fraction', 'boolean'))
def test_parameter_insert_control_forms_share_exact_bigint_guard(sql, path, value, expected):
    raw = duckdb.connect(':memory:')
    db = Connection(raw)
    db.execute('CREATE TABLE probe(a BIGINT PRIMARY KEY,b BIGINT)'
               if 'OR IGNORE' in sql else 'CREATE TABLE probe(a BIGINT,b BIGINT)')
    if expected is None:
        with pytest.raises(ValueError, match='^invalid_database_integer$'):
            run_parameter_insert(db, path, sql, (0, value))
        assert db.execute('SELECT a,b FROM probe').fetchall() == []
    else:
        run_parameter_insert(db, path, sql, (0, value))
        assert db.execute('SELECT a,b FROM probe').fetchall() == [expected]
    raw.close()


@pytest.mark.parametrize('path', INSERT_PATHS)
def test_parameter_insert_trims_reordered_columns_and_accepts_multiline_whitespace(path):
    raw = duckdb.connect(':memory:')
    db = Connection(raw)
    db.execute('CREATE TABLE probe(a BIGINT,b BIGINT)')
    run_parameter_insert(db, path, 'INSERT INTO probe ( b, a ) VALUES ( ?, ? )', (2, 1))
    run_parameter_insert(db, path,
                         'INSERT\nINTO\tprobe\n( a,\tb )\nVALUES\t( ?,\n? )',
                         (3, 4))
    assert db.execute('SELECT a,b FROM probe ORDER BY a').fetchall() == [(1, 2), (3, 4)]
    raw.close()


@pytest.mark.parametrize('path', INSERT_PATHS)
def test_parameter_insert_allows_nullable_bigint_without_inventing_zero(path):
    raw = duckdb.connect(':memory:')
    db = Connection(raw)
    db.execute('CREATE TABLE probe(a BIGINT,b BIGINT)')
    run_parameter_insert(db, path, 'INSERT INTO probe ( a, b ) VALUES ( ?, ? )', (0, None))
    assert db.execute('SELECT a,b FROM probe').fetchall() == [(0, None)]
    raw.close()


@pytest.mark.parametrize('path', INSERT_PATHS)
def test_parameter_insert_or_ignore_validates_before_duplicate_suppression(path):
    raw = duckdb.connect(':memory:')
    db = Connection(raw)
    db.execute('CREATE TABLE probe(a BIGINT PRIMARY KEY,b BIGINT)')
    db.execute('INSERT INTO probe VALUES(?,?)', (0, 1))
    sql = 'INSERT OR IGNORE INTO probe ( a, b ) VALUES ( ?, ? )'
    run_parameter_insert(db, path, sql, (0, 2))
    assert db.execute('SELECT a,b FROM probe').fetchall() == [(0, 1)]
    with pytest.raises(ValueError, match='^invalid_database_integer$'):
        run_parameter_insert(db, path, sql, (0, 1.5))
    assert db.execute('SELECT a,b FROM probe').fetchall() == [(0, 1)]
    raw.close()


def test_public_transaction_rolls_back_prior_batch_group_after_spaced_invalid_insert(tmp_path):
    path = tmp_path / 'ledger.duckdb'
    store = Storage(path)
    with store.connect(write=True) as db:
        db.execute('CREATE TABLE integer_probe(a BIGINT,b BIGINT)')
    with pytest.raises(ValueError, match='^invalid_database_integer$'):
        with store.connect(write=True) as db:
            with db.batch():
                db.execute('INSERT INTO integer_probe(a,b) VALUES(?,?)', (0, 1))
                db.execute('INSERT INTO integer_probe ( a, b ) VALUES ( ?, ? )', (1, 1.5))
    store.close()
    reopened = Storage(path)
    with reopened.connect() as db:
        assert db.execute('SELECT a,b FROM integer_probe').fetchall() == []
    with reopened.connect(write=True) as db:
        db.execute('INSERT INTO integer_probe ( a, b ) VALUES ( ?, ? )', (2, 3))
    with reopened.connect() as db:
        assert db.execute('SELECT a,b FROM integer_probe').fetchall() == [(2, 3)]
    reopened.close()


def test_parameter_insert_arrow_transfer_preserves_typed_values_and_unregisters_batch():
    raw = duckdb.connect(':memory:')
    db = Connection(raw)
    db.execute('CREATE TABLE probe(id BIGINT PRIMARY KEY,label VARCHAR,amount BIGINT)')
    db.execute('INSERT INTO probe(id,label,amount) VALUES(?,?,?)', (1, 'direct-compact', None))
    db.execute('INSERT INTO probe ( id, label, amount ) VALUES ( ?, ?, ? )',
               (2, 'direct-spaced', 2))
    db.executemany('INSERT INTO probe ( id, label, amount ) VALUES ( ?, ?, ? )', (
        (3, 'many-null', None), (4, 'many-known', 4),
    ))
    with db.batch():
        db.execute('INSERT INTO probe ( id, label, amount ) VALUES ( ?, ?, ? )',
                   (5, 'queued-null', None))
        db.execute('INSERT INTO probe ( id, label, amount ) VALUES ( ?, ?, ? )',
                   (6, 'queued-known', 6))
    db.executemany('INSERT OR IGNORE INTO probe ( id, label, amount ) VALUES ( ?, ?, ? )', (
        (4, 'ignored', 40), (7, 'many-ignore', 7),
    ))
    with db.batch():
        db.execute('INSERT OR IGNORE INTO probe ( id, label, amount ) VALUES ( ?, ?, ? )',
                   (6, 'ignored', 60))
        db.execute('INSERT OR IGNORE INTO probe ( id, label, amount ) VALUES ( ?, ?, ? )',
                   (8, 'queued-ignore', None))
    assert db.execute('SELECT id,label,amount FROM probe ORDER BY id').fetchall() == [
        (1, 'direct-compact', None), (2, 'direct-spaced', 2),
        (3, 'many-null', None), (4, 'many-known', 4),
        (5, 'queued-null', None), (6, 'queued-known', 6),
        (7, 'many-ignore', 7), (8, 'queued-ignore', None),
    ]
    with pytest.raises(duckdb.CatalogException):
        raw.execute('SELECT * FROM _ledger_batch')
    raw.close()


def test_bounded_parameter_insert_guard_leaves_other_sql_shapes_to_duckdb():
    raw = duckdb.connect(':memory:')
    db = Connection(raw)
    db.execute('CREATE TABLE probe(a BIGINT PRIMARY KEY,b BIGINT)')
    db.execute('INSERT INTO probe VALUES(1,2)')
    db.execute('INSERT INTO "probe"(a,b) VALUES(?,?)', (3, 4))
    db.execute('INSERT INTO probe SELECT 5,6')
    db.execute('UPDATE probe SET b=? WHERE a=?', (7, 5))
    db.execute('INSERT INTO probe VALUES(?,?) ON CONFLICT DO NOTHING', (5, 8))
    assert db.execute('SELECT a,b FROM probe ORDER BY a').fetchall() == [(1, 2), (3, 4), (5, 7)]
    raw.close()


def test_native_duckdb_ledger_roundtrip_and_rollback(tmp_path):
    store = Storage(tmp_path / 'ledger.duckdb')
    data = (Path(__file__).parent / 'fixtures/pi/source/ordinary.jsonl').read_bytes()
    store.import_source('/fixture', data)
    expected = store.snapshot()
    with store.connect() as db:
        assert db.execute('SELECT version()').fetchone()[0].startswith('v1.5.')
    with pytest.raises(RuntimeError):
        with store.connect() as db:
            db.execute("UPDATE source_generation SET availability='missing'")
            raise RuntimeError('abort')
    assert store.snapshot() == expected
    assert Storage(store.path).snapshot() == expected


def test_one_running_import_and_progress_are_checked(tmp_path):
    store = Storage(tmp_path / 'ledger.duckdb')
    assert store.begin_import('one').run_id == 'one'
    assert store.begin_import('two').run_id == 'one'
    store.finish_import('one', 0, None)
    assert store.begin_import('two').run_id == 'two'


@pytest.mark.parametrize('committed', [False, True])
def test_abrupt_process_exit_recovers_only_committed_imports(tmp_path, committed):
    import os
    import subprocess
    import sys
    path = tmp_path / 'ledger.duckdb'
    store = Storage(path)
    store.close()
    fixture = Path(__file__).parent / 'fixtures/pi/source/ordinary.jsonl'
    code = '''
import os, sys
from pathlib import Path
from harness_usage.storage import Storage
store = Storage(Path(sys.argv[1]))
if sys.argv[3] == 'False':
    original = store._reconcile
    def crash(*args, **kwargs):
        original(*args, **kwargs)
        os._exit(17)
    store._reconcile = crash
store.import_source('/fixture', Path(sys.argv[2]).read_bytes())
os._exit(17)
'''
    result = subprocess.run([sys.executable, '-c', code, str(path), str(fixture), str(committed)],
                            env={**os.environ, 'PYTHONPATH': str(Path.cwd() / 'src')}, timeout=30)
    assert result.returncode == 17
    restored = Storage(path)
    assert restored.snapshot().revision == int(committed)
    assert bool(restored.snapshot().observations) is committed
    assert restored.import_source('/fixture', fixture.read_bytes()) == 1
    restored.close()


def test_native_batch_rejects_fractional_integers(tmp_path):
    store = Storage(tmp_path / 'ledger.duckdb')
    with pytest.raises(ValueError, match='invalid_database_integer'):
        with store.connect() as db:
            db.executemany('INSERT INTO ledger_meta VALUES(?,?,?)', [(1,5,1.5)])
    assert store.snapshot().revision == 0


def test_native_parameter_insert_rejects_fractional_token_amount(tmp_path):
    store = Storage(tmp_path / 'ledger.duckdb')
    with store.connect(write=True) as db:
        db.execute("INSERT INTO session(id,harness,native_id) VALUES('pi:s','pi','s')")
        db.execute("INSERT INTO session_attribution VALUES('pi:s',NULL,NULL,'unknown')")
        db.execute("INSERT INTO observation(id,session_id,native_entry_id,fingerprint,kind,time_kind,at_us,time_reason,safe_facts_json) VALUES('o','pi:s','e','digest','assistant','point',1,'response_recorded_at','{}')")
    with pytest.raises(ValueError, match='invalid_database_integer'), store.connect(write=True) as db:
        db.execute('INSERT INTO token_value VALUES(?,?,?,?,?)', ('o', 'output', 'known', 1.5, None))
    with store.connect() as db:
        assert db.execute('SELECT count(*) FROM token_value').one()[0] == 0


@pytest.mark.parametrize('kind', ['request_summary', 'usage_checkpoint'])
def test_schema_six_observation_kinds_round_trip_through_storage(tmp_path, kind):
    store = Storage(tmp_path / 'ledger.duckdb')
    data = (Path(__file__).parent / 'fixtures/pi/source/ordinary.jsonl').read_bytes()
    store.import_source('/fixture', data)
    base = store.snapshot().observations[0].record
    with store.connect(write=True) as db:
        source_id = db.execute("SELECT id FROM source_generation WHERE locator='/fixture'").one()[0]
        store._insert_observation(db, source_id, 'pi:74811efe-ab6b-5971-88ac-bce0d809fe49',
                                  replace(base, entry=EntryIdentity(kind, None, 100), kind=kind))
    assert kind in {row.record.kind for row in store.snapshot().observations}
