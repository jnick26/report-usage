"""Reuse the reviewed forbidden combinations against the installed migration."""
from pathlib import Path
import duckdb
import pytest
from harness_usage.storage import Storage
from harness_usage.domain import ContractViolation


def test_reviewed_schema_constraints_on_real_initialized_database(tmp_path):
    store = Storage(tmp_path / 'ledger.duckdb')
    with store.connect(write=True) as db:
        db.execute("INSERT INTO session(id,harness,native_id) VALUES('pi:s','pi','s')")
        db.execute("INSERT INTO session_attribution VALUES('pi:s',NULL,NULL,'unknown')")
        db.execute("INSERT INTO observation(id,session_id,native_entry_id,fingerprint,kind,time_kind,at_us,time_reason,safe_facts_json) VALUES('o','pi:s','e','digest','assistant','point',1,'response_recorded_at','{}')")
    # The same sixteen forbidden combinations as fixtures/pi/check_design.py.
    # Each failing native statement needs its own transaction; DuckDB has no savepoints.
    forbidden = [
        "INSERT INTO token_value VALUES('o','output','known',-1,NULL)",
        "INSERT INTO token_value VALUES('o','output','unknown',0,'missing')",
        "INSERT INTO token_value VALUES('o','output','not_applicable',0,'unsupported')",
        "INSERT INTO token_value VALUES('o','output','known',NULL,NULL)",
        "INSERT INTO token_value VALUES('missing','output','known',0,NULL)",
        "INSERT INTO token_value VALUES('o','output','known',1.5,NULL)",
        "INSERT INTO token_value VALUES('o','output','known',0,NULL)",
        "INSERT INTO recorded_estimate VALUES('o','missing','0','USD','{}','source','missing')",
        "INSERT INTO recorded_estimate VALUES('o','known','0',NULL,'{}','source',NULL)",
        "INSERT INTO decision VALUES('o','output','selected',NULL,NULL,'rule','1')",
        "INSERT INTO decision VALUES('o','output','unresolved','pi:s',NULL,'reason','1')",
        "INSERT INTO decision VALUES('o','output','excluded',NULL,'o','rule','1')",
        "INSERT INTO observation(id,session_id,native_entry_id,fingerprint,kind,time_kind,at_us,safe_facts_json) VALUES('bad','pi:s','bad','x','assistant','point',1,'{}')",
        "INSERT INTO observation(id,session_id,native_entry_id,fingerprint,kind,time_kind,start_us,end_us,safe_facts_json) VALUES('bad','pi:s','bad','x','compaction','interval',5,4,'{}')",
        "INSERT INTO import_run(id,state,started_us,finished_us,files_processed,revision,error_code) VALUES('run2','running',1,NULL,0,0,NULL)",
        "INSERT INTO import_run(id,state,started_us,finished_us,files_processed,revision,error_code) VALUES('run3','failed',1,2,0,0,NULL)",
    ]
    for index, sql in enumerate(forbidden):
        if index == 5:
            with pytest.raises(ValueError, match='invalid_database_integer'), store.connect(write=True) as db:
                db.execute('INSERT INTO token_value VALUES(?,?,?,?,?)', ('o', 'output', 'known', 1.5, None))
            continue
        with store.connect(write=True) as db:
            if index == 6:
                db.execute("INSERT INTO token_value VALUES('o','output','known',0,NULL)")
            if index == 9:
                db.execute("INSERT INTO recorded_estimate VALUES('o','known','0','USD','{}','source',NULL)")
            if index == 14:
                db.execute("INSERT INTO import_run(id,state,started_us,finished_us,files_processed,revision,error_code) VALUES('run','running',1,NULL,0,0,NULL)")
        with pytest.raises(duckdb.ConstraintException), store.connect(write=True) as db:
            db.execute(sql)
    assert len(forbidden) == 16
    with store.connect() as db:
        before = db.execute('SELECT revision FROM ledger_meta').fetchone()[0]
    with pytest.raises(RuntimeError, match='rollback'), store.connect(write=True) as db:
        db.execute('UPDATE ledger_meta SET revision=revision+1')
        raise RuntimeError('rollback')
    with store.connect() as db:
        assert db.execute('SELECT revision FROM ledger_meta').fetchone()[0] == before


def test_reconstitution_rejects_cross_row_corrupted_subset(tmp_path):
    store = Storage(tmp_path / 'ledger.db')
    store.import_source('/fixture', (Path(__file__).parent / 'fixtures/pi/source/ordinary.jsonl').read_bytes())
    with store.connect(write=True) as db:
        db.execute("UPDATE token_value SET amount=99999 WHERE measure='reasoning'")
    with pytest.raises(ContractViolation, match='invalid_subset'):
        store.snapshot()


def test_schema_six_accepts_new_sources_and_rejects_invalid_accounting_values(tmp_path):
    store = Storage(tmp_path / 'ledger.duckdb')
    with store.connect(write=True) as db:
        db.execute("INSERT INTO session(id,harness,native_id) VALUES('copilot-vscode:s','copilot-vscode','s')")
        db.execute("INSERT INTO session_attribution VALUES('copilot-vscode:s',NULL,NULL,'unknown')")
        db.execute("INSERT INTO observation(id,session_id,native_entry_id,fingerprint,kind,time_kind,at_us,time_reason,safe_facts_json) VALUES('o','copilot-vscode:s','e','digest','request_summary','point',1,'response_recorded_at','{}')")
        db.execute("INSERT INTO quantity_value VALUES('o','ai_credits','known','0',NULL,0,'vscode:s')")
        db.execute("INSERT INTO quantity_decision VALUES('o','ai_credits','selected','copilot-vscode:s',NULL,'independent_evidence','quantity-1')")
    for sql in (
        "INSERT INTO session(id,harness,native_id) VALUES('bad','other','bad')",
        "INSERT INTO observation(id,session_id,native_entry_id,fingerprint,kind,time_kind,at_us,time_reason,safe_facts_json) VALUES('bad','copilot-vscode:s','bad','x','other','point',1,'response_recorded_at','{}')",
        "INSERT INTO quantity_value VALUES('o','nano_aiu','unknown','0','not_recorded',0,NULL)",
        "INSERT INTO quantity_value VALUES('o','premium_requests','known','-1',NULL,0,'cli:shutdown')",
    ):
        with pytest.raises(duckdb.ConstraintException), store.connect(write=True) as db:
            db.execute(sql)
    with store.connect() as db:
        names = {row[0] for row in db.execute("SELECT table_name FROM information_schema.tables WHERE table_name IN ('claude_evidence','copilot_vscode_evidence','copilot_cli_evidence')")}
    assert names == {'claude_evidence', 'copilot_vscode_evidence', 'copilot_cli_evidence'}
