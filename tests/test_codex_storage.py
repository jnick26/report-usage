"""Codex persistence and schema upgrade checks use synthetic sources only."""
from contextlib import contextmanager
from pathlib import Path
import sqlite3

import pytest

from harness_usage.storage import Storage

PI = Path(__file__).parent / 'fixtures/pi/source/ordinary.jsonl'


def v3_fixture(path: Path) -> None:
    """Freeze the actual preceding schema shape, without consulting a live ledger."""
    schema = Path('src/harness_usage/legacy_schema.sql').read_text().split('-- Codex source accounting evidence.')[0]
    schema = schema.replace('schema v4', 'schema v3').replace('schema_version=4', 'schema_version=3').replace('VALUES(1,4,0)', 'VALUES(1,3,0)')
    schema = schema.replace("CHECK(harness IN ('pi','codex'))", "CHECK(harness='pi')")
    with sqlite3.connect(path) as db:
        db.executescript(schema)
        db.execute("INSERT INTO session(id,harness,native_id,attribution_reason) VALUES('pi:retained','pi','retained','not_resolved')")
        db.execute('UPDATE ledger_meta SET revision=7')


def test_v3_upgrade_backs_up_and_preserves_pi_identity_and_revision(tmp_path: Path) -> None:
    path = tmp_path / 'ledger.sqlite3'
    v3_fixture(path)
    storage = Storage(path)
    assert storage.snapshot().revision == 7
    assert storage.snapshot().sessions[0].id == 'pi:retained'
    with storage.connect() as db:
        assert db.execute('SELECT schema_version FROM ledger_meta').fetchone()[0] == 6
        db.execute("INSERT INTO session(id,harness,native_id) VALUES('codex:retained','codex','retained')")
        db.execute("INSERT INTO session_attribution VALUES('codex:retained',NULL,NULL,'not_resolved')")
    backups = [path]
    assert len(backups) == 1
    with sqlite3.connect(backups[0]) as db:
        assert db.execute('SELECT schema_version FROM ledger_meta').fetchone()[0] == 3
        assert db.execute('SELECT revision FROM ledger_meta').fetchone()[0] == 7
        assert db.execute('SELECT COUNT(*) FROM session').fetchone()[0] == 1
    Storage(path)
    assert path.exists()


def test_v3_upgrade_failure_rolls_back_and_keeps_backup(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / 'ledger.sqlite3'; v3_fixture(path)
    import harness_usage.migrate_sqlite as migration
    def fail(*args):
        raise RuntimeError('injected migration failure')
    monkeypatch.setattr(migration, '_copy', fail)
    with pytest.raises(RuntimeError, match='injected'):
        Storage(path)
    assert not path.with_suffix('.duckdb').exists()
    with sqlite3.connect(path) as db:
        assert db.execute('SELECT schema_version FROM ledger_meta').fetchone()[0] == 3
        assert db.execute('SELECT id FROM session').fetchone()[0] == 'pi:retained'
    assert path.exists()


import json
from harness_usage.domain import Known
from harness_usage.reporting import AllTime, ReportQuery, build_report


def usage(output=10, input=100):
    return {'input_tokens':input, 'cached_input_tokens':20, 'cache_write_input_tokens':0,
            'output_tokens':output, 'reasoning_output_tokens':2, 'total_tokens':input+output}


def codex_source(identity, *events, parent=None, fork=None):
    meta = {'id':identity, 'timestamp':'2026-09-12T08:00:00Z', 'cwd':'/project', 'model_provider':'openai'}
    if parent: meta['parent_thread_id'] = parent
    if fork: meta['forked_from_id'] = fork
    records = [{'type':'session_meta', 'timestamp':'2026-09-12T08:00:00Z', 'payload':meta},
               {'type':'turn_context', 'timestamp':'2026-09-12T08:00:00Z', 'payload':{'turn_id':'turn', 'model':'gpt-5', 'cwd':'/project'}}, *events]
    return ('\n'.join(map(json.dumps, records))+'\n').encode()


def modern(response='response', thread='root', output=10, at='2026-09-12T09:00:00Z'):
    return {'type':'token_usage_record','timestamp':at,'payload':{'response_id':response,'thread_id':thread,'turn_id':'turn','usage':usage(output),'thread_token_usage':usage(output)}}


def legacy(last, cumulative, at='2026-09-12T09:00:00Z'):
    return {'type':'event_msg','timestamp':at,'payload':{'type':'token_count','info':{'last_token_usage':last,'total_token_usage':cumulative}}}


def selected_output(storage):
    return sum(o.record.tokens.buckets.output.value for o in storage.snapshot().observations
               if o.decisions['output']=='selected' and isinstance(o.record.tokens.buckets.output, Known))


def test_modern_import_dedupes_archive_and_cross_thread_copies_without_pi_collision(tmp_path):
    store=Storage(tmp_path/'ledger.sqlite3')
    store.import_source('/pi', PI.read_bytes())
    root=codex_source('root', modern())
    store.import_source('/root', root)
    store.import_source('/archive/root', root)
    store.import_source('/copy', codex_source('fork', modern(at='2026-09-13T09:00:00Z'), fork='root'))
    assert selected_output(store)==30
    assert any(s.id=='codex:root' for s in store.snapshot().sessions)
    assert any(s.id.startswith('pi:') for s in store.snapshot().sessions)
    assert all(o.record.money.reason=='not_recorded' for o in store.snapshot().observations if o.session_id.startswith('codex:'))
    revision=store.snapshot().revision
    store.import_source('/root', root)
    assert store.snapshot().revision==revision


def test_modern_conflict_preserves_agreeing_measures_and_quarantines_output(tmp_path):
    store=Storage(tmp_path/'ledger.sqlite3')
    store.import_source('/first', codex_source('root',modern(output=10)))
    store.import_source('/second', codex_source('root',modern(output=11)))
    snapshot=store.snapshot()
    assert selected_output(store)==0
    assert sum(o.record.tokens.buckets.input.value for o in snapshot.observations if o.decisions['input']=='selected')==80
    assert all(o.decisions['output']=='unresolved' for o in snapshot.observations)


def test_legacy_replay_collapses_only_explicit_ancestor_and_subagent_groups(tmp_path):
    first=usage(10)
    second=usage(5,50); cumulative={name:first[name]+second[name] for name in first}
    store=Storage(tmp_path/'ledger.sqlite3')
    parent=codex_source('root',legacy(first,first))
    child=codex_source('child',legacy(first,first),legacy(second,cumulative, '2026-09-12T10:00:00Z'),parent='root')
    store.import_source('/child',child)
    store.import_source('/root',parent)
    assert selected_output(store)==15
    report=build_report(store.report_input(ReportQuery(None,AllTime())).contributions, revision=store.snapshot().revision,query=ReportQuery(None,AllTime()))
    assert report.total_session_count==1
    assert report.sessions[0].subagent_count==1
    independent=codex_source('independent',legacy(first,first))
    store.import_source('/independent',independent)
    assert selected_output(store)==25


def test_mixed_modern_legacy_mirror_counts_once_and_keeps_older_legacy(tmp_path):
    old=usage(3,30); new=usage(10)
    cumulative={name:old[name]+new[name] for name in old}
    response=modern(); response['payload']['thread_token_usage']=cumulative
    data=codex_source('root',legacy(old,old),response,legacy(new,cumulative))
    store=Storage(tmp_path/'ledger.sqlite3'); store.import_source('/mixed',data)
    assert selected_output(store)==13
    assert len(store.snapshot().observations)>=3


def test_codex_title_sidecar_survives_reimport_and_never_changes_pi(tmp_path):
    store=Storage(tmp_path/'ledger.sqlite3')
    store.import_source('/root',codex_source('root',modern()))
    store.import_source('/pi',PI.read_bytes())
    store.save_codex_titles({'codex:root':'Native Codex title','pi:other':'Wrong harness'})
    store.import_source('/root',codex_source('root',modern(),modern('next',output=3)))
    assert next(s for s in store.snapshot().sessions if s.id=='codex:root').display_name=='Native Codex title'
    assert all(s.display_name!='Wrong harness' for s in store.snapshot().sessions if s.id.startswith('pi:'))


def test_later_independent_legacy_match_is_not_a_copied_prefix(tmp_path):
    first=usage(2,30); second=usage(3,40); third=usage(4,50)
    prefix={name:first[name]+second[name] for name in first}
    final={name:prefix[name]+third[name] for name in first}
    store=Storage(tmp_path/'ledger.sqlite3')
    store.import_source('/parent',codex_source('root',legacy(first,first),legacy(second,prefix),legacy(third,final)))
    store.import_source('/child',codex_source('child',legacy(prefix,prefix),legacy(third,final),fork='root'))
    assert selected_output(store)==18


def test_duplicate_thread_header_timestamp_does_not_invalidate_responses(tmp_path):
    store=Storage(tmp_path/'ledger.sqlite3')
    store.import_source('/empty',codex_source('root'))
    later=codex_source('root',modern()).replace(b'2026-09-12T08:00:00Z',b'2026-09-12T08:01:00Z')
    store.import_source('/live',later)
    assert selected_output(store)==10
    session=store.snapshot().sessions[0]
    assert session.started.isoformat()=='2026-09-12T08:00:00+00:00'
    assert session.last_observed.isoformat()=='2026-09-12T09:00:00+00:00'
    assert 'codex_header_timestamp_variation' in store.snapshot().diagnostics


def test_deep_unknown_header_does_not_abort_other_sources(tmp_path):
    store=Storage(tmp_path/'ledger.sqlite3')
    store.import_sources((('/pi',PI.read_bytes()),('/broken',b'['*10000+b']'*10000+b'\n')))
    assert selected_output(store)==20
    assert 'invalid_header' in store.snapshot().diagnostics


def test_native_name_delegation_never_uses_other_harness(tmp_path):
    store=Storage(tmp_path/'ledger.sqlite3')
    original=PI.read_bytes()
    native=json.loads(original.splitlines()[0])['id']
    store.import_source('/pi-main',original.replace(native.encode(),b'main'))
    child=original.replace(native.encode(),b'child')+b'{"type":"session_info","id":"name","name":"same title"}\n'
    store.import_source('/pi-child',child)
    store.import_source('/codex',codex_source('codex-child',modern(thread='codex-child')))
    store.save_codex_titles({'codex:codex-child':'same title'})
    with store.connect() as db:
        sid=db.execute("SELECT id FROM source_generation WHERE locator='/pi-main'").fetchone()[0]
        db.execute('INSERT INTO delegation_ref VALUES(?,?,?,?,?)',(sid,'delegate','child_name','same title',''))
    inputs=store.report_input(ReportQuery(None,AllTime()))
    families={str(c.session_id):str(c.family.id) if c.family else None for c in inputs.contributions}
    assert families['pi:child']=='pi:main'
    assert families['codex:codex-child'] is None


def test_response_model_conflict_keeps_counts_but_does_not_choose_a_price_identity(tmp_path):
    store=Storage(tmp_path/'ledger.sqlite3')
    first=codex_source('root',modern())
    second=first.replace(b'gpt-5',b'gpt-6')
    store.import_sources((('/first',first),('/second',second)))
    assert selected_output(store)==10
    snapshot=store.snapshot()
    assert all(o.record.model.model is None and o.record.model.provider is None for o in snapshot.observations)
    with store.connect() as db:
        assert {r[0] for r in db.execute('SELECT model FROM observation')}=={'gpt-5','gpt-6'}
    inputs=store.report_input(ReportQuery(None,AllTime()))
    assert all(c.model.model is None for c in inputs.contributions)
    assert 'codex_model_conflict' in inputs.diagnostics
    from harness_usage.pricing import Catalog
    rates=Catalog.from_bytes(b'{"openai":{"models":{"gpt-5":{"cost":{"input":1,"output":2}},"gpt-6":{"cost":{"input":3,"output":4}}}}}',snapshot_date='2026-09-12',sha256='fixture')
    for contribution in inputs.contributions:
        priced=rates.price(contribution.model,contribution.tokens,contribution.decisions)
        assert priced.known==0 and priced.unpriced


def test_crossfile_legacy_and_modern_exact_cumulative_mirror_counts_once(tmp_path):
    store=Storage(tmp_path/'ledger.sqlite3')
    store.import_source('/legacy',codex_source('root',legacy(usage(),usage())))
    store.import_source('/modern',codex_source('root',modern()))
    assert selected_output(store)==10


def test_v3_upgrade_preserves_complete_pi_evidence_graph(tmp_path):
    current=Storage(tmp_path/'current.sqlite3')
    current.import_source('/pi',PI.read_bytes())
    before=current.snapshot()
    target=tmp_path/'v3.sqlite3'; v3_fixture(target)
    from legacy_fixture import export_legacy
    target.unlink()
    export_legacy(current, target)
    with sqlite3.connect(target) as db:
        db.execute('ALTER TABLE ledger_meta RENAME TO old_meta')
        db.execute('CREATE TABLE ledger_meta(singleton INTEGER PRIMARY KEY,schema_version INTEGER CHECK(schema_version=3),revision INTEGER)')
        db.execute('INSERT INTO ledger_meta SELECT singleton,3,revision FROM old_meta')
        db.execute('DROP TABLE old_meta')
    migrated=Storage(target)
    assert migrated.snapshot()==before
    with migrated.connect() as db:
        assert db.execute('SELECT COUNT(*) FROM decision').fetchone()[0]>0


def test_modern_child_work_ends_inherited_legacy_prefix(tmp_path):
    store=Storage(tmp_path/'ledger.sqlite3')
    store.import_source('/parent',codex_source('root',legacy(usage(),usage())))
    context={'type':'turn_context','timestamp':'2026-09-12T09:30:00Z','payload':{'turn_id':'next','model':'gpt-5'}}
    store.import_source('/child',codex_source('child',modern('own',thread='child',output=5),context,legacy(usage(),usage()),fork='root'))
    assert selected_output(store)==25
