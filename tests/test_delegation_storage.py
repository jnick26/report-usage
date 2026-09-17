"""Subagent ownership changes presentation, never accounting selection."""
import json
from pathlib import Path
import sqlite3

from harness_usage.domain import Assigned, ProjectId
from harness_usage.reporting import AllTime, ReportQuery, build_report
from harness_usage.storage import Storage


def source(identity, amount=10, *, parent=None, name=None):
    header = {'type':'session','version':3,'id':identity,'timestamp':'2026-09-12T00:00:00Z','cwd':'/project'}
    if parent is not None: header['parentSession']=parent
    rows=[header]
    if name is not None: rows.append({'type':'session_info','id':identity+'-name','name':name})
    rows.append({'type':'message','id':identity+'-usage','timestamp':'2026-09-12T01:00:00Z','message':{'role':'assistant','stopReason':'stop','provider':'test','model':'test','usage':{'input':amount,'output':0,'cacheRead':0,'cacheWrite':0,'totalTokens':amount,'cost':{'total':0}}}})
    return ('\n'.join(json.dumps(row) for row in rows)+'\n').encode()


def reference(storage, owner, child, *, entry='delegation', kind='child_path', owner_path=None):
    with storage.connect() as db:
        sid=db.execute('SELECT id FROM source_generation WHERE locator=?',(owner,)).fetchone()[0]
        db.execute('INSERT INTO delegation_ref VALUES(?,?,?,?,?)',(sid,entry,kind,child,owner_path or ''))


def report(storage, project=None):
    query=ReportQuery(project,AllTime())
    data=storage.report_input(query)
    return build_report(data.contributions, revision=data.revision, query=query),data


def test_explicit_children_use_main_project_and_family_totals(tmp_path):
    storage=Storage(tmp_path/'ledger.sqlite3')
    storage.import_sources([('/main',source('main')),('/child',source('child',20)),('/other',source('other',5))])
    reference(storage,'/main','/child')
    storage.save_attributions({'pi:main':Assigned(ProjectId('directory:/main-project'),'/main-project','recorded_directory'),'pi:child':Assigned(ProjectId('directory:/child-project'),'/child-project','recorded_directory')})
    whole,_=report(storage)
    scoped,_=report(storage,ProjectId('directory:/main-project'))
    assert whole.tokens.total.known==35 and whole.total_session_count==2
    assert scoped.tokens.total.known==30 and scoped.total_session_count==1
    assert scoped.sessions[0].id=='pi:main' and scoped.sessions[0].subagent_count==1
    assert report(storage,ProjectId('directory:/child-project'))[0].total_session_count==0


def test_parent_header_alone_is_not_delegation_and_copied_reference_has_one_owner(tmp_path):
    storage=Storage(tmp_path/'ledger.sqlite3')
    storage.import_sources([('/main',source('main')),('/fork',source('fork',parent='/main')),('/child',source('child'))])
    assert report(storage)[0].total_session_count==3
    reference(storage,'/main','/child')
    reference(storage,'/fork','/child')
    result,_=report(storage)
    assert result.total_session_count==2
    assert {s.id:s.subagent_count for s in result.sessions}=={'pi:main':1,'pi:fork':0}


def test_ambiguous_owners_and_cycles_remain_separate_with_coverage(tmp_path):
    storage=Storage(tmp_path/'ledger.sqlite3')
    storage.import_sources([(p,source(p[1:])) for p in ('/one','/two','/child')])
    reference(storage,'/one','/child')
    reference(storage,'/two','/child')
    result,data=report(storage)
    assert result.total_session_count==3 and 'ambiguous_subagent_owner' in data.diagnostics
    with storage.connect() as db: db.execute('DELETE FROM delegation_ref')
    reference(storage,'/one','/two')
    reference(storage,'/two','/one')
    result,data=report(storage)
    assert result.total_session_count==3 and 'cyclic_subagent_owner' in data.diagnostics


def test_agent_identity_needs_matching_parent_header_and_unique_suffix(tmp_path):
    storage=Storage(tmp_path/'ledger.sqlite3')
    storage.import_sources([('/main',source('main')),('/child',source('child',parent='/main',name='worker#12345678')),('/unproven',source('unproven',name='worker#87654321'))])
    reference(storage,'/main','12345678-agent',kind='agent_id')
    reference(storage,'/main','87654321-agent',entry='other',kind='agent_id')
    result,_=report(storage)
    assert result.total_session_count==2 and next(s for s in result.sessions if s.id=='pi:main').subagent_count==1


def downgrade_fixture_to_v1(path):
    with sqlite3.connect(path) as db:
        db.executescript('''
            DROP TABLE delegation_ref;
            DROP TABLE delegation_scan;
            ALTER TABLE session DROP COLUMN title_excerpt;
            CREATE TABLE old_meta(singleton INTEGER PRIMARY KEY CHECK(singleton=1),schema_version INTEGER NOT NULL CHECK(schema_version=1),revision INTEGER NOT NULL CHECK(revision>=0)) STRICT;
            INSERT INTO old_meta SELECT singleton,1,revision FROM ledger_meta;
            DROP TABLE ledger_meta;
            ALTER TABLE old_meta RENAME TO ledger_meta;
        ''')


def test_v1_migration_backs_up_and_backfills_metadata_without_accounting_replay(tmp_path, monkeypatch):
    import harness_usage.storage as module
    path=tmp_path/'ledger.sqlite3'
    storage=Storage(tmp_path/'seed.duckdb')
    delegated=source('main')+json.dumps({'type':'message','id':'spawn','message':{'role':'toolResult','toolName':'subagent','details':{'results':[{'sessionFile':'/child'}]}}}).encode()+b'\n'
    storage.import_sources([('/main',delegated),('/child',source('child'))])
    before=storage.snapshot()
    from legacy_fixture import export_legacy
    export_legacy(storage, path)
    downgrade_fixture_to_v1(path)
    migrated=Storage(path)
    backups=[path]
    assert len(backups)==1
    with sqlite3.connect(backups[0]) as backup:
        assert backup.execute('SELECT schema_version FROM ledger_meta').fetchone()[0]==1
        assert backup.execute('SELECT count(*) FROM observation').fetchone()[0]==2
    assert migrated.snapshot()==before
    def forbidden(*args,**kwargs):
        raise AssertionError('Metadata backfill must not replay accounting')
    monkeypatch.setattr(module,'read_pi',forbidden)
    monkeypatch.setattr(migrated,'_reconcile',forbidden)
    revision=migrated.import_sources([('/main',delegated),('/child',source('child'))])
    assert revision==before.revision+1
    after=migrated.snapshot()
    assert after.observations==before.observations
    assert report(migrated)[0].total_session_count==1
    assert migrated.import_sources([('/main',delegated),('/child',source('child'))])==revision
    assert path.exists()


def test_migration_failure_rolls_back_whole_schema(tmp_path, monkeypatch):
    from contextlib import contextmanager
    path=tmp_path/'ledger.sqlite3'
    from legacy_fixture import export_legacy
    export_legacy(Storage(tmp_path/'seed.duckdb'), path)
    downgrade_fixture_to_v1(path)
    import harness_usage.migrate_sqlite as migration
    def fail(*args):
        raise RuntimeError('injected migration failure')
    monkeypatch.setattr(migration, '_copy', fail)
    import pytest
    with pytest.raises(RuntimeError, match='injected'): Storage(path)
    assert not path.with_suffix('.duckdb').exists()
    with sqlite3.connect(path) as db:
        assert db.execute('SELECT schema_version FROM ledger_meta').fetchone()[0]==1
        assert db.execute("SELECT count(*) FROM sqlite_master WHERE name IN ('delegation_ref','delegation_scan','ledger_meta_v3')").fetchone()[0]==0
    assert path.exists()


def test_future_schema_is_rejected_without_mutation_or_backup(tmp_path):
    import pytest
    path=tmp_path/'ledger.sqlite3'
    with sqlite3.connect(path) as db:
        db.executescript('CREATE TABLE ledger_meta(singleton INTEGER,schema_version INTEGER,revision INTEGER);INSERT INTO ledger_meta VALUES(1,99,0);')
    before=path.read_bytes()
    with pytest.raises(ValueError,match='unsupported_legacy_schema'): Storage(path)
    assert path.read_bytes()==before
    assert not list(tmp_path.glob('*.v1-backup-*.sqlite3'))


def test_self_delegation_is_reported_as_cycle(tmp_path):
    storage=Storage(tmp_path/'ledger.sqlite3')
    storage.import_source('/main',source('main'))
    reference(storage,'/main','/main')
    result,data=report(storage)
    assert result.total_session_count==1 and 'cyclic_subagent_owner' in data.diagnostics


def test_canonical_path_hash_is_persisted_and_report_does_not_resolve_paths(tmp_path, monkeypatch):
    from harness_usage.storage import digest
    target=tmp_path/'actual.jsonl'
    alias=tmp_path/'alias.jsonl'
    target.write_bytes(source('child'))
    alias.symlink_to(target)
    storage=Storage(tmp_path/'ledger.sqlite3')
    storage.import_sources([('/main',source('main')),(str(alias),target.read_bytes())])
    reference(storage,'/main',digest(str(target)),kind='child_path_hash')
    alias.unlink()
    def no_filesystem(*args,**kwargs):
        raise AssertionError('Report must use stored canonical identity')
    monkeypatch.setattr('harness_usage.storage.os.path.realpath',no_filesystem)
    result,_=report(storage)
    assert result.total_session_count==1 and result.sessions[0].subagent_count==1


def test_hash_with_multiple_native_identities_is_not_resolved(tmp_path):
    from harness_usage.storage import digest
    target=tmp_path/'actual.jsonl'
    alias=tmp_path/'alias.jsonl'
    target.write_bytes(source('child'))
    alias.symlink_to(target)
    storage=Storage(tmp_path/'ledger.sqlite3')
    storage.import_sources([('/main',source('main')),(str(target),source('child')),(str(alias),source('other'))])
    reference(storage,'/main',digest(str(target)),kind='child_path_hash')
    result,data=report(storage)
    assert result.total_session_count==3 and 'missing_subagent_history' in data.diagnostics


def test_nested_descendants_keep_root_identity_when_only_grandchild_is_in_range(tmp_path):
    from harness_usage.reporting import parse_range
    storage=Storage(tmp_path/'ledger.sqlite3')
    storage.import_sources([('/main',source('main')),('/child',source('child').replace(b'2026-09-12',b'2026-09-13')),('/grandchild',source('grandchild').replace(b'2026-09-12',b'2026-09-14'))])
    reference(storage,'/main','/child')
    reference(storage,'/child','/grandchild')
    query=ReportQuery(None,parse_range('2026-09-14T00:00Z','2026-09-15T00:00Z','UTC'))
    data=storage.report_input(query)
    result=build_report(data.contributions,revision=data.revision,query=query)
    assert result.total_session_count==1 and result.sessions[0].id=='pi:main'
    assert result.sessions[0].started.day==12 and data.contributions[0].family.last_observed.day==14
    assert result.tokens.total.known==10 and result.sessions[0].subagent_count==1


def test_explicit_intercom_name_matches_exactly_and_requires_unique_identity(tmp_path):
    storage=Storage(tmp_path/'ledger.sqlite3')
    storage.import_sources([('/main',source('main')),('/child',source('child',name='subagent-worker-exact-run')),('/unproven',source('unproven',name='subagent-worker-other-run'))])
    reference(storage,'/main','subagent-worker-exact-run',kind='child_name')
    result,_=report(storage)
    assert result.total_session_count==2 and next(s for s in result.sessions if s.id=='pi:main').subagent_count==1
    storage.import_source('/collision',source('collision',name='subagent-worker-exact-run'))
    result,data=report(storage)
    assert result.total_session_count==4 and 'ambiguous_subagent_owner' in data.diagnostics


def test_updated_delegation_profile_backfills_once_without_accounting_replay(tmp_path, monkeypatch):
    import harness_usage.storage as module
    storage=Storage(tmp_path/'ledger.sqlite3')
    data=source('main')
    storage.import_source('/main',data)
    before=storage.snapshot()
    monkeypatch.setattr(module,'DELEGATION_PROFILE','new-linkage-profile')
    def forbidden(*args): raise AssertionError('Accounting replay is unnecessary')
    monkeypatch.setattr(storage,'_reconcile',forbidden)
    assert storage.import_source('/main',data)==before.revision+1
    assert storage.snapshot().observations==before.observations
    assert storage.import_source('/main',data)==before.revision+1
