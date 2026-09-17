"""Transcript access stays inside registered roots and outside usage accounting."""
import json
from pathlib import Path

import pytest

from harness_usage.storage import Storage
from harness_usage.transcript import TranscriptUnavailable
from harness_usage.transcript_access import read_transcript_page, source_snapshot


def register(storage, path, native='test-session', generation=0):
    sid = 'pi:' + native
    with storage.connect() as db:
        db.execute("INSERT OR IGNORE INTO session(id,harness,native_id,display_name) VALUES(?,'pi',?,'Readable title')", (sid,native))
        db.execute("INSERT OR IGNORE INTO session_attribution VALUES(?,NULL,NULL,'unknown')", (sid,))
        db.execute("INSERT INTO source_generation VALUES(?,?,?,?,?,?,?,0,'available')", (sid+str(generation),str(path),generation,'0'*64,sid,'pi-v3',path.stat().st_size))
    return sid


def pi_source(path, native='test-session'):
    rows = [dict(type='session',version=3,id=native,cwd='/example',timestamp='2026-09-13T00:00:00Z'),
            dict(type='message',id='one',parentId=None,message=dict(role='user',content=[dict(type='text',text='A real message')]))]
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))


def test_registered_source_and_family_reads_do_not_touch_observations(tmp_path, monkeypatch):
    source = tmp_path/'session.jsonl'
    pi_source(source)
    storage = Storage(tmp_path/'ledger.sqlite3')
    sid = register(storage,source)
    original_connect = storage.connect
    from contextlib import contextmanager
    @contextmanager
    def guarded():
        with original_connect() as db:
            import re
            original = db.execute
            def checked(sql, args=()):
                assert not re.search(r'\b(?:observation|token_value|decision)\b', sql)
                return original(sql, args)
            monkeypatch.setattr(db, 'execute', checked)
            yield db
    monkeypatch.setattr(storage,'connect',guarded)
    result = read_transcript_page(storage,(str(tmp_path),),sid)
    assert result.transcript.title == 'Readable title'
    assert result.transcript.entries[0].blocks[0].text == 'A real message'
    with pytest.raises(TranscriptUnavailable,match='not registered'):
        read_transcript_page(storage,(str(tmp_path),),'pi:absent')
    source.write_text(source.read_text().replace('test-session','different-session'))
    with pytest.raises(TranscriptUnavailable):
        read_transcript_page(storage,(str(tmp_path),),sid)


def test_source_snapshot_rejects_outside_symlinks_and_oversize(tmp_path, monkeypatch):
    root = tmp_path/'root';root.mkdir()
    outside = tmp_path/'private.jsonl';outside.write_bytes(b'private')
    for path in (outside,root/'link.jsonl'):
        if path != outside:path.symlink_to(outside)
        with pytest.raises(TranscriptUnavailable):source_snapshot(str(path),(str(root),))
    folder = root/'linked-folder';folder.symlink_to(tmp_path,target_is_directory=True)
    with pytest.raises(TranscriptUnavailable):source_snapshot(str(folder/'private.jsonl'),(str(root),))
    safe=root/'safe.jsonl';safe.write_bytes(b'12345')
    assert source_snapshot(str(safe),(str(root),)) == b'12345'
    monkeypatch.setattr('harness_usage.transcript_access.MAX_TRANSCRIPT_BYTES',4)
    with pytest.raises(TranscriptUnavailable) as result:source_snapshot(str(safe),(str(root),))
    assert result.value.kind == 'too_large'


def test_old_generation_cannot_open_reassigned_source(tmp_path):
    path=tmp_path/'session.jsonl';pi_source(path)
    storage=Storage(tmp_path/'ledger.sqlite3');sid=register(storage,path)
    register(storage,path,native='replacement',generation=1)
    with pytest.raises(TranscriptUnavailable,match='No original source'):
        read_transcript_page(storage,(str(tmp_path),),sid)


def test_family_navigation_reuses_explicit_codex_ownership(tmp_path):
    # The child view returns to its root; a root lists each child once.
    source=tmp_path/'root.jsonl'
    source.write_text(json.dumps(dict(type='session_meta',payload=dict(id='root',cwd='/example')))+'\n'+json.dumps(dict(type='response_item',payload=dict(type='message',role='user',content=[dict(type='input_text',text='Root request')])) )+'\n')
    child=tmp_path/'child.jsonl';child.write_text(source.read_text().replace('"root"','"child"'))
    storage=Storage(tmp_path/'ledger.sqlite3')
    with storage.connect() as db:
        for name,path in [('root',source),('child',child)]:
            db.execute("INSERT INTO session(id,harness,native_id,display_name) VALUES(?,'codex',?,?)",('codex:'+name,name,name.title()))
            db.execute("INSERT INTO session_attribution VALUES(?,NULL,NULL,'unknown')", ('codex:'+name,))
            db.execute("INSERT INTO source_generation VALUES(?,?,0,?,?,'codex-legacy',?,0,'available')",(name,str(path),'0'*64,'codex:'+name,path.stat().st_size))
        db.execute("INSERT INTO codex_source(source_id,parent_thread_id) VALUES('child','root')")
    root_page=read_transcript_page(storage,(str(tmp_path),),'codex:root')
    assert [link.session_id for link in root_page.children] == ['codex:child']
    assert root_page.parent is None
    child_page=read_transcript_page(storage,(str(tmp_path),),'codex:child')
    assert child_page.parent.session_id == 'codex:root'
    assert child_page.children == ()


def test_source_changed_during_snapshot_is_rejected(tmp_path,monkeypatch):
    import os
    from types import SimpleNamespace
    source=tmp_path/'session.jsonl';source.write_bytes(b'snapshot')
    original=os.fstat
    calls=0
    def changed(fd):
        nonlocal calls
        current=original(fd);calls+=1
        return SimpleNamespace(st_mode=current.st_mode,st_size=current.st_size,
                               st_mtime_ns=current.st_mtime_ns+(calls>1),st_ctime_ns=current.st_ctime_ns)
    monkeypatch.setattr('harness_usage.transcript_access.os.fstat',changed)
    with pytest.raises(TranscriptUnavailable) as result:source_snapshot(str(source),(str(tmp_path),))
    assert result.value.kind == 'changed'
