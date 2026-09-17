"""Real HTTP routes, branch downloads and hostile content use the same safe reader."""
import json
from pathlib import Path

from fastapi.testclient import TestClient

from harness_usage.application import Application
from harness_usage.web import create_app

CLAUDE = '11111111-1111-4111-8111-111111111111'


def test_live_and_download_routes_share_registered_readonly_source(tmp_path):
    root=tmp_path/'sources';root.mkdir()
    source=root/'session.jsonl'
    rows=[dict(type='session',version=3,id='test',cwd='/project'),
          dict(type='message',id='a',parentId=None,message=dict(role='user',content='First request')),
          dict(type='message',id='b',parentId='a',message=dict(role='assistant',content=[dict(type='text',text='<script>alert(1)</script>')])),
          dict(type='message',id='c',parentId='a',message=dict(role='assistant',content=[dict(type='text',text='Alternate reply')]))]
    source.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    original=source.read_bytes()
    app=Application(tmp_path/'data');app.set_roots((str(root),))
    with app.storage.connect() as db:
        db.execute("INSERT INTO session(id,harness,native_id) VALUES('pi:test','pi','test')")
        db.execute("INSERT INTO session_attribution VALUES('pi:test',NULL,NULL,'unknown')")
        db.execute("INSERT INTO source_generation VALUES('source',?,0,?,'pi:test','pi-v3',?,0,'available')",(str(source),'0'*64,len(original)))
    with TestClient(create_app(app),base_url='http://127.0.0.1:8765') as client:
        live=client.get('/sessions/pi%3Atest/transcript')
        assert live.status_code == 200
        assert 'Alternate reply' in live.text
        assert 'script-src \'self\'' in live.headers['content-security-policy']
        branch=client.get('/sessions/pi%3Atest/transcript?branch=b')
        assert branch.status_code == 200
        assert '<script>alert(1)</script>' not in branch.text
        assert '&lt;script&gt;' in branch.text
        download=client.get('/sessions/pi%3Atest/transcript.html?branch=b')
        assert download.status_code == 200
        assert download.headers['content-disposition'].startswith('attachment;')
        assert 'sha256-' in download.headers['content-security-policy']
        assert 'data:font/' in download.text
        assert 'Alternate reply' not in download.text
        assert client.get('/sessions/pi%3Atest/transcript?branch=absent').status_code == 422
        assert client.get('/sessions/pi%3Aabsent/transcript').status_code == 404
        source.unlink()
        assert client.get('/sessions/pi%3Atest/transcript').status_code == 404
    assert not source.exists()


def test_claude_live_and_download_routes_escape_content_without_fetching_references(tmp_path):
    root = tmp_path / 'sources'; root.mkdir()
    source = root / f'{CLAUDE}.jsonl'
    rows = [
        {'type': 'user', 'uuid': 'user', 'parentUuid': None, 'sessionId': CLAUDE,
         'message': {'content': [{'type': 'text', 'text': '<script>claude()</script>'}]}},
        {'type': 'assistant', 'uuid': 'answer', 'parentUuid': 'user', 'sessionId': CLAUDE,
         'requestId': 'request', 'message': {'id': 'message', 'model': 'invented-model',
         'usage': {'input_tokens': 2}, 'content': [{'type': 'image', 'source': {'url': 'https://invalid.example/canary'}}]}},
    ]
    source.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    application = Application(tmp_path / 'data'); application.set_roots((str(root),))
    application.storage.import_source(str(source), source.read_bytes())
    with TestClient(create_app(application), base_url='http://127.0.0.1:8765') as client:
        for suffix in ('transcript', 'transcript.html'):
            response = client.get(f'/sessions/claude%3A{CLAUDE}/{suffix}')
            assert response.status_code == 200
            assert '<script>claude()</script>' not in response.text
            assert '&lt;script&gt;claude()&lt;/script&gt;' in response.text
            assert 'https://invalid.example' not in response.text
        source.write_text(source.read_text().replace(CLAUDE, '22222222-2222-4222-8222-222222222222'))
        assert client.get(f'/sessions/claude%3A{CLAUDE}/transcript').status_code == 409
        source.unlink()
        assert client.get(f'/sessions/claude%3A{CLAUDE}/transcript').status_code == 404


def test_copilot_vscode_route_uses_core_canonical_source_and_keeps_urls_inert(tmp_path):
    fixture = Path(__file__).parent / 'fixtures/copilot_vscode/session-v3.jsonl'
    root = tmp_path / 'User'
    chats = root / 'workspaceStorage/one/chatSessions'
    chats.mkdir(parents=True)
    source = chats / 'session.jsonl'
    source.write_bytes(fixture.read_bytes())
    extension = root / 'workspaceStorage/one/GitHub.copilot-chat/transcripts'
    extension.mkdir(parents=True)
    (extension / 'session.jsonl').write_text('{"private":"EXTENSION_CANARY"}\n')
    application = Application(tmp_path / 'data'); application.set_roots((str(root),))
    application.start_import(); application.close()
    session_id = str(application.storage.snapshot().sessions[0].id)
    with TestClient(create_app(application), base_url='http://127.0.0.1:8765') as client:
        response = client.get('/sessions/' + session_id.replace(':', '%3A') + '/transcript')
        assert response.status_code == 200
        assert '&lt;b&gt;safe&lt;/b&gt;' in response.text
        assert 'href="javascript:' not in response.text
        assert 'user:secret' not in response.text
        assert 'EXTENSION_CANARY' not in response.text
        assert 'Attachment: reference' in response.text


def test_cli_agent_transcript_links_and_metadata_only_unavailable_route(tmp_path):
    fixture = Path(__file__).parent / 'fixtures/copilot_cli'
    root = tmp_path / 'copilot'
    directory = root / 'session-state' / CLAUDE
    directory.mkdir(parents=True)
    source = directory / 'events.jsonl'
    source.write_bytes((fixture / 'current/events.jsonl').read_bytes())
    application = Application(tmp_path / 'data'); application.set_roots((str(root),))
    application.storage.import_source(str(source), source.read_bytes())
    parent = 'copilot-cli:' + CLAUDE
    child = parent + ':agent:agent-child'
    legacy_id = 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa'
    legacy_dir = root / 'session-state' / legacy_id
    legacy_dir.mkdir()
    legacy = legacy_dir / 'workspace.yaml'
    legacy.write_bytes((fixture / 'legacy/workspace.yaml').read_bytes())
    application.storage.import_source(str(legacy), legacy.read_bytes())
    with TestClient(create_app(application), base_url='http://127.0.0.1:8765') as client:
        parent_response = client.get('/sessions/' + parent.replace(':', '%3A') + '/transcript')
        assert parent_response.status_code == 200
        assert 'SYNTHETIC_USER' in parent_response.text and 'SYNTHETIC_CHILD' not in parent_response.text
        assert 'Subagents' in parent_response.text
        child_response = client.get('/sessions/' + child.replace(':', '%3A') + '/transcript')
        assert child_response.status_code == 200
        assert 'SYNTHETIC_CHILD' in child_response.text and 'SYNTHETIC_USER' not in child_response.text
        unavailable = client.get('/sessions/copilot-cli%3A' + legacy_id + '/transcript')
        assert unavailable.status_code == 422
        application.storage.mark_missing((str(source),))
        assert client.get('/sessions/' + parent.replace(':', '%3A') + '/transcript').status_code == 404
        assert client.get('/sessions/' + child.replace(':', '%3A') + '/transcript').status_code == 404
    application.close()
