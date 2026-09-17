import json
import pytest

from harness_usage.source_input import SourcePayload, detect_source
from harness_usage.storage import Storage


PI_HEADER = b'{"type":"session","version":3,"id":"pi-session","cwd":"/work"}\n'
CODEX_HEADER = b'{"type":"session_meta","payload":{"id":"codex-session","cwd":"/work"}}\n'
CLAUDE_RECORD = b'{"type":"assistant","sessionId":"claude-session","uuid":"entry"}\n'
VSCODE_V3 = json.dumps({
    'version': 3,
    'sessionId': 'vs-session',
    'requests': [{'requestId': 'request', 'agent': {'extensionId': {'value': 'GitHub.copilot-chat'}}}],
}, separators=(',', ':')).encode()
VSCODE_V3_LOG = json.dumps({'kind': 0, 'v': json.loads(VSCODE_V3)}, separators=(',', ':')).encode() + b'\n'
CLI_START = b'{"id":"event","parentId":null,"timestamp":"2026-09-16T00:00:00Z","type":"session.start","data":{"sessionId":"cli-session","version":1,"producer":"copilot-cli","copilotVersion":"1.0.0","startTime":"2026-09-16T00:00:00Z"}}\n'
CLI_CHECKPOINT = b'{"id":"checkpoint","parentId":"event","timestamp":"2026-09-16T01:00:00Z","type":"session.usage_checkpoint","data":{"totalNanoAiu":1000000000,"totalPremiumRequests":2}}\n'
CLI_SHUTDOWN = b'{"id":"shutdown","parentId":"checkpoint","timestamp":"2026-09-16T02:00:00Z","type":"session.shutdown","data":{"sessionStartTime":1789516800000,"shutdownType":"routine","currentModel":"gpt-5","totalApiDurationMs":1200,"totalNanoAiu":1000000000,"totalPremiumRequests":2,"modelMetrics":{"gpt-5":{"requests":{"count":2},"usage":{"inputTokens":10,"outputTokens":5,"cacheReadTokens":0,"cacheWriteTokens":0}}},"codeChanges":{"filesModified":["file.py"],"linesAdded":2,"linesRemoved":0}}}\n'
CLI_EPHEMERAL_USAGE = b'{"id":"usage","parentId":"event","timestamp":"2026-09-16T01:00:00Z","type":"assistant.usage","data":{"model":"gpt-5"},"ephemeral":true}\n'


def test_structural_detection_distinguishes_all_five_sources():
    assert detect_source(SourcePayload('/pi.jsonl', PI_HEADER)) == 'pi'
    assert detect_source(SourcePayload('/codex.jsonl', CODEX_HEADER)) == 'codex'
    assert detect_source(SourcePayload('/claude.jsonl', CLAUDE_RECORD)) == 'claude'
    assert detect_source(SourcePayload('/chatSessions/s.json', VSCODE_V3)) == 'copilot-vscode'
    assert detect_source(SourcePayload('/chatSessions/s.jsonl', VSCODE_V3_LOG)) == 'copilot-vscode'
    assert detect_source(SourcePayload('/session-state/s/events.jsonl', CLI_START)) == 'copilot-cli'
    assert detect_source(SourcePayload('/notes.jsonl', b'{"event":"other"}\n')) is None


def test_vscode_empty_initial_operation_log_is_structurally_detected_before_participant_filtering():
    empty = b'{"kind":0,"v":{"version":3,"sessionId":"vs-session","requests":[]}}\n'
    assert detect_source(SourcePayload('/workspaceStorage/key/chatSessions/s.jsonl', empty)) == 'copilot-vscode'


@pytest.mark.parametrize('version', (3.0, True, '3'))
def test_vscode_discovery_requires_exact_integer_schema_version(version):
    value = json.loads(VSCODE_V3)
    value['version'] = version
    assert detect_source(SourcePayload('/chatSessions/s.json', json.dumps(value).encode())) is None


def test_existing_pi_and_codex_direct_imports_do_not_depend_on_locator_suffix():
    assert detect_source(SourcePayload('/pi', PI_HEADER)) == 'pi'
    assert detect_source(SourcePayload('/codex', CODEX_HEADER + b'{"type":"event_msg"}\n')) == 'codex'


def test_vscode_structure_reaches_reader_while_cli_still_requires_exact_producer():
    wrong_vscode = VSCODE_V3.replace(b'GitHub.copilot-chat', b'other.extension')
    wrong_cli = CLI_START.replace(b'copilot-cli', b'other-writer')
    assert detect_source(SourcePayload('/chatSessions/s.json', wrong_vscode)) == 'copilot-vscode'
    assert detect_source(SourcePayload('/session-state/s/events.jsonl', wrong_cli)) is None


def test_cli_durable_usage_events_are_structurally_detected():
    locator = '/session-state/s/events.jsonl'
    assert detect_source(SourcePayload(locator, CLI_CHECKPOINT)) == 'copilot-cli'
    assert detect_source(SourcePayload(locator, CLI_SHUTDOWN)) == 'copilot-cli'


def test_cli_usage_detection_rejects_wrong_boundaries_envelopes_and_ephemeral_events():
    locator = '/session-state/s/events.jsonl'
    bad_checkpoint = CLI_CHECKPOINT.replace(b'"totalNanoAiu":1000000000', b'"totalNanoAiu":"1000000000"')
    bad_shutdown_value = json.loads(CLI_SHUTDOWN)
    bad_shutdown_value['data']['modelMetrics'] = []
    bad_shutdown = json.dumps(bad_shutdown_value).encode()
    bad_envelope = CLI_CHECKPOINT.replace(b'"id":"checkpoint"', b'"id":1')
    bad_start_version = CLI_START.replace(b'"version":1', b'"version":"1"')
    assert detect_source(SourcePayload('/events.jsonl', CLI_CHECKPOINT)) is None
    assert detect_source(SourcePayload(locator, bad_checkpoint)) is None
    assert detect_source(SourcePayload(locator, bad_shutdown)) is None
    assert detect_source(SourcePayload(locator, bad_envelope)) is None
    assert detect_source(SourcePayload(locator, bad_start_version)) is None
    assert detect_source(SourcePayload(locator, CLI_EPHEMERAL_USAGE)) is None


@pytest.mark.parametrize('source', (
    b'{"id":"checkpoint","timestamp":"2026-09-16T01:00:00Z","type":"session.usage_checkpoint","data":{"totalNanoAiu":1}}\n',
    b'{"id":"checkpoint","parentId":"event","timestamp":"2026-09-16T01:00:00Z","type":"session.usage_checkpoint","data":{"totalNanoAiu":1},"ephemeral":true}\n',
    b'{"id":"checkpoint","parentId":"event","timestamp":"2026-09-16T01:00:00Z","type":"session.usage_checkpoint","data":{"totalNanoAiu":1},"agentId":"subagent"}\n',
    b'{"id":"","parentId":"event","timestamp":"2026-09-16T01:00:00Z","type":"session.usage_checkpoint","data":{"totalNanoAiu":1}}\n',
    b'{"id":"checkpoint","parentId":"event","timestamp":"not-a-time","type":"session.usage_checkpoint","data":{"totalNanoAiu":1}}\n',
), ids=('missing-parent', 'ephemeral', 'agent-control', 'empty-id', 'invalid-time'))
def test_cli_durable_controls_require_complete_top_level_persisted_envelopes(source):
    assert detect_source(SourcePayload('/session-state/s/events.jsonl', source)) is None


def test_sidecar_bytes_are_ordered_and_fingerprinted_with_the_primary():
    first = SourcePayload('/chatSessions/s.json', VSCODE_V3, (('workspace.json', b'{"folder":"one"}'),))
    same = SourcePayload('/chatSessions/s.json', VSCODE_V3, tuple(reversed(first.context)))
    changed = SourcePayload('/chatSessions/s.json', VSCODE_V3, (('workspace.json', b'{"folder":"two"}'),))
    assert first.fingerprint() == same.fingerprint()
    assert first.fingerprint() != changed.fingerprint()


def test_storage_uses_exact_new_source_dispatch_and_sidecar_fingerprints(tmp_path):
    store = Storage(tmp_path / 'ledger.duckdb')
    first_vscode = SourcePayload('/chatSessions/s.json', VSCODE_V3, (('workspace.json', b'{"folder":"one"}'),))
    sources = (
        SourcePayload('/claude.jsonl', CLAUDE_RECORD),
        first_vscode,
        SourcePayload('/session-state/s/events.jsonl', CLI_START),
    )
    store.import_sources(sources)
    with store.connect() as db:
        assert dict(db.execute('SELECT locator,profile FROM source_generation')) == {
            '/claude.jsonl': 'claude-code/2.1.273-shape-5',
            '/chatSessions/s.json': 'vscode-chat-v3/copilot-shape-2',
            '/session-state/s/events.jsonl': 'copilot-cli-events/e60d903-shape-1',
        }
        assert {row[0] for row in db.execute('SELECT code FROM diagnostic')} == {
            'conflicting_session_identity', 'missing_copilot_vscode_scope'}
    changed_vscode = SourcePayload('/chatSessions/s.json', VSCODE_V3, (('workspace.json', b'{"folder":"two"}'),))
    store.import_sources((changed_vscode,))
    with store.connect() as db:
        assert db.execute("SELECT count(*) FROM source_generation WHERE locator='/chatSessions/s.json'").one()[0] == 2


def test_storage_keeps_context_free_tuple_import_compatibility(tmp_path):
    store = Storage(tmp_path / 'ledger.duckdb')
    store.import_sources((('/pi.jsonl', PI_HEADER),))
    assert store.snapshot().sessions[0].id == 'pi:pi-session'
