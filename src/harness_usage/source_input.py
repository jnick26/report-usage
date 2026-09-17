"""Immutable local source bytes and bounded structural classification."""
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
from typing import Literal, cast


SourceKind = Literal['pi', 'codex', 'claude', 'copilot-vscode', 'copilot-cli']
MAX_PROBE_BYTES = 1024 * 1024


@dataclass(frozen=True, slots=True)
class SourcePayload:
    locator: str
    data: bytes
    context: tuple[tuple[str, bytes], ...] = ()

    def fingerprint(self) -> str:
        parts = [self.data]
        for name, value in sorted(self.context):
            parts.extend((b'\0', name.encode(), b'\0', value))
        return sha256(b''.join(parts)).hexdigest()


def _first_object(data: bytes, *, jsonl: bool) -> dict[str, object] | None:
    if jsonl:
        data = data.split(b'\n', 1)[0]
    if not data or len(data) > MAX_PROBE_BYTES:
        return None
    try:
        value = json.loads(data)
    except (ValueError, UnicodeError, RecursionError):
        return None
    return cast(dict[str, object], value) if isinstance(value, dict) else None


def _copilot_request(value: object) -> bool:
    if not isinstance(value, list):
        return False
    for request in value:
        if not isinstance(request, dict):
            continue
        agent = request.get('agent')
        extension = agent.get('extensionId') if isinstance(agent, dict) else None
        identity = extension.get('value') if isinstance(extension, dict) else None
        if isinstance(identity, str) and identity.casefold() == 'github.copilot-chat':
            return True
    return False


def _number(value: object) -> bool:
    return type(value) in (int, float)


def _valid_timestamp(value: object) -> bool:
    try:
        return isinstance(value, str) and datetime.fromisoformat(value).tzinfo is not None
    except (ValueError, OverflowError):
        return False


def _claude_record(value: dict[str, object]) -> bool:
    """Recognize Claude metadata headers without routing arbitrary JSONL."""
    if not isinstance(value.get('sessionId'), str) or not value['sessionId']:
        return False
    kind = value.get('type')
    if kind in ('assistant', 'user', 'system', 'summary', 'progress'):
        return True
    if kind == 'queue-operation':
        return (isinstance(value.get('operation'), str) and bool(value['operation'])
                and _valid_timestamp(value.get('timestamp'))
                and ('content' not in value or isinstance(value['content'], str)))
    if kind == 'last-prompt':
        return isinstance(value.get('leafUuid'), str) and bool(value['leafUuid'])
    if kind == 'ai-title':
        return isinstance(value.get('aiTitle'), str) and bool(value['aiTitle'])
    return False


def _cli_envelope(value: dict[str, object], *, durable: bool) -> dict[str, object] | None:
    data = value.get('data')
    identity, parent = value.get('id'), value.get('parentId')
    if (not isinstance(identity, str) or not identity
            or durable and (not isinstance(parent, str) or not parent)
            or not durable and parent is not None and not isinstance(parent, str)
            or not _valid_timestamp(value.get('timestamp'))
            or not isinstance(data, dict)
            or durable and 'agentId' in value
            or not durable and 'agentId' in value and not isinstance(value['agentId'], str)
            or durable and value.get('ephemeral', False) is not False
            or not durable and 'ephemeral' in value and type(value['ephemeral']) is not bool):
        return None
    return cast(dict[str, object], data)


def _cli_checkpoint(data: dict[str, object]) -> bool:
    return (_number(data.get('totalNanoAiu'))
            and ('totalPremiumRequests' not in data or _number(data['totalPremiumRequests'])))


def _cli_model_metric(value: object) -> bool:
    if not isinstance(value, dict) or not isinstance(value.get('requests'), dict) or not isinstance(value.get('usage'), dict):
        return False
    requests, usage = value['requests'], value['usage']
    return (all(field not in requests or _number(requests[field]) for field in ('count', 'cost'))
            and all(_number(usage.get(field)) for field in ('inputTokens', 'outputTokens', 'cacheReadTokens', 'cacheWriteTokens'))
            and ('reasoningTokens' not in usage or _number(usage['reasoningTokens'])))


def _cli_shutdown(data: dict[str, object]) -> bool:
    metrics, changes = data.get('modelMetrics'), data.get('codeChanges')
    return (_number(data.get('sessionStartTime'))
            and data.get('shutdownType') in ('routine', 'error')
            and _number(data.get('totalApiDurationMs'))
            and isinstance(metrics, dict) and all(_cli_model_metric(metric) for metric in metrics.values())
            and isinstance(changes, dict)
            and isinstance(changes.get('filesModified'), list)
            and all(isinstance(path, str) for path in changes['filesModified'])
            and _number(changes.get('linesAdded')) and _number(changes.get('linesRemoved')))


def detect_source(payload: SourcePayload) -> SourceKind | None:
    path = Path(payload.locator)
    if path.name == 'workspace.yaml' and path.parent.parent.name == 'session-state':
        from .copilot_cli_reader import _locator_session
        return 'copilot-cli' if _locator_session(payload.locator) is not None else None
    first = _first_object(payload.data, jsonl=True)
    if first is not None and first.get('type') == 'session' and first.get('version') == 3 and isinstance(first.get('id'), str):
        return 'pi'
    codex = first.get('payload') if first is not None else None
    if first is not None and first.get('type') == 'session_meta' and isinstance(codex, dict) and isinstance(codex.get('id'), str):
        return 'codex'
    if first is not None and _claude_record(first):
        return 'claude'
    jsonl = path.suffix == '.jsonl'
    value = first if jsonl else _first_object(payload.data, jsonl=False)
    if value is None:
        probe_size = len(payload.data.split(b'\n', 1)[0]) if jsonl else len(payload.data)
        if probe_size > MAX_PROBE_BYTES:
            from .copilot_vscode_reader import copilot_vscode_scope
            # Route oversized candidates only; the full reader qualifies schema and participants.
            if copilot_vscode_scope(payload.locator) is not None:
                return 'copilot-vscode'
        return None
    if path.parent.name == 'chatSessions' and path.suffix in ('.json', '.jsonl'):
        state = value.get('v') if value.get('kind') == 0 else value
        if (isinstance(state, dict) and type(state.get('version')) is int and state.get('version') == 3
                and isinstance(state.get('sessionId'), str) and bool(state['sessionId'])):
            return 'copilot-vscode'
    if (path.name == 'events.jsonl' and path.parent.parent.name == 'session-state'
            and value.get('type') in ('session.start', 'session.usage_checkpoint', 'session.shutdown')):
        data = _cli_envelope(value, durable=value.get('type') != 'session.start')
        if data is None:
            return None
        if (value.get('type') == 'session.start' and isinstance(data.get('sessionId'), str)
                and _number(data.get('version'))
                and data.get('producer') in ('copilot-cli', 'github-copilot-cli', 'copilot-agent')
                and isinstance(data.get('copilotVersion'), str)
                and isinstance(data.get('startTime'), str)):
            return 'copilot-cli'
        if value.get('type') == 'session.usage_checkpoint' and _cli_checkpoint(data):
            return 'copilot-cli'
        if value.get('type') == 'session.shutdown' and _cli_shutdown(data):
            return 'copilot-cli'
    return None
