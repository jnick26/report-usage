"""Privacy-bounded Claude Code JSONL accounting reader."""
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Literal, cast
from uuid import UUID

from .domain import Known, MissingEstimate, ModelIdentity, Point, SessionId, TokenBreakdown, TokenEvidence, Undated, Unknown
from .pi_reader import (Diagnostic, EntryIdentity, ReadBatch, RejectedSource, SafeFact,
                        SessionMetadata, UsageRecord, _pending, _unique_object)

CLAUDE_PROFILE = 'claude-code/2.1.273-shape-5'
PROFILE = CLAUDE_PROFILE

_SEMVER = re.compile(
    r'(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)'
    r'(?:-((?:0|[1-9][0-9]*|[0-9A-Za-z-]*[A-Za-z-][0-9A-Za-z-]*)'
    r'(?:\.(?:0|[1-9][0-9]*|[0-9A-Za-z-]*[A-Za-z-][0-9A-Za-z-]*))*))?'
    r'(?:\+([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?', re.ASCII)


@dataclass(frozen=True, slots=True)
class ClaudeEvidence:
    message_id: str | None
    request_id: str | None
    entry_uuid: str
    agent_id: str | None
    state: Literal['usable', 'unresolved']
    reason: str | None


@dataclass(frozen=True, slots=True)
class ClaudeReadBatch(ReadBatch):
    evidence: tuple[ClaudeEvidence, ...] = ()
    parent_session_id: str | None = None
    cwd_state: Literal['absent', 'valid', 'invalid'] = 'absent'
    cwd_line: int | None = None


def _uuid(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = UUID(value)
    except (ValueError, AttributeError):
        return None
    return str(parsed) if str(parsed) == value.casefold() else None


def claude_locator_identity(locator: str) -> tuple[str, str | None, str | None] | None:
    """Return the verified parent UUID, agent ID, and canonical main locator."""
    path = Path(locator)
    parts = path.parts
    if 'subagents' in parts:
        index = len(parts) - 1 - tuple(reversed(parts)).index('subagents')
        if index == 0 or index == len(parts) - 1:
            return None
        parent = _uuid(parts[index - 1])
        name = path.name
        agent = name[6:-6] if name.startswith('agent-') and name.endswith('.jsonl') else ''
        if parent is None or not agent or any(char in agent for char in '/\\'):
            return None
        parent_dir = Path(*parts[:index - 1])
        main = parent_dir / f'{parent}.jsonl'
        return parent, agent, str(main)
    parent = _uuid(path.stem) if path.suffix == '.jsonl' else None
    return (parent, None, None) if parent is not None else None


def _text(value: object) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _instant(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
        return parsed if parsed.tzinfo is not None and parsed.utcoffset() is not None else None
    except (ValueError, OverflowError):
        return None


def _count(usage: dict[str, object], name: str) -> Known | Unknown | None:
    if name not in usage:
        return Unknown('not_reported')
    value = usage[name]
    if type(value) is not int or not 0 <= value < 2**63:
        return None
    return Known(value)


def _writer_output_kind(
        value: object) -> tuple[Literal['final', 'placeholder', 'unqualified'], str | None]:
    if not isinstance(value, str) or len(value) > 128:
        return 'unqualified', None
    match = _SEMVER.fullmatch(value)
    if match is None:
        return 'unqualified', None
    release = tuple(int(match.group(index)) for index in range(1, 4))
    token = sha256(b'claude-writer-version-v1\0' + value.encode()).hexdigest()
    if release > (2, 1, 97) or release == (2, 1, 97) and match.group(4) is None:
        return 'final', token
    return 'placeholder', token


def read_claude(data: bytes, *, locator: str, profile: str = PROFILE) -> ClaudeReadBatch | RejectedSource:
    identity = claude_locator_identity(locator)
    if profile != PROFILE or identity is None:
        return RejectedSource((Diagnostic('conflicting_session_identity', None, None, None),))
    expected, agent_id, parent_locator = identity
    records: list[tuple[int, dict[str, object]]] = []
    diagnostics: list[Diagnostic] = []
    complete_bytes = 0
    offset = 0
    lines = data.splitlines(keepends=True)
    pending = False
    for line_number, raw_line in enumerate(lines, 1):
        complete = raw_line.endswith((b'\n', b'\r'))
        body = raw_line.rstrip(b'\r\n')
        if not body:
            offset += len(raw_line)
            complete_bytes = offset
            continue
        try:
            value = json.loads(body, object_pairs_hook=_unique_object)
        except json.JSONDecodeError as error:
            if not complete and line_number == len(lines) and _pending(error, body.decode(errors='replace')):
                pending = True
                break
            offset += len(raw_line)
            complete_bytes = offset
            diagnostics.append(Diagnostic('malformed_json', line_number, None, None))
            continue
        except (ValueError, UnicodeError, RecursionError):
            offset += len(raw_line)
            complete_bytes = offset
            diagnostics.append(Diagnostic('malformed_json', line_number, None, None))
            continue
        offset += len(raw_line)
        complete_bytes = offset
        if isinstance(value, dict):
            records.append((line_number, cast(dict[str, object], value)))
        else:
            diagnostics.append(Diagnostic('invalid_record', line_number, None, None))
    identity_records = [record for _, record in records
                        if _text(record.get('uuid')) is not None or _text(record.get('requestId')) is not None
                        or isinstance(record.get('message'), dict)]
    session_ids = {_uuid(record.get('sessionId')) for record in identity_records}
    if not identity_records or None in session_ids or session_ids != {expected}:
        return RejectedSource((Diagnostic('conflicting_session_identity', None, None, None),))

    entries: list[EntryIdentity] = []
    usage_records: list[UsageRecord] = []
    evidence: list[ClaudeEvidence] = []
    times: list[datetime] = []
    cwd_fields = [(line, record['cwd']) for line, record in records if 'cwd' in record]
    valid_cwds = [(line, value) for line, value in cwd_fields
                  if isinstance(value, str) and value.strip() and Path(value).is_absolute()]
    invalid_cwds = [line for line, value in cwd_fields
                    if not (isinstance(value, str) and value.strip() and Path(value).is_absolute())]
    distinct_cwds = {value for _, value in valid_cwds}
    if invalid_cwds or len(distinct_cwds) > 1:
        cwd_state: Literal['absent', 'valid', 'invalid'] = 'invalid'
        cwd = None
        cwd_line = invalid_cwds[0] if invalid_cwds else next(
            line for line, value in valid_cwds if value != valid_cwds[0][1])
        diagnostics.append(Diagnostic('cwd_attribution_unavailable', cwd_line, None, None))
    elif valid_cwds:
        cwd_state = 'valid'
        cwd, cwd_line = valid_cwds[0][1], valid_cwds[0][0]
    else:
        cwd_state, cwd, cwd_line = 'absent', None, None
        diagnostics.append(Diagnostic('cwd_attribution_unavailable', None, None, None))
    for line_number, record in records:
        entry_uuid = _text(record.get('uuid'))
        parent_uuid = _text(record.get('parentUuid'))
        if entry_uuid is not None:
            entries.append(EntryIdentity(entry_uuid, parent_uuid, line_number))
        timestamp = _instant(record.get('timestamp'))
        if timestamp is not None:
            times.append(timestamp)
        elif 'timestamp' in record:
            diagnostics.append(Diagnostic('invalid_timestamp', line_number, entry_uuid, None))
        message = record.get('message')
        if record.get('type') != 'assistant' or not isinstance(message, dict) or not isinstance(message.get('usage'), dict):
            continue
        if entry_uuid is None:
            diagnostics.append(Diagnostic('invalid_entry_identity', line_number, None, None))
            continue
        raw_usage = cast(dict[str, object], message['usage'])
        counts = tuple(_count(raw_usage, name) for name in
                       ('input_tokens', 'cache_read_input_tokens', 'cache_creation_input_tokens'))
        if any(value is None for value in counts):
            diagnostics.append(Diagnostic('invalid_count', line_number, entry_uuid, None))
            continue
        model = _text(message.get('model'))
        if model is None:
            diagnostics.append(Diagnostic('invalid_model', line_number, entry_uuid, None))
            continue
        message_id, request_id = _text(message.get('id')), _text(record.get('requestId'))
        state: Literal['usable', 'unresolved'] = 'usable' if message_id is not None or request_id is not None else 'unresolved'
        reason = None if state == 'usable' else 'missing_claude_identity'
        writer_kind, writer_token = _writer_output_kind(record.get('version'))
        if writer_kind == 'unqualified':
            diagnostics.append(Diagnostic('writer_version_unavailable', line_number, entry_uuid, None))
        raw_output = _count(raw_usage, 'output_tokens')
        output: Known | Unknown
        safe_facts: tuple[SafeFact, ...]
        writer_fact: tuple[SafeFact, ...] = (
            (('usage.writerVersionToken', writer_token),) if writer_token is not None else ())
        if 'output_tokens' not in raw_usage:
            output = Unknown('not_reported')
            finality = 'final_missing' if writer_kind == 'final' else 'not_reported'
            safe_facts = (('usage.outputFinality', finality),) + writer_fact
        elif raw_output is None:
            diagnostics.append(Diagnostic('invalid_count', line_number, entry_uuid, None))
            output = Unknown('invalid_count')
            finality = 'invalid'
            safe_facts = (('usage.outputFinality', finality),) + writer_fact
            state, reason = 'unresolved', 'invalid_count'
        else:
            assert isinstance(raw_output, Known)
            output = (raw_output if writer_kind == 'final'
                      else Unknown('stream_start_placeholder') if writer_kind == 'placeholder'
                      else Unknown('writer_version_unavailable'))
            finality = writer_kind
            safe_facts = (('usage.output', raw_output.value),
                          ('usage.outputFinality', finality)) + writer_fact
        usage_record = UsageRecord(
            EntryIdentity(entry_uuid, parent_uuid, line_number), 'assistant',
            Point(timestamp) if timestamp is not None else Undated('timestamp_unavailable'),
            ModelIdentity(None, model),
            TokenEvidence(TokenBreakdown(cast(Known | Unknown, counts[0]), output,
                                         cast(Known | Unknown, counts[1]), cast(Known | Unknown, counts[2])),
                          Unknown('partial_total'), Unknown('not_reported'), Unknown('not_reported')),
            MissingEstimate('not_recorded'), None, None, safe_facts, ())
        usage_records.append(usage_record)
        evidence.append(ClaudeEvidence(message_id, request_id, entry_uuid, agent_id, state, reason))
    native_id = expected if agent_id is None else f'{expected}:agent:{agent_id}'
    session_id = SessionId('claude:' + native_id)
    session = SessionMetadata(session_id, native_id, min(times) if times else None,
                              cwd, parent_locator, None,
                              last_observed=max(times) if times else None)
    return ClaudeReadBatch(session, tuple(entries), tuple(usage_records), tuple(diagnostics), complete_bytes,
                           pending, (), tuple(evidence), 'claude:' + expected if agent_id is not None else None,
                           cwd_state, cwd_line)
