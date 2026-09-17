"""Bounded reader for the pinned Copilot CLI durable event shape."""
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime
from decimal import Decimal
from hashlib import sha256
import json
from pathlib import Path
from typing import Literal, cast
from uuid import UUID

from .domain import (
    Known, MeasuredQuantity, MissingEstimate, ModelIdentity, NotApplicable,
    Point, SessionId, TokenBreakdown, TokenEvidence, Undated, Unknown,
)
from .pi_reader import (
    Diagnostic, EntryIdentity, ReadBatch, RejectedSource, SessionMetadata,
    UsageRecord, _pending, _unique_object,
)

CLI_PROFILE = 'copilot-cli-events/e60d903-shape-1'
WORKSPACE_PROFILE = 'copilot-cli-workspace/unversioned'
_LIMIT = 2**63
_MAX_MAP = 1024
_MAX_DECIMAL_DIGITS = 128
_MAX_DECIMAL_EXPONENT = 128
_MAX_DECIMAL_TEXT = 256
_SESSION_PARTS = ('session-state',)


@dataclass(frozen=True, slots=True)
class WorkspaceMetadata:
    session_id: str | None
    cwd: str | None
    diagnostics: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CopilotCLIEvidence:
    profile: str
    source_kind: Literal['shutdown', 'usage_checkpoint', 'metadata']
    event_id: str | None
    parent_event_id: str | None
    event_schema_version: str | None
    writer_version: str | None
    session_start_us: int | None
    current_model: str | None
    agent_id: str | None
    counter_epoch: str | None
    counters_json: str
    state: Literal['usable', 'unresolved']
    reason: str | None
    line: int
    scope: Literal['session', 'model', 'agent', 'metadata']


@dataclass(frozen=True, slots=True)
class CopilotCLIReadBatch(ReadBatch):
    evidence: tuple[CopilotCLIEvidence, ...] = ()
    raw_session_id: str = ''
    agent_ids: tuple[str, ...] = ()


def _uuid4(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = UUID(value)
    except (ValueError, AttributeError):
        return None
    return value if parsed.version == 4 and str(parsed) == value.casefold() else None


def _locator_session(locator: str) -> str | None:
    path = Path(locator)
    if path.name not in ('events.jsonl', 'workspace.yaml') or path.parent.parent.name not in _SESSION_PARTS:
        return None
    return _uuid4(path.parent.name)


def _time(value: object, *, millisecond: bool = False) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    except (ValueError, OverflowError):
        return None
    try:
        if parsed.tzinfo is None or parsed.utcoffset() is None or (millisecond and parsed.microsecond % 1000):
            return None
        return parsed.astimezone(UTC)
    except (ValueError, OverflowError, OSError):
        return None


def _text(value: object) -> str | None:
    return value if isinstance(value, str) and value.strip() and len(value) <= 256 else None


def _count(value: object) -> int | None:
    return value if type(value) is int and 0 <= value < _LIMIT else None


def _decimal(value: object) -> Decimal | None:
    if type(value) is int:
        result = Decimal(value)
    elif isinstance(value, Decimal):
        result = value
    else:
        return None
    if not result.is_finite() or not 0 <= result < _LIMIT:
        return None
    _, digits, exponent = result.as_tuple()
    if len(digits) > _MAX_DECIMAL_DIGITS or abs(int(exponent)) > _MAX_DECIMAL_EXPONENT:
        return None
    try:
        if len(format(result, 'f')) > _MAX_DECIMAL_TEXT:
            return None
    except (ValueError, OverflowError):
        return None
    return result


def _object(value: object) -> dict[str, object] | None:
    return cast(dict[str, object], value) if isinstance(value, dict) and len(value) <= _MAX_MAP else None


def _json(value: object) -> str:
    def default(item: object) -> str:
        if isinstance(item, Decimal):
            return str(item)
        raise TypeError
    return json.dumps(value, sort_keys=True, separators=(',', ':'), default=default)


def _epoch_millis(value: datetime) -> int | None:
    """Convert an aware instant without float timestamp overflow/rounding."""
    try:
        utc = value.astimezone(UTC)
        days = utc.date().toordinal() - date(1970, 1, 1).toordinal()
        seconds = days * 86_400 + utc.hour * 3_600 + utc.minute * 60 + utc.second
        result = seconds * 1_000 + utc.microsecond // 1_000
    except (ValueError, OverflowError, OSError):
        return None
    return result if abs(result) <= 2**63 - 1 else None


def _native(value: object) -> int | str | None:
    """A native scalar never admits source-defined container keys."""
    if type(value) is int:
        return value if 0 <= value < _LIMIT else None
    if isinstance(value, Decimal):
        bounded = _decimal(value)
        return str(bounded) if bounded is not None else None
    return None


def _token_details(value: object) -> dict[str, object] | None:
    """Retain only the pinned token-type -> tokenCount breakdown shape."""
    details = _object(value)
    if details is None:
        return None
    output: dict[str, object] = {}
    for token_type, raw_detail in details.items():
        name = _text(token_type)
        detail = _object(raw_detail)
        if name is None or detail is None:
            continue
        count = _count(detail.get('tokenCount'))
        if count is not None:
            output[name] = {'tokenCount': count}
    return output


def _metric_details(value: object) -> dict[str, object] | None:
    """Project only numeric fields from a native model-metric breakdown."""
    metric = _object(value)
    if metric is None:
        return None
    output: dict[str, object] = {}
    invalid: list[str] = []
    requests = _object(metric.get('requests')) if 'requests' in metric else None
    if requests is not None:
        request_values: dict[str, object] = {}
        if 'count' in requests:
            count = _count(requests['count'])
            if count is not None:
                request_values['count'] = count
            else:
                invalid.append('request_count')
        if 'cost' in requests:
            cost = _decimal(requests['cost'])
            if cost is not None:
                request_values['cost'] = str(cost)
            else:
                invalid.append('request_cost')
        output['requests'] = request_values
        if invalid:
            output['invalid'] = invalid
    usage = _object(metric.get('usage')) if 'usage' in metric else None
    if requests is None or usage is None or any(
            _count(usage.get(name)) is None for name in
            ('inputTokens', 'outputTokens', 'cacheReadTokens', 'cacheWriteTokens')):
        return None
    if usage is not None:
        usage_values: dict[str, object] = {}
        for name in ('inputTokens', 'outputTokens', 'cacheReadTokens', 'cacheWriteTokens', 'reasoningTokens'):
            if name in usage:
                count = _count(usage[name])
                if count is not None:
                    usage_values[name] = count
        output['usage'] = usage_values
    if 'totalNanoAiu' in metric:
        total = _native(metric['totalNanoAiu'])
        if total is not None:
            output['total_nano_aiu'] = total
    if 'tokenDetails' in metric:
        details = _token_details(metric['tokenDetails'])
        if details is not None:
            output['token_details'] = details
    return output


def _metric_map(value: object) -> dict[str, object] | None:
    metrics = _object(value)
    if metrics is None:
        return None
    output: dict[str, object] = {}
    for model, raw_metric in metrics.items():
        model_name = _text(model)
        details = _metric_details(raw_metric)
        if model_name is not None and details is not None:
            output[model_name] = details
        else:
            return None
    return output


def read_workspace_metadata(data: bytes) -> WorkspaceMetadata:
    """Read only the two pinned scalar workspace fields; YAML features are unsupported."""
    try:
        text = data.decode('utf-8')
    except UnicodeError:
        return WorkspaceMetadata(None, None, ('invalid_workspace_metadata',))
    values: dict[str, str] = {}
    diagnostics: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith('#'):
            continue
        if raw_line[:1].isspace():
            diagnostics.append('invalid_workspace_metadata')
            continue
        if ':' not in line or line.startswith(('-', '{', '[', '&', '*', '!', '|', '>')):
            diagnostics.append('invalid_workspace_metadata')
            continue
        key, raw = (part.strip() for part in line.split(':', 1))
        if key not in ('id', 'cwd'):
            continue
        if key in values:
            diagnostics.append('invalid_workspace_metadata')
            continue
        try:
            value = json.loads(raw) if raw.startswith(('"', "'")) else raw
        except (ValueError, TypeError):
            diagnostics.append('invalid_workspace_metadata')
            continue
        if not isinstance(value, str) or not value.strip() or len(value) > 4096:
            diagnostics.append('invalid_workspace_metadata')
            continue
        values[key] = value
    session_id = _uuid4(values.get('id'))
    cwd = values.get('cwd')
    if session_id is None:
        diagnostics.append('invalid_workspace_metadata')
    if cwd is not None and not Path(cwd).is_absolute():
        diagnostics.append('invalid_workspace_metadata')
        cwd = None
    return WorkspaceMetadata(session_id, cwd, tuple(dict.fromkeys(diagnostics)))


def _unknown_tokens(reason: str) -> TokenEvidence:
    value = Unknown(reason)
    return TokenEvidence(TokenBreakdown(value, value, value, value), value, value, value)


def _na_tokens() -> TokenEvidence:
    value = NotApplicable('not_applicable')
    return TokenEvidence(TokenBreakdown(value, value, value, value), value, value, value)


def _quantity(measure: Literal['nano_aiu', 'premium_requests', 'request_count'], value: Decimal,
              source: str) -> MeasuredQuantity:
    return MeasuredQuantity(measure, 'known', value, None, True, source)


def _unknown_quantities(reason: str) -> tuple[MeasuredQuantity, ...]:
    measures: tuple[Literal['nano_aiu', 'premium_requests', 'request_count'], ...] = (
        'nano_aiu', 'premium_requests', 'request_count')
    return tuple(MeasuredQuantity(measure, 'unknown', None, reason, False, None)
                 for measure in measures)


def _entry_id(event_id: str, scope: str, discriminator: str | None = None) -> str:
    identity = [event_id, scope, discriminator]
    return 'copilot-cli:' + sha256(_json(identity).encode()).hexdigest()


def _workspace_batch(data: bytes, locator: str, expected: str) -> CopilotCLIReadBatch | RejectedSource:
    metadata = read_workspace_metadata(data)
    if metadata.session_id is not None and metadata.session_id != expected:
        return RejectedSource((Diagnostic('conflicting_session_identity', None, None, None),))
    entry = EntryIdentity(_entry_id(expected, 'metadata'), None, 1)
    record = UsageRecord(entry, 'usage_checkpoint', Undated('events_unavailable'), ModelIdentity(None, None),
                         _unknown_tokens('events_unavailable'), MissingEstimate('not_recorded'),
                         None, None, (), _unknown_quantities('events_unavailable'))
    evidence = CopilotCLIEvidence(WORKSPACE_PROFILE, 'metadata', None, None, None, None, None, None, None,
                                  None, _json({'scope': 'metadata'}), 'unresolved', 'events_unavailable', 1,
                                  'metadata')
    diagnostics = (Diagnostic('events_unavailable', None, None, None), *(
        Diagnostic(code, None, None, None) for code in metadata.diagnostics))
    session = SessionMetadata(SessionId('copilot-cli:' + expected), expected, None, metadata.cwd, None, None)
    return CopilotCLIReadBatch(session, (entry,), (record,), tuple(diagnostics), len(data), False, (),
                               (evidence,), expected, ())


def _parse_lines(data: bytes) -> tuple[list[tuple[int, dict[str, object]]], list[Diagnostic], int, bool]:
    rows: list[tuple[int, dict[str, object]]] = []
    diagnostics: list[Diagnostic] = []
    complete_bytes = 0
    pending = False
    offset = 0
    lines = data.splitlines(keepends=True)
    for line_number, raw_line in enumerate(lines, 1):
        complete = raw_line.endswith((b'\n', b'\r'))
        body = raw_line.rstrip(b'\r\n')
        try:
            text = body.decode('utf-8')
            value = json.loads(text, parse_int=int, parse_float=Decimal,
                               parse_constant=lambda _value: (_ for _ in ()).throw(ValueError('constant')),
                               object_pairs_hook=_unique_object)
        except (ValueError, UnicodeError, RecursionError) as error:
            if (not complete and line_number == len(lines) and isinstance(error, json.JSONDecodeError)
                    and _pending(error, body.decode(errors='replace'))):
                pending = True
                break
            offset += len(raw_line)
            complete_bytes = offset
            diagnostics.append(Diagnostic('malformed_json', line_number, None, None))
            continue
        offset += len(raw_line)
        complete_bytes = offset
        if not isinstance(value, dict) or len(value) > _MAX_MAP:
            diagnostics.append(Diagnostic('invalid_cli_event', line_number, None, None))
        else:
            rows.append((line_number, cast(dict[str, object], value)))
    return rows, diagnostics, complete_bytes, pending


def _envelope(row: dict[str, object], *, start: bool = False) -> tuple[str, str | None, datetime, dict[str, object]] | None:
    event_id = _uuid4(row.get('id'))
    if 'parentId' not in row:
        return None
    parent = row['parentId']
    parent_id = None if parent is None else _uuid4(parent)
    timestamp = _time(row.get('timestamp'), millisecond=True)
    data = _object(row.get('data'))
    if (event_id is None or timestamp is None or data is None or (start and parent is not None)
            or (start and 'agentId' in row) or (not start and parent_id is None)):
        return None
    if 'ephemeral' in row and row['ephemeral'] is not False:
        return None
    return event_id, parent_id, timestamp, data


def _conversation_agent(row: dict[str, object]) -> tuple[bool, str | None]:
    if 'agentId' not in row:
        return True, None
    value = _text(row['agentId'])
    return value is not None, value


def _unknown_quantity(measure: Literal['nano_aiu', 'premium_requests', 'request_count'], reason: str) -> MeasuredQuantity:
    return MeasuredQuantity(measure, 'unknown', None, reason, False, None)


def _model_record(event_id: str, parent_id: str, line: int, at: datetime, epoch: str,
                  model: str, raw: dict[str, object] | None,
                  current_model: str | None) -> tuple[UsageRecord, CopilotCLIEvidence] | None:
    if raw is None:
        return None
    requests = _object(raw.get('requests'))
    usage = _object(raw.get('usage'))
    required_names = ('inputTokens', 'outputTokens', 'cacheReadTokens', 'cacheWriteTokens')
    required = {name: _count(usage.get(name)) if usage is not None and name in usage else None
                for name in required_names}
    invalid: list[str] = []
    invalid_shape = requests is None or usage is None
    if invalid_shape:
        invalid.append('model_shape')
    presence = {name.removesuffix('Tokens'): int(usage is not None and name in usage)
                for name in required_names}
    if usage is None:
        invalid.extend(name.removesuffix('Tokens') for name in required_names)
    else:
        invalid.extend(name.removesuffix('Tokens') for name, value in required.items()
                       if value is None)
    reasoning_present = usage is not None and 'reasoningTokens' in usage
    reasoning = _count(usage.get('reasoningTokens')) if reasoning_present and usage is not None else None
    if reasoning_present and reasoning is None:
        invalid.append('reasoning')
    if reasoning is not None and required['outputTokens'] is not None and reasoning > required['outputTokens']:
        invalid.append('reasoning')

    def token(name: str) -> Known | Unknown:
        value = required[name + 'Tokens']
        return Known(value) if value is not None else Unknown('invalid_cli_counter')

    cache_read = required['cacheReadTokens']
    cache_write = required['cacheWriteTokens']
    input_raw = required['inputTokens']
    input_value: Known | Unknown
    if input_raw is None:
        input_value = Unknown('invalid_cli_counter')
    elif cache_read is None or cache_write is None:
        input_value = Unknown('invalid_cli_counter')
    elif cache_read or cache_write:
        input_value = Unknown('cache_inclusion_unknown')
    else:
        input_value = Known(input_raw)
    tokens = TokenEvidence(
        TokenBreakdown(input_value, token('output'), token('cacheRead'), token('cacheWrite')),
        Unknown('not_reported'),
        Known(reasoning) if reasoning is not None and 'reasoning' not in invalid else Unknown('invalid_cli_counter') if reasoning_present else Unknown('not_reported'),
        Unknown('not_reported'))

    quantities: list[MeasuredQuantity] = []
    count_presence = int(requests is not None and 'count' in requests)
    cost_presence = int(requests is not None and 'cost' in requests)
    count = _count(requests.get('count')) if requests is not None and count_presence else None
    cost = _decimal(requests.get('cost')) if requests is not None and cost_presence else None
    if count_presence:
        if count is None:
            invalid.append('request_count')
            quantities.append(_unknown_quantity('request_count', 'invalid_cli_counter'))
        else:
            quantities.append(_quantity('request_count', Decimal(count), 'copilot-cli-model'))
    if cost_presence and cost is None:
        invalid.append('request_cost')
    if not count_presence:
        quantities.append(_unknown_quantity('request_count', 'not_reported'))
    if invalid_shape:
        tokens = _unknown_tokens('invalid_cli_counter')
        quantities = [_unknown_quantity('request_count', 'invalid_cli_counter')]

    native: dict[str, object] = {}
    for name in ('totalNanoAiu', 'tokenDetails'):
        if name not in raw:
            continue
        bounded = _token_details(raw[name]) if name == 'tokenDetails' else _native(raw[name])
        if bounded is None:
            invalid.append(name)
        else:
            native['total_nano_aiu' if name == 'totalNanoAiu' else 'token_details'] = bounded

    entry = EntryIdentity(_entry_id(event_id, 'model', model), parent_id, line)
    record = UsageRecord(entry, 'request_summary', Point(at), ModelIdentity(None, model), tokens,
                         MissingEstimate('not_recorded'), None, None, (), tuple(quantities))
    counters: dict[str, object] = {
        'scope': 'model', 'model': model,
        'requests_presence': count_presence, 'cost_presence': cost_presence,
        'input': required['inputTokens'], 'output': required['outputTokens'],
        'cache_read': required['cacheReadTokens'], 'cache_write': required['cacheWriteTokens'],
        'reasoning_presence': int(reasoning_present), 'reasoning': reasoning,
        'presence': presence,
    }
    if count is not None:
        counters['requests'] = count
    if cost is not None:
        counters['cost'] = str(cost)
    if invalid:
        counters['invalid'] = sorted(set(invalid))
    counters.update(native)
    evidence = CopilotCLIEvidence(CLI_PROFILE, 'shutdown', event_id, parent_id, None, None, None,
                                  current_model, None, epoch, _json(counters), 'unresolved' if invalid else 'usable',
                                  'invalid_cli_counter' if invalid else None, line, 'model')
    return record, evidence


def _valid_context(value: object) -> bool:
    context = _object(value)
    cwd = context.get('cwd') if context is not None else None
    if context is None or not (isinstance(cwd, str) and cwd.strip()
                               and len(cwd) <= 4096 and Path(cwd).is_absolute()):
        return False
    if 'pendingGitContext' in context and type(context['pendingGitContext']) is not bool:
        return False
    root = context.get('gitRoot')
    return 'gitRoot' not in context or (
        isinstance(root, str) and bool(root.strip()) and len(root) <= 4096 and Path(root).is_absolute())


def _context_identity(value: dict[str, object]) -> str | None:
    """Return the settled attribution identity; pending contexts are preliminary."""
    if not _valid_context(value) or value.get('pendingGitContext') is True:
        return None
    cwd = value.get('cwd')
    cwd_value = cwd if isinstance(cwd, str) and cwd.strip() and Path(cwd).is_absolute() else None
    root = value.get('gitRoot')
    root_value = root if isinstance(root, str) and root.strip() and len(root) <= 4096 and Path(root).is_absolute() else None
    return root_value or cwd_value


Envelope = tuple[str, str | None, datetime, dict[str, object]]


def _qualify_events(rows: list[tuple[int, dict[str, object]]], expected: str,
                    diagnostics: list[Diagnostic]) -> tuple[int, Envelope, datetime, int, dict[int, Envelope]] | RejectedSource:
    """Shared content-free start, envelope and physical-chain qualification."""
    starts = [(line, row) for line, row in rows if row.get('type') == 'session.start']
    if len(starts) != 1:
        return RejectedSource((Diagnostic('invalid_session_start', None, None, None),))
    start_line, start_row = starts[0]
    start = _envelope(start_row, start=True)
    if start is None or any(line < start_line and _envelope(row) is not None for line, row in rows):
        return RejectedSource((Diagnostic('invalid_session_start', start_line, None, None),))
    start_id, _, _, body = start
    started = _time(body.get('startTime'), millisecond=True)
    millis = _epoch_millis(started) if started is not None else None
    if (body.get('sessionId') != expected or type(body.get('version')) is not int
            or body.get('version') != 1 or body.get('producer') not in ('copilot-cli', 'github-copilot-cli')
            or _text(body.get('copilotVersion')) is None or started is None or millis is None
            or 'context' in body and not _valid_context(body['context'])):
        return RejectedSource((Diagnostic('conflicting_session_identity', start_line, start_id, None),))
    envelopes: dict[int, Envelope] = {}
    positions: dict[str, int] = {}
    for index, (line, row) in enumerate(rows):
        envelope = start if line == start_line else _envelope(row)
        if envelope is None or not _conversation_agent(row)[0]:
            continue
        event_id = envelope[0]
        if event_id in positions:
            diagnostics.append(Diagnostic('duplicate_event_id', line, event_id, None))
            continue
        envelopes[index] = envelope
        positions[event_id] = index
    rejected: set[str] = set()
    for index, envelope in tuple(envelopes.items()):
        event_id, parent, at, _ = envelope
        if at < started:
            return RejectedSource((Diagnostic('invalid_session_lifetime', start_line, start_id, None),))
        if parent is not None and (positions.get(parent, -1) >= index or parent in rejected):
            diagnostics.append(Diagnostic('invalid_event_chain', rows[index][0], event_id, None))
            rejected.add(event_id)
            del envelopes[index]
        elif parent is not None and parent not in positions:
            diagnostics.append(Diagnostic('event_chain_gap', rows[index][0], event_id, None))
    return start_line, start, started, millis, envelopes


def read_copilot_cli(data: bytes, *, locator: str, context: tuple[tuple[str, bytes], ...] = (),
                     profile: str = CLI_PROFILE) -> CopilotCLIReadBatch | RejectedSource:
    expected = _locator_session(locator)
    if expected is None:
        return RejectedSource((Diagnostic('conflicting_session_identity', None, None, None),))
    if Path(locator).name == 'workspace.yaml':
        return _workspace_batch(data, locator, expected)
    if profile != CLI_PROFILE:
        return RejectedSource((Diagnostic('unsupported_profile', None, None, None),))
    rows, diagnostics, complete_bytes, pending = _parse_lines(data)
    qualified = _qualify_events(rows, expected, diagnostics)
    if isinstance(qualified, RejectedSource):
        return qualified
    start_line, start_envelope, start_time, start_ms, envelopes = qualified
    start_id, _, start_at, start_data = start_envelope
    context_data = _object(start_data.get('context')) or {}
    epoch = sha256(_json([expected, 1, start_ms]).encode()).hexdigest()

    entries: list[EntryIdentity] = [EntryIdentity(start_id, None, start_line)]
    usage_records: list[UsageRecord] = []
    evidence: list[CopilotCLIEvidence] = []
    observed_times: list[datetime] = [start_at]
    contexts: list[str] = []
    initial_context = _context_identity(context_data)
    if initial_context is not None:
        contexts.append(initial_context)
    conversation_agents: set[str] = set()

    start_entry = EntryIdentity(_entry_id(start_id, 'metadata'), None, start_line)
    usage_records.append(UsageRecord(start_entry, 'usage_checkpoint', Point(start_at), ModelIdentity(None, None),
                                     _unknown_tokens('metadata_only'), MissingEstimate('not_recorded'), None, None, (),
                                     _unknown_quantities('events_unavailable')))
    evidence.append(CopilotCLIEvidence(CLI_PROFILE, 'metadata', start_id, None, '1',
                                       cast(str, start_data['copilotVersion']), start_ms * 1000,
                                       _text(start_data.get('selectedModel')), None, epoch,
                                       _json({'scope': 'metadata'}), 'usable', None, start_line, 'metadata'))

    final_shutdown_line: int | None = None
    final_shutdown_type: str | None = None
    for index, (line, row) in enumerate(rows):
        kind = row.get('type')
        if kind == 'session.start':
            continue
        envelope = envelopes.get(index)
        if envelope is None:
            if kind in ('session.shutdown', 'session.usage_checkpoint', 'assistant.usage'):
                code = 'ephemeral_event_in_history' if row.get('ephemeral') is True else 'invalid_cli_event'
                diagnostics.append(Diagnostic(code, line, _uuid4(row.get('id')), None))
            continue
        event_id, parent_id, at, body = envelope
        assert parent_id is not None
        observed_times.append(at)
        entries.append(EntryIdentity(event_id, parent_id, line))
        if kind == 'assistant.usage':
            diagnostics.append(Diagnostic('ephemeral_event_in_history', line, event_id, None))
            continue
        valid_agent, agent_id = _conversation_agent(row)
        if not valid_agent:
            diagnostics.append(Diagnostic('invalid_cli_event', line, event_id, None))
            continue
        if kind in ('user.message', 'assistant.message', 'assistant.reasoning',
                    'tool.execution_start', 'tool.execution_complete'):
            if agent_id not in (None, 'main') and agent_id is not None:
                conversation_agents.add(agent_id)
            continue
        if kind == 'session.context_changed':
            if not _valid_context(body):
                diagnostics.append(Diagnostic('invalid_cli_event', line, event_id, None))
                continue
            changed = _context_identity(body)
            if changed is not None:
                contexts.append(changed)
            continue
        if kind not in ('session.shutdown', 'session.usage_checkpoint'):
            continue
        if 'agentId' in row:
            diagnostics.append(Diagnostic('invalid_cli_event', line, event_id, None))
            continue
        session_quantities: list[MeasuredQuantity] = []
        raw_quantities: dict[str, str] = {}
        invalid_fields: list[str] = []
        presence: dict[str, int] = {}
        for field, measure in (('totalNanoAiu', 'nano_aiu'), ('totalPremiumRequests', 'premium_requests')):
            presence[measure] = int(field in body)
            if field not in body:
                if kind == 'session.usage_checkpoint' and field == 'totalNanoAiu':
                    invalid_fields.append(measure)
                    session_quantities.append(_unknown_quantity(cast(Literal['nano_aiu', 'premium_requests', 'request_count'], measure), 'invalid_cli_counter'))
                continue
            value = _decimal(body[field])
            if value is None:
                diagnostics.append(Diagnostic('invalid_cli_counter', line, event_id, measure))
                invalid_fields.append(measure)
                session_quantities.append(_unknown_quantity(cast(Literal['nano_aiu', 'premium_requests', 'request_count'], measure), 'invalid_cli_counter'))
            else:
                session_quantities.append(_quantity(cast(Literal['nano_aiu', 'premium_requests'], measure), value,
                                                    'copilot-cli-session'))
                raw_quantities[measure] = str(value)
        if kind == 'session.usage_checkpoint':
            counters: dict[str, object] = {'scope': 'session', 'presence': presence, **raw_quantities}
            if invalid_fields:
                counters['invalid'] = sorted(set(invalid_fields))
            entry = EntryIdentity(_entry_id(event_id, 'session'), parent_id, line)
            usage_records.append(UsageRecord(entry, 'usage_checkpoint', Point(at), ModelIdentity(None, None),
                                             _na_tokens(), MissingEstimate('not_recorded'), None, None, (),
                                             tuple(session_quantities)))
            evidence.append(CopilotCLIEvidence(CLI_PROFILE, 'usage_checkpoint', event_id, parent_id, None,
                                               None, start_ms * 1000, None, None, epoch, _json(counters),
                                               'unresolved' if invalid_fields else 'usable', 'invalid_cli_counter' if invalid_fields else None,
                                               line, 'session'))
            continue

        shutdown_start = _count(body.get('sessionStartTime'))
        shutdown_type = _text(body.get('shutdownType'))
        metrics = _object(body.get('modelMetrics'))
        changes = _object(body.get('codeChanges'))
        valid_changes = (changes is not None and isinstance(changes.get('filesModified'), list)
                         and all(isinstance(name, str) for name in cast(list[object], changes['filesModified']))
                         and _count(changes.get('linesAdded')) is not None
                         and _count(changes.get('linesRemoved')) is not None)
        valid_shutdown = (shutdown_start == start_ms and shutdown_type in ('routine', 'error')
                          and _count(body.get('totalApiDurationMs')) is not None
                          and metrics is not None and valid_changes)
        if shutdown_start != start_ms:
            diagnostics.append(Diagnostic('incompatible_session_epoch', line, event_id, None))
            invalid_fields.append('epoch')
        if shutdown_type not in ('routine', 'error') or _count(body.get('totalApiDurationMs')) is None:
            diagnostics.append(Diagnostic('invalid_cli_event', line, event_id, None))
            invalid_fields.append('shutdown')
        if metrics is None or not valid_changes:
            diagnostics.append(Diagnostic('invalid_cli_event', line, event_id, None))
            invalid_fields.append('shutdown')
        current_model = _text(body.get('currentModel'))
        agents = _object(body.get('agentMetrics'))
        if 'agentMetrics' in body and agents is None:
            invalid_fields.append('agentMetrics')
            diagnostics.append(Diagnostic('invalid_cli_counter', line, event_id, None))
        counters = {'scope': 'session', 'presence': presence, **raw_quantities}
        counters['duration'] = _count(body.get('totalApiDurationMs'))
        counters['current_model_presence'] = int('currentModel' in body)
        counters['agent_metrics_presence'] = int('agentMetrics' in body)
        if valid_changes and changes is not None:
            counters['code_changes'] = {name: changes[name] for name in ('linesAdded', 'linesRemoved')}
        if shutdown_type is not None:
            counters['shutdown_type'] = shutdown_type
        if 'tokenDetails' in body:
            token_details = _token_details(body['tokenDetails'])
            if token_details is None:
                invalid_fields.append('tokenDetails')
            else:
                counters['token_details'] = token_details
        if invalid_fields:
            counters['invalid'] = sorted(set(invalid_fields))
        entry = EntryIdentity(_entry_id(event_id, 'session'), parent_id, line)
        usage_records.append(UsageRecord(entry, 'usage_checkpoint', Point(at), ModelIdentity(None, None),
                                         _na_tokens(), MissingEstimate('not_recorded'), None, None, (),
                                         tuple(session_quantities)))
        evidence.append(CopilotCLIEvidence(CLI_PROFILE, 'shutdown', event_id, parent_id, None, None,
                                           start_ms * 1000, current_model, None, epoch, _json(counters),
                                           'unresolved' if invalid_fields else 'usable', 'invalid_cli_event' if invalid_fields else None,
                                           line, 'session'))
        if not valid_shutdown:
            continue
        final_shutdown_line = line
        final_shutdown_type = cast(str, shutdown_type)
        assert metrics is not None
        model_names = sorted(name for name in metrics if _text(name) is not None)
        counters['models'] = model_names
        # The persisted session evidence must carry the presence snapshot used
        # to quarantine a removed model on the next cumulative control.
        latest_evidence = replace(evidence[-1], counters_json=_json(counters))
        evidence[-1] = latest_evidence
        for model, raw_metric in metrics.items():
            model_name = _text(model)
            if model_name is None:
                diagnostics.append(Diagnostic('invalid_cli_counter', line, event_id, None))
                continue
            model_item = _model_record(event_id, parent_id, line, at, epoch, model_name,
                                       _object(raw_metric) or {}, current_model)
            if model_item is None:
                diagnostics.append(Diagnostic('invalid_cli_counter', line, event_id, None))
                continue
            record, proof = model_item
            usage_records.append(record)
            evidence.append(proof)
        if agents is not None:
            for raw_agent, raw_metrics in agents.items():
                agent = _text(raw_agent)
                metrics_value = _object(raw_metrics)
                if agent is None or metrics_value is None:
                    diagnostics.append(Diagnostic('invalid_cli_counter', line, event_id, None))
                    continue
                native: dict[str, object] = {}
                for name in ('totalApiDurationMs', 'totalNanoAiu', 'modelMetrics'):
                    if name in metrics_value:
                        native_value = (_metric_map(metrics_value[name]) if name == 'modelMetrics'
                                        else _count(metrics_value[name]) if name == 'totalApiDurationMs'
                                        else _native(metrics_value[name]))
                        if native_value is not None:
                            native[name] = native_value
                agent_entry = EntryIdentity(_entry_id(event_id, 'agent', agent), parent_id, line)
                usage_records.append(UsageRecord(agent_entry, 'request_summary', Point(at), ModelIdentity(None, None),
                                                 _unknown_tokens('non_additive_agent_breakdown'),
                                                 MissingEstimate('not_recorded'), None, None, (), ()))
                agent_counters: dict[str, object] = {'scope': 'agent', 'agent': agent, **native}
                projected_models = _object(native.get('modelMetrics'))
                valid_agent_metric = len(native) == 3 and projected_models is not None and not any(
                    isinstance(metric, dict) and metric.get('invalid') for metric in projected_models.values())
                if not valid_agent_metric:
                    diagnostics.append(Diagnostic('invalid_cli_counter', line, event_id, None))
                    agent_counters['invalid'] = ['agent_core']
                evidence.append(CopilotCLIEvidence(CLI_PROFILE, 'shutdown', event_id, parent_id, None, None,
                                                   start_ms * 1000, current_model, agent, epoch,
                                                   _json(agent_counters), 'usable' if valid_agent_metric else 'unresolved',
                                                   None if valid_agent_metric else 'invalid_cli_counter', line, 'agent'))

    later_durable = pending
    if final_shutdown_line is not None:
        later_durable = later_durable or any(
            line > final_shutdown_line and row.get('type') != 'assistant.usage'
            and row.get('ephemeral') is not True for line, row in rows)
        later_durable = later_durable or any(
            item.line is not None and item.line > final_shutdown_line
            and item.code in ('malformed_json', 'invalid_cli_event', 'invalid_cli_counter',
                              'invalid_event_chain', 'event_chain_gap', 'duplicate_event_id')
            for item in diagnostics)
    complete_lifetime = bool(final_shutdown_line is not None and final_shutdown_type == 'routine'
                             and not later_durable)
    if not complete_lifetime and any(item.source_kind != 'metadata' for item in evidence):
        diagnostics.append(Diagnostic('cli_partial_lifetime', final_shutdown_line, None, None))
    adjusted: list[UsageRecord] = []
    for record, evidence_item in zip(usage_records, evidence, strict=True):
        quantities = tuple(replace(quantity, lower_bound=evidence_item.source_kind == 'usage_checkpoint')
                           if quantity.state == 'known' else quantity for quantity in record.quantities)
        adjusted.append(replace(record, quantities=quantities))
    usage_records = adjusted

    distinct_contexts = tuple(dict.fromkeys(contexts))
    if len(distinct_contexts) == 1:
        settled_cwd = distinct_contexts[0]
    else:
        settled_cwd = None
        if len(distinct_contexts) > 1:
            diagnostics.append(Diagnostic('workspace_changed_cumulative_scope', None, None, None))
    sidecar = next((value for name, value in context if name == 'workspace.yaml'), None)
    if not distinct_contexts and sidecar is not None:
        metadata = read_workspace_metadata(sidecar)
        if metadata.session_id not in (None, expected):
            diagnostics.append(Diagnostic('invalid_workspace_metadata', None, None, None))
        else:
            settled_cwd = metadata.cwd
            diagnostics.extend(Diagnostic(code, None, None, None) for code in metadata.diagnostics)
    if any(value < start_time for value in observed_times):
        return RejectedSource((Diagnostic('invalid_session_lifetime', start_line, start_id, None),))
    try:
        session = SessionMetadata(SessionId('copilot-cli:' + expected), expected, start_time, settled_cwd,
                                  None, None, last_observed=max(observed_times))
    except (ValueError, OverflowError):
        return RejectedSource((Diagnostic('invalid_session_lifetime', start_line, start_id, None),))
    agent_ids = tuple(sorted(conversation_agents))
    # Child sessions are persisted by storage from the supported conversation IDs.
    return CopilotCLIReadBatch(session, tuple(entries), tuple(usage_records), tuple(diagnostics), complete_bytes,
                               pending, (), tuple(evidence), expected, agent_ids)
