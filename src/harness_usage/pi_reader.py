"""Pi's source boundary: accounting evidence and an approved bounded title excerpt."""
from dataclasses import dataclass, replace
from datetime import datetime
from decimal import Decimal
import json
import re
from typing import Literal, cast

from .domain import (
    ContractViolation, Interval, Known, MissingEstimate, ModelIdentity, Point,
    MeasuredQuantity, RecordedEstimate, RecordedMoney, SessionId, TimeEvidence, TokenBreakdown,
    TokenEvidence, TokenValue, Undated, Unknown, instant,
)

PROFILE = 'pi-v3/0.85.1-shape-1'
UsageKind = Literal['assistant', 'compaction', 'branch_summary', 'tool_result', 'request_summary', 'usage_checkpoint']
SafeFact = tuple[str, str | int | Decimal | None]
SAFE_FACT_NAMES = frozenset(
    ['entry.type', 'entry.timestamp']
    + [prefix + name for prefix in ('', 'message.') for name in ('role', 'provider', 'model', 'stopReason', 'timestamp', 'toolCallId')]
    + ['usage.' + name for name in ('input', 'output', 'cacheRead', 'cacheWrite', 'totalTokens', 'reasoning', 'cacheWrite1h', 'outputFinality', 'writerVersionToken')]
    + ['usage.cost.' + name for name in ('input', 'output', 'cacheRead', 'cacheWrite', 'total')]
)


@dataclass(frozen=True, slots=True)
class SessionMetadata:
    id: SessionId
    native_id: str
    started: datetime | None
    cwd: str | None
    parent_locator: str | None
    display_name: str | None
    last_observed: datetime | None = None
    title_excerpt: str | None = None

    def __post_init__(self) -> None:
        if any(not isinstance(value, str) or not value.strip() for value in (self.id, self.native_id)):
            raise ContractViolation('invalid_session_identity')
        if any(value is not None and not isinstance(value, str) for value in (self.cwd, self.parent_locator, self.display_name)):
            raise ContractViolation('invalid_session_metadata')
        if self.title_excerpt is not None and (not isinstance(self.title_excerpt, str) or not 1 <= len(self.title_excerpt) <= 160):
            raise ContractViolation('invalid_title_excerpt')
        for name in ('started', 'last_observed'):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, instant(value))
        if self.started is not None and self.last_observed is not None and self.last_observed < self.started:
            raise ContractViolation('invalid_session_lifetime')


@dataclass(frozen=True, slots=True)
class EntryIdentity:
    native_id: str
    parent_id: str | None
    line: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.native_id, str) or not self.native_id.strip() or (self.parent_id is not None and (not isinstance(self.parent_id, str) or not self.parent_id.strip())):
            raise ContractViolation('invalid_entry_identity')
        if type(self.line) is not int or self.line < 0:
            raise ContractViolation('invalid_entry_line')


@dataclass(frozen=True, slots=True)
class UsageRecord:
    entry: EntryIdentity
    kind: UsageKind
    time: TimeEvidence
    model: ModelIdentity
    tokens: TokenEvidence
    money: RecordedMoney
    stop_reason: str | None
    tool_call_id: str | None
    safe_facts: tuple[SafeFact, ...]
    quantities: tuple[MeasuredQuantity, ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in ('assistant', 'compaction', 'branch_summary', 'tool_result', 'request_summary', 'usage_checkpoint'):
            raise ContractViolation('invalid_usage_kind')
        if not isinstance(self.entry, EntryIdentity) or not isinstance(self.time, (Point, Interval, Undated)) or not isinstance(self.model, ModelIdentity) or not isinstance(self.tokens, TokenEvidence) or not isinstance(self.money, (RecordedEstimate, MissingEstimate)):
            raise ContractViolation('invalid_usage_variant')
        if any(value is not None and (not isinstance(value, str) or not value.strip()) for value in (self.stop_reason, self.tool_call_id)):
            raise ContractViolation('invalid_usage_identity')
        if type(self.safe_facts) is not tuple:
            raise ContractViolation('immutable_safe_facts_required')
        if (type(self.quantities) is not tuple
                or any(not isinstance(value, MeasuredQuantity) for value in self.quantities)
                or len({value.measure for value in self.quantities}) != len(self.quantities)):
            raise ContractViolation('invalid_quantities')
        seen: set[str] = set()
        for pair in self.safe_facts:
            if type(pair) is not tuple or len(pair) != 2:
                raise ContractViolation('invalid_safe_fact')
            name, value = pair
            if name not in SAFE_FACT_NAMES or name in seen or not (value is None or type(value) in (str, int) or isinstance(value, Decimal) and value.is_finite()):
                raise ContractViolation('invalid_safe_fact')
            seen.add(name)


@dataclass(frozen=True, slots=True)
class Diagnostic:
    code: str
    line: int | None
    entry_id: str | None
    measure: str | None


@dataclass(frozen=True, slots=True)
class DelegationRef:
    entry_id: str
    kind: Literal['child_path', 'child_path_hash', 'child_name', 'child_id', 'agent_id']
    value: str
    owner_path: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in ('child_path', 'child_path_hash', 'child_name', 'child_id', 'agent_id') or any(
            not isinstance(value, str) or not value.strip() for value in (self.entry_id, self.value)
        ) or (self.owner_path is not None and (not isinstance(self.owner_path, str) or not self.owner_path.strip())):
            raise ContractViolation('invalid_delegation_reference')
        if self.kind == 'child_path_hash' and re.fullmatch(r'[0-9a-f]{64}', self.value) is None:
            raise ContractViolation('invalid_delegation_reference')


@dataclass(frozen=True, slots=True)
class ReadBatch:
    session: SessionMetadata
    entries: tuple[EntryIdentity, ...]
    usage: tuple[UsageRecord, ...]
    diagnostics: tuple[Diagnostic, ...]
    complete_bytes: int
    pending_tail: bool
    delegations: tuple[DelegationRef, ...] = ()


@dataclass(frozen=True, slots=True)
class RejectedSource:
    diagnostics: tuple[Diagnostic, ...]


ReadResult = ReadBatch | RejectedSource


def _object(value: object) -> dict[str, object] | None:
    return cast(dict[str, object], value) if isinstance(value, dict) else None


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate_json_key')
        result[key] = value
    return result


def _text(value: object) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _child_name(run_id: str, agent: object, index: object) -> str | None:
    if _text(agent) is None or type(index) is not int or index < 0:
        return None
    # Exact Pi intercom writer formula, with recorded agent/run/index coordinates.
    parts = [re.sub(r'[^a-z0-9_-]+', '-', part.strip().lower()).strip('-') or 'agent' for part in (cast(str, agent), run_id)]
    return f'subagent-{parts[0]}-{parts[1]}-{index + 1}'


def _single_delegations(calls: dict[str, str | None], runs: list[tuple[str, str, str]]) -> tuple[DelegationRef, ...]:
    refs = []
    for entry_id, run_id, call_id in runs:
        name = _child_name(run_id, calls.get(call_id), 0)
        if name is not None:
            refs.append(DelegationRef(entry_id, 'child_name', name))
    return tuple(dict.fromkeys(refs))


def _delegations(raw: dict[str, object], calls: dict[str, str | None],
                 single_runs: list[tuple[str, str, str]]) -> tuple[DelegationRef, ...]:
    """Read only linkage fields from known tool formats, never arbitrary result content."""
    entry_id = _text(raw.get('id'))
    if entry_id is None or (raw.get('parentId') is not None and _text(raw['parentId']) is None):
        return ()
    refs: list[DelegationRef] = []
    if raw.get('type') == 'custom' and raw.get('customType') == 'subagents:record':
        data = _object(raw.get('data')) or {}
        owner = _text(data.get('originParentSessionFile'))
        if data.get('originParentSessionFile') is not None and owner is None:
            return ()
        for field, kind in (('sessionFile', 'child_path'), ('id', 'agent_id')):
            value = _text(data.get(field))
            if value is not None:
                refs.append(DelegationRef(entry_id, cast(Literal['child_path', 'agent_id'], kind), value, owner))
    if raw.get('type') == 'custom_message' and raw.get('customType') == 'subagent_control_notice':
        target = _text((_object(raw.get('details')) or {}).get('childIntercomTarget'))
        if target is not None:
            refs.append(DelegationRef(entry_id, 'child_name', target))
    body = _object(raw.get('message')) or {}
    content = body.get('content')
    if raw.get('type') == 'message' and body.get('role') == 'assistant' and isinstance(content, list):
        for item in content:
            call = _object(item) or {}
            call_id = _text(call.get('id'))
            if call.get('type') == 'toolCall' and call_id is not None:
                arguments = _object(call.get('arguments')) or {}
                agent = _text(arguments.get('agent')) if call.get('name') == 'subagent' and not any(key in arguments for key in ('tasks', 'chain')) else None
                if call_id in calls and calls[call_id] != agent:
                    calls[call_id] = None
                else:
                    calls.setdefault(call_id, agent)
    if raw.get('type') == 'message' and body.get('role') == 'toolResult':
        details = _object(body.get('details')) or {}
        if body.get('toolName') == 'Agent':
            agent_id = _text(details.get('agentId'))
            if agent_id is not None:
                refs.append(DelegationRef(entry_id, 'agent_id', agent_id))
        elif body.get('toolName') == 'subagent':
            run_id = _text(details.get('runId'))
            call_id = _text(body.get('toolCallId'))
            if run_id is not None and call_id is not None and details.get('mode') == 'single' and details.get('asyncId') == run_id:
                single_runs.append((entry_id, run_id, call_id))
            graph = _object(details.get('workflowGraph')) or {}
            if run_id is not None and graph.get('runId') == run_id:
                graph_nodes = graph.get('nodes')
                nodes = list(graph_nodes) if isinstance(graph_nodes, list) else []
                while nodes:
                    node = _object(nodes.pop()) or {}
                    children = node.get('children')
                    if isinstance(children, list):
                        nodes.extend(children)
                    if node.get('kind') in ('agent', 'step'):
                        child_name = _child_name(run_id, node.get('agent'), node.get('flatIndex'))
                        if child_name is not None:
                            refs.append(DelegationRef(entry_id, 'child_name', child_name))
            results = details.get('results')
            if run_id is not None and isinstance(results, list):
                for result in results:
                    fields = _object(result) or {}
                    child_name = _child_name(run_id, fields.get('agent'), fields.get('index'))
                    if child_name is not None:
                        refs.append(DelegationRef(entry_id, 'child_name', child_name))
            lifecycle = _object(details.get('lifecycleStatus')) or {}
            terminal = _object(lifecycle.get('processTerminal')) or {}
            canonical = _object(terminal.get('canonicalSession')) or {}
            path_hash = _text(canonical.get('canonicalSessionId'))
            if path_hash is not None and re.fullmatch(r'[0-9a-f]{64}', path_hash):
                refs.append(DelegationRef(entry_id, 'child_path_hash', path_hash))
            workflow = _object(details.get('workflow')) or {}
            workflow_value = _object(workflow.get('value')) or {}
            for results in (details.get('results'), workflow_value.get('results')):
                if isinstance(results, list):
                    for result in results:
                        child = _text((_object(result) or {}).get('sessionFile'))
                        if child is not None:
                            refs.append(DelegationRef(entry_id, 'child_path', child))
    return tuple(dict.fromkeys(refs))


def read_delegations(data: bytes) -> tuple[DelegationRef, ...]:
    """Backfill allowlisted links without reconstructing accounting observations."""
    return read_metadata(data)[0]


def _title_excerpt(raw: dict[str, object]) -> str | None:
    body = _object(raw.get('message'))
    if raw.get('type') != 'message' or body is None or body.get('role') != 'user':
        return None
    content = body.get('content')
    if isinstance(content, list):
        content = ' '.join(block['text'] for block in content if isinstance(block, dict) and block.get('type') == 'text' and isinstance(block.get('text'), str))
    if not isinstance(content, str):
        return None
    text = ' '.join(content.split())
    return (text[:159] + '…' if len(text) > 160 else text) or None


def _session_header(raw: dict[str, object] | None) -> SessionMetadata | None:
    if raw is None or raw.get('type') != 'session' or type(raw.get('version')) is not int or raw['version'] != 3 or _text(raw.get('id')) is None:
        return None
    if any(raw.get(key) is not None and not isinstance(raw[key], str) for key in ('cwd', 'parentSession')) or ('timestamp' in raw and _timestamp(raw['timestamp']) is None):
        return None
    native_id = cast(str, raw['id'])
    started = _timestamp(raw.get('timestamp'))
    return SessionMetadata(SessionId('pi:' + native_id), native_id, started, _text(raw.get('cwd')), _text(raw.get('parentSession')), None, started)


def read_metadata(data: bytes) -> tuple[tuple[DelegationRef, ...], str | None]:
    """Scan links and an unnamed session's excerpt without decoding usage evidence."""
    lines = data.splitlines()
    try:
        header = _object(json.loads(lines[0], parse_float=Decimal, parse_constant=Decimal, object_pairs_hook=_unique_object)) if lines else None
    except (ValueError, UnicodeError, RecursionError):
        return (), None
    if _session_header(header) is None:
        return (), None
    refs: list[DelegationRef] = []
    calls: dict[str, str | None] = {}
    single_runs: list[tuple[str, str, str]] = []
    excerpt = None
    name = None
    for line in lines[1:]:
        try:
            raw = _object(json.loads(line, parse_float=Decimal, parse_constant=Decimal, object_pairs_hook=_unique_object))
        except (ValueError, UnicodeError, RecursionError):
            continue
        if raw is not None:
            refs.extend(_delegations(raw, calls, single_runs))
            if _text(raw.get('id')) is None or (raw.get('parentId') is not None and _text(raw['parentId']) is None):
                continue
            if raw.get('type') == 'session_info' and (raw.get('name') is None or isinstance(raw['name'], str)):
                name = _text(raw.get('name'))
            if excerpt is None:
                excerpt = _title_excerpt(raw)
    return tuple(dict.fromkeys((*refs, *_single_delegations(calls, single_runs)))), None if name else excerpt


def _timestamp(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return instant(datetime.fromisoformat(value))
    except (ValueError, OverflowError):
        return None


def _decimal(value: object) -> Decimal | None:
    number = Decimal(value) if type(value) is int else value
    return number if isinstance(number, Decimal) and number.is_finite() and number >= 0 else None


def _pending(error: json.JSONDecodeError, text: str) -> bool:
    # Only JSON prefixes cut off at EOF are retryable; malformed terminated lines are not.
    return error.pos >= len(text.rstrip()) or error.msg.startswith('Unterminated string') or (
        error.msg == "Expecting ',' delimiter" and bool(re.search(r'[0-9](?:[eE][+-]?|\.)$', text))
    ) or (
        error.msg == 'Invalid \\uXXXX escape' and bool(re.search(r'\\u[0-9a-fA-F]{0,3}$', text))
    ) or (
        error.msg == 'Expecting value' and text[error.pos:] in ('-', 't', 'tr', 'tru', 'f', 'fa', 'fal', 'fals', 'n', 'nu', 'nul')
    )


def _record(raw: dict[str, object], body: dict[str, object], entry: EntryIdentity,
            kind: UsageKind, diagnostics: list[Diagnostic]) -> UsageRecord:
    facts: list[SafeFact] = []

    def diagnose(code: str, measure: str | None = None) -> None:
        diagnostics.append(Diagnostic(code, entry.line, entry.native_id, measure))

    def fact(path: str, value: object) -> None:
        if path.startswith('usage.') and type(value) not in (int, Decimal):
            return
        if value is None or type(value) in (str, int) or (isinstance(value, Decimal) and value.is_finite()):
            facts.append((path, cast(str | int | Decimal | None, value)))

    for name in ('type', 'timestamp'):
        if name in raw:
            fact('entry.' + name, raw[name])
    prefix = 'message.' if raw.get('type') == 'message' else ''
    for name in ('role', 'provider', 'model', 'stopReason', 'timestamp', 'toolCallId'):
        if name in body and (prefix or name != 'timestamp'):
            fact(prefix + name, body[name])

    usage = _object(body.get('usage'))
    if usage is None:
        diagnose('missing_usage' if body.get('usage') is None else 'invalid_usage_shape')
        usage = {}
    fields = (('input', 'input'), ('output', 'output'), ('cache_read', 'cacheRead'),
              ('cache_write', 'cacheWrite'), ('total', 'totalTokens'),
              ('reasoning', 'reasoning'), ('cache_write_1h', 'cacheWrite1h'))
    counts: dict[str, TokenValue] = {}
    for measure, name in fields:
        if name not in usage:
            counts[measure] = Unknown('not_reported')
            continue
        value = usage[name]
        fact('usage.' + name, value)
        if type(value) is int and 0 <= value < 2**63:
            counts[measure] = Known(value)
        else:
            counts[measure] = Unknown('invalid_count')
            diagnose('invalid_count', measure)
    for subset, bucket in (('reasoning', 'output'), ('cache_write_1h', 'cache_write')):
        sub, whole = counts[subset], counts[bucket]
        if isinstance(sub, Known) and isinstance(whole, Known) and sub.value > whole.value:
            counts[subset] = Unknown('invalid_subset')
            diagnose('invalid_subset', subset)
    tokens = TokenEvidence(TokenBreakdown(*(counts[name] for name in ('input', 'output', 'cache_read', 'cache_write'))),
                           counts['total'], counts['reasoning'], counts['cache_write_1h'])
    if isinstance(tokens.total, Unknown) and tokens.total.reason == 'total_conflict':
        diagnose('total_conflict', 'total')

    money: RecordedMoney = MissingEstimate('not_reported')
    cost = _object(usage.get('cost'))
    if cost is not None:
        components: list[tuple[str, Decimal]] = []
        for measure, name in fields[:4]:
            if name in cost:
                fact('usage.cost.' + name, cost[name])
                value_money = _decimal(cost[name])
                if value_money is None:
                    diagnose('invalid_money', 'recorded_usd.' + measure)
                else:
                    components.append((measure, value_money))
        if 'total' in cost:
            fact('usage.cost.total', cost['total'])
            amount = _decimal(cost['total'])
            if amount is None:
                money = MissingEstimate('invalid_money')
                diagnose('invalid_money', 'recorded_usd')
            else:
                money = RecordedEstimate(amount, 'USD', tuple(components), 'pi:usage.cost.total')
                if len(components) == 4 and sum((v for _, v in components), Decimal(0)) != amount:
                    diagnose('cost_component_mismatch', 'recorded_usd')
    elif 'cost' in usage:
        diagnose('invalid_money', 'recorded_usd')
        money = MissingEstimate('invalid_money')

    timestamp = _timestamp(raw.get('timestamp'))
    time: TimeEvidence
    if timestamp is None:
        time = Undated('invalid_timestamp' if 'timestamp' in raw else 'missing_timestamp')
        diagnose(time.reason)
    else:
        time = Point(timestamp) if kind == 'assistant' else Interval(None, timestamp)
    for name in ('provider', 'model', 'stopReason', 'toolCallId'):
        if body.get(name) is not None and _text(body[name]) is None:
            diagnose('invalid_' + name)
    return UsageRecord(entry, kind, time, ModelIdentity(_text(body.get('provider')), _text(body.get('model'))),
                       tokens, money, _text(body.get('stopReason')), _text(body.get('toolCallId')), tuple(facts))


def read_pi(data: bytes, *, locator: str, profile: str = PROFILE) -> ReadResult:
    """Read one snapshot; its locator never determines session or observation identity."""
    if profile != PROFILE:
        return RejectedSource((Diagnostic('unsupported_profile', None, None, None),))
    session: SessionMetadata | None = None
    entries: list[EntryIdentity] = []
    records: list[UsageRecord] = []
    delegations: list[DelegationRef] = []
    calls: dict[str, str | None] = {}
    single_runs: list[tuple[str, str, str]] = []
    diagnostics: list[Diagnostic] = []
    seen: set[str] = set()
    complete_bytes = 0
    pending_tail = False
    excerpt = None
    lines = data.splitlines(keepends=True)
    for line, chunk in enumerate(lines, 1):
        try:
            text = chunk.decode('utf-8')
            raw = _object(json.loads(text, parse_float=Decimal, parse_constant=Decimal, object_pairs_hook=_unique_object))
        except (ValueError, UnicodeError, RecursionError) as error:
            if line == 1:
                return RejectedSource((Diagnostic('invalid_header', 1, None, None),))
            incomplete = ((isinstance(error, json.JSONDecodeError) and _pending(error, text))
                          or (isinstance(error, UnicodeDecodeError) and error.reason == 'unexpected end of data'))
            if line == len(lines) and not chunk.endswith(b'\n') and incomplete:
                pending_tail = True
                break
            diagnostics.append(Diagnostic('malformed_line', line, None, None))
            complete_bytes += len(chunk)
            continue
        complete_bytes += len(chunk)
        if line == 1:
            session = _session_header(raw)
            if session is None:
                return RejectedSource((Diagnostic('invalid_header', 1, None, None),))
            continue
        assert session is not None
        if raw is None:
            diagnostics.append(Diagnostic('invalid_entry', line, None, None))
            continue
        if raw.get('type') == 'session':
            diagnostics.append(Diagnostic('conflicting_header', line, None, None))
            continue
        native_entry = _text(raw.get('id'))
        if native_entry is None or (raw.get('parentId') is not None and _text(raw['parentId']) is None):
            diagnostics.append(Diagnostic('invalid_entry_identity', line, native_entry, None))
            continue
        entry = EntryIdentity(native_entry, _text(raw.get('parentId')), line)
        if excerpt is None:
            excerpt = _title_excerpt(raw)
        entries.append(entry)
        delegations.extend(_delegations(raw, calls, single_runs))
        if native_entry in seen:
            diagnostics.append(Diagnostic('duplicate_entry_id', line, native_entry, None))
        seen.add(native_entry)
        at = _timestamp(raw.get('timestamp'))
        if at is not None and (session.last_observed is None or at > session.last_observed):
            session = replace(session, last_observed=at)
        entry_type = raw.get('type')
        body = _object(raw.get('message'))
        if entry_type == 'session_info':
            if raw.get('name') is not None and not isinstance(raw['name'], str):
                diagnostics.append(Diagnostic('invalid_session_name', line, native_entry, None))
            else:
                session = replace(session, display_name=_text(raw.get('name')))
        if entry_type == 'message' and body is not None and body.get('role') in ('assistant', 'toolResult'):
            if body['role'] == 'toolResult' and 'usage' not in body:
                continue
            kind: UsageKind = 'assistant' if body['role'] == 'assistant' else 'tool_result'
            records.append(_record(raw, body, entry, kind, diagnostics))
        elif entry_type in ('compaction', 'branch_summary'):
            records.append(_record(raw, raw, entry, entry_type, diagnostics))
        elif 'usage' in raw or (body is not None and 'usage' in body):
            diagnostics.append(Diagnostic('unsupported_usage_shape', line, native_entry, None))
        elif entry_type == 'message' and body is None:
            diagnostics.append(Diagnostic('invalid_message_shape', line, native_entry, None))
    if session is None:
        return RejectedSource((Diagnostic('invalid_header', 1, None, None),))
    session = replace(session, title_excerpt=None if session.display_name else excerpt)
    return ReadBatch(session, tuple(entries), tuple(records), tuple(diagnostics), complete_bytes, pending_tail, tuple(dict.fromkeys((*delegations, *_single_delegations(calls, single_runs)))))
