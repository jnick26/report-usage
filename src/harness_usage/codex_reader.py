"""Codex rollout usage boundary: response deltas and qualified legacy counters."""
from dataclasses import dataclass, replace
from datetime import datetime
from decimal import Decimal
import hashlib
import json
import re
from typing import Literal, cast

from .domain import ContractViolation, Known, ModelIdentity, MissingEstimate, Point, SessionId, TokenBreakdown, TokenEvidence, TokenValue, Undated, Unknown, instant
from .pi_reader import Diagnostic, EntryIdentity, ReadBatch, RejectedSource, SessionMetadata, UsageRecord, _pending, _unique_object

PROFILE = 'codex-rollout-v1'
CODEX_PROFILE = PROFILE
FIELDS = ('input_tokens', 'cached_input_tokens', 'cache_write_input_tokens', 'output_tokens', 'reasoning_output_tokens', 'total_tokens')
TokenVector = tuple[tuple[str, int | None], ...]


@dataclass(frozen=True, slots=True)
class CodexEvidence:
    entry_id: str
    source_kind: Literal['response', 'legacy']
    response_id: str | None
    thread_id: str | None
    turn_id: str | None
    cumulative: TokenVector | None
    last: TokenVector | None
    state: Literal['usable', 'unresolved'] = 'usable'
    reason: str | None = None
    mirror_response_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.entry_id, str) or not self.entry_id.strip() or self.source_kind not in ('response', 'legacy') or self.state not in ('usable', 'unresolved'):
            raise ContractViolation('invalid_codex_evidence')
        if any(value is not None and (not isinstance(value, str) or not value.strip()) for value in (self.response_id, self.thread_id, self.turn_id, self.reason, self.mirror_response_id)):
            raise ContractViolation('invalid_codex_evidence_identity')
        if (self.state == 'unresolved') != (self.reason is not None):
            raise ContractViolation('invalid_codex_evidence_state')
        for vector in (self.cumulative, self.last):
            if vector is None:
                continue
            if type(vector) is not tuple or len({field for field, _ in vector}) != len(vector):
                raise ContractViolation('immutable_codex_vector_required')
            if any(field not in FIELDS or value is not None and (type(value) is not int or not 0 <= value < 2**63) for field, value in vector):
                raise ContractViolation('invalid_codex_vector')


@dataclass(frozen=True, slots=True)
class CodexReadBatch(ReadBatch):
    evidence: tuple[CodexEvidence, ...] = ()
    parent_thread_id: str | None = None
    forked_from_id: str | None = None
    root_session_id: str | None = None


def _object(value: object) -> dict[str, object]:
    return cast(dict[str, object], value) if isinstance(value, dict) else {}


def _text(value: object) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _time(value: object) -> datetime | None:
    try:
        return instant(datetime.fromisoformat(value)) if isinstance(value, str) else None
    except (ValueError, OverflowError):
        return None


def _vector(value: object) -> TokenVector | None:
    if not isinstance(value, dict):
        return None
    return tuple((field, value[field] if type(value[field]) is int and 0 <= value[field] < 2**63 else None) for field in FIELDS if field in value)


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, separators=(',', ':'), sort_keys=True).encode()).hexdigest()


def _tokens(vector: TokenVector | None, line: int, entry: str, diagnostics: list[Diagnostic]) -> TokenEvidence:
    raw = dict(vector or ())
    values: dict[str, TokenValue] = {}
    for field in FIELDS:
        if field not in raw:
            values[field] = Unknown('not_reported')
        elif raw[field] is None:
            values[field] = Unknown('invalid_count')
            diagnostics.append(Diagnostic('invalid_count', line, entry, field))
        else:
            values[field] = Known(cast(int, raw[field]))
    inclusive, cached, written = (values[field] for field in FIELDS[:3])
    fresh: TokenValue = Unknown('missing_cache_breakdown')
    if all(isinstance(value, Known) for value in (inclusive, cached, written)):
        total_in, cache_in, write_in = (cast(Known, value).value for value in (inclusive, cached, written))
        if cache_in + write_in <= total_in:
            fresh = Known(total_in - cache_in - write_in)
        else:
            fresh = cached = written = Unknown('invalid_count')
            diagnostics.append(Diagnostic('invalid_input_subsets', line, entry, 'input'))
    elif any(isinstance(value, Unknown) and value.reason == 'invalid_count' for value in (inclusive, cached, written)):
        fresh = Unknown('invalid_count')
    output, reasoning, reported = (values[field] for field in FIELDS[3:])
    if isinstance(reasoning, Known) and isinstance(output, Known) and reasoning.value > output.value:
        reasoning = Unknown('invalid_count')
        diagnostics.append(Diagnostic('invalid_reasoning_subset', line, entry, 'reasoning'))
    if isinstance(inclusive, Known) and isinstance(output, Known) and isinstance(reported, Known) and inclusive.value + output.value != reported.value:
        reported = Unknown('invalid_count')
        diagnostics.append(Diagnostic('total_mismatch', line, entry, 'total'))
    return TokenEvidence(TokenBreakdown(fresh, output, cached, written), reported, reasoning)


def _context_only(vector: TokenVector | None) -> bool:
    # TokenUsageInfo.fill_to_context_window emits this synthetic context delta.
    raw = dict(vector or ())
    return all(raw.get(field) == 0 for field in FIELDS[:-1]) and isinstance(raw.get('total_tokens'), int) and cast(int, raw['total_tokens']) > 0


def _advance(previous: TokenVector | None, current: TokenVector | None, last: TokenVector | None) -> str | None:
    if current is None or last is None:
        return 'missing_cumulative_usage' if current is None else 'missing_last_usage'
    now, delta = dict(current), dict(last)
    prior = dict(previous or ())
    core = ('input_tokens', 'output_tokens', 'total_tokens')
    if any(now.get(field) is None or delta.get(field) is None for field in core):
        return 'incomplete_cumulative_usage'
    if previous is None:
        return None if current == last else 'missing_cumulative_baseline'
    if any(prior.get(field) is None for field in core):
        return 'incomplete_cumulative_baseline'
    for field in set(now) | set(prior) | set(delta):
        if field not in now or field not in prior or field not in delta or now[field] is None or prior[field] is None or delta[field] is None:
            return 'incomplete_cumulative_usage'
        difference = cast(int, now[field]) - cast(int, prior[field])
        if difference < 0:
            return 'cumulative_regression'
        if difference != delta[field]:
            return 'cumulative_delta_mismatch'
    return None


def _excerpt(kind: object, payload: dict[str, object]) -> str | None:
    content: object = None
    if kind == 'event_msg' and payload.get('type') == 'user_message':
        content = payload.get('message')
    elif kind == 'response_item' and payload.get('type') == 'message' and payload.get('role') == 'user':
        blocks = payload.get('content')
        if isinstance(blocks, list):
            content = ' '.join(cast(str, block['text']) for block in blocks if isinstance(block, dict) and block.get('type') == 'input_text' and isinstance(block.get('text'), str))
    if not isinstance(content, str):
        return None
    if kind == 'response_item' and re.fullmatch(
        r'\s*(?:(?:<(environment_context|recommended_plugins)>.*?</\1>'
        r'|# AGENTS\.md instructions for [^\r\n]+\s+<INSTRUCTIONS>.*?</INSTRUCTIONS>)\s*)+', content, re.DOTALL,
    ):
        return None
    text = ' '.join(content.split())
    return (text[:159] + '…' if len(text) > 160 else text) or None


def read_codex(data: bytes, *, locator: str, profile: str = PROFILE) -> CodexReadBatch | RejectedSource:
    if profile != PROFILE:
        return RejectedSource((Diagnostic('unsupported_profile', None, None, None),))
    rows = data.splitlines(keepends=True)
    if not rows:
        return RejectedSource((Diagnostic('invalid_header', 1, None, None),))
    diagnostics: list[Diagnostic] = []
    session: SessionMetadata | None = None
    records: list[UsageRecord] = []
    evidence: list[CodexEvidence] = []
    parent = fork = root = excerpt = user_excerpt = None
    complete = 0
    pending = False
    current_turn: str | None = None
    current_model = ModelIdentity(None, None)
    model_by_turn: dict[str, ModelIdentity | None] = {}
    model_by_thread: dict[str, ModelIdentity] = {}
    previous: TokenVector | None = None
    seen_legacy: set[tuple[TokenVector | None, TokenVector | None]] = set()
    adjacent: tuple[str, str, str | None, TokenVector | None] | None = None
    for line, chunk in enumerate(rows, 1):
        try:
            text = chunk.decode('utf-8')
            raw = _object(json.loads(text, parse_float=Decimal, parse_constant=Decimal, object_pairs_hook=_unique_object))
        except (ValueError, UnicodeError, RecursionError) as error:
            if line == 1:
                return RejectedSource((Diagnostic('invalid_header', 1, None, None),))
            incomplete = isinstance(error, json.JSONDecodeError) and _pending(error, text) or isinstance(error, UnicodeDecodeError) and error.reason == 'unexpected end of data'
            if line == len(rows) and not chunk.endswith(b'\n') and incomplete:
                pending = True
                break
            diagnostics.append(Diagnostic('malformed_line', line, None, None))
            complete += len(chunk)
            continue
        complete += len(chunk)
        payload = _object(raw.get('payload'))
        kind = raw.get('type')
        if line == 1:
            identity = _text(payload.get('id'))
            started = _time(payload.get('timestamp', raw.get('timestamp')))
            if kind != 'session_meta' or identity is None or ('timestamp' in payload and started is None) or (payload.get('cwd') is not None and not isinstance(payload['cwd'], str)):
                return RejectedSource((Diagnostic('invalid_header', 1, None, None),))
            parent = _text(payload.get('parent_thread_id'))
            source = _object(payload.get('source'))
            spawned = _object(_object(source.get('subagent')).get('thread_spawn'))
            parent = parent or _text(spawned.get('parent_thread_id'))
            fork, root = _text(payload.get('forked_from_id')), _text(payload.get('session_id'))
            session = SessionMetadata(SessionId('codex:' + identity), identity, started, _text(payload.get('cwd')), None, _text(payload.get('agent_nickname')), started)
            current_model = ModelIdentity(_text(payload.get('model_provider')), None)
            model_by_thread[identity] = current_model
            continue
        assert session is not None
        at = _time(raw.get('timestamp'))
        if at is not None and (session.last_observed is None or at > session.last_observed):
            session = replace(session, last_observed=at)
        if kind == 'session_meta':
            continue  # Copied history metadata never changes this rollout's identity.
        if kind == 'event_msg' and user_excerpt is None:
            user_excerpt = _excerpt(kind, payload)
        elif excerpt is None:
            excerpt = _excerpt(kind, payload)
        if kind == 'turn_context':
            current_turn = _text(payload.get('turn_id'))
            model_name = _text(payload.get('model')) if 'model' in payload else current_model.model
            current_model = ModelIdentity(current_model.provider, model_name)
            if current_turn is not None:
                if current_turn in model_by_turn and model_by_turn[current_turn] != current_model:
                    model_by_turn[current_turn] = None
                else:
                    model_by_turn.setdefault(current_turn, current_model)
            model_by_thread[session.native_id] = current_model
            continue
        if kind == 'event_msg' and payload.get('type') == 'thread_settings_applied':
            settings = _object(payload.get('thread_settings'))
            owner = _text(payload.get('thread_id')) or session.native_id
            model_by_thread[owner] = ModelIdentity(_text(settings.get('model_provider_id')), _text(settings.get('model')))
            if owner == session.native_id:
                current_model = model_by_thread[owner]
                if current_turn is not None:
                    model_by_turn[current_turn] = current_model
            continue
        source_kind: Literal['response', 'legacy']
        response_id = mirror = reason = None
        state: Literal['usable', 'unresolved'] = 'usable'
        if kind == 'token_usage_record':
            source_kind = 'response'
            last = _vector(payload.get('usage'))
            cumulative = _vector(payload.get('thread_token_usage'))
            owner = _text(payload.get('thread_id')) or session.native_id
            turn = _text(payload.get('turn_id'))
            response_id = _text(payload.get('response_id'))
            entry_id = 'codex-response:' + response_id if response_id else 'codex-unkeyed:' + _digest((owner, turn, cumulative, last))
            if response_id is None:
                reason = 'missing_response_id'
            elif last is None:
                reason = 'missing_response_usage'
            adjacent = (response_id, owner, turn, last) if response_id is not None else None
            model = model_by_thread.get(owner, ModelIdentity(None, None))
            if turn in model_by_turn:
                selected = model_by_turn[turn]
                model = ModelIdentity(model.provider, selected.model if selected is not None else None)
            elif turn is not None:
                model = ModelIdentity(model.provider, None)
        elif kind == 'event_msg' and payload.get('type') == 'token_count':
            info = _object(payload.get('info'))
            if not info:
                continue  # Rate-limit-only updates are not accounting observations.
            source_kind = 'legacy'
            last, cumulative = _vector(info.get('last_token_usage')), _vector(info.get('total_token_usage'))
            owner, turn = session.native_id, current_turn
            if _context_only(last):
                previous = cumulative
                continue
            replay = (cumulative, last)
            if replay in seen_legacy or cumulative is not None and cumulative == previous:
                continue
            seen_legacy.add(replay)
            entry_id = 'codex-legacy:' + _digest((session.native_id, cumulative, last))
            reason = _advance(previous, cumulative, last)
            previous = cumulative
            if adjacent is not None and adjacent[2] == turn and turn is not None and adjacent[3] == last:
                mirror, owner = adjacent[0], adjacent[1]
                reason = 'modern_legacy_overlap'
            adjacent = None
            model = current_model
        else:
            continue
        if reason is not None:
            state = 'unresolved'
            diagnostics.append(Diagnostic(reason, line, entry_id, None))
        tokens = _tokens(last, line, entry_id, diagnostics)
        entry = EntryIdentity(entry_id, None, line)
        timing = Point(at) if at is not None else Undated('missing_timestamp')
        records.append(UsageRecord(entry, 'assistant', timing, model, tokens, MissingEstimate('not_recorded'), 'stop', None, ()))
        evidence.append(CodexEvidence(entry_id, source_kind, response_id, owner, turn, cumulative, last, state, reason, mirror))
    if session is None:
        return RejectedSource((Diagnostic('invalid_header', 1, None, None),))
    for index, item in enumerate(evidence):
        if item.source_kind == 'response' and records[index].model.model is None and item.turn_id in model_by_turn:
            selected = model_by_turn[item.turn_id]
            records[index] = replace(records[index], model=ModelIdentity(records[index].model.provider, selected.model if selected is not None else None))
    session = replace(session, title_excerpt=None if session.display_name else user_excerpt or excerpt)
    return CodexReadBatch(session, tuple(record.entry for record in records), tuple(records), tuple(diagnostics), complete, pending,
                          evidence=tuple(evidence), parent_thread_id=parent, forked_from_id=fork, root_session_id=root)
