"""Pinned VS Code core chat-v3 replay and accounting normalization."""
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from hashlib import sha256
import json
import os
from pathlib import Path
import re
from typing import Literal, Mapping, cast
from urllib.parse import unquote, urlsplit

from .domain import Known, MeasuredQuantity, MissingEstimate, ModelIdentity, Point, SessionId, TokenBreakdown, TokenEvidence, Undated, Unknown
from .pi_reader import (Diagnostic, EntryIdentity, ReadBatch, RejectedSource,
                        SessionMetadata, UsageRecord, _pending, _unique_object)

PROFILE = 'vscode-chat-v3/copilot-shape-2'
SUPPORTED_PROFILES = ('vscode-chat-v3/copilot-shape-1', PROFILE)
PathKey = str | int


@dataclass(frozen=True, slots=True)
class ChatSnapshot:
    value: dict[str, object]
    complete_bytes: int
    pending_tail: bool
    representation: Literal['flat', 'operation_log']
    provenance: Mapping[tuple[PathKey, ...], int]

    def line_for(self, path: tuple[PathKey, ...]) -> int:
        return self.provenance.get(path, 1)


@dataclass(frozen=True, slots=True)
class CopilotVscodeEvidence:
    profile: str
    session_native_id: str
    request_id: str | None
    response_id: str | None
    selected_model: str | None
    representation: Literal['flat', 'operation_log']
    request_ordinal: int
    evidence_kind: Literal['model_totals', 'turn_summary', 'session_control', 'unavailable']
    raw_input: int | None
    raw_output: int | None
    raw_cache_read: int | None
    raw_cache_write: int | None
    presence: str
    state: Literal['usable', 'unresolved']
    reason: str | None
    line: int
    output_lower_bound: bool = False
    retained_conflict: tuple[int, int] | None = None


@dataclass(frozen=True, slots=True)
class CopilotVscodeReadBatch(ReadBatch):
    evidence: tuple[CopilotVscodeEvidence, ...] = ()
    raw_session_id: str = ''
    source_scope: str = ''


def copilot_vscode_scope(locator: str) -> str | None:
    path = Path(locator)
    if not path.is_absolute() or path.suffix not in ('.json', '.jsonl'):
        return None
    parts = path.parts
    if len(parts) < 5 or parts[-2] != 'chatSessions' or parts[-4] != 'workspaceStorage' or not parts[-3]:
        return None
    root = os.path.normcase(os.path.realpath(os.path.abspath(str(Path(*parts[:-4])))))
    material = b'harness-usage/copilot-vscode-scope-v1\0' + os.fsencode(root)
    return sha256(material).hexdigest()


def _record_tree(lines: dict[tuple[PathKey, ...], int], value: object,
                 path: tuple[PathKey, ...], line: int) -> None:
    lines[path] = line
    children = value.items() if isinstance(value, dict) else enumerate(value) if isinstance(value, list) else ()
    for key, child in children:
        _record_tree(lines, child, (*path, key), line)


def _drop_tree(lines: dict[tuple[PathKey, ...], int], path: tuple[PathKey, ...]) -> None:
    for existing in tuple(lines):
        if existing[:len(path)] == path:
            del lines[existing]


def _container(value: object, path: list[object]) -> object:
    current = value
    for key in path:
        if type(key) is str and isinstance(current, dict) and key in current:
            current = current[key]
        elif type(key) is int and isinstance(current, list) and 0 <= key < len(current):
            current = current[key]
        else:
            raise ValueError('invalid_operation_path')
    return current


def _path(entry: dict[str, object], *, nonempty: bool) -> list[object]:
    value = entry.get('k')
    if not isinstance(value, list) or (nonempty and not value) or any(type(item) not in (str, int) for item in value):
        raise ValueError('invalid_operation_path')
    return value


def _apply(entry: dict[str, object], state: object | None,
           lines: dict[tuple[PathKey, ...], int], line: int) -> object:
    kind = entry.get('kind')
    if type(kind) is not int:
        raise ValueError('unsupported_chat_operation')
    if kind == 0:
        if state is not None:
            raise ValueError('duplicate_initial_chat_state')
        if 'v' not in entry:
            raise ValueError('missing_initial_chat_state')
        state = deepcopy(entry['v'])
        _record_tree(lines, state, (), line)
        return state
    if state is None:
        raise ValueError('missing_initial_chat_state')
    path = _path(entry, nonempty=kind in (1, 3))
    if kind == 2:
        target = _container(state, path)
        if not isinstance(target, list):
            raise ValueError('invalid_push_target')
        if 'i' in entry:
            index = entry['i']
            if type(index) is not int or not 0 <= index <= len(target):
                raise ValueError('invalid_push_index')
            del target[index:]
            prefix = cast(tuple[PathKey, ...], tuple(path))
            for existing in tuple(lines):
                child = existing[len(prefix)] if len(existing) > len(prefix) else None
                if existing[:len(prefix)] == prefix and type(child) is int and child >= index:
                    del lines[existing]
        additions = entry.get('v', [])
        if not isinstance(additions, list):
            raise ValueError('invalid_push_value')
        start = len(target)
        target.extend(deepcopy(additions))
        for offset, child in enumerate(additions):
            _record_tree(lines, child, (*cast(tuple[PathKey, ...], tuple(path)), start + offset), line)
        lines[cast(tuple[PathKey, ...], tuple(path))] = line
        return state
    if kind not in (1, 3):
        raise ValueError('unsupported_chat_operation')
    parent = _container(state, path[:-1])
    key = path[-1]
    if type(key) is str and isinstance(parent, dict):
        if kind == 1:
            if 'v' not in entry:
                raise ValueError('missing_set_value')
            _drop_tree(lines, cast(tuple[PathKey, ...], tuple(path)))
            parent[key] = deepcopy(entry['v'])
            _record_tree(lines, parent[key], cast(tuple[PathKey, ...], tuple(path)), line)
        elif key in parent:
            del parent[key]
            _drop_tree(lines, cast(tuple[PathKey, ...], tuple(path)))
        else:
            raise ValueError('invalid_operation_path')
        return state
    if kind == 3:
        raise ValueError('invalid_delete_path')
    raise ValueError('invalid_operation_path')


def _loads(data: bytes) -> object:
    return json.loads(data, parse_float=Decimal, parse_int=int, object_pairs_hook=_unique_object)


def replay_chat_v3(data: bytes, *, representation: Literal['flat', 'operation_log']) -> ChatSnapshot:
    provenance: dict[tuple[PathKey, ...], int] = {}
    if representation == 'flat':
        try:
            value = _loads(data)
        except (ValueError, UnicodeError, RecursionError) as error:
            raise ValueError('malformed_chat_state') from error
        if not isinstance(value, dict):
            raise ValueError('invalid_chat_state')
        _record_tree(provenance, value, (), 1)
        return ChatSnapshot(cast(dict[str, object], value), len(data), False, representation, provenance)
    state: object | None = None
    offset = complete_bytes = 0
    pending_tail = False
    lines = data.splitlines(keepends=True)
    for number, raw in enumerate(lines, 1):
        body = raw.rstrip(b'\r\n')
        if not body:
            raise ValueError('malformed_chat_operation')
        try:
            entry = _loads(body)
        except json.JSONDecodeError as error:
            if number == len(lines) and not raw.endswith((b'\n', b'\r')) and _pending(error, body.decode(errors='replace')):
                pending_tail = True
                break
            raise ValueError('malformed_chat_operation') from error
        except (ValueError, UnicodeError, RecursionError) as error:
            raise ValueError('malformed_chat_operation') from error
        if not isinstance(entry, dict):
            raise ValueError('invalid_chat_operation')
        state = _apply(cast(dict[str, object], entry), state, provenance, number)
        offset += len(raw)
        complete_bytes = offset
    if state is None or not isinstance(state, dict):
        raise ValueError('missing_initial_chat_state')
    return ChatSnapshot(cast(dict[str, object], state), complete_bytes, pending_tail, representation, provenance)


def _text(value: object) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _count(value: object, *, absent: bool = False) -> Known | Unknown | None:
    if absent:
        return Unknown('not_reported')
    return Known(value) if type(value) is int and 0 <= value < 2**63 else None


def _credit(value: object) -> Decimal | None:
    number = Decimal(value) if type(value) is int else value
    return number if isinstance(number, Decimal) and number.is_finite() and 0 <= number < 2**63 else None


def _time(value: object) -> datetime | None:
    if type(value) is not int or not 0 <= value < 2**63:
        return None
    try:
        return datetime(1970, 1, 1, tzinfo=UTC) + timedelta(milliseconds=value)
    except OverflowError:
        return None


def _file_uri(value: object) -> str | None:
    if (not isinstance(value, str) or re.search(r'[\x00-\x1f\x7f]', value)
            or re.search(r'%(?![0-9A-Fa-f]{2})', value)):
        return None
    try:
        parsed = urlsplit(value)
    except ValueError:
        return None
    if (parsed.scheme.casefold() != 'file' or parsed.netloc not in ('', 'localhost')
            or parsed.query or parsed.fragment):
        return None
    try:
        path = unquote(parsed.path, errors='strict')
    except (UnicodeError, ValueError):
        return None
    if (re.search(r'[\x00-\x1f\x7f]', path) or path.startswith('//')
            or re.match(r'^/[A-Za-z]:', path) or not Path(path).is_absolute()):
        return None
    return path


def _workspace(context: tuple[tuple[str, bytes], ...]) -> str | None:
    raw = next((value for name, value in context if name == 'workspace.json'), None)
    if raw is None:
        return None
    try:
        value = _loads(raw)
    except (ValueError, UnicodeError, RecursionError):
        return None
    return _file_uri(value.get('folder')) if isinstance(value, dict) else None


def _tokens(input_value: Known | Unknown, output: Known | Unknown,
            cache_read: Known | Unknown, *, reason: str) -> TokenEvidence:
    return TokenEvidence(TokenBreakdown(input_value, output, cache_read, Unknown('not_reported')),
                         Unknown(reason), Unknown('not_reported'), Unknown('not_reported'))


def read_copilot_vscode(data: bytes, *, locator: str,
                        context: tuple[tuple[str, bytes], ...] = (),
                        profile: str = PROFILE) -> CopilotVscodeReadBatch | RejectedSource:
    scope = copilot_vscode_scope(locator)
    if profile != PROFILE:
        return RejectedSource((Diagnostic('unsupported_profile', None, None, None),))
    if scope is None:
        return RejectedSource((Diagnostic('missing_copilot_vscode_scope', None, None, None),))
    representation: Literal['flat', 'operation_log'] = 'operation_log' if Path(locator).suffix == '.jsonl' else 'flat'
    try:
        snapshot = replay_chat_v3(data, representation=representation)
    except ValueError as error:
        return RejectedSource((Diagnostic(str(error), None, None, None),))
    state = snapshot.value
    raw_session_id = _text(state.get('sessionId'))
    if (type(state.get('version')) is not int or state.get('version') != 3
            or raw_session_id is None or not isinstance(state.get('requests'), list)):
        return RejectedSource((Diagnostic('invalid_vscode_chat_state', None, None, None),))
    native_id = f'{scope}:{raw_session_id}'
    session_id = SessionId('copilot-vscode:' + native_id)
    cwd = _file_uri(state.get('workingDirectory')) or _workspace(context)
    title = _text(state.get('customTitle'))
    if title is not None:
        title = ' '.join(title.split())[:160] or None
    creation = _time(state.get('creationDate'))
    diagnostics: list[Diagnostic] = []
    usage: list[UsageRecord] = []
    evidence: list[CopilotVscodeEvidence] = []
    seen_records: set[tuple[object, ...]] = set()
    times: list[datetime] = [creation] if creation is not None else []

    def add(request: dict[str, object], ordinal: int, request_id: str | None,
            response_id: str | None, selected: str | None, evidence_kind: Literal['model_totals', 'turn_summary', 'session_control', 'unavailable'],
            entry_suffix: str, line: int, tokens: TokenEvidence, model: ModelIdentity,
            quantity: MeasuredQuantity | None, raw: tuple[int | None, int | None, int | None, int | None],
            present: tuple[bool, bool, bool, bool], reason: str | None,
            output_lower_bound: bool = False, retained_conflict: tuple[int, int] | None = None) -> None:
        state_value: Literal['usable', 'unresolved'] = 'usable' if reason is None else 'unresolved'
        source_key = sha256(locator.encode()).hexdigest()[:16]
        request_key: object = ('recorded', request_id) if request_id is not None else ('missing', source_key, ordinal)
        compatibility = (
            tuple(zip(present, raw, strict=True)), state_value, reason,
            quantity.state if quantity is not None else None,
            str(quantity.amount) if quantity is not None and quantity.amount is not None else None,
            output_lower_bound,
            retained_conflict,
        )
        identity = (request_key, ('response', response_id), ('evidence', entry_suffix))
        native_entry = 'vscode:' + sha256(json.dumps(identity, separators=(',', ':')).encode()).hexdigest()
        entry = EntryIdentity(native_entry, None, line)
        at = _time(request.get('responseTimestamp')) or _time(request.get('timestamp')) or creation
        if at is not None:
            times.append(at)
        record = UsageRecord(entry, 'usage_checkpoint' if evidence_kind == 'session_control' else 'request_summary',
            Point(at) if at is not None else Undated('timestamp_unavailable'), model, tokens,
            MissingEstimate('partial_tokens' if evidence_kind == 'model_totals' else 'selected_model_unpriced'),
            None, None, (('usage.outputFinality', 'last_successful_call'),) if output_lower_bound else (),
            (quantity,) if quantity is not None else ())
        normalized = (native_entry, record.kind, record.time, record.model, record.tokens, record.money,
                      record.stop_reason, record.tool_call_id, record.safe_facts, record.quantities,
                      compatibility)
        if normalized in seen_records:
            return
        seen_records.add(normalized)
        usage.append(record)
        evidence.append(CopilotVscodeEvidence(PROFILE, raw_session_id, request_id, response_id, selected,
            representation, ordinal, evidence_kind, *raw,
            ''.join('1' if value else '0' for value in present), state_value, reason, line,
            output_lower_bound, retained_conflict))

    requests = cast(list[object], state['requests'])
    for ordinal, item in enumerate(requests):
        if not isinstance(item, dict):
            diagnostics.append(Diagnostic('invalid_vscode_request', snapshot.line_for(('requests', ordinal)), None, None))
            continue
        request = cast(dict[str, object], item)
        agent = request.get('agent')
        extension = agent.get('extensionId') if isinstance(agent, dict) else None
        extension_id = extension.get('value') if isinstance(extension, dict) else None
        if not isinstance(extension_id, str) or extension_id.casefold() != 'github.copilot-chat':
            continue
        request_id = _text(request.get('requestId'))
        response_id = _text(request.get('responseId'))
        selected = _text(request.get('modelId'))
        base_reason = None if request_id is not None else 'missing_request_identity'
        if 'responseId' in request and response_id is None:
            base_reason = 'invalid_response_identity'
        models = request.get('modelTotals')
        model_rows = cast(list[object], models) if isinstance(models, list) else []
        duplicate_models = {name for name in (_text(row.get('model')) if isinstance(row, dict) else None for row in model_rows)
                            if name is not None and sum(1 for row in model_rows if isinstance(row, dict) and _text(row.get('model')) == name) > 1}
        numeric_invalid = False
        for field in ('promptTokens', 'completionTokens'):
            if field in request and _count(request[field]) is None:
                numeric_invalid = True
        for field in ('copilotCredits', 'sessionCopilotCredits'):
            if field in request and _credit(request[field]) is None:
                numeric_invalid = True
        if models is not None and not isinstance(models, list):
            numeric_invalid = True
        request_line = snapshot.line_for(('requests', ordinal))
        for model_ordinal, raw_model in enumerate(model_rows):
            model_path = ('requests', ordinal, 'modelTotals', model_ordinal)
            line = snapshot.line_for(model_path)
            if not isinstance(raw_model, dict):
                numeric_invalid = True
                continue
            model_row = cast(dict[str, object], raw_model)
            line = max((line, *(snapshot.line_for((*model_path, name)) for name in
                                ('model', 'inputTokens', 'cachedTokens', 'outputTokens')
                                if name in model_row),
                        *(snapshot.line_for(('requests', ordinal, name)) for name in
                          ('requestId', 'responseId') if name in request)))
            model = _text(model_row.get('model'))
            values = tuple(_count(model_row.get(name), absent=name not in model_row)
                           for name in ('inputTokens', 'cachedTokens', 'outputTokens'))
            reason = ('invalid_numeric_value' if any(value is None for value in values) or model is None
                      else 'duplicate_actual_model' if model in duplicate_models else base_reason)
            if reason == 'invalid_numeric_value':
                numeric_invalid = True
            input_raw = values[0].value if isinstance(values[0], Known) else None
            cache_raw = values[1].value if isinstance(values[1], Known) else None
            output_raw = values[2].value if isinstance(values[2], Known) else None
            cache_value = values[1] if values[1] is not None else Unknown('invalid_numeric_value')
            output_value = values[2] if values[2] is not None else Unknown('invalid_numeric_value')
            add(request, ordinal, request_id, response_id, selected, 'model_totals', f'model:{model or model_ordinal}', line,
                _tokens(Unknown('cache_inclusion_unknown'), output_value, cache_value, reason='cache_inclusion_unknown'),
                ModelIdentity('github-copilot', model), None, (input_raw, output_raw, cache_raw, None),
                ('inputTokens' in model_row, 'outputTokens' in model_row,
                 'cachedTokens' in model_row, False),
                reason)
        prompt = _count(request.get('promptTokens'), absent='promptTokens' not in request)
        completion = _count(request.get('completionTokens'), absent='completionTokens' not in request)
        result = request.get('result')
        metadata = result.get('metadata') if isinstance(result, dict) else None
        last_prompt = _count(metadata.get('promptTokens')) if isinstance(metadata, dict) else None
        last_output = _count(metadata.get('outputTokens')) if isinstance(metadata, dict) else None
        retained_reason = None
        retained_conflict = None
        retained_path, retained_fields = 'metadata', ('promptTokens', 'outputTokens')
        core_present = any(name in request for name in ('modelTotals', 'promptTokens', 'completionTokens'))
        if not core_present and isinstance(result, dict) and 'usage' in result:
            legacy = result['usage']
            legacy_prompt = _count(legacy.get('promptTokens')) if isinstance(legacy, dict) else None
            legacy_output = _count(legacy.get('completionTokens')) if isinstance(legacy, dict) else None
            if not isinstance(legacy_prompt, Known) or not isinstance(legacy_output, Known):
                retained_reason = 'invalid_retained_usage'
            elif (isinstance(last_prompt, Known) and isinstance(last_output, Known)
                  and (last_prompt, last_output) != (legacy_prompt, legacy_output)):
                retained_reason = 'conflicting_retained_usage'
                retained_conflict = (last_prompt.value, last_output.value)
            last_prompt, last_output = legacy_prompt, legacy_output
            retained_path, retained_fields = 'usage', ('promptTokens', 'completionTokens')
        last_call = (isinstance(last_prompt, Known) and isinstance(last_output, Known)
                     and not core_present)
        summary_kind: Literal['turn_summary', 'unavailable'] = ('turn_summary' if 'promptTokens' in request or 'completionTokens' in request or 'copilotCredits' in request or model_rows else 'unavailable')
        summary_reason = retained_reason or ('invalid_numeric_value' if numeric_invalid else
                                             'duplicate_actual_model' if duplicate_models else base_reason)
        credit = _credit(request.get('copilotCredits')) if 'copilotCredits' in request else None
        quantity = (MeasuredQuantity('ai_credits', 'known', credit, None, True, 'copilot-vscode-turn')
                    if credit is not None else MeasuredQuantity('ai_credits', 'unknown', None,
                        'invalid_numeric_value' if 'copilotCredits' in request else 'not_reported', False, None))
        summary_line = max(request_line, snapshot.line_for(('requests', ordinal, 'promptTokens')),
                           snapshot.line_for(('requests', ordinal, 'completionTokens')),
                           snapshot.line_for(('requests', ordinal, 'copilotCredits')))
        if last_call:
            assert isinstance(last_prompt, Known) and isinstance(last_output, Known)
            prompt, completion = last_prompt, last_output
            summary_kind = 'turn_summary'
        if last_call or retained_reason:
            summary_line = max(summary_line, *(snapshot.line_for(('requests', ordinal, 'result', retained_path, name))
                                               for name in retained_fields),
                               *(snapshot.line_for(('requests', ordinal, name))
                                 for name in ('requestId', 'responseId') if name in request))
            if retained_conflict is not None:
                summary_line = max(summary_line, *(snapshot.line_for(('requests', ordinal, 'result', 'metadata', name))
                                                   for name in ('promptTokens', 'outputTokens')))
            if retained_reason:
                diagnostics.append(Diagnostic(retained_reason, summary_line, request_id, None))
        if model_rows:
            summary_tokens = _tokens(Unknown('covered_by_model_totals'), Unknown('covered_by_model_totals'),
                                     Unknown('covered_by_model_totals'), reason='covered_by_model_totals')
        else:
            summary_tokens = _tokens(Unknown('cache_inclusion_unknown') if last_call else
                                     prompt if prompt is not None else Unknown('invalid_count'),
                                     completion if completion is not None else Unknown('invalid_count'),
                                     Unknown('not_reported'), reason='partial_total')
            if summary_kind == 'turn_summary' and not last_call:
                diagnostics.append(Diagnostic('lossy_turn_summary', summary_line, request_id, None))
        add(request, ordinal, request_id, response_id, selected, summary_kind,
            'summary' if summary_kind == 'turn_summary' else 'unavailable', summary_line, summary_tokens,
            ModelIdentity(None if last_call or retained_reason else 'github-copilot', None), quantity,
            (prompt.value if isinstance(prompt, Known) else None,
             completion.value if isinstance(completion, Known) else None,
             None, None),
            ('promptTokens' in request or last_call, 'completionTokens' in request or last_call, False, False),
            summary_reason, output_lower_bound=last_call, retained_conflict=retained_conflict)
        if 'sessionCopilotCredits' in request:
            control = _credit(request['sessionCopilotCredits'])
            line = snapshot.line_for(('requests', ordinal, 'sessionCopilotCredits'))
            control_quantity = (MeasuredQuantity('ai_credits', 'known', control, None, False, 'copilot-vscode-session')
                                if control is not None else MeasuredQuantity('ai_credits', 'unknown', None, 'invalid_numeric_value', False, None))
            add(request, ordinal, request_id, response_id, selected, 'session_control', 'session-control', line,
                _tokens(Unknown('not_applicable'), Unknown('not_applicable'), Unknown('not_applicable'), reason='not_applicable'),
                ModelIdentity('github-copilot', None), control_quantity, (None, None, None, None),
                (False, False, False, False),
                'invalid_numeric_value' if control is None else base_reason)
    session = SessionMetadata(session_id, native_id, creation, cwd, None, title,
                              last_observed=max(times) if times else None)
    return CopilotVscodeReadBatch(session, (), tuple(usage), tuple(diagnostics),
                                  snapshot.complete_bytes, snapshot.pending_tail, (), tuple(evidence),
                                  raw_session_id, scope)
