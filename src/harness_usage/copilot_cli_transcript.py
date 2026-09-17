"""On-demand projection for the pinned Copilot CLI durable event subset."""
import json
from decimal import Decimal
from typing import Literal, cast

from .copilot_cli_reader import (
    _conversation_agent, _object, _text, _parse_lines, _qualify_events,
)
from .pi_reader import RejectedSource
from .transcript import (
    Message, Notice, ReasoningBlock, RecordedOutput, TextBlock, ToolBlock, Transcript,
    TranscriptEntry, TranscriptUnavailable, JsonlRecord, MAX_TRANSCRIPT_BYTES,
)


def _json_arguments(value: object) -> str | None:
    if value is None:
        return ''
    if isinstance(value, (str, int, Decimal, bool, list, dict)):
        try:
            rendered = value if isinstance(value, str) else json.dumps(
                value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                default=lambda item: float(item) if isinstance(item, Decimal) else None, allow_nan=False)
        except (TypeError, ValueError, RecursionError):
            return None
        return rendered if len(rendered) <= 64 * 1024 else None
    return None


def _tool_request(value: object, line: int, *, assistant: bool = False) -> ToolBlock | None:
    request = _object(value)
    if request is None:
        return None
    call_id = _text(request.get('toolCallId'))
    # Assistant tool requests use `name`; execution-start events use the
    # envelope's exact `toolName`.  Do not accept generic aliases here.
    name = _text(request.get('name') if assistant else request.get('toolName'))
    arguments = _json_arguments(request.get('arguments'))
    if call_id is None or name is None or arguments is None:
        return None
    return ToolBlock(call_id, name, arguments, source_line=line)


def _notice(line: int) -> Notice:
    return Notice(f'copilot-cli-unsupported-{line}', 'Unsupported Copilot CLI entry',
                  f'A recorded entry at line {line} could not be displayed.')


def _completion(data: dict[str, object], line: int) -> RecordedOutput | None:
    if type(data.get('success')) is not bool:
        return None
    result = _object(data.get('result'))
    if data['success'] is True and 'error' not in data and result is not None and isinstance(result.get('content'), str):
        return RecordedOutput((TextBlock(cast(str, result['content'])),), 'succeeded', line)
    error = _object(data.get('error'))
    if data['success'] is False and 'result' not in data and error is not None and isinstance(error.get('message'), str):
        return RecordedOutput((TextBlock(cast(str, error['message'])),), 'failed', line)
    return None


def parse_copilot_cli_transcript(payload: bytes, expected_id: str, scoped_session_id: str,
                                 *, agent_id: str | None = None,
                                 branch: str | None = None) -> Transcript:
    if branch is not None:
        raise TranscriptUnavailable('Copilot CLI transcripts expose one recorded view.', 'invalid_branch')
    if len(payload) > MAX_TRANSCRIPT_BYTES:
        raise TranscriptUnavailable('Transcript exceeds the 128 MiB read limit.', 'too_large')
    rows, diagnostics, _, pending = _parse_lines(payload)
    qualified = _qualify_events(rows, expected_id, diagnostics)
    if isinstance(qualified, RejectedSource):
        if not any(row.get('type') == 'session.start' for _, row in rows):
            raise TranscriptUnavailable('Usage unavailable for this Copilot CLI source.', 'unsupported')
        raise TranscriptUnavailable('The source no longer satisfies the Copilot CLI event contract.', 'changed')
    start_line, _, _, _, envelopes = qualified
    records = tuple(JsonlRecord(line, row) for line, row in rows)
    start_record = next(record for record in records if record.line == start_line)
    warnings = tuple(f'Line {item.line} could not be displayed.' for item in diagnostics)
    if pending:
        warnings += ('The final line is incomplete and was skipped.',)

    preferred_tools: dict[str, ToolBlock] = {}
    fallback_tools: dict[str, ToolBlock] = {}
    completion_candidates: dict[str, list[tuple[int, dict[str, object]]]] = {}
    visible_indices: set[int] = set()
    for index, record in enumerate(records):
        value = record.value
        envelope = envelopes.get(index)
        if envelope is None:
            continue
        _, parent_id, _, data = envelope
        valid_agent, recorded_agent = _conversation_agent(value)
        visible = recorded_agent == agent_id if agent_id is not None else recorded_agent in (None, 'main')
        if not valid_agent or not visible:
            continue
        kind = value.get('type')
        if kind in ('session.start', 'session.shutdown', 'session.usage_checkpoint',
                    'session.context_changed', 'assistant.usage'):
            continue
        visible_indices.add(index)
        if kind == 'assistant.message' and isinstance(data.get('toolRequests'), list):
            if _text(data.get('messageId')) is None or not isinstance(data.get('content'), str):
                continue
            for raw in cast(list[object], data['toolRequests']):
                tool = _tool_request(raw, record.line, assistant=True)
                if tool is not None and tool.call_id is not None:
                    preferred_tools.setdefault(tool.call_id, tool)
        elif kind == 'tool.execution_start':
            tool = _tool_request(data, record.line)
            if tool is not None and tool.call_id is not None:
                fallback_tools.setdefault(tool.call_id, tool)
        elif kind == 'tool.execution_complete':
            call_id = _text(data.get('toolCallId'))
            if call_id is not None:
                completion_candidates.setdefault(call_id, []).append((record.line, data))

    completions: dict[str, RecordedOutput] = {}
    for call_id, candidates in completion_candidates.items():
        origin = preferred_tools.get(call_id) or fallback_tools.get(call_id)
        if origin is None:
            continue
        candidate = next(((line, data) for line, data in candidates if line > origin.source_line), None)
        if candidate is None:
            continue
        line, data = candidate
        rendered = _completion(data, line)
        if rendered is not None:
            completions[call_id] = rendered

    entries: list[TranscriptEntry] = []
    emitted_tools: set[str] = set()
    for index, record in enumerate(records):
        if index not in visible_indices:
            continue
        value = record.value
        kind = value.get('type')
        event_id, _, _, data = envelopes[index]
        timestamp = _text(value.get('timestamp'))
        if kind in ('user.message', 'assistant.message'):
            content = data.get('content')
            if (not isinstance(content, str)
                    or kind == 'assistant.message' and _text(data.get('messageId')) is None):
                entries.append(_notice(record.line))
                continue
            blocks: list[TextBlock | ToolBlock] = [TextBlock(content)]
            requests = data.get('toolRequests')
            if kind == 'assistant.message' and isinstance(requests, list):
                for raw in cast(list[object], requests):
                    tool = _tool_request(raw, record.line, assistant=True)
                    if tool is not None and tool.call_id is not None and tool.call_id not in emitted_tools:
                        blocks.append(ToolBlock(tool.call_id, tool.name, tool.arguments,
                                                completions.get(tool.call_id), tool.source_line))
                        emitted_tools.add(tool.call_id)
            role: Literal['user', 'assistant'] = 'user' if kind == 'user.message' else 'assistant'
            entries.append(Message(event_id, role, tuple(blocks), timestamp=timestamp))
        elif kind == 'assistant.reasoning':
            content = data.get('content')
            if isinstance(content, str) and _text(data.get('reasoningId')) is not None:
                entries.append(Message(event_id, 'assistant', (ReasoningBlock(content),), timestamp=timestamp))
            else:
                entries.append(_notice(record.line))
        elif kind == 'tool.execution_start':
            tool = _tool_request(data, record.line)
            if tool is not None and tool.call_id is not None and tool.call_id not in preferred_tools \
                    and tool.call_id not in emitted_tools:
                entries.append(Message(event_id, 'assistant',
                                       (ToolBlock(tool.call_id, tool.name, tool.arguments,
                                                  completions.get(tool.call_id), tool.source_line),),
                                       timestamp=timestamp))
                emitted_tools.add(tool.call_id)
        elif kind == 'tool.execution_complete':
            if _completion(data, record.line) is None:
                entries.append(_notice(record.line))
            continue
        else:
            entries.append(_notice(record.line))
    if not entries:
        raise TranscriptUnavailable('Usage unavailable for this Copilot CLI source.', 'unsupported')
    native_id = expected_id if agent_id is None else f'{expected_id}:agent:{agent_id}'
    return Transcript(scoped_session_id, native_id, 'Copilot CLI session ' + expected_id[:8],
                      'copilot-cli', tuple(entries), started=_text(start_record.value.get('timestamp')),
                      warnings=warnings)
