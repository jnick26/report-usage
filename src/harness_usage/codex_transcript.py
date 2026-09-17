"""Read supported Codex legacy rollout JSONL without reconstructing history."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import Literal, cast

from .transcript import (
    AttachmentBlock,
    ConflictingOutput,
    ImageBlock,
    Message,
    Notice,
    ReasoningBlock,
    RecordedOutput,
    TextBlock,
    ToolBlock,
    Transcript,
    TranscriptUnavailable,
    read_jsonl,
    text_and_media_blocks,
)

Role = Literal["user", "assistant", "system", "developer", "tool", "context"]
MirrorKind = Literal["raw", "event"]
PlainBlock = TextBlock | AttachmentBlock | ImageBlock
MIRROR_LINE_WINDOW = 4


@dataclass(slots=True)
class _State:
    entry: Message | Notice
    suppressed: bool = False


@dataclass(frozen=True, slots=True)
class _Mirror:
    state_index: int
    line: int
    scope: str
    kind: MirrorKind
    role: Role
    phase: str | None
    signature: tuple[tuple[str, str, str], ...]
    reasoning_texts: tuple[str, ...]
    native_id: str | None


@dataclass(frozen=True, slots=True)
class _Call:
    state_index: int
    line: int
    scope: str
    call_id: str | None


@dataclass(frozen=True, slots=True)
class _Result:
    state_index: int
    line: int
    scope: str
    call_id: str | None
    output: RecordedOutput


def _object(value: object) -> dict[str, object]:
    return cast(dict[str, object], value) if isinstance(value, dict) else {}


def _string(value: object) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _discriminant(value: object) -> str:
    """Return a bounded marker for an untrusted schema discriminator."""
    if isinstance(value, str) and value.strip() and len(value) <= 80:
        return value
    return "<unrecognized>"


def _safe_role(value: object, line: int, warnings: list[str]) -> Role:
    if value in ("user", "assistant", "system", "developer", "tool", "context"):
        return value
    warnings.append(f"Line {line} has an unknown role; shown as context.")
    return "context"


def _id(line: int) -> str:
    return f"codex-{line}-entry"


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _reference(kind: str, value: object, line: int) -> AttachmentBlock:
    if isinstance(value, str) and value:
        return AttachmentBlock(f"{kind} reference: {value}")
    return AttachmentBlock(f"{kind} content unavailable at line {line}")


def _image(value: object, line: int) -> tuple[PlainBlock, tuple[str, ...]]:
    if not isinstance(value, str) or not value.startswith("data:"):
        return _reference("Image", value, line), ()
    header, comma, data = value.partition(",")
    mime = header[5:].removesuffix(";base64")
    if comma and header.endswith(";base64"):
        blocks, warnings = text_and_media_blocks(
            [{"type": "image", "mimeType": mime, "data": data}], source_line=line
        )
        return blocks[0], warnings
    return _reference("Image", value, line), (
        f"Line {line} contains an unsupported inline image reference.",
    )


def _content_blocks(content: object, line: int) -> tuple[tuple[PlainBlock, ...], tuple[str, ...]]:
    if isinstance(content, str):
        return (TextBlock(content),), ()
    if not isinstance(content, list):
        return text_and_media_blocks(content, source_line=line)
    blocks: list[PlainBlock] = []
    warnings: list[str] = []
    for raw in content:
        item = _object(raw)
        kind = item.get("type")
        if kind in ("input_text", "output_text", "text", "Text") and isinstance(item.get("text"), str):
            blocks.append(TextBlock(cast(str, item["text"])))
        elif kind in ("input_image", "image_url"):
            block, extra = _image(item.get("image_url", item.get("url")), line)
            blocks.append(block)
            warnings.extend(extra)
        elif kind == "image" and "data" in item:
            shared, extra = text_and_media_blocks([item], source_line=line)
            blocks.extend(shared)
            warnings.extend(extra)
        elif kind in ("input_audio", "audio", "local_audio", "local_image"):
            value = item.get("audio_url", item.get("image_url", item.get("url", item.get("path"))))
            blocks.append(_reference("Audio" if "audio" in kind else "Local image", value, line))
        elif kind == "encrypted_content":
            blocks.append(AttachmentBlock(f"Encrypted content unavailable at line {line}"))
        else:
            label = _discriminant(kind)
            blocks.append(AttachmentBlock(f"Unsupported {label} content at line {line}"))
            warnings.append(f"Line {line} contains unsupported {label} content.")
    return tuple(blocks), tuple(warnings)


def _event_blocks(payload: dict[str, object], line: int) -> tuple[tuple[PlainBlock, ...], tuple[str, ...]]:
    blocks: list[PlainBlock] = []
    warnings: list[str] = []
    message = payload.get("message")
    if isinstance(message, str):
        blocks.append(TextBlock(message))
    elif message is not None:
        warnings.append(f"Line {line} has a non-text legacy message.")
    for field, label in (
        ("images", "Image"),
        ("local_images", "Local image"),
        ("audio", "Audio"),
        ("local_audio", "Local audio"),
    ):
        value = payload.get(field)
        values = value if isinstance(value, list) else [value] if value is not None else []
        for reference in values:
            if field == "images":
                block, extra = _image(reference, line)
                blocks.append(block)
                warnings.extend(extra)
            else:
                blocks.append(_reference(label, reference, line))
    if not blocks:
        blocks.append(AttachmentBlock(f"Legacy message content unavailable at line {line}"))
        warnings.append(f"Line {line} has no displayable legacy message content.")
    return tuple(blocks), tuple(warnings)


def _signature(blocks: tuple[PlainBlock, ...]) -> tuple[tuple[str, str, str], ...]:
    result: list[tuple[str, str, str]] = []
    for block in blocks:
        if isinstance(block, TextBlock):
            result.append(("text", block.text, ""))
        elif isinstance(block, ImageBlock):
            result.append(("image", block.mime, block.data))
        else:
            result.append(("attachment", block.label, ""))
    return tuple(result)


def _mirror_matches(event: _Mirror, raw: _Mirror) -> bool:
    if event.scope != raw.scope or event.role != raw.role:
        return False
    if event.phase is not None and raw.phase is not None and event.phase != raw.phase:
        return False
    if event.reasoning_texts:
        return len(event.reasoning_texts) == 1 and event.reasoning_texts[0] in raw.reasoning_texts
    return event.signature == raw.signature


def _reconcile_mirrors(states: list[_State], mirrors: list[_Mirror], warnings: list[str]) -> None:
    nearby: dict[tuple[str, Role, int], list[_Mirror]] = {}
    identified: dict[tuple[str, Role, str], list[_Mirror]] = {}
    for candidate in (item for item in mirrors if item.kind == "raw"):
        nearby.setdefault((candidate.scope, candidate.role, candidate.line), []).append(candidate)
        if candidate.native_id is not None:
            identified.setdefault((candidate.scope, candidate.role, candidate.native_id), []).append(candidate)
    used_messages: set[int] = set()
    used_reasoning: set[tuple[int, str]] = set()
    for event in (candidate for candidate in mirrors if candidate.kind == "event"):
        possible: dict[int, _Mirror] = {}
        for line in range(event.line - MIRROR_LINE_WINDOW, event.line + MIRROR_LINE_WINDOW + 1):
            for candidate in nearby.get((event.scope, event.role, line), ()):
                possible[candidate.state_index] = candidate
        if event.native_id is not None:
            for candidate in identified.get((event.scope, event.role, event.native_id), ()):
                possible[candidate.state_index] = candidate
        reasoning = event.reasoning_texts[0] if len(event.reasoning_texts) == 1 else None
        candidates = []
        for candidate in possible.values():
            available = (
                (candidate.state_index, reasoning) not in used_reasoning
                if reasoning is not None
                else candidate.state_index not in used_messages
            )
            if available and _mirror_matches(event, candidate):
                candidates.append(candidate)
        if len(candidates) == 1:
            states[event.state_index].suppressed = True
            if reasoning is not None:
                used_reasoning.add((candidates[0].state_index, reasoning))
            else:
                used_messages.add(candidates[0].state_index)
        elif len(candidates) > 1:
            warnings.append(
                f"Line {event.line} may mirror multiple raw records; all ambiguous records were retained."
            )


def _set_tool_output(state: _State, output: RecordedOutput | ConflictingOutput) -> None:
    message = cast(Message, state.entry)
    state.entry = replace(
        message,
        blocks=tuple(replace(block, output=output) if isinstance(block, ToolBlock) else block for block in message.blocks),
    )


def _pair_tools(
    states: list[_State], calls: list[_Call], results: list[_Result], warnings: list[str]
) -> None:
    call_groups: dict[tuple[str, str], list[_Call]] = {}
    result_groups: dict[tuple[str, str], list[_Result]] = {}
    keys: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for call in calls:
        if call.call_id is None:
            continue
        key = (call.scope, call.call_id)
        call_groups.setdefault(key, []).append(call)
        if key not in seen:
            keys.append(key)
            seen.add(key)
    for result in results:
        if result.call_id is None:
            continue
        key = (result.scope, result.call_id)
        result_groups.setdefault(key, []).append(result)
        if key not in seen:
            keys.append(key)
            seen.add(key)
    for scope, call_id in keys:
        matched_calls = call_groups.get((scope, call_id), [])
        matched_results = result_groups.get((scope, call_id), [])
        if len(matched_calls) == len(matched_results) == 1:
            _set_tool_output(states[matched_calls[0].state_index], matched_results[0].output)
            states[matched_results[0].state_index].suppressed = True
            continue
        if len(matched_calls) > 1 or len(matched_results) > 1:
            reason = f"Duplicate tool identity {call_id!r} in one turn"
            for call in matched_calls:
                _set_tool_output(states[call.state_index], ConflictingOutput(reason))
            warnings.append(reason + ".")
        elif matched_calls:
            warnings.append(f"No result was recorded for tool call {call_id!r} at line {matched_calls[0].line}.")
        else:
            warnings.append(f"Tool result {call_id!r} at line {matched_results[0].line} has no matching call.")
    for call in (item for item in calls if item.call_id is None):
        warnings.append(f"Tool call at line {call.line} has no call ID and cannot be paired.")
    for result in (item for item in results if item.call_id is None):
        warnings.append(f"Tool result at line {result.line} has no call ID and cannot be paired.")


def _tool_name(payload: dict[str, object], fallback: str) -> str:
    name = _string(payload.get("name")) or fallback
    namespace = _string(payload.get("namespace"))
    return f"{namespace}.{name}" if namespace else name


def _tool_call(payload: dict[str, object], line: int) -> ToolBlock:
    kind = cast(str, payload.get("type"))
    if kind == "function_call":
        arguments = payload.get("arguments")
    elif kind == "custom_tool_call":
        arguments = payload.get("input")
    else:
        arguments = payload.get("action", {key: value for key, value in payload.items() if key not in {"type", "id", "call_id", "status"}})
    if not isinstance(arguments, str):
        arguments = _json(arguments)
    return ToolBlock(
        _string(payload.get("call_id")),
        _tool_name(payload, kind),
        arguments,
        source_line=line,
    )


def _status(value: object) -> Literal["recorded", "succeeded", "failed"]:
    if value in ("succeeded", "success", "completed"):
        return "succeeded"
    if value in ("failed", "error"):
        return "failed"
    return "recorded"


def _result(payload: dict[str, object], line: int) -> tuple[RecordedOutput, tuple[str, ...]]:
    blocks, warnings = _content_blocks(payload.get("output"), line)
    return RecordedOutput(blocks, _status(payload.get("status")), line), warnings


def _reasoning_blocks(payload: dict[str, object], line: int) -> tuple[ReasoningBlock, ...]:
    blocks: list[ReasoningBlock] = []
    for label, field in (("Reasoning summary", "summary"), ("Recorded reasoning", "content")):
        value = payload.get(field)
        if not isinstance(value, list):
            continue
        for raw in value:
            item = _object(raw)
            if isinstance(item.get("text"), str):
                blocks.append(ReasoningBlock(cast(str, item["text"]), label))
    if not blocks and payload.get("encrypted_content") is not None:
        blocks.append(ReasoningBlock("Opaque reasoning was recorded but is unavailable."))
    if not blocks:
        blocks.append(ReasoningBlock(f"Reasoning content unavailable at line {line}"))
    return tuple(blocks)


def parse_codex_transcript(
    payload: bytes, session_id: str, branch: str | None = None
) -> Transcript:
    """Project one self-contained legacy rollout; reconstruction remains unsupported."""
    if branch is not None:
        raise TranscriptUnavailable("Codex legacy transcripts do not record selectable branches.", "invalid_branch")
    if not isinstance(session_id, str) or not session_id.strip():
        raise TranscriptUnavailable("Session identity is missing.", "invalid_source")
    records, initial_warnings = read_jsonl(payload)
    if not records or records[0].line != 1 or records[0].value.get("type") != "session_meta":
        raise TranscriptUnavailable("Codex transcript has no valid first-line session metadata.", "invalid_source")
    header = records[0].value
    metadata = _object(header.get("payload"))
    native_id = _string(metadata.get("id"))
    if native_id is None:
        raise TranscriptUnavailable("Codex session metadata has no native identity.", "invalid_source")
    history_mode = metadata.get("history_mode")
    if metadata.get("history_base") is not None:
        raise TranscriptUnavailable("External Codex history prefix reconstruction is unsupported.", "unsupported")
    if history_mode == "paginated":
        from .codex_paginated import project_codex_paginated

        return project_codex_paginated(records, initial_warnings, session_id, metadata, header)
    if history_mode not in (None, "legacy"):
        raise TranscriptUnavailable("Codex history mode is unsupported.", "unsupported")

    warnings = list(initial_warnings)
    states: list[_State] = []
    mirrors: list[_Mirror] = []
    calls: list[_Call] = []
    results: list[_Result] = []
    scope = "file"
    model = _string(metadata.get("model"))
    first_user_text: str | None = None
    history_start = metadata.get("subagent_history_start_ordinal")
    inherited_cutoff = history_start if type(history_start) is int and history_start >= 0 else None

    def inherited(raw: dict[str, object]) -> bool:
        ordinal = raw.get("ordinal")
        record_metadata = _object(raw.get("metadata"))
        return record_metadata.get("inherited_user_message") is True or (
            inherited_cutoff is not None and type(ordinal) is int and ordinal < inherited_cutoff
        )

    def add_message(
        line: int,
        role: Role,
        blocks: tuple[TextBlock | ReasoningBlock | AttachmentBlock | ImageBlock | ToolBlock, ...],
        *,
        phase: str | None = None,
        timestamp: str | None = None,
    ) -> int:
        states.append(_State(Message(_id(line), role, blocks, model, phase, timestamp)))
        return len(states) - 1

    for record in records[1:]:
        raw = record.value
        line = record.line
        kind = raw.get("type")
        item = _object(raw.get("payload"))
        item_kind = item.get("type")
        timestamp = _string(raw.get("timestamp"))
        if kind in ("thread_rolled_back", "rollback") or item_kind in ("thread_rolled_back", "rollback"):
            raise TranscriptUnavailable("Codex rollback requires unsupported current-history reconstruction.", "unsupported")
        if kind == "session_meta":
            repeated_id = _string(_object(raw.get("payload")).get("id"))
            if repeated_id != native_id:
                raise TranscriptUnavailable(
                    "Codex transcript contains conflicting session metadata.", "invalid_source"
                )
            continue
        if kind == "turn_context":
            turn_id = _string(item.get("turn_id"))
            if turn_id:
                scope = "turn:" + turn_id
            model = _string(item.get("model")) or model
            continue
        if kind in ("token_usage_record", "token_count", "world_state") or kind == "event_msg" and item_kind in (
            "token_count",
            "thread_settings_applied",
            "task_started",
            "task_complete",
            "inter_agent_communication_metadata",
        ):
            continue

        if kind == "response_item" and item_kind == "message":
            role = _safe_role(item.get("role"), line, warnings)
            blocks, extra = _content_blocks(item.get("content"), line)
            warnings.extend(extra)
            phase = _string(item.get("phase"))
            if inherited(raw):
                role = "context"
                phase = "inherited"
            index = add_message(line, role, blocks, phase=phase, timestamp=timestamp)
            mirrors.append(
                _Mirror(index, line, scope, "raw", role, phase, _signature(blocks), (), _string(item.get("id")))
            )
            if role == "user" and first_user_text is None:
                first_user_text = next((block.text for block in blocks if isinstance(block, TextBlock)), None)
            continue

        if kind == "event_msg" and item_kind in ("user_message", "agent_message"):
            role = "user" if item_kind == "user_message" else "assistant"
            blocks, extra = _event_blocks(item, line)
            warnings.extend(extra)
            phase = _string(item.get("phase"))
            if inherited(raw):
                role = "context"
                phase = "inherited"
            index = add_message(line, role, blocks, phase=phase, timestamp=timestamp)
            mirrors.append(
                _Mirror(
                    index,
                    line,
                    scope,
                    "event",
                    role,
                    phase,
                    _signature(blocks),
                    (),
                    _string(item.get("id")) or _string(item.get("client_id")),
                )
            )
            if role == "user" and first_user_text is None:
                first_user_text = next((block.text for block in blocks if isinstance(block, TextBlock)), None)
            continue

        if kind == "response_item" and item_kind == "reasoning":
            reasoning_blocks = _reasoning_blocks(item, line)
            index = add_message(line, "assistant", reasoning_blocks, phase="reasoning", timestamp=timestamp)
            mirrors.append(
                _Mirror(index, line, scope, "raw", "assistant", "reasoning", (), tuple(block.text for block in reasoning_blocks), _string(item.get("id")))
            )
            continue

        if kind == "event_msg" and item_kind in ("agent_reasoning", "agent_reasoning_raw_content"):
            value = item.get("text")
            reasoning = value if isinstance(value, str) else f"Reasoning content unavailable at line {line}"
            label = "Reasoning summary" if item_kind == "agent_reasoning" else "Recorded reasoning"
            event_reasoning_blocks = (ReasoningBlock(reasoning, label),)
            index = add_message(line, "assistant", event_reasoning_blocks, phase="reasoning", timestamp=timestamp)
            mirrors.append(_Mirror(index, line, scope, "event", "assistant", "reasoning", (), (reasoning,), _string(item.get("id"))))
            continue

        if kind == "response_item" and item_kind == "agent_message":
            blocks, extra = _content_blocks(item.get("content"), line)
            warnings.extend(extra)
            author = _string(item.get("author")) or "agent"
            recipient = _string(item.get("recipient")) or "agent"
            add_message(line, "context", (TextBlock(f"{author} → {recipient}"), *blocks), phase="inter-agent", timestamp=timestamp)
            continue

        if kind == "response_item" and item_kind in (
            "function_call",
            "custom_tool_call",
            "local_shell_call",
            "web_search_call",
            "tool_search_call",
            "image_generation_call",
        ):
            block = _tool_call(item, line)
            index = add_message(line, "assistant", (block,), phase="tool", timestamp=timestamp)
            calls.append(_Call(index, line, scope, block.call_id))
            continue

        if kind == "response_item" and item_kind in (
            "function_call_output",
            "custom_tool_call_output",
            "tool_search_output",
        ):
            output, extra = _result(item, line)
            warnings.extend(extra)
            index = add_message(line, "tool", output.blocks, phase="tool", timestamp=timestamp)
            results.append(_Result(index, line, scope, _string(item.get("call_id")), output))
            continue

        if kind == "event_msg" and item_kind == "item_completed":
            completed = _object(item.get("item"))
            completed_kind = completed.get("type")
            explicit_turn = _string(item.get("turn_id"))
            completed_scope = "turn:" + explicit_turn if explicit_turn else scope
            if completed_kind == "Plan":
                blocks, extra = _content_blocks(completed.get("content"), line)
                warnings.extend(extra)
                add_message(line, "assistant", blocks, phase="plan", timestamp=timestamp)
            elif completed_kind == "FunctionCallOutput":
                output, extra = _result(completed, line)
                warnings.extend(extra)
                index = add_message(line, "tool", output.blocks, phase="tool", timestamp=timestamp)
                results.append(_Result(index, line, completed_scope, _string(completed.get("call_id")), output))
            else:
                label = "Unsupported Codex item dialect" if isinstance(completed_kind, str) and completed_kind[:1].islower() else "Unsupported Codex item"
                marker = _discriminant(completed_kind)
                states.append(_State(Notice(_id(line), label, f"Codex item {marker} at line {line} is not supported.")))
                warnings.append(f"Line {line} contains unsupported Codex item {marker}.")
            continue

        if kind == "compacted" or kind == "response_item" and item_kind in (
            "compaction",
            "compaction_summary",
            "context_compaction",
        ) or kind == "event_msg" and item_kind == "context_compaction":
            summary = item.get("message", item.get("summary"))
            notice_text = summary if isinstance(summary, str) else "A compaction checkpoint was recorded."
            states.append(_State(Notice(_id(line), "Compaction checkpoint", notice_text)))
            if item.get("replacement_history") is not None:
                warnings.append(f"Line {line} replacement history was not appended to the transcript.")
            continue

        label = "Unsupported Codex event" if kind == "event_msg" else "Unsupported Codex record"
        record_name = item_kind if kind in ("event_msg", "response_item") else kind
        marker = _discriminant(record_name)
        states.append(_State(Notice(_id(line), label, f"Codex record {marker} at line {line} is not supported.")))
        warnings.append(f"Line {line} contains unsupported Codex record {marker}.")

    _reconcile_mirrors(states, mirrors, warnings)
    _pair_tools(states, calls, results, warnings)
    entries = tuple(state.entry for state in states if not state.suppressed)
    if not entries:
        raise TranscriptUnavailable("Codex source contains no supported conversation records.", "unsupported")
    title = _string(metadata.get("agent_nickname")) or first_user_text or "Codex session"
    title = " ".join(title.split())[:160] or "Codex session"
    return Transcript(
        session_id,
        native_id,
        title,
        "codex",
        entries,
        _string(metadata.get("cwd")),
        model,
        _string(metadata.get("timestamp")) or _string(header.get("timestamp")),
        warnings=tuple(warnings),
    )


__all__ = ["parse_codex_transcript"]
