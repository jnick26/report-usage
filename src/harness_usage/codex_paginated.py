"""Project self-contained Codex paginated JSONL as a stored-record transcript."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import cast

from .codex_transcript import (
    Role,
    _Call,
    _content_blocks,
    _discriminant,
    _id,
    _json,
    _object,
    _pair_tools,
    _reasoning_blocks,
    _Result,
    _result,
    _safe_role,
    _signature,
    _State,
    _status,
    _string,
    _tool_call,
)
from .transcript import (
    AttachmentBlock,
    ConflictingOutput,
    JsonlRecord,
    Message,
    MessageBlock,
    Notice,
    ReasoningBlock,
    RecordedOutput,
    TextBlock,
    ToolBlock,
    Transcript,
    TranscriptUnavailable,
)

_TOOL_NEIGHBORHOOD = 4


@dataclass(frozen=True, slots=True)
class _IdentityRecord:
    state_index: int
    line: int
    scope: str
    kind: str
    native_id: str | None
    typed: bool


@dataclass(frozen=True, slots=True)
class _UserRecord:
    state_index: int
    line: int
    scope: str
    signature: tuple[tuple[str, str, str], ...]
    typed: bool


@dataclass(frozen=True, slots=True)
class _TypedTool:
    state_index: int
    line: int
    scope: str
    native_id: str | None
    kind: str


@dataclass(frozen=True, slots=True)
class _TypedOutput:
    state_index: int
    line: int
    scope: str
    native_id: str | None


def _message_blocks(state: _State) -> tuple[MessageBlock, ...]:
    return cast(Message, state.entry).blocks


def _replace_tool(state: _State, block: ToolBlock) -> None:
    message = cast(Message, state.entry)
    state.entry = replace(message, blocks=(block,))


def _tool(state: _State) -> ToolBlock:
    return next(block for block in cast(Message, state.entry).blocks if isinstance(block, ToolBlock))


def _merge_output(
    typed: RecordedOutput | ConflictingOutput | None,
    raw: RecordedOutput | ConflictingOutput | None,
    line: int,
) -> RecordedOutput | ConflictingOutput | None:
    if typed is None:
        return raw
    if raw is None:
        return typed
    if isinstance(typed, ConflictingOutput) or isinstance(raw, ConflictingOutput):
        return ConflictingOutput("Conflicting typed and raw tool output evidence")
    terminal = {typed.status, raw.status} - {"recorded"}
    if len(terminal) > 1:
        return ConflictingOutput("Typed and raw tool outputs record conflicting terminal statuses")
    blocks = list(typed.blocks)
    blocks.extend(block for block in raw.blocks if block not in blocks)
    status = typed.status if typed.status != "recorded" else raw.status
    return RecordedOutput(tuple(blocks), status, line)


def _output_from_values(
    values: tuple[object, ...], status: object, line: int
) -> RecordedOutput | None:
    blocks: list[TextBlock] = []
    recorded = False
    for value in values:
        if isinstance(value, str):
            recorded = True
            if value and TextBlock(value) not in blocks:
                blocks.append(TextBlock(value))
    return RecordedOutput(tuple(blocks), _status(status), line) if recorded else None


def _typed_output(item: dict[str, object], kind: str, line: int) -> tuple[
    RecordedOutput | None, tuple[str, ...]
]:
    if kind == "CommandExecution":
        return _output_from_values(
            tuple(item.get(field) for field in ("formatted_output", "aggregated_output", "stdout", "stderr")),
            item.get("status"),
            line,
        ), ()
    if kind == "FileChange":
        return _output_from_values(
            (item.get("stdout"), item.get("stderr")), item.get("status"), line
        ), ()
    if kind == "DynamicToolCall":
        content = item.get("content_items")
        if content is None and isinstance(item.get("error"), str):
            content = cast(str, item["error"])
        if content is None:
            return None, ()
        blocks, warnings = _content_blocks(content, line)
        success = item.get("success")
        status = "succeeded" if success is True else "failed" if success is False else item.get("status")
        return RecordedOutput(blocks, _status(status), line), warnings
    if kind == "McpToolCall":
        error = _object(item.get("error"))
        if isinstance(error.get("message"), str):
            return RecordedOutput((TextBlock(cast(str, error["message"])),), "failed", line), ()
        result = item.get("result")
        result_object = _object(result)
        content = result_object.get("content") if isinstance(result, dict) else result
        if content is None:
            return None, ()
        blocks, warnings = _content_blocks(content, line)
        status = "failed" if result_object.get("isError") is True else item.get("status")
        return RecordedOutput(blocks, _status(status), line), warnings
    if kind == "WebSearch":
        results = item.get("results")
        if results is None:
            return None, ()
        return RecordedOutput((TextBlock(_json(results)),), "recorded", line), ()
    if kind == "ImageGeneration":
        saved = _string(item.get("saved_path"))
        label = f"Generated image reference: {saved}" if saved else "Generated image data was recorded."
        return RecordedOutput((AttachmentBlock(label),), _status(item.get("status")), line), ()
    if kind == "Extension":
        value = item.get("output", item.get("result"))
        if value is None:
            return None, ()
        blocks, warnings = _content_blocks(value, line)
        return RecordedOutput(blocks, _status(item.get("status")), line), warnings
    return None, ()


def _typed_tool(item: dict[str, object], kind: str, line: int) -> tuple[ToolBlock, tuple[str, ...]]:
    native_id = _string(item.get("id"))
    if kind == "CommandExecution":
        name = "command_execution"
        arguments: object = {
            key: item[key]
            for key in ("command", "cwd", "source", "interaction_input")
            if key in item
        }
    elif kind == "DynamicToolCall":
        tool = _string(item.get("tool")) or "dynamic_tool"
        namespace = _string(item.get("namespace"))
        name = f"{namespace}.{tool}" if namespace else tool
        arguments = item.get("arguments")
    elif kind == "CollabAgentToolCall":
        name = "collaboration." + _discriminant(item.get("tool"))
        arguments = {
            key: item[key]
            for key in (
                "prompt",
                "model",
                "reasoning_effort",
                "sender_thread_id",
                "receiver_thread_ids",
                "receiver_agents",
            )
            if key in item
        }
    elif kind == "SubAgentActivity":
        name = "subagent." + _discriminant(item.get("kind"))
        arguments = {
            key: item[key] for key in ("agent_thread_id", "agent_path") if key in item
        }
    elif kind == "WebSearch":
        name = "web_search"
        arguments = {key: item[key] for key in ("query", "action") if key in item}
    elif kind == "ImageView":
        name = "view_image"
        arguments = {"path": item.get("path")}
    elif kind == "Extension":
        name = "extension." + _discriminant(item.get("kind"))
        arguments = {
            key: value
            for key, value in item.items()
            if key not in {"type", "id", "status", "output", "result"}
        }
    elif kind == "ImageGeneration":
        name = "image_generation"
        arguments = {"revised_prompt": item.get("revised_prompt")}
    elif kind == "FileChange":
        name = "file_change"
        arguments = {"changes": item.get("changes")}
    else:
        server = _string(item.get("server")) or "mcp"
        tool = _string(item.get("tool")) or "tool"
        name = f"{server}.{tool}"
        arguments = item.get("arguments")
    argument_text = arguments if isinstance(arguments, str) else _json(arguments)
    output, warnings = _typed_output(item, kind, line)
    return ToolBlock(native_id, name, argument_text, output, line), warnings


def _typed_reasoning(item: dict[str, object], line: int) -> tuple[ReasoningBlock, ...]:
    blocks: list[ReasoningBlock] = []
    summary = item.get("summary_text")
    if isinstance(summary, list):
        blocks.extend(ReasoningBlock(value, "Reasoning summary") for value in summary if isinstance(value, str))
    raw = item.get("raw_content")
    if isinstance(raw, list):
        blocks.extend(ReasoningBlock(value, "Recorded reasoning") for value in raw if isinstance(value, str))
    return tuple(blocks) or (ReasoningBlock(f"Reasoning content unavailable at line {line}"),)


def _reconcile_identities(
    states: list[_State], records: list[_IdentityRecord], warnings: list[str]
) -> None:
    groups: dict[tuple[str, str, str], list[_IdentityRecord]] = {}
    identity_participants: set[int] = set()
    for record in records:
        if record.native_id is not None:
            groups.setdefault((record.scope, record.kind, record.native_id), []).append(record)
    for (_, kind, native_id), group in groups.items():
        raw_records = [record for record in group if not record.typed]
        typed_records = [record for record in group if record.typed]
        if raw_records and typed_records:
            identity_participants.update(record.state_index for record in group)
        if len(raw_records) == len(typed_records) == 1:
            raw_state = states[raw_records[0].state_index]
            typed_state = states[typed_records[0].state_index]
            raw_message = cast(Message, raw_state.entry)
            typed_message = cast(Message, typed_state.entry)
            merged = list(typed_message.blocks)
            merged.extend(block for block in _message_blocks(raw_state) if block not in merged)
            if kind == "reasoning":
                available = [
                    block
                    for block in merged
                    if not (
                        isinstance(block, ReasoningBlock)
                        and "unavailable" in block.text.lower()
                    )
                ]
                merged = available or merged[:1]
            phase = typed_message.phase or raw_message.phase
            if kind != "reasoning" and typed_message.blocks != raw_message.blocks:
                warnings.append(
                    f"Shared Codex {kind} identity {_discriminant(native_id)} has conflicting content; both block sets were retained."
                )
            if typed_message.phase and raw_message.phase and typed_message.phase != raw_message.phase:
                warnings.append(
                    f"Shared Codex {kind} identity {_discriminant(native_id)} has conflicting phases."
                )
            typed_state.entry = replace(typed_message, blocks=tuple(merged), phase=phase)
            raw_state.suppressed = True
        elif raw_records and typed_records:
            warnings.append(
                f"Duplicate Codex {kind} identity {_discriminant(native_id)} was retained as conflicting evidence."
            )

    # Older paginated sources persist the same completed assistant item first as
    # item-N and then as an adjacent raw response message with a msg_ identity.
    raw_messages_by_line: dict[tuple[str, int], list[_IdentityRecord]] = {}
    for record in records:
        if record.kind == "assistant message" and not record.typed:
            raw_messages_by_line.setdefault((record.scope, record.line), []).append(record)
    for typed in (record for record in records if record.kind == "assistant message" and record.typed):
        if states[typed.state_index].suppressed or typed.state_index in identity_participants:
            continue
        typed_id = typed.native_id
        if typed_id is None or not typed_id.startswith("item-") or not typed_id[5:].isdigit():
            continue
        possible = [
            raw
            for raw in raw_messages_by_line.get((typed.scope, typed.line + 1), ())
            if not states[raw.state_index].suppressed
            and raw.state_index not in identity_participants
            and raw.native_id is not None
            and raw.native_id.startswith("msg_")
        ]
        if len(possible) != 1:
            continue
        raw_record = possible[0]
        raw_message = cast(Message, states[raw_record.state_index].entry)
        typed_message = cast(Message, states[typed.state_index].entry)
        if (
            typed_message.phase is not None
            and typed_message.phase == raw_message.phase
            and typed_message.blocks == raw_message.blocks
        ):
            states[raw_record.state_index].suppressed = True


def _reconcile_users(
    states: list[_State], users: list[_UserRecord], warnings: list[str]
) -> None:
    raw_by_line: dict[tuple[str, int], list[_UserRecord]] = {}
    for record in users:
        if not record.typed:
            raw_by_line.setdefault((record.scope, record.line), []).append(record)
    used: set[int] = set()
    for typed in (record for record in users if record.typed):
        possible: dict[int, _UserRecord] = {}
        for line in range(typed.line - _TOOL_NEIGHBORHOOD, typed.line + _TOOL_NEIGHBORHOOD + 1):
            for raw in raw_by_line.get((typed.scope, line), ()):
                if raw.state_index not in used and raw.signature == typed.signature:
                    possible[raw.state_index] = raw
        if len(possible) == 1:
            matched = next(iter(possible.values()))
            states[matched.state_index].suppressed = True
            used.add(matched.state_index)
        elif len(possible) > 1:
            warnings.append(
                f"Line {typed.line} may mirror multiple raw user records; all ambiguous records were retained."
            )
    for raw in (record for record in users if not record.typed):
        state = states[raw.state_index]
        if state.suppressed:
            continue
        message = cast(Message, state.entry)
        if message.role == "user":
            state.entry = replace(message, role="context", phase=message.phase or "model-context")


def _reconcile_tools(
    states: list[_State], calls: list[_Call], results: list[_Result], typed: list[_TypedTool], warnings: list[str]
) -> None:
    call_groups: dict[tuple[str, str], list[_Call]] = {}
    result_groups: dict[tuple[str, str], list[_Result]] = {}
    typed_groups: dict[tuple[str, str], list[_TypedTool]] = {}
    for call in calls:
        if call.call_id is not None:
            call_groups.setdefault((call.scope, call.call_id), []).append(call)
    for raw_result in results:
        if raw_result.call_id is not None:
            result_groups.setdefault((raw_result.scope, raw_result.call_id), []).append(raw_result)
    for item in typed:
        if item.native_id is not None:
            typed_groups.setdefault((item.scope, item.native_id), []).append(item)

    conflicting_typed: set[int] = set()
    for (_, native_id), group in typed_groups.items():
        if len(group) < 2:
            continue
        reason = f"Duplicate typed tool identity {_discriminant(native_id)} in one turn"
        for item in group:
            block = _tool(states[item.state_index])
            conflict = replace(block, output=ConflictingOutput(reason))
            evidence = block.output.blocks if isinstance(block.output, RecordedOutput) else ()
            message = cast(Message, states[item.state_index].entry)
            states[item.state_index].entry = replace(
                message, blocks=(conflict, *evidence)
            )
            conflicting_typed.add(item.state_index)
        warnings.append(reason + ".")

    used_calls: set[int] = set()
    used_results: set[int] = set()
    used_typed: set[int] = set()

    def contains(container: object, target: object) -> bool:
        if container == target:
            return True
        if isinstance(container, dict):
            return any(contains(value, target) for value in container.values())
        if isinstance(container, list):
            return any(contains(value, target) for value in container)
        return False

    def command_text(value: object) -> str | None:
        if isinstance(value, str):
            return value
        if isinstance(value, list) and value and all(isinstance(part, str) for part in value):
            return " ".join(cast(str, part) for part in value)
        if isinstance(value, dict):
            for key in ("cmd", "command"):
                if key in value:
                    return command_text(value[key])
        return None

    def compatible(item: _TypedTool, call: _Call) -> bool:
        typed_block = _tool(states[item.state_index])
        raw_block = _tool(states[call.state_index])
        try:
            raw_arguments = json.loads(raw_block.arguments)
            typed_arguments = json.loads(typed_block.arguments)
        except (TypeError, ValueError, RecursionError):
            raw_arguments = raw_block.arguments
            try:
                typed_arguments = json.loads(typed_block.arguments)
            except (TypeError, ValueError, RecursionError):
                typed_arguments = typed_block.arguments
        if item.kind == "CommandExecution":
            if raw_block.name not in {"exec", "exec_command", "command_execution"}:
                return False
            command = typed_arguments.get("command") if isinstance(typed_arguments, dict) else None
            if isinstance(command, list) and command and isinstance(command[-1], str):
                recorded = command[-1]
                if json.dumps(recorded) in raw_block.arguments:
                    return True
            expected = command_text(typed_arguments)
            candidates = {command_text(raw_arguments), raw_block.arguments}
            return expected is not None and expected in candidates
        if item.kind == "FileChange":
            if raw_block.name not in {"exec", "exec_command", "apply_patch", "file_change"}:
                return False
            return "apply_patch" in raw_block.arguments or "*** Begin Patch" in raw_block.arguments
        if typed_arguments in ({}, [], None, ""):
            return False
        names_match = raw_block.name == typed_block.name
        return names_match and contains(raw_arguments, typed_arguments)

    def associate(item: _TypedTool, call: _Call, result: _Result | None) -> None:
        typed_block = _tool(states[item.state_index])
        raw_block = _tool(states[call.state_index])
        output = _merge_output(typed_block.output, result.output if result else None, item.line)
        merged_block = ToolBlock(
            call.call_id, raw_block.name, raw_block.arguments, output, item.line
        )
        if isinstance(output, ConflictingOutput):
            evidence = (
                typed_block.output.blocks
                if isinstance(typed_block.output, RecordedOutput)
                else ()
            )
            message = cast(Message, states[item.state_index].entry)
            states[item.state_index].entry = replace(
                message, blocks=(merged_block, *evidence)
            )
            warnings.append(
                f"Line {item.line} has conflicting typed and raw tool output status; both output bodies were retained."
            )
        else:
            _replace_tool(states[item.state_index], merged_block)
        states[call.state_index].suppressed = True
        used_calls.add(call.state_index)
        used_typed.add(item.state_index)
        if result is not None and not isinstance(output, ConflictingOutput):
            states[result.state_index].suppressed = True
            used_results.add(result.state_index)

    for item in typed:
        if item.state_index in conflicting_typed or item.native_id is None:
            continue
        key = (item.scope, item.native_id)
        matched_calls = call_groups.get(key, [])
        matched_results = result_groups.get(key, [])
        if len(matched_calls) == 1 and len(matched_results) <= 1:
            associate(item, matched_calls[0], matched_results[0] if matched_results else None)

    pair_by_line: dict[tuple[str, int], list[tuple[_Call, _Result | None]]] = {}
    for key, grouped_calls in call_groups.items():
        grouped_results = result_groups.get(key, [])
        if len(grouped_calls) == 1 and len(grouped_results) <= 1:
            call = grouped_calls[0]
            pair_by_line.setdefault((call.scope, call.line), []).append(
                (call, grouped_results[0] if grouped_results else None)
            )
    for item in typed:
        if item.state_index in conflicting_typed or item.state_index in used_typed:
            continue
        possible: list[tuple[_Call, _Result | None]] = []
        for line in range(item.line - _TOOL_NEIGHBORHOOD, item.line + 2):
            for call, pair_result in pair_by_line.get((item.scope, line), ()):
                if call.state_index in used_calls or not compatible(item, call):
                    continue
                end_line = pair_result.line if pair_result is not None else call.line
                if call.line - 1 <= item.line <= end_line + _TOOL_NEIGHBORHOOD:
                    possible.append((call, pair_result))
        if len(possible) == 1:
            associate(item, *possible[0])
        elif len(possible) > 1:
            warnings.append(
                f"Line {item.line} may represent multiple raw tool calls; all ambiguous records were retained."
            )

    remaining_calls = [
        call
        for call in calls
        if call.state_index not in used_calls and not states[call.state_index].suppressed
    ]
    remaining_results = [
        result
        for result in results
        if result.state_index not in used_results and not states[result.state_index].suppressed
    ]
    _pair_tools(states, remaining_calls, remaining_results, warnings)


def _reconcile_function_outputs(
    states: list[_State], results: list[_Result], typed: list[_TypedOutput], warnings: list[str]
) -> None:
    raw_groups: dict[tuple[str, str], list[_Result]] = {}
    for result in results:
        if result.call_id is not None:
            raw_groups.setdefault((result.scope, result.call_id), []).append(result)
    for item in typed:
        if item.native_id is None:
            continue
        matched = raw_groups.get((item.scope, item.native_id), [])
        if len(matched) != 1:
            if len(matched) > 1:
                warnings.append(
                    f"Typed function output {_discriminant(item.native_id)} matches multiple raw results; all were retained."
                )
            continue
        typed_blocks = cast(Message, states[item.state_index].entry).blocks
        if typed_blocks == matched[0].output.blocks:
            states[item.state_index].suppressed = True
        else:
            warnings.append(
                f"Typed and raw function output {_discriminant(item.native_id)} contain conflicting evidence; both were retained."
            )


def project_codex_paginated(
    records: tuple[JsonlRecord, ...],
    initial_warnings: tuple[str, ...],
    session_id: str,
    metadata: dict[str, object],
    header: dict[str, object],
) -> Transcript:
    """Project one self-contained paginated rollout without reconstructing history."""
    native_id = cast(str, _string(metadata.get("id")))
    warnings = list(initial_warnings)
    states = [
        _State(
            Notice(
                "codex-stored-record-view",
                "Stored-record view",
                "This paginated transcript shows records stored in the local rollout file.",
            )
        )
    ]
    identities: list[_IdentityRecord] = []
    users: list[_UserRecord] = []
    calls: list[_Call] = []
    results: list[_Result] = []
    typed_tools: list[_TypedTool] = []
    typed_outputs: list[_TypedOutput] = []
    scope = "file"
    model = _string(metadata.get("model"))
    first_typed_user_text: str | None = None
    first_raw_user_text: str | None = None
    conversation_seen = False
    history_start = metadata.get("subagent_history_start_ordinal")
    inherited_cutoff = history_start if type(history_start) is int and history_start >= 0 else None

    def inherited(raw: dict[str, object]) -> bool:
        ordinal = raw.get("ordinal")
        record_metadata = _object(raw.get("metadata"))
        return record_metadata.get("inherited_user_message") is True or (
            inherited_cutoff is not None and type(ordinal) is int and ordinal < inherited_cutoff
        )

    def add_message(
        raw: dict[str, object],
        line: int,
        role: Role,
        blocks: tuple[object, ...],
        *,
        phase: str | None = None,
        timestamp: str | None = None,
    ) -> int:
        if inherited(raw):
            role = "context"
            phase = "inherited"
        states.append(
            _State(
                Message(
                    _id(line),
                    role,
                    cast(tuple[TextBlock | ReasoningBlock | AttachmentBlock | ToolBlock, ...], blocks),
                    model,
                    phase,
                    timestamp,
                )
            )
        )
        return len(states) - 1

    for record in records[1:]:
        raw = record.value
        line = record.line
        kind = raw.get("type")
        payload = _object(raw.get("payload"))
        item_kind = payload.get("type")
        timestamp = _string(raw.get("timestamp"))
        if kind in ("thread_rolled_back", "rollback") or item_kind in ("thread_rolled_back", "rollback"):
            raise TranscriptUnavailable(
                "Codex rollback requires unsupported current-history reconstruction.", "unsupported"
            )
        if kind == "session_meta":
            if _string(payload.get("id")) != native_id:
                raise TranscriptUnavailable(
                    "Codex transcript contains conflicting session metadata.", "invalid_source"
                )
            continue
        if kind == "turn_context":
            turn_id = _string(payload.get("turn_id"))
            if turn_id:
                scope = "turn:" + turn_id
            model = _string(payload.get("model")) or model
            continue
        if kind in ("token_usage_record", "token_count", "world_state") or kind == "event_msg" and item_kind in (
            "token_count",
            "thread_settings_applied",
            "task_started",
            "task_complete",
            "inter_agent_communication_metadata",
        ):
            continue
        if kind == "inter_agent_communication_metadata":
            continue

        if kind == "response_item" and item_kind == "message":
            role = _safe_role(payload.get("role"), line, warnings)
            blocks, extra = _content_blocks(payload.get("content"), line)
            warnings.extend(extra)
            index = add_message(raw, line, role, blocks, phase=_string(payload.get("phase")), timestamp=timestamp)
            if role == "user":
                users.append(_UserRecord(index, line, scope, _signature(blocks), False))
                if not inherited(raw) and first_raw_user_text is None:
                    first_raw_user_text = next(
                        (block.text for block in blocks if isinstance(block, TextBlock)),
                        None,
                    )
            elif role == "assistant":
                identities.append(_IdentityRecord(index, line, scope, "assistant message", _string(payload.get("id")), False))
            conversation_seen = True
            continue

        if kind == "response_item" and item_kind == "reasoning":
            reasoning_blocks = _reasoning_blocks(payload, line)
            index = add_message(raw, line, "assistant", reasoning_blocks, phase="reasoning", timestamp=timestamp)
            identities.append(_IdentityRecord(index, line, scope, "reasoning", _string(payload.get("id")), False))
            conversation_seen = True
            continue

        if kind == "response_item" and item_kind == "agent_message":
            blocks, extra = _content_blocks(payload.get("content"), line)
            warnings.extend(extra)
            author = _string(payload.get("author")) or "agent"
            recipient = _string(payload.get("recipient")) or "agent"
            add_message(raw, line, "context", (TextBlock(f"{author} → {recipient}"), *blocks), phase="inter-agent", timestamp=timestamp)
            conversation_seen = True
            continue

        if kind == "response_item" and item_kind in (
            "function_call",
            "custom_tool_call",
            "local_shell_call",
            "web_search_call",
            "tool_search_call",
            "image_generation_call",
        ):
            block = _tool_call(payload, line)
            index = add_message(raw, line, "assistant", (block,), phase="tool", timestamp=timestamp)
            calls.append(_Call(index, line, scope, block.call_id))
            conversation_seen = True
            continue

        if kind == "response_item" and item_kind in (
            "function_call_output",
            "custom_tool_call_output",
            "tool_search_output",
        ):
            output, extra = _result(payload, line)
            warnings.extend(extra)
            index = add_message(raw, line, "tool", output.blocks, phase="tool", timestamp=timestamp)
            results.append(_Result(index, line, scope, _string(payload.get("call_id")), output))
            if _string(payload.get("call_id")) is None:
                identities.append(
                    _IdentityRecord(
                        index,
                        line,
                        scope,
                        "function output",
                        _string(payload.get("id")),
                        False,
                    )
                )
            conversation_seen = True
            continue

        if kind == "event_msg" and item_kind == "item_completed":
            item = _object(payload.get("item"))
            completed_kind = item.get("type")
            explicit_turn = _string(payload.get("turn_id"))
            item_scope = "turn:" + explicit_turn if explicit_turn else scope
            if completed_kind == "UserMessage":
                blocks, extra = _content_blocks(item.get("content"), line)
                warnings.extend(extra)
                index = add_message(raw, line, "user", blocks, timestamp=timestamp)
                users.append(_UserRecord(index, line, item_scope, _signature(blocks), True))
                if not inherited(raw) and first_typed_user_text is None:
                    first_typed_user_text = next(
                        (block.text for block in blocks if isinstance(block, TextBlock)),
                        None,
                    )
            elif completed_kind == "AgentMessage":
                blocks, extra = _content_blocks(item.get("content"), line)
                warnings.extend(extra)
                index = add_message(raw, line, "assistant", blocks, phase=_string(item.get("phase")), timestamp=timestamp)
                identities.append(_IdentityRecord(index, line, item_scope, "assistant message", _string(item.get("id")), True))
            elif completed_kind == "Reasoning":
                typed_reasoning_blocks = _typed_reasoning(item, line)
                index = add_message(raw, line, "assistant", typed_reasoning_blocks, phase="reasoning", timestamp=timestamp)
                identities.append(_IdentityRecord(index, line, item_scope, "reasoning", _string(item.get("id")), True))
            elif completed_kind == "Plan":
                value = item.get("text")
                blocks = (TextBlock(value),) if isinstance(value, str) else (AttachmentBlock(f"Plan content unavailable at line {line}"),)
                add_message(raw, line, "assistant", blocks, phase="plan", timestamp=timestamp)
            elif completed_kind == "FunctionCallOutput":
                output, extra = _result(item, line)
                warnings.extend(extra)
                index = add_message(raw, line, "tool", output.blocks, phase="tool", timestamp=timestamp)
                identities.append(_IdentityRecord(index, line, item_scope, "function output", _string(item.get("id")), True))
                typed_outputs.append(
                    _TypedOutput(index, line, item_scope, _string(item.get("id")))
                )
            elif completed_kind in {
                "CommandExecution",
                "DynamicToolCall",
                "CollabAgentToolCall",
                "SubAgentActivity",
                "WebSearch",
                "ImageView",
                "Extension",
                "ImageGeneration",
                "FileChange",
                "McpToolCall",
            }:
                block, extra = _typed_tool(item, completed_kind, line)
                warnings.extend(extra)
                index = add_message(raw, line, "assistant", (block,), phase="tool", timestamp=timestamp)
                typed_tools.append(
                    _TypedTool(
                        index,
                        line,
                        item_scope,
                        _string(item.get("id")),
                        completed_kind,
                    )
                )
            elif completed_kind == "ContextCompaction":
                states.append(_State(Notice(_id(line), "Compaction checkpoint", "A context compaction checkpoint was recorded.")))
            elif completed_kind in ("HookPrompt", "EnteredReviewMode", "ExitedReviewMode"):
                marker = _discriminant(completed_kind)
                states.append(_State(Notice(_id(line), "Codex activity", f"Codex {marker} activity was recorded at line {line}.")))
            else:
                marker = _discriminant(completed_kind)
                states.append(_State(Notice(_id(line), "Unsupported Codex item", f"Codex item {marker} at line {line} is not supported.")))
                warnings.append(f"Line {line} contains unsupported Codex item {marker}.")
            conversation_seen = True
            continue

        if kind == "compacted" or kind == "response_item" and item_kind in (
            "compaction",
            "compaction_summary",
            "context_compaction",
        ) or kind == "event_msg" and item_kind == "context_compaction":
            states.append(_State(Notice(_id(line), "Compaction checkpoint", "A compaction checkpoint was recorded.")))
            if payload.get("replacement_history") is not None:
                warnings.append(f"Line {line} replacement history was not appended to the transcript.")
            conversation_seen = True
            continue

        label = "Unsupported Codex event" if kind == "event_msg" else "Unsupported Codex record"
        record_name = item_kind if kind in ("event_msg", "response_item") else kind
        marker = _discriminant(record_name)
        states.append(_State(Notice(_id(line), label, f"Codex record {marker} at line {line} is not supported.")))
        warnings.append(f"Line {line} contains unsupported Codex record {marker}.")
        conversation_seen = True

    _reconcile_identities(states, identities, warnings)
    _reconcile_users(states, users, warnings)
    _reconcile_function_outputs(states, results, typed_outputs, warnings)
    _reconcile_tools(states, calls, results, typed_tools, warnings)
    if not conversation_seen:
        raise TranscriptUnavailable("Codex source contains no supported conversation records.", "unsupported")
    entries = tuple(state.entry for state in states if not state.suppressed)
    title = (
        _string(metadata.get("agent_nickname"))
        or first_typed_user_text
        or first_raw_user_text
        or "Codex session"
    )
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


__all__ = ["project_codex_paginated"]
