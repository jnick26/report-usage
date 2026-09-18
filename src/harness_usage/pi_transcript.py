"""Read-only projection of one Pi v3 JSONL session branch."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
import json
from typing import cast

from .transcript import (
    AttachmentBlock,
    Branch,
    ConflictingOutput,
    ImageBlock,
    Message,
    MessageBlock,
    Notice,
    OutputBlock,
    OutputStatus,
    ReasoningBlock,
    RecordedOutput,
    TextBlock,
    ToolBlock,
    Transcript,
    TranscriptEntry,
    TranscriptUnavailable,
    read_jsonl,
    text_and_media_blocks,
)


@dataclass(frozen=True, slots=True)
class _Node:
    id: str
    parent_id: str | None
    line: int
    value: dict[str, object]


@dataclass(frozen=True, slots=True)
class _PendingResult:
    call_id: str | None
    name: str
    output: RecordedOutput
    timestamp: str | None
    id: str


def _text(value: object) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _safe_name(value: object, fallback: str) -> str:
    text = _text(value)
    return (text[:77] + "…") if text is not None and len(text) > 78 else (text or fallback)


def _timestamp(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _entry_id(prefix: str, line: int, suffix: int | None = None) -> str:
    return f"{prefix}-{line}" + (f"-{suffix}" if suffix is not None else "")


def _assistant_blocks(content: object, line: int, warnings: list[str]) -> tuple[MessageBlock, ...]:
    if not isinstance(content, list):
        warnings.append(f"Line {line} has missing or unsupported assistant content.")
        return (AttachmentBlock(f"Assistant content unavailable at line {line}"),)
    blocks: list[MessageBlock] = []
    for raw in content:
        item = raw if isinstance(raw, dict) else {}
        kind = item.get("type")
        if kind == "text" and isinstance(item.get("text"), str):
            blocks.append(TextBlock(cast(str, item["text"])))
        elif kind == "thinking":
            thinking = item.get("thinking")
            if isinstance(thinking, str) and thinking:
                blocks.append(ReasoningBlock(thinking))
            elif item.get("redacted") is True:
                blocks.append(ReasoningBlock("Recorded reasoning was redacted."))
            else:
                warnings.append(f"Line {line} contains reasoning without readable text.")
        elif kind == "toolCall":
            call_id = _text(item.get("id"))
            name = _text(item.get("name"))
            arguments = item.get("arguments")
            if isinstance(arguments, dict):
                rendered_arguments = json.dumps(arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            elif isinstance(arguments, str):
                rendered_arguments = arguments
            else:
                rendered_arguments = ""
                warnings.append(f"Line {line} has unsupported tool arguments.")
            if call_id is None:
                warnings.append(f"Line {line} has a tool call without a usable id.")
            if name is None:
                warnings.append(f"Line {line} has a tool call without a usable name.")
            blocks.append(ToolBlock(call_id, name or "Unknown tool", rendered_arguments, source_line=line))
        elif kind == "image":
            media, media_warnings = text_and_media_blocks([item], source_line=line)
            blocks.extend(media)
            warnings.extend(media_warnings)
        else:
            label = _safe_name(kind, "unknown")
            blocks.append(AttachmentBlock(f"Unsupported {label} content at line {line}"))
            warnings.append(f"Line {line} contains unsupported {label} assistant content.")
    return tuple(blocks)


def _project_message(
    body: dict[str, object], line: int, message_id: str, warnings: list[str],
    *, phase: str | None = None, timestamp: str | None = None,
) -> Message | Notice | _PendingResult:
    role = body.get("role")
    if role == "toolResult":
        blocks, block_warnings = text_and_media_blocks(body.get("content"), source_line=line)
        warnings.extend(block_warnings)
        failed = body.get("isError")
        status: OutputStatus = "failed" if failed is True else "succeeded" if failed is False else "recorded"
        if type(failed) is not bool:
            warnings.append(f"Line {line} has a tool result without a recorded outcome.")
        return _PendingResult(
            _text(body.get("toolCallId")), _safe_name(body.get("toolName"), "Unknown tool"),
            RecordedOutput(blocks, status, line), timestamp, message_id,
        )
    if role == "assistant":
        assistant_blocks = _assistant_blocks(body.get("content"), line, warnings)
        return Message(
            message_id, "assistant", assistant_blocks, _text(body.get("model")),
            phase or _text(body.get("stopReason")), timestamp,
        )
    if role == "bashExecution":
        command = body.get("command") if isinstance(body.get("command"), str) else ""
        output_text = body.get("output")
        output: list[OutputBlock] = [TextBlock(output_text if isinstance(output_text, str) else "")]
        if body.get("truncated") is True or _text(body.get("fullOutputPath")) is not None:
            output.append(AttachmentBlock("Additional command output was recorded externally and was not opened."))
        exit_code = body.get("exitCode")
        if exit_code is not None and type(exit_code) is not int:
            warnings.append(f"Line {line} has an invalid command exit code.")
        bash_status: OutputStatus = "failed" if body.get("cancelled") is True or (type(exit_code) is int and exit_code != 0) else "succeeded" if type(exit_code) is int and exit_code == 0 else "recorded"
        call = ToolBlock(None, "bash", cast(str, command), RecordedOutput(tuple(output), bash_status, line), line)
        context = "excluded from model context" if body.get("excludeFromContext") is True else phase
        return Message(message_id, "tool", (call,), phase=context, timestamp=timestamp)
    if role == "custom":
        blocks, block_warnings = text_and_media_blocks(body.get("content"), source_line=line)
        warnings.extend(block_warnings)
        visibility = "extension context" if body.get("display") is True else "hidden extension context"
        return Message(message_id, "context", blocks, phase=phase or visibility, timestamp=timestamp)
    if role in ("branchSummary", "compactionSummary"):
        summary = body.get("summary")
        text = summary if isinstance(summary, str) else "Summary content is unavailable."
        label = "Branch summary" if role == "branchSummary" else "Context compacted"
        return Notice(message_id, label, text)
    if role in ("user", "system", "developer", "tool", "context"):
        blocks, block_warnings = text_and_media_blocks(body.get("content"), source_line=line)
        warnings.extend(block_warnings)
        return Message(message_id, role, cast(tuple[MessageBlock, ...], blocks), phase=phase, timestamp=timestamp)
    label = _safe_name(role, "unknown")
    warnings.append(f"Line {line} contains unsupported message role {label}.")
    return Message(
        message_id, "context", (AttachmentBlock(f"Unsupported message role {label} at line {line}"),),
        phase="unsupported", timestamp=timestamp,
    )


def _project_node(
    node: _Node, known_ids: set[str], warnings: list[str]
) -> list[Message | Notice | _PendingResult]:
    raw = node.value
    kind = raw.get("type")
    timestamp = _timestamp(raw.get("timestamp"))
    local_id = _entry_id("pi-entry", node.line)
    if kind == "message":
        body = raw.get("message")
        if not isinstance(body, dict):
            warnings.append(f"Line {node.line} has an invalid message object.")
            return [Notice(local_id, "Unreadable Pi message", "The recorded message shape is unsupported.")]
        projected = _project_message(cast(dict[str, object], body), node.line, local_id, warnings, timestamp=timestamp)
        if isinstance(projected, Message) and projected.role == 'assistant':
            projected = replace(projected, usage_id=node.id)
        return [projected]
    if kind == "custom_message":
        body = {
            "role": "custom", "content": raw.get("content"), "display": raw.get("display"),
        }
        custom_type = _safe_name(raw.get("customType"), "unknown extension")
        visibility = "visible" if raw.get("display") is True else "hidden"
        custom_projected = _project_message(body, node.line, local_id, warnings, phase=f"{visibility} extension: {custom_type}", timestamp=timestamp)
        return [custom_projected]
    if kind == "compaction":
        summary = raw.get("summary")
        if not isinstance(summary, str):
            warnings.append(f"Line {node.line} has a compaction without a readable summary.")
        checkpoint_notice = Notice(local_id, "Context compacted", summary if isinstance(summary, str) else "Summary content is unavailable.")
        checkpoint_projected: list[Message | Notice | _PendingResult] = []
        first_kept = _text(raw.get("firstKeptEntryId"))
        retained = raw.get("retainedTail")
        if first_kept is not None and first_kept not in known_ids:
            warnings.append(f"Line {node.line} references a missing first-kept entry.")
        if first_kept is not None and retained is not None:
            warnings.append(f"Line {node.line} has both compaction retention shapes; model context is ambiguous.")
        if retained is not None:
            if not isinstance(retained, list):
                warnings.append(f"Line {node.line} has an invalid retained checkpoint.")
            else:
                for index, body in enumerate(retained, 1):
                    if not isinstance(body, dict):
                        warnings.append(f"Line {node.line} has an unsupported retained checkpoint item.")
                        continue
                    checkpoint_projected.append(_project_message(
                        cast(dict[str, object], body), node.line,
                        _entry_id("pi-checkpoint", node.line, index), warnings,
                        phase="retained checkpoint", timestamp=timestamp,
                    ))
        return [checkpoint_notice, *_pair_tools(checkpoint_projected, warnings)]
    if kind == "branch_summary":
        summary = raw.get("summary")
        if not isinstance(summary, str):
            warnings.append(f"Line {node.line} has a branch summary without readable text.")
        return [Notice(local_id, "Branch summary", summary if isinstance(summary, str) else "Summary content is unavailable.")]
    if kind == "model_change":
        provider = _safe_name(raw.get("provider"), "unknown provider")
        model = _safe_name(raw.get("modelId"), "unknown model")
        return [Notice(local_id, "Model changed", f"{provider} / {model}")]
    if kind == "thinking_level_change":
        return [Notice(local_id, "Reasoning level changed", _safe_name(raw.get("thinkingLevel"), "unknown"))]
    if kind == "label":
        label = _text(raw.get("label"))
        return [Notice(local_id, "Branch label updated", f"Label set to {label}." if label else "Label cleared.")]
    if kind == "session_info":
        name = _text(raw.get("name"))
        return [Notice(local_id, "Session name updated", name or "Session name cleared.")]
    if kind == "custom":
        custom_type = _safe_name(raw.get("customType"), "unknown extension")
        return [Notice(local_id, "Extension state", f"Recorded state for {custom_type}; opaque data was not interpreted.")]
    label = _safe_name(kind, "unknown")
    warnings.append(f"Line {node.line} contains unsupported Pi entry type {label}.")
    return [Notice(local_id, "Unsupported Pi entry", f"Recorded {label} entry at line {node.line}.")]


def _pair_tools(
    projected: list[Message | Notice | _PendingResult], warnings: list[str],
    *, skip_phase: str | None = None,
) -> tuple[TranscriptEntry, ...]:
    calls: dict[str, list[tuple[int, int, ToolBlock]]] = {}
    results: dict[str, list[tuple[int, _PendingResult]]] = {}
    for entry_index, entry in enumerate(projected):
        if isinstance(entry, Message):
            if skip_phase is not None and entry.phase == skip_phase:
                continue
            for block_index, block in enumerate(entry.blocks):
                if isinstance(block, ToolBlock) and block.call_id is not None and block.output is None:
                    calls.setdefault(block.call_id, []).append((entry_index, block_index, block))
        elif isinstance(entry, _PendingResult) and entry.call_id is not None:
            results.setdefault(entry.call_id, []).append((entry_index, entry))

    replacements: dict[tuple[int, int], ToolBlock] = {}
    standalone: dict[int, str] = {}
    for call_id, call_group in calls.items():
        result_group = results.get(call_id, [])
        conflict = len(call_group) != 1 or len(result_group) > 1
        if len(call_group) == 1 and len(result_group) == 1:
            conflict = call_group[0][2].name != result_group[0][1].name
        if conflict:
            reason = "Recorded tool identity is ambiguous."
            for entry_index, block_index, block in call_group:
                replacements[(entry_index, block_index)] = replace(block, output=ConflictingOutput(reason))
            for entry_index, _ in result_group:
                standalone[entry_index] = "conflicting"
            warnings.append("A tool call id is conflicting on the selected branch.")
        elif result_group:
            entry_index, block_index, block = call_group[0]
            result_index, result = result_group[0]
            replacements[(entry_index, block_index)] = replace(block, output=result.output)
            standalone[result_index] = "paired"

    entries: list[TranscriptEntry] = []
    for entry_index, entry in enumerate(projected):
        if isinstance(entry, _PendingResult):
            state = standalone.get(entry_index)
            if state == "paired":
                continue
            phase = state or "unmatched"
            if phase == "unmatched":
                warnings.append(f"Line {entry.output.source_line} has an unmatched tool result.")
            tool = ToolBlock(entry.call_id, entry.name, "", entry.output, entry.output.source_line)
            entries.append(Message(entry.id, "tool", (tool,), phase=phase, timestamp=entry.timestamp))
            continue
        if isinstance(entry, Message):
            blocks = tuple(replacements.get((entry_index, index), block) for index, block in enumerate(entry.blocks))
            entries.append(replace(entry, blocks=blocks))
        else:
            entries.append(entry)
    return tuple(entries)


def _path(leaf: _Node, by_id: dict[str, _Node], warnings: list[str]) -> list[_Node] | None:
    path: list[_Node] = []
    seen: set[str] = set()
    current: _Node | None = leaf
    while current is not None:
        if current.id in seen:
            warnings.append(f"Branch ending at line {leaf.line} contains a parent cycle.")
            return None
        seen.add(current.id)
        path.append(current)
        if current.parent_id is None:
            break
        parent = by_id.get(current.parent_id)
        if parent is None:
            warnings.append(f"Line {current.line} has a missing parent; the recorded ancestry is incomplete.")
            break
        current = parent
    path.reverse()
    return path


def _cyclic_ancestry(nodes: list[_Node], by_id: dict[str, _Node]) -> set[str]:
    """Find cycles and nodes leading into them in one iterative graph pass."""
    state: dict[str, int] = {}
    invalid: set[str] = set()
    for start in nodes:
        if state.get(start.id) == 2:
            continue
        chain: list[_Node] = []
        positions: dict[str, int] = {}
        current: _Node | None = start
        while current is not None and state.get(current.id, 0) == 0:
            state[current.id] = 1
            positions[current.id] = len(chain)
            chain.append(current)
            current = by_id.get(current.parent_id) if current.parent_id is not None else None
        if current is not None and state.get(current.id) == 1 and current.id in positions:
            invalid.update(node.id for node in chain)
        elif current is not None and current.id in invalid:
            invalid.update(node.id for node in chain)
        for node in chain:
            state[node.id] = 2
    return invalid


def _leaf_label(
    leaf: _Node, by_id: dict[str, _Node], labels: dict[str, str], cache: dict[str, str | None]
) -> str | None:
    chain: list[_Node] = []
    current: _Node | None = leaf
    while current is not None and current.id not in cache:
        chain.append(current)
        current = by_id.get(current.parent_id) if current.parent_id is not None else None
    inherited = cache.get(current.id) if current is not None else None
    for node in reversed(chain):
        inherited = labels.get(node.id, inherited)
        cache[node.id] = inherited
    return cache.get(leaf.id)


def parse_pi_transcript(payload: bytes, session_id: str, branch: str | None = None) -> Transcript:
    """Parse one immutable Pi v3 snapshot without invoking Pi or following references."""
    if not isinstance(session_id, str) or not session_id.strip():
        raise TranscriptUnavailable("The session identity is invalid.", "invalid_source")
    records, source_warnings = read_jsonl(payload)
    if not records or records[0].line != 1 or records[0].value.get("type") != "session":
        raise TranscriptUnavailable("The Pi session header is missing or invalid.", "invalid_source")
    header = records[0].value
    version = header.get("version")
    if type(version) is not int or version != 3:
        raise TranscriptUnavailable("This Pi session version is unsupported.", "unsupported")
    native_id = _text(header.get("id"))
    if native_id is None:
        raise TranscriptUnavailable("The Pi session identity is missing or invalid.", "invalid_source")
    if header.get("cwd") is not None and not isinstance(header.get("cwd"), str):
        raise TranscriptUnavailable("The Pi session working directory is invalid.", "invalid_source")

    warnings = list(source_warnings)
    candidates: list[_Node] = []
    for record in records[1:]:
        raw = record.value
        if raw.get("type") == "session":
            if _text(raw.get("id")) != native_id:
                raise TranscriptUnavailable(
                    "The Pi source contains a conflicting session identity.", "invalid_source"
                )
            warnings.append(f"Line {record.line} contains an extra session header and was skipped.")
            continue
        entry_id = _text(raw.get("id"))
        parent = raw.get("parentId")
        if entry_id is None or (parent is not None and _text(parent) is None):
            warnings.append(f"Line {record.line} has invalid Pi entry identity and was skipped.")
            continue
        candidates.append(_Node(entry_id, _text(parent), record.line, raw))
    if not candidates:
        raise TranscriptUnavailable("No readable transcript entries were recorded.", "invalid_source")

    counts = Counter(node.id for node in candidates)
    for _duplicate in (entry_id for entry_id, count in counts.items() if count > 1):
        warnings.append("A Pi entry id is duplicated; its ancestry is ambiguous.")
    nodes = [node for node in candidates if counts[node.id] == 1]
    by_id = {node.id: node for node in nodes}
    invalid = _cyclic_ancestry(nodes, by_id)
    if invalid:
        warnings.append("The Pi entry graph contains a parent cycle.")
        nodes = [node for node in nodes if node.id not in invalid]
        by_id = {node.id: node for node in nodes}
    for node in nodes:
        if node.parent_id is not None and node.parent_id not in by_id:
            warnings.append(f"Line {node.line} has a missing parent; the recorded ancestry is incomplete.")
    parent_ids = {node.parent_id for node in nodes if node.parent_id in by_id}
    leaves = [node for node in nodes if node.id not in parent_ids]
    if not leaves:
        raise TranscriptUnavailable("No complete recorded Pi branch can be selected.", "invalid_source")
    leaf_ids = {leaf.id for leaf in leaves}
    if branch is not None and branch not in leaf_ids:
        raise TranscriptUnavailable("The requested Pi branch is not a recorded leaf.", "invalid_branch")
    latest_id = max(leaves, key=lambda leaf: leaf.line).id
    selected_id = branch or latest_id
    selected_path = _path(by_id[selected_id], by_id, warnings)
    assert selected_path is not None

    labels: dict[str, str] = {}
    for node in nodes:
        if node.value.get("type") != "label":
            continue
        target = _text(node.value.get("targetId"))
        if target is None:
            warnings.append(f"Line {node.line} has an invalid label target.")
            continue
        label = _text(node.value.get("label"))
        if label is None:
            labels.pop(target, None)
        else:
            labels[target] = label

    branch_records: list[Branch] = []
    label_cache: dict[str, str | None] = {}
    for leaf in leaves:
        label = _leaf_label(leaf, by_id, labels, label_cache)
        if label is None:
            label = "Latest saved branch" if leaf.id == latest_id else f"Branch ending at line {leaf.line}"
        branch_records.append(Branch(leaf.id, label))

    projected: list[Message | Notice | _PendingResult] = []
    known_ids: set[str] = set()
    for node in selected_path:
        projected.extend(_project_node(node, known_ids, warnings))
        known_ids.add(node.id)
    entries = _pair_tools(projected, warnings, skip_phase="retained checkpoint")

    title = "Pi session"
    model: str | None = None
    for node in selected_path:
        if node.value.get("type") == "session_info":
            title = _text(node.value.get("name")) or "Pi session"
        elif node.value.get("type") == "model_change":
            model = _text(node.value.get("modelId"))
        elif node.value.get("type") == "message" and isinstance(node.value.get("message"), dict):
            body = cast(dict[str, object], node.value["message"])
            if body.get("role") == "assistant" and _text(body.get("model")) is not None:
                model = cast(str, body["model"])
    return Transcript(
        session_id, native_id, title, "pi", entries,
        cast(str | None, header.get("cwd")), model, _timestamp(header.get("timestamp")),
        tuple(branch_records), selected_id, tuple(dict.fromkeys(warnings)),
    )


__all__ = ["parse_pi_transcript"]
