"""Independent edge checks for the Codex paginated stored-record projection."""

from __future__ import annotations

import json
from typing import cast

import pytest

from harness_usage.codex_transcript import parse_codex_transcript
from harness_usage.transcript import (
    ConflictingOutput,
    Message,
    Notice,
    RecordedOutput,
    ReasoningBlock,
    TextBlock,
    ToolBlock,
    Transcript,
    TranscriptPage,
    TranscriptUnavailable,
)
from harness_usage.transcript_rendering import render_transcript


HEADER: dict[str, object] = {
    "type": "session_meta",
    "timestamp": "2026-09-13T08:00:00Z",
    "payload": {
        "id": "native-paginated",
        "cwd": "/workspace",
        "history_mode": "paginated",
    },
}


def source(*records: dict[str, object], header: dict[str, object] = HEADER) -> bytes:
    return ("\n".join(json.dumps(record) for record in (header, *records)) + "\n").encode()


def turn(turn_id: str) -> dict[str, object]:
    return {"type": "turn_context", "payload": {"turn_id": turn_id}}


def completed(
    turn_id: str, item: object, *, ordinal: int | None = None
) -> dict[str, object]:
    record: dict[str, object] = {
        "type": "event_msg",
        "payload": {
            "type": "item_completed",
            "thread_id": "native-paginated",
            "turn_id": turn_id,
            "item": item,
            "completed_at_ms": 1,
        },
    }
    if ordinal is not None:
        record["ordinal"] = ordinal
    return record


def raw_message(
    role: str, text: str, item_id: str, *, ordinal: int | None = None,
    inherited: bool = False,
) -> dict[str, object]:
    record: dict[str, object] = {
        "type": "response_item",
        "payload": {
            "type": "message",
            "id": item_id,
            "role": role,
            "content": [
                {
                    "type": "input_text" if role == "user" else "output_text",
                    "text": text,
                }
            ],
        },
    }
    if ordinal is not None:
        record["ordinal"] = ordinal
    if inherited:
        record["metadata"] = {"inherited_user_message": True}
    return record


def messages(transcript: Transcript) -> list[Message]:
    return [entry for entry in transcript.entries if isinstance(entry, Message)]


def texts(message: Message) -> list[str]:
    return [
        block.text
        for block in message.blocks
        if isinstance(block, (TextBlock, ReasoningBlock))
    ]


def tools(transcript: Transcript) -> list[ToolBlock]:
    return [
        block
        for message in messages(transcript)
        for block in message.blocks
        if isinstance(block, ToolBlock)
    ]


def test_shared_agent_and_reasoning_ids_suppress_paginated_typed_mirrors() -> None:
    raw_reasoning = {
        "type": "response_item",
        "payload": {
            "type": "reasoning",
            "id": "reasoning-shared",
            "summary": [{"type": "summary_text", "text": "summary"}],
            "content": [{"type": "reasoning_text", "text": "details"}],
        },
    }
    transcript = parse_codex_transcript(
        source(
            turn("turn-1"),
            raw_message("assistant", "answer", "assistant-shared"),
            completed(
                "turn-1",
                {
                    "type": "AgentMessage",
                    "id": "assistant-shared",
                    "content": [{"type": "Text", "text": "answer"}],
                    "phase": "final_answer",
                },
            ),
            raw_reasoning,
            completed(
                "turn-1",
                {
                    "type": "Reasoning",
                    "id": "reasoning-shared",
                    "summary_text": ["summary"],
                    "raw_content": ["details"],
                },
            ),
        ),
        "ledger",
    )

    assert sum("answer" in texts(message) for message in messages(transcript)) == 1
    reasoning = [
        message
        for message in messages(transcript)
        if any(isinstance(block, ReasoningBlock) for block in message.blocks)
    ]
    assert len(reasoning) == 1
    assert texts(reasoning[0]) == ["summary", "details"]


def test_shared_assistant_id_with_different_content_is_reported_as_conflicting() -> None:
    transcript = parse_codex_transcript(
        source(
            turn("turn-1"),
            raw_message("assistant", "raw answer", "assistant-shared"),
            completed(
                "turn-1",
                {
                    "type": "AgentMessage",
                    "id": "assistant-shared",
                    "content": [{"type": "Text", "text": "different typed answer"}],
                },
            ),
        ),
        "ledger",
    )

    rendered_text = [text for message in messages(transcript) for text in texts(message)]
    assert "raw answer" in rendered_text
    assert "different typed answer" in rendered_text
    assert any("conflict" in warning.lower() for warning in transcript.warnings)


def test_typed_assistant_without_phase_keeps_matching_raw_phase() -> None:
    raw = raw_message("assistant", "commentary", "assistant-shared")
    cast(dict[str, object], raw["payload"])["phase"] = "commentary"
    transcript = parse_codex_transcript(
        source(
            turn("turn-1"),
            raw,
            completed(
                "turn-1",
                {
                    "type": "AgentMessage",
                    "id": "assistant-shared",
                    "content": [{"type": "Text", "text": "commentary"}],
                },
            ),
        ),
        "ledger",
    )

    matching = [message for message in messages(transcript) if "commentary" in texts(message)]
    assert len(matching) == 1
    assert matching[0].phase == "commentary"


def test_distinct_typed_user_ids_match_raw_messages_one_to_one_per_turn() -> None:
    transcript = parse_codex_transcript(
        source(
            turn("turn-1"),
            raw_message("user", "repeatable request", "raw-user-1"),
            completed(
                "turn-1",
                {
                    "type": "UserMessage",
                    "id": "typed-user-1",
                    "content": [
                        {"type": "text", "text": "repeatable request", "text_elements": []}
                    ],
                },
            ),
            turn("turn-2"),
            raw_message("user", "repeatable request", "raw-user-2"),
            completed(
                "turn-2",
                {
                    "type": "UserMessage",
                    "id": "typed-user-2",
                    "content": [
                        {"type": "text", "text": "repeatable request", "text_elements": []}
                    ],
                },
            ),
        ),
        "ledger",
    )

    requests = [
        message
        for message in messages(transcript)
        if message.role == "user" and "repeatable request" in texts(message)
    ]
    assert len(requests) == 2


def test_command_execution_with_a_distinct_typed_id_does_not_duplicate_raw_call() -> None:
    raw_call = {
        "type": "response_item",
        "payload": {
            "type": "custom_tool_call",
            "id": "raw-function-item",
            "call_id": "raw-call-id",
            "name": "exec",
            "input": '{"cmd":"echo done"}',
        },
    }
    raw_output = {
        "type": "response_item",
        "payload": {
            "type": "custom_tool_call_output",
            "call_id": "raw-call-id",
            "output": "done\n",
        },
    }
    transcript = parse_codex_transcript(
        source(
            turn("turn-1"),
            raw_call,
            raw_output,
            completed(
                "turn-1",
                {
                    "type": "CommandExecution",
                    "id": "typed-command-id",
                    "command": ["echo", "done"],
                    "cwd": "/workspace",
                    "parsed_cmd": [{"type": "unknown", "cmd": "echo done"}],
                    "source": "agent",
                    "status": "completed",
                    "stdout": "done\n",
                    "stderr": "",
                    "aggregated_output": "done\n",
                    "exit_code": 0,
                },
            ),
        ),
        "ledger",
    )

    projected = tools(transcript)
    assert len(projected) == 1
    assert projected[0].output is not None
    assert "done" in repr(projected[0].output)


def test_completed_command_with_known_empty_output_is_not_missing_output() -> None:
    transcript = parse_codex_transcript(
        source(
            completed(
                "turn-1",
                {
                    "type": "CommandExecution",
                    "id": "empty-command",
                    "command": ["true"],
                    "cwd": "/workspace",
                    "parsed_cmd": [],
                    "source": "agent",
                    "status": "completed",
                    "stdout": "",
                    "stderr": "",
                    "aggregated_output": "",
                    "exit_code": 0,
                },
            )
        ),
        "ledger",
    )

    output = tools(transcript)[0].output
    assert isinstance(output, RecordedOutput)
    assert output.blocks == ()
    assert output.status == "succeeded"


def test_dynamic_tool_explicit_success_is_succeeded_when_status_is_missing() -> None:
    transcript = parse_codex_transcript(
        source(
            completed(
                "turn-1",
                {
                    "type": "DynamicToolCall",
                    "id": "dynamic",
                    "tool": "lookup",
                    "arguments": {"query": "value"},
                    "content_items": [{"type": "text", "text": "found"}],
                    "success": True,
                },
            )
        ),
        "ledger",
    )

    output = tools(transcript)[0].output
    assert isinstance(output, RecordedOutput)
    assert output.status == "succeeded"


def test_mcp_result_error_flag_overrides_completed_lifecycle_status() -> None:
    transcript = parse_codex_transcript(
        source(
            completed(
                "turn-1",
                {
                    "type": "McpToolCall",
                    "id": "mcp",
                    "server": "server",
                    "tool": "lookup",
                    "arguments": {},
                    "status": "completed",
                    "result": {
                        "content": [{"type": "text", "text": "tool failed"}],
                        "isError": True,
                    },
                },
            )
        ),
        "ledger",
    )

    output = tools(transcript)[0].output
    assert isinstance(output, RecordedOutput)
    assert output.status == "failed"


def test_contradictory_raw_and_typed_tool_status_is_explicitly_conflicting() -> None:
    transcript = parse_codex_transcript(
        source(
            turn("turn-1"),
            {
                "type": "response_item",
                "payload": {
                    "type": "custom_tool_call",
                    "call_id": "raw-call",
                    "name": "exec",
                    "input": "true",
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "custom_tool_call_output",
                    "call_id": "raw-call",
                    "output": "raw success",
                    "status": "succeeded",
                },
            },
            completed(
                "turn-1",
                {
                    "type": "CommandExecution",
                    "id": "typed-command",
                    "command": ["true"],
                    "cwd": "/workspace",
                    "parsed_cmd": [],
                    "source": "agent",
                    "status": "failed",
                    "aggregated_output": "typed failure",
                    "exit_code": 1,
                },
            ),
        ),
        "ledger",
    )

    assert isinstance(tools(transcript)[0].output, ConflictingOutput)


def test_adjacent_semantically_unrelated_raw_and_typed_tools_are_not_merged() -> None:
    transcript = parse_codex_transcript(
        source(
            turn("turn-1"),
            {
                "type": "response_item",
                "payload": {
                    "type": "custom_tool_call",
                    "call_id": "raw-call",
                    "name": "database_lookup",
                    "input": '{"query":"customer"}',
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "custom_tool_call_output",
                    "call_id": "raw-call",
                    "output": "customer record",
                },
            },
            completed(
                "turn-1",
                {
                    "type": "CommandExecution",
                    "id": "typed-command",
                    "command": ["echo", "safe"],
                    "cwd": "/workspace",
                    "parsed_cmd": [],
                    "source": "agent",
                    "status": "completed",
                    "aggregated_output": "safe\n",
                    "exit_code": 0,
                },
            ),
        ),
        "ledger",
    )

    projected = tools(transcript)
    assert len(projected) == 2
    assert {tool.name for tool in projected} == {"database_lookup", "command_execution"}
    assert any("customer record" in repr(tool.output) for tool in projected)
    assert any("safe" in repr(tool.output) for tool in projected)


def test_adjacent_exec_tools_with_different_commands_are_not_merged() -> None:
    transcript = parse_codex_transcript(
        source(
            turn("turn-1"),
            {
                "type": "response_item",
                "payload": {
                    "type": "custom_tool_call",
                    "call_id": "raw-call",
                    "name": "exec",
                    "input": '{"cmd":"echo one"}',
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "custom_tool_call_output",
                    "call_id": "raw-call",
                    "output": "one\n",
                },
            },
            completed(
                "turn-1",
                {
                    "type": "CommandExecution",
                    "id": "typed-command",
                    "command": ["echo", "two"],
                    "cwd": "/workspace",
                    "parsed_cmd": [],
                    "source": "agent",
                    "status": "completed",
                    "aggregated_output": "two\n",
                    "exit_code": 0,
                },
            ),
        ),
        "ledger",
    )

    projected = tools(transcript)
    assert len(projected) == 2
    assert {tool.name for tool in projected} == {"exec", "command_execution"}
    assert any("one" in tool.arguments for tool in projected)
    assert any("two" in tool.arguments for tool in projected)


def test_typed_function_output_does_not_duplicate_matching_raw_result() -> None:
    transcript = parse_codex_transcript(
        source(
            turn("turn-1"),
            {
                "type": "response_item",
                "payload": {
                    "type": "function_call",
                    "call_id": "call-1",
                    "name": "lookup",
                    "arguments": "{}",
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "function_call_output",
                    "id": "raw-output-id",
                    "call_id": "call-1",
                    "output": "done",
                },
            },
            completed(
                "turn-1",
                {
                    "type": "FunctionCallOutput",
                    "id": "call-1",
                    "name": "lookup",
                    "output": "done",
                },
            ),
        ),
        "ledger",
    )

    projected = tools(transcript)
    assert len(projected) == 1
    assert "done" in repr(projected[0].output)
    assert not [message for message in messages(transcript) if message.role == "tool"]


def test_paginated_inherited_cutoff_labels_copied_raw_history_as_context() -> None:
    header = {
        **HEADER,
        "payload": {
            **cast(dict[str, object], HEADER["payload"]),
            "parent_thread_id": "parent",
            "subagent_history_start_ordinal": 3,
        },
    }
    transcript = parse_codex_transcript(
        source(
            raw_message("user", "parent request", "raw-inherited", ordinal=1),
            turn("turn-child"),
            raw_message("user", "child request", "raw-local", ordinal=4),
            completed(
                "turn-child",
                {
                    "type": "UserMessage",
                    "id": "typed-local",
                    "content": [{"type": "text", "text": "child request", "text_elements": []}],
                },
                ordinal=5,
            ),
            header=header,
        ),
        "ledger",
    )

    by_text = {text: message for message in messages(transcript) for text in texts(message)}
    assert by_text["parent request"].role == "context"
    assert by_text["parent request"].phase == "inherited"
    assert by_text["child request"].role == "user"


def test_inherited_parent_prompt_does_not_become_paginated_session_title() -> None:
    header = {
        **HEADER,
        "payload": {
            **cast(dict[str, object], HEADER["payload"]),
            "subagent_history_start_ordinal": 2,
        },
    }
    transcript = parse_codex_transcript(
        source(
            raw_message("user", "copied parent prompt", "parent", ordinal=1),
            completed(
                "turn-child",
                {
                    "type": "UserMessage",
                    "id": "typed-local",
                    "content": [{"type": "text", "text": "actual child request"}],
                },
                ordinal=2,
            ),
            header=header,
        ),
        "ledger",
    )

    assert transcript.title == "actual child request"


@pytest.mark.parametrize(
    "item",
    [
        {"type": "UserMessage", "id": "user", "content": {"text": "wrong"}},
        {"type": "AgentMessage", "id": {"bad": "id"}, "content": "wrong"},
        {
            "type": "Reasoning",
            "id": "reasoning",
            "summary_text": {"text": "wrong"},
            "raw_content": [1, {"text": "wrong"}],
        },
        {
            "type": "CommandExecution",
            "id": {"bad": "id"},
            "command": {"cmd": "wrong"},
            "cwd": {"path": "/workspace"},
            "parsed_cmd": "wrong",
            "source": ["agent"],
            "status": {"bad": "status"},
            "aggregated_output": ["wrong"],
            "exit_code": False,
        },
        ["not", "an", "item"],
    ],
)
def test_malformed_paginated_items_remain_renderable_and_utf8_safe(item: object) -> None:
    transcript = parse_codex_transcript(
        source(
            raw_message("user", "healthy sibling", "healthy"),
            completed("turn-1", item),
        ),
        "ledger",
    )

    rendered = render_transcript(TranscriptPage(transcript))
    assert "healthy sibling" in rendered
    rendered.encode("utf-8")


@pytest.mark.parametrize(
    ("header_change", "record", "expected"),
    [
        (
            {"history_base": {"thread_id": "parent", "end_byte_offset": 10}},
            None,
            "prefix",
        ),
        (
            {},
            {"type": "event_msg", "payload": {"type": "thread_rolled_back"}},
            "rollback",
        ),
    ],
)
def test_paginated_prefix_and_rollback_remain_explicitly_unsupported(
    header_change: dict[str, object], record: dict[str, object] | None, expected: str
) -> None:
    header = {
        **HEADER,
        "payload": {**cast(dict[str, object], HEADER["payload"]), **header_change},
    }
    with pytest.raises(TranscriptUnavailable) as caught:
        parse_codex_transcript(source(*(record,) if record else (), header=header), "ledger")
    assert caught.value.kind == "unsupported"
    assert expected in caught.value.reason.lower()


def test_paginated_projection_discloses_that_it_is_a_stored_record_view() -> None:
    transcript = parse_codex_transcript(
        source(raw_message("user", "visible request", "raw-user")), "ledger"
    )
    labels = [entry.label for entry in transcript.entries if isinstance(entry, Notice)]
    disclosure = " ".join((*labels, *transcript.warnings)).lower()
    assert "stored-record" in disclosure or "stored record" in disclosure
