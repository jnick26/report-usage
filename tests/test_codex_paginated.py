from __future__ import annotations

import json
from pathlib import Path

import pytest

from harness_usage.codex_transcript import parse_codex_transcript
from harness_usage.transcript import (
    ConflictingOutput,
    Message,
    Notice,
    ReasoningBlock,
    RecordedOutput,
    TextBlock,
    ToolBlock,
    TranscriptUnavailable,
)

FIXTURE = Path(__file__).parent / "fixtures" / "transcripts" / "codex" / "paginated.jsonl"
HEADER = {
    "type": "session_meta",
    "payload": {"id": "native", "cwd": "/workspace", "history_mode": "paginated"},
}


def source(*records: dict[str, object], header: dict[str, object] = HEADER) -> bytes:
    return ("\n".join(json.dumps(record) for record in (header, *records)) + "\n").encode()


def text(entry: Message) -> str:
    return "".join(block.text for block in entry.blocks if isinstance(block, TextBlock))


def tool_blocks(transcript: object) -> list[ToolBlock]:
    return [
        block
        for entry in transcript.entries
        if isinstance(entry, Message)
        for block in entry.blocks
        if isinstance(block, ToolBlock)
    ]


def output_text(block: ToolBlock) -> list[str]:
    assert isinstance(block.output, RecordedOutput)
    return [item.text for item in block.output.blocks if isinstance(item, TextBlock)]


def test_paginated_projection_keeps_main_requests_and_raw_only_context_without_duplicates() -> None:
    transcript = parse_codex_transcript(FIXTURE.read_bytes(), "ledger")
    messages = [entry for entry in transcript.entries if isinstance(entry, Message)]

    assert transcript.native_id == "native-paginated"
    assert transcript.title == "worker"
    assert [text(entry) for entry in messages if entry.role == "user"] == ["actual request"]
    assert [text(entry) for entry in messages if entry.role == "assistant" and entry.phase == "final_answer"] == ["answer"]
    contexts = {text(entry): entry for entry in messages if entry.role == "context"}
    assert contexts["copied parent prompt"].phase == "inherited"
    assert contexts["model-only context"].phase == "inherited"
    assert "worker → parentinter-agent update" in contexts
    assert any(entry.role == "developer" and text(entry) == "developer context" for entry in messages)

    reasoning = [
        block
        for entry in messages
        for block in entry.blocks
        if isinstance(block, ReasoningBlock)
    ]
    assert [(block.label, block.text) for block in reasoning] == [
        ("Reasoning summary", "summary"),
        ("Recorded reasoning", "details"),
    ]
    assert "PRIVATE" not in repr(transcript)


def test_paginated_item_message_mirror_with_distinct_storage_ids_is_suppressed() -> None:
    transcript = parse_codex_transcript(
        source(
            {
                "type": "turn_context",
                "payload": {"type": "turn_context", "turn_id": "turn-1"},
            },
            {
                "type": "event_msg",
                "payload": {
                    "type": "item_completed",
                    "turn_id": "turn-1",
                    "item": {
                        "type": "AgentMessage",
                        "id": "item-7",
                        "phase": "commentary",
                        "content": [{"type": "Text", "text": "one reply"}],
                    },
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "id": "msg_abc123",
                    "role": "assistant",
                    "phase": "commentary",
                    "content": [{"type": "output_text", "text": "one reply"}],
                },
            },
        ),
        "ledger",
    )

    replies = [
        text(entry)
        for entry in transcript.entries
        if isinstance(entry, Message) and entry.role == "assistant"
    ]
    assert replies == ["one reply"]


def test_paginated_item_message_mirror_rule_preserves_legitimate_repeated_text() -> None:
    def completed(turn_id: str, item_id: str) -> dict[str, object]:
        return {
            "type": "event_msg",
            "payload": {
                "type": "item_completed",
                "turn_id": turn_id,
                "item": {
                    "type": "AgentMessage",
                    "id": item_id,
                    "phase": "final_answer",
                    "content": [{"type": "Text", "text": "same answer"}],
                },
            },
        }

    def raw(message_id: str) -> dict[str, object]:
        return {
            "type": "response_item",
            "payload": {
                "type": "message",
                "id": message_id,
                "role": "assistant",
                "phase": "final_answer",
                "content": [{"type": "output_text", "text": "same answer"}],
            },
        }

    transcript = parse_codex_transcript(
        source(
            {"type": "turn_context", "payload": {"turn_id": "turn-1"}},
            completed("turn-1", "item-1"),
            raw("msg_one"),
            {"type": "turn_context", "payload": {"turn_id": "turn-2"}},
            completed("turn-2", "item-2"),
            raw("msg_two"),
            raw("msg_three"),
        ),
        "ledger",
    )

    replies = [
        text(entry)
        for entry in transcript.entries
        if isinstance(entry, Message) and entry.role == "assistant"
    ]
    assert replies == ["same answer", "same answer", "same answer"]


@pytest.mark.parametrize(
    ("typed_id", "raw_id", "typed_phase", "raw_phase", "raw_text", "between"),
    [
        ("other-1", "msg_one", "commentary", "commentary", "same answer", ()),
        ("item-1", "other-1", "commentary", "commentary", "same answer", ()),
        ("item-1", "msg_one", "commentary", "final_answer", "same answer", ()),
        ("item-1", "msg_one", "commentary", "commentary", "different answer", ()),
        (
            "item-1",
            "msg_one",
            "commentary",
            "commentary",
            "same answer",
            ({"type": "token_usage_record", "payload": {}},),
        ),
    ],
)
def test_paginated_item_message_mirror_rule_requires_exact_dialect_evidence(
    typed_id: str,
    raw_id: str,
    typed_phase: str,
    raw_phase: str,
    raw_text: str,
    between: tuple[dict[str, object], ...],
) -> None:
    transcript = parse_codex_transcript(
        source(
            {"type": "turn_context", "payload": {"turn_id": "turn-1"}},
            {
                "type": "event_msg",
                "payload": {
                    "type": "item_completed",
                    "turn_id": "turn-1",
                    "item": {
                        "type": "AgentMessage",
                        "id": typed_id,
                        "phase": typed_phase,
                        "content": [{"type": "Text", "text": "same answer"}],
                    },
                },
            },
            *between,
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "id": raw_id,
                    "role": "assistant",
                    "phase": raw_phase,
                    "content": [{"type": "output_text", "text": raw_text}],
                },
            },
        ),
        "ledger",
    )

    replies = [
        text(entry)
        for entry in transcript.entries
        if isinstance(entry, Message) and entry.role == "assistant"
    ]
    assert replies == ["same answer", raw_text]


def test_paginated_item_message_mirror_rule_does_not_match_reverse_order() -> None:
    raw = {
        "type": "response_item",
        "payload": {
            "type": "message",
            "id": "msg_one",
            "role": "assistant",
            "phase": "commentary",
            "content": [{"type": "output_text", "text": "same answer"}],
        },
    }
    typed = {
        "type": "event_msg",
        "payload": {
            "type": "item_completed",
            "turn_id": "turn-1",
            "item": {
                "type": "AgentMessage",
                "id": "item-1",
                "phase": "commentary",
                "content": [{"type": "Text", "text": "same answer"}],
            },
        },
    }
    transcript = parse_codex_transcript(
        source(
            {"type": "turn_context", "payload": {"turn_id": "turn-1"}},
            raw,
            typed,
        ),
        "ledger",
    )

    replies = [
        text(entry)
        for entry in transcript.entries
        if isinstance(entry, Message) and entry.role == "assistant"
    ]
    assert replies == ["same answer", "same answer"]


def test_paginated_item_message_with_exact_identity_evidence_is_not_reused() -> None:
    def message(message_id: str) -> dict[str, object]:
        return {
            "type": "response_item",
            "payload": {
                "type": "message",
                "id": message_id,
                "role": "assistant",
                "phase": "commentary",
                "content": [{"type": "output_text", "text": "same answer"}],
            },
        }

    transcript = parse_codex_transcript(
        source(
            {"type": "turn_context", "payload": {"turn_id": "turn-1"}},
            message("item-1"),
            {
                "type": "event_msg",
                "payload": {
                    "type": "item_completed",
                    "turn_id": "turn-1",
                    "item": {
                        "type": "AgentMessage",
                        "id": "item-1",
                        "phase": "commentary",
                        "content": [{"type": "Text", "text": "same answer"}],
                    },
                },
            },
            message("msg_one"),
        ),
        "ledger",
    )

    replies = [
        text(entry)
        for entry in transcript.entries
        if isinstance(entry, Message) and entry.role == "assistant"
    ]
    assert replies == ["same answer", "same answer"]


def test_paginated_conflicting_identity_candidate_is_not_reused_as_storage_mirror() -> None:
    def raw(value: str) -> dict[str, object]:
        return {
            "type": "response_item",
            "payload": {
                "type": "message",
                "id": "msg_shared",
                "role": "assistant",
                "phase": "commentary",
                "content": [{"type": "output_text", "text": value}],
            },
        }

    def typed(item_id: str, value: str) -> dict[str, object]:
        return {
            "type": "event_msg",
            "payload": {
                "type": "item_completed",
                "turn_id": "turn-1",
                "item": {
                    "type": "AgentMessage",
                    "id": item_id,
                    "phase": "commentary",
                    "content": [{"type": "Text", "text": value}],
                },
            },
        }

    transcript = parse_codex_transcript(
        source(
            {"type": "turn_context", "payload": {"turn_id": "turn-1"}},
            typed("item-1", "same answer"),
            raw("same answer"),
            raw("different answer"),
            typed("msg_shared", "typed conflict"),
        ),
        "ledger",
    )

    replies = [
        text(entry)
        for entry in transcript.entries
        if isinstance(entry, Message) and entry.role == "assistant"
    ]
    assert replies == ["same answer", "same answer", "different answer", "typed conflict"]
    assert any("conflict" in warning.lower() for warning in transcript.warnings)


def test_typed_local_request_takes_title_precedence_over_raw_model_context() -> None:
    raw_context = {
        "type": "response_item",
        "payload": {
            "type": "message",
            "id": "context",
            "role": "user",
            "content": [{"type": "input_text", "text": "injected context"}],
        },
    }
    raw_request = {
        "type": "response_item",
        "payload": {
            "type": "message",
            "id": "raw-request",
            "role": "user",
            "content": [{"type": "input_text", "text": "real request"}],
        },
    }
    typed_request = {
        "type": "event_msg",
        "payload": {
            "type": "item_completed",
            "item": {
                "type": "UserMessage",
                "id": "typed-request",
                "content": [{"type": "text", "text": "real request"}],
            },
        },
    }
    transcript = parse_codex_transcript(
        source(raw_context, raw_request, typed_request), "ledger"
    )
    assert transcript.title == "real request"


def test_paginated_tools_use_typed_presentation_and_preserve_raw_only_evidence() -> None:
    transcript = parse_codex_transcript(FIXTURE.read_bytes(), "ledger")
    tools = tool_blocks(transcript)
    assert len(tools) == 4
    by_name = {block.name: block for block in tools}

    command = by_name["exec"]
    assert command.call_id == "call-command"
    assert command.arguments == '{"cmd":"echo hi"}'
    assert output_text(command) == ["hi\n", "raw-only diagnostic"]
    assert command.output.status == "succeeded"

    mcp = by_name["js"]
    assert mcp.call_id == "call-mcp"
    assert mcp.arguments == '{"code":"1+1"}'
    assert output_text(mcp) == ["2"]

    subagent = by_name["spawn_agent"]
    assert subagent.call_id == "call-subagent"
    assert "task_name" in subagent.arguments
    assert output_text(subagent) == ["spawned"]

    raw_only = by_name["standalone"]
    assert raw_only.arguments == "raw args"
    assert output_text(raw_only) == ["raw result"]
    assert sum(block.call_id == "call-command" for block in tools) == 1
    assert sum(block.call_id == "call-mcp" for block in tools) == 1


def test_paginated_plan_compaction_and_unknown_typed_items_are_visible_and_bounded() -> None:
    transcript = parse_codex_transcript(FIXTURE.read_bytes(), "ledger")
    assert any(isinstance(entry, Message) and entry.phase == "plan" and text(entry) == "plan text" for entry in transcript.entries)
    notices = [entry for entry in transcript.entries if isinstance(entry, Notice)]
    assert any(entry.label == "Compaction checkpoint" for entry in notices)
    assert any("FutureThing" in entry.text for entry in notices)
    assert "DO-NOT-EXPOSE" not in repr(transcript)

    secret = "PRIVATE-TYPE-" + "x" * 10_000
    bad = {
        "type": "event_msg",
        "payload": {"type": "item_completed", "item": {"type": secret}},
    }
    safe = parse_codex_transcript(source(bad), "ledger")
    assert max(len(entry.text) for entry in safe.entries if isinstance(entry, Notice)) < 500
    assert secret not in repr(safe)


def test_paginated_duplicate_identities_are_retained_as_conflicts() -> None:
    raw = {
        "type": "response_item",
        "payload": {
            "type": "message",
            "id": "same-message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "raw answer"}],
        },
    }
    typed_one = {
        "type": "event_msg",
        "payload": {
            "type": "item_completed",
            "item": {"type": "AgentMessage", "id": "same-message", "content": [{"type": "Text", "text": "one"}]},
        },
    }
    typed_two = {
        "type": "event_msg",
        "payload": {
            "type": "item_completed",
            "item": {"type": "AgentMessage", "id": "same-message", "content": [{"type": "Text", "text": "two"}]},
        },
    }
    tool_one = {
        "type": "event_msg",
        "payload": {
            "type": "item_completed",
            "item": {"type": "CommandExecution", "id": "same-tool", "command": ["one"], "cwd": "/", "parsed_cmd": [], "source": "agent", "status": "completed", "stdout": "one"},
        },
    }
    tool_two = {
        "type": "event_msg",
        "payload": {
            "type": "item_completed",
            "item": {"type": "CommandExecution", "id": "same-tool", "command": ["two"], "cwd": "/", "parsed_cmd": [], "source": "agent", "status": "completed", "stdout": "two"},
        },
    }
    transcript = parse_codex_transcript(source(raw, typed_one, typed_two, tool_one, tool_two), "ledger")
    assistant = [
        text(entry)
        for entry in transcript.entries
        if isinstance(entry, Message)
        and entry.role == "assistant"
        and any(isinstance(block, TextBlock) for block in entry.blocks)
        and not any(isinstance(block, ToolBlock) for block in entry.blocks)
    ]
    assert assistant == ["raw answer", "one", "two"]
    duplicates = [block for block in tool_blocks(transcript) if block.call_id == "same-tool"]
    assert len(duplicates) == 2
    assert all(isinstance(block.output, ConflictingOutput) for block in duplicates)
    assert "one" in repr(transcript) and "two" in repr(transcript)
    assert any("duplicate" in warning.lower() for warning in transcript.warnings)


def test_conflicting_raw_and_typed_statuses_preserve_both_output_bodies() -> None:
    raw_call = {
        "type": "response_item",
        "payload": {
            "type": "custom_tool_call",
            "call_id": "call",
            "name": "exec",
            "input": "true",
        },
    }
    raw_output = {
        "type": "response_item",
        "payload": {
            "type": "custom_tool_call_output",
            "call_id": "call",
            "output": "raw success body",
            "status": "succeeded",
        },
    }
    typed = {
        "type": "event_msg",
        "payload": {
            "type": "item_completed",
            "item": {
                "type": "CommandExecution",
                "id": "typed",
                "command": ["true"],
                "cwd": "/",
                "parsed_cmd": [],
                "source": "agent",
                "status": "failed",
                "aggregated_output": "typed failure body",
            },
        },
    }
    transcript = parse_codex_transcript(source(raw_call, raw_output, typed), "ledger")
    tool = tool_blocks(transcript)[0]
    assert isinstance(tool.output, ConflictingOutput)
    assert "raw success body" in repr(transcript)
    assert "typed failure body" in repr(transcript)


def test_command_prefix_similarity_is_not_enough_to_merge_tools() -> None:
    raw_call = {
        "type": "response_item",
        "payload": {
            "type": "custom_tool_call",
            "call_id": "raw",
            "name": "exec",
            "input": '{"cmd":"echo one-more"}',
        },
    }
    raw_output = {
        "type": "response_item",
        "payload": {
            "type": "custom_tool_call_output",
            "call_id": "raw",
            "output": "one-more",
        },
    }
    typed = {
        "type": "event_msg",
        "payload": {
            "type": "item_completed",
            "item": {
                "type": "CommandExecution",
                "id": "typed",
                "command": ["echo one"],
                "cwd": "/",
                "parsed_cmd": [],
                "source": "agent",
                "status": "completed",
                "aggregated_output": "one",
            },
        },
    }
    transcript = parse_codex_transcript(source(raw_call, raw_output, typed), "ledger")
    assert len(tool_blocks(transcript)) == 2


def test_same_arguments_do_not_merge_different_dynamic_tool_names() -> None:
    raw_call = {
        "type": "response_item",
        "payload": {
            "type": "function_call",
            "call_id": "raw",
            "name": "customer_lookup",
            "arguments": '{"query":"same"}',
        },
    }
    raw_output = {
        "type": "response_item",
        "payload": {
            "type": "function_call_output",
            "call_id": "raw",
            "output": "customer",
        },
    }
    typed = {
        "type": "event_msg",
        "payload": {
            "type": "item_completed",
            "item": {
                "type": "DynamicToolCall",
                "id": "typed",
                "tool": "order_lookup",
                "arguments": {"query": "same"},
                "status": "completed",
                "content_items": [{"type": "text", "text": "order"}],
                "success": True,
            },
        },
    }
    transcript = parse_codex_transcript(source(raw_call, raw_output, typed), "ledger")
    assert {block.name for block in tool_blocks(transcript)} == {
        "customer_lookup",
        "order_lookup",
    }


def test_command_matching_preserves_significant_inner_whitespace() -> None:
    raw_call = {
        "type": "response_item",
        "payload": {
            "type": "custom_tool_call",
            "call_id": "raw",
            "name": "exec",
            "input": '{"cmd":"echo a b"}',
        },
    }
    raw_output = {
        "type": "response_item",
        "payload": {
            "type": "custom_tool_call_output",
            "call_id": "raw",
            "output": "a b",
        },
    }
    typed = {
        "type": "event_msg",
        "payload": {
            "type": "item_completed",
            "item": {
                "type": "CommandExecution",
                "id": "typed",
                "command": ["echo a  b"],
                "cwd": "/",
                "parsed_cmd": [],
                "source": "agent",
                "status": "completed",
                "aggregated_output": "a  b",
            },
        },
    }
    transcript = parse_codex_transcript(source(raw_call, raw_output, typed), "ledger")
    assert len(tool_blocks(transcript)) == 2


@pytest.mark.parametrize(
    "record",
    [
        {"type": "event_msg", "payload": {"type": "thread_rolled_back"}},
        {"type": "thread_rolled_back", "payload": {}},
    ],
)
def test_paginated_rollback_remains_unsupported(record: dict[str, object]) -> None:
    with pytest.raises(TranscriptUnavailable) as caught:
        parse_codex_transcript(source(record), "ledger")
    assert caught.value.kind == "unsupported"
    assert "rollback" in caught.value.reason.lower()


def test_paginated_prefix_and_accounting_only_sources_are_not_false_successes() -> None:
    prefix_header = {
        **HEADER,
        "payload": {**HEADER["payload"], "history_base": {"thread_id": "parent", "end_byte_offset": 10}},
    }
    with pytest.raises(TranscriptUnavailable) as prefix:
        parse_codex_transcript(source(header=prefix_header), "ledger")
    assert prefix.value.kind == "unsupported"
    assert "prefix" in prefix.value.reason.lower()

    accounting = {"type": "token_usage_record", "payload": {"usage": {"input_tokens": 1}}}
    with pytest.raises(TranscriptUnavailable) as empty:
        parse_codex_transcript(source(accounting), "ledger")
    assert empty.value.kind == "unsupported"
    assert "conversation" in empty.value.reason.lower()
