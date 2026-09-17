from __future__ import annotations

import json
from pathlib import Path
from time import monotonic

import pytest

from harness_usage.codex_transcript import parse_codex_transcript
from harness_usage.transcript import (
    AttachmentBlock,
    ConflictingOutput,
    ImageBlock,
    Message,
    Notice,
    ReasoningBlock,
    RecordedOutput,
    TextBlock,
    ToolBlock,
    TranscriptUnavailable,
)

FIXTURES = Path(__file__).parent / "fixtures" / "transcripts" / "codex"
HEADER = {
    "timestamp": "2026-09-13T08:00:00Z",
    "type": "session_meta",
    "payload": {
        "id": "native-thread",
        "timestamp": "2026-09-13T08:00:00Z",
        "cwd": "/project",
        "history_mode": "legacy",
    },
}


def source(*records: object, header: dict[str, object] = HEADER) -> bytes:
    return ("\n".join(json.dumps(row) for row in (header, *records)) + "\n").encode()


def text(blocks: tuple[object, ...]) -> list[str]:
    return [block.text for block in blocks if isinstance(block, TextBlock)]


def tools(transcript: object) -> list[ToolBlock]:
    return [
        block
        for entry in transcript.entries
        if isinstance(entry, Message)
        for block in entry.blocks
        if isinstance(block, ToolBlock)
    ]


def test_legacy_messages_reconcile_only_one_to_one_mirrors_and_keep_context() -> None:
    transcript = parse_codex_transcript(
        (FIXTURES / "conversation.jsonl").read_bytes(), "ledger-session"
    )

    messages = [entry for entry in transcript.entries if isinstance(entry, Message)]
    user_text = [value for entry in messages if entry.role == "user" for value in text(entry.blocks)]
    assistant_text = [
        value for entry in messages if entry.role == "assistant" for value in text(entry.blocks)
    ]
    assert transcript.session_id == "ledger-session"
    assert transcript.native_id == "native-thread"
    assert transcript.cwd == "/project"
    assert transcript.model == "gpt-5.6-sol"
    assert user_text == ["Repeat me", "Repeat me"]
    assert assistant_text == ["Working", "Done", "Plan item"]
    assert all("unsafe/native:id" not in entry.id for entry in messages)

    inter_agent = next(entry for entry in messages if "Inter-agent note" in text(entry.blocks))
    assert inter_agent.role == "context"
    assert any("researcher" in value and "reviewer" in value for value in text(inter_agent.blocks))


def test_reasoning_compaction_unknown_and_bad_lines_stay_visible_without_payload_leaks() -> None:
    transcript = parse_codex_transcript(
        (FIXTURES / "conversation.jsonl").read_bytes(), "ledger-session"
    )

    reasoning = [
        block
        for entry in transcript.entries
        if isinstance(entry, Message)
        for block in entry.blocks
        if isinstance(block, ReasoningBlock)
    ]
    notices = [entry for entry in transcript.entries if isinstance(entry, Notice)]
    assert [(block.label, block.text) for block in reasoning] == [
        ("Reasoning summary", "Reasoning summary"),
        ("Recorded reasoning", "Recorded details"),
    ]
    assert any(entry.label == "Compaction checkpoint" for entry in notices)
    assert any(entry.label == "Unsupported Codex record" for entry in notices)
    assert "DO NOT APPEND" not in repr(transcript)
    assert "DO-NOT-EXPOSE" not in repr(transcript)
    assert any("18" in warning for warning in transcript.warnings)
    assert any("19" in warning for warning in transcript.warnings)


def test_encrypted_reasoning_is_an_opaque_recorded_placeholder() -> None:
    encrypted = {
        "type": "response_item",
        "payload": {"type": "reasoning", "encrypted_content": "PRIVATE-CIPHERTEXT"},
    }
    transcript = parse_codex_transcript(source(encrypted), "ledger-session")
    block = next(
        block
        for entry in transcript.entries
        if isinstance(entry, Message)
        for block in entry.blocks
        if isinstance(block, ReasoningBlock)
    )
    assert "unavailable" in block.text.lower()
    assert "PRIVATE-CIPHERTEXT" not in repr(transcript)


def test_tools_pair_by_turn_and_call_id_while_preserving_outputs() -> None:
    transcript = parse_codex_transcript(
        (FIXTURES / "tools.jsonl").read_bytes(), "ledger-tools"
    )
    blocks = tools(transcript)
    by_name = {block.name: block for block in blocks}

    assert by_name["alpha"].arguments == "{broken json"
    assert isinstance(by_name["alpha"].output, RecordedOutput)
    assert text(by_name["alpha"].output.blocks) == ["alpha result"]
    assert isinstance(by_name["freeform"].output, RecordedOutput)
    assert text(by_name["freeform"].output.blocks) == ["free result"]
    assert any(isinstance(block, ImageBlock) for block in by_name["freeform"].output.blocks)
    assert any(isinstance(block, AttachmentBlock) for block in by_name["freeform"].output.blocks)
    assert by_name["no_result"].output is None
    assert text(by_name["same_id_new_turn"].output.blocks) == ["new turn result"]

    tool_messages = [
        entry for entry in transcript.entries if isinstance(entry, Message) and entry.role == "tool"
    ]
    assert any("orphan result" in text(entry.blocks) for entry in tool_messages)
    assert any("missing id result" in text(entry.blocks) for entry in tool_messages)
    assert any(
        isinstance(block, AttachmentBlock)
        for entry in tool_messages
        for block in entry.blocks
    )


def test_duplicate_tool_identities_are_conflicts_not_last_write_wins() -> None:
    transcript = parse_codex_transcript(
        (FIXTURES / "tools.jsonl").read_bytes(), "ledger-tools"
    )
    blocks = tools(transcript)
    duplicates = [block for block in blocks if block.call_id == "duplicate-call"]
    repeated_output = next(block for block in blocks if block.call_id == "duplicate-output")
    assert len(duplicates) == 2
    assert all(isinstance(block.output, ConflictingOutput) for block in duplicates)
    assert isinstance(repeated_output.output, ConflictingOutput)
    assert any("duplicate" in warning.lower() for warning in transcript.warnings)


@pytest.mark.parametrize(
    ("header_change", "record", "expected"),
    [
        ({"history_base": {"thread_id": "parent", "end_byte_offset": 10}}, None, "prefix"),
        ({}, {"type": "event_msg", "payload": {"type": "thread_rolled_back"}}, "rollback"),
    ],
)
def test_history_requiring_reconstruction_is_explicitly_unsupported(
    header_change: dict[str, object], record: dict[str, object] | None, expected: str
) -> None:
    header = {**HEADER, "payload": {**HEADER["payload"], **header_change}}
    payload = source(*(record,) if record else (), header=header)
    with pytest.raises(TranscriptUnavailable) as caught:
        parse_codex_transcript(payload, "ledger-session")
    assert caught.value.kind == "unsupported"
    assert expected in caught.value.reason.lower()


def test_exact_internal_turn_item_dialect_is_parsed_or_marked_without_guessing() -> None:
    legacy_output = {
        "type": "event_msg",
        "payload": {
            "type": "item_completed",
            "turn_id": "turn",
            "item": {
                "type": "FunctionCallOutput",
                "call_id": "call",
                "output": "internal result",
            },
        },
    }
    call = {
        "type": "response_item",
        "payload": {"type": "function_call", "call_id": "call", "name": "tool", "arguments": "{}"},
    }
    context = {"type": "turn_context", "payload": {"turn_id": "turn"}}
    public_dialect = {
        "type": "event_msg",
        "payload": {
            "type": "item_completed",
            "item": {"type": "agentMessage", "id": "public", "text": "not legacy"},
        },
    }
    transcript = parse_codex_transcript(source(context, call, legacy_output, public_dialect), "ledger")
    tool = tools(transcript)[0]
    assert isinstance(tool.output, RecordedOutput)
    assert text(tool.output.blocks) == ["internal result"]
    assert any(
        isinstance(entry, Notice)
        and entry.label == "Unsupported Codex item dialect"
        and "agentMessage" in entry.text
        for entry in transcript.entries
    )
    assert "not legacy" not in repr(transcript)


def test_accounting_only_source_is_not_an_empty_success() -> None:
    accounting = {
        "type": "token_usage_record",
        "payload": {"response_id": "response", "usage": {"input_tokens": 1}},
    }
    with pytest.raises(TranscriptUnavailable) as caught:
        parse_codex_transcript(source(accounting), "ledger-session")
    assert caught.value.kind == "unsupported"
    assert "conversation" in caught.value.reason.lower()


def test_known_nonconversation_bookkeeping_does_not_create_warnings() -> None:
    message = {
        "type": "response_item",
        "payload": {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "hello"}],
        },
    }
    transcript = parse_codex_transcript(
        source(
            {"type": "event_msg", "payload": {"type": "task_started"}},
            message,
            {"type": "event_msg", "payload": {"type": "task_complete"}},
            {"type": "world_state", "payload": {"cwd": "/project"}},
        ),
        "ledger",
    )
    assert [entry for entry in transcript.entries if isinstance(entry, Message)]
    assert not transcript.warnings


def test_codex_rejects_branch_selection_without_changing_the_default_view() -> None:
    message = {
        "type": "response_item",
        "payload": {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "hello"}],
        },
    }
    assert parse_codex_transcript(source(message), "ledger", None).entries
    with pytest.raises(TranscriptUnavailable) as caught:
        parse_codex_transcript(source(message), "ledger", "branch-id")
    assert caught.value.kind == "invalid_branch"


@pytest.mark.parametrize("history_mode", ["legacy", "paginated"])
def test_concatenated_codex_sources_with_conflicting_headers_are_rejected(
    history_mode: str,
) -> None:
    header = {**HEADER, "payload": {**HEADER["payload"], "history_mode": history_mode}}
    repeated = {
        "type": "session_meta",
        "payload": {"id": "different-native", "history_mode": history_mode},
    }
    message = {
        "type": "response_item",
        "payload": {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "first source"}],
        },
    }
    with pytest.raises(TranscriptUnavailable) as caught:
        parse_codex_transcript(source(message, repeated, header=header), "ledger")
    assert caught.value.kind == "invalid_source"
    assert "conflicting" in caught.value.reason.lower()


def test_ambiguous_same_turn_mirrors_are_retained_with_a_warning() -> None:
    context = {"type": "turn_context", "payload": {"turn_id": "turn"}}
    raw = {
        "type": "response_item",
        "payload": {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "same"}],
        },
    }
    event = {"type": "event_msg", "payload": {"type": "user_message", "message": "same"}}
    transcript = parse_codex_transcript(source(context, raw, raw, event), "ledger")
    messages = [entry for entry in transcript.entries if isinstance(entry, Message)]
    assert [value for entry in messages for value in text(entry.blocks)] == ["same", "same", "same"]
    assert any("ambiguous" in warning.lower() for warning in transcript.warnings)


def test_opaque_and_invalid_multimodal_outputs_remain_inert_placeholders() -> None:
    context = {"type": "turn_context", "payload": {"turn_id": "turn"}}
    call = {
        "type": "response_item",
        "payload": {"type": "custom_tool_call", "call_id": "media", "name": "media", "input": "free form"},
    }
    result = {
        "type": "response_item",
        "payload": {
            "type": "custom_tool_call_output",
            "call_id": "media",
            "output": [
                {"type": "encrypted_content", "data": "PRIVATE"},
                {"type": "input_image", "image_url": "data:image/svg+xml;base64,PHN2Zz4="},
                {"type": "unknown_media", "value": "PRIVATE-UNKNOWN"},
            ],
        },
    }
    transcript = parse_codex_transcript(source(context, call, result), "ledger")
    tool = tools(transcript)[0]
    assert tool.arguments == "free form"
    assert isinstance(tool.output, RecordedOutput)
    assert tool.output.status == "recorded"
    assert all(isinstance(block, AttachmentBlock) for block in tool.output.blocks)
    assert "PRIVATE" not in repr(transcript)
    assert any("unsupported" in warning.lower() for warning in transcript.warnings)


def test_summary_and_raw_reasoning_events_can_mirror_one_raw_reasoning_record() -> None:
    raw = {
        "type": "response_item",
        "payload": {
            "type": "reasoning",
            "summary": [{"type": "summary_text", "text": "summary"}],
            "content": [{"type": "reasoning_text", "text": "details"}],
        },
    }
    summary = {"type": "event_msg", "payload": {"type": "agent_reasoning", "text": "summary"}}
    details = {
        "type": "event_msg",
        "payload": {"type": "agent_reasoning_raw_content", "text": "details"},
    }
    transcript = parse_codex_transcript(source(raw, summary, details), "ledger")
    reasoning = [
        block
        for entry in transcript.entries
        if isinstance(entry, Message)
        for block in entry.blocks
        if isinstance(block, ReasoningBlock)
    ]
    assert [(block.label, block.text) for block in reasoning] == [
        ("Reasoning summary", "summary"),
        ("Recorded reasoning", "details"),
    ]


def test_large_legacy_session_reconciliation_remains_linear_enough() -> None:
    records: list[object] = []
    for index in range(3_000):
        records.extend(
            (
                {
                    "type": "response_item",
                    "payload": {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": f"message {index}"}],
                    },
                },
                {
                    "type": "event_msg",
                    "payload": {"type": "user_message", "message": f"message {index}"},
                },
                {
                    "type": "response_item",
                    "payload": {
                        "type": "function_call",
                        "call_id": f"call-{index}",
                        "name": "tool",
                        "arguments": "{}",
                    },
                },
                {
                    "type": "response_item",
                    "payload": {
                        "type": "function_call_output",
                        "call_id": f"call-{index}",
                        "output": "ok",
                    },
                },
            )
        )
    started = monotonic()
    transcript = parse_codex_transcript(source(*records), "large-ledger")
    elapsed = monotonic() - started
    assert len(transcript.entries) == 6_000
    assert elapsed < 1.0, f"12,000 legacy records took {elapsed:.2f}s"
