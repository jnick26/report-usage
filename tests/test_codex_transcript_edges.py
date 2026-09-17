"""Independent adversarial checks for the supported Codex legacy projection."""

from __future__ import annotations

import json

import pytest

from harness_usage.codex_transcript import parse_codex_transcript
from harness_usage.transcript import Message, Notice, ReasoningBlock, TextBlock, TranscriptUnavailable


HEADER = {
    "type": "session_meta",
    "timestamp": "2026-09-13T08:00:00Z",
    "payload": {
        "id": "native",
        "cwd": "/workspace",
        "history_mode": "legacy",
    },
}


def source(*records: dict[str, object], header: dict[str, object] = HEADER) -> bytes:
    return ("\n".join(json.dumps(record) for record in (header, *records)) + "\n").encode()


def response_message(
    text: str, *, role: object = "user", ordinal: int | None = None,
    metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    record: dict[str, object] = {
        "type": "response_item",
        "payload": {
            "type": "message",
            "role": role,
            "content": [{"type": "input_text", "text": text}],
        },
    }
    if ordinal is not None:
        record["ordinal"] = ordinal
    if metadata is not None:
        record["metadata"] = metadata
    return record


def message_text(entry: Message) -> str:
    return "".join(block.text for block in entry.blocks if isinstance(block, TextBlock))


def test_summary_and_raw_reasoning_events_can_mirror_one_raw_reasoning_item() -> None:
    raw = {
        "type": "response_item",
        "payload": {
            "type": "reasoning",
            "summary": [{"type": "summary_text", "text": "summary"}],
            "content": [{"type": "reasoning_text", "text": "details"}],
        },
    }
    summary_event = {"type": "event_msg", "payload": {"type": "agent_reasoning", "text": "summary"}}
    raw_event = {"type": "event_msg", "payload": {"type": "agent_reasoning_raw_content", "text": "details"}}
    transcript = parse_codex_transcript(source(raw, summary_event, raw_event), "ledger")
    reasoning_messages = [
        entry
        for entry in transcript.entries
        if isinstance(entry, Message)
        and any(isinstance(block, ReasoningBlock) for block in entry.blocks)
    ]
    assert len(reasoning_messages) == 1
    assert [(block.label, block.text) for block in reasoning_messages[0].blocks] == [
        ("Reasoning summary", "summary"),
        ("Recorded reasoning", "details"),
    ]


def test_subagent_inherited_messages_are_not_presented_as_fresh_user_intent() -> None:
    header = {
        **HEADER,
        "payload": {
            **HEADER["payload"],
            "parent_thread_id": "parent",
            "subagent_history_start_ordinal": 2,
        },
    }
    transcript = parse_codex_transcript(
        source(
            response_message("copied parent prompt", ordinal=1),
            response_message("child request", ordinal=2),
            response_message(
                "explicitly inherited prompt",
                ordinal=3,
                metadata={"inherited_user_message": True},
            ),
            header=header,
        ),
        "ledger",
    )
    messages = {message_text(entry): entry for entry in transcript.entries if isinstance(entry, Message)}
    assert messages["child request"].role == "user"
    for text in ("copied parent prompt", "explicitly inherited prompt"):
        assert messages[text].role == "context"
        assert messages[text].phase is not None and "inherited" in messages[text].phase


def test_malformed_discriminants_are_bounded_markers_and_do_not_echo_opaque_payloads() -> None:
    secret = "PRIVATE-OPAQUE-" + "x" * 10_000
    transcript = parse_codex_transcript(
        source(
            response_message("", role={"opaque": secret}),
            {"type": secret, "payload": {}},
            {"type": "event_msg", "payload": {"type": secret}},
        ),
        "ledger",
    )
    notices = [entry for entry in transcript.entries if isinstance(entry, Notice)]
    assert notices
    assert max(len(entry.text) for entry in notices) < 500
    assert max(len(warning) for warning in transcript.warnings) < 500
    assert secret not in repr(transcript)


@pytest.mark.parametrize(
    "bad_mode",
    [
        {"opaque": "PRIVATE-HISTORY-MODE"},
        ["PRIVATE-HISTORY-MODE"],
        "PRIVATE-HISTORY-MODE" + "x" * 10_000,
    ],
)
def test_invalid_history_mode_returns_a_short_safe_boundary_error(bad_mode: object) -> None:
    header = {**HEADER, "payload": {**HEADER["payload"], "history_mode": bad_mode}}
    with pytest.raises(TranscriptUnavailable) as caught:
        parse_codex_transcript(source(header=header), "ledger")
    assert caught.value.kind in ("unsupported", "invalid_source")
    assert len(caught.value.reason) < 500
    assert "PRIVATE-HISTORY-MODE" not in caught.value.reason
