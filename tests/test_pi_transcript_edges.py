"""Adversarial Pi transcript and registered-source boundary checks."""

from __future__ import annotations

import base64
import json

import pytest

from harness_usage.pi_transcript import parse_pi_transcript
from harness_usage.transcript import (
    ConflictingOutput,
    ImageBlock,
    Message,
    RecordedOutput,
    TextBlock,
    ToolBlock,
    Transcript,
    TranscriptPage,
    TranscriptUnavailable,
)
from harness_usage.transcript_access import source_snapshot
from harness_usage.transcript_rendering import render_transcript


HEADER = {"type": "session", "version": 3, "id": "native", "cwd": "/work"}


def source(*entries: dict[str, object]) -> bytes:
    return "\n".join(json.dumps(row, ensure_ascii=True) for row in (HEADER, *entries)).encode()


def message(entry_id: str, parent_id: str | None, body: dict[str, object]) -> dict[str, object]:
    return {"type": "message", "id": entry_id, "parentId": parent_id, "message": body}


def calls(transcript: Transcript) -> list[ToolBlock]:
    return [
        block
        for entry in transcript.entries
        if isinstance(entry, Message)
        for block in entry.blocks
        if isinstance(block, ToolBlock)
    ]


@pytest.mark.parametrize(
    "entry",
    [
        message("entry", None, {"role": "user", "content": "lone " + chr(0xD800) + " surrogate"}),
        message(chr(0xD800), None, {"role": "user", "content": "valid text"}),
        {"type": chr(0xD800), "id": "entry", "parentId": None},
        message("entry", None, {"role": chr(0xD800), "content": "valid text"}),
    ],
)
def test_json_surrogates_cannot_escape_as_unencodable_response_text(
    entry: dict[str, object],
) -> None:
    transcript = parse_pi_transcript(source(entry), "pi:ledger")
    rendered = render_transcript(TranscriptPage(transcript))

    rendered.encode("utf-8")


@pytest.mark.parametrize("invalid_zero", [False, 0.0])
def test_bash_exit_status_requires_an_exact_integer(invalid_zero: object) -> None:
    transcript = parse_pi_transcript(
        source(
            message(
                "bash",
                None,
                {"role": "bashExecution", "command": "check", "output": "done", "exitCode": invalid_zero},
            )
        ),
        "pi:ledger",
    )

    output = calls(transcript)[0].output
    assert isinstance(output, RecordedOutput)
    assert output.status == "recorded"
    assert any("exit" in warning.casefold() for warning in transcript.warnings)


@pytest.mark.parametrize("payload", [b"", b"plain text", b"<svg onload=alert(1)></svg>"])
def test_raster_mime_requires_matching_file_signature(payload: bytes) -> None:
    with pytest.raises(ValueError, match="image"):
        ImageBlock("image/png", base64.b64encode(payload).decode())


def test_registered_source_boundary_translates_embedded_nul_to_domain_error(tmp_path) -> None:
    root = tmp_path.resolve()
    locator = str(root) + "/" + chr(0) + ".jsonl"

    for candidate, roots in (
        (locator, (str(root),)),
        (str(root / "session.jsonl"), (str(root) + chr(0),)),
    ):
        with pytest.raises(TranscriptUnavailable) as caught:
            source_snapshot(candidate, roots)
        assert caught.value.kind == "invalid_source"


def test_duplicate_parent_and_cycle_damage_never_join_unrelated_history() -> None:
    transcript = parse_pi_transcript(
        source(
            message("good", None, {"role": "user", "content": "good branch"}),
            message("good-leaf", "good", {"role": "assistant", "content": []}),
            message("duplicate", None, {"role": "user", "content": "first duplicate"}),
            message("duplicate", None, {"role": "user", "content": "second duplicate"}),
            message("orphan-child", "duplicate", {"role": "user", "content": "isolated orphan"}),
            message("cycle", "cycle", {"role": "user", "content": "must disappear"}),
        ),
        "pi:ledger",
        branch="good-leaf",
    )

    assert "good branch" in repr(transcript.entries)
    assert "isolated orphan" not in repr(transcript.entries)
    assert "must disappear" not in repr(transcript)
    assert {branch.id for branch in transcript.branches} == {"good-leaf", "orphan-child"}
    assert any("duplicated" in warning for warning in transcript.warnings)
    assert any("cycle" in warning for warning in transcript.warnings)
    assert any("missing parent" in warning for warning in transcript.warnings)


def test_duplicate_checkpoint_calls_conflict_only_inside_that_checkpoint() -> None:
    retained = [
        {
            "role": "assistant",
            "content": [
                {"type": "toolCall", "id": "same", "name": "read", "arguments": {"path": "one"}},
                {"type": "toolCall", "id": "same", "name": "read", "arguments": {"path": "two"}},
            ],
        },
        {"role": "toolResult", "toolCallId": "same", "toolName": "read", "content": "checkpoint", "isError": False},
    ]
    transcript = parse_pi_transcript(
        source(
            message(
                "main-call",
                None,
                {"role": "assistant", "content": [{"type": "toolCall", "id": "same", "name": "read", "arguments": {}}]},
            ),
            message(
                "main-result",
                "main-call",
                {"role": "toolResult", "toolCallId": "same", "toolName": "read", "content": "main", "isError": False},
            ),
            {
                "type": "compaction",
                "id": "checkpoint",
                "parentId": "main-result",
                "summary": "summary",
                "retainedTail": retained,
            },
        ),
        "pi:ledger",
    )

    tool_calls = calls(transcript)
    assert isinstance(tool_calls[0].output, RecordedOutput)
    assert tool_calls[0].output.blocks == (TextBlock("main"),)
    assert all(isinstance(tool.output, ConflictingOutput) for tool in tool_calls[1:3])
    assert isinstance(tool_calls[3].output, RecordedOutput)
    conflicting_result = next(
        entry for entry in transcript.entries if isinstance(entry, Message) and entry.phase == "conflicting"
    )
    assert conflicting_result.role == "tool"


@pytest.mark.parametrize(
    ("role", "content"),
    [
        (None, None),
        (False, {"unexpected": "object"}),
        (0, ["wrong block", {"type": "text", "text": "readable sibling"}]),
        ("assistant", [None, {"type": "text", "text": "readable sibling"}]),
        ("toolResult", [{"type": "image", "mimeType": "image/png", "data": "%%%"}]),
    ],
)
def test_wrong_shaped_message_variants_are_domain_results_or_renderable(
    role: object, content: object,
) -> None:
    payload = source(message("entry", None, {"role": role, "content": content}))
    try:
        transcript = parse_pi_transcript(payload, "pi:ledger")
    except TranscriptUnavailable:
        return

    render_transcript(TranscriptPage(transcript)).encode("utf-8")
