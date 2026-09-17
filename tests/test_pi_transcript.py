from pathlib import Path
import json

import pytest

from harness_usage.pi_transcript import parse_pi_transcript
from harness_usage.transcript import (
    AttachmentBlock,
    ConflictingOutput,
    Message,
    Notice,
    ReasoningBlock,
    RecordedOutput,
    TextBlock,
    ToolBlock,
    TranscriptUnavailable,
)


FIXTURES = Path(__file__).parent / "fixtures/transcripts/pi"


def parse(name: str, *, branch: str | None = None):
    return parse_pi_transcript((FIXTURES / name).read_bytes(), "pi:ledger", branch)


def test_preserves_ordered_content_and_pairs_reverse_tool_results() -> None:
    transcript = parse("core.jsonl")
    assistant = next(entry for entry in transcript.entries if isinstance(entry, Message) and entry.role == "assistant")
    assert [type(block) for block in assistant.blocks] == [
        TextBlock,
        ReasoningBlock,
        ToolBlock,
        TextBlock,
        ToolBlock,
    ]
    read, run = (block for block in assistant.blocks if isinstance(block, ToolBlock))
    assert read.arguments == '{"path":"notes.txt"}'
    assert read.output == RecordedOutput((TextBlock("notes"),), "failed", 5)
    assert run.output == RecordedOutput((TextBlock("done"),), "succeeded", 4)
    assert "SECRET-THINKING" not in repr(transcript)
    assert "SECRET-TOOL" not in repr(transcript)
    assert "SECRET-EMPTY" not in repr(transcript)
    assert any(isinstance(entry, Notice) and entry.label == "Context compacted" for entry in transcript.entries)
    assert any(isinstance(entry, Notice) and entry.label == "Unsupported Pi entry" for entry in transcript.entries)
    assert "SECRET-UNKNOWN" not in repr(transcript)


def test_follows_metadata_nodes_and_scopes_same_call_id_to_selected_branch() -> None:
    latest = parse("branches.jsonl")
    assert latest.selected_branch == "leaf-b"
    assert {branch.id: branch.label for branch in latest.branches} == {
        "leaf-a": "Approach A",
        "leaf-b": "Approach B",
    }
    assert not any(isinstance(entry, Message) and entry.model == "a" for entry in latest.entries)
    selected = parse("branches.jsonl", branch="leaf-a")
    assert any(isinstance(entry, Notice) and entry.label == "Model changed" for entry in selected.entries)
    call = next(block for entry in selected.entries if isinstance(entry, Message) for block in entry.blocks if isinstance(block, ToolBlock))
    assert call.name == "read"
    assert isinstance(call.output, RecordedOutput)
    assert call.output.blocks == (TextBlock("A"),)


def test_latest_saved_label_stays_on_physical_latest_leaf_when_selecting_an_older_branch() -> None:
    payload = b"\n".join(
        [
            b'{"type":"session","version":3,"id":"native","cwd":"/w"}',
            b'{"type":"message","id":"root","parentId":null,"message":{"role":"user","content":"root"}}',
            b'{"type":"message","id":"older","parentId":"root","message":{"role":"assistant","content":[]}}',
            b'{"type":"message","id":"latest","parentId":"root","message":{"role":"assistant","content":[]}}',
        ]
    )

    transcript = parse_pi_transcript(payload, "pi:ledger", branch="older")

    assert {item.id: item.label for item in transcript.branches} == {
        "older": "Branch ending at line 3",
        "latest": "Latest saved branch",
    }


def test_conflicting_additional_session_header_cannot_cross_native_identity() -> None:
    payload = b"\n".join(
        [
            b'{"type":"session","version":3,"id":"first","cwd":"/w"}',
            b'{"type":"message","id":"first-entry","parentId":null,"message":{"role":"user","content":"first"}}',
            b'{"type":"session","version":3,"id":"second","cwd":"/w"}',
            b'{"type":"message","id":"second-entry","parentId":null,"message":{"role":"user","content":"second"}}',
        ]
    )

    with pytest.raises(TranscriptUnavailable) as caught:
        parse_pi_transcript(payload, "pi:ledger")

    assert caught.value.kind == "invalid_source"
    assert "identity" in caught.value.reason


def test_matching_additional_session_header_is_skipped_with_a_warning() -> None:
    payload = b"\n".join(
        [
            b'{"type":"session","version":3,"id":"native","cwd":"/w"}',
            b'{"type":"message","id":"first","parentId":null,"message":{"role":"user","content":"first"}}',
            b'{"type":"session","version":3,"id":"native","cwd":"/w"}',
            b'{"type":"message","id":"second","parentId":"first","message":{"role":"assistant","content":[]}}',
        ]
    )

    transcript = parse_pi_transcript(payload, "pi:ledger")

    assert transcript.selected_branch == "second"
    assert any("extra session header" in warning for warning in transcript.warnings)


def test_duplicate_conflicting_and_unmatched_tool_results_remain_explicit() -> None:
    payload = b"\n".join(
        [
            b'{"type":"session","version":3,"id":"native","cwd":"/w"}',
            b'{"type":"message","id":"a","parentId":null,"message":{"role":"assistant","content":[{"type":"toolCall","id":"dup","name":"read","arguments":{}},{"type":"toolCall","id":"dup","name":"write","arguments":{}},{"type":"toolCall","id":"missing","name":"bash","arguments":{}}]}}',
            b'{"type":"message","id":"r1","parentId":"a","message":{"role":"toolResult","toolCallId":"dup","toolName":"read","content":"one","isError":false}}',
            b'{"type":"message","id":"r2","parentId":"r1","message":{"role":"toolResult","toolCallId":"orphan","toolName":"other","content":"two","isError":true}}',
        ]
    )
    transcript = parse_pi_transcript(payload, "pi:ledger")
    calls = [block for entry in transcript.entries if isinstance(entry, Message) for block in entry.blocks if isinstance(block, ToolBlock)]
    assert isinstance(calls[0].output, ConflictingOutput)
    assert isinstance(calls[1].output, ConflictingOutput)
    assert calls[2].output is None
    standalone = [entry for entry in transcript.entries if isinstance(entry, Message) and entry.role == "tool"]
    assert [entry.phase for entry in standalone] == ["conflicting", "unmatched"]


def test_both_compaction_shapes_are_explicit_checkpoint_evidence() -> None:
    transcript = parse("compaction.jsonl")
    assert any(isinstance(entry, Notice) and entry.label == "Context compacted" and entry.text == "Checkpoint summary" for entry in transcript.entries)
    checkpoint = [entry for entry in transcript.entries if isinstance(entry, Message) and entry.phase == "retained checkpoint"]
    assert len(checkpoint) == 2
    call = next(block for entry in checkpoint for block in entry.blocks if isinstance(block, ToolBlock))
    assert isinstance(call.output, RecordedOutput)
    assert call.output.blocks == (TextBlock("retained result"),)
    assert any("both compaction retention shapes" in warning for warning in transcript.warnings)
    assert any("missing first-kept entry" in warning for warning in transcript.warnings)


def test_invalid_media_unknown_roles_and_local_output_references_are_inert() -> None:
    payload = b"\n".join(
        [
            b'{"type":"session","version":3,"id":"native","cwd":"/w"}',
            b'{"type":"message","id":"u","parentId":null,"message":{"role":"user","content":[{"type":"image","mimeType":"image/svg+xml","data":"PHN2Zy8+"},{"type":"file","path":"/private/file"}]}}',
            b'{"type":"message","id":"b","parentId":"u","message":{"role":"bashExecution","command":"false","output":"bad","exitCode":1,"cancelled":false,"truncated":true,"fullOutputPath":"/private/output"}}',
            b'{"type":"message","id":"x","parentId":"b","message":{"role":"mystery","content":"PRIVATE UNKNOWN"}}',
        ]
    )
    transcript = parse_pi_transcript(payload, "pi:ledger")
    assert "/private/file" not in repr(transcript)
    assert "/private/output" not in repr(transcript)
    assert "PRIVATE UNKNOWN" not in repr(transcript)
    assert any(isinstance(block, AttachmentBlock) for entry in transcript.entries if isinstance(entry, Message) for block in entry.blocks)
    bash = next(block for entry in transcript.entries if isinstance(entry, Message) for block in entry.blocks if isinstance(block, ToolBlock))
    assert isinstance(bash.output, RecordedOutput) and bash.output.status == "failed"


@pytest.mark.parametrize(
    ("exit_code", "cancelled", "status"),
    [
        (0, False, "succeeded"),
        (1, False, "failed"),
        (False, False, "recorded"),
        (0.0, False, "recorded"),
        (None, False, "recorded"),
        (0, True, "failed"),
    ],
)
def test_bash_status_requires_an_exact_integer_exit_code(
    exit_code: object, cancelled: bool, status: str
) -> None:
    header = {"type": "session", "version": 3, "id": "native", "cwd": "/w"}
    execution = {
        "type": "message",
        "id": "bash",
        "parentId": None,
        "message": {
            "role": "bashExecution",
            "command": "command",
            "output": "output",
            "exitCode": exit_code,
            "cancelled": cancelled,
        },
    }
    payload = (json.dumps(header) + "\n" + json.dumps(execution)).encode()
    transcript = parse_pi_transcript(payload, "pi:ledger")
    block = next(
        block
        for entry in transcript.entries
        if isinstance(entry, Message)
        for block in entry.blocks
        if isinstance(block, ToolBlock)
    )
    assert isinstance(block.output, RecordedOutput)
    assert block.output.status == status


def test_damage_invalid_branch_and_empty_sources_are_not_silent_success() -> None:
    payload = (FIXTURES / "core.jsonl").read_bytes() + b"{broken}\n{\"partial\":"
    transcript = parse_pi_transcript(payload, "pi:ledger")
    assert any("malformed JSON" in warning for warning in transcript.warnings)
    assert any("incomplete or malformed" in warning for warning in transcript.warnings)
    with pytest.raises(TranscriptUnavailable) as invalid:
        parse("branches.jsonl", branch="root")
    assert invalid.value.kind == "invalid_branch"
    with pytest.raises(TranscriptUnavailable) as empty:
        parse_pi_transcript(b'{"type":"session","version":3,"id":"native","cwd":"/w"}\n', "pi:ledger")
    assert empty.value.kind == "invalid_source"


def test_multiple_roots_dangling_parents_cycles_and_duplicate_ids_are_diagnosed() -> None:
    payload = b"\n".join(
        [
            b'{"type":"session","version":3,"id":"native","cwd":"/w"}',
            b'{"type":"message","id":"root","parentId":null,"message":{"role":"user","content":"root"}}',
            b'{"type":"message","id":"leaf","parentId":"root","message":{"role":"assistant","content":[]}}',
            b'{"type":"message","id":"cycle-a","parentId":"cycle-b","message":{"role":"user","content":"a"}}',
            b'{"type":"message","id":"cycle-b","parentId":"cycle-a","message":{"role":"user","content":"b"}}',
            b'{"type":"message","id":"dup","parentId":null,"message":{"role":"user","content":"first"}}',
            b'{"type":"message","id":"dup","parentId":null,"message":{"role":"user","content":"second"}}',
            b'{"type":"message","id":"orphan","parentId":"missing","message":{"role":"user","content":"orphan"}}',
        ]
    )
    transcript = parse_pi_transcript(payload, "pi:ledger")
    assert transcript.selected_branch == "orphan"
    assert {item.id for item in transcript.branches} == {"leaf", "orphan"}
    assert any("missing parent" in warning for warning in transcript.warnings)
    assert any("parent cycle" in warning for warning in transcript.warnings)
    assert any("duplicated" in warning for warning in transcript.warnings)


def test_unicode_redaction_custom_visibility_outcomes_and_source_bytes_are_preserved_safely() -> None:
    payload = b"\n".join(
        [
            b'{"type":"session","version":3,"id":"native","cwd":"/w","parentSession":"/must/not/open"}',
            '{"type":"message","id":"u","parentId":null,"message":{"role":"user","content":[{"type":"text","text":"Привіт € ```code``` ![remote](https://example.invalid/x)"},{"type":"text","text":""}]}}'.encode(),
            b'{"type":"message","id":"a","parentId":"u","message":{"role":"assistant","stopReason":"deferred","content":[{"type":"thinking","redacted":true,"thinkingSignature":"SECRET"},{"type":"toolCall","id":"call","name":"read","arguments":{}}]}}',
            b'{"type":"message","id":"r","parentId":"a","message":{"role":"toolResult","toolCallId":"call","toolName":"read","content":"result","isError":1}}',
            b'{"type":"custom_message","id":"hidden","parentId":"r","customType":"memory","content":"hidden but inspectable","display":false}',
        ]
    )
    before = bytes(payload)
    transcript = parse_pi_transcript(payload, "pi:ledger")
    assert payload == before
    assert "Привіт €" in repr(transcript)
    assert "https://example.invalid/x" in repr(transcript)
    assert "SECRET" not in repr(transcript)
    assert any(isinstance(block, ReasoningBlock) and "redacted" in block.text for entry in transcript.entries if isinstance(entry, Message) for block in entry.blocks)
    assert any(isinstance(entry, Message) and entry.phase == "hidden extension: memory" for entry in transcript.entries)
    assert any("without a recorded outcome" in warning for warning in transcript.warnings)


@pytest.mark.parametrize("stop_reason", ["stop", "length", "error", "aborted", "deferred", "pending"])
def test_assistant_completion_state_is_preserved_without_inference(stop_reason: str) -> None:
    payload = (
        b'{"type":"session","version":3,"id":"native","cwd":"/w"}\n'
        + ('{"type":"message","id":"a","parentId":null,"message":{"role":"assistant","stopReason":"'
           + stop_reason + '","content":[]}}').encode()
    )
    transcript = parse_pi_transcript(payload, "pi:ledger")
    message = next(entry for entry in transcript.entries if isinstance(entry, Message))
    assert message.phase == stop_reason


def test_legacy_compaction_pointer_must_target_an_earlier_selected_ancestor() -> None:
    payload = b"\n".join(
        [
            b'{"type":"session","version":3,"id":"native","cwd":"/w"}',
            b'{"type":"compaction","id":"c","parentId":null,"summary":"summary","firstKeptEntryId":"later","tokensBefore":1}',
            b'{"type":"message","id":"later","parentId":"c","message":{"role":"user","content":"later"}}',
        ]
    )
    transcript = parse_pi_transcript(payload, "pi:ledger")
    assert any("missing first-kept entry" in warning for warning in transcript.warnings)


def test_retained_checkpoint_tool_ids_do_not_conflict_with_recorded_history() -> None:
    payload = b"\n".join(
        [
            b'{"type":"session","version":3,"id":"native","cwd":"/w"}',
            b'{"type":"message","id":"a","parentId":null,"message":{"role":"assistant","content":[{"type":"toolCall","id":"same","name":"read","arguments":{}}]}}',
            b'{"type":"message","id":"r","parentId":"a","message":{"role":"toolResult","toolCallId":"same","toolName":"read","content":"original","isError":false}}',
            b'{"type":"compaction","id":"c","parentId":"r","summary":"summary","retainedTail":[{"role":"assistant","content":[{"type":"toolCall","id":"same","name":"read","arguments":{}}]},{"role":"toolResult","toolCallId":"same","toolName":"read","content":"checkpoint","isError":false}]}'
        ]
    )
    transcript = parse_pi_transcript(payload, "pi:ledger")
    calls = [block for entry in transcript.entries if isinstance(entry, Message) for block in entry.blocks if isinstance(block, ToolBlock)]
    assert [cast.output.blocks for cast in calls if isinstance(cast.output, RecordedOutput)] == [
        (TextBlock("original"),),
        (TextBlock("checkpoint"),),
    ]
    assert not any(isinstance(call.output, ConflictingOutput) for call in calls)


def test_long_linear_branch_uses_iterative_bounded_graph_work() -> None:
    rows: list[dict[str, object]] = [
        {"type": "session", "version": 3, "id": "native", "cwd": "/w"}
    ]
    parent: str | None = None
    for index in range(12_000):
        entry_id = f"e{index}"
        rows.append({
            "type": "message", "id": entry_id, "parentId": parent,
            "message": {"role": "user", "content": ""},
        })
        parent = entry_id
    payload = "\n".join(json.dumps(row) for row in rows).encode()
    transcript = parse_pi_transcript(payload, "pi:ledger")
    assert transcript.selected_branch == "e11999"
    assert len(transcript.entries) == 12_000


@pytest.mark.parametrize("version", [None, 1, 2, 4, True])
def test_only_explicit_pi_v3_is_supported(version: object) -> None:
    field = b"" if version is None else b',"version":' + str(version).lower().encode()
    payload = b'{"type":"session","id":"native","cwd":"/w"' + field + b"}\n"
    with pytest.raises(TranscriptUnavailable) as caught:
        parse_pi_transcript(payload, "pi:ledger")
    assert caught.value.kind == "unsupported"
