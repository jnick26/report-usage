import base64
import json

import pytest

from harness_usage.transcript import (
    AttachmentBlock,
    ImageBlock,
    Message,
    TextBlock,
    TranscriptUnavailable,
    read_jsonl,
    text_and_media_blocks,
)


def test_jsonl_reader_is_bounded_and_recovers_after_bad_lines() -> None:
    payload = b'{"type":"session"}\n{bad}\n{"type":"message"}'
    records, warnings = read_jsonl(payload)
    assert [(record.line, record.value["type"]) for record in records] == [
        (1, "session"),
        (3, "message"),
    ]
    assert warnings == ("Line 2 is malformed JSON and was skipped.",)
    with pytest.raises(TranscriptUnavailable) as caught:
        read_jsonl(b"x" * (128 * 1024 * 1024 + 1))
    assert caught.value.kind == "too_large"


def test_jsonl_reader_marks_unterminated_tail_without_exposing_it() -> None:
    records, warnings = read_jsonl(b'{"type":"session"}\n{"private":')
    assert len(records) == 1
    assert warnings == ("Final line 2 is incomplete or malformed and was skipped.",)
    assert "private" not in warnings[0]


def test_jsonl_reader_default_keeps_legacy_final_malformed_tail_warning() -> None:
    records, warnings = read_jsonl(b'{bad')
    assert records == ()
    assert warnings == ("Final line 1 is incomplete or malformed and was skipped.",)


def test_raster_images_are_validated_and_other_media_stays_inert() -> None:
    png_bytes = b"\x89PNG\r\n\x1a\nminimal"
    png = base64.b64encode(png_bytes).decode()
    image = ImageBlock("image/png", png)
    assert image.data_url == f"data:image/png;base64,{png}"
    with pytest.raises(ValueError):
        ImageBlock("image/svg+xml", base64.b64encode(b"<svg/>").decode())
    with pytest.raises(ValueError):
        ImageBlock("image/png", "not base64")
    with pytest.raises(ValueError):
        ImageBlock("image/png", base64.b64encode(b"GIF89a-not-a-png").decode())
    with pytest.raises(ValueError):
        ImageBlock("image/png", base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"x" * (4 * 1024 * 1024)).decode())

    blocks, warnings = text_and_media_blocks(
        [
            {"type": "text", "text": "hello"},
            {"type": "image", "mimeType": "image/svg+xml", "data": "PHN2Zy8+"},
            {"type": "file", "path": "/private/secret"},
        ],
        source_line=7,
    )
    assert blocks == (
        TextBlock("hello"),
        AttachmentBlock("Unsupported image at line 7"),
        AttachmentBlock("Unsupported file content at line 7"),
    )
    assert len(warnings) == 2
    assert "/private/secret" not in repr(blocks)


@pytest.mark.parametrize(
    ("mime", "payload"),
    [
        ("image/png", b"\x89PNG\r\n\x1a\nminimal"),
        ("image/jpeg", b"\xff\xd8\xffminimal"),
        ("image/gif", b"GIF89aminimal"),
        ("image/webp", b"RIFF\x04\x00\x00\x00WEBPminimal"),
    ],
)
def test_each_supported_raster_type_requires_its_native_signature(mime: str, payload: bytes) -> None:
    encoded = base64.b64encode(payload).decode()
    assert ImageBlock(mime, encoded).data_url == f"data:{mime};base64,{encoded}"


def test_shared_records_are_frozen_and_validate_discriminants() -> None:
    message = Message("message-1", "user", (TextBlock("hello"),))
    with pytest.raises(AttributeError):
        message.role = "assistant"  # type: ignore[misc]
    with pytest.raises(ValueError):
        Message("message-1", "mystery", ())
    with pytest.raises(ValueError):
        TranscriptUnavailable("bad", "other")  # type: ignore[arg-type]


def test_jsonl_reader_rejects_duplicate_keys_and_non_objects() -> None:
    payload = b'{"a":1,"a":2}\n[]\n' + json.dumps({"ok": True}).encode() + b"\n"
    records, warnings = read_jsonl(payload)
    assert [record.line for record in records] == [3]
    assert warnings == (
        "Line 1 is malformed JSON and was skipped.",
        "Line 2 is not a JSON object and was skipped.",
    )


def test_jsonl_reader_replaces_unpaired_surrogates_before_projection() -> None:
    records, warnings = read_jsonl(
        b'{"type":"message","payload":{"text":"before\\ud800after","valid":"\\ud83d\\ude00"}}\n'
    )
    assert records[0].value["payload"] == {"text": "before\ufffdafter", "valid": "😀"}
    assert warnings == ("Line 1 contains invalid Unicode; replacement markers were inserted.",)
    repr(records).encode("utf-8")

    collision_records, collision_warnings = read_jsonl(b'{"\\ud800":1,"\\ud801":2}\n')
    assert collision_records == ()
    assert collision_warnings == ("Line 1 is malformed JSON and was skipped.",)


def test_recorded_branches_require_a_selection_before_rendering():
    from harness_usage.transcript import Branch, Transcript
    with pytest.raises(ValueError,match='must be selected'):
        Transcript('session','native','Title','pi',(),branches=(Branch('leaf','Latest'),))
