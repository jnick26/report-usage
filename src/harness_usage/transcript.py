"""Framework-free transcript records and bounded source helpers."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import base64
import binascii
import json
import re
from typing import Literal, cast

from .pricing import CostLine


MAX_TRANSCRIPT_BYTES = 128 * 1024 * 1024
MAX_IMAGE_BYTES = 4 * 1024 * 1024
SUPPORTED_IMAGE_MIMES = frozenset({"image/gif", "image/jpeg", "image/png", "image/webp"})
_SURROGATE_RE = re.compile("[\ud800-\udfff]")

TranscriptUnavailableKind = Literal[
    "missing", "unsupported", "too_large", "changed", "invalid_branch", "invalid_source"
]
MessageRole = Literal["user", "assistant", "system", "developer", "tool", "context"]
OutputStatus = Literal["recorded", "succeeded", "failed"]


def _nonempty(value: str, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")


def _line(value: int) -> None:
    if type(value) is not int or value < 0:
        raise ValueError("source_line must be a non-negative integer")


def _raster_matches(mime: str, data: bytes) -> bool:
    if mime == "image/png":
        return data.startswith(b"\x89PNG\r\n\x1a\n")
    if mime == "image/jpeg":
        return data.startswith(b"\xff\xd8\xff")
    if mime == "image/gif":
        return data.startswith((b"GIF87a", b"GIF89a"))
    return mime == "image/webp" and len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP"


@dataclass(frozen=True, slots=True)
class TextBlock:
    text: str

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise ValueError("text must be a string")


@dataclass(frozen=True, slots=True)
class ReasoningBlock:
    text: str
    label: str = "Recorded reasoning"

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise ValueError("text must be a string")
        _nonempty(self.label, "label")


@dataclass(frozen=True, slots=True)
class AttachmentBlock:
    label: str

    def __post_init__(self) -> None:
        _nonempty(self.label, "label")


@dataclass(frozen=True, slots=True)
class ImageBlock:
    mime: str
    data: str

    def __post_init__(self) -> None:
        if self.mime not in SUPPORTED_IMAGE_MIMES or not isinstance(self.data, str):
            raise ValueError("unsupported image")
        if len(self.data) > ((MAX_IMAGE_BYTES + 2) // 3) * 4:
            raise ValueError("image exceeds size limit")
        try:
            decoded = base64.b64decode(self.data, validate=True)
        except (binascii.Error, ValueError) as error:
            raise ValueError("invalid image data") from error
        if len(decoded) > MAX_IMAGE_BYTES:
            raise ValueError("image exceeds size limit")
        if not _raster_matches(self.mime, decoded):
            raise ValueError("image data does not match its raster type")

    @property
    def data_url(self) -> str:
        return f"data:{self.mime};base64,{self.data}"


OutputBlock = TextBlock | AttachmentBlock | ImageBlock


@dataclass(frozen=True, slots=True)
class RecordedOutput:
    blocks: tuple[OutputBlock, ...]
    status: OutputStatus = "recorded"
    source_line: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.blocks, tuple) or not all(isinstance(block, (TextBlock, AttachmentBlock, ImageBlock)) for block in self.blocks):
            raise ValueError("blocks must be immutable output blocks")
        if self.status not in ("recorded", "succeeded", "failed"):
            raise ValueError("invalid output status")
        _line(self.source_line)


@dataclass(frozen=True, slots=True)
class ConflictingOutput:
    reason: str

    def __post_init__(self) -> None:
        _nonempty(self.reason, "reason")


@dataclass(frozen=True, slots=True)
class ToolBlock:
    call_id: str | None
    name: str
    arguments: str
    output: RecordedOutput | ConflictingOutput | None = None
    source_line: int = 0

    def __post_init__(self) -> None:
        if self.call_id is not None:
            _nonempty(self.call_id, "call_id")
        _nonempty(self.name, "name")
        if not isinstance(self.arguments, str):
            raise ValueError("arguments must be a string")
        if self.output is not None and not isinstance(self.output, (RecordedOutput, ConflictingOutput)):
            raise ValueError("invalid tool output")
        _line(self.source_line)


MessageBlock = TextBlock | ReasoningBlock | AttachmentBlock | ImageBlock | ToolBlock


@dataclass(frozen=True, slots=True)
class Message:
    id: str
    role: MessageRole
    blocks: tuple[MessageBlock, ...]
    model: str | None = None
    phase: str | None = None
    timestamp: str | None = None
    usage_id: str | None = None

    def __post_init__(self) -> None:
        _nonempty(self.id, "id")
        if self.role not in ("user", "assistant", "system", "developer", "tool", "context"):
            raise ValueError("invalid message role")
        if not isinstance(self.blocks, tuple) or not all(isinstance(block, (TextBlock, ReasoningBlock, AttachmentBlock, ImageBlock, ToolBlock)) for block in self.blocks):
            raise ValueError("blocks must be immutable message blocks")
        for name, value in (("model", self.model), ("phase", self.phase), ("timestamp", self.timestamp)):
            if value is not None and not isinstance(value, str):
                raise ValueError(f"{name} must be a string or None")
        if self.usage_id is not None:
            _nonempty(self.usage_id, 'usage_id')
            if self.role != 'assistant':
                raise ValueError('usage identity requires an assistant request')


@dataclass(frozen=True, slots=True)
class Notice:
    id: str
    label: str
    text: str

    def __post_init__(self) -> None:
        _nonempty(self.id, "id")
        _nonempty(self.label, "label")
        if not isinstance(self.text, str):
            raise ValueError("text must be a string")


@dataclass(frozen=True, slots=True)
class Branch:
    id: str
    label: str

    def __post_init__(self) -> None:
        _nonempty(self.id, "id")
        _nonempty(self.label, "label")


TranscriptEntry = Message | Notice


@dataclass(frozen=True, slots=True)
class Transcript:
    session_id: str
    native_id: str
    title: str
    harness: Literal["pi", "codex", "claude", "copilot-vscode", "copilot-cli"]
    entries: tuple[TranscriptEntry, ...]
    cwd: str | None = None
    model: str | None = None
    started: str | None = None
    branches: tuple[Branch, ...] = ()
    selected_branch: str | None = None
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field, value in (("session_id", self.session_id), ("native_id", self.native_id), ("title", self.title)):
            _nonempty(value, field)
        if self.harness not in ("pi", "codex", "claude", "copilot-vscode", "copilot-cli"):
            raise ValueError("invalid harness")
        if not isinstance(self.entries, tuple) or not all(isinstance(entry, (Message, Notice)) for entry in self.entries):
            raise ValueError("entries must be immutable transcript entries")
        if not isinstance(self.branches, tuple) or not all(isinstance(branch, Branch) for branch in self.branches):
            raise ValueError("branches must be immutable")
        if len({branch.id for branch in self.branches}) != len(self.branches):
            raise ValueError("branch ids must be unique")
        if self.branches and self.selected_branch is None:
            raise ValueError("a recorded branch must be selected")
        if self.selected_branch is not None and self.selected_branch not in {branch.id for branch in self.branches}:
            raise ValueError("selected branch is not recorded")
        if not isinstance(self.warnings, tuple) or not all(isinstance(warning, str) for warning in self.warnings):
            raise ValueError("warnings must be immutable strings")


class TranscriptUnavailable(Exception):
    def __init__(self, reason: str, kind: TranscriptUnavailableKind) -> None:
        _nonempty(reason, "reason")
        if kind not in ("missing", "unsupported", "too_large", "changed", "invalid_branch", "invalid_source"):
            raise ValueError("invalid transcript unavailability kind")
        self.reason = reason
        self.kind = kind
        super().__init__(reason)


@dataclass(frozen=True, slots=True)
class TranscriptLink:
    session_id: str
    title: str

    def __post_init__(self) -> None:
        _nonempty(self.session_id, "session_id")
        _nonempty(self.title, "title")


@dataclass(frozen=True, slots=True)
class TranscriptCostPoint:
    message_id: str
    amount: Decimal | None
    cumulative: Decimal
    incomplete: bool
    cumulative_incomplete: bool
    calculations: tuple[CostLine, ...] = ()

    def __post_init__(self) -> None:
        _nonempty(self.message_id, 'message_id')
        for value in (self.amount, self.cumulative):
            if value is not None and (not isinstance(value, Decimal) or not value.is_finite() or value < 0):
                raise ValueError('cost must be a finite non-negative decimal')
        if self.cumulative is None or type(self.incomplete) is not bool or type(self.cumulative_incomplete) is not bool:
            raise ValueError('invalid cumulative cost state')
        if not isinstance(self.calculations, tuple) or not all(isinstance(line, CostLine) for line in self.calculations):
            raise ValueError('calculations must be immutable cost lines')


@dataclass(frozen=True, slots=True)
class TranscriptPage:
    transcript: Transcript
    parent: TranscriptLink | None = None
    children: tuple[TranscriptLink, ...] = ()
    costs: tuple[TranscriptCostPoint, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.transcript, Transcript):
            raise ValueError("transcript must be a Transcript")
        if self.parent is not None and not isinstance(self.parent, TranscriptLink):
            raise ValueError("parent must be a TranscriptLink or None")
        if not isinstance(self.children, tuple) or not all(isinstance(child, TranscriptLink) for child in self.children):
            raise ValueError("children must be immutable transcript links")
        if not isinstance(self.costs, tuple) or not all(isinstance(point, TranscriptCostPoint) for point in self.costs):
            raise ValueError('costs must be immutable cost points')


@dataclass(frozen=True, slots=True)
class JsonlRecord:
    line: int
    value: dict[str, object]


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON key")
        value[key] = item
    return value


def _replace_surrogates(value: object) -> tuple[object, bool]:
    if isinstance(value, str):
        cleaned = _SURROGATE_RE.sub("\ufffd", value)
        return cleaned, cleaned != value
    if isinstance(value, list):
        cleaned_items: list[object] = []
        changed = False
        for item in value:
            cleaned_value, item_changed = _replace_surrogates(item)
            cleaned_items.append(cleaned_value)
            changed |= item_changed
        return cleaned_items, changed
    if isinstance(value, dict):
        cleaned_object: dict[str, object] = {}
        changed = False
        for key, item in value.items():
            cleaned_key_value, key_changed = _replace_surrogates(key)
            cleaned_key = cast(str, cleaned_key_value)
            if cleaned_key in cleaned_object:
                raise ValueError("Unicode replacement created a duplicate JSON key")
            cleaned_item, item_changed = _replace_surrogates(item)
            cleaned_object[cleaned_key] = cleaned_item
            changed |= key_changed or item_changed
        return cleaned_object, changed
    return value, False


def read_jsonl(payload: bytes, *, strict_tail: bool = False) -> tuple[tuple[JsonlRecord, ...], tuple[str, ...]]:
    """Decode one bounded snapshot, retaining valid records around damage.

    Legacy callers keep the historical final-fragment wording.  The Copilot
    CLI reader opts into strict tail classification because its durable
    lifetime proof distinguishes a complete malformed row from a pending one.
    """
    if len(payload) > MAX_TRANSCRIPT_BYTES:
        raise TranscriptUnavailable("Transcript exceeds the 128 MiB read limit.", "too_large")
    records: list[JsonlRecord] = []
    warnings: list[str] = []
    lines = payload.splitlines(keepends=True)
    for number, raw_line in enumerate(lines, 1):
        if not raw_line.strip():
            warnings.append(f"Line {number} is blank and was skipped.")
            continue
        try:
            value = json.loads(raw_line, object_pairs_hook=_unique_object)
            value, replaced_surrogates = _replace_surrogates(value)
        except (UnicodeError, ValueError, RecursionError) as error:
            text = raw_line.decode(errors='replace')
            incomplete = isinstance(error, json.JSONDecodeError) and (
                error.pos >= len(text.rstrip()) or error.msg.startswith('Unterminated string') or
                error.msg == 'Invalid \\uXXXX escape' and bool(re.search(r'\\u[0-9a-fA-F]{0,3}$', text)) or
                error.msg == 'Expecting value' and text[error.pos:] in
                ('-', 't', 'tr', 'tru', 'f', 'fa', 'fal', 'fals', 'n', 'nu', 'nul'))
            final_partial = (number == len(lines) and not raw_line.endswith((b"\n", b"\r"))
                             and (incomplete or not strict_tail))
            prefix = f"Final line {number} is incomplete or malformed" if final_partial else f"Line {number} is malformed JSON"
            warnings.append(prefix + " and was skipped.")
            continue
        if not isinstance(value, dict):
            warnings.append(f"Line {number} is not a JSON object and was skipped.")
            continue
        if replaced_surrogates:
            warnings.append(f"Line {number} contains invalid Unicode; replacement markers were inserted.")
        records.append(JsonlRecord(number, cast(dict[str, object], value)))
    return tuple(records), tuple(warnings)


def text_and_media_blocks(
    content: object, *, source_line: int
) -> tuple[tuple[OutputBlock, ...], tuple[str, ...]]:
    """Translate shared text/image content without following any attachment reference."""
    _line(source_line)
    if isinstance(content, str):
        return (TextBlock(content),), ()
    if not isinstance(content, list):
        return (AttachmentBlock(f"Content unavailable at line {source_line}"),), (
            f"Line {source_line} has missing or unsupported content.",
        )
    blocks: list[OutputBlock] = []
    warnings: list[str] = []
    for raw in content:
        item = raw if isinstance(raw, dict) else {}
        kind = item.get("type")
        if kind == "text" and isinstance(item.get("text"), str):
            blocks.append(TextBlock(cast(str, item["text"])))
        elif kind == "image":
            try:
                blocks.append(ImageBlock(cast(str, item.get("mimeType")), cast(str, item.get("data"))))
            except (TypeError, ValueError):
                blocks.append(AttachmentBlock(f"Unsupported image at line {source_line}"))
                warnings.append(f"Line {source_line} contains an invalid or unsupported image.")
        else:
            label = kind if isinstance(kind, str) and kind.strip() else "unknown"
            blocks.append(AttachmentBlock(f"Unsupported {label} content at line {source_line}"))
            warnings.append(f"Line {source_line} contains unsupported {label} content.")
    return tuple(blocks), tuple(warnings)


__all__ = [
    "AttachmentBlock", "Branch", "ConflictingOutput", "ImageBlock", "JsonlRecord",
    "MAX_IMAGE_BYTES", "MAX_TRANSCRIPT_BYTES", "Message", "MessageBlock", "MessageRole",
    "Notice", "OutputBlock", "OutputStatus", "ReasoningBlock", "RecordedOutput",
    "TextBlock", "ToolBlock", "Transcript", "TranscriptEntry", "TranscriptLink",
    "TranscriptPage", "TranscriptCostPoint", "TranscriptUnavailable", "read_jsonl", "text_and_media_blocks",
]
