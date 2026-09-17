"""Safe HTML rendering for live and standalone transcripts."""

from __future__ import annotations

from base64 import b64encode
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from urllib.parse import quote

from jinja2 import Environment, FileSystemLoader, select_autoescape
from markdown_it import MarkdownIt
from markupsafe import Markup

from .presentation import harness_label
from .transcript import (
    AttachmentBlock,
    ConflictingOutput,
    ImageBlock,
    Message,
    Notice,
    ReasoningBlock,
    RecordedOutput,
    TextBlock,
    ToolBlock,
    TranscriptPage,
)

_ROOT = Path(__file__).parent
_STATIC = _ROOT / "static"
_TEMPLATES = _ROOT / "templates"
_CSS = _STATIC / "transcript.css"
_JS = _STATIC / "transcript.js"
_FONT_FILES = {
    "/static/vendor/NotoSans.ttf": ("font/ttf", _STATIC / "vendor/NotoSans.ttf"),
    "/static/vendor/NotoSansMono.ttf": ("font/ttf", _STATIC / "vendor/NotoSansMono.ttf"),
    "/static/vendor/SymbolsNerdFontMono-subset.woff": (
        "font/woff",
        _STATIC / "vendor/SymbolsNerdFontMono-subset.woff",
    ),
}
_LICENSE_FILES = (
    _STATIC / "vendor/NotoSans-LICENSE",
    _STATIC / "vendor/SymbolsNerdFont-LICENSE.txt",
    _STATIC / "vendor/Codicons-LICENSE.txt",
    _STATIC / "vendor/MaterialDesign-LICENSE.txt",
)
_MARKDOWN = (
    MarkdownIt("commonmark", {"html": False, "linkify": False})
    .enable("table")
    .disable(("image", "link", "autolink"))
)


@dataclass(frozen=True, slots=True)
class _ToolView:
    block: ToolBlock
    anchor: str
    category: str
    icon: str
    label: str
    status: str | None
    failed: bool


@dataclass(frozen=True, slots=True)
class _ToolGroup:
    tools: tuple[_ToolView, ...]
    summary: str
    failed: bool


type _Segment = TextBlock | ReasoningBlock | AttachmentBlock | ImageBlock | _ToolGroup


@dataclass(frozen=True, slots=True)
class _MessageView:
    entry: Message
    anchor: str
    label: str
    model: str | None
    segments: tuple[_Segment, ...]


@dataclass(frozen=True, slots=True)
class _NoticeView:
    entry: Notice
    anchor: str


@dataclass(frozen=True, slots=True)
class _ContextGroup:
    messages: tuple[_MessageView, ...]


@dataclass(frozen=True, slots=True)
class _ActivityBatch:
    messages: tuple[_MessageView, ...]
    summary: str


type _EntryView = _MessageView | _NoticeView | _ContextGroup | _ActivityBatch


@dataclass(frozen=True, slots=True)
class _RailItem:
    anchor: str
    label: str
    excerpt: str
    tool: bool = False


@dataclass(frozen=True, slots=True)
class _View:
    page: TranscriptPage
    entries: tuple[_EntryView, ...]
    rail: tuple[_RailItem, ...]
    parent_href: str | None
    children: tuple[tuple[str, str], ...]
    transcript_href: str
    download_href: str


def _path(session_id: str, suffix: str = "/transcript") -> str:
    return f"/sessions/{quote(session_id, safe='')}{suffix}"


def _entry_anchor(entry: Message | Notice) -> str:
    return f"entry-{sha256(entry.id.encode()).hexdigest()[:16]}"


def _tool_category(name: str) -> tuple[str, str]:
    value = name.casefold()
    if "agent" in value:
        return "Subagent", "\U000f167a"
    if any(word in value for word in ("read", "open", "view")):
        return "Read", "\uea7b"
    if any(word in value for word in ("search", "find", "grep", "glob", "query")):
        return "Search", "\uea6d"
    if any(word in value for word in ("edit", "write", "patch", "replace")):
        return "Edit", "\uea73"
    if any(word in value for word in ("run", "exec", "command", "shell", "bash", "terminal", "test")):
        return "Run", "\uea85"
    return "Tool", "\ueb6d"


def _tool_status(tool: ToolBlock) -> tuple[str | None, bool]:
    if tool.output is None:
        return "Result not recorded", False
    if isinstance(tool.output, ConflictingOutput):
        return "Conflicting result", False
    if tool.output.status == "failed":
        return "Failed", True
    if len(tool.output.blocks) > 1:
        return f"{len(tool.output.blocks)} result items", False
    if tool.output.status == "recorded":
        return "Recorded result", False
    return None, False


def _tool_label(tool: ToolBlock) -> str:
    try:
        arguments = json.loads(tool.arguments)
    except (ValueError, TypeError, RecursionError):
        return tool.name
    if isinstance(arguments, dict):
        for key in ("path", "file_path", "command", "cmd", "query"):
            value = arguments.get(key)
            if isinstance(value, str) and value.strip():
                return value
    return tool.name


def _excerpt(message: Message) -> str:
    values: list[str] = []
    for block in message.blocks:
        if isinstance(block, (TextBlock, ReasoningBlock)):
            values.append(block.text)
        elif isinstance(block, AttachmentBlock):
            values.append(block.label)
        elif isinstance(block, ToolBlock):
            values.append(block.name)
    return " ".join(" ".join(values).split())[:180] or "Recorded message"


def _view(page: TranscriptPage) -> _View:
    entries: list[_EntryView] = []
    rail: list[_RailItem] = []
    previous_model = page.transcript.model
    role_labels = {
        "user": "User",
        "assistant": "Assistant",
        "system": "System",
        "developer": "Developer",
        "tool": "Tool result",
        "context": "Context",
    }
    for entry in page.transcript.entries:
        anchor = _entry_anchor(entry)
        if isinstance(entry, Notice):
            entries.append(_NoticeView(entry, anchor))
            continue
        label = role_labels[entry.role]
        segments: list[_Segment] = []
        pending: list[_ToolView] = []
        tool_number = 0

        def flush_tools() -> None:
            if not pending:
                return
            categories = tuple(dict.fromkeys(tool.category for tool in pending))
            count = len(pending)
            segments.append(
                _ToolGroup(
                    tuple(pending),
                    f"{', '.join(categories)} · {count} {'action' if count == 1 else 'actions'}",
                    any(tool.failed for tool in pending),
                )
            )
            pending.clear()

        for block in entry.blocks:
            if isinstance(block, ToolBlock):
                tool_number += 1
                category, icon = _tool_category(block.name)
                status, failed = _tool_status(block)
                tool = _ToolView(
                    block, f"{anchor}-tool-{tool_number}", category, icon,
                    _tool_label(block), status, failed,
                )
                pending.append(tool)
            else:
                flush_tools()
                segments.append(block)
        flush_tools()
        model = entry.model if entry.model and entry.model != previous_model else None
        if entry.model:
            previous_model = entry.model
        entries.append(_MessageView(entry, anchor, label, model, tuple(segments)))
    grouped_entries: list[_EntryView] = []
    context_messages: list[_MessageView] = []

    def flush_context() -> None:
        if context_messages:
            grouped_entries.append(_ContextGroup(tuple(context_messages)))
            context_messages.clear()

    for item in entries:
        if isinstance(item, _MessageView) and item.entry.role in ("system", "developer", "context"):
            context_messages.append(item)
        else:
            flush_context()
            grouped_entries.append(item)
    flush_context()
    display_entries: list[_EntryView] = []
    activity_messages: list[_MessageView] = []

    def flush_activity() -> None:
        if not activity_messages:
            return
        actions = sum(
            len(segment.tools)
            for message in activity_messages
            for segment in message.segments
            if isinstance(segment, _ToolGroup)
        )
        reasoning = sum(
            isinstance(segment, ReasoningBlock)
            for message in activity_messages
            for segment in message.segments
        )
        parts = []
        if actions:
            parts.append(f"{actions} {'action' if actions == 1 else 'actions'}")
        if reasoning:
            parts.append(
                f"{reasoning} reasoning {'entry' if reasoning == 1 else 'entries'}"
            )
        display_entries.append(_ActivityBatch(tuple(activity_messages), " · ".join(parts)))
        activity_messages.clear()

    for item in grouped_entries:
        if (
            isinstance(item, _MessageView)
            and item.entry.role == "assistant"
            and bool(item.segments)
            and all(isinstance(segment, (_ToolGroup, ReasoningBlock)) for segment in item.segments)
        ):
            activity_messages.append(item)
        else:
            flush_activity()
            display_entries.append(item)
    flush_activity()
    rail.extend(
        _RailItem(item.anchor, "User message", _excerpt(item.entry))
        for item in display_entries
        if isinstance(item, _MessageView) and item.entry.role == "user"
    )
    selected = page.transcript.selected_branch
    query = f"branch={quote(selected, safe='')}" if selected else ""
    transcript_href = _path(page.transcript.session_id)
    download_href = _path(page.transcript.session_id, "/transcript.html")
    if query:
        download_href = f"{download_href}?{query}"
    return _View(
        page,
        tuple(display_entries),
        tuple(rail),
        _path(page.parent.session_id) if page.parent else None,
        tuple((_path(child.session_id), child.title) for child in page.children),
        transcript_href,
        download_href,
    )


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _markdown(text: str) -> Markup:
    return Markup(_MARKDOWN.render(text))


def _standalone_css() -> str:
    css = _source(_CSS)
    for url, (mime, path) in _FONT_FILES.items():
        data_url = f"data:{mime};base64,{b64encode(path.read_bytes()).decode('ascii')}"
        css = css.replace(url, data_url)
    return css


def _digest(source: str) -> str:
    return b64encode(sha256(source.encode()).digest()).decode("ascii")


def transcript_csp(*, standalone: bool = False) -> str:
    """Return the CSP matching the renderer's trusted live or inline assets."""
    if standalone:
        scripts = f"'sha256-{_digest(_source(_JS))}'"
        styles = f"'sha256-{_digest(_standalone_css())}'"
        fonts = "data:"
        forms = "'none'"
    else:
        scripts = styles = fonts = "'self'"
        forms = "'self'"
    return (
        f"default-src 'none'; script-src {scripts}; style-src {styles}; "
        f"font-src {fonts}; img-src data:; connect-src 'none'; "
        f"frame-ancestors 'none'; base-uri 'none'; form-action {forms}"
    )


def render_transcript(page: TranscriptPage, *, standalone: bool = False) -> str:
    """Render one transcript projection without interpreting transcript-provided HTML."""
    environment = Environment(
        loader=FileSystemLoader(_TEMPLATES),
        autoescape=select_autoescape(("html",)),
        enable_async=False,
    )
    environment.globals.update(
        ActivityBatch=_ActivityBatch,
        AttachmentBlock=AttachmentBlock,
        ConflictingOutput=ConflictingOutput,
        ContextGroup=_ContextGroup,
        ImageBlock=ImageBlock,
        MessageView=_MessageView,
        NoticeView=_NoticeView,
        ReasoningBlock=ReasoningBlock,
        RecordedOutput=RecordedOutput,
        TextBlock=TextBlock,
        ToolGroup=_ToolGroup,
        isinstance=isinstance,
    )
    environment.filters["safe_markdown"] = _markdown
    environment.filters["harness_label"] = harness_label
    template = environment.get_template("transcript.html")
    return template.render(
        view=_view(page),
        standalone=standalone,
        css=_standalone_css() if standalone else None,
        javascript=_source(_JS) if standalone else None,
        csp=transcript_csp(standalone=standalone),
        font_licenses="\n\n".join(_source(path) for path in _LICENSE_FILES) if standalone else None,
    )


__all__ = ["render_transcript", "transcript_csp"]
