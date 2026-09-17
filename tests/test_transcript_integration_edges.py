"""Independent integration regressions for transcript routing and rendering."""

from pathlib import Path

from fastapi.testclient import TestClient

from harness_usage.transcript import Message, TextBlock, ToolBlock, Transcript, TranscriptPage
from harness_usage.transcript_rendering import render_transcript
from harness_usage.web import create_app


def page(session_id: str, *blocks: TextBlock | ToolBlock) -> TranscriptPage:
    return TranscriptPage(
        Transcript(
            session_id,
            "native",
            "Recorded session",
            "pi",
            (Message("message", "assistant", blocks),),
        )
    )


def test_deep_recorded_tool_arguments_do_not_crash_label_extraction() -> None:
    arguments = "[" * 10_000 + "]" * 10_000

    rendered = render_transcript(page("pi:deep", ToolBlock("call", "Run", arguments)))

    assert "Run" in rendered
    assert arguments in rendered


def test_percent_encoded_opaque_session_identity_reaches_both_routes() -> None:
    class TranscriptApplication:
        data_dir = Path("/tmp")
        timezone = "UTC"

        def __init__(self) -> None:
            self.requested: list[str] = []

        def get_roots(self) -> tuple[str, ...]:
            return ("/tmp",)

        def transcript(self, session_id: str, branch: str | None = None) -> TranscriptPage:
            self.requested.append(session_id)
            return page(session_id, TextBlock("Visible transcript"))

        def close(self) -> None:
            pass

    backend = TranscriptApplication()
    with TestClient(create_app(backend), base_url="http://127.0.0.1:8765") as client:  # type: ignore[arg-type]
        live = client.get("/sessions/session%20%2F%20one/transcript")
        download = client.get("/sessions/session%20%2F%20one/transcript.html")

    assert live.status_code == 200
    assert download.status_code == 200
    assert backend.requested == ["session / one", "session / one"]
