"""On-demand projection of the pinned VS Code core chat-v3 subset."""
from datetime import UTC, datetime, timedelta
from typing import Literal, cast

from .copilot_vscode_reader import replay_chat_v3
from .transcript import (AttachmentBlock, Message, Notice, ReasoningBlock, RecordedOutput,
                         TextBlock, ToolBlock, Transcript, TranscriptEntry,
                         TranscriptUnavailable)


def _text(value: object) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _timestamp(value: object) -> str | None:
    if type(value) is not int or not 0 <= value < 2**63:
        return None
    try:
        return (datetime(1970, 1, 1, tzinfo=UTC) + timedelta(milliseconds=value)).isoformat()
    except OverflowError:
        return None


def _copilot(request: dict[str, object]) -> bool:
    agent = request.get('agent')
    extension = agent.get('extensionId') if isinstance(agent, dict) else None
    identity = extension.get('value') if isinstance(extension, dict) else None
    return isinstance(identity, str) and identity.casefold() == 'github.copilot-chat'


def parse_copilot_vscode_transcript(payload: bytes, expected_raw_id: str,
                                    scoped_session_id: str, scoped_native_id: str, *,
                                    representation: Literal['flat', 'operation_log'],
                                    branch: str | None = None) -> Transcript:
    if branch is not None:
        raise TranscriptUnavailable('VS Code chat transcripts expose one recorded view.', 'invalid_branch')
    try:
        snapshot = replay_chat_v3(payload, representation=representation)
    except ValueError as error:
        raise TranscriptUnavailable('The registered VS Code chat source is malformed.', 'changed') from error
    state = snapshot.value
    if (type(state.get('version')) is not int or state.get('version') != 3
            or _text(state.get('sessionId')) != expected_raw_id):
        raise TranscriptUnavailable('The source now contains a different session. Import sources again.', 'changed')
    requests = state.get('requests')
    if not isinstance(requests, list):
        raise TranscriptUnavailable('No supported VS Code chat records were found.', 'unsupported')
    entries: list[TranscriptEntry] = []
    for ordinal, item in enumerate(requests):
        if not isinstance(item, dict) or not _copilot(cast(dict[str, object], item)):
            continue
        request = cast(dict[str, object], item)
        request_id = _text(request.get('requestId')) or f'unidentified-request-{ordinal}'
        line = snapshot.line_for(('requests', ordinal))
        message = request.get('message')
        text = _text(message.get('text')) if isinstance(message, dict) else None
        if text is not None:
            entries.append(Message(request_id, 'user', (TextBlock(text),),
                                   timestamp=_timestamp(request.get('timestamp'))))
        response = request.get('response')
        response_line = snapshot.line_for(('requests', ordinal, 'response'))
        blocks: list[TextBlock | ReasoningBlock | AttachmentBlock | ToolBlock] = []
        unsupported = False
        if isinstance(response, list):
            for part in response:
                if not isinstance(part, dict):
                    unsupported = True
                    continue
                kind = part.get('kind')
                if kind == 'markdownContent':
                    content = part.get('content')
                    value = content.get('value') if isinstance(content, dict) else None
                    if isinstance(value, str):
                        blocks.append(TextBlock(value))
                    else:
                        unsupported = True
                elif kind == 'thinking' and isinstance(part.get('value'), str):
                    blocks.append(ReasoningBlock(cast(str, part['value'])))
                elif kind == 'attachment':
                    name = ' '.join((_text(part.get('name')) or 'recorded item').split())
                    blocks.append(AttachmentBlock('Attachment: ' + name[:160]))
                elif kind == 'toolInvocation' and _text(part.get('toolName')) is not None:
                    call_id = _text(part.get('toolCallId'))
                    arguments = part.get('arguments') if isinstance(part.get('arguments'), str) else '{}'
                    result = part.get('result')
                    output = RecordedOutput((TextBlock(result),), source_line=response_line) if isinstance(result, str) else None
                    blocks.append(ToolBlock(call_id, cast(str, part['toolName']), cast(str, arguments), output,
                                            response_line))
                else:
                    unsupported = True
        elif response is not None:
            unsupported = True
        if blocks:
            response_id = _text(request.get('responseId')) or request_id + ':response'
            entries.append(Message(response_id, 'assistant', tuple(blocks),
                                   timestamp=(_timestamp(request.get('responseTimestamp')) or
                                              _timestamp(request.get('timestamp')) or
                                              _timestamp(state.get('creationDate')))))
        if unsupported:
            entries.append(Notice(f'copilot-vscode-unsupported-{ordinal}',
                                  'Unsupported Copilot response part',
                                  f'A recorded response part at line {response_line} could not be displayed.'))
    if not entries:
        raise TranscriptUnavailable('No supported Copilot chat records were found.', 'unsupported')
    warnings = ('The final VS Code operation is incomplete; the last complete state is shown.',) if snapshot.pending_tail else ()
    title = _text(state.get('customTitle')) or 'Copilot chat ' + expected_raw_id[:8]
    return Transcript(scoped_session_id, scoped_native_id, title[:160], 'copilot-vscode',
                      tuple(entries), cwd=None, started=_timestamp(state.get('creationDate')),
                      warnings=warnings)
