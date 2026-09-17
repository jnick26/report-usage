"""On-demand Claude Code transcript projection from registered source bytes."""
import json
from typing import Literal, cast

from .transcript import (AttachmentBlock, Message, Notice, ReasoningBlock, RecordedOutput,
                         TextBlock, ToolBlock, Transcript, TranscriptEntry,
                         TranscriptUnavailable, read_jsonl)

CHAIN_TYPES = frozenset(('user', 'assistant', 'system', 'progress', 'attachment'))


def _text(value: object) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _chain_type(value: object) -> bool:
    return isinstance(value, str) and value in CHAIN_TYPES


def _result_blocks(content: object, line: int) -> tuple[TextBlock | AttachmentBlock, ...]:
    if isinstance(content, str):
        return (TextBlock(content),)
    if not isinstance(content, list):
        return (AttachmentBlock(f'Unsupported Claude tool result at line {line}'),)
    blocks: list[TextBlock | AttachmentBlock] = []
    for value in content:
        if isinstance(value, dict) and value.get('type') == 'text' and isinstance(value.get('text'), str):
            blocks.append(TextBlock(cast(str, value['text'])))
        else:
            blocks.append(AttachmentBlock(f'Unsupported Claude tool result at line {line}'))
    return tuple(blocks)


def parse_claude_transcript(payload: bytes, expected_id: str, agent_id: str | None = None,
                            branch: str | None = None) -> Transcript:
    if branch is not None:
        raise TranscriptUnavailable('Claude transcripts expose one reconstructed display chain.', 'invalid_branch')
    records, warnings = read_jsonl(payload)
    if any(_text(record.value.get('uuid')) is not None
           and _text(record.value.get('sessionId')) != expected_id for record in records):
        raise TranscriptUnavailable('The source now contains a different session. Import sources again.', 'changed')
    ignored = [record for record in records if _text(record.value.get('uuid')) is not None
               and not _chain_type(record.value.get('type'))]
    nodes = [(record.line, record.value) for record in records
             if _text(record.value.get('uuid')) is not None and _chain_type(record.value.get('type'))]
    if not nodes:
        raise TranscriptUnavailable('No supported Claude transcript records were found.', 'unsupported')
    by_id = {cast(str, value['uuid']): (line, value) for line, value in nodes}
    visible = [(line, value) for line, value in nodes if value.get('type') in ('user', 'assistant')]
    if not visible:
        raise TranscriptUnavailable('No supported Claude transcript records were found.', 'unsupported')
    if agent_id is not None:
        leaf = visible[-1]
    else:
        parent_ids = {_text(value.get('parentUuid')) for _, value in nodes}
        terminals = [(line, value) for line, value in visible if value['uuid'] not in parent_ids]
        candidates = terminals or visible
        preferred = [(line, value) for line, value in candidates
                     if not value.get('isSidechain') and not value.get('teamName') and not value.get('isMeta')]
        leaf = (preferred or candidates)[-1]
    chain: list[tuple[int, dict[str, object]]] = []
    current: tuple[int, dict[str, object]] | None = leaf
    seen: set[str] = set()
    chain_notice: Notice | None = None
    while current is not None:
        line, value = current
        identity = cast(str, value['uuid'])
        if identity in seen:
            chain_notice = Notice('claude-chain-cycle', 'Incomplete Claude chain', 'A cycle in the recorded parent chain was not followed.')
            break
        seen.add(identity)
        chain.append(current)
        parent = _text(value.get('parentUuid'))
        if parent is None:
            break
        current = by_id.get(parent)
        if current is None:
            chain_notice = Notice('claude-chain-missing', 'Incomplete Claude chain', 'A recorded parent was unavailable.')
            break
    chain.reverse()

    results: dict[str, RecordedOutput] = {}
    for line, value in chain:
        if value.get('type') not in ('user', 'assistant'):
            continue
        message = value.get('message')
        if not isinstance(message, dict) or not isinstance(message.get('content'), list):
            continue
        for item in cast(list[object], message['content']):
            if isinstance(item, dict) and item.get('type') == 'tool_result' and _text(item.get('tool_use_id')) is not None:
                results[cast(str, item['tool_use_id'])] = RecordedOutput(_result_blocks(item.get('content'), line), source_line=line)

    entries: list[TranscriptEntry] = []
    entries.extend(Notice(f'claude-unsupported-{record.line}', 'Unsupported Claude entry',
                          f'A recorded entry at line {record.line} could not be displayed.')
                   for record in ignored)
    if chain_notice is not None:
        entries.append(chain_notice)
    for line, value in chain:
        record_type = value.get('type')
        if record_type not in ('user', 'assistant'):
            entries.append(Notice(f'claude-internal-{line}', 'Recorded internal entry',
                                  f'A recorded internal entry at line {line} is not displayed.'))
            continue
        message = value.get('message')
        if not isinstance(message, dict):
            entries.append(Notice(f'claude-message-{line}', 'Unsupported Claude entry',
                                  f'A recorded entry at line {line} could not be displayed.'))
            continue
        content = message.get('content')
        raw_items: list[object] = content if isinstance(content, list) else ([{'type': 'text', 'text': content}] if isinstance(content, str) else [])
        blocks: list[TextBlock | ReasoningBlock | AttachmentBlock | ToolBlock] = []
        for raw in raw_items:
            if not isinstance(raw, dict):
                blocks.append(AttachmentBlock(f'Unsupported Claude content at line {line}'))
            elif raw.get('type') == 'text' and isinstance(raw.get('text'), str):
                blocks.append(TextBlock(cast(str, raw['text'])))
            elif raw.get('type') == 'thinking' and isinstance(raw.get('thinking'), str):
                blocks.append(ReasoningBlock(cast(str, raw['thinking'])))
            elif raw.get('type') == 'tool_use' and _text(raw.get('name')) is not None:
                call_id = _text(raw.get('id'))
                arguments = json.dumps(raw.get('input'), sort_keys=True, separators=(',', ':'), ensure_ascii=False)
                blocks.append(ToolBlock(call_id, cast(str, raw['name']), arguments,
                                        results.get(call_id) if call_id is not None else None, line))
            elif raw.get('type') == 'tool_result':
                continue
            else:
                blocks.append(AttachmentBlock(f'Unsupported Claude content at line {line}'))
        if not blocks:
            continue
        role: Literal['user', 'assistant'] = 'user' if record_type == 'user' else 'assistant'
        entries.append(Message(cast(str, value['uuid']), role, tuple(blocks),
                               _text(message.get('model')),
                               'compact summary' if value.get('isCompactSummary') is True else None,
                               _text(value.get('timestamp'))))
    native_id = expected_id if agent_id is None else f'{expected_id}:agent:{agent_id}'
    return Transcript('claude:' + native_id, native_id, 'Claude session ' + expected_id[:8], 'claude',
                      tuple(entries), warnings=warnings)
