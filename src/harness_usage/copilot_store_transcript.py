"""Read retained Copilot turn summaries on demand, outside the usage ledger."""
from pathlib import Path
import sqlite3

from .copilot_store_reader import session_store_connection
from .transcript import MAX_TRANSCRIPT_BYTES, Message, Notice, TextBlock, Transcript, TranscriptUnavailable

MAX_TURNS = 100_000


def read_store_transcript(path: Path, roots: tuple[str, ...], native_id: str,
                          branch: str | None = None) -> Transcript:
    if branch is not None:
        raise TranscriptUnavailable('Turn summaries do not contain recorded branches.', 'invalid_branch')
    if '\0' in str(path) or '..' in path.parts or any('\0' in root for root in roots):
        raise TranscriptUnavailable('The registered source path is invalid.', 'invalid_source')
    root = next((Path(value).resolve() for value in roots
                 if path.is_relative_to(Path(value).resolve())), None)
    if root is None or path == root:
        raise TranscriptUnavailable('This source is outside the configured source folders.', 'invalid_source')
    sidecars = tuple(path.with_name(path.name + suffix) for suffix in ('-wal', '-shm'))
    if any(part.is_symlink() for part in (path, *path.parents, *sidecars) if part.is_relative_to(root)):
        raise TranscriptUnavailable('The registered source path contains a symbolic link.', 'invalid_source')
    entries: list[Message] = []
    try:
        with session_store_connection(path, max_bytes=MAX_TRANSCRIPT_BYTES) as db:
            if db.execute('SELECT 1 FROM sessions WHERE id=?', (native_id,)).fetchone() is None:
                raise TranscriptUnavailable('The source no longer contains this session.', 'changed')
            columns = {row[1]: row[2].upper() for row in db.execute('PRAGMA table_info(turns)')}
            required = {'session_id': 'TEXT', 'turn_index': 'INTEGER', 'user_message': 'TEXT',
                        'assistant_response': 'TEXT', 'timestamp': 'TEXT'}
            if any(columns.get(name) != kind for name, kind in required.items()):
                raise TranscriptUnavailable('Retained turn summaries are unavailable.', 'unsupported')
            count, size = db.execute(
                'SELECT count(*),COALESCE(sum(COALESCE(length(CAST(user_message AS BLOB)),0) '
                '+ COALESCE(length(CAST(assistant_response AS BLOB)),0)),0) FROM turns WHERE session_id=?',
                (native_id,)).fetchone()
            if size > MAX_TRANSCRIPT_BYTES or count > MAX_TURNS:
                raise TranscriptUnavailable('Retained turn summaries exceed the transcript limit.', 'too_large')
            seen: set[int] = set()
            for index, user, assistant, timestamp in db.execute(
                    'SELECT turn_index,user_message,assistant_response,timestamp FROM turns '
                    'WHERE session_id=? ORDER BY turn_index', (native_id,)):
                if type(index) is not int or index < 0 or index in seen:
                    raise TranscriptUnavailable('Retained turn ordering is unsupported.', 'unsupported')
                seen.add(index)
                stamp = timestamp if isinstance(timestamp, str) and len(timestamp) <= 64 else None
                for role, text in (('user', user), ('assistant', assistant)):
                    if text is None or text == '':
                        continue
                    if not isinstance(text, str):
                        raise TranscriptUnavailable('Retained turn content is unsupported.', 'unsupported')
                    entries.append(Message(f'turn-{index}-{role}', 'user' if role == 'user' else 'assistant',
                                           (TextBlock(text),), timestamp=stamp))
    except OSError as error:
        raise TranscriptUnavailable('The registered session store is unavailable.', 'missing') from error
    except (sqlite3.Error, ValueError) as error:
        if str(error) == 'session_store_too_large':
            raise TranscriptUnavailable('The session store exceeds the 128 MiB source limit.', 'too_large') from error
        if str(error) == 'source_changed_during_snapshot':
            raise TranscriptUnavailable('The session store changed while reading. Reload to try again.', 'changed') from error
        if str(error) in ('symlink_session_store', 'invalid_session_store_file'):
            raise TranscriptUnavailable('The registered source path is invalid.', 'invalid_source') from error
        raise TranscriptUnavailable('Retained turn summaries are unavailable.', 'unsupported') from error
    if not entries:
        raise TranscriptUnavailable('No retained turn summaries are available.', 'unsupported')
    return Transcript('copilot-cli:' + native_id, native_id, 'Copilot CLI session ' + native_id[:8],
                      'copilot-cli', (Notice('lossy-turn-summary', 'Lossy turn-summary view',
                          'Retained database text only; tool calls, reasoning, and event history '
                          'are not reconstructed.'), *entries))
