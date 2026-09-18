"""Read a registered session source on demand, without loading usage observations."""
from dataclasses import replace
import os
from pathlib import Path
import stat

from .storage import Storage
from .pricing import Catalog, load_bundled_catalog
from .transcript import MAX_TRANSCRIPT_BYTES, TranscriptLink, TranscriptPage, TranscriptUnavailable


def source_snapshot(locator: str, roots: tuple[str, ...]) -> bytes:
    if '\0' in locator or any('\0' in value for value in roots):
        raise TranscriptUnavailable('The registered source path is invalid.', 'invalid_source')
    path = Path(locator)
    root = next((Path(value).resolve() for value in roots
                 if path.is_relative_to(Path(value).resolve())), None)
    if root is None or path == root or '..' in path.parts:
        raise TranscriptUnavailable('This source is outside the configured source folders.', 'invalid_source')
    descriptors: list[int] = []
    try:
        # Walk from the authorized directory without following replaceable symlinks.
        descriptors.append(os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW))
        parts = path.relative_to(root).parts
        for part in parts[:-1]:
            descriptors.append(os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                                       dir_fd=descriptors[-1]))
        fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                     dir_fd=descriptors[-1])
        descriptors.append(fd)
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise TranscriptUnavailable('The registered source is not a regular file.', 'invalid_source')
        if before.st_size > MAX_TRANSCRIPT_BYTES:
            raise TranscriptUnavailable('This transcript exceeds the 128 MiB source limit.', 'too_large')
        with os.fdopen(os.dup(fd), 'rb') as stream:
            payload = stream.read(MAX_TRANSCRIPT_BYTES + 1)
        after = os.fstat(fd)
        if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise TranscriptUnavailable('The session changed while reading. Reload to try again.', 'changed')
        if len(payload) > MAX_TRANSCRIPT_BYTES:
            raise TranscriptUnavailable('This transcript exceeds the 128 MiB source limit.', 'too_large')
        return payload
    except OSError as error:
        raise TranscriptUnavailable('The registered session source is unavailable.', 'missing') from error
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def read_transcript_page(storage: Storage, roots: tuple[str, ...], session_id: str,
                         branch: str | None = None, *, catalog: Catalog | None = None) -> TranscriptPage:
    from .codex_transcript import parse_codex_transcript
    from .claude_reader import claude_locator_identity
    from .claude_transcript import parse_claude_transcript
    from .copilot_vscode_reader import SUPPORTED_PROFILES as VSCODE_PROFILES
    from .copilot_vscode_transcript import parse_copilot_vscode_transcript
    from .copilot_cli_transcript import parse_copilot_cli_transcript
    from .copilot_store_transcript import read_store_transcript
    from .pi_transcript import parse_pi_transcript

    with storage.connect() as db:
        db.execute('BEGIN')
        sessions = {row['id']: row for row in db.execute(
            'SELECT s.*,COALESCE(t.title,s.display_name) AS native_title FROM session_view s '
            'LEFT JOIN codex_title t ON t.session_id=s.id')}
        session = sessions.get(session_id)
        if session is None:
            raise TranscriptUnavailable('This session is not registered.', 'missing')
        if session['harness'] == 'copilot-vscode':
            sources = db.execute(
                'SELECT DISTINCT g.locator,e.session_native_id,e.representation FROM source_generation g '
                'JOIN copilot_vscode_evidence e ON e.source_id=g.id '
                'JOIN decision d ON d.observation_id=e.observation_id '
                'WHERE g.session_id=? AND g.profile IN (?,?) AND g.availability=\'available\' AND d.state=\'selected\' '
                'AND NOT EXISTS (SELECT 1 FROM source_generation newer '
                'WHERE newer.locator=g.locator AND newer.generation>g.generation) '
                "ORDER BY (e.representation='operation_log') DESC,g.locator",
                (session_id, *VSCODE_PROFILES)).fetchall()
            if not sources:
                locators = tuple(row[0] for row in db.execute(
                    'SELECT DISTINCT locator FROM source_generation WHERE session_id=? AND profile IN (?,?)',
                    (session_id, *VSCODE_PROFILES)))
                current = tuple(db.execute(
                    'SELECT availability FROM source_generation WHERE locator=? ORDER BY generation DESC LIMIT 1',
                    (locator,)).one()[0] for locator in locators)
                if current and all(state == 'missing' for state in current):
                    raise TranscriptUnavailable('The registered VS Code chat sources are unavailable.', 'missing')
                raise TranscriptUnavailable('No unambiguous current VS Code chat source is available.', 'changed')
        elif session['harness'] == 'copilot-cli':
            locator_predicate = 'g.locator=?' if session['parent_locator'] is not None else 'g.session_id=?'
            locator_value = session['parent_locator'] or session_id
            sources = db.execute(
                'SELECT g.locator,0 AS store FROM source_generation g WHERE ' + locator_predicate +
                " AND g.availability='available' "
                'AND NOT EXISTS (SELECT 1 FROM source_generation newer '
                'WHERE newer.locator=g.locator AND newer.generation>g.generation) '
                "ORDER BY (g.availability='available') DESC,g.complete_bytes DESC",
                (locator_value,)).fetchall()
            sources.extend(db.execute(
                'SELECT g.locator,1 AS store FROM copilot_store_session m '
                'JOIN source_generation g ON g.id=m.source_id WHERE m.session_id=? '
                "AND g.availability='available' "
                'AND NOT EXISTS (SELECT 1 FROM source_generation newer '
                'WHERE newer.locator=g.locator AND newer.generation>g.generation) ORDER BY g.locator',
                (session_id,)).fetchall())
            if not sources:
                registered = db.execute(
                    'SELECT 1 FROM source_generation g WHERE ' + locator_predicate + ' LIMIT 1',
                    (locator_value,)).fetchone()
                if registered:
                    raise TranscriptUnavailable('The registered Copilot CLI source is unavailable.', 'missing')
        else:
            sources = db.execute(
                'SELECT g.locator FROM source_generation g WHERE g.session_id=? '
                'AND NOT EXISTS (SELECT 1 FROM source_generation newer '
                'WHERE newer.locator=g.locator AND newer.generation>g.generation) '
                "ORDER BY (g.availability='available') DESC,g.complete_bytes DESC,g.locator",
                (session_id,)).fetchall()
        families, _ = storage._families(db, sessions, storage._attributions(db))

    def link(identity: str) -> TranscriptLink:
        row = sessions[identity]
        return TranscriptLink(identity, row['native_title'] or row['title_excerpt'] or 'Session ' + row['native_id'][:8])

    family = families.get(session_id)
    parent = link(str(family.id)) if family and family.id != session_id else None
    children = tuple(link(sid) for sid, member in families.items()
                     if family and member.id == session_id and sid != session_id)
    error = TranscriptUnavailable('No original source is registered for this session.', 'missing')
    for source in sources:
        try:
            if session['harness'] == 'copilot-cli' and source['store']:
                transcript = read_store_transcript(Path(source['locator']), roots, session['native_id'], branch)
                title = session['native_title'] or session['title_excerpt'] or transcript.title
                return TranscriptPage(replace(transcript, title=title), parent, children)
            payload = source_snapshot(source['locator'], roots)
        except TranscriptUnavailable as unavailable:
            if unavailable.kind == 'invalid_branch':
                raise
            error = unavailable
            continue
        if session['harness'] == 'pi':
            transcript = parse_pi_transcript(payload, session_id, branch)
        elif session['harness'] == 'codex':
            transcript = parse_codex_transcript(payload, session_id, branch)
        elif session['harness'] == 'claude':
            identity = claude_locator_identity(source['locator'])
            if identity is None:
                error = TranscriptUnavailable('The registered Claude source path is invalid.', 'changed')
                continue
            parent_id, agent_id, _ = identity
            transcript = parse_claude_transcript(payload, parent_id, agent_id, branch)
        elif session['harness'] == 'copilot-vscode':
            transcript = parse_copilot_vscode_transcript(
                payload, source['session_native_id'], session_id, session['native_id'],
                representation=source['representation'], branch=branch)
        elif session['harness'] == 'copilot-cli':
            native = session['native_id']
            raw_id, separator, agent_id = native.partition(':agent:')
            try:
                transcript = parse_copilot_cli_transcript(
                    payload, raw_id, session_id,
                    agent_id=agent_id if separator else None, branch=branch)
            except TranscriptUnavailable as unavailable:
                if unavailable.kind != 'unsupported':
                    raise
                error = unavailable
                continue
        else:
            error = TranscriptUnavailable('This transcript source is not supported.', 'unsupported')
            continue
        if transcript.native_id != session['native_id']:
            error = TranscriptUnavailable('The source now contains a different session. Import sources again.', 'changed')
            continue
        title = session['native_title'] or session['title_excerpt'] or transcript.title
        transcript = replace(transcript, title=title)
        if transcript.harness == 'pi':
            from .transcript_costs import pi_transcript_costs
            costs = pi_transcript_costs(storage, transcript, source['locator'], payload, catalog or load_bundled_catalog())
            return TranscriptPage(transcript, parent, children, costs)
        return TranscriptPage(transcript, parent, children)
    raise error
