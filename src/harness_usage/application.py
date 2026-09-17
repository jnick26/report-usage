"""Local use cases: one importer and reports over committed evidence."""
from datetime import datetime
from itertools import batched
import json
import os
from pathlib import Path
from threading import Lock, Thread
from typing import Iterator, Literal
from uuid import uuid4
from zoneinfo import ZoneInfo


from .discovery import resolve_project
from .domain import Attribution, SessionId, Unassigned
from .pricing import load_bundled_catalog
from .transcript import TranscriptPage
from .transcript_access import read_transcript_page
from .reporting import Report, ReportQuery, SessionSort
from .storage import Storage, ImportStatus as ImportStatus
from .aggregate_reporting import build_aggregate_report
from .source_input import MAX_PROBE_BYTES, SourcePayload, detect_source

def local_timezone() -> str:
    configured = os.environ.get('TZ')
    if configured:
        try:
            return ZoneInfo(configured).key
        except (ValueError, KeyError):
            pass
    target = str(Path('/etc/localtime').resolve())
    return target.split('zoneinfo/', 1)[1] if 'zoneinfo/' in target else 'UTC'


def codex_titles(roots: tuple[str, ...]) -> dict[str, str]:
    """Read only the native title index beside explicitly configured rollout roots."""
    indices = {Path(root).parent / 'session_index.jsonl' for root in roots
               if Path(root).name in ('sessions', 'archived_sessions')}
    latest: dict[str, tuple[datetime, str]] = {}
    for index in sorted(indices):
        try:
            with index.open() as source:
                for line in source:
                    try:
                        row = json.loads(line)
                        if not isinstance(row, dict):
                            continue
                        identity, name = row.get('id'), row.get('thread_name')
                        updated = datetime.fromisoformat(row.get('updated_at', ''))
                        if not isinstance(identity, str) or not identity.strip() or not isinstance(name, str) or updated.tzinfo is None:
                            continue
                        name = ' '.join(name.split())
                        name = name[:159] + '…' if len(name) > 160 else name
                        key = 'codex:' + identity
                        if name and (key not in latest or updated >= latest[key][0]):
                            latest[key] = updated, name
                    except (ValueError, TypeError, RecursionError):
                        continue
        except (OSError, UnicodeError):
            continue  # Optional metadata; rollout titles and excerpts remain available.
    return {identity: item[1] for identity, item in latest.items()}


class SourceChangeDuringImport(ValueError):
    pass


class SourceFailure(Exception):
    def __init__(self, source: int, reason: Literal['unavailable', 'read_failed', 'sidecar_too_large']):
        self.code = f'source_{source}_{reason}'
        super().__init__(self.code)


class Application:
    def __init__(self, data_dir: Path, *, timezone: str | None = None):
        data_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir = data_dir
        self.timezone = ZoneInfo(timezone or local_timezone()).key
        self.storage = Storage(data_dir / 'ledger.duckdb')
        self.catalog = load_bundled_catalog()
        self._lock = Lock()
        self._worker: Thread | None = None
        self._config = data_dir / 'sources.json'
        self.storage.interrupt_runs()

    def get_roots(self) -> tuple[str, ...]:
        if not self._config.exists():
            return ()
        roots = json.loads(self._config.read_text())
        if not isinstance(roots, list) or any(not isinstance(root, str) or not Path(root).is_absolute() for root in roots):
            raise ValueError('invalid_source_configuration')
        return tuple(roots)

    def set_roots(self, roots: tuple[str, ...]) -> None:
        if type(roots) is not tuple or any(not isinstance(root, str) or not Path(root).is_absolute() for root in roots):
            raise ValueError('Source roots must be absolute directory paths')
        normalized = tuple(dict.fromkeys(str(Path(root).resolve()) for root in roots))
        with self._lock:
            if self._worker is not None and self._worker.is_alive() and normalized != self.get_roots():
                raise SourceChangeDuringImport('Wait for the current import to finish, then save your sources again.')
            pending = self._config.with_suffix('.tmp')
            pending.write_text(json.dumps(normalized))
            pending.replace(self._config)

    def transcript(self, session_id: str, branch: str | None = None) -> TranscriptPage:
        return read_transcript_page(self.storage, self.get_roots(), session_id, branch)

    def status(self) -> ImportStatus:
        return self.storage.import_status()

    def start_import(self) -> ImportStatus:
        with self._lock:
            if self._worker is not None and self._worker.is_alive():
                return self.status()
            roots = self.get_roots()
            run_id = str(uuid4())
            status = self.storage.begin_import(run_id)
            if status.run_id != run_id:
                return status
            self._worker = Thread(target=self._import, args=(run_id, roots), name='usage-import', daemon=False)
            self._worker.start()
            return status

    def _scan(self, roots: tuple[str, ...]) -> Iterator[SourcePayload]:
        files: dict[Path, int] = {}
        authorized_roots = tuple(Path(value) for value in roots)
        for index, value in enumerate(roots, 1):
            def fail(error: OSError) -> None:
                raise SourceFailure(index, 'unavailable') from error
            root = Path(value)
            try:
                if root.is_symlink() or not root.is_dir():
                    raise SourceFailure(index, 'unavailable')
                for directory, children, names in os.walk(root, followlinks=False, onerror=fail):
                    children[:] = sorted(name for name in children if not (Path(directory) / name).is_symlink())
                    for name in sorted(names):
                        path = Path(directory) / name
                        metadata_only = (name == 'workspace.yaml' and path.parent.parent.name == 'session-state'
                                         and not (path.parent / 'events.jsonl').is_file())
                        if (path.suffix == '.jsonl' or path.parent.name == 'chatSessions' and path.suffix == '.json'
                                or metadata_only) and not path.is_symlink():
                            files.setdefault(path, index)
            except OSError as error:
                raise SourceFailure(index, 'unavailable') from error
        for path, index in sorted(files.items()):
            try:
                data = path.read_bytes()
                context: tuple[tuple[str, bytes], ...] = ()
                sidecar = (path.parent.parent / 'workspace.json' if path.parent.name == 'chatSessions'
                           else path.parent / 'workspace.yaml' if path.name == 'events.jsonl' and path.parent.parent.name == 'session-state'
                           else None)
                authorized = sidecar is not None and any(
                    sidecar.is_relative_to(root) and sidecar.resolve().is_relative_to(root.resolve())
                    for root in authorized_roots
                )
                if sidecar is not None and authorized and not sidecar.is_symlink() and sidecar.is_file():
                    with sidecar.open('rb') as source:
                        sidecar_data = source.read(MAX_PROBE_BYTES + 1)
                    if len(sidecar_data) > MAX_PROBE_BYTES:
                        raise SourceFailure(index, 'sidecar_too_large')
                    context = ((sidecar.name, sidecar_data),)
            except OSError as failure:
                raise SourceFailure(index, 'read_failed') from failure
            payload = SourcePayload(str(path), data, context)
            kind = detect_source(payload)
            if kind is None and path.suffix == '.jsonl':
                first = data.split(b'\n', 1)[0]
                if len(first) <= MAX_PROBE_BYTES:
                    try:
                        header = json.loads(first)
                    except (ValueError, UnicodeError, RecursionError):
                        header = None
                    if not isinstance(header, dict) or header.get('type') not in ('session', 'session_meta'):
                        continue
                else:
                    continue
            if kind is not None or path.suffix == '.jsonl':
                yield payload

    def _import(self, run_id: str, roots: tuple[str, ...]) -> None:
        processed = 0
        error: str | None = None
        try:
            found: set[str] = set()
            for batch in batched(self._scan(roots), 32):
                self.storage.import_sources(batch)
                found.update(payload.locator for payload in batch)
                processed += len(batch)
                self.storage.advance_import(run_id, processed)
            self.storage.mark_missing(tuple(locator for locator in self.storage.locators() if locator not in found))
        except SourceFailure as failure:
            error = failure.code
        except Exception:
            # Source text and exception messages can contain transcript material.
            error = 'import_failed'
        finally:
            try:
                self._assign_projects()
            except Exception:
                error = error or 'project_discovery_failed'
            self.storage.finish_import(run_id, processed, error)

    def _assign_projects(self) -> None:
        sessions, previous = self.storage.attribution_input()
        resolved: dict[tuple[str | None, Attribution | None], Attribution] = {}
        attributions: dict[str, Attribution] = {}
        for session in sessions:
            old = previous.get(session.id)
            if (str(session.id).startswith('copilot-cli:') and session.cwd is None
                    and isinstance(old, Unassigned)
                    and old.reason == 'workspace_changed_cumulative_scope'):
                attributions[session.id] = old
                continue
            key = (session.cwd, old)
            if key not in resolved:
                resolved[key] = resolve_project(session.cwd, previous=old)
            attributions[session.id] = resolved[key]
        self.storage.save_attributions(attributions)
        self.storage.save_codex_titles(codex_titles(self.get_roots()))

    def report(self, query: ReportQuery, *, include_sessions: bool = True, session_page: int | None = None,
               page_size: int = 50, session_id: SessionId | None = None, session_sort: SessionSort = 'started') -> Report:
        return build_aggregate_report(self.storage, query, self.catalog, include_sessions=include_sessions,
                                    session_page=session_page, page_size=page_size, session_id=session_id,
                                    session_sort=session_sort)

    def close(self) -> None:
        worker = self._worker
        if worker is not None:
            worker.join()
        self.storage.close()
