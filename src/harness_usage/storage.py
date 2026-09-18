"""DuckDB-owned representations and atomic imports. No transcript retention."""
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, cast
import duckdb

from .database import Connection, Row, open_database, writer_lock

from .accounting import Candidate, MEASURES, reconcile
from .codex_accounting import CodexAppearance, CodexCandidate, reconcile_codex
from .claude_accounting import ClaudeCandidate, claude_output_variations, reconcile_claude
from .copilot_vscode_accounting import CopilotVscodeCandidate, CopilotVscodeSource, reconcile_copilot_vscode
from .copilot_cli_accounting import CopilotCLICandidate, reconcile_copilot_cli
from .domain import (Known, Unknown, NotApplicable, TokenValue, TokenBreakdown, TokenEvidence,
                     Point, Interval, Undated, TimeEvidence, RecordedEstimate, MissingEstimate,
                     RecordedMoney, MeasuredQuantity, ModelIdentity, SessionId, ObservationId, Assigned, Unassigned, Attribution, ProjectId, ContractViolation)
from .pi_reader import (read_pi, Diagnostic, ReadBatch, RejectedSource, UsageRecord, EntryIdentity,
                        SessionMetadata, DelegationRef, read_metadata)
from .reporting import AllTime, DIAGNOSTIC_CODES, DateRange, ReportQuery, SelectedContribution, SessionFamily
from .source_input import SourcePayload, detect_source

if TYPE_CHECKING:
    from .copilot_store_reader import CopilotStoreCall, CopilotStoreSnapshot

EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
PROFILE = 'pi-v3/0.85.1-shape-1'
DELEGATION_PROFILE = 'pi-delegation-3-title-1'
VSCODE_PRESENCE = 'copilot_vscode_presence_v1'
VSCODE_OUTPUT_LOWER_BOUND = 'last_successful_call'
VSCODE_RETAINED_CONFLICT = 'copilot_vscode_retained_conflict_v1'


def micros(at: datetime | None) -> int | None:
    if at is None:
        return None
    delta = at - EPOCH
    return (delta.days * 86400 + delta.seconds) * 1_000_000 + delta.microseconds


def from_micros(value: int | None) -> datetime | None:
    return EPOCH + timedelta(microseconds=value) if value is not None else None


def report_diagnostics(db: Connection, population: Literal['base', 'relevant']) -> dict[str, set[str]]:
    """Read fixed diagnostic codes attached to this transaction's scoped evidence."""
    allowed = ','.join('?' for _ in DIAGNOSTIC_CODES)
    query = (f'SELECT DISTINCT r.id,d.code FROM {population} r JOIN appearance a ON a.observation_id=r.id '
             'JOIN diagnostic d ON d.source_id=a.source_id AND '
             '(d.observation_id=r.id OR (d.observation_id IS NULL AND (d.line IS NULL OR d.line=a.line))) '
             'JOIN source_generation g ON g.id=d.source_id '
             f'WHERE d.code IN ({allowed}) '
             'AND NOT EXISTS(SELECT 1 FROM source_generation newer WHERE newer.locator=g.locator AND newer.generation>g.generation) '
             f"UNION SELECT r.id,'saved_history' FROM {population} r JOIN appearance a ON a.observation_id=r.id "
             "JOIN source_generation s ON s.id=a.source_id WHERE s.availability<>'available' "
             "AND NOT EXISTS(SELECT 1 FROM appearance current JOIN source_generation available ON available.id=current.source_id "
             "WHERE current.observation_id=r.id AND available.availability='available')")
    result: dict[str, set[str]] = {}
    for identity, code in db.execute(query, DIAGNOSTIC_CODES):
        result.setdefault(identity, set()).add(code)
    return result


def digest(value: str | bytes) -> str:
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def vscode_compatibility(presence: str, raw_input: int | None, raw_output: int | None,
                         raw_cache_read: int | None, raw_cache_write: int | None,
                         state: str, reason: str | None, quantity_state: str | None,
                         quantity_amount: Decimal | None, output_lower_bound: bool = False,
                         retained_conflict: str | None = None) -> str:
    return digest(json.dumps((presence, raw_input, raw_output, raw_cache_read, raw_cache_write,
                              state, reason, quantity_state,
                              str(quantity_amount) if quantity_amount is not None else None)
                             + ((True,) if output_lower_bound else ())
                             + ((retained_conflict,) if retained_conflict is not None else ())))


def token_data(value: TokenValue) -> tuple[str, int | None, str | None]:
    if isinstance(value, Known):
        return 'known', value.value, None
    return ('unknown' if isinstance(value, Unknown) else 'not_applicable'), None, value.reason


def signature(record: UsageRecord, *, include_safe_facts: bool = False) -> str:
    tokens = [token_data(v) for v in (*record.tokens.buckets.values, record.tokens.reported_total, record.tokens.reasoning, record.tokens.cache_write_1h)]
    money = (str(record.money.amount), tuple((k, str(v)) for k, v in record.money.components)) if isinstance(record.money, RecordedEstimate) else record.money.reason
    if isinstance(record.time, Point):
        temporal: object = ('point', micros(record.time.at))
    elif isinstance(record.time, Interval):
        temporal = ('interval', micros(record.time.start), micros(record.time.end))
    else:
        temporal = ('undated', record.time.reason)
    payload: tuple[object, ...] = (record.entry.parent_id, record.kind, temporal, record.model.provider,
                                  record.model.model, tokens, money, record.stop_reason, record.tool_call_id)
    if record.quantities:
        payload += ([(value.measure, value.state, str(value.amount) if value.amount is not None else None,
                      value.reason, value.lower_bound, value.source_ref) for value in record.quantities],)
    if include_safe_facts:
        payload += (record.safe_facts,)
    return digest(json.dumps(payload, sort_keys=True))


@dataclass(frozen=True, slots=True)
class LedgerObservation:
    id: str
    session_id: str
    record: UsageRecord
    decisions: Mapping[str, str]
    reasons: tuple[str, ...]
    diagnostics: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Snapshot:
    revision: int
    sessions: tuple[SessionMetadata, ...]
    observations: tuple[LedgerObservation, ...]
    diagnostics: tuple[str, ...]
    attributions: Mapping[str, Attribution]


@dataclass(frozen=True, slots=True)
class ReportInput:
    revision: int
    contributions: tuple[SelectedContribution, ...]
    diagnostics: tuple[str, ...]


RunState = Literal['idle', 'running', 'succeeded', 'failed', 'interrupted']
ImportPhase = Literal['discovering', 'checking', 'finalizing']


@dataclass(frozen=True, slots=True)
class ImportStatus:
    run_id: str | None
    state: RunState
    files_processed: int
    revision: int
    error: str | None
    phase: ImportPhase | None = None
    files_checked: int = 0
    files_total: int | None = None

    def __post_init__(self) -> None:
        if self.state not in ('idle', 'running', 'succeeded', 'failed', 'interrupted'):
            raise ContractViolation('invalid_run_state')
        if any(type(value) is not int or value < 0 for value in (self.files_processed, self.revision)):
            raise ContractViolation('invalid_run_progress')
        if (self.state == 'idle') != (self.run_id is None) or (self.run_id is not None and (not isinstance(self.run_id, str) or not self.run_id)):
            raise ContractViolation('invalid_run_identity')
        if self.state in ('failed', 'interrupted'):
            if not isinstance(self.error, str) or not self.error:
                raise ContractViolation('run_error_required')
        elif self.error is not None:
            raise ContractViolation('unexpected_run_error')


class Storage:
    def __init__(self, path: Path):
        self.path = path.with_suffix('.duckdb') if path.suffix == '.sqlite3' else path
        self.path = self.path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._write_lock = writer_lock(self.path)
        with self._write_lock:
            legacy = self.path.with_suffix('.sqlite3')
            if not self.path.exists() and legacy.exists():
                from .migrate_sqlite import migrate_sqlite
                migrate_sqlite(legacy, self.path)
            if self.path.exists():
                from .migrate_duckdb import migrate_duckdb_v5, migrate_duckdb_v6, schema_version
                version = schema_version(self.path)
                if version == 5:
                    migrate_duckdb_v5(self.path)
                elif version == 6:
                    migrate_duckdb_v6(self.path)
                elif version not in (None, 7):
                    raise ValueError('unsupported_schema')
            self._db: duckdb.DuckDBPyConnection | None = open_database(self.path)
            with self.connect() as db:
                exists = db.execute("SELECT 1 FROM information_schema.tables WHERE table_name='ledger_meta'").fetchone()
                if not exists:
                    db.raw.execute(Path(__file__).with_name('schema.sql').read_text())
                elif db.execute('SELECT schema_version FROM ledger_meta').one()[0] != 7:
                    raise ValueError('unsupported_schema')
                db.execute('CREATE INDEX IF NOT EXISTS decision_nonexcluded ON decision(observation_id)')

    @contextmanager
    def connect(self, *, write: bool = False) -> Iterator[Connection]:
        with self._write_lock if write else nullcontext():
            if self._db is None:
                self._db = open_database(self.path)
            db = Connection(self._db.cursor())
            try:
                db.execute('BEGIN')
                yield db
                db.commit()
            except BaseException:
                try:
                    db.rollback()
                except duckdb.TransactionException:
                    pass  # A failed commit can already have rolled back.
                raise
            finally:
                db.raw.close()

    def close(self) -> None:
        if self._db is not None:
            self._db.close()
            self._db = None

    def interrupt_runs(self) -> None:
        with self.connect(write=True) as db:
            db.execute("UPDATE import_run SET state='interrupted',finished_us=GREATEST(started_us,?),error_code='process_interrupted' WHERE state='running'", (micros(datetime.now(UTC)),))

    def import_status(self) -> ImportStatus:
        with self.connect() as db:
            db.execute('BEGIN')
            return self._import_status(db)

    @staticmethod
    def _import_status(db: Connection) -> ImportStatus:
        row = db.execute('SELECT * FROM import_run ORDER BY ordinal DESC LIMIT 1').fetchone()
        revision = int(db.execute('SELECT revision FROM ledger_meta').one()[0])
        return ImportStatus(row['id'], cast(RunState, row['state']), row['files_processed'], revision, row['error_code']) if row else ImportStatus(None, 'idle', 0, revision, None)

    def begin_import(self, run_id: str) -> ImportStatus:
        ImportStatus(run_id, 'running', 0, 0, None)
        with self.connect(write=True) as db:
            db.execute('BEGIN IMMEDIATE')
            current = self._import_status(db)
            if current.state == 'running':
                return current
            db.execute("INSERT INTO import_run(id,state,started_us,finished_us,files_processed,revision,error_code) VALUES(?,'running',?,NULL,0,?,NULL)", (run_id, micros(datetime.now(UTC)), current.revision))
            return ImportStatus(run_id, 'running', 0, current.revision, None)

    def advance_import(self, run_id: str, processed: int) -> None:
        ImportStatus(run_id, 'running', processed, 0, None)
        with self.connect(write=True) as db:
            cursor = db.execute("UPDATE import_run SET files_processed=?,revision=(SELECT revision FROM ledger_meta) WHERE id=? AND state='running' AND files_processed<=?", (processed, run_id, processed))
            if cursor.rowcount != 1:
                raise ContractViolation('invalid_run_progress')

    def finish_import(self, run_id: str, processed: int, error: str | None) -> None:
        state: RunState = 'failed' if error is not None else 'succeeded'
        ImportStatus(run_id, state, processed, 0, error)
        with self.connect(write=True) as db:
            cursor = db.execute("UPDATE import_run SET state=?,finished_us=GREATEST(started_us,?),files_processed=?,revision=(SELECT revision FROM ledger_meta),error_code=? WHERE id=? AND state='running' AND files_processed<=?", (state, micros(datetime.now(UTC)), processed, error, run_id, processed))
            if cursor.rowcount != 1:
                raise ContractViolation('invalid_run_completion')

    def import_source(self, locator: str, data: bytes) -> int:
        return self.import_sources((SourcePayload(locator, data),))

    def import_session_store(self, snapshot: 'CopilotStoreSnapshot') -> int:
        return self.import_sources((snapshot,))

    def import_sources(self, sources: Iterable['SourcePayload | CopilotStoreSnapshot | tuple[str, bytes]']) -> int:
        """Commit a bounded batch and reconcile once; callers own batch size/progress."""
        with self.connect(write=True) as db:
            db.execute('BEGIN IMMEDIATE')
            changed = False
            changed_harnesses: set[str] = set()
            changed_locators: set[str] = set()
            before = db.total_changes
            for source in sources:
                payload = SourcePayload(*source) if isinstance(source, tuple) else source
                if not isinstance(payload, SourcePayload):
                    with db.batch():
                        imported = self._import_session_store(db, payload)
                else:
                    imported = self._import_source(db, payload)
                changed = imported or changed
                if imported:
                    changed_locators.add(payload.locator)
                    row = db.execute('SELECT s.harness FROM source_generation g JOIN session_view s ON s.id=g.session_id WHERE g.locator=? ORDER BY g.generation DESC LIMIT 1', (payload.locator,)).fetchone()
                    if row:
                        changed_harnesses.add(row[0])
            if changed:
                self._reconcile(db, changed_harnesses, changed_locators=changed_locators)
            if changed or db.total_changes != before:
                db.execute('UPDATE ledger_meta SET revision=revision+1')
            return int(db.execute('SELECT revision FROM ledger_meta').one()[0])

    def _import_session_store(self, db: Connection, snapshot: 'CopilotStoreSnapshot') -> bool:
        from .copilot_store_accounting import store_record
        previous = db.execute('SELECT * FROM source_generation WHERE locator=? ORDER BY generation DESC LIMIT 1', (snapshot.locator,)).fetchone()
        if previous and previous['sha256'] == snapshot.fingerprint and previous['profile'] == snapshot.profile and previous['availability'] == 'available':
            return False
        generation = previous['generation'] + 1 if previous else 0
        source_id = digest(snapshot.locator + ':' + str(generation))
        db.execute('INSERT INTO source_generation VALUES(?,?,?,?,NULL,?,0,0,?)',
                   (source_id, snapshot.locator, generation, snapshot.fingerprint, snapshot.profile, 'available'))
        owners = {session.session_id for session in snapshot.sessions}
        if len(owners) != len(snapshot.sessions) or any(call.session_id not in owners for call in snapshot.calls):
            raise ValueError('invalid_store_owner')
        calls_by_session: dict[str, list[CopilotStoreCall]] = {}
        for native_call in snapshot.calls:
            calls_by_session.setdefault(native_call.session_id, []).append(native_call)
        ordinal = 0
        for session in snapshot.sessions:
            sid = 'copilot-cli:' + session.session_id
            db.execute('INSERT OR IGNORE INTO session(id,harness,native_id,started_us,last_seen_us,cwd) VALUES(?,?,?,?,?,?)',
                       (sid, 'copilot-cli', session.session_id, micros(session.created_at), micros(session.updated_at), session.cwd))
            db.execute('INSERT OR IGNORE INTO session_attribution VALUES(?,?,?,?)', (sid, None, None, 'not_resolved'))
            db.execute('INSERT INTO copilot_store_session VALUES(?,?,?,?,?)',
                       (source_id, sid, micros(session.created_at), micros(session.updated_at), session.host_type))
            for call in tuple(calls_by_session.get(session.session_id, ())) or (None,):
                ordinal += 1
                raw = asdict(call) if call is not None else {'unavailable': True}
                counters = json.dumps(raw, sort_keys=True, default=str, separators=(',', ':'))
                compatibility = digest(json.dumps({key: value for key, value in raw.items() if key != 'row_id'}, sort_keys=True, default=str, separators=(',', ':')))
                record = store_record(call, source_id, ordinal, snapshot.profile)
                oid = self._insert_observation(db, source_id, sid, record)
                db.execute('INSERT INTO copilot_store_evidence VALUES(?,?,?,?,?,?,?,?)',
                           (source_id, call.row_id if call else None, oid, call.turn_index if call else None,
                            call.agent_id if call else None, call.parent_tool_call_id if call else None, counters, compatibility))
        return True

    def _import_source(self, db: Connection, payload: SourcePayload) -> bool:
        locator, data = payload.locator, payload.data
        fingerprint = payload.fingerprint()
        source_kind = detect_source(payload)
        previous = db.execute('SELECT * FROM source_generation WHERE locator=? ORDER BY generation DESC LIMIT 1', (locator,)).fetchone()
        from .claude_reader import PROFILE as CLAUDE_PROFILE
        from .copilot_vscode_reader import PROFILE as VSCODE_PROFILE, SUPPORTED_PROFILES as VSCODE_PROFILES
        from .copilot_cli_reader import CLI_PROFILE as COPILOT_CLI_PROFILE
        legacy_vscode_projection = bool(previous and previous['profile'] in VSCODE_PROFILES
                                        and previous['profile'] != VSCODE_PROFILE)
        legacy_claude_projection = bool(
            previous and str(previous['profile']).startswith('claude-code/')
            and previous['profile'] != CLAUDE_PROFILE)
        legacy_cli_projection = bool(
            previous and str(previous['profile']).startswith('copilot-cli-events/')
            and previous['profile'] != COPILOT_CLI_PROFILE)
        same_byte_claude_reprojection = bool(
            legacy_claude_projection and previous is not None
            and previous['sha256'] == fingerprint
            and previous['availability'] == 'available')
        same_byte_reclassified_rejection = bool(
            previous and previous['session_id'] is None
            and previous['profile'] == PROFILE
            and source_kind in ('codex', 'claude', 'copilot-vscode', 'copilot-cli'))
        if (previous and previous['sha256'] == fingerprint and previous['availability'] == 'available'
                and not legacy_claude_projection and not legacy_vscode_projection and not legacy_cli_projection
                and not same_byte_reclassified_rejection):
            if previous['profile'] == PROFILE:
                scanned = db.execute('SELECT profile FROM delegation_scan WHERE source_id=?', (previous['id'],)).fetchone()
                if not scanned or scanned[0] != DELEGATION_PROFILE:
                    references, excerpt = read_metadata(data)
                    self._save_delegations(db, previous['id'], references, locator)
                    db.execute('UPDATE session SET title_excerpt=? WHERE id=?', (excerpt, previous['session_id']))
            return False
        from .codex_reader import CODEX_PROFILE, CodexReadBatch, read_codex
        from .claude_reader import ClaudeReadBatch, read_claude
        from .copilot_vscode_reader import PROFILE as VSCODE_PROFILE, CopilotVscodeReadBatch, read_copilot_vscode
        from .copilot_cli_reader import (CLI_PROFILE, CopilotCLIReadBatch,
                                         read_copilot_cli)
        is_codex = source_kind == 'codex'
        is_claude = source_kind == 'claude'
        is_vscode = source_kind == 'copilot-vscode'
        is_cli = source_kind == 'copilot-cli'
        read: ReadBatch | RejectedSource
        if source_kind == 'codex':
            profile, read = CODEX_PROFILE, read_codex(data, locator=locator)
        elif source_kind == 'claude':
            profile, read = CLAUDE_PROFILE, read_claude(data, locator=locator)
        elif source_kind == 'copilot-vscode':
            profile, read = VSCODE_PROFILE, read_copilot_vscode(data, locator=locator, context=payload.context)
        elif source_kind == 'copilot-cli':
            read = read_copilot_cli(data, locator=locator, context=payload.context)
            profile = (read.evidence[0].profile
                       if isinstance(read, CopilotCLIReadBatch) and read.evidence else CLI_PROFILE)
        elif source_kind == 'pi' or source_kind is None:
            profile, read = PROFILE, read_pi(data, locator=locator, profile=PROFILE)
        else:
            profile = source_kind
            read = RejectedSource((Diagnostic('unsupported_profile', None, None, None),))
        generation = previous['generation'] + 1 if previous else 0
        source_id = digest(locator + ':' + str(generation))
        session = read.session if isinstance(read, ReadBatch) else None
        claude_cwd_conflict_line: int | None = None
        vscode_cwd_conflict = False
        cli_cwd_conflict = False
        if session:
            prior = db.execute('SELECT * FROM session_view WHERE id=?', (session.id,)).fetchone()
            if prior and is_codex and not db.execute('SELECT 1 FROM source_generation WHERE session_id=? LIMIT 1', (session.id,)).fetchone():
                db.execute('UPDATE session SET display_name=?,title_excerpt=?,started_us=?,last_seen_us=?,cwd=?,parent_locator=? WHERE id=?', (session.display_name, session.title_excerpt, micros(session.started), micros(session.last_observed), session.cwd, session.parent_locator, session.id))
                prior = None
            db.execute('INSERT OR IGNORE INTO session(id,harness,native_id,display_name,title_excerpt,started_us,last_seen_us,cwd,parent_locator) VALUES(?,?,?,?,?,?,?,?,?)',
                       (session.id, source_kind if source_kind in ('codex', 'claude', 'copilot-vscode', 'copilot-cli') else 'pi', session.native_id, session.display_name, session.title_excerpt, micros(session.started), micros(session.last_observed), session.cwd, session.parent_locator))
            db.execute("INSERT OR IGNORE INTO session_attribution VALUES(?,NULL,NULL,'not_resolved')", (session.id,))
            if is_cli and any(item.code == 'workspace_changed_cumulative_scope' for item in read.diagnostics):
                db.execute("UPDATE session_attribution SET project_id=NULL,worktree=NULL,attribution_reason='workspace_changed_cumulative_scope' WHERE session_id=?", (session.id,))
            if not prior or is_claude or (prior['cwd'] == session.cwd and (is_codex or prior['started_us'] == micros(session.started)) and prior['parent_locator'] == session.parent_locator):
                db.execute('UPDATE session SET last_seen_us=CASE WHEN last_seen_us IS NULL THEN ? WHEN ? IS NULL THEN last_seen_us ELSE GREATEST(last_seen_us,?) END WHERE id=?', (micros(session.last_observed), micros(session.last_observed), micros(session.last_observed), session.id))
            if (is_codex or is_claude) and prior:
                db.execute('UPDATE session SET started_us=CASE WHEN started_us IS NULL THEN ? WHEN ? IS NULL THEN started_us ELSE LEAST(started_us,?) END WHERE id=?', (micros(session.started), micros(session.started), micros(session.started), session.id))
            if is_vscode and prior:
                conflict = db.execute(
                    "SELECT 1 FROM diagnostic d JOIN source_generation g ON g.id=d.source_id "
                    "WHERE g.session_id=? AND d.code='workspace_attribution_conflict' LIMIT 1",
                    (session.id,)).fetchone() is not None
                merged = prior['cwd']
                if conflict:
                    merged = None
                elif session.cwd is not None:
                    if merged is None:
                        merged = session.cwd
                    elif merged != session.cwd:
                        merged = None
                        vscode_cwd_conflict = True
                db.execute('UPDATE session SET cwd=?,started_us=CASE WHEN started_us IS NULL THEN ? WHEN ? IS NULL THEN started_us ELSE LEAST(started_us,?) END WHERE id=?',
                           (merged, micros(session.started), micros(session.started), micros(session.started), session.id))
            if is_claude and prior:
                assert isinstance(read, ClaudeReadBatch)
                cwd_poisoned = db.execute(
                    "SELECT 1 FROM diagnostic d JOIN source_generation g ON g.id=d.source_id "
                    "WHERE g.session_id=? AND d.code='cwd_attribution_unavailable' "
                    "AND d.line IS NOT NULL LIMIT 1", (session.id,)).fetchone() is not None
                merged_cwd = prior['cwd']
                if cwd_poisoned or read.cwd_state == 'invalid':
                    merged_cwd = None
                elif read.cwd_state == 'valid':
                    if prior['cwd'] is None:
                        merged_cwd = session.cwd
                    elif prior['cwd'] != session.cwd:
                        merged_cwd = None
                        claude_cwd_conflict_line = read.cwd_line
                db.execute('UPDATE session SET cwd=?,parent_locator=CASE WHEN parent_locator IS DISTINCT FROM ? THEN NULL ELSE parent_locator END WHERE id=?',
                           (merged_cwd, session.parent_locator, session.id))
            if is_cli and prior:
                cumulative_changed = any(item.code == 'workspace_changed_cumulative_scope'
                                         for item in read.diagnostics)
                cwd_poisoned = db.execute(
                    "SELECT 1 FROM diagnostic d JOIN source_generation g ON g.id=d.source_id "
                    "WHERE g.session_id=? AND d.code='workspace_changed_cumulative_scope' LIMIT 1",
                    (session.id,)).fetchone() is not None
                cli_cwd_conflict = cumulative_changed or cwd_poisoned
                merged_cwd = None if cumulative_changed or cwd_poisoned else prior['cwd'] or session.cwd
                if prior['cwd'] is not None and session.cwd is not None and prior['cwd'] != session.cwd:
                    merged_cwd = None
                    cli_cwd_conflict = True
                db.execute('UPDATE session SET cwd=?,started_us=CASE WHEN started_us IS NULL THEN ? WHEN ? IS NULL THEN started_us ELSE LEAST(started_us,?) END,last_seen_us=CASE WHEN last_seen_us IS NULL THEN ? WHEN ? IS NULL THEN last_seen_us ELSE GREATEST(last_seen_us,?) END WHERE id=?',
                           (merged_cwd, micros(session.started), micros(session.started), micros(session.started),
                            micros(session.last_observed), micros(session.last_observed), micros(session.last_observed), session.id))
        db.execute('INSERT INTO source_generation VALUES(?,?,?,?,?,?,?,?,?)',
                   (source_id, locator, generation, fingerprint, session.id if session else None, profile, read.complete_bytes if isinstance(read, ReadBatch) else 0, int(read.pending_tail) if isinstance(read, ReadBatch) else 0, 'available'))
        if session and prior and not is_claude and not is_vscode and not is_cli and (prior['cwd'] != session.cwd or (not is_codex and prior['started_us'] != micros(session.started)) or prior['parent_locator'] != session.parent_locator):
            db.execute('INSERT INTO diagnostic(source_id,code) VALUES(?,?)', (source_id, 'header_conflict'))
        if session and prior and is_codex and prior['started_us'] != micros(session.started):
            db.execute('INSERT INTO diagnostic(source_id,code) VALUES(?,?)', (source_id, 'codex_header_timestamp_variation'))
        for diagnostic in read.diagnostics:
            db.execute('INSERT INTO diagnostic(source_id,line,code,measure) VALUES(?,?,?,?)', (source_id, diagnostic.line, diagnostic.code, diagnostic.measure))
        if same_byte_claude_reprojection and isinstance(read, ClaudeReadBatch):
            db.execute(
                "DELETE FROM diagnostic WHERE code='writer_version_unavailable' AND line IS NULL "
                "AND source_id IN (SELECT id FROM source_generation WHERE locator=? AND sha256=? "
                "AND profile LIKE 'claude-code/%' AND profile<>?)",
                (locator, fingerprint, CLAUDE_PROFILE))
        if claude_cwd_conflict_line is not None:
            db.execute('INSERT INTO diagnostic(source_id,line,code) VALUES(?,?,?)',
                       (source_id, claude_cwd_conflict_line, 'cwd_attribution_unavailable'))
        if vscode_cwd_conflict:
            db.execute('INSERT INTO diagnostic(source_id,code) VALUES(?,?)',
                       (source_id, 'workspace_attribution_conflict'))
        if cli_cwd_conflict:
            db.execute('INSERT INTO diagnostic(source_id,code) VALUES(?,?)',
                       (source_id, 'workspace_changed_cumulative_scope'))
        if isinstance(read, ReadBatch):
            with db.batch():
                for edge in read.entries:
                    db.execute('INSERT INTO entry_edge VALUES(?,?,?,?)', (source_id, edge.line, edge.native_id, edge.parent_id))
                if isinstance(read, CodexReadBatch):
                    db.execute('INSERT INTO codex_source VALUES(?,?,?,?)', (source_id, read.parent_thread_id, read.forked_from_id, read.root_session_id))
                    for record, evidence in zip(read.usage, read.evidence, strict=True):
                        if evidence.entry_id != record.entry.native_id:
                            raise ContractViolation('codex_evidence_identity_mismatch')
                        owner = 'codex:' + evidence.thread_id if evidence.source_kind == 'response' and evidence.thread_id else str(read.session.id)
                        db.execute('INSERT OR IGNORE INTO session(id,harness,native_id) VALUES(?,?,?)', (owner, 'codex', owner.removeprefix('codex:')))
                        db.execute('INSERT OR IGNORE INTO session_attribution VALUES(?,?,?,?)', (owner, None, None, 'missing_thread_metadata'))
                        oid = self._insert_observation(db, source_id, owner, record)
                        db.execute('INSERT INTO codex_evidence(source_id,line,observation_id,source_kind,response_id,thread_id,turn_id,mirror_response_id,cumulative_json,last_json,state,reason) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)', (source_id, record.entry.line, oid, evidence.source_kind, evidence.response_id, evidence.thread_id, evidence.turn_id, evidence.mirror_response_id,
                                   json.dumps(evidence.cumulative) if evidence.cumulative is not None else None, json.dumps(evidence.last) if evidence.last is not None else None, evidence.state, evidence.reason))
                elif isinstance(read, ClaudeReadBatch):
                    for record, item in zip(read.usage, read.evidence, strict=True):
                        oid = self._insert_observation(db, source_id, read.session.id, record, include_safe_facts=True)
                        db.execute('INSERT INTO claude_evidence(source_id,line,observation_id,entry_id,message_id,request_id,entry_uuid,agent_id,state,reason) VALUES(?,?,?,?,?,?,?,?,?,?)',
                                   (source_id, record.entry.line, oid, oid, item.message_id, item.request_id,
                                    item.entry_uuid, item.agent_id, item.state, item.reason))
                elif isinstance(read, CopilotVscodeReadBatch):
                    for record, vscode_item in zip(read.usage, read.evidence, strict=True):
                        retained_conflict = (json.dumps(vscode_item.retained_conflict)
                                             if vscode_item.retained_conflict is not None else None)
                        quantity = next((value for value in record.quantities
                                         if value.measure == 'ai_credits'), None)
                        compatibility = vscode_compatibility(
                            vscode_item.presence, vscode_item.raw_input, vscode_item.raw_output,
                            vscode_item.raw_cache_read, vscode_item.raw_cache_write,
                            vscode_item.state, vscode_item.reason,
                            quantity.state if quantity is not None else None,
                            quantity.amount if quantity is not None else None,
                            vscode_item.output_lower_bound, retained_conflict)
                        oid = self._insert_observation(
                            db, source_id, read.session.id, record,
                            fingerprint_discriminator=compatibility,
                            safe_metadata={VSCODE_PRESENCE: vscode_item.presence,
                                           **({VSCODE_RETAINED_CONFLICT: retained_conflict}
                                              if retained_conflict is not None else {})})
                        db.execute('INSERT INTO copilot_vscode_evidence(source_id,line,observation_id,profile,session_native_id,request_id,response_id,selected_model,representation,request_ordinal,evidence_kind,raw_input,raw_output,raw_cache_read,raw_cache_write,state,reason) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                                   (source_id, vscode_item.line, oid, vscode_item.profile,
                                    vscode_item.session_native_id, vscode_item.request_id,
                                    vscode_item.response_id, vscode_item.selected_model,
                                    vscode_item.representation, vscode_item.request_ordinal,
                                    vscode_item.evidence_kind, vscode_item.raw_input,
                                    vscode_item.raw_output, vscode_item.raw_cache_read,
                                    vscode_item.raw_cache_write, vscode_item.state, vscode_item.reason))
                elif isinstance(read, CopilotCLIReadBatch):
                    for record, cli_item in zip(read.usage, read.evidence, strict=True):
                        compatibility = digest(json.dumps((record.kind, record.entry.parent_id,
                                                           cli_item.event_id, cli_item.parent_event_id,
                                                           cli_item.counter_epoch, cli_item.current_model,
                                                           cli_item.counters_json),
                                                          sort_keys=True))
                        oid = self._insert_observation(db, source_id, read.session.id, record,
                                                       fingerprint_discriminator=compatibility)
                        db.execute('INSERT INTO copilot_cli_evidence(source_id,line,observation_id,profile,source_kind,event_id,parent_event_id,event_schema_version,writer_version,session_start_us,current_model,agent_id,counter_epoch,counters_json,state,reason) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                                   (source_id, cli_item.line, oid, cli_item.profile, cli_item.source_kind,
                                    cli_item.event_id, cli_item.parent_event_id, cli_item.event_schema_version,
                                    cli_item.writer_version, cli_item.session_start_us, cli_item.current_model,
                                    cli_item.agent_id, cli_item.counter_epoch, cli_item.counters_json,
                                    cli_item.state, cli_item.reason))
                    for agent_id in read.agent_ids:
                        child_native = read.raw_session_id + ':agent:' + agent_id
                        child_id = 'copilot-cli:' + child_native
                        db.execute('INSERT OR IGNORE INTO session(id,harness,native_id,started_us,last_seen_us,cwd,parent_locator) VALUES(?,?,?,?,?,?,?)',
                                   (child_id, 'copilot-cli', child_native, micros(read.session.started),
                                    micros(read.session.last_observed), read.session.cwd, locator))
                        db.execute('INSERT OR IGNORE INTO session_attribution VALUES(?,?,?,?)',
                                   (child_id, None, None, 'not_resolved'))
                else:
                    for record in read.usage:
                        self._insert_observation(db, source_id, read.session.id, record)
            # Family updates must see children inserted by the flushed batch.
            if isinstance(read, CopilotCLIReadBatch) and (cli_cwd_conflict or any(
                    item.code == 'workspace_changed_cumulative_scope' for item in read.diagnostics)):
                family = (read.session.id, read.raw_session_id + ':agent:')
                db.execute("UPDATE session SET cwd=NULL WHERE id=? OR (harness='copilot-cli' AND starts_with(native_id,?))", family)
                db.execute("UPDATE session_attribution SET project_id=NULL,worktree=NULL,attribution_reason='workspace_changed_cumulative_scope' WHERE session_id IN (SELECT id FROM session WHERE id=? OR (harness='copilot-cli' AND starts_with(native_id,?)))", family)
        if isinstance(read, ReadBatch):
            previous_meta = db.execute('SELECT complete_sha256 FROM source_metadata WHERE source_id=?', (previous['id'],)).fetchone() if previous else None
            extends = bool(previous and previous_meta and len(data) >= previous['complete_bytes'] and digest(data[:previous['complete_bytes']]) == previous_meta[0])
            existing_conflict = db.execute("SELECT 1 FROM diagnostic d JOIN source_generation s ON d.source_id=s.id WHERE s.session_id=? AND d.code='name_conflict' LIMIT 1", (read.session.id,)).fetchone()
            name_conflict = bool(prior and prior['display_name'] != read.session.display_name and not extends)
            if name_conflict:
                db.execute('INSERT INTO diagnostic(source_id,code) VALUES(?,?)', (source_id, 'name_conflict'))
            db.execute('UPDATE session SET display_name=? WHERE id=?', (None if name_conflict or existing_conflict else read.session.display_name, read.session.id))
            db.execute('UPDATE session SET title_excerpt=? WHERE id=?', (read.session.title_excerpt, read.session.id))
            db.execute('INSERT INTO source_metadata VALUES(?,?)', (source_id, digest(data[:read.complete_bytes])))
        self._save_delegations(db, source_id, read.delegations if isinstance(read, ReadBatch) else (), locator)
        return True

    @staticmethod
    def _save_delegations(db: Connection, source_id: str, references: tuple[DelegationRef, ...], locator: str) -> None:
        db.execute('DELETE FROM delegation_ref WHERE source_id=?', (source_id,))
        db.executemany('INSERT OR IGNORE INTO delegation_ref VALUES(?,?,?,?,?)', ((source_id, ref.entry_id, ref.kind, ref.value, ref.owner_path or '') for ref in references))
        canonical_path = os.path.realpath(locator)
        if os.name == 'nt':
            canonical_path = canonical_path.lower()
        db.execute('INSERT OR REPLACE INTO delegation_scan VALUES(?,?,?)', (source_id, DELEGATION_PROFILE, digest(canonical_path)))

    def _insert_observation(self, db: Connection, source: str, session: str, record: UsageRecord,
                            *, include_safe_facts: bool = False,
                            fingerprint_discriminator: str | None = None,
                            safe_metadata: Mapping[str, str] = MappingProxyType({})) -> str:
        fingerprint = signature(record, include_safe_facts=include_safe_facts)
        if fingerprint_discriminator is not None:
            fingerprint = digest(json.dumps((fingerprint, fingerprint_discriminator)))
        oid = digest(session + ':' + record.entry.native_id + ':' + fingerprint)
        at = start = end = None
        reason = None
        if isinstance(record.time, Point):
            kind, at, reason = 'point', micros(record.time.at), 'response_recorded_at'
        elif isinstance(record.time, Interval):
            kind, start, end = 'interval', micros(record.time.start), micros(record.time.end)
        else:
            kind, reason = 'undated', record.time.reason
        facts = {**dict(record.safe_facts), **safe_metadata}
        db.execute('INSERT OR IGNORE INTO observation VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                   (oid, session, record.entry.native_id, fingerprint, record.kind, kind, at, start, end, reason, record.model.provider, record.model.model, record.stop_reason, record.tool_call_id, json.dumps(facts, default=str)))
        db.execute('INSERT INTO appearance(source_id,line,observation_id) VALUES(?,?,?)', (source, record.entry.line, oid))
        values = (*record.tokens.buckets.values, record.tokens.reported_total, record.tokens.reasoning, record.tokens.cache_write_1h)
        names = ('input', 'output', 'cache_read', 'cache_write', 'reported_total', 'reasoning', 'cache_write_1h')
        for name, value in zip(names, values):
            db.execute('INSERT OR IGNORE INTO token_value VALUES(?,?,?,?,?)', (oid, name, *token_data(value)))
        if isinstance(record.money, RecordedEstimate):
            money = record.money
            db.execute('INSERT OR IGNORE INTO recorded_estimate VALUES(?,?,?,?,?,?,?)', (oid, 'known', str(money.amount), 'USD', json.dumps(dict(money.components), default=str), money.source_ref, None))
        else:
            db.execute('INSERT OR IGNORE INTO recorded_estimate VALUES(?,?,?,?,?,?,?)', (oid, 'missing', None, None, None, None, record.money.reason))
        for quantity in record.quantities:
            db.execute('INSERT OR IGNORE INTO quantity_value VALUES(?,?,?,?,?,?,?)',
                       (oid, quantity.measure, quantity.state, str(quantity.amount) if quantity.amount is not None else None,
                        quantity.reason, int(quantity.lower_bound), quantity.source_ref))
            db.execute('INSERT OR IGNORE INTO quantity_decision VALUES(?,?,?,?,?,?,?)',
                       (oid, quantity.measure, 'selected', session, None, 'independent_evidence', 'quantity-1'))
        return oid

    @staticmethod
    def _affected_sessions(db: Connection, locators: set[str]) -> set[str]:
        """Dependency closure over retained generations, including unresolved identities."""
        graph: dict[tuple[str, str], set[tuple[str, str]]] = {}

        def link(a: tuple[str, str], b: tuple[str, str]) -> None:
            graph.setdefault(a, set()).add(b)
            graph.setdefault(b, set()).add(a)

        for row in db.execute('SELECT locator,session_id FROM source_generation WHERE session_id IS NOT NULL'):
            link(('locator', row['locator']), ('session', row['session_id']))
        for row in db.execute('SELECT id,parent_locator FROM session_view WHERE parent_locator IS NOT NULL'):
            link(('session', row['id']), ('locator', row['parent_locator']))
        for row in db.execute('SELECT g.session_id,COALESCE(c.forked_from_id,c.parent_thread_id) AS parent FROM codex_source c JOIN source_generation g ON g.id=c.source_id WHERE g.session_id IS NOT NULL'):
            if row['parent']:
                link(('session', row['session_id']), ('session', 'codex:' + row['parent']))
        # A source can contain other owners; source order affects legacy replay boundaries.
        for row in db.execute('SELECT DISTINCT g.locator,o.session_id FROM appearance a JOIN source_generation g ON g.id=a.source_id JOIN observation o ON o.id=a.observation_id'):
            link(('locator', row['locator']), ('session', row['session_id']))
        for row in db.execute('SELECT g.locator,m.session_id FROM copilot_store_session m JOIN source_generation g ON g.id=m.source_id'):
            link(('locator', row['locator']), ('session', row['session_id']))
        for row in db.execute('SELECT DISTINCT o.session_id,e.response_id,e.mirror_response_id FROM codex_evidence e JOIN observation o ON o.id=e.observation_id'):
            for name in ('response_id', 'mirror_response_id'):
                if row[name] is not None:
                    link(('session', row['session_id']), ('response', row[name]))
        for row in db.execute('SELECT DISTINCT o.session_id,e.message_id,e.request_id,e.entry_uuid FROM claude_evidence e JOIN observation o ON o.id=e.observation_id'):
            for name, kind in (('message_id', 'claude_message'), ('request_id', 'claude_request'), ('entry_uuid', 'claude_entry')):
                if row[name] is not None:
                    link(('session', row['session_id']), (kind, row[name]))
        for row in db.execute('SELECT DISTINCT o.session_id,e.event_id,e.parent_event_id,e.counter_epoch FROM copilot_cli_evidence e JOIN observation o ON o.id=e.observation_id'):
            for name, kind in (('event_id', 'cli_event'), ('parent_event_id', 'cli_event'),
                               ('counter_epoch', 'cli_epoch')):
                if row[name] is not None:
                    link(('session', row['session_id']), (kind, row[name]))
        pending = [('locator', locator) for locator in locators]
        seen: set[tuple[str, str]] = set(pending)
        while pending:
            for node in graph.get(pending.pop(), ()):
                if node not in seen:
                    seen.add(node)
                    pending.append(node)
        # Absent parents connect siblings but must remain absent to the accounting rules.
        return {row[0] for row in db.execute('SELECT id FROM session_view') if ('session', row[0]) in seen}

    def _reconcile(self, db: Connection, harnesses: set[str] | None = None,
                   *, changed_locators: set[str] | None = None) -> None:
        db.execute('DROP TABLE IF EXISTS temp.reconcile_sessions')
        db.execute('CREATE TEMP TABLE reconcile_sessions(id TEXT PRIMARY KEY)')
        if changed_locators is None:
            db.execute('INSERT INTO reconcile_sessions SELECT id FROM session_view')
        else:
            affected = self._affected_sessions(db, changed_locators)
            db.executemany('INSERT INTO reconcile_sessions VALUES(?)', ((sid,) for sid in affected))
            harnesses = {row[0] for row in db.execute('SELECT DISTINCT harness FROM session_view WHERE id IN (SELECT id FROM reconcile_sessions)')}
        if harnesses is None or 'codex' in harnesses:
            self._reconcile_codex(db)
        if harnesses is None or 'claude' in harnesses:
            self._reconcile_claude(db)
        if harnesses is None or 'copilot-vscode' in harnesses:
            self._reconcile_copilot_vscode(db)
        if harnesses is None or 'copilot-cli' in harnesses:
            self._reconcile_copilot_cli(db)
            from .copilot_store_accounting import reconcile_store
            reconcile_store(db)
        if harnesses is not None and 'pi' not in harnesses:
            return
        # ponytail: rebuild decisions for whole affected components; narrower deltas need rule-level dependencies.
        locator_sessions: dict[str, set[str]] = {}
        for row in db.execute('SELECT locator,session_id FROM source_generation WHERE session_id IS NOT NULL'):
            locator_sessions.setdefault(row['locator'], set()).add(row['session_id'])
        locators = {locator: next(iter(ids)) for locator, ids in locator_sessions.items() if len(ids) == 1}
        parents: dict[str, str | None] = {}
        missing: set[str] = set()
        for row in db.execute('SELECT id,parent_locator FROM session_view'):
            parents[row['id']] = locators.get(row['parent_locator'])
            if row['parent_locator'] and row['parent_locator'] not in locators:
                missing.add(row['id'])
        conflicts = {r[0] for r in db.execute("SELECT DISTINCT s.session_id FROM diagnostic d JOIN source_generation s ON d.source_id=s.id WHERE d.code IN ('header_conflict','conflicting_header','duplicate_entry_id')")}
        edges: dict[str, dict[str, str | None]] = {}
        for row in db.execute('SELECT s.session_id,e.native_entry_id,e.parent_entry_id FROM entry_edge e JOIN source_generation s ON e.source_id=s.id WHERE s.session_id IN (SELECT id FROM reconcile_sessions)'):
            current = edges.setdefault(row['session_id'], {})
            if row['native_entry_id'] in current and current[row['native_entry_id']] != row['parent_entry_id']:
                conflicts.add(row['session_id'])
            current[row['native_entry_id']] = row['parent_entry_id']
        observations = tuple(db.execute("SELECT * FROM observation WHERE session_id LIKE 'pi:%' AND session_id IN (SELECT id FROM reconcile_sessions)"))
        by_observation: dict[str, dict[str, TokenValue]] = {}
        for row in db.execute("SELECT v.* FROM token_value v JOIN observation o ON o.id=v.observation_id WHERE o.session_id LIKE 'pi:%' AND o.session_id IN (SELECT id FROM reconcile_sessions)"):
            by_observation.setdefault(row['observation_id'], {})[row['measure']] = Known(row['amount']) if row['state'] == 'known' else Unknown(row['reason']) if row['state'] == 'unknown' else NotApplicable(row['reason'])
        prices = {row['observation_id']: row for row in db.execute("SELECT v.* FROM recorded_estimate v JOIN observation o ON o.id=v.observation_id WHERE o.session_id LIKE 'pi:%' AND o.session_id IN (SELECT id FROM reconcile_sessions)")}
        evidence: dict[str, TokenEvidence] = {}
        for oid, values in by_observation.items():
            evidence[oid] = TokenEvidence(TokenBreakdown(*(values[name] for name in MEASURES[:4])), values['reported_total'], values['reasoning'], values['cache_write_1h'])
        owners = {row['id']: row['session_id'] for row in observations}
        db.execute("DELETE FROM decision WHERE observation_id IN (SELECT id FROM observation WHERE session_id LIKE 'pi:%' AND session_id IN (SELECT id FROM reconcile_sessions))")
        for measure in MEASURES:
            candidates: list[Candidate] = []
            for row in observations:
                parent_id = edges.get(row['session_id'], {}).get(row['native_entry_id'])
                context = tuple(row[name] for name in ('kind', 'time_kind', 'at_us', 'start_us', 'end_us', 'time_reason', 'provider', 'model', 'stop_reason', 'tool_call_id'))
                value: object
                if measure == 'recorded_usd':
                    price = prices[row['id']]
                    if price['state'] == 'known':
                        decimal = Decimal(price['amount_decimal'])
                        sign, digits, exponent = decimal.as_tuple()
                        exponent = int(exponent)
                        while digits and digits[-1] == 0:
                            digits, exponent = digits[:-1], exponent + 1
                        value = ('known', sign, digits, exponent if digits else 0)
                    else:
                        value = ('missing', price['reason'])
                else:
                    tokens = evidence[row['id']]
                    value = token_data(tokens.total if measure == 'total' else by_observation[row['id']][measure])
                fingerprint = digest(json.dumps((parent_id, context, value)))
                pending = row['kind'] == 'assistant' and row['stop_reason'] not in ('stop', 'length', 'toolUse', 'error', 'aborted')
                candidates.append(Candidate(row['id'], row['session_id'], row['native_entry_id'], parent_id, fingerprint, row['kind'], pending))
            db.executemany('INSERT INTO decision VALUES(?,?,?,?,?,?,?)',
                ((decision.observation_id, measure, decision.state, owners[decision.observation_id] if decision.state == 'selected' else None, decision.canonical, decision.reason, 'pi-1')
                 for decision in reconcile(tuple(candidates), parents, edges, missing, conflicts)))

    def _reconcile_copilot_cli(self, db: Connection) -> None:
        # A present rejected replacement has no active evidence but must still
        # retire its superseded decisions. Missing valid evidence is kept below.
        db.execute(
            "DELETE FROM decision WHERE observation_id IN (SELECT id FROM observation "
            "WHERE session_id LIKE 'copilot-cli:%' AND session_id IN (SELECT id FROM reconcile_sessions))")
        db.execute(
            "DELETE FROM quantity_decision WHERE observation_id IN (SELECT id FROM observation "
            "WHERE session_id LIKE 'copilot-cli:%' AND session_id IN (SELECT id FROM reconcile_sessions))")
        rows = tuple(db.execute(
            "SELECT o.*,e.source_id,e.line AS source_line,e.event_id,e.parent_event_id,e.counter_epoch,"
            "e.source_kind,e.current_model,e.agent_id,e.state AS evidence_state,e.reason AS evidence_reason,e.counters_json,g.generation,g.availability "
            "FROM observation o JOIN copilot_cli_evidence e ON e.observation_id=o.id "
            "JOIN source_generation g ON g.id=e.source_id "
            "WHERE o.session_id IN (SELECT id FROM reconcile_sessions) "
            "AND NOT EXISTS (SELECT 1 FROM source_generation newer WHERE newer.locator=g.locator AND newer.generation>g.generation) "
            "ORDER BY e.source_id,e.line,e.ordinal"))
        if not rows:
            return
        live_sessions = {row['session_id'] for row in rows if row['availability'] == 'available'}
        rows = tuple(row for row in rows if row['availability'] == 'available' or row['session_id'] not in live_sessions)
        active_sources = {row['source_id'] for row in rows}
        session_by_event: dict[str, set[str]] = {}
        for row in db.execute(
                "SELECT DISTINCT o.session_id,e.event_id FROM copilot_cli_evidence e "
                "JOIN observation o ON o.id=e.observation_id WHERE e.event_id IS NOT NULL"):
            session_by_event.setdefault(row['event_id'], set()).add(row['session_id'])
        cross_session = {event for event, sessions in session_by_event.items() if len(sessions) > 1}
        parent_by_event: dict[tuple[str, str], set[str | None]] = {}
        for row in rows:
            parent_by_event.setdefault((row['session_id'], row['event_id']), set()).add(row['parent_event_id'])
        conflicting_parent = {key for key, parents in parent_by_event.items() if len(parents) > 1}
        token_values: dict[str, dict[str, Row]] = {}
        for row in db.execute(
                "SELECT v.* FROM token_value v JOIN observation o ON o.id=v.observation_id "
                "WHERE o.session_id IN (SELECT id FROM reconcile_sessions)"):
            token_values.setdefault(row['observation_id'], {})[row['measure']] = row
        model_rows: list[tuple[Row, dict[str, object]]] = []
        metadata_rows: list[tuple[Row, dict[str, object]]] = []
        session_rows: list[tuple[Row, dict[str, object]]] = []
        for row in rows:
            raw = json.loads(row['counters_json']) if row['counters_json'] else {}
            if not isinstance(raw, dict):
                continue
            scope = raw.get('scope')
            if scope == 'model':
                model_rows.append((row, cast(dict[str, object], raw)))
            elif scope == 'metadata':
                metadata_rows.append((row, cast(dict[str, object], raw)))
            elif scope == 'session':
                session_rows.append((row, cast(dict[str, object], raw)))
        model_sessions = {row['session_id'] for row, _ in model_rows}
        control_sessions = {row['session_id'] for row, _ in session_rows}
        started_sessions = {row['session_id'] for row, _ in metadata_rows if row['event_id'] is not None}
        metadata_rows = [(row, raw) for row, raw in metadata_rows
                         if row['event_id'] is not None or row['session_id'] not in started_sessions]
        event_facts: dict[tuple[str, str], list[tuple[object, ...]]] = {}
        for row in rows:
            event_id = row['event_id']
            if event_id is None:
                continue
            event_facts.setdefault((row['source_id'], event_id), []).append((
                row['kind'], row['source_kind'], row['parent_event_id'], row['counter_epoch'],
                row['current_model'], row['agent_id'], row['counters_json'],
                row['evidence_state'], row['evidence_reason']))
        event_signatures = {key: digest(json.dumps(sorted(facts, key=repr), default=str))
                            for key, facts in event_facts.items()}
        signatures_by_event: dict[tuple[str, str], set[str]] = {}
        for row in rows:
            if row['event_id'] is not None:
                signatures_by_event.setdefault((row['session_id'], row['event_id']), set()).add(
                    event_signatures[row['source_id'], row['event_id']])
        conflicting_event = {key for key, signatures in signatures_by_event.items()
                             if len(signatures) > 1}
        model_names = {row['model'] for row, raw in model_rows if row['model'] is not None}

        def invalid_for(raw: dict[str, object], measure: str) -> bool:
            invalid = raw.get('invalid')
            if not isinstance(invalid, list):
                return False
            names = {value for value in invalid if isinstance(value, str)}
            if 'model_shape' in names:
                return True
            if measure == 'total':
                return bool(names & {'input', 'output', 'cacheRead', 'cacheWrite', 'inputTokens',
                                     'outputTokens', 'cache_read', 'cache_write'})
            return measure in names or (measure == 'cache_read' and 'cacheRead' in names) \
                or (measure == 'cache_write' and 'cacheWrite' in names)

        def candidate(row: Row, raw: dict[str, object], measure: str,
                      value: int | Decimal | None) -> CopilotCLICandidate:
            event_id = row['event_id']
            compatibility = event_signatures.get((row['source_id'], event_id), row['counters_json'])
            unusable = invalid_for(raw, measure)
            if (row['evidence_state'] != 'usable' and not raw.get('invalid') and raw.get('scope') != 'metadata') or event_id in cross_session \
                    or (row['session_id'], event_id) in conflicting_parent \
                    or (row['session_id'], event_id) in conflicting_event:
                unusable = True
            reason = ('copilot_cli_cross_session_event' if event_id in cross_session else
                      'copilot_cli_parent_conflict' if (row['session_id'], event_id) in conflicting_parent else
                      'copilot_cli_event_conflict' if (row['session_id'], event_id) in conflicting_event else
                      'invalid_cli_counter' if invalid_for(raw, measure) else row['evidence_reason'])
            return CopilotCLICandidate(
                row['id'], row['session_id'], event_id, row['source_line'],
                ('model', row['model']), value, not unusable, reason,
                row['source_id'], row['parent_event_id'], row['counter_epoch'], compatibility,
                raw.get('scope') == 'metadata' and row['session_id'] not in model_sessions)

        for measure in MEASURES:
            candidates: list[CopilotCLICandidate] = []
            for row, raw in model_rows:
                if measure == 'recorded_usd':
                    candidate_value: int | Decimal | None = None
                elif measure == 'total':
                    parts = [raw.get(name) for name in ('input', 'output', 'cache_read', 'cache_write')]
                    candidate_value = sum(cast(list[int], parts)) \
                        if all(type(part) is int for part in parts) else None
                else:
                    candidate_value = cast(int | Decimal | None, raw.get(measure))
                candidates.append(candidate(row, raw, measure, candidate_value))
            for row, raw in metadata_rows:
                if row['session_id'] not in model_sessions:
                    candidates.append(candidate(row, raw, measure, None))
            owners = {item.observation_id: item.session_id for item in candidates}
            db.executemany('INSERT INTO decision VALUES(?,?,?,?,?,?,?)',
                ((decision.observation_id, measure, decision.state,
                  owners[decision.observation_id] if decision.state == 'selected' else None,
                  decision.canonical, decision.reason, 'cli-1')
                 for decision in reconcile_copilot_cli(tuple(candidates))))

        quantity_rows = tuple(db.execute(
            "SELECT q.*,o.session_id,o.model,e.source_id,e.event_id,e.parent_event_id,e.counter_epoch,"
            "e.line AS source_line,e.state AS evidence_state,"
            "e.reason AS evidence_reason,e.counters_json FROM quantity_value q "
            "JOIN observation o ON o.id=q.observation_id "
            "JOIN copilot_cli_evidence e ON e.observation_id=o.id "
            "WHERE o.session_id IN (SELECT id FROM reconcile_sessions)"))
        known_by_measure: dict[str, list[CopilotCLICandidate]] = {}
        metadata_by_measure: dict[str, list[CopilotCLICandidate]] = {}
        for row in quantity_rows:
            if row['source_id'] not in active_sources:
                continue
            raw = json.loads(row['counters_json']) if row['counters_json'] else {}
            scope = raw.get('scope') if isinstance(raw, dict) else None
            if scope == 'metadata' and row['event_id'] is None and row['session_id'] in started_sessions:
                continue
            group = ('model', row['model']) if row['measure'] == 'request_count' else ('session', None)
            candidate_item = CopilotCLICandidate(
                row['observation_id'], row['session_id'], row['event_id'], row['source_line'], group,
                Decimal(row['amount_decimal']) if row['state'] == 'known' else None,
                ((row['evidence_state'] == 'usable' or scope == 'metadata' or bool(raw.get('invalid')))
                 and not invalid_for(raw, row['measure']) and row['event_id'] not in cross_session
                 and (row['session_id'], row['event_id']) not in conflicting_parent
                 and (row['session_id'], row['event_id']) not in conflicting_event),
                ('copilot_cli_cross_session_event' if row['event_id'] in cross_session else
                 'copilot_cli_parent_conflict' if (row['session_id'], row['event_id']) in conflicting_parent
                 else 'copilot_cli_event_conflict' if (row['session_id'], row['event_id']) in conflicting_event
                 else row['evidence_reason']), row['source_id'], row['parent_event_id'],
                row['counter_epoch'], event_signatures.get((row['source_id'], row['event_id']), row['counters_json']),
                scope == 'metadata' and row['session_id'] not in control_sessions | model_sessions)
            target = metadata_by_measure if scope == 'metadata' else known_by_measure
            target.setdefault(row['measure'], []).append(candidate_item)
        for measure in ('nano_aiu', 'premium_requests', 'request_count'):
            candidates = known_by_measure.get(measure, [])
            known_sessions = {item.session_id for item in candidates}
            candidates += [item for item in metadata_by_measure.get(measure, []) if item.session_id not in known_sessions]
            owners = {item.observation_id: item.session_id for item in candidates}
            db.executemany('INSERT INTO quantity_decision VALUES(?,?,?,?,?,?,?)',
                ((decision.observation_id, measure, decision.state,
                  owners[decision.observation_id] if decision.state == 'selected' else None,
                  decision.canonical, decision.reason, 'cli-1')
                 for decision in reconcile_copilot_cli(tuple(candidates))))

        # A later valid cumulative shutdown is also a presence snapshot.  If a
        # model disappeared, old model observations cannot remain selected.
        presence_candidates: list[CopilotCLICandidate] = []
        models_by_observation: dict[str, set[str]] = {}
        owners_by_observation: dict[str, str] = {}
        for row, raw in session_rows:
            if row['source_kind'] != 'shutdown' or row['event_id'] is None:
                continue
            models = raw.get('models')
            if not isinstance(models, list) or not all(isinstance(name, str) for name in models):
                continue
            presence_candidates.append(candidate(row, raw, 'presence', 0))
            models_by_observation[row['id']] = set(models)
            owners_by_observation[row['id']] = row['session_id']
        presence_decisions = reconcile_copilot_cli(tuple(presence_candidates))
        latest_models = {owners_by_observation[decision.observation_id]: models_by_observation[decision.observation_id]
                         for decision in presence_decisions if decision.state == 'selected'}
        ambiguous_presence = {owners_by_observation[decision.observation_id]
                              for decision in presence_decisions if decision.state == 'unresolved'}
        for session_id in ambiguous_presence:
            for table in ('decision', 'quantity_decision'):
                db.execute(f"UPDATE {table} SET state='unresolved',owner_session=NULL,canonical=NULL,"
                           "reason='copilot_cli_incomparable_history' WHERE observation_id IN "
                           "(SELECT id FROM observation WHERE session_id=?)", (session_id,))
        for session_id, present in latest_models.items():
            removed = model_names - present
            if not removed:
                continue
            db.execute(
                "UPDATE decision SET state='unresolved',owner_session=NULL,canonical=NULL,"
                "reason='copilot_cli_model_removed' WHERE observation_id IN "
                "(SELECT id FROM observation WHERE session_id=? AND model IN (SELECT UNNEST(?)))",
                (session_id, list(removed)))
            db.execute(
                "UPDATE quantity_decision SET state='unresolved',owner_session=NULL,canonical=NULL,"
                "reason='copilot_cli_model_removed' WHERE observation_id IN "
                "(SELECT id FROM observation WHERE session_id=? AND model IN (SELECT UNNEST(?)))",
                (session_id, list(removed)))

        invalid_sessions = {
            row['session_id'] for row, raw in session_rows
            if isinstance(raw.get('invalid'), list)
            and any(value in ('epoch', 'shutdown') for value in cast(list[object], raw['invalid']))
        }
        for session_id in invalid_sessions:
            db.execute(
                "UPDATE decision SET state='unresolved',owner_session=NULL,canonical=NULL,"
                "reason='copilot_cli_invalid_control' WHERE observation_id IN "
                "(SELECT id FROM observation WHERE session_id=?)", (session_id,))
            db.execute(
                "UPDATE quantity_decision SET state='unresolved',owner_session=NULL,canonical=NULL,"
                "reason='copilot_cli_invalid_control' WHERE observation_id IN "
                "(SELECT id FROM observation WHERE session_id=?)", (session_id,))

    def _reconcile_codex(self, db: Connection) -> None:
        rows = tuple(db.execute("SELECT o.*,e.source_id,e.line AS source_line,e.source_kind,e.response_id,e.thread_id,e.turn_id,e.mirror_response_id,e.cumulative_json,e.last_json,e.state AS evidence_state,e.reason AS evidence_reason FROM observation o JOIN codex_evidence e ON e.observation_id=o.id WHERE o.session_id IN (SELECT id FROM reconcile_sessions) ORDER BY o.id,e.state DESC,e.ordinal"))
        if not rows:
            return
        metadata: dict[str, Row] = {}
        for row in rows:
            previous = metadata.get(row['id'])
            if previous is None or (bool(row['mirror_response_id']), row['evidence_state'] == 'usable') > (bool(previous['mirror_response_id']), previous['evidence_state'] == 'usable'):
                metadata[row['id']] = row
        # Model disagreement affects pricing attribution, not agreeing token counts.
        # Keep original observation metadata and derive the projection warning.
        response_models: dict[str, set[tuple[str | None, str | None]]] = {}
        for row in rows:
            if row['response_id'] is not None:
                response_models.setdefault(row['response_id'], set()).add((row['provider'], row['model']))
        model_conflicts = {rid for rid, models in response_models.items() if len(models) > 1}
        db.execute("DELETE FROM diagnostic WHERE code='codex_model_conflict' AND observation_id IN (SELECT id FROM observation WHERE session_id IN (SELECT id FROM reconcile_sessions))")
        db.executemany('INSERT INTO diagnostic(source_id,line,observation_id,code) VALUES(?,?,?,?)',
            ((row['source_id'], row['source_line'], row['id'], 'codex_model_conflict') for row in rows if row['response_id'] in model_conflicts))
        session_ids = {row[0] for row in db.execute("SELECT id FROM session_view WHERE harness='codex'")}
        lineage: dict[str, set[str]] = {}
        for row in db.execute('SELECT g.session_id,c.parent_thread_id,c.forked_from_id FROM codex_source c JOIN source_generation g ON g.id=c.source_id'):
            parent = row['forked_from_id'] or row['parent_thread_id']
            if parent:
                lineage.setdefault(row['session_id'], set()).add('codex:' + parent)
        conflicts = {sid for sid, parents in lineage.items() if len(parents) != 1}
        conflicts.update(row[0] for row in db.execute("SELECT DISTINCT g.session_id FROM diagnostic d JOIN source_generation g ON g.id=d.source_id WHERE g.session_id LIKE 'codex:%' AND d.code IN ('header_conflict','conflicting_header')"))
        parents = {sid: next(iter(ancestors)) for sid, ancestors in lineage.items() if len(ancestors) == 1}
        missing = {sid for sid, parent in parents.items() if parent not in session_ids}
        values: dict[str, dict[str, TokenValue]] = {}
        for row in db.execute("SELECT v.* FROM token_value v JOIN observation o ON o.id=v.observation_id WHERE o.session_id LIKE 'codex:%' AND o.session_id IN (SELECT id FROM reconcile_sessions)"):
            values.setdefault(row['observation_id'], {})[row['measure']] = Known(row['amount']) if row['state'] == 'known' else Unknown(row['reason']) if row['state'] == 'unknown' else NotApplicable(row['reason'])
        totals = {oid: TokenEvidence(TokenBreakdown(*(measures[name] for name in MEASURES[:4])), measures['reported_total'], measures['reasoning'], measures['cache_write_1h']).total for oid, measures in values.items()}
        db.execute("DELETE FROM decision WHERE observation_id IN (SELECT id FROM observation WHERE session_id LIKE 'codex:%' AND session_id IN (SELECT id FROM reconcile_sessions))")
        owners = {oid: row['session_id'] for oid, row in metadata.items()}
        appearances = tuple(CodexAppearance(row['source_id'], row['session_id'], row['id'], row['source_line'], row['cumulative_json'] if row['source_kind'] == 'legacy' else None, row['last_json'] if row['source_kind'] == 'legacy' else None) for row in rows)
        for measure in MEASURES:
            candidates = tuple(CodexCandidate(oid, row['session_id'], row['source_kind'], row['response_id'], row['cumulative_json'], row['last_json'],
                json.dumps(('missing', 'not_recorded') if measure == 'recorded_usd' else token_data(totals[oid] if measure == 'total' else values[oid][measure])), row['at_us'], row['evidence_state'] == 'usable', row['evidence_reason'], row['mirror_response_id']) for oid, row in metadata.items())
            db.executemany('INSERT INTO decision VALUES(?,?,?,?,?,?,?)',
                ((decision.observation_id, measure, decision.state, owners[decision.observation_id] if decision.state == 'selected' else None, decision.canonical, decision.reason, 'codex-1')
                 for decision in reconcile_codex(candidates, parents, missing, conflicts, appearances)))

    def _reconcile_claude(self, db: Connection) -> None:
        from .claude_reader import PROFILE as CLAUDE_PROFILE
        rows = tuple(db.execute(
            "SELECT o.*,e.source_id,e.line AS source_line,e.message_id,e.request_id,e.entry_uuid,"
            "e.state AS evidence_state,e.reason AS evidence_reason,g.locator,g.profile,"
            "(g.profile<>? AND EXISTS(SELECT 1 FROM source_generation current "
            "WHERE current.locator=g.locator AND current.sha256=g.sha256 AND current.profile=? "
            "AND current.session_id IS NOT NULL)) "
            "AS superseded_projection "
            "FROM observation o JOIN claude_evidence e ON e.observation_id=o.id "
            "JOIN source_generation g ON g.id=e.source_id "
            "WHERE o.session_id IN (SELECT id FROM reconcile_sessions) ORDER BY o.id,e.ordinal",
            (CLAUDE_PROFILE, CLAUDE_PROFILE)))
        if not rows:
            return
        values: dict[str, dict[str, tuple[str, int | None, str | None]]] = {}
        for row in db.execute(
                "SELECT v.observation_id,v.measure,v.state,v.amount,v.reason FROM token_value v "
                "JOIN observation o ON o.id=v.observation_id WHERE o.session_id IN (SELECT id FROM reconcile_sessions)"):
            values.setdefault(row['observation_id'], {})[row['measure']] = (row['state'], row['amount'], row['reason'])
        candidates: list[ClaudeCandidate] = []
        for row in rows:
            token = values[row['id']]
            compatible = digest(json.dumps((row['provider'], row['model'], token['input'], token['cache_read'], token['cache_write'])))
            safe = json.loads(row['safe_facts_json'])
            output = safe.get('usage.output')
            finality = safe.get('usage.outputFinality', 'unqualified')
            writer_token = safe.get('usage.writerVersionToken')
            missing_time = int(row['time_kind'] != 'point')
            candidates.append(ClaudeCandidate(
                row['id'], row['session_id'], row['message_id'], row['request_id'], row['entry_uuid'], compatible,
                (missing_time, row['at_us'] or 0, row['locator'], row['source_line'], row['entry_uuid']),
                row['evidence_state'], row['evidence_reason'],
                ('present', output) if 'usage.output' in safe else ('missing', None),
                finality if isinstance(finality, str) else 'unqualified',
                row['profile'] == CLAUDE_PROFILE,
                writer_token if isinstance(writer_token, str) and len(writer_token) == 64
                and all(char in '0123456789abcdef' for char in writer_token) else None,
                bool(row['superseded_projection'])))
        db.execute("DELETE FROM diagnostic WHERE code='claude_output_variation' AND observation_id IN "
                   "(SELECT id FROM observation WHERE session_id IN (SELECT id FROM reconcile_sessions))")
        varying = claude_output_variations(tuple(candidates))
        db.executemany('INSERT INTO diagnostic(source_id,line,observation_id,code) VALUES(?,?,?,?)',
                       ((row['source_id'], row['source_line'], row['id'], 'claude_output_variation') for row in rows
                        if row['id'] in varying))
        db.execute("DELETE FROM decision WHERE observation_id IN "
                   "(SELECT id FROM observation WHERE session_id IN (SELECT id FROM reconcile_sessions) "
                   "AND session_id IN (SELECT id FROM session_view WHERE harness='claude'))")
        owners = {row['id']: row['session_id'] for row in rows}
        decisions = reconcile_claude(tuple(candidates))
        for measure in MEASURES:
            db.executemany('INSERT INTO decision VALUES(?,?,?,?,?,?,?)',
                           ((decision.observation_id, measure, decision.state,
                             owners[decision.observation_id] if decision.state == 'selected' else None,
                             decision.canonical, decision.reason, 'claude-1') for decision in decisions))

    def _reconcile_copilot_vscode(self, db: Connection) -> None:
        rows = tuple(db.execute(
            "SELECT o.id,o.session_id,o.native_entry_id,o.fingerprint,o.model,o.safe_facts_json,"
            "e.*,g.locator,g.generation,g.availability,"
            "q.state AS quantity_state,q.amount_decimal "
            "FROM observation o JOIN copilot_vscode_evidence e ON e.observation_id=o.id "
            "JOIN source_generation g ON g.id=e.source_id "
            "LEFT JOIN quantity_value q ON q.observation_id=o.id AND q.measure='ai_credits' "
            "WHERE o.session_id IN (SELECT id FROM reconcile_sessions) ORDER BY o.id,e.ordinal"))
        if not rows:
            return
        latest = {row['locator']: row['generation'] for row in db.execute(
            'SELECT locator,max(generation) AS generation FROM source_generation GROUP BY locator')}
        candidates = tuple(CopilotVscodeCandidate(
            row['id'], row['session_id'], row['source_id'], row['locator'],
            row['generation'] == latest[row['locator']], row['availability'] == 'available',
            row['representation'], row['request_id'], row['response_id'], row['model'],
            row['evidence_kind'], row['request_ordinal'], row['line'], row['fingerprint'],
            vscode_compatibility(
                json.loads(row['safe_facts_json']).get(VSCODE_PRESENCE, 'legacy'),
                row['raw_input'], row['raw_output'], row['raw_cache_read'], row['raw_cache_write'],
                row['state'], row['reason'], row['quantity_state'],
                Decimal(row['amount_decimal']) if row['amount_decimal'] is not None else None,
                json.loads(row['safe_facts_json']).get('usage.outputFinality') == VSCODE_OUTPUT_LOWER_BOUND,
                json.loads(row['safe_facts_json']).get(VSCODE_RETAINED_CONFLICT)),
            row['state'], row['reason'], row['quantity_state'],
            Decimal(row['amount_decimal']) if row['amount_decimal'] is not None else None)
            for row in rows)
        from .copilot_vscode_reader import SUPPORTED_PROFILES as VSCODE_PROFILES
        current_sources = tuple(CopilotVscodeSource(
            row['id'], row['session_id'], row['locator'], row['availability'] == 'available',
            'operation_log' if Path(row['locator']).suffix == '.jsonl' else 'flat')
            for row in db.execute(
                "SELECT id,session_id,locator,generation,availability FROM source_generation "
                "WHERE profile IN (?,?) AND session_id IN "
                "(SELECT id FROM reconcile_sessions)", VSCODE_PROFILES)
            if row['generation'] == latest[row['locator']])
        token_decisions, quantity_decisions = reconcile_copilot_vscode(candidates, current_sources)
        scope = "SELECT id FROM observation WHERE session_id IN (SELECT id FROM reconcile_sessions) AND session_id IN (SELECT id FROM session_view WHERE harness='copilot-vscode')"
        db.execute('DELETE FROM decision WHERE observation_id IN (' + scope + ')')
        db.execute('DELETE FROM quantity_decision WHERE observation_id IN (' + scope + ')')
        owners = {row['id']: row['session_id'] for row in rows}
        for measure in MEASURES:
            db.executemany('INSERT INTO decision VALUES(?,?,?,?,?,?,?)',
                ((decision.observation_id, measure, decision.state,
                  owners[decision.observation_id] if decision.state == 'selected' else None,
                  decision.canonical, decision.reason, 'copilot-vscode-1') for decision in token_decisions))
        db.executemany('INSERT INTO quantity_decision VALUES(?,?,?,?,?,?,?)',
            ((decision.observation_id, 'ai_credits', decision.state,
              owners[decision.observation_id] if decision.state == 'selected' else None,
              decision.canonical, decision.reason, 'copilot-vscode-1') for decision in quantity_decisions))

    def save_codex_titles(self, mapping: Mapping[str, str]) -> None:
        with self.connect(write=True) as db:
            before = db.total_changes
            for sid, title in mapping.items():
                if not sid.startswith('codex:') or not db.execute("SELECT 1 FROM session_view WHERE id=? AND harness='codex'", (sid,)).fetchone():
                    continue
                if not isinstance(title, str) or not 1 <= len(title) <= 160:
                    raise ContractViolation('invalid_codex_title')
                db.execute('INSERT INTO codex_title VALUES(?,?) ON CONFLICT(session_id) DO UPDATE SET title=excluded.title WHERE title<>excluded.title', (sid, title))
            if db.total_changes != before:
                db.execute('UPDATE ledger_meta SET revision=revision+1')

    def mark_missing(self, locators: tuple[str, ...]) -> None:
        with self.connect(write=True) as db:
            before = db.total_changes
            for locator in locators:
                db.execute("UPDATE source_generation SET availability='missing' WHERE locator=? AND availability<>'missing'", (locator,))
            if db.total_changes != before:
                self._reconcile(db, changed_locators=set(locators))
                db.execute('UPDATE ledger_meta SET revision=revision+1')

    def locators(self) -> tuple[str, ...]:
        with self.connect() as db:
            return tuple(row[0] for row in db.execute('SELECT DISTINCT locator FROM source_generation ORDER BY locator'))

    @staticmethod
    def _sessions(db: Connection) -> tuple[SessionMetadata, ...]:
        return tuple(SessionMetadata(SessionId(r['id']), r['native_id'], from_micros(r['started_us']), r['cwd'], r['parent_locator'], r['native_title'], last_observed=from_micros(r['last_seen_us']), title_excerpt=r['title_excerpt']) for r in db.execute('SELECT s.*,COALESCE(t.title,s.display_name) AS native_title FROM session_view s LEFT JOIN codex_title t ON t.session_id=s.id'))

    def attribution_input(self) -> tuple[tuple[SessionMetadata, ...], Mapping[str, Attribution]]:
        """Read only discovery metadata, from one consistent ledger snapshot."""
        with self.connect() as db:
            db.execute('BEGIN')
            return self._sessions(db), MappingProxyType(self._attributions(db))

    def attributions(self) -> dict[str, Attribution]:
        with self.connect() as db:
            return self._attributions(db)

    @staticmethod
    def _attributions(db: Connection) -> dict[str, Attribution]:
        return {row['id']: Assigned(ProjectId(row['project_id']), row['worktree'], row['attribution_reason']) if row['project_id'] is not None else Unassigned(row['attribution_reason']) for row in db.execute('SELECT id,project_id,worktree,attribution_reason FROM session_view')}

    def save_attributions(self, mapping: Mapping[str, Attribution]) -> None:
        with self.connect(write=True) as db:
            db.execute('BEGIN IMMEDIATE')
            changed = False
            for session_id, attribution in mapping.items():
                if isinstance(attribution, Assigned):
                    project_id, worktree, reason = str(attribution.project_id), attribution.worktree, attribution.basis
                    kind = 'repository' if project_id.startswith('git:') else 'directory'
                    identity = project_id.split(':', 1)[-1]
                    label_path = Path(identity).parent if Path(identity).name == '.git' else Path(identity)
                    db.execute('INSERT OR IGNORE INTO project VALUES(?,?,?,?)', (project_id, project_id, kind, label_path.name or identity))
                else:
                    project_id, worktree, reason = None, None, attribution.reason
                row = db.execute('SELECT project_id,worktree,attribution_reason FROM session_view WHERE id=?', (session_id,)).fetchone()
                if row is None:
                    raise ValueError('unknown_session')
                if tuple(row) != (project_id, worktree, reason):
                    db.execute('UPDATE session_attribution SET project_id=?,worktree=?,attribution_reason=? WHERE session_id=?', (project_id, worktree, reason, session_id))
                    changed = True
            if changed:
                db.execute('UPDATE ledger_meta SET revision=revision+1')

    @staticmethod
    def _families(db: Connection, sessions: Mapping[str, Row],
                  attributions: Mapping[str, Attribution]) -> tuple[dict[str, SessionFamily], set[str]]:
        """Resolve explicit ownership; parentSession alone describes only lineage."""
        locator_ids: dict[str, set[str]] = {}
        for row in db.execute('SELECT locator,session_id FROM source_generation WHERE session_id IS NOT NULL'):
            locator_ids.setdefault(os.path.normpath(row['locator']), set()).add(row['session_id'])
        locators = {path: next(iter(ids)) for path, ids in locator_ids.items() if len(ids) == 1}
        native_ids = {(sid.split(':', 1)[0], row['native_id']): sid for sid, row in sessions.items()}
        hashed_ids: dict[str, set[str]] = {}
        for row in db.execute('SELECT d.path_hash,s.session_id FROM delegation_scan d JOIN source_generation s ON s.id=d.source_id WHERE s.session_id IS NOT NULL'):
            hashed_ids.setdefault(row['path_hash'], set()).add(row['session_id'])
        path_hashes = {key: next(iter(ids)) for key, ids in hashed_ids.items() if len(ids) == 1}
        parents = {sid: locators.get(os.path.normpath(row['parent_locator'])) if row['parent_locator'] else None for sid, row in sessions.items()}
        suffixes: dict[str, list[str]] = {}
        named_ids: dict[str, list[str]] = {}
        for sid, row in sessions.items():
            if row['display_name']:
                named_ids.setdefault(sid.split(':', 1)[0] + ':' + row['display_name'], []).append(sid)
            if row['display_name'] and '#' in row['display_name']:
                suffixes.setdefault(sid.split(':', 1)[0] + ':' + row['display_name'].rsplit('#', 1)[1], []).append(sid)
        # Group repeated records before resolving owners. A copied fork keeps the
        # same native entry identity; unrelated references must not be collapsed.
        evidence: dict[tuple[str, str, str, str], set[str]] = {}
        diagnostics: set[str] = set()
        for ref in db.execute('SELECT r.*,s.session_id FROM delegation_ref r JOIN source_generation s ON s.id=r.source_id WHERE s.session_id IS NOT NULL'):
            owner = locators.get(os.path.normpath(ref['owner_path'])) if ref['owner_path'] else ref['session_id']
            if owner is None or owner not in sessions:
                diagnostics.add('missing_subagent_owner')
                continue
            if ref['kind'] == 'child_path':
                child = locators.get(os.path.normpath(ref['value']))
            elif ref['kind'] == 'child_id':
                child = native_ids.get((owner.split(':', 1)[0], ref['value']))
            elif ref['kind'] == 'child_path_hash':
                child = path_hashes.get(ref['value'])
            elif ref['kind'] == 'child_name':
                named = named_ids.get(owner.split(':', 1)[0] + ':' + ref['value'], ())
                if len(named) > 1:
                    diagnostics.add('ambiguous_subagent_owner')
                    continue
                child = named[0] if named else None
            else:
                # The installed writer sets <type>#<agent-id first 8>. Require
                # the independent header parent link as corroboration.
                matches = [sid for sid in suffixes.get(owner.split(':', 1)[0] + ':' + ref['value'][:8], ()) if parents[sid] == owner]
                child = matches[0] if len(matches) == 1 else None
                if len(matches) > 1:
                    diagnostics.add('ambiguous_subagent_owner')
                if not matches:
                    continue  # In-memory agents legitimately have no history file.
            if child is None:
                diagnostics.add('missing_subagent_history')
            elif child.split(':', 1)[0] != owner.split(':', 1)[0]:
                diagnostics.add('unsupported_cross_harness_delegation')
            elif child == owner:
                diagnostics.add('cyclic_subagent_owner')
            else:
                evidence.setdefault((child, ref['entry_id'], ref['kind'], ref['value']), set()).add(owner)

        def ancestor(older: str, newer: str) -> bool:
            seen: set[str] = set()
            current: str | None = newer
            while current is not None and current not in seen:
                seen.add(current)
                current = parents.get(current)
                if current == older:
                    return True
            return False

        candidates: dict[str, set[str]] = {}
        for row in db.execute("SELECT id,parent_locator FROM session_view WHERE harness='claude' AND parent_locator IS NOT NULL"):
            parent = locators.get(os.path.normpath(row['parent_locator']))
            if parent is not None:
                candidates.setdefault(row['id'], set()).add(parent)
            else:
                diagnostics.add('missing_subagent_owner')
        cli_locator_ids: dict[str, set[str]] = {}
        for row in db.execute(
                "SELECT g.locator,g.session_id FROM source_generation g JOIN session_view s "
                "ON s.id=g.session_id WHERE s.harness='copilot-cli' AND g.session_id IS NOT NULL"):
            cli_locator_ids.setdefault(os.path.normpath(row['locator']), set()).add(row['session_id'])
        for row in db.execute(
                "SELECT id,parent_locator FROM session_view WHERE harness='copilot-cli' "
                "AND parent_locator IS NOT NULL"):
            possible = cli_locator_ids.get(os.path.normpath(row['parent_locator']), set())
            if len(possible) == 1:
                candidates.setdefault(row['id'], set()).update(possible)
            elif possible:
                diagnostics.add('ambiguous_subagent_owner')
            else:
                diagnostics.add('missing_subagent_owner')
        for (child, _entry, _kind, _value), possible_owners in evidence.items():
            original = {owner for owner in possible_owners if not any(ancestor(other, owner) for other in possible_owners if other != owner)}
            candidates.setdefault(child, set()).update(original)
        for row in db.execute('SELECT g.session_id,c.parent_thread_id FROM codex_source c JOIN source_generation g ON g.id=c.source_id WHERE c.parent_thread_id IS NOT NULL'):
            parent = 'codex:' + row['parent_thread_id']
            if parent in sessions:
                candidates.setdefault(row['session_id'], set()).add(parent)
            else:
                diagnostics.add('missing_subagent_owner')
        owners: dict[str, str] = {}
        for child, possible in candidates.items():
            if len(possible) == 1:
                owners[child] = next(iter(possible))
            else:
                diagnostics.add('ambiguous_subagent_owner')
        roots: dict[str, str] = {}
        for child in owners:
            current = child
            seen: set[str] = set()
            while current in owners and current not in seen:
                seen.add(current)
                current = owners[current]
            if current in seen:
                diagnostics.add('cyclic_subagent_owner')
            else:
                roots[child] = current
        members: dict[str, list[str]] = {}
        for child, root in roots.items():
            members.setdefault(root, [root]).append(child)
        families: dict[str, SessionFamily] = {}
        for root, children in members.items():
            row = sessions[root]
            last = [sessions[sid]['last_seen_us'] for sid in children if sessions[sid]['last_seen_us'] is not None]
            family = SessionFamily(SessionId(root), row['display_name'] or row['title_excerpt'], row['cwd'], attributions[root],
                                   from_micros(row['started_us']), from_micros(max(last)) if last else None, row['harness'])
            families.update((sid, family) for sid in children)
        return families, diagnostics

    def report_input(self, query: ReportQuery) -> ReportInput:
        """Read report evidence and bounded session labels in one transaction."""
        predicates = ["(EXISTS (SELECT 1 FROM decision d WHERE d.observation_id=o.id AND d.state<>'excluded') OR EXISTS (SELECT 1 FROM quantity_decision d WHERE d.observation_id=o.id AND d.state<>'excluded'))"]
        parameters: list[str | int | None] = []
        if query.project_id == 'unassigned':
            predicates.append('s.project_id IS NULL')
        elif query.project_id is not None:
            predicates.append('s.project_id=?')
            parameters.append(str(query.project_id))
        if isinstance(query.range, DateRange):
            predicates.append("((o.time_kind='point' AND o.at_us>=? AND o.at_us<?) OR (o.time_kind='interval' AND (o.start_us IS NULL OR (o.start_us<? AND o.end_us>?))) OR o.time_kind='undated')")
            parameters.extend((micros(query.range.start), micros(query.range.end), micros(query.range.end), micros(query.range.start)))
        scope = 'SELECT o.id FROM observation o JOIN session_owner f ON f.session_id=o.session_id JOIN session_view s ON s.id=f.owner_id WHERE ' + ' AND '.join(predicates)
        with self.connect() as db:
            db.execute('BEGIN')
            revision = int(db.execute('SELECT revision FROM ledger_meta').one()[0])
            sessions = {row['id']: row for row in db.execute('SELECT s.id,s.harness,COALESCE(t.title,s.display_name) AS display_name,s.title_excerpt,s.cwd,s.started_us,s.last_seen_us,s.project_id,s.worktree,s.attribution_reason,s.native_id,s.parent_locator FROM session_view s LEFT JOIN codex_title t ON t.session_id=s.id')}
            attributions: dict[str, Attribution] = {
                sid: Assigned(ProjectId(row['project_id']), row['worktree'], row['attribution_reason']) if row['project_id'] is not None else Unassigned(row['attribution_reason'])
                for sid, row in sessions.items()
            }
            families, family_diagnostics = self._families(db, sessions, attributions)
            db.execute('CREATE TEMP TABLE session_owner(session_id TEXT PRIMARY KEY,owner_id TEXT NOT NULL)')
            db.executemany('INSERT INTO session_owner VALUES(?,?)', ((sid, families[sid].id if sid in families else sid) for sid in sessions))
            # Materialize once in the connection's temporary database; the ledger
            # remains read-only and every evidence table uses the same snapshot.
            db.execute('CREATE TEMP TABLE relevant(id TEXT PRIMARY KEY)')
            db.execute('INSERT INTO relevant ' + scope, parameters)
            scoped_diagnostics = report_diagnostics(db, 'relevant')
            # Existing family-resolution metadata has no observation attachment;
            # retain it only for the complete population, never a filtered report.
            diagnostics = tuple(sorted({code for codes in scoped_diagnostics.values() for code in codes}
                                       | (family_diagnostics if query.project_id is None and isinstance(query.range, AllTime) else set())))
            if db.execute('SELECT 1 FROM relevant LIMIT 1').fetchone() is None:
                return ReportInput(revision, (), diagnostics)
            token_values: dict[str, dict[str, TokenValue]] = {}
            for value in db.execute('SELECT v.* FROM relevant r JOIN token_value v ON v.observation_id=r.id'):
                token_values.setdefault(value['observation_id'], {})[value['measure']] = Known(value['amount']) if value['state'] == 'known' else Unknown(value['reason']) if value['state'] == 'unknown' else NotApplicable(value['reason'])
            prices: dict[str, RecordedMoney] = {}
            for price in db.execute('SELECT p.* FROM relevant r JOIN recorded_estimate p ON p.observation_id=r.id'):
                prices[price['observation_id']] = (RecordedEstimate(Decimal(price['amount_decimal']), price['currency'], tuple((k, Decimal(v)) for k, v in json.loads(price['components_json']).items()), price['source_ref'])
                                                   if price['state'] == 'known' else MissingEstimate(price['reason']))
            quantities: dict[str, list[MeasuredQuantity]] = {}
            for value in db.execute("SELECT v.*,d.reason AS decision_reason FROM relevant r JOIN quantity_value v ON v.observation_id=r.id LEFT JOIN quantity_decision d ON d.observation_id=v.observation_id AND d.measure=v.measure"):
                quantities.setdefault(value['observation_id'], []).append(MeasuredQuantity(
                    value['measure'], value['state'], Decimal(value['amount_decimal']) if value['amount_decimal'] is not None else None,
                    value['reason'], bool(value['lower_bound']) or value['state'] == 'known' and value['decision_reason'] == 'copilot_store_partial_index', value['source_ref']))
            decisions: dict[str, list[tuple[str, str]]] = {}
            reasons: dict[str, set[str]] = {}
            lower_bounds: dict[str, list[str]] = {}
            for decision in db.execute('SELECT d.observation_id,d.measure,d.state,d.reason FROM relevant r JOIN decision d ON d.observation_id=r.id'):
                decisions.setdefault(decision['observation_id'], []).append((decision['measure'], decision['state']))
                if decision['reason'] == 'copilot_store_partial_index' and decision['measure'] != 'recorded_usd':
                    lower_bounds.setdefault(decision['observation_id'], []).append(decision['measure'])
                if decision['state'] != 'selected' and decision['reason'] in DIAGNOSTIC_CODES:
                    reasons.setdefault(decision['observation_id'], set()).add(decision['reason'])
            quantity_decisions: dict[str, list[tuple[str, str]]] = {}
            for decision in db.execute('SELECT d.observation_id,d.measure,d.state FROM relevant r JOIN quantity_decision d ON d.observation_id=r.id'):
                quantity_decisions.setdefault(decision['observation_id'], []).append((decision['measure'], decision['state']))
            contributions: list[SelectedContribution] = []
            for row in db.execute('SELECT o.id,o.session_id,o.time_kind,o.at_us,o.start_us,o.end_us,o.time_reason,o.provider,o.model,o.safe_facts_json FROM relevant r JOIN observation o ON o.id=r.id'):
                values = token_values.pop(row['id'])
                tokens = TokenEvidence(TokenBreakdown(*(values[name] for name in ('input', 'output', 'cache_read', 'cache_write'))), values['reported_total'], values['reasoning'], values['cache_write_1h'])
                time: TimeEvidence
                if row['time_kind'] == 'point':
                    time = Point(cast(datetime, from_micros(row['at_us'])))
                elif row['time_kind'] == 'interval':
                    time = Interval(from_micros(row['start_us']), cast(datetime, from_micros(row['end_us'])))
                else:
                    time = Undated(row['time_reason'])
                session = sessions[row['session_id']]
                row_diagnostics = scoped_diagnostics.pop(row['id'], set())
                contributions.append(SelectedContribution(ObservationId(row['id']), SessionId(row['session_id']), session['display_name'] or session['title_excerpt'], session['cwd'],
                    ModelIdentity(None, None) if 'codex_model_conflict' in row_diagnostics else ModelIdentity(row['provider'], row['model']), time, tokens, prices.pop(row['id']), attributions[row['session_id']], tuple(decisions.pop(row['id'], ())),
                    from_micros(session['started_us']), from_micros(session['last_seen_us']), tuple(sorted(reasons.pop(row['id'], set()))) + tuple(sorted(row_diagnostics - {'saved_history'})), families.get(row['session_id']),
                    tuple(quantities.pop(row['id'], ())), tuple(quantity_decisions.pop(row['id'], ())), session['harness'], 'saved_history' in row_diagnostics,
                    json.loads(row['safe_facts_json']).get('usage.outputFinality') == VSCODE_OUTPUT_LOWER_BOUND,
                    tuple(lower_bounds.get(row['id'], ()))))
            return ReportInput(revision, tuple(contributions), diagnostics)

    def snapshot(self) -> Snapshot:
        with self.connect() as db:
            db.execute('BEGIN')
            revision = int(db.execute('SELECT revision FROM ledger_meta').one()[0])
            sessions = self._sessions(db)
            attributions = MappingProxyType(self._attributions(db))
            scoped_diagnostics: dict[str, set[str]] = {}
            for diagnostic in db.execute('SELECT a.observation_id,d.code FROM diagnostic d JOIN appearance a ON a.source_id=d.source_id AND a.line=d.line UNION SELECT a.observation_id,d.code FROM diagnostic d JOIN appearance a ON a.source_id=d.source_id WHERE d.line IS NULL'):
                scoped_diagnostics.setdefault(diagnostic[0], set()).add(diagnostic[1])
            token_values: dict[str, dict[str, TokenValue]] = {}
            for value in db.execute('SELECT * FROM token_value'):
                token_values.setdefault(value['observation_id'], {})[value['measure']] = Known(value['amount']) if value['state'] == 'known' else Unknown(value['reason']) if value['state'] == 'unknown' else NotApplicable(value['reason'])
            prices: dict[str, RecordedMoney] = {}
            for price in db.execute('SELECT * FROM recorded_estimate'):
                prices[price['observation_id']] = (RecordedEstimate(Decimal(price['amount_decimal']), price['currency'], tuple((k, Decimal(v)) for k, v in json.loads(price['components_json']).items()), price['source_ref'])
                                                   if price['state'] == 'known' else MissingEstimate(price['reason']))
            quantities: dict[str, list[MeasuredQuantity]] = {}
            for value in db.execute('SELECT * FROM quantity_value'):
                quantities.setdefault(value['observation_id'], []).append(MeasuredQuantity(
                    value['measure'], value['state'], Decimal(value['amount_decimal']) if value['amount_decimal'] is not None else None,
                    value['reason'], bool(value['lower_bound']), value['source_ref']))
            parents: dict[str, str | None] = {}
            for parent in db.execute('SELECT a.observation_id,e.parent_entry_id FROM appearance a JOIN entry_edge e ON e.source_id=a.source_id AND e.line=a.line ORDER BY a.observation_id,a.ordinal'):
                parents.setdefault(parent['observation_id'], parent['parent_entry_id'])
            decisions: dict[str, dict[str, str]] = {}
            reasons: dict[str, set[str]] = {}
            for decision in db.execute('SELECT observation_id,measure,state,reason FROM decision ORDER BY observation_id,measure'):
                decisions.setdefault(decision['observation_id'], {})[decision['measure']] = decision['state']
                if decision['state'] != 'selected':
                    reasons.setdefault(decision['observation_id'], set()).add(decision['reason'])
            observations: list[LedgerObservation] = []
            for row in db.execute('SELECT * FROM observation ORDER BY id'):
                values = token_values.pop(row['id'])
                tokens = TokenEvidence(TokenBreakdown(*(values[n] for n in ('input','output','cache_read','cache_write'))), values['reported_total'], values['reasoning'], values['cache_write_1h'])
                time: TimeEvidence
                if row['time_kind'] == 'point':
                    time = Point(cast(datetime, from_micros(row['at_us'])))
                elif row['time_kind'] == 'interval':
                    time = Interval(from_micros(row['start_us']), cast(datetime, from_micros(row['end_us'])))
                else:
                    time = Undated(row['time_reason'])
                money = prices.pop(row['id'])
                parent_id = parents.pop(row['id'], None)
                raw: dict[str, Any] = json.loads(row['safe_facts_json'])
                raw.pop(VSCODE_PRESENCE, None)
                raw.pop(VSCODE_RETAINED_CONFLICT, None)
                raw.pop('usage.writerVersionToken', None)
                record = UsageRecord(EntryIdentity(row['native_entry_id'], parent_id), row['kind'], time, ModelIdentity(None, None) if 'codex_model_conflict' in scoped_diagnostics.get(row['id'], ()) else ModelIdentity(row['provider'], row['model']), tokens, money, row['stop_reason'], row['tool_call_id'], tuple(raw.items()), tuple(quantities.pop(row['id'], ())))
                observations.append(LedgerObservation(row['id'], row['session_id'], record, MappingProxyType(decisions.pop(row['id'], {})), tuple(sorted(reasons.pop(row['id'], set()))), tuple(sorted(scoped_diagnostics.pop(row['id'], ())))))
            diagnostics = tuple(sorted({r[0] for r in db.execute('SELECT code FROM diagnostic')} | { 'saved_history' for _ in db.execute("SELECT 1 FROM source_generation WHERE availability<>'available' LIMIT 1")}))
            return Snapshot(revision, sessions, tuple(observations), diagnostics, attributions)
