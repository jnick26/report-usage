# Copilot session-store Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Import retained Copilot CLI database usage without double counting JSONL.

**Architecture:** A read-only SQLite snapshot feeds one atomic DuckDB generation.
Common observations serve reports; separate provenance supports a DB/JSON
reconciliation step after native CLI accounting. On-demand turns are lossy views.

**Tech Stack:** Python 3.13, stdlib sqlite3, existing DuckDB and pytest.

**Spec:** `docs/superpowers/specs/2026-09-17-copilot-session-store-design.md`

## Global Constraints

- DuckDB is the sole authoritative ledger; SQLite is read-only source input.
- Preserve existing five harness identities and Pi/Codex behavior.
- No double counting, no invented fresh/cache split, cost or credits.
- Unknown, unresolved, lower-bound and measured zero remain distinguishable.
- Preserve source files, existing fixes and retained provenance.
- No new dependencies; no push or deployment.
- Use synthetic public fixtures only; native validation stays in temporary files.

## Reader contract shared by Tasks 1 and 2

New module `src/harness_usage/copilot_store_reader.py` exports frozen dataclasses:

```python
@dataclass(frozen=True, slots=True)
class CopilotStoreSession:
    session_id: str
    cwd: str | None
    created_at: datetime | None
    updated_at: datetime | None
    host_type: str | None

@dataclass(frozen=True, slots=True)
class CopilotStoreCall:
    row_id: int
    session_id: str
    turn_index: int | None
    created_at: datetime | None
    model: str | None
    agent_id: str | None
    parent_tool_call_id: str | None
    input_tokens: int | None
    output_tokens: int | None
    cache_read_tokens: int | None
    cache_write_tokens: int | None
    reasoning_tokens: int | None
    total_nano_aiu: Decimal | None
    request_multiplier: Decimal | None
    token_details_json: str | None
    invalid_fields: tuple[str, ...] = ()

@dataclass(frozen=True, slots=True)
class CopilotStoreSnapshot:
    locator: str
    fingerprint: str
    sessions: tuple[CopilotStoreSession, ...]
    calls: tuple[CopilotStoreCall, ...]
    profile: str = 'copilot-cli-session-store/schema-8-shape-1'
    schema_version: int = 8

def read_session_store(path: Path) -> CopilotStoreSnapshot: ...
```

The callable raises a content-free `ValueError` or OSError on unqualified input.
Snapshot contains accounting fields only, not session summaries or turns.
Canonical fingerprint includes approved logical fields and local IDs, not mtime.
Invalid native counters use None plus their name in invalid_fields; absent optional
values use None alone. Use integers <2**63 and existing bounded Decimal semantics.

### Task 1: Qualified reader

**Files:** create `src/harness_usage/copilot_store_reader.py`,
`tests/test_copilot_store_reader.py`.

**Interfaces:** produce the exact dataclasses and callable above; expose the
read-only snapshot context helper for the later on-demand transcript reader.

- [ ] Add synthetic SQLite fixtures with version table, sessions and usage tables.
- [ ] Demonstrate RED before implementation: a committed row present only in WAL
  must appear in `read_session_store(path).calls`; source main/WAL hashes unchanged.
- [ ] Implement sqlite3 backup, feature checks and bounded native field decoding.
- [ ] Assert known/zero/invalid/null counters, missing turn joins, schema rejection,
  malformed identities, repeated fingerprint equality, changed/deleted call handling.
- [ ] Run focused pytest and mypy. Report RED/GREEN evidence for review.

### Task 2: Ledger generations, reconciliation and report bounds

**Files:** create `src/harness_usage/copilot_store_accounting.py` and focused
`tests/test_copilot_store_storage.py`; modify `storage.py`, `schema.sql`,
`migrate_duckdb.py`, `migrate_sqlite.py`, `reporting.py`, `aggregate_reporting.py`.

**Interfaces:** consume the exact snapshot contract; `Storage.import_sources`
accepts `SourcePayload | CopilotStoreSnapshot` and existing tuple inputs. Expose
`Storage.import_session_store(snapshot) -> int` as a one-snapshot wrapper.

- [ ] Test schema-6 populated ledger upgrade and restart before changing storage.
- [ ] Add schema-7 provenance tables and preserve SQLite/v5 conversion paths.
- [ ] Add atomic snapshot import using existing `_insert_observation`, generation
  hashing, affected-session closure and batched writes. Register empty sessions
  with unavailable evidence rather than zero. Namespace local call IDs by generation.
- [ ] Add RED/GREEN overlap cases with independent literal totals:
  JSON output10 + equal DB calls4/6 =>10; JSON10 + partial DB4 =>10;
  conflicting DB12 => unresolved, not22; DB-only4 =>lower-bound4.
- [ ] Cover copied DBs, multiplicity, rebuilt row IDs, removed rows/sessions,
  import-order independence, missing/reappearing source and model-presence controls.
- [ ] Keep raw/itemized accounting distinct; no sum of subagent and top-level totals.
- [ ] Plumb per-measure lower bounds through ordinary and SQL reports; assert parity
  for all-time, date filters, sessions/models and zero/unknown values.
- [ ] Run focused pytest and mypy; provide migration and reconciliation evidence.

### Task 3: Discovery, progress and applicable transcript views

**Files:** modify `application.py`, `__main__.py`, `transcript_access.py`,
`README.md`; create `copilot_store_transcript.py` and focused application/transcript
tests. Task 1 implementer owns this after its reader contract is stable.

**Interfaces:** `_scan` yields the typed snapshot for real session-store.db paths;
existing import_sources dispatch consumes it. Use Task 2 store_session membership
to find DB-only sessions. Accounting never reads turns.

- [ ] Test authorized Copilot-home discovery plus missing optional database.
- [ ] Include the DB in existing import progress/error accounting; preserve explicit
  saved roots and document the parent-root requirement for legacy configurations.
- [ ] Prefer native JSONL for transcripts. Test DB-only on-demand turn strings with
  a lossy-summary notice, HTML escaping, branch rejection, absent turns and bounds.
- [ ] Test symlink/out-of-root denial without reading adjacent unauthorized sources.
- [ ] Update capability documentation and run focused source/web tests.

### Task 4: Integration verification and review

**Files:** synthetic regression tests as needed, existing docs above.

- [ ] Run full tests and strict mypy from an isolated exported tree.
- [ ] Run the supplied WAL-aware DB through reader/import/reimport/report in a
  temporary DuckDB; report safe aggregate counts only.
- [ ] Verify representative overlap using source JSONL; no arithmetic sum fallback.
- [ ] Review the final diff, address concrete findings, scan for secrets.
- [ ] Leave local changes ready for publication; do not push.
