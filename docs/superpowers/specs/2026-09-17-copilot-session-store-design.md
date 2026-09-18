# Copilot CLI session-store input

## Approved intent

Continue the retained-history repairs by importing the optional Copilot CLI
`session-store.db`, without double counting its overlapping JSONL history.
This remains Copilot CLI, not a sixth harness or a second reporting database.

## Input and storage

Use Python's sqlite3 read-only URI connection and a WAL-aware backup to a
temporary snapshot. Feature-qualify native `schema_version.version = 8` and
the required columns. Never run producer code or modify the source database.
Return immutable, content-free session/call records; do not manufacture native
JSONL events. Preserve flat counters separately from itemized nano-AIU details.
Invalid counters remain unknown; invalid ownership identities reject the snapshot.

One physical source_generation represents an atomic whole-database snapshot.
DuckDB schema 7 adds session membership and call provenance tables, reusing
common observations, appearances, quantities, and decisions. Rebuild-local row
IDs are namespaced by generation. Replacement snapshots retire all preceding
rows, including deleted sessions. Missing files retain labeled saved history.
Logical fingerprints make unchanged imports no-ops. Copied index snapshots are
compared as multisets excluding local row IDs, preserving call multiplicity.

## Accounting

Reconcile native CLI events first, then apply a separate database reconciliation
step. Never pass per-call database rows through cumulative event reconciliation.
DB sessions.created_at is display metadata, not an epoch marker. Overlap
qualification requires the same session UUID, one unambiguous retained native
JSON counter epoch, all relevant calls within its start-to-selected-shutdown
interval, and comparable full raw usage/request-count vectors. Time alone is
not a proof of completeness; equality of the vectors permits detail substitution.

- Qualified, equal DB and JSON usage vectors: choose one representation, using
  database detail where the complete compatible scope is proven.
- A qualified JSON total covers a smaller database vector: keep the JSON total,
  retaining database detail non-additively. Do not fabricate residual calls.
- Conflicting vectors, differing database copies, or unproven overlapping epochs:
  mark the affected measure/scope unresolved, never add both or pick by mtime.
- DB-only evidence: count retained calls once as a lower bound. No row means
  unavailable historical usage, not measured zero.
- Respect later JSON model-presence controls; absent models cannot automatically
  resurrect through stale database rows. Agent fields are attribution, not extra
  usage on top of all-call totals.

Flat input is cache-inclusive but its relationship with the flat cache-write
counter is not sufficiently qualified for subtraction: fresh input stays unknown
when cache use is present. Keep reasoning supplemental. Nano-AIU, request count,
premium requests and estimated API USD retain distinct semantics. Retain the
native multiplier/itemized details as provenance, not invented USD or credits.

## Discovery, reports, transcripts

Discover session-store.db only inside authorized roots. New default Copilot
configuration uses its home directory, covering session-state and the sibling
database. Preserve explicitly saved roots; document adding the parent for old
session-state-only configurations. Do not silently read outside a selected root.

Use the existing progress display and failure states. All known token categories
from partial database scopes carry lower-bound presentation, not only output.
Keep SQL and ordinary report calculations equivalent. Preserve recorded model
identity and reuse the official-provider pricing resolver.

Prefer retained native JSONL transcripts. For database-only sessions, provide a
clearly labeled lossy turn-summary view when supported turns are retained; read
those strings on demand with path/size validation, never into the usage ledger.
Do not invent tool calls, reasoning, branches or per-call transcript alignment.
Absent turns produce an explicit unavailable view.

## Validation and exclusions

Synthetic tests cover WAL-only commits, invalid schema/scalars, empty snapshots,
rebuilds/deletions, identical and differing copies, all overlap outcomes, import
order, missing/reappearing sources, restart/migration, bounds and transcript
fallback. Test actual supplied accounting data only in temporary ledgers, without
committing traces. Preserve Pi, Codex, Claude and VS Code behavior. No new
dependencies, server deployment, publication, or future capture system.
