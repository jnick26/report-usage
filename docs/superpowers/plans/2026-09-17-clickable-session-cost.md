# Clickable session cost implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Implement the approved clickable cumulative-cost transcript UI for Pi.

**Architecture:** Join displayed original Pi assistant entries to reconciled ledger observations using verified source provenance and native entry identity. Price through the existing catalog; render immutable branch-scoped points with existing transcript HTML/JS. No secondary database or new dependencies.

**Tech Stack:** Python, DuckDB, Jinja, native SVG, JavaScript.

**Spec:** User-approved clickable-cost-timeline mockup in the thread. Scope is the Pi transcript first, not invented per-tool billing or speculative cross-source joins.

## Global constraints

- Preserve existing uncommitted import, migration, refresh and collapsed-quantity changes.
- Keep the authoritative DuckDB ledger and existing reconciliation/pricing rules.
- Unknown is not zero; partial cumulative costs are lower bounds, never complete totals.
- Cost belongs to a model request; do not repeat it per tool or count replayed compaction context again.
- Selected-branch totals are not whole-session or subagent rollups.
- No new dependencies, network calls, raw trace commits, commits or pushes by implementers.
- Retain transcript search, branches, tool expansion, hashes, accessibility and standalone downloads.
- Follow-up UX correction: main labels distinguish branch total, this response,
  and total so far; use readable cents and retain exact amounts in details.
  Nonzero sub-cent values must not appear as zero. Axis labels must not wrap.
- Two implementers own disjoint files; root integrates and validates. User requested parallel execution and context reuse.

## Shared interface

Task 1 defines these frozen dataclasses/fields in transcript.py. Task 2 consumes them:

```python
@dataclass(frozen=True, slots=True)
class TranscriptCostPoint:
    message_id: str
    amount: Decimal | None
    cumulative: Decimal
    incomplete: bool
    cumulative_incomplete: bool
    calculations: tuple[CostLine, ...] = ()

# Append defaulted fields to existing models:
# Message.usage_id: str | None = None
# TranscriptPage.costs: tuple[TranscriptCostPoint, ...] = ()
```

`message_id` is the displayed Message.id; usage_id is original Pi native entry id,
not a tool-call id or synthetic checkpoint id. Amount is None if no component is
priced, including unmatched/conflicting evidence; known partial subtotal is
allowed only with incomplete=True. Cumulative is exact_sum of known selected
amounts in displayed branch order. cumulative_incomplete becomes sticky after
any unavailable/partial point. Zero remains a valid known cost.

### Task 1: Trusted branch-scoped cost projection

**Own files:** transcript.py, pi_transcript.py, transcript_access.py, application.py,
new transcript_costs.py, focused new tests/test_transcript_costs.py; storage.py only
if a small session-scoped query helper is necessary. Do not edit UI files.

**Consumes:** Storage's reconciled decisions, source provenance, Catalog.price.
**Produces:** Shared dataclasses and populated TranscriptPage.costs for Pi.

- [x] Write failing synthetic Pi import/transcript tests: two requests with literal
  prices produce amount and cumulative values; alternate branch excludes sibling
  request; unknown model produces unavailable point and sticky incomplete total.
- [x] Add usage identity only to original Pi assistant entries; no synthetic
  compaction replay identity. Preserve existing parser behavior and defaults.
- [x] Verify source bytes against imported provenance before joining to selected
  observations. Changed/unimported/ambiguous source must never get stale costs.
  Limit reads to the selected session/source; no full-ledger snapshot.
- [x] Populate costs via Catalog.price and exact_sum; preserve pricing reasons in
  calculations. Tool-only assistant responses can carry one request cost; user
  and tool-result messages cannot. Excluded/conflicting evidence must not count.
- [x] Test duplicate sources, invalid/missing usage, known zero, partial cache
  pricing and branch replay; run focused pytest and strict mypy. Report RED/GREEN.

### Task 2: Clickable timeline and response labels

**Own files:** transcript_rendering.py, templates/transcript.html,
static/transcript.css, static/transcript.js, focused new tests/test_transcript_cost_ui.py,
tests/check_transcript_cost.cjs and necessary existing UI tests. No backend edits.

**Consumes:** TranscriptPage.costs, linked by Message.id to existing safe anchors.
**Produces:** Compact cumulative chart, response labels and cost details.

- [x] First write failing render/JS behavior checks using literal cost points:
  chart and per-response labels, point target correct even inside collapsed tool
  activity; unknown not zero, lower-bound labels, standalone self-contained.
- [x] Render selected-branch API-equivalent estimate and priced request coverage.
  Reuse cost calculations for expandable token/cache/rate details, no tool price
  allocation. Keep exact decimal strings in labels; JS numbers only for geometry.
- [x] Native SVG chart: request order on x, cumulative USD on y; every request
  keyboard/click reachable. Gap/missing indicators and >= lower-bound labels.
  Thousands of requests must not generate overlapping tabbable dots: native
  range request selector plus pointer mapping is acceptable with visible selection.
- [x] Click selects/highlights actual response, opens enclosing details, scrolls
  to it without replacing transcript. Hover previews; scrolling follows current
  response. Respect reduced motion, narrow widths, no extra scroll container.
- [x] Remove the old independently scrolling history rail, retaining search,
  existing hash navigation and Latest. Collapse extension metadata as a group,
  not one noisy row per record. Preserve other source transcripts without costs.
- [x] Run focused render/frontend tests and strict mypy; report RED/GREEN evidence.

### Task 3: Integration and review

**Own files:** plan/progress and targeted integration tests if needed.

- [x] Verify actual supplied Pi session renders priced points and correct anchors
  without exposing trace content in outputs; use copied ledger for read checks.
- [x] Run isolated full pytest, frontend JS checks, strict mypy and secrets scan.
- [x] Review focused delta from preserved baseline, fix concrete issues via owner.
- [x] Restart only the owned local preview safely after import has completed,
  preserve its ledger/roots and verify the live transcript route. Do not push.

## Follow-up visual refinement (2026-09-18)

User requested the visible response slider be replaced by vertical bars, with
height representing summed costs over short spans of responses. Retain click
and keyboard navigation, use grey chart styling and a white selected marker.
Bins describe response ranges, not elapsed-time periods. Unknown/partial costs
remain distinct from known zero. Existing accounting is unchanged.
