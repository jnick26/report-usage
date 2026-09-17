# Pi accounting fixtures

All source files are **synthetic**, shaped from the locally inspected Pi 0.85.1 TypeScript declarations and earlier source-contract research. Empty content/summary fields are structural placeholders, not copied conversation text. `/fixtures/...` paths and UUIDs are invented. No real-session numerical reconciliation is claimed.

Run from the repository root:

```sh
python3 tests/fixtures/pi/check_design.py
```

This checks independent expected arithmetic and draft SQLite constraints only. It is intentionally not a reference importer: the `selected` lists are explicit oracle decisions. Future importer/query tests must run the sources and assert those decisions, not copy the lists into implementation logic.

## Hand-calculated oracle

| Case | Expected result and reason |
|---|---|
| Ordinary | 100 input + 20 output + 800 cache read + 50 cache write = 970. Reasoning 5 is already inside output. Recorded USD 0.01. |
| Auxiliary all-time | Ordinary 970 + compaction (10 + 5 + 20 = 35) + branch summary (5 + 5 = 10) = 1015. Recorded USD 0.013. Auxiliary 45 is unbucketed and has unknown provider/model. |
| Auxiliary dated | Only the 970 recorded assistant tokens are attributable to the requested day. Aggregate start times are unavailable; do not distribute 45 onto their append date. |
| Distinct equal attempts | Different entry IDs each retain 970, giving 1940 and USD 0.02. |
| Proven fork | Original 100 plus new child 30 = 130. The copied original entry is an appearance, not another 100. Both import orders converge. |
| Fresh child | Parent 100 plus independent child 50 = 150. Parent linkage alone is not copied history. |
| Tool overlap | Child detail 50 counts. Parent aggregate 50 remains unresolved and is not added; its USD 0.004 does not join exact selected money. |
| Pending tail | First read retains 970. Completing the same file adds 30 exactly once: 1000. |
| Malformed middle | Malformed line 3 produces coverage; valid earlier/later entries contribute 970 + 30 = 1000. |
| Partial plus zero | Source total 120 is compatible with 100 input + 20 output; read is observed 0, write is unknown. Recorded USD zero stays zero. |
| Unknown identity | Output 50 still counts in unknown provider/model groups. |
| Aborted attempt | Recorded output 7 counts despite stop reason. |
| Identity conflict | Same session/entry has contradictory 10 and 20. Neither is an exact selected contribution; both remain unresolved. An empty known subtotal of 0 is not observed-zero usage. |
| Midnight | Europe/Kyiv midnight on this date is 21:00 UTC. Output 11 just before the boundary belongs to the earlier day; output 13 at the boundary belongs to the next. |
| Invalid count | Negative output is invalid, never clamped. Known subtotal is empty; output and total are unknown-invalid. Other observed zero categories remain valid. |
| Explicit name | Use session_info.name, not content. Ordinary accounting remains 970. |
| Repeat/move/delete | The ordinary observation remains 970 through repeated import, archive relocation and later source deletion. No new consumption identity is created. |

`cases.json` retains selected references, duplicate/unresolved references, range boundaries and sequential file operations. The checker verifies references and arithmetic; it does not simulate reimport, file moves, lineage resolution or deletion. Those sequences are required future integration tests.

## What the checks establish

- Source files and selected entries exist; literal oracle arithmetic reconciles.
- Only the two deliberately malformed source lines fail JSON decoding.
- Draft SQL creates successfully in a fresh in-memory SQLite database.
- Sixteen forbidden SQL combinations fail, including negative/unknown-valued counts, fractional integer storage, duplicate measures, broken foreign keys, missing currency, contradictory temporal variants, invalid selection states and concurrent running imports.
- A rollback restores the report revision in the checked database transaction.

## What they do not establish

No Python domain constructors, Pi importer, real filesystem snapshots, cross-file identity algorithm, per-measure reconciliation, HTTP reports, process recovery or bundle were implemented. SQL does not independently check Decimal grammar, acyclic lineage or cross-row subset relationships. The contract assigns those checks to validated construction and the commit transaction. Missing fields in permissive synthetic cases exercise uncertainty policy, not a claim that all official Pi records omit them.
