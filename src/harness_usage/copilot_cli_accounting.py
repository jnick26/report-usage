"""Cumulative counter selection for the pinned Copilot CLI event profile."""
from dataclasses import dataclass
from decimal import Decimal

from .accounting import Decision


@dataclass(frozen=True, slots=True)
class CopilotCLICandidate:
    observation_id: str
    session_id: str
    event_id: str | None
    line: int
    group: tuple[str, str | None]
    value: int | Decimal | None
    usable: bool
    reason: str | None
    source_id: str | None = None
    parent_event_id: str | None = None
    counter_epoch: str | None = None
    compatibility: str | None = None
    allow_unknown: bool = False


def reconcile_copilot_cli(candidates: tuple[CopilotCLICandidate, ...]) -> tuple[Decision, ...]:
    """Select only a physically proven cumulative maximum.

    Source-local line numbers are meaningful inside one source, never between
    copied files.  A copied source can extend another only when it retains the
    earlier compatible event identity in its own physical chain.
    """
    groups: dict[tuple[str, tuple[str, str | None]], list[CopilotCLICandidate]] = {}
    for item in candidates:
        groups.setdefault((item.session_id, item.group), []).append(item)
    decisions: list[Decision] = []
    for group in groups.values():
        # Event nodes collapse only when their full compatibility proof agrees.
        nodes: dict[tuple[str, str | None], list[CopilotCLICandidate]] = {}
        for item in group:
            event_key = (item.event_id or item.observation_id, item.compatibility)
            nodes.setdefault(event_key, []).append(item)
        node_items = {key: min(items, key=lambda item: item.observation_id)
                      for key, items in nodes.items()}
        keys = tuple(node_items)
        edges: set[tuple[tuple[str, str | None], tuple[str, str | None]]] = set()

        # ponytail: quadratic event comparison; index per-source neighbors if large histories require it.
        for left in keys:
            for right in keys:
                if left == right:
                    continue
                # Common compatible nodes retain every appearance. Each edge
                # is proved inside one physical source; shared nodes join paths.
                if any(a.source_id is not None and a.source_id == b.source_id and a.line < b.line
                       for a in nodes[left] for b in nodes[right]):
                    edges.add((left, right))

        # Different compatibility signatures for the same native event are a
        # hard conflict, even when their scalar values happen to agree.
        event_signatures: dict[str, set[str | None]] = {}
        for key, item in node_items.items():
            if item.event_id is not None:
                event_signatures.setdefault(item.event_id, set()).add(item.compatibility)
        incompatible_event = any(len(signatures) > 1 for signatures in event_signatures.values())
        incomparable = [key for key in keys if not any(left == key for left, _ in edges)]
        unresolved = incompatible_event or len(incomparable) != 1 or any(not item.usable for item in group)
        remaining = set(keys)
        while remaining:
            roots = {key for key in remaining if not any(right == key and left in remaining
                                                       for left, right in edges)}
            if not roots:
                unresolved = True
                break
            remaining -= roots
        if not unresolved:
            maximum = node_items[incomparable[0]]
            unresolved = maximum.value is None and not maximum.allow_unknown

        # Validate monotonicity only along a proven physical edge.
        if not unresolved:
            for left, right in edges:
                before, after = node_items[left], node_items[right]
                if before.value is not None and after.value is not None and after.value < before.value:
                    unresolved = True
                    break
        if unresolved:
            reason = next((item.reason for item in group if item.reason),
                          'copilot_cli_incomparable_history')
            decisions.extend(Decision(oid, 'unresolved', None, reason)
                             for oid in {item.observation_id for item in group})
            continue
        canonical_key = incomparable[0]
        canonical = node_items[canonical_key].observation_id
        for item in {item.observation_id: item for item in group}.values():
            if item.observation_id == canonical:
                decisions.append(Decision(item.observation_id, 'selected', None,
                                          'copilot_cli_cumulative_counter'))
            else:
                decisions.append(Decision(item.observation_id, 'excluded', canonical,
                                          'covered_by_later_cli_counter'))
    return tuple(sorted(decisions, key=lambda item: item.observation_id))
