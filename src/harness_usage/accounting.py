"""Conservative identity selection; no file or database access."""
from dataclasses import dataclass
from collections import defaultdict
from typing import Literal

from .domain import ContractViolation

MEASURES = ('input', 'output', 'cache_read', 'cache_write', 'total', 'recorded_usd')


@dataclass(frozen=True, slots=True)
class Candidate:
    id: str
    session_id: str
    native_id: str
    parent_id: str | None
    fingerprint: str
    kind: str
    pending: bool = False


@dataclass(frozen=True, slots=True)
class Decision:
    observation_id: str
    state: Literal['selected', 'excluded', 'unresolved']
    canonical: str | None
    reason: str

    def __post_init__(self) -> None:
        if self.state not in ('selected', 'excluded', 'unresolved') or not isinstance(self.observation_id, str) or not self.observation_id.strip() or not isinstance(self.reason, str) or not self.reason.strip():
            raise ContractViolation('invalid_decision')
        if self.state == 'excluded':
            if not isinstance(self.canonical, str) or not self.canonical.strip() or self.canonical == self.observation_id:
                raise ContractViolation('invalid_exclusion_target')
        elif self.canonical is not None:
            raise ContractViolation('unexpected_exclusion_target')


def reconcile(candidates: tuple[Candidate, ...], parents: dict[str, str | None],
              edges: dict[str, dict[str, str | None]], missing_parents: set[str],
              conflicts: set[str]) -> tuple[Decision, ...]:
    by_entry: dict[tuple[str, str], list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        by_entry[candidate.session_id, candidate.native_id].append(candidate)
    representatives = {key: min(group, key=lambda item: item.id) for key, group in by_entry.items()}
    conflicting_entries = {key for key, group in by_entry.items() if len({item.fingerprint for item in group}) > 1}
    decisions: dict[str, Decision] = {}
    visiting: set[str] = set()

    path_cache: dict[tuple[str, str, str], bool] = {}

    def matching_path(ancestor: str, child: str, node: str | None) -> bool:
        parent_edges, child_edges = edges.get(ancestor, {}), edges.get(child, {})
        visited: set[str] = set()
        matched = True
        while node is not None:
            cached = path_cache.get((ancestor, child, node))
            if cached is not None:
                matched = cached
                break
            if node in visited or node not in parent_edges or node not in child_edges or parent_edges[node] != child_edges[node]:
                matched = False
                break
            visited.add(node)
            parent_usage, child_usage = representatives.get((ancestor, node)), representatives.get((child, node))
            if (ancestor, node) in conflicting_entries or (child, node) in conflicting_entries:
                matched = False
                break
            if (parent_usage is None) != (child_usage is None) or (parent_usage is not None and child_usage is not None and parent_usage.fingerprint != child_usage.fingerprint):
                matched = False
                break
            node = child_edges[node]
        for entry in visited:
            path_cache[ancestor, child, entry] = matched
        return matched

    def select(c: Candidate) -> Decision:
        if c.id in decisions:
            return decisions[c.id]
        result = Decision(c.id, 'selected', None, 'independent_entry')
        if c.id in visiting:
            return Decision(c.id, 'unresolved', None, 'lineage_cycle')
        visiting.add(c.id)
        if c.session_id in conflicts or (c.session_id, c.native_id) in conflicting_entries:
            result = Decision(c.id, 'unresolved', None, 'identity_conflict')
        elif c.kind == 'tool_result':
            result = Decision(c.id, 'unresolved', None, 'tool_overlap_unproven')
        elif c.pending:
            result = Decision(c.id, 'unresolved', None, 'unfinished_attempt')
        elif c.session_id in missing_parents:
            result = Decision(c.id, 'unresolved', None, 'missing_ancestor')
        elif representatives[c.session_id, c.native_id].id != c.id:
            canonical_decision = select(representatives[c.session_id, c.native_id])
            result = (Decision(c.id, 'unresolved', None, canonical_decision.reason) if canonical_decision.state == 'unresolved'
                      else Decision(c.id, 'excluded', canonical_decision.canonical or canonical_decision.observation_id, 'identical_measure_variant'))
        else:
            ancestor = parents.get(c.session_id)
            seen = {c.session_id}
            while ancestor:
                if ancestor in seen:
                    result = Decision(c.id, 'unresolved', None, 'lineage_cycle')
                    break
                seen.add(ancestor)
                representative = representatives.get((ancestor, c.native_id))
                matches = [representative] if representative is not None else []
                if matches:
                    if len(matches) != 1 or matches[0].fingerprint != c.fingerprint or not matching_path(ancestor, c.session_id, c.native_id):
                        result = Decision(c.id, 'unresolved', None, 'copied_identity_conflict')
                    else:
                        selected = select(matches[0])
                        if selected.state == 'unresolved':
                            result = Decision(c.id, 'unresolved', None, selected.reason)
                        else:
                            canonical = selected.canonical or selected.observation_id
                            result = Decision(c.id, 'excluded', canonical, 'proven_copied_path')
                    break
                if ancestor in missing_parents:
                    result = Decision(c.id, 'unresolved', None, 'missing_ancestor')
                    break
                ancestor = parents.get(ancestor)
        visiting.remove(c.id)
        decisions[c.id] = result
        return result

    return tuple(select(c) for c in candidates)
