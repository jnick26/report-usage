"""Codex request identities and explicitly related legacy replay evidence."""
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from .accounting import Decision


@dataclass(frozen=True, slots=True)
class CodexCandidate:
    id: str
    session_id: str
    source_kind: Literal['response', 'legacy']
    response_id: str | None
    cumulative_key: str | None
    last_key: str | None
    value: str
    at: int | None
    usable: bool
    reason: str | None
    mirror_response_id: str | None = None


@dataclass(frozen=True, slots=True)
class CodexAppearance:
    source_id: str
    session_id: str
    observation_id: str
    line: int
    cumulative_key: str | None
    last_key: str | None


def reconcile_codex(candidates: tuple[CodexCandidate, ...], parents: Mapping[str, str | None],
                    missing: set[str], conflicts: set[str], appearances: tuple[CodexAppearance, ...]) -> tuple[Decision, ...]:
    responses: dict[str, list[CodexCandidate]] = defaultdict(list)
    legacy: dict[tuple[str, str], list[CodexCandidate]] = defaultdict(list)
    for candidate in candidates:
        if candidate.source_kind == 'response' and candidate.response_id is not None:
            responses[candidate.response_id].append(candidate)
        elif candidate.cumulative_key is not None:
            legacy[candidate.session_id, candidate.cumulative_key].append(candidate)
    own_keys: dict[str, set[tuple[str, str | None]]] = defaultdict(set)
    modern_keys: dict[tuple[str, str, str], list[CodexCandidate]] = defaultdict(list)
    for candidate in candidates:
        if candidate.cumulative_key is not None:
            if candidate.source_kind == 'legacy':
                own_keys[candidate.session_id].add((candidate.cumulative_key, candidate.last_key))
            elif candidate.last_key is not None and candidate.response_id is not None:
                modern_keys[candidate.session_id, candidate.cumulative_key, candidate.last_key].append(candidate)
    ancestor_keys: dict[str, set[tuple[str, str | None]]] = {}
    for sid in {c.session_id for c in candidates}:
        keys: set[tuple[str, str | None]] = set()
        ancestor = parents.get(sid)
        seen = {sid}
        while ancestor is not None and ancestor not in seen:
            seen.add(ancestor)
            keys.update(own_keys.get(ancestor, ()))
            ancestor = parents.get(ancestor)
        ancestor_keys[sid] = keys
    sources: dict[str, list[CodexAppearance]] = defaultdict(list)
    for appearance in appearances:
        sources[appearance.source_id].append(appearance)
    replay_prefix: set[str] = set()
    after_boundary: set[str] = set()
    for source in sources.values():
        inherited = True
        for appearance in sorted(source, key=lambda item: item.line):
            inherited = inherited and (appearance.cumulative_key, appearance.last_key) in ancestor_keys.get(appearance.session_id, set())
            (replay_prefix if inherited else after_boundary).add(appearance.observation_id)
    replay_prefix.difference_update(after_boundary)
    decisions: dict[str, Decision] = {}
    visiting: set[str] = set()

    def canonical(group: list[CodexCandidate]) -> CodexCandidate:
        return min(group, key=lambda c: (c.at is None, c.at or 0, c.id))

    def exclude(candidate: CodexCandidate, original: CodexCandidate, reason: str) -> Decision:
        selected = select(original)
        if selected.state == 'unresolved':
            return Decision(candidate.id, 'unresolved', None, selected.reason)
        return Decision(candidate.id, 'excluded', selected.canonical or selected.observation_id, reason)

    def select(c: CodexCandidate) -> Decision:
        if c.id in decisions:
            return decisions[c.id]
        if c.id in visiting:
            return Decision(c.id, 'unresolved', None, 'codex_lineage_cycle')
        visiting.add(c.id)
        result = Decision(c.id, 'selected', None, 'codex_response' if c.source_kind == 'response' else 'codex_cumulative_advance')
        if c.session_id in conflicts:
            result = Decision(c.id, 'unresolved', None, 'codex_identity_conflict')
        elif c.mirror_response_id is not None:
            matches = responses.get(c.mirror_response_id, [])
            if matches and all(item.value == c.value for item in matches):
                result = exclude(c, canonical(matches), 'codex_modern_legacy_mirror')
            else:
                result = Decision(c.id, 'unresolved', None, 'codex_modern_legacy_overlap')
        elif c.source_kind == 'legacy' and c.cumulative_key is not None and c.last_key is not None and (mirror_matches := modern_keys.get((c.session_id, c.cumulative_key, c.last_key))):
            if len({item.response_id for item in mirror_matches}) == 1 and all(item.value == c.value for item in mirror_matches):
                result = exclude(c, canonical(mirror_matches), 'codex_modern_legacy_mirror')
            else:
                result = Decision(c.id, 'unresolved', None, 'codex_modern_legacy_overlap')
        elif not c.usable:
            result = Decision(c.id, 'unresolved', None, c.reason or 'codex_usage_unresolved')
        elif c.source_kind == 'response':
            group = responses.get(c.response_id or '', [c])
            if len({item.session_id for item in group}) != 1:
                result = Decision(c.id, 'unresolved', None, 'codex_response_owner_conflict')
            elif len({item.value for item in group}) != 1:
                result = Decision(c.id, 'unresolved', None, 'codex_response_conflict')
            else:
                original = canonical(group)
                if original.id != c.id:
                    result = exclude(c, original, 'codex_response_duplicate')
        elif c.cumulative_key is None or c.last_key is None:
            result = Decision(c.id, 'unresolved', None, 'codex_missing_cumulative')
        else:
            group = legacy[c.session_id, c.cumulative_key]
            if len({item.value for item in group}) != 1:
                result = Decision(c.id, 'unresolved', None, 'codex_cumulative_conflict')
            elif canonical(group).id != c.id:
                result = exclude(c, canonical(group), 'codex_legacy_duplicate')
            else:
                ancestor: str | None = c.session_id
                seen: set[str] = set()
                while ancestor is not None:
                    if ancestor in seen:
                        result = Decision(c.id, 'unresolved', None, 'codex_lineage_cycle')
                        break
                    seen.add(ancestor)
                    if ancestor in missing:
                        result = Decision(c.id, 'unresolved', None, 'codex_missing_ancestor')
                        break
                    ancestor = parents.get(ancestor)
                    if ancestor is None:
                        break
                    matches = [item for item in legacy.get((ancestor, c.cumulative_key), []) if item.last_key == c.last_key] if c.id in replay_prefix else []
                    if matches:
                        if any(item.value != c.value for item in matches):
                            result = Decision(c.id, 'unresolved', None, 'codex_replay_conflict')
                        else:
                            result = exclude(c, canonical(matches), 'codex_legacy_replay')
                        break
        visiting.remove(c.id)
        decisions[c.id] = result
        return result

    return tuple(select(candidate) for candidate in candidates)
