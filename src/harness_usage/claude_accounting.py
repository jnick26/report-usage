"""Deterministic Claude call identity reconciliation."""
from dataclasses import dataclass

from .accounting import Decision


@dataclass(frozen=True, slots=True)
class ClaudeCandidate:
    observation_id: str
    session_id: str
    message_id: str | None
    request_id: str | None
    entry_uuid: str
    signature: str
    sort_key: tuple[int, int, str, int, str]
    state: str
    reason: str | None
    output_provenance: tuple[str, int | None] = ('missing', None)
    output_finality: str = 'unqualified'
    current_projection: bool = True
    writer_token: str | None = None
    superseded_projection: bool = False


def _components(candidates: tuple[ClaudeCandidate, ...]) -> tuple[tuple[ClaudeCandidate, ...], ...]:
    parent = list(range(len(candidates)))

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left, right = root(left), root(right)
        if left != right:
            parent[right] = left

    nodes: dict[tuple[str, str], int] = {}
    observations: dict[str, int] = {}
    for index, candidate in enumerate(candidates):
        previous = observations.setdefault(candidate.observation_id, index)
        union(index, previous)
        for kind, value in (('message', candidate.message_id), ('request', candidate.request_id), ('entry', candidate.entry_uuid)):
            if value is None:
                continue
            node = (kind, value)
            previous = nodes.setdefault(node, index)
            union(index, previous)
    groups: dict[int, list[ClaudeCandidate]] = {}
    for index, candidate in enumerate(candidates):
        groups.setdefault(root(index), []).append(candidate)
    return tuple(tuple(group) for group in groups.values())


def claude_output_variations(candidates: tuple[ClaudeCandidate, ...]) -> set[str]:
    return {candidate.observation_id for group in _components(candidates)
            for active in (tuple(candidate for candidate in group
                                 if not candidate.superseded_projection) or group,)
            if len({candidate.output_provenance for candidate in active}) > 1
            for candidate in active}


def reconcile_claude(candidates: tuple[ClaudeCandidate, ...]) -> tuple[Decision, ...]:
    decisions: dict[str, Decision] = {}
    for group in _components(candidates):
        active_group = tuple(candidate for candidate in group
                             if not candidate.superseded_projection) or group
        by_observation: dict[str, ClaudeCandidate] = {}
        identified_by_observation: dict[str, ClaudeCandidate] = {}
        def rank(candidate: ClaudeCandidate) -> tuple[int, tuple[int, int, str, int, str]]:
            return (0 if candidate.current_projection else 1, candidate.sort_key)

        for candidate in group:
            previous = by_observation.get(candidate.observation_id)
            if previous is None or rank(candidate) < rank(previous):
                by_observation[candidate.observation_id] = candidate
            if (not candidate.superseded_projection
                    and (candidate.message_id is not None or candidate.request_id is not None)):
                previous = identified_by_observation.get(candidate.observation_id)
                if previous is None or rank(candidate) < rank(previous):
                    identified_by_observation[candidate.observation_id] = candidate
        messages: dict[str, set[str]] = {}
        requests: dict[str, set[str]] = {}
        entry_messages: dict[str, set[str]] = {}
        entry_requests: dict[str, set[str]] = {}
        for candidate in group:
            if candidate.message_id is not None:
                entry_messages.setdefault(candidate.entry_uuid, set()).add(candidate.message_id)
            if candidate.request_id is not None:
                entry_requests.setdefault(candidate.entry_uuid, set()).add(candidate.request_id)
            if candidate.message_id is not None and candidate.request_id is not None:
                messages.setdefault(candidate.message_id, set()).add(candidate.request_id)
                requests.setdefault(candidate.request_id, set()).add(candidate.message_id)
        mapping_conflict = any(len(values) > 1 for values in (*messages.values(), *requests.values()))
        entry_identity_conflict = any(
            len(values) > 1 for values in (*entry_messages.values(), *entry_requests.values()))
        identified_entries = {candidate.entry_uuid for candidate in active_group
                              if candidate.message_id is not None or candidate.request_id is not None}
        identityless = {candidate.observation_id for candidate in active_group
                        if candidate.message_id is None and candidate.request_id is None
                        and candidate.entry_uuid not in identified_entries}
        unusable = any(
            candidate.state != 'usable'
            and not (candidate.message_id is None and candidate.request_id is None
                     and candidate.entry_uuid in identified_entries
                     and candidate.reason == 'missing_claude_identity')
            for candidate in active_group if candidate.observation_id not in identityless)
        final_values = {candidate.output_provenance[1] for candidate in active_group
                        if candidate.output_finality == 'final'}
        current_entry_finalities: dict[str, set[str]] = {}
        for candidate in active_group:
            if candidate.current_projection:
                current_entry_finalities.setdefault(candidate.entry_uuid, set()).add(
                    candidate.output_finality)
        contradictory_entry_finality = any(
            'final' in values and values != {'final'}
            for values in current_entry_finalities.values())
        writer_tokens = {candidate.writer_token for candidate in active_group
                         if candidate.writer_token is not None}
        final_writer_tokens = {candidate.writer_token for candidate in active_group
                               if candidate.output_finality == 'final'
                               and candidate.writer_token is not None}
        placeholder_writer_tokens = {candidate.writer_token for candidate in active_group
                                     if candidate.output_finality == 'placeholder'
                                     and candidate.writer_token is not None}
        refinement_writer_pair = (
            len(final_writer_tokens) == 1
            and len(placeholder_writer_tokens) <= 1
            and all(candidate.output_finality == 'final'
                    and candidate.writer_token in final_writer_tokens
                    or candidate.output_finality == 'placeholder'
                    or candidate.output_finality == 'unqualified'
                    and candidate.writer_token is None
                    for candidate in active_group))
        writer_conflict = len(writer_tokens) > 1 and not refinement_writer_pair
        unproven_final_equality = (
            any(candidate.output_finality == 'final' and candidate.writer_token is None
                for candidate in active_group)
            and any(candidate.output_finality == 'final' and candidate.writer_token is not None
                    for candidate in active_group))
        output_conflict = (len(final_values) > 1
                           or bool(final_values) and any(
                               candidate.output_finality in ('final_missing', 'invalid')
                               for candidate in active_group)
                           or any(candidate.output_finality == 'invalid'
                                  for candidate in active_group)
                           or contradictory_entry_finality or writer_conflict
                           or unproven_final_equality)
        conflict = (mapping_conflict or entry_identity_conflict
                    or len({candidate.signature for candidate in active_group}) > 1 or unusable)
        if output_conflict or conflict:
            reason = 'claude_output_conflict' if output_conflict else 'claude_identity_conflict'
            for observation_id in by_observation:
                decisions[observation_id] = Decision(observation_id, 'unresolved', None, reason)
            continue
        if identified_by_observation:
            final_candidates = [candidate for candidate in identified_by_observation.values()
                                if candidate.output_finality == 'final']
            canonical = min(final_candidates or list(identified_by_observation.values()), key=rank)
            active_observations = {candidate.observation_id for candidate in active_group}
            for observation_id in by_observation:
                if observation_id not in active_observations:
                    decisions[observation_id] = Decision(
                        observation_id, 'excluded', canonical.observation_id,
                        'claude_superseded_projection')
                elif observation_id in identityless:
                    decisions[observation_id] = Decision(observation_id, 'unresolved', None, 'missing_claude_identity')
                elif observation_id == canonical.observation_id:
                    decisions[observation_id] = Decision(observation_id, 'selected', None, 'claude_call')
                else:
                    decisions[observation_id] = Decision(observation_id, 'excluded', canonical.observation_id, 'claude_duplicate')
        else:
            for observation_id in by_observation:
                decisions[observation_id] = Decision(observation_id, 'unresolved', None, 'missing_claude_identity')
    return tuple(decisions[key] for key in sorted(decisions))
