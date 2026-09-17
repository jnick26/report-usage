"""Current-alternate and credit-control decisions for VS Code chat evidence."""
from dataclasses import dataclass
from decimal import Decimal

from .accounting import Decision


@dataclass(frozen=True, slots=True)
class CopilotVscodeCandidate:
    observation_id: str
    session_id: str
    source_id: str
    locator: str
    current: bool
    available: bool
    representation: str
    request_id: str | None
    response_id: str | None
    model: str | None
    evidence_kind: str
    request_ordinal: int
    line: int
    fingerprint: str
    compatibility: str
    state: str
    reason: str | None
    quantity_state: str | None = None
    quantity_amount: Decimal | None = None

    @property
    def logical_key(self) -> tuple[object, ...]:
        identity: object = self.request_id if self.request_id is not None else (
            self.source_id, self.request_ordinal)
        return (self.session_id, identity, self.response_id, self.evidence_kind,
                self.model if self.evidence_kind == 'model_totals' else None)

    @property
    def sort_key(self) -> tuple[int, str, int, str]:
        return (0 if self.representation == 'operation_log' else 1,
                self.locator, self.line, self.observation_id)


@dataclass(frozen=True, slots=True)
class CopilotVscodeSource:
    source_id: str
    session_id: str
    locator: str
    available: bool
    representation: str

    @property
    def sort_key(self) -> tuple[int, str]:
        return (0 if self.representation == 'operation_log' else 1, self.locator)


def reconcile_copilot_vscode(candidates: tuple[CopilotVscodeCandidate, ...],
                             current_sources: tuple[CopilotVscodeSource, ...] = ()) -> tuple[tuple[Decision, ...], tuple[Decision, ...]]:
    token: dict[str, Decision] = {}
    by_session: dict[str, list[CopilotVscodeCandidate]] = {}
    for candidate in candidates:
        by_session.setdefault(candidate.session_id, []).append(candidate)
    sources_by_session: dict[str, list[CopilotVscodeSource]] = {}
    canonical_sources: dict[str, str] = {}
    for source in current_sources:
        sources_by_session.setdefault(source.session_id, []).append(source)
        by_session.setdefault(source.session_id, [])
    for session_id, session in by_session.items():
        current = [item for item in session if item.current]
        available = [item for item in current if item.available]
        active_rows = available or current
        if not active_rows:
            if not sources_by_session.get(session_id):
                for observation_id in {item.observation_id for item in session}:
                    token[observation_id] = Decision(observation_id, 'unresolved', None,
                                                     'copilot_vscode_no_current_usage')
                continue
            active_rows = list(session)
        by_source: dict[str, list[CopilotVscodeCandidate]] = {}
        for item in active_rows:
            by_source.setdefault(item.source_id, []).append(item)
        sources = sources_by_session.get(session_id, [])
        active_sources = [source for source in sources if source.available] or sources
        for source in active_sources:
            by_source.setdefault(source.source_id, [])
        signatures = {
            source_id: tuple(sorted({(repr(item.logical_key[1:]), item.fingerprint, item.compatibility) for item in rows}))
            for source_id, rows in by_source.items()
        }
        if len(set(signatures.values())) > 1:
            for observation_id in {item.observation_id for item in session}:
                token[observation_id] = Decision(observation_id, 'unresolved', None,
                                                 'copilot_vscode_alternate_conflict')
            continue
        canonical_source = (min(active_sources, key=lambda source: source.sort_key).source_id
                            if active_sources else min(active_rows, key=lambda item: item.sort_key).source_id)
        canonical_sources[session_id] = canonical_source
        canonical_rows = by_source[canonical_source]
        if not canonical_rows:
            for observation_id in {item.observation_id for item in session}:
                token[observation_id] = Decision(observation_id, 'unresolved', None,
                                                 'copilot_vscode_no_current_usage')
            continue
        fallback = min(canonical_rows, key=lambda item: item.sort_key).observation_id
        groups: dict[tuple[object, ...], list[CopilotVscodeCandidate]] = {}
        for item in session:
            groups.setdefault(item.logical_key, []).append(item)
        for group in groups.values():
            active = [item for item in group if item.source_id == canonical_source]
            observations = {item.observation_id for item in group}
            if not active:
                for observation_id in observations:
                    token[observation_id] = (Decision(observation_id, 'excluded', fallback, 'stale_vscode_generation')
                                             if observation_id != fallback else
                                             Decision(observation_id, 'selected', None, 'copilot_vscode_request'))
                continue
            conflict = (any(item.state != 'usable' or item.request_id is None for item in active)
                        or len({(item.fingerprint, item.compatibility) for item in active}) > 1)
            if conflict:
                reason = next((item.reason for item in active if item.reason), 'copilot_vscode_identity_conflict')
                for observation_id in observations:
                    token[observation_id] = Decision(observation_id, 'unresolved', None, reason)
                continue
            canonical = min(active, key=lambda item: item.sort_key).observation_id
            for observation_id in observations:
                token[observation_id] = (Decision(observation_id, 'selected', None, 'copilot_vscode_request')
                                         if observation_id == canonical else
                                         Decision(observation_id, 'excluded', canonical, 'copilot_vscode_alternate'))

    alternate = dict(token)
    by_request: dict[tuple[str, str | None, str | None], list[CopilotVscodeCandidate]] = {}
    for candidate in candidates:
        by_request.setdefault((candidate.session_id, candidate.request_id, candidate.response_id), []).append(candidate)
    for group in by_request.values():
        selected_models = sorted((item for item in group if item.evidence_kind == 'model_totals'
                                  and token[item.observation_id].state == 'selected'), key=lambda item: item.sort_key)
        selected_summaries = sorted((item for item in group if item.evidence_kind in ('turn_summary', 'unavailable')
                                     and token[item.observation_id].state == 'selected'), key=lambda item: item.sort_key)
        target = selected_models[0] if selected_models else selected_summaries[0] if selected_summaries else None
        if selected_models:
            assert target is not None
            for item in selected_summaries:
                token[item.observation_id] = Decision(item.observation_id, 'excluded', target.observation_id,
                                                       'covered_by_model_totals')
        if target is not None:
            for item in group:
                if item.evidence_kind == 'session_control' and token[item.observation_id].state == 'selected':
                    token[item.observation_id] = Decision(item.observation_id, 'excluded', target.observation_id,
                                                           'quantity_only_control')

    quantity = {item.observation_id: alternate[item.observation_id] for item in candidates
                if item.quantity_state is not None}
    by_session = {}
    for item in candidates:
        if item.quantity_state is not None:
            by_session.setdefault(item.session_id, []).append(item)
    for group in by_session.values():
        quantity_source = canonical_sources.get(group[0].session_id)
        canonical_rows = [item for item in group if item.source_id == quantity_source]
        controls = [item for item in canonical_rows if item.evidence_kind == 'session_control']
        if any(quantity[item.observation_id].state == 'unresolved' for item in controls):
            for item in group:
                quantity[item.observation_id] = Decision(item.observation_id, 'unresolved', None,
                                                          'copilot_vscode_control_conflict')
            continue
        selected = [item for item in controls if quantity[item.observation_id].state == 'selected'
                    and item.quantity_state == 'known' and item.quantity_amount is not None]
        if not selected:
            continue
        amounts = [item.quantity_amount for item in selected if item.quantity_amount is not None]
        maximum = max(amounts)
        control = min((item for item in selected if item.quantity_amount == maximum),
                      key=lambda item: (-item.request_ordinal, item.sort_key))
        for item in group:
            decision = quantity[item.observation_id]
            appearances = [current for current in canonical_rows
                           if current.observation_id == item.observation_id]
            if item.observation_id == control.observation_id:
                quantity[item.observation_id] = Decision(item.observation_id, 'selected', None,
                                                          'copilot_vscode_session_control')
            elif decision.state != 'unresolved' and (
                item.evidence_kind == 'session_control' or
                any(current.request_ordinal <= control.request_ordinal for current in appearances)
            ):
                quantity[item.observation_id] = Decision(item.observation_id, 'excluded', control.observation_id,
                                                          'covered_by_session_control')
    return (tuple(token[key] for key in sorted(token)),
            tuple(quantity[key] for key in sorted(quantity)))
