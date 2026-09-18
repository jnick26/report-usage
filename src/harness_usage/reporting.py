"""Immutable reports over one committed population of reconciled evidence."""
from bisect import bisect_right
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal, localcontext
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .domain import (
    Assigned, Attribution, ContractViolation, Interval, Known, MeasuredQuantity, ModelIdentity,
    NotApplicable, ObservationId, Point, ProjectId, RecordedEstimate, RecordedMoney,
    SessionId, TimeEvidence, TokenEvidence, TokenValue, Unassigned, Undated, instant,
)

from .pricing import Catalog, CostLine, ObservationCost, aggregate, exact_sum

SessionSort = Literal['started', 'cost_desc', 'cost_asc']

MEASURES = ('input', 'output', 'cache_read', 'cache_write', 'total', 'recorded_usd')

# Only fixed categories cross the ledger-to-presentation boundary.
COVERAGE_TEXT = {
    'identity_conflict': 'Conflicting evidence remains unresolved.',
    'unresolved_evidence': 'Unresolved usage is excluded from recorded totals.',
    'interval_only': 'Cumulative usage has no per-call timing.',
    'timing_unavailable': 'Usage with unavailable or partially overlapping timing cannot be allocated to this range.',
    'unknown_model_identity': 'Recorded model or provider identity is incomplete.',
    'unassigned': 'Project attribution is unavailable.',
    'usage_unavailable': 'Usage unavailable',
    'usage_partial': 'Some usage is unavailable.',
    'not_applicable': 'Some measures are not applicable.',
    'lower_bound': 'Lower bound: retained evidence may omit usage.',
    'writer_version_unavailable': 'Writer version is unavailable; final output may be unavailable.',
    'cwd_attribution_unavailable': 'Project attribution is unavailable.',
    'claude_output_variation': 'Conflicting Claude Code output counters remain unresolved.',
    'lossy_turn_summary': 'Copilot in VS Code retained a lossy usage summary; cache and actual-model detail may be unavailable.',
    'events_unavailable': 'Older Copilot CLI history has no recoverable usage events.',
    'cli_partial_lifetime': 'Copilot CLI history covers only part of the session lifetime.',
    'invalid_cli_counter': 'Invalid Copilot CLI counters remain unavailable.',
    'copilot_store_conflict': 'Copilot CLI index and history evidence cannot be reconciled safely.',
    'incompatible_session_epoch': 'Conflicting Copilot CLI lifetime evidence remains unresolved.',
    'invalid_event_chain': 'Copilot CLI event history is incomplete or inconsistent.',
    'event_chain_gap': 'Copilot CLI event history has gaps.',
    'workspace_changed_cumulative_scope': 'Cumulative Copilot CLI usage cannot be allocated across workspace changes.',
    'saved_history': 'Saved history includes sources that are currently unavailable.',
    'unsupported_profile': 'This writer profile is unsupported; usage may be unavailable.',
    'codex_model_conflict': 'Conflicting model evidence cannot be priced.',
    'subagent_parent_unresolved': 'Subagent ownership is unresolved.',
    'copilot_cross_source_join_unavailable': 'Copilot CLI and Copilot in VS Code were recorded independently and may overlap. No exact cross-source request join is available; combined Copilot usage is not an exact billing total.',
}
DIAGNOSTIC_CODES = tuple(code for code in COVERAGE_TEXT if code not in {
    'unresolved_evidence', 'interval_only', 'timing_unavailable', 'unknown_model_identity',
    'unassigned', 'usage_unavailable', 'usage_partial', 'not_applicable', 'lower_bound',
    'copilot_cross_source_join_unavailable', 'saved_history',
})
OVERLAP_USAGE_CODES = frozenset(('_known_usage', 'usage_partial', 'unresolved_evidence'))


@dataclass(frozen=True, slots=True)
class AllTime:
    pass


@dataclass(frozen=True, slots=True)
class DateRange:
    start: datetime
    end: datetime
    timezone: str

    def __post_init__(self) -> None:
        object.__setattr__(self, 'start', instant(self.start))
        object.__setattr__(self, 'end', instant(self.end))
        _zone(self.timezone)
        if self.start >= self.end:
            raise ContractViolation('range_start_must_precede_end')


ReportRange = AllTime | DateRange


@dataclass(frozen=True, slots=True)
class ReportQuery:
    project_id: ProjectId | None
    range: ReportRange
    bucket: Literal['day', 'hour', 'five_minutes'] = 'day'

    def __post_init__(self) -> None:
        if not isinstance(self.range, (AllTime, DateRange)) or self.bucket not in ('day', 'hour', 'five_minutes'):
            raise ContractViolation('invalid_report_query')


@dataclass(frozen=True, slots=True)
class MetricSum:
    known: int = 0
    unknown_observations: int = 0
    not_applicable_observations: int = 0
    lower_bound_observations: int = 0


@dataclass(frozen=True, slots=True)
class TokenSums:
    input: MetricSum
    output: MetricSum
    cache_read: MetricSum
    cache_write: MetricSum
    total: MetricSum


@dataclass(frozen=True, slots=True)
class MoneySum:
    known: Decimal
    missing_observations: int
    basis: Literal['pi_recorded_usd', 'models_dev_usd'] = 'pi_recorded_usd'
    calculations: tuple[CostLine, ...] = ()
    snapshot_date: str | None = None


@dataclass(frozen=True, slots=True)
class Coverage:
    code: str
    observation_ids: tuple[ObservationId, ...]


@dataclass(frozen=True, slots=True)
class Bucket:
    start: datetime
    end: datetime
    tokens: TokenSums
    money: MoneySum | None = None


@dataclass(frozen=True, slots=True)
class ElapsedSpan:
    first: datetime
    last: datetime
    partial_history: bool


@dataclass(frozen=True, slots=True)
class UnknownElapsed:
    reason: str


@dataclass(frozen=True, slots=True)
class SessionRow:
    id: SessionId
    name: str
    attribution: Attribution
    elapsed: ElapsedSpan | UnknownElapsed
    tokens: TokenSums
    money: MoneySum
    buckets: tuple[Bucket, ...]
    coverage: tuple[Coverage, ...]
    started: datetime | None = None
    models: tuple[ModelIdentity, ...] = ()
    subagent_count: int = 0
    harness: str = 'pi'


@dataclass(frozen=True, slots=True)
class ProjectRow:
    id: ProjectId | None
    label: str
    tokens: TokenSums
    money: MoneySum
    session_count: int


@dataclass(frozen=True, slots=True)
class ModelRow:
    model: ModelIdentity
    tokens: TokenSums
    money: MoneySum
    harness: str = 'pi'


@dataclass(frozen=True, slots=True)
class QuantityRow:
    harness: str
    measure: str
    known: Decimal
    known_observations: int
    unknown_observations: int
    lower_bound_observations: int
    unresolved_observations: int = 0
    not_applicable_observations: int = 0


def quantity_rows(values: Iterable[tuple[str, MeasuredQuantity, str]]) -> tuple[QuantityRow, ...]:
    grouped: dict[tuple[str, str], list[tuple[MeasuredQuantity, str]]] = defaultdict(list)
    for harness, value, decision in values:
        if decision != 'excluded':
            grouped[harness, value.measure].append((value, decision))
    result = []
    for (harness, measure), members in sorted(grouped.items()):
        known = [value.amount for value, decision in members if decision == 'selected' and value.state == 'known' and value.amount is not None]
        unknown = sum(decision == 'selected' and value.state == 'unknown' for value, decision in members)
        result.append(QuantityRow(harness, measure, exact_sum(known), len(known), unknown,
                                  sum(decision == 'selected' and value.state == 'known' and value.lower_bound for value, decision in members),
                                  sum(decision == 'unresolved' for _, decision in members),
                                  sum(decision == 'selected' and value.state == 'not_applicable' for value, decision in members)))
    return tuple(result)


@dataclass(frozen=True, slots=True)
class Report:
    revision: int
    query: ReportQuery
    projects: tuple[ProjectRow, ...]
    sessions: tuple[SessionRow, ...]
    tokens: TokenSums
    money: MoneySum
    buckets: tuple[Bucket, ...]
    unbucketed: TokenSums
    coverage: tuple[Coverage, ...]
    models: tuple[ModelRow, ...] = ()
    total_session_count: int = 0
    session_page: int | None = None
    page_size: int = 50
    session_sort: SessionSort = 'started'
    quantities: tuple[QuantityRow, ...] = ()


@dataclass(frozen=True, slots=True)
class SessionFamily:
    id: SessionId
    name: str | None
    cwd: str | None
    attribution: Attribution
    started: datetime | None
    last_observed: datetime | None
    harness: str = 'pi'

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise ContractViolation('invalid_session_identity')
        if any(value is not None and not isinstance(value, str) for value in (self.name, self.cwd)):
            raise ContractViolation('invalid_session_metadata')
        if not isinstance(self.attribution, (Assigned, Unassigned)):
            raise ContractViolation('invalid_session_attribution')
        for field in ('started', 'last_observed'):
            value = getattr(self, field)
            if value is not None:
                object.__setattr__(self, field, instant(value))
        if self.started is not None and self.last_observed is not None and self.last_observed < self.started:
            raise ContractViolation('invalid_session_lifetime')


@dataclass(frozen=True, slots=True)
class SelectedContribution:
    observation_id: ObservationId
    session_id: SessionId
    name: str | None
    cwd: str | None
    model: ModelIdentity
    time: TimeEvidence
    tokens: TokenEvidence
    money: RecordedMoney
    attribution: Attribution
    decisions: tuple[tuple[str, str], ...]
    started: datetime | None = None
    last_observed: datetime | None = None
    diagnostics: tuple[str, ...] = ()
    family: SessionFamily | None = None
    quantities: tuple[MeasuredQuantity, ...] = ()
    quantity_decisions: tuple[tuple[str, str], ...] = ()
    harness: str = 'pi'
    saved_history: bool = False
    output_lower_bound: bool = False
    lower_bound_measures: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.family is not None and not isinstance(self.family, SessionFamily):
            raise ContractViolation('invalid_session_family')
        if (type(self.decisions) is not tuple or type(self.diagnostics) is not tuple
                or type(self.quantities) is not tuple or type(self.quantity_decisions) is not tuple):
            raise ContractViolation('immutable_contribution_required')
        seen: set[str] = set()
        for measure, state in self.decisions:
            if measure not in MEASURES or measure in seen or state not in ('selected', 'excluded', 'unresolved'):
                raise ContractViolation('invalid_report_decision')
            seen.add(measure)
        quantity_measures = {value.measure for value in self.quantities}
        if len(quantity_measures) != len(self.quantities):
            raise ContractViolation('invalid_report_quantity')
        seen.clear()
        for measure, state in self.quantity_decisions:
            if measure not in quantity_measures or measure in seen or state not in ('selected', 'excluded', 'unresolved'):
                raise ContractViolation('invalid_report_quantity_decision')
            seen.add(measure)
        for field in ('started', 'last_observed'):
            value = getattr(self, field)
            if value is not None:
                object.__setattr__(self, field, instant(value))


def _zone(timezone: str) -> ZoneInfo:
    try:
        return ZoneInfo(timezone)
    except (ZoneInfoNotFoundError, ValueError) as error:
        raise ValueError('Unknown IANA timezone') from error


def _parse_local(value: str, zone: ZoneInfo) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is not None:
        return instant(parsed)
    candidates = {
        candidate.astimezone(UTC)
        for fold in (0, 1)
        if (candidate := parsed.replace(tzinfo=zone, fold=fold)).astimezone(UTC).astimezone(zone).replace(tzinfo=None) == parsed
    }
    if len(candidates) != 1:
        raise ValueError('Local time is ambiguous or nonexistent; supply an explicit UTC offset')
    return candidates.pop()


def parse_range(start: str, end: str, timezone: str) -> DateRange:
    zone = _zone(timezone)
    return DateRange(_parse_local(start, zone), _parse_local(end, zone), timezone)


def format_mtok(value: int) -> str:
    if type(value) is not int or value < 0:
        raise ValueError('Token count must be a nonnegative integer')
    if value == 0:
        return '0'
    return '<0.01' if value < 10_000 else f'{Decimal(value) / 1_000_000:.2f}'


def project_label(project_id: ProjectId | None) -> str:
    if project_id is None or project_id == 'unassigned':
        return 'Unassigned'
    identity_path = str(project_id).split(':', 1)[-1]
    path = Path(identity_path)
    return (path.parent if path.name == '.git' else path).name or identity_path


def _project(row: SelectedContribution) -> ProjectId | None:
    attribution = (row.family or row).attribution
    return attribution.project_id if isinstance(attribution, Assigned) else None


def _session_id(row: SelectedContribution) -> SessionId:
    return row.family.id if row.family is not None else row.session_id


def _sums(rows: Sequence[SelectedContribution], prices: Mapping[ObservationId, ObservationCost] | None = None,
          snapshot_date: str | None = None) -> tuple[TokenSums, MoneySum]:
    counts = {measure: [0, 0, 0, 0] for measure in MEASURES[:-1]}
    amounts: list[Decimal] = []
    missing = 0
    for row in rows:
        for measure, state in row.decisions:
            if state == 'excluded':
                continue
            if measure == 'recorded_usd':
                if prices is not None:
                    continue
                if state == 'selected' and isinstance(row.money, RecordedEstimate):
                    amounts.append(row.money.amount)
                else:
                    missing += 1
                continue
            counters = counts[measure]
            value: TokenValue = row.tokens.total if measure == 'total' else getattr(row.tokens.buckets, measure)
            if state == 'unresolved':
                counters[1] += 1
            elif isinstance(value, Known):
                counters[0] += value.value
                counters[3] += (measure == 'output' and row.output_lower_bound) or measure in row.lower_bound_measures
            elif isinstance(value, NotApplicable):
                counters[2] += 1
            else:
                counters[1] += 1
    with localcontext() as context:
        if amounts:
            context.prec = max(len(value.as_tuple().digits) + abs(int(value.as_tuple().exponent)) + max(value.adjusted(), 0) for value in amounts) + len(str(len(amounts))) + 1
        amount = sum(amounts, Decimal(0))
    money = MoneySum(amount, missing)
    if prices is not None:
        amount, missing, calculations = aggregate([prices[row.observation_id] for row in rows])
        money = MoneySum(amount, missing, 'models_dev_usd', calculations, snapshot_date)
    return TokenSums(*(MetricSum(*counts[measure]) for measure in MEASURES[:-1])), money


def _coverage_facts(rows: Sequence[SelectedContribution]) -> list[tuple[str, ObservationId]]:
    codes: dict[str, set[ObservationId]] = defaultdict(set)
    for row in rows:
        if row.saved_history:
            codes['saved_history'].add(row.observation_id)
        for code in row.diagnostics:
            if code not in DIAGNOSTIC_CODES:
                continue
            codes[code].add(row.observation_id)
        if any(state == 'unresolved' for _, state in (*row.decisions, *row.quantity_decisions)):
            codes['unresolved_evidence'].add(row.observation_id)
        selected_values = [row.tokens.total if measure == 'total' else getattr(row.tokens.buckets, measure)
                           for measure, state in row.decisions if state == 'selected' and measure != 'recorded_usd']
        selected_quantities = [value for value in row.quantities if (value.measure, 'selected') in row.quantity_decisions]
        if any(isinstance(value, Known) for value in selected_values) or any(value.state == 'known' for value in selected_quantities):
            codes['_known_usage'].add(row.observation_id)
        if any(not isinstance(value, (Known, NotApplicable)) for value in selected_values) or any(value.state == 'unknown' for value in selected_quantities):
            codes['usage_partial'].add(row.observation_id)
        if any(isinstance(value, NotApplicable) for value in selected_values) or any(value.state == 'not_applicable' for value in selected_quantities):
            codes['not_applicable'].add(row.observation_id)
        if (any(value.lower_bound for value in selected_quantities)
                or row.output_lower_bound and ('output', 'selected') in row.decisions
                and isinstance(row.tokens.buckets.output, Known)
                or any(measure in row.lower_bound_measures and state == 'selected'
                       and isinstance(row.tokens.total if measure == 'total' else getattr(row.tokens.buckets, measure), Known)
                       for measure, state in row.decisions if measure != 'recorded_usd')):
            codes['lower_bound'].add(row.observation_id)
        if isinstance(row.time, (Interval, Undated)):
            codes['interval_only' if isinstance(row.time, Interval) else 'timing_unavailable'].add(row.observation_id)
        if row.model.model is None or row.model.provider is None:
            codes['unknown_model_identity'].add(row.observation_id)
        if not isinstance((row.family or row).attribution, Assigned):
            codes['unassigned'].add(row.observation_id)
    return [(code, identity) for code, ids in codes.items() for identity in ids]


def _coverage(rows: Sequence[SelectedContribution]) -> tuple[Coverage, ...]:
    return coverage_rows(_coverage_facts(rows))


def coverage_rows(facts: Iterable[tuple[str, ObservationId]]) -> tuple[Coverage, ...]:
    codes: dict[str, set[ObservationId]] = defaultdict(set)
    for code, identity in facts:
        if code in COVERAGE_TEXT or code == '_known_usage':
            codes[code].add(identity)
    known = codes.pop('_known_usage', set())
    if not known and 'usage_partial' in codes:
        codes['usage_unavailable'] = codes.pop('usage_partial')
    return tuple(Coverage(code, tuple(sorted(ids))) for code, ids in sorted(codes.items()))


def _axis(rows: Sequence[SelectedContribution], query: ReportQuery) -> tuple[tuple[datetime, datetime], ...]:
    points = [row.time.at for row in rows if isinstance(row.time, Point)]
    return time_axis(min(points) if points else None, max(points) if points else None, query)


def time_axis(first: datetime | None, last: datetime | None, query: ReportQuery) -> tuple[tuple[datetime, datetime], ...]:
    if first is None or last is None:
        return ()
    zone = _zone(query.range.timezone if isinstance(query.range, DateRange) else 'UTC')
    start = query.range.start if isinstance(query.range, DateRange) else first
    end = query.range.end if isinstance(query.range, DateRange) else last + timedelta(microseconds=1)
    local = start.astimezone(zone)
    if query.bucket == 'day':
        cursor = local.replace(hour=0, minute=0, second=0, microsecond=0).astimezone(UTC)
    elif query.bucket == 'hour':
        cursor = local.replace(minute=0, second=0, microsecond=0).astimezone(UTC)
    else:
        cursor = local.replace(minute=local.minute // 5 * 5, second=0, microsecond=0).astimezone(UTC)
    axis: list[tuple[datetime, datetime]] = []
    while cursor < end:
        if len(axis) >= 100_000:
            raise ValueError('Range is too large for this bucket size')
        following = ((cursor.astimezone(zone) + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0, fold=0).astimezone(UTC) if query.bucket == 'day'
                     else cursor + timedelta(minutes=60 if query.bucket == 'hour' else 5))
        axis.append((max(cursor, start) if isinstance(query.range, DateRange) else cursor,
                     min(following, end) if isinstance(query.range, DateRange) else following))
        cursor = following
    return tuple(axis)


def _buckets(rows: Sequence[SelectedContribution], axis: tuple[tuple[datetime, datetime], ...],
             prices: Mapping[ObservationId, ObservationCost] | None = None, snapshot_date: str | None = None) -> tuple[Bucket, ...]:
    # ponytail: dense axes cost visible sessions * buckets; use sparse rendering if long histories dominate.
    grouped: dict[int, list[SelectedContribution]] = defaultdict(list)
    starts = [start for start, _ in axis]
    for row in rows:
        if isinstance(row.time, Point):
            index = bisect_right(starts, row.time.at) - 1
            if 0 <= index < len(axis) and row.time.at < axis[index][1]:
                grouped[index].append(row)
    return tuple(Bucket(start, end, *_sums(grouped[index], prices, snapshot_date)) for index, (start, end) in enumerate(axis))


def build_report(
    contributions: Sequence[SelectedContribution], *, revision: int, query: ReportQuery,
    include_sessions: bool = True, session_page: int | None = None, page_size: int = 50,
    session_id: SessionId | None = None, session_sort: SessionSort = 'started', catalog: Catalog | None = None,
) -> Report:
    if session_sort not in ('started', 'cost_desc', 'cost_asc'):
        raise ValueError('Invalid session sort')
    if session_page is not None and (type(session_page) is not int or session_page < 1):
        raise ValueError('Session page must be a positive integer')
    if type(page_size) is not int or page_size < 1:
        raise ValueError('Session page size must be a positive integer')
    if (not include_sessions and (session_page is not None or session_id is not None)
            or session_page is not None and session_id is not None):
        raise ValueError('Choose one session projection')
    scoped = [row for row in contributions if query.project_id is None or _project(row) == query.project_id or (query.project_id == 'unassigned' and _project(row) is None)]
    first_by_session: dict[SessionId, datetime] = {}
    last_by_session: dict[SessionId, datetime] = {}
    for row in scoped:
        identity = _session_id(row)
        metadata = row.family or row
        if metadata.started is not None:
            first_by_session[identity] = min(first_by_session.get(identity, metadata.started), metadata.started)
        if metadata.last_observed is not None:
            last_by_session[identity] = max(last_by_session.get(identity, metadata.last_observed), metadata.last_observed)
    rows: list[SelectedContribution] = []
    unavailable: list[SelectedContribution] = []
    for row in scoped:
        if not any(state != 'excluded' for _, state in (*row.decisions, *row.quantity_decisions)):
            continue
        timing = row.time
        if isinstance(query.range, AllTime):
            rows.append(row)
        elif isinstance(timing, Point):
            if query.range.start <= timing.at < query.range.end:
                rows.append(row)
        elif isinstance(timing, Interval) and timing.start is not None:
            if query.range.start <= timing.start and timing.end <= query.range.end:
                rows.append(row)
            elif timing.start < query.range.end and timing.end > query.range.start:
                unavailable.append(row)
        else:
            unavailable.append(row)
    prices = {row.observation_id: catalog.price(row.model, row.tokens, row.decisions, single_response=isinstance(row.time, Point)) for row in rows} if catalog is not None else None
    snapshot_date = catalog.snapshot_date if catalog is not None else None

    def sums(members: Sequence[SelectedContribution]) -> tuple[TokenSums, MoneySum]:
        return _sums(members, prices, snapshot_date)

    axis = _axis(rows, query)
    by_session: dict[SessionId, list[SelectedContribution]] = defaultdict(list)
    by_project: dict[ProjectId | None, list[SelectedContribution]] = defaultdict(list)
    for row in rows:
        by_session[_session_id(row)].append(row)
        by_project[_project(row)].append(row)
    session_ids = sorted(by_session, key=lambda identity: (first_by_session.get(identity, datetime.max.replace(tzinfo=UTC)), identity)) if include_sessions else []
    if include_sessions and session_sort != 'started':
        def cost_key(identity: SessionId) -> tuple[bool, Decimal]:
            members = by_session[identity]
            money = sums(members)[1]
            known = (any(isinstance(row.money, RecordedEstimate) and ('recorded_usd', 'selected') in row.decisions for row in members) if catalog is None
                     else money.missing_observations == 0 or any(line.subtotal is not None for line in money.calculations))
            amount = money.known
            return not known, amount.copy_negate() if session_sort == 'cost_desc' else amount
        session_ids.sort(key=cost_key)
    if session_page is not None:
        session_page = min(session_page, max(1, (len(session_ids) + page_size - 1) // page_size))
        session_ids = session_ids[(session_page - 1) * page_size:session_page * page_size]
    elif session_id is not None:
        session_ids = [session_id] if session_id in by_session else []
    sessions: list[SessionRow] = []
    for identity in session_ids:
        members = by_session[identity]
        member = members[0].family or members[0]
        first, last = first_by_session.get(identity), last_by_session.get(identity)
        elapsed = ElapsedSpan(first, last, True) if first is not None and last is not None and first <= last else UnknownElapsed('missing_endpoints')
        name = member.name or ('Session ' + str(identity).partition(':')[2][:8])
        tokens, money = sums(members)
        session_models = tuple(sorted({item.model for item in members}, key=lambda m: (m.provider or '', m.model or '')))
        subagent_count = len({row.session_id for row in members if row.session_id != identity})
        sessions.append(SessionRow(identity, name, member.attribution, elapsed, tokens, money, _buckets(members, axis, prices, snapshot_date), _coverage(members), first, session_models, subagent_count, member.harness))
    projects: list[ProjectRow] = []
    for project_id, members in sorted(by_project.items(), key=lambda item: str(item[0] or '')):
        label = project_label(project_id)
        tokens, money = sums(members)
        session_count = len({_session_id(row) for row in members if any(state == 'selected' for _, state in (*row.decisions, *row.quantity_decisions))})
        projects.append(ProjectRow(project_id, label, tokens, money, session_count))
    by_model: dict[tuple[str, ModelIdentity], list[SelectedContribution]] = defaultdict(list)
    for row in rows:
        by_model[row.harness, row.model].append(row)
    models = tuple(ModelRow(model, *sums(members), harness) for (harness, model), members in sorted(by_model.items(), key=lambda item: (item[0][0], item[0][1].provider or '', item[0][1].model or '')))
    tokens, money = sums(rows)
    coverage = list(_coverage(rows + unavailable))
    if unavailable:
        coverage.append(Coverage('timing_unavailable', tuple(sorted({row.observation_id for row in unavailable}))))
    if {'copilot-vscode', 'copilot-cli'} <= {row.harness for row in rows + unavailable
                                           if any(code in OVERLAP_USAGE_CODES for code, _ in _coverage_facts([row]))}:
        coverage.append(Coverage('copilot_cross_source_join_unavailable', ()))
    merged: dict[str, set[ObservationId]] = defaultdict(set)
    for entry in coverage:
        merged[entry.code].update(entry.observation_ids)
    coverage = [Coverage(code, tuple(sorted(ids))) for code, ids in sorted(merged.items())]
    return Report(revision, query, tuple(projects), tuple(sessions), tokens, money, _buckets(rows, axis, prices, snapshot_date),
                  sums([row for row in rows if not isinstance(row.time, Point)])[0], tuple(coverage), models,
                  len(by_session), session_page, page_size, session_sort,
                  quantity_rows((row.harness, value, dict(row.quantity_decisions).get(value.measure, 'excluded'))
                                for row in rows for value in row.quantities))
