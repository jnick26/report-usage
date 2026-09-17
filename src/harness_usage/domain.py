"""Framework-free accounting values. Validation lives at construction."""
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Literal, NewType

SessionId = NewType('SessionId', str)
ObservationId = NewType('ObservationId', str)
ProjectId = NewType('ProjectId', str)


class ContractViolation(ValueError):
    pass


def instant(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ContractViolation('aware_instant_required')
    return value.astimezone(UTC)


@dataclass(frozen=True, slots=True)
class Known:
    value: int

    def __post_init__(self) -> None:
        if type(self.value) is not int or not 0 <= self.value < 2**63:
            raise ContractViolation('invalid_count')


@dataclass(frozen=True, slots=True)
class Unknown:
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.reason, str) or not self.reason:
            raise ContractViolation('reason_required')


@dataclass(frozen=True, slots=True)
class NotApplicable:
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.reason, str) or not self.reason:
            raise ContractViolation('reason_required')


TokenValue = Known | Unknown | NotApplicable

QuantityMeasure = Literal['ai_credits', 'nano_aiu', 'premium_requests', 'request_count']
QuantityState = Literal['known', 'unknown', 'not_applicable']


@dataclass(frozen=True, slots=True)
class MeasuredQuantity:
    measure: QuantityMeasure
    state: QuantityState
    amount: Decimal | None
    reason: str | None
    lower_bound: bool
    source_ref: str | None

    def __post_init__(self) -> None:
        if self.measure not in ('ai_credits', 'nano_aiu', 'premium_requests', 'request_count'):
            raise ContractViolation('invalid_quantity_measure')
        if self.state not in ('known', 'unknown', 'not_applicable'):
            raise ContractViolation('invalid_quantity_state')
        known = self.state == 'known'
        if (known and (not isinstance(self.amount, Decimal) or not self.amount.is_finite() or self.amount < 0
                       or self.reason is not None or not isinstance(self.source_ref, str) or not self.source_ref)
                or not known and (self.amount is not None or not isinstance(self.reason, str) or not self.reason
                                  or self.source_ref is not None or self.lower_bound)):
            raise ContractViolation('invalid_quantity_state')


@dataclass(frozen=True, slots=True)
class TokenBreakdown:
    input: TokenValue
    output: TokenValue
    cache_read: TokenValue
    cache_write: TokenValue

    def __post_init__(self) -> None:
        if not all(isinstance(v, (Known, Unknown, NotApplicable)) for v in self.values):
            raise ContractViolation('invalid_token_variant')

    @property
    def values(self) -> tuple[TokenValue, ...]:
        return self.input, self.output, self.cache_read, self.cache_write


@dataclass(frozen=True, slots=True)
class TokenEvidence:
    buckets: TokenBreakdown
    reported_total: TokenValue
    reasoning: TokenValue = Unknown('not_reported')
    cache_write_1h: TokenValue = Unknown('not_reported')

    def __post_init__(self) -> None:
        if not isinstance(self.buckets, TokenBreakdown):
            raise ContractViolation('invalid_breakdown')
        for v in (self.reported_total, self.reasoning, self.cache_write_1h):
            if not isinstance(v, (Known, Unknown, NotApplicable)):
                raise ContractViolation('invalid_token_variant')
        for sub, whole in ((self.reasoning, self.buckets.output), (self.cache_write_1h, self.buckets.cache_write)):
            if isinstance(sub, Known) and isinstance(whole, Known) and sub.value > whole.value:
                raise ContractViolation('invalid_subset')

    @property
    def total(self) -> TokenValue:
        values = self.buckets.values
        if any(isinstance(v, Unknown) and v.reason == 'invalid_count' for v in values):
            return Unknown('invalid_count')
        subtotal = sum(v.value for v in values if isinstance(v, Known))
        if subtotal >= 2**63:
            return Unknown('invalid_count')
        if isinstance(self.reported_total, Known):
            if self.reported_total.value < subtotal or (all(isinstance(v, Known) for v in values) and subtotal != self.reported_total.value):
                return Unknown('total_conflict')
            return self.reported_total
        if isinstance(self.reported_total, Unknown) and self.reported_total.reason == 'invalid_count':
            return self.reported_total
        return Known(subtotal) if all(isinstance(v, Known) for v in values) else Unknown('partial_total')


@dataclass(frozen=True, slots=True)
class Point:
    at: datetime
    basis: Literal['response_recorded_at'] = 'response_recorded_at'

    def __post_init__(self) -> None:
        object.__setattr__(self, 'at', instant(self.at))
        if self.basis != 'response_recorded_at':
            raise ContractViolation('invalid_time_basis')


@dataclass(frozen=True, slots=True)
class Interval:
    start: datetime | None
    end: datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, 'end', instant(self.end))
        if self.start is not None:
            object.__setattr__(self, 'start', instant(self.start))
            if self.start >= self.end:
                raise ContractViolation('invalid_interval')


@dataclass(frozen=True, slots=True)
class Undated:
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.reason, str) or not self.reason:
            raise ContractViolation('reason_required')


TimeEvidence = Point | Interval | Undated


@dataclass(frozen=True, slots=True)
class RecordedEstimate:
    amount: Decimal
    currency: Literal['USD']
    components: tuple[tuple[str, Decimal], ...]
    source_ref: str

    def __post_init__(self) -> None:
        if not isinstance(self.amount, Decimal) or not self.amount.is_finite() or self.amount < 0:
            raise ContractViolation('invalid_money')
        if self.currency != 'USD' or not isinstance(self.source_ref, str) or not self.source_ref or type(self.components) is not tuple:
            raise ContractViolation('invalid_money_basis')
        names: set[str] = set()
        for pair in self.components:
            if type(pair) is not tuple or len(pair) != 2:
                raise ContractViolation('invalid_money_component')
            name, value = pair
            if name not in ('input', 'output', 'cache_read', 'cache_write') or name in names or not isinstance(value, Decimal) or not value.is_finite() or value < 0:
                raise ContractViolation('invalid_money_component')
            names.add(name)


@dataclass(frozen=True, slots=True)
class MissingEstimate:
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.reason, str) or not self.reason:
            raise ContractViolation('reason_required')


RecordedMoney = RecordedEstimate | MissingEstimate


@dataclass(frozen=True, slots=True)
class ModelIdentity:
    provider: str | None
    model: str | None

    def __post_init__(self) -> None:
        for value in (self.provider, self.model):
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ContractViolation('invalid_model_identity')


@dataclass(frozen=True, slots=True)
class Assigned:
    project_id: ProjectId
    worktree: str
    basis: str

    def __post_init__(self) -> None:
        if not all(isinstance(value, str) and value for value in (self.project_id, self.worktree, self.basis)):
            raise ContractViolation('invalid_attribution')


@dataclass(frozen=True, slots=True)
class Unassigned:
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.reason, str) or not self.reason:
            raise ContractViolation('reason_required')


Attribution = Assigned | Unassigned
