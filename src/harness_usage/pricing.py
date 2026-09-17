"""Exact API-equivalent estimates from a bundled models.dev pricing snapshot."""
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, localcontext
from functools import cache
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Literal, cast

from .domain import ContractViolation, Known, ModelIdentity, NotApplicable, TokenEvidence

Category = Literal['input', 'output', 'cache_read', 'cache_write']
CATEGORIES: tuple[Category, ...] = ('input', 'output', 'cache_read', 'cache_write')
ALIASES = {'openai-codex': 'openai'}
COPILOT_MODEL_ALIASES = {
    'claude-haiku-4.5': 'claude-haiku-4-5',
    'claude-opus-4.6': 'claude-opus-4-6',
    'claude-opus-4.7': 'claude-opus-4-7',
    'claude-opus-4.8': 'claude-opus-4-8',
    'claude-sonnet-4.5': 'claude-sonnet-4-5',
    'claude-sonnet-4.6': 'claude-sonnet-4-6',
}


def exact_sum(values: Sequence[Decimal]) -> Decimal:
    with localcontext() as context:
        if values:
            context.prec = max(len(value.as_tuple().digits) + abs(int(value.as_tuple().exponent)) + max(value.adjusted(), 0) for value in values) + len(str(len(values))) + 2
        return sum(values, Decimal(0))


def _subtotal(tokens: int, rate: Decimal) -> Decimal:
    with localcontext() as context:
        context.prec = len(str(tokens)) + len(rate.as_tuple().digits) + abs(int(rate.as_tuple().exponent)) + 10
        return Decimal(tokens) * rate / Decimal(1_000_000)


@dataclass(frozen=True, slots=True)
class CostLine:
    model: ModelIdentity
    category: Category
    tokens: int | None
    rate: Decimal | None
    subtotal: Decimal | None
    source_provider: str | None
    context_threshold: int | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.model, ModelIdentity) or self.category not in CATEGORIES:
            raise ContractViolation('invalid_cost_identity')
        if self.tokens is not None and (type(self.tokens) is not int or self.tokens < 0):
            raise ContractViolation('invalid_cost_tokens')
        if any(value is not None and (not isinstance(value, Decimal) or not value.is_finite() or value < 0) for value in (self.rate, self.subtotal)):
            raise ContractViolation('invalid_cost_amount')
        if self.subtotal is not None and (self.tokens is None or self.rate is None or self.reason is not None):
            raise ContractViolation('invalid_priced_cost_line')
        if self.subtotal is None and not self.reason:
            raise ContractViolation('unpriced_cost_reason_required')


@dataclass(frozen=True, slots=True)
class ObservationCost:
    known: Decimal
    unpriced: bool
    calculations: tuple[CostLine, ...]


@dataclass(frozen=True, slots=True)
class _Rates:
    base: tuple[Decimal | None, ...]
    tiers: tuple[tuple[int, tuple[Decimal | None, ...]], ...]
    invalid_tier: bool = False


def _object(value: object) -> dict[str, object]:
    return cast(dict[str, object], value) if isinstance(value, dict) else {}


def _rates(raw: Mapping[str, object]) -> tuple[Decimal | None, ...]:
    values = []
    for category in CATEGORIES:
        value = raw.get(category)
        decimal = Decimal(value) if type(value) is int else value
        values.append(decimal if isinstance(decimal, Decimal) and decimal.is_finite() and decimal >= 0 else None)
    return tuple(values)


@dataclass(frozen=True, slots=True)
class Catalog:
    models: Mapping[tuple[str, str], _Rates]
    snapshot_date: str
    sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, 'models', MappingProxyType(dict(self.models)))

    @classmethod
    def from_bytes(cls, data: bytes, *, snapshot_date: str, sha256: str) -> 'Catalog':
        raw = _object(json.loads(data, parse_float=Decimal, parse_constant=Decimal))
        models: dict[tuple[str, str], _Rates] = {}
        for provider, record in raw.items():
            for identity, model in _object(_object(record).get('models')).items():
                cost = _object(_object(model).get('cost'))
                tiers: list[tuple[int, tuple[Decimal | None, ...]]] = []
                invalid = False
                if 'tiers' in cost:
                    entries = cost['tiers']
                    if not isinstance(entries, list):
                        invalid = True
                    else:
                        for entry in entries:
                            fields = _object(entry)
                            tier = _object(fields.get('tier'))
                            size = tier.get('size')
                            if tier.get('type') != 'context' or type(size) is not int or size < 0:
                                invalid = True
                            else:
                                tiers.append((size, _rates(fields)))
                elif 'context_over_200k' in cost:
                    fields = _object(cost['context_over_200k'])
                    invalid = not bool(fields)
                    tiers.append((200_000, _rates(fields)))
                if len({size for size, _ in tiers}) != len(tiers):
                    invalid = True
                models[(provider, identity)] = _Rates(_rates(cost), tuple(sorted(tiers)), invalid)
        return cls(models, snapshot_date, sha256)

    def pricing_key(self, model: ModelIdentity) -> tuple[str, str] | None:
        """Resolve a price reference, never an assertion about actual provider routing."""
        provider = ALIASES.get(model.provider, model.provider) if model.provider is not None else None
        if model.model is None:
            return None
        if provider is not None and (provider, model.model) in self.models:
            return provider, model.model
        official = ('anthropic',) if provider == 'claude-bridge' else (
            ('openai', 'anthropic', 'google') if provider in (None, 'github-copilot') else ())
        matches = [(name, model.model) for name in official if (name, model.model) in self.models]
        if len(matches) == 1:
            return matches[0]
        if not matches and provider in (None, 'github-copilot'):
            canonical = COPILOT_MODEL_ALIASES.get(model.model)
            if canonical is not None and ('anthropic', canonical) in self.models:
                return 'anthropic', canonical
        return None

    def price(self, model: ModelIdentity, tokens: TokenEvidence,
              decisions: Sequence[tuple[str, str]], *, single_response: bool = True) -> ObservationCost:
        key = self.pricing_key(model)
        profile = self.models.get(key) if key is not None else None
        threshold = None
        context_reason = None
        if profile is not None and not profile.invalid_tier and profile.tiers:
            prompt = (tokens.buckets.input, tokens.buckets.cache_read, tokens.buckets.cache_write)
            if not single_response:
                context_reason = 'aggregate_context'
            elif any(not isinstance(value, (Known, NotApplicable)) for value in prompt):
                context_reason = 'unknown_context_size'
            else:
                size = sum(value.value for value in prompt if isinstance(value, Known))
                for boundary, _ in profile.tiers:
                    if size > boundary:
                        threshold = boundary
        selected = dict(decisions)
        counts = tuple(value.value if isinstance(value, Known) else 0 if isinstance(value, NotApplicable) else None for value in tokens.buckets.values)
        return self.price_group(model, counts, tuple(selected.get(category, 'excluded') for category in CATEGORIES),
                                context_threshold=threshold, context_reason=context_reason,
                                cache_write_1h=isinstance(tokens.cache_write_1h, Known) and tokens.cache_write_1h.value > 0)

    def price_group(self, model: ModelIdentity, counts: tuple[int | None, ...], states: tuple[str, ...], *,
                    context_threshold: int | None = None, context_reason: str | None = None,
                    cache_write_1h: bool = False) -> ObservationCost:
        """Price homogeneous categories using their original response tier, never the summed context.

        Counts and states follow CATEGORIES; selected NotApplicable is represented by zero.
        The caller weights ``unpriced`` by the number of observations in the group.
        """
        if len(counts) != len(CATEGORIES) or len(states) != len(CATEGORIES):
            raise ContractViolation('invalid_pricing_group')
        key = self.pricing_key(model)
        provider = key[0] if key is not None else None
        profile = self.models.get(key) if key is not None else None
        rates = profile.base if profile is not None else (None,) * 4
        profile_reason = None if profile is not None else 'model_not_in_catalog'
        if profile is not None:
            if profile.invalid_tier:
                profile_reason = 'unsupported_pricing_tier'
            elif context_threshold is not None:
                rates = dict(profile.tiers)[context_threshold]
        lines: list[CostLine] = []
        for category_index, (category, rate) in enumerate(zip(CATEGORIES, rates)):
            state = states[category_index]
            if state == 'excluded':
                continue
            count = counts[category_index] if state == 'selected' else None
            if state == 'selected' and count == 0:
                continue
            category_reason = profile_reason or context_reason
            if context_reason is not None and profile is not None:
                alternatives = {profile.base[category_index], *(tier_rates[category_index] for _, tier_rates in profile.tiers)}
                if len(alternatives) == 1 and None not in alternatives:
                    category_reason = None  # This category has one rate for every possible context.
            if category == 'cache_write' and cache_write_1h:
                category_reason = 'unsupported_cache_write_duration'
            reason = ('unresolved_tokens' if state != 'selected' else 'unknown_tokens') if count is None else category_reason or ('missing_category_rate' if rate is None else None)
            amount = _subtotal(count, rate) if reason is None and count is not None and rate is not None else None
            lines.append(CostLine(model, category, count, rate if category_reason is None else None, amount, provider if profile is not None else None, context_threshold, reason))
        return ObservationCost(exact_sum([line.subtotal for line in lines if line.subtotal is not None]), any(line.subtotal is None for line in lines), tuple(lines))


def aggregate(costs: Sequence[ObservationCost]) -> tuple[Decimal, int, tuple[CostLine, ...]]:
    grouped: dict[tuple[ModelIdentity, Category, Decimal | None, str | None, int | None, str | None], list[CostLine]] = {}
    for cost in costs:
        for line in cost.calculations:
            key = (line.model, line.category, line.rate, line.source_provider, line.context_threshold, line.reason)
            grouped.setdefault(key, []).append(line)
    lines = []
    for (model, category, rate, provider, threshold, reason), members in grouped.items():
        counts = [line.tokens for line in members]
        amounts = [line.subtotal for line in members]
        lines.append(CostLine(model, category, sum(cast(list[int], counts)) if all(value is not None for value in counts) else None,
                              rate, exact_sum(cast(list[Decimal], amounts)) if all(value is not None for value in amounts) else None,
                              provider, threshold, reason))
    lines.sort(key=lambda line: (line.model.provider or '', line.model.model or '', CATEGORIES.index(line.category), line.rate is None, line.rate or Decimal(0), line.context_threshold or 0, line.reason or ''))
    return exact_sum([cost.known for cost in costs]), sum(cost.unpriced for cost in costs), tuple(lines)


@cache
def load_bundled_catalog() -> Catalog:
    directory = Path(__file__).with_name('data')
    data = (directory / 'models-dev.json').read_bytes()
    metadata = json.loads((directory / 'models-dev.metadata.json').read_bytes())
    fingerprint = hashlib.sha256(data).hexdigest()
    if fingerprint != metadata['sha256']:
        raise ValueError('pricing_catalog_checksum_mismatch')
    return Catalog.from_bytes(data, snapshot_date=metadata['snapshot_date'], sha256=fingerprint)
