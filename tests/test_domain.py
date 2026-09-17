from datetime import datetime, UTC
from decimal import Decimal

import pytest


def test_invalid_counts_cannot_enter_accounting():
    from harness_usage.domain import Known, ContractViolation
    for value in (-1, True, 1.5, '1', 2**63):
        with pytest.raises(ContractViolation):
            Known(value)
    assert Known(0).value == 0


def test_total_preserves_partial_and_conflicting_evidence():
    from harness_usage.domain import Known, Unknown, TokenBreakdown, TokenEvidence
    partial = TokenEvidence(TokenBreakdown(Known(100), Known(20), Known(0), Unknown('missing')), Known(120))
    assert partial.total == Known(120)
    complete = TokenEvidence(TokenBreakdown(Known(100), Known(20), Known(800), Known(50)), Known(970), Known(5))
    assert complete.total == Known(970)
    conflict = TokenEvidence(complete.buckets, Known(971))
    assert isinstance(conflict.total, Unknown)


def test_subsets_and_money_are_valid_by_construction():
    from harness_usage.domain import Known, TokenBreakdown, TokenEvidence, RecordedEstimate, ContractViolation
    with pytest.raises(ContractViolation):
        TokenEvidence(TokenBreakdown(Known(0), Known(2), Known(0), Known(0)), Known(2), Known(3))
    for bad in (Decimal('-1'), Decimal('NaN'), Decimal('Infinity'), 0.1):
        with pytest.raises(ContractViolation):
            RecordedEstimate(bad, 'USD', (), 'source')
    assert RecordedEstimate(Decimal('0'), 'USD', (), 'source').amount == 0


def test_time_evidence_requires_aware_ordered_instants():
    from harness_usage.domain import Point, Interval, ContractViolation
    with pytest.raises(ContractViolation):
        Point(datetime(2026, 9, 12))
    now = datetime(2026, 9, 12, tzinfo=UTC)
    with pytest.raises(ContractViolation):
        Interval(now, now)
    assert Interval(None, now).start is None
