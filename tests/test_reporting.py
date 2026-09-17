from dataclasses import replace
from collections.abc import Sequence
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from harness_usage.domain import (
    Assigned, Interval, Known, MissingEstimate, ModelIdentity, ObservationId,
    Point, ProjectId, RecordedEstimate, SessionId, TokenBreakdown, TokenEvidence,
    Unassigned, Undated, Unknown,
)
from harness_usage.reporting import (
    AllTime, Bucket, DateRange, ElapsedSpan, ReportQuery, SelectedContribution,
    build_report, format_mtok, parse_range,
)


def at(hour: int) -> datetime:
    return datetime(2026, 9, 12, hour, tzinfo=UTC)


def contribution(identity: str, hour: int = 12) -> SelectedContribution:
    return SelectedContribution(
        ObservationId(identity), SessionId('pi:s'), 'Saved title', '/repo',
        ModelIdentity('provider', 'model'), Point(at(hour)),
        TokenEvidence(TokenBreakdown(Known(10), Known(20), Known(30), Known(40)), Known(100)),
        RecordedEstimate(Decimal('0.01'), 'USD', (), identity),
        Assigned(ProjectId('git:/repo/.git'), '/repo', 'git_common_dir'),
        tuple((name, 'selected') for name in ('input', 'output', 'cache_read', 'cache_write', 'total', 'recorded_usd')),
        started=at(1), last_observed=at(23),
    )


def test_quantity_only_family_counts_and_private_diagnostics_are_not_reported():
    from harness_usage.domain import MeasuredQuantity
    from harness_usage.reporting import SessionFamily
    owner = SessionFamily(SessionId('opaque-owner'), 'Owner', None,
                          Assigned(ProjectId('owner-project'), '/owner', 'fixture'), at(1), at(23), 'claude')
    row = replace(contribution('opaque-child'), harness='copilot-cli', family=owner,
                  decisions=(), diagnostics=('secret /private/token credential',),
                  quantities=(MeasuredQuantity('request_count', 'known', Decimal('0'), None, False, 'private'),),
                  quantity_decisions=(('request_count', 'selected'),))
    report = build_report([row], revision=1, query=ReportQuery(ProjectId('owner-project'), AllTime()))
    assert report.projects[0].session_count == 1
    assert report.sessions[0].harness == 'claude'
    assert report.quantities[0].harness == 'copilot-cli'
    assert not report.coverage
    assert not build_report([row], revision=1, query=ReportQuery(ProjectId('git:/repo/.git'), AllTime())).sessions


@pytest.mark.parametrize('state', ['known', 'unknown', 'unresolved', 'not_applicable', 'excluded', 'outside_project', 'outside_date'])
def test_cross_source_warning_requires_both_visible_populations(state):
    from harness_usage.domain import MeasuredQuantity
    first = replace(contribution('same'), harness='copilot-vscode', diagnostics=('copilot_cross_source_join_unavailable', 'saved_history'))
    value = MeasuredQuantity('request_count', 'unknown', None, 'private', False, None)
    second = replace(contribution('same'), session_id=SessionId('opaque-cli'), harness='copilot-cli', decisions=(),
                     quantities=(value,), quantity_decisions=(('request_count', 'selected'),))
    if state == 'known':
        second = replace(second, quantities=(MeasuredQuantity('request_count', 'known', Decimal('0'), None, False, 'private'),))
    if state == 'not_applicable':
        second = replace(second, quantities=(MeasuredQuantity('request_count', 'not_applicable', None, 'private', False, None),))
    if state in {'unresolved', 'excluded'}:
        second = replace(second, quantity_decisions=(('request_count', state),))
    if state == 'outside_project':
        second = replace(second, attribution=Assigned(ProjectId('elsewhere'), '/elsewhere', 'fixture'))
    if state == 'outside_date':
        second = replace(second, time=Point(at(14)))
    report = build_report([first, second], revision=1, query=ReportQuery(ProjectId('git:/repo/.git'), DateRange(at(12), at(14), 'UTC')))
    codes = [entry.code for entry in report.coverage]
    assert 'saved_history' not in codes
    assert codes.count('copilot_cross_source_join_unavailable') == (1 if state in {'known', 'unknown', 'unresolved'} else 0)
    assert report.tokens.total.known == 100
    assert len(report.sessions) == (2 if state in {'known', 'unknown', 'unresolved', 'not_applicable'} else 1)


def test_range_rejects_ambiguous_nonexistent_and_reversed_times() -> None:
    for start, end in [('2026-03-29T03:30', '2026-03-29T05:00'), ('2026-10-25T03:30', '2026-10-25T05:00')]:
        with pytest.raises(ValueError):
            parse_range(start, end, 'Europe/Kyiv')
    with pytest.raises(ValueError):
        parse_range('2026-09-12T13:00', '2026-09-12T12:00', 'UTC')
    with pytest.raises(ValueError):
        parse_range('2026-09-12T12:00', '2026-09-12T13:00', 'Unknown/Zone')
    disambiguated = parse_range('2026-10-25T03:30+03:00', '2026-10-25T03:30+02:00', 'Europe/Kyiv')
    assert (disambiguated.end - disambiguated.start).total_seconds() == 3600
    day = parse_range('2026-03-29T00:00', '2026-03-30T00:00', 'Europe/Kyiv')
    assert (day.end - day.start).total_seconds() == 23 * 3600


def test_report_uses_one_population_and_lifetime_elapsed() -> None:
    data = [contribution('before', 11), contribution('inside', 12), contribution('end', 13)]
    report = build_report(data, revision=7, query=ReportQuery(None, DateRange(at(12), at(13), 'UTC'), 'hour'))
    assert report.revision == 7
    assert report.tokens.total.known == 100
    assert report.money.known == Decimal('0.01')
    assert sum(row.tokens.total.known for row in report.projects) == 100
    assert sum(row.tokens.total.known for row in report.sessions) == 100
    assert sum(bucket.tokens.total.known for bucket in report.buckets) == 100
    assert report.projects[0].session_count == 1
    assert report.sessions[0].name == 'Saved title'
    assert report.sessions[0].elapsed == ElapsedSpan(at(1), at(23), True)


def test_intervals_are_never_interpolated_and_unknowns_are_not_zero() -> None:
    point = contribution('point')
    interval = replace(contribution('interval'), time=Interval(at(12), at(13)))
    unknown_start = replace(contribution('start'), time=Interval(None, at(13)))
    undated = replace(contribution('undated'), time=Undated('missing_time'), attribution=Unassigned('missing_cwd'), session_id=SessionId('pi:u'))
    unknown = replace(contribution('unknown'), tokens=TokenEvidence(TokenBreakdown(Known(0), Unknown('missing'), Known(0), Known(0)), Unknown('missing')), money=MissingEstimate('missing'))
    query = ReportQuery(None, DateRange(at(12), at(14), 'UTC'), 'hour')
    report = build_report([point, interval, unknown_start, undated, unknown], revision=1, query=query)
    assert report.tokens.total.known == 200
    assert report.tokens.output.unknown_observations == 1
    assert report.money.missing_observations == 1
    assert report.unbucketed.total.known == 100
    assert sum(bucket.tokens.total.known for bucket in report.buckets) == 100
    assert {'interval_only', 'timing_unavailable'} <= {item.code for item in report.coverage}
    alltime = build_report([point, interval, unknown_start, undated], revision=1, query=ReportQuery(None, AllTime(), 'hour'))
    assert alltime.tokens.total.known == 400
    assert alltime.unbucketed.total.known == 300
    assert len(alltime.projects) == 2
    assert any(row.id is None and row.label == 'Unassigned' for row in alltime.projects)


def test_decisions_are_per_measure_and_preserve_unresolved_coverage() -> None:
    row = replace(contribution('partial'), decisions=(('input', 'selected'), ('total', 'unresolved'), ('recorded_usd', 'excluded')), diagnostics=('identity_conflict',))
    report = build_report([row], revision=0, query=ReportQuery(None, AllTime(), 'hour'))
    assert report.tokens.input.known == 10
    assert report.tokens.total.known == 0
    assert report.tokens.total.unknown_observations == 1
    assert report.money.known == 0
    assert {'identity_conflict'} <= {c.code for c in report.coverage}


def test_empty_and_interval_only_reports_have_no_synthetic_axis() -> None:
    query = ReportQuery(None, AllTime(), 'hour')
    empty = build_report([], revision=0, query=query)
    assert empty.sessions == empty.buckets == empty.projects == ()
    interval = build_report([replace(contribution('x'), time=Interval(None, at(13)))], revision=0, query=query)
    assert interval.buckets == ()
    assert interval.tokens.total.known == interval.unbucketed.total.known == 100


def test_mtok_never_displays_small_positive_as_zero() -> None:
    assert format_mtok(0) == '0'
    assert format_mtok(1) == '<0.01'
    assert format_mtok(9999) == '<0.01'
    assert format_mtok(10_000) == '0.01'
    assert format_mtok(1_230_000) == '1.23'


def test_model_groups_keep_unknown_identity_and_unassigned_filter() -> None:
    known = contribution('known')
    unknown = replace(contribution('unknown'), model=ModelIdentity(None, None), attribution=Unassigned('missing'))
    report = build_report([known, unknown], revision=1, query=ReportQuery(None, AllTime()))
    assert {row.model for row in report.models} == {ModelIdentity('provider', 'model'), ModelIdentity(None, None)}
    assert [row.tokens.total.known for row in report.models] == [100, 100]
    unassigned = build_report([known, unknown], revision=1, query=ReportQuery(ProjectId('unassigned'), AllTime()))
    assert unassigned.tokens.total.known == 100
    assert unassigned.projects[0].id is None


def test_money_aggregation_preserves_decimal_digits() -> None:
    first = replace(contribution('one'), money=RecordedEstimate(Decimal('0.12345678901234567890123456781'), 'USD', (), 'one'))
    second = replace(contribution('two'), money=RecordedEstimate(Decimal('0.00000000000000000000000000001'), 'USD', (), 'two'))
    report = build_report([first, second], revision=0, query=ReportQuery(None, AllTime()))
    assert report.money.known == Decimal('0.12345678901234567890123456782')


def test_calendar_day_buckets_handle_dst_and_share_session_axes() -> None:
    query = ReportQuery(None, parse_range('2026-03-28T00:00', '2026-03-31T00:00', 'Europe/Kyiv'), 'day')
    first = replace(contribution('first'), time=Point(datetime(2026, 3, 28, 12, tzinfo=UTC)))
    second = replace(contribution('second'), time=Point(datetime(2026, 3, 30, 12, tzinfo=UTC)), session_id=SessionId('pi:second'))
    report = build_report([first, second], revision=0, query=query)
    assert [(b.end - b.start).total_seconds() / 3600 for b in report.buckets] == [24, 23, 24]
    assert all([(b.start, b.end) for b in row.buckets] == [(b.start, b.end) for b in report.buckets] for row in report.sessions)


def test_partial_interval_overlap_is_qualified_without_contributing_amounts() -> None:
    overlap = replace(contribution('overlap'), time=Interval(at(11), at(13)))
    report = build_report([overlap], revision=0, query=ReportQuery(None, DateRange(at(12), at(14), 'UTC')))
    assert report.tokens.total.known == report.unbucketed.total.known == 0
    assert report.sessions == report.buckets == ()
    assert 'timing_unavailable' in {entry.code for entry in report.coverage}


def test_report_preserves_not_applicable_separately_from_unknown() -> None:
    from harness_usage.domain import NotApplicable

    row = replace(contribution('missing'), tokens=TokenEvidence(TokenBreakdown(NotApplicable('not_supported'), Unknown('not_recorded'), Known(0), Known(0)), Unknown('partial')))
    report = build_report([row], revision=0, query=ReportQuery(None, AllTime()))
    assert report.tokens.input.not_applicable_observations == 1
    assert report.tokens.input.unknown_observations == 0
    assert report.tokens.output.unknown_observations == 1
    assert report.tokens.output.not_applicable_observations == 0
    assert report.tokens.cache_read.known == report.tokens.cache_read.unknown_observations == 0


def test_midnight_dst_jump_keeps_following_buckets_at_local_midnight() -> None:
    query = ReportQuery(None, parse_range('2026-09-05T00:00', '2026-09-08T00:00', 'America/Santiago'), 'day')
    rows = [replace(contribution('first'), time=Point(datetime(2026, 9, 5, 12, tzinfo=UTC))), replace(contribution('last'), time=Point(datetime(2026, 9, 7, 12, tzinfo=UTC)))]
    report = build_report(rows, revision=0, query=query)
    assert [(bucket.end - bucket.start).total_seconds() / 3600 for bucket in report.buckets] == [24, 23, 24]


def test_session_rows_are_oldest_first_independent_of_session_ids() -> None:
    early = replace(contribution('early'), session_id=SessionId('pi:z'), started=at(1))
    late = replace(contribution('late'), session_id=SessionId('pi:a'), started=at(2))
    unknown = replace(contribution('unknown'), session_id=SessionId('pi:0'), started=None)
    report = build_report([late, unknown, early], revision=0, query=ReportQuery(None, AllTime()))
    assert [row.id for row in report.sessions] == ['pi:z', 'pi:a', 'pi:0']


def test_project_label_is_available_when_a_filter_has_no_contributions() -> None:
    from harness_usage.reporting import project_label
    assert project_label(ProjectId('git:/projects/example/.git')) == 'example'
    assert project_label(ProjectId('directory:/projects/notes')) == 'notes'
    assert project_label(ProjectId('unassigned')) == project_label(None) == 'Unassigned'


def test_session_pages_and_details_preserve_full_population_totals_and_axis() -> None:
    data = [
        replace(contribution('later', 15), session_id=SessionId('pi:a'), started=at(2)),
        replace(contribution('first', 12), session_id=SessionId('pi:b'), started=at(1)),
        replace(contribution('tied', 13), session_id=SessionId('pi:c'), started=at(1)),
        replace(contribution('unresolved', 14), session_id=SessionId('pi:d'), started=None,
                decisions=(('total', 'unresolved'),)),
    ]
    query = ReportQuery(None, DateRange(at(12), at(16), 'UTC'), 'hour')
    full = build_report(data, revision=7, query=query)
    first = build_report(data, revision=7, query=query, session_page=1, page_size=2)
    last = build_report(data, revision=7, query=query, session_page=99, page_size=2)
    overview = build_report(data, revision=7, query=query, include_sessions=False)
    detail = build_report(data, revision=7, query=query, session_id=SessionId('pi:a'))
    assert [row.id for row in first.sessions + last.sessions] == ['pi:b', 'pi:c', 'pi:a', 'pi:d']
    assert first.sessions + last.sessions == full.sessions
    assert (first.session_page, last.session_page, first.page_size) == (1, 2, 2)
    assert overview.sessions == ()
    assert detail.sessions == (full.sessions[2],)
    for report in (full, first, last, overview, detail):
        assert report.total_session_count == 4
        assert report.projects[0].session_count == 3  # Unresolved-only rows still occupy a page slot.
        assert report.tokens.total.known == 300
        assert report.tokens.total.unknown_observations == 1
        assert report.money.known == Decimal('0.03')
        assert (report.revision, report.projects, report.models, report.tokens, report.money,
                report.buckets, report.unbucketed, report.coverage) == (
            full.revision, full.projects, full.models, full.tokens, full.money,
            full.buckets, full.unbucketed, full.coverage)
        assert [(b.start, b.end) for b in report.buckets] == [(at(i), at(i + 1)) for i in range(12, 16)]
        for row in report.sessions:
            assert [(b.start, b.end) for b in row.buckets] == [(at(i), at(i + 1)) for i in range(12, 16)]
    assert detail.sessions[0].elapsed == ElapsedSpan(at(2), at(23), True)


def test_pagination_handles_empty_and_missing_sessions() -> None:
    query = ReportQuery(None, AllTime())
    empty = build_report([], revision=0, query=query, session_page=99)
    assert (empty.sessions, empty.total_session_count, empty.session_page) == ((), 0, 1)
    missing = build_report([contribution('one')], revision=0, query=query, session_id=SessionId('pi:missing'))
    assert missing.sessions == ()
    assert missing.total_session_count == 1
    assert missing.tokens.total.known == 100


@pytest.mark.parametrize('page', [0, -1, True])
def test_invalid_session_page_is_rejected(page: int) -> None:
    with pytest.raises(ValueError):
        build_report([], revision=0, query=ReportQuery(None, AllTime()), session_page=page)


@pytest.mark.parametrize('size', [0, -1, True])
def test_invalid_session_page_size_is_rejected(size: int) -> None:
    with pytest.raises(ValueError):
        build_report([], revision=0, query=ReportQuery(None, AllTime()), page_size=size)


def test_incompatible_session_projections_are_rejected() -> None:
    query = ReportQuery(None, AllTime())
    with pytest.raises(ValueError):
        build_report([], revision=0, query=query, include_sessions=False, session_page=1)
    with pytest.raises(ValueError):
        build_report([], revision=0, query=query, include_sessions=False, session_id=SessionId('pi:s'))
    with pytest.raises(ValueError):
        build_report([], revision=0, query=query, session_page=1, session_id=SessionId('pi:s'))


def test_projection_only_allocates_buckets_for_visible_sessions(monkeypatch: pytest.MonkeyPatch) -> None:
    from harness_usage import reporting

    data = [replace(contribution(str(i), 12 + i), session_id=SessionId(f'pi:{i}')) for i in range(4)]
    query = ReportQuery(None, AllTime(), 'hour')
    original = reporting._buckets
    allocated: list[int] = []

    def counted(rows: Sequence[SelectedContribution], axis: tuple[tuple[datetime, datetime], ...], prices=None, snapshot_date=None) -> tuple[Bucket, ...]:
        buckets = original(rows, axis, prices, snapshot_date)
        allocated.append(len(buckets))
        return buckets

    monkeypatch.setattr(reporting, '_buckets', counted)
    build_report(data, revision=0, query=query, include_sessions=False)
    assert allocated == [4]  # Only the aggregate axis.
    allocated.clear()
    build_report(data, revision=0, query=query, session_page=1, page_size=2)
    assert allocated == [4, 4, 4]  # Two visible rows and the aggregate axis.


def test_session_families_keep_accounting_and_group_before_project_filters_and_pages() -> None:
    from harness_usage.reporting import SessionFamily

    root_owner = Assigned(ProjectId('git:/main/.git'), '/main', 'git_common_dir')
    family = SessionFamily(SessionId('pi:root'), 'Main task', '/main', root_owner, at(1), at(23))
    worker_owner = Assigned(ProjectId('git:/worker/.git'), '/worker', 'git_common_dir')
    child = replace(contribution('child', 12), session_id=SessionId('pi:child'), attribution=worker_owner,
                    started=at(2), last_observed=at(15), family=family)
    nested = replace(contribution('nested', 13), session_id=SessionId('pi:nested'), attribution=Unassigned('missing_cwd'),
                     started=at(3), last_observed=at(15), family=family,
                     decisions=(('input', 'selected'), ('output', 'excluded'), ('cache_read', 'selected'),
                                ('cache_write', 'selected'), ('total', 'unresolved'), ('recorded_usd', 'selected')))
    independent = replace(contribution('independent', 15), session_id=SessionId('pi:independent'),
                          attribution=root_owner, started=at(4), diagnostics=('subagent_parent_unresolved',))
    data = [replace(contribution('root', 11), session_id=family.id, attribution=root_owner, family=family),
            child, nested, replace(child, observation_id=ObservationId('child-again'), time=Point(at(14))),
            replace(nested, observation_id=ObservationId('excluded-copy'), decisions=(('total', 'excluded'),)),
            independent]
    all_query = ReportQuery(None, AllTime(), 'hour')
    original = build_report([replace(row, family=None) for row in data], revision=3, query=all_query)
    grouped = build_report(data, revision=3, query=all_query)
    assert grouped.total_session_count == 2
    assert (grouped.tokens, grouped.money, grouped.buckets, grouped.unbucketed, grouped.models) == (
        original.tokens, original.money, original.buckets, original.unbucketed, original.models)
    assert grouped.tokens.total.known == 400 and grouped.tokens.total.unknown_observations == 1
    assert grouped.tokens.output.known == 80  # The nested output and duplicate remain excluded.
    assert grouped.money.known == Decimal('0.05')

    query = ReportQuery(root_owner.project_id, DateRange(at(12), at(16), 'UTC'), 'hour')
    first = build_report(data, revision=3, query=query, session_page=1, page_size=1)
    second = build_report(data, revision=3, query=query, session_page=2, page_size=1)
    assert first.total_session_count == second.total_session_count == 2
    assert first.projects[0].session_count == 2
    assert first.tokens == second.tokens and first.buckets == second.buckets
    assert first.tokens.total.known == 300 and first.tokens.total.unknown_observations == 1
    row = first.sessions[0]
    assert (row.id, row.name, row.attribution, row.started) == (family.id, 'Main task', root_owner, at(1))
    assert row.elapsed == ElapsedSpan(at(1), at(23), True)
    assert row.subagent_count == 2  # Distinct contributing child sessions, not observations.
    assert row.tokens.total.known == 200 and row.tokens.total.unknown_observations == 1
    assert row.money.known == Decimal('0.03')
    assert [(bucket.start, bucket.end) for bucket in row.buckets] == [(at(i), at(i + 1)) for i in range(12, 16)]
    assert second.sessions[0].id == independent.session_id and second.sessions[0].subagent_count == 0
    assert 'subagent_parent_unresolved' in {item.code for item in second.sessions[0].coverage}
    assert build_report(data, revision=3, query=query, session_id=family.id).sessions == (row,)
    assert build_report(data, revision=3, query=query, session_id=child.session_id).sessions == ()
    assert build_report(data, revision=3, query=replace(query, project_id=worker_owner.project_id)).sessions == ()


def test_unknown_family_start_is_not_replaced_by_a_child_start() -> None:
    from harness_usage.reporting import SessionFamily, UnknownElapsed

    family = SessionFamily(SessionId('pi:root'), None, None, Unassigned('missing_cwd'), None, at(23))
    child = replace(contribution('child'), family=family, started=at(2))
    report = build_report([child], revision=0, query=ReportQuery(ProjectId('unassigned'), AllTime()))
    row = report.sessions[0]
    assert row.id == family.id and row.name == 'Session root'
    assert row.started is None and isinstance(row.elapsed, UnknownElapsed)
    assert row.subagent_count == 1


def test_family_boundary_rejects_invalid_identity_metadata_and_lifetime() -> None:
    from harness_usage.domain import ContractViolation
    from harness_usage.reporting import SessionFamily

    valid = SessionFamily(SessionId('pi:root'), None, None, Unassigned('missing_cwd'), at(1), at(2))
    for changes in ({'id': ''}, {'name': []}, {'cwd': []}, {'attribution': {}}, {'started': at(3)}):
        with pytest.raises(ContractViolation):
            replace(valid, **changes)
    with pytest.raises(ContractViolation):
        replace(contribution('invalid-family'), family={})


def test_cost_sort_uses_family_total_before_paging_and_keeps_unpriced_last():
    from harness_usage.reporting import SessionFamily
    base = contribution('base')
    family = SessionFamily(SessionId('pi:family'), 'Main', base.cwd, base.attribution, at(1), at(23))
    rows = [replace(base, observation_id=ObservationId('root'), session_id=family.id, family=family, money=RecordedEstimate(Decimal('2'), 'USD', (), 'root')),
            replace(base, observation_id=ObservationId('child'), session_id=SessionId('pi:child'), family=family, money=RecordedEstimate(Decimal('8'), 'USD', (), 'child')),
            replace(base, observation_id=ObservationId('other'), session_id=SessionId('pi:other'), money=RecordedEstimate(Decimal('9'), 'USD', (), 'other')),
            replace(base, observation_id=ObservationId('zero'), session_id=SessionId('pi:zero'), money=RecordedEstimate(Decimal(0), 'USD', (), 'zero')),
            replace(base, observation_id=ObservationId('missing'), session_id=SessionId('pi:missing'), money=MissingEstimate('not_reported'))]
    query = ReportQuery(None, AllTime())
    descending = build_report(rows, revision=1, query=query, session_sort='cost_desc', session_page=1, page_size=1)
    assert [row.id for row in descending.sessions] == ['pi:family']
    ascending = build_report(rows, revision=1, query=query, session_sort='cost_asc')
    assert [row.id for row in ascending.sessions] == ['pi:zero', 'pi:other', 'pi:family', 'pi:missing']
    assert ascending.money == descending.money
    assert ascending.buckets == descending.buckets
    with pytest.raises(ValueError):
        build_report(rows, revision=1, query=query, session_sort='invalid')


def test_bucket_money_preserves_exact_per_measure_selection_and_missing_values() -> None:
    first = replace(contribution('bucket-first'), money=RecordedEstimate(Decimal('0.12345678901234567890123456781'), 'USD', (), 'first'))
    second = replace(contribution('bucket-second'), session_id=SessionId('pi:second'), money=RecordedEstimate(Decimal('0.00000000000000000000000000001'), 'USD', (), 'second'))
    excluded = replace(contribution('bucket-excluded'), decisions=tuple((name, 'excluded' if name == 'recorded_usd' else state) for name, state in contribution('base').decisions))
    unresolved = replace(contribution('bucket-unresolved', 13), decisions=tuple((name, 'unresolved' if name == 'recorded_usd' else state) for name, state in contribution('base').decisions))
    missing = replace(contribution('bucket-missing', 13), money=MissingEstimate('not_recorded'))
    interval = replace(contribution('bucket-interval'), time=Interval(at(12), at(14)))
    report = build_report([first, second, excluded, unresolved, missing, interval], revision=1,
                          query=ReportQuery(None, DateRange(at(12), at(15), 'UTC'), 'hour'))
    assert report.buckets[0].money is not None
    assert report.buckets[0].money.known == Decimal('0.12345678901234567890123456782')
    assert report.buckets[0].money.missing_observations == 0
    assert report.buckets[1].money is not None
    assert report.buckets[1].money.known == 0 and report.buckets[1].money.missing_observations == 2
    assert report.buckets[2].money is not None
    assert report.buckets[2].money.known == 0 and report.buckets[2].money.missing_observations == 0
    first_session = next(session for session in report.sessions if session.id == 'pi:s')
    assert first_session.buckets[0].money is not None
    assert first_session.buckets[0].money.known == first.money.amount
    assert first_session.buckets[0].tokens.total.known == 200
    assert report.money.known == Decimal('0.13345678901234567890123456782')


def test_manual_bucket_can_omit_money_for_backwards_compatibility() -> None:
    report = build_report([contribution('manual')], revision=0, query=ReportQuery(None, AllTime(), 'hour'))
    bucket = report.buckets[0]
    assert Bucket(bucket.start, bucket.end, bucket.tokens).money is None


def test_catalog_costs_conserve_family_pages_buckets_and_category_breakdowns() -> None:
    from harness_usage.pricing import Catalog
    from harness_usage.reporting import SessionFamily
    rates = Catalog.from_bytes(b'{"provider":{"models":{"model":{"cost":{"input":1,"output":2,"cache_read":0.1,"cache_write":0.2,"tiers":[{"tier":{"type":"context","size":100},"input":3,"output":4,"cache_read":0.3,"cache_write":0.6}]}}}}}',snapshot_date='2026-09-12',sha256='fixture')
    family = SessionFamily(SessionId('pi:main'),'Main','/repo',contribution('sample').attribution,at(1),at(23))
    first = replace(contribution('catalog-first'),family=family,money=RecordedEstimate(Decimal(999),'USD',(),'ignored'),decisions=tuple((name,'excluded' if name=='recorded_usd' else state) for name,state in contribution('sample').decisions))
    second = replace(contribution('catalog-second',13),family=family,session_id=SessionId('pi:child'),money=MissingEstimate('ignored'))
    unknown = replace(contribution('catalog-unknown',13),session_id=SessionId('pi:unknown'),model=ModelIdentity('provider','typo'),money=RecordedEstimate(Decimal(99999),'USD',(),'ignored'))
    query=ReportQuery(None,DateRange(at(12),at(14),'UTC'),'hour')
    report=build_report([first,second,unknown],revision=1,query=query,catalog=rates,session_page=1,page_size=1,session_sort='cost_desc')
    assert report.money.basis=='models_dev_usd' and report.money.snapshot_date=='2026-09-12'
    assert report.money.known==Decimal('0.000122') and report.money.missing_observations==1
    assert report.total_session_count==2 and report.sessions[0].id=='pi:main'
    assert report.sessions[0].money.known==report.money.known
    assert report.projects[0].money.known==report.money.known
    assert sum(bucket.money.known for bucket in report.buckets)==report.money.known
    assert {line.context_threshold for line in report.sessions[0].money.calculations}=={None}
    assert [(line.category,line.tokens,line.rate,line.subtotal) for line in report.sessions[0].money.calculations]==[
        ('input',20,Decimal(1),Decimal('0.00002')),('output',40,Decimal(2),Decimal('0.00008')),
        ('cache_read',60,Decimal('0.1'),Decimal('0.000006')),('cache_write',80,Decimal('0.2'),Decimal('0.000016'))]
    next_page=build_report([first,second,unknown],revision=1,query=query,catalog=rates,session_page=2,page_size=1,session_sort='cost_desc')
    assert next_page.sessions[0].id=='pi:unknown' and next_page.money==report.money and next_page.buckets==report.buckets


def test_catalog_interval_usage_does_not_assume_one_request_context() -> None:
    from harness_usage.pricing import Catalog
    rates=Catalog.from_bytes(b'{"provider":{"models":{"model":{"cost":{"input":1,"output":2,"cache_read":0.1,"cache_write":0.2,"tiers":[{"tier":{"type":"context","size":100},"input":3,"output":2,"cache_read":0.1,"cache_write":0.2}]}}}}}',snapshot_date='2026-09-12',sha256='fixture')
    interval=replace(contribution('aggregate-cost'),time=Interval(at(12),at(13)))
    report=build_report([interval],revision=0,query=ReportQuery(None,AllTime()),catalog=rates)
    assert report.money.known==Decimal('0.000051') and report.money.missing_observations==1
    assert report.buckets==()
    assert next(line.reason for line in report.money.calculations if line.category=='input')=='aggregate_context'


def test_mixed_harnesses_share_project_totals_but_keep_model_identity():
    pi = contribution('pi-event')
    codex = replace(contribution('codex-event'), session_id=SessionId('codex:12345678-rest'), name=None,
                    harness='codex')
    report = build_report([pi, codex], revision=1, query=ReportQuery(None, AllTime()))
    assert len(report.projects) == 1
    assert report.projects[0].session_count == 2
    assert report.tokens.total.known == 200
    assert [(row.harness, row.tokens.total.known) for row in report.models] == [('codex', 100), ('pi', 100)]
    assert next(row.name for row in report.sessions if row.id == codex.session_id) == 'Session 12345678'
