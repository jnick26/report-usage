"""Dashboard read model: reduce ledger evidence in DuckDB before crossing into Python."""
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import cast

from .domain import Assigned, Attribution, MeasuredQuantity, ModelIdentity, ObservationId, ProjectId, SessionId, Unassigned
from .pricing import Catalog, ObservationCost, aggregate
from .reporting import (DIAGNOSTIC_CODES, OVERLAP_USAGE_CODES, Coverage, Bucket, DateRange, ElapsedSpan, MetricSum, ModelRow, MoneySum,
                        ProjectRow, Report, ReportQuery, SessionRow, SessionSort, TokenSums,
                        UnknownElapsed, coverage_rows, project_label, quantity_rows, time_axis)
from .storage import Storage, VSCODE_OUTPUT_LOWER_BOUND, from_micros, micros, report_diagnostics

MEASURES = ('input', 'output', 'cache_read', 'cache_write', 'total')


@dataclass(frozen=True, slots=True)
class _Group:
    owner: str
    project: ProjectId | None
    harness: str
    model: ModelIdentity
    bucket: int | None
    tokens: TokenSums
    price: ObservationCost
    unpriced: int
    selected: bool


def _sums(groups: Sequence[_Group], catalog: Catalog) -> tuple[TokenSums, MoneySum]:
    tokens = TokenSums(*(MetricSum(sum(getattr(g.tokens, m).known for g in groups),
                                  sum(getattr(g.tokens, m).unknown_observations for g in groups),
                                  sum(getattr(g.tokens, m).not_applicable_observations for g in groups),
                                  sum(getattr(g.tokens, m).lower_bound_observations for g in groups)) for m in MEASURES))
    amount, _, lines = aggregate([g.price for g in groups])
    return tokens, MoneySum(amount, sum(g.unpriced for g in groups), 'models_dev_usd', lines, catalog.snapshot_date)


def build_aggregate_report(storage: Storage, query: ReportQuery, catalog: Catalog, *,
                           include_sessions: bool = True, session_page: int | None = None,
                           page_size: int = 50, session_id: SessionId | None = None,
                           session_sort: SessionSort = 'started') -> Report:
    if session_sort not in ('started', 'cost_desc', 'cost_asc'):
        raise ValueError('Invalid session sort')
    if session_page is not None and (type(session_page) is not int or session_page < 1):
        raise ValueError('Session page must be a positive integer')
    if type(page_size) is not int or page_size < 1:
        raise ValueError('Session page size must be a positive integer')
    if (not include_sessions and (session_page is not None or session_id is not None)
            or session_page is not None and session_id is not None):
        raise ValueError('Choose one session projection')
    with storage.connect() as db:
        if not db.in_transaction:
            db.execute('BEGIN')
        revision = db.execute('SELECT revision FROM ledger_meta').one()[0]
        sessions = {r['id']: r for r in db.execute('SELECT s.id,s.harness,COALESCE(t.title,s.display_name) AS display_name,s.title_excerpt,s.cwd,s.started_us,s.last_seen_us,s.project_id,s.worktree,s.attribution_reason,s.native_id,s.parent_locator FROM session_view s LEFT JOIN codex_title t ON t.session_id=s.id')}
        # Reuse the established ownership resolver; accounting and reporting share the authoritative ledger.
        attributions: dict[str, Attribution] = {sid: Assigned(ProjectId(r['project_id']), r['worktree'], r['attribution_reason']) if r['project_id'] is not None else Unassigned(r['attribution_reason']) for sid, r in sessions.items()}
        families, _ = storage._families(db, sessions, attributions)
        db.execute('CREATE TEMP TABLE owners(session_id TEXT PRIMARY KEY,owner_id TEXT,project_id TEXT)')
        owners = []
        for sid in sessions:
            owner = str(families[sid].id) if sid in families else sid
            owners.append((sid, owner, sessions[owner]['project_id']))
        db.executemany('INSERT INTO owners VALUES(?,?,?)', owners)
        predicates = ["(EXISTS(SELECT 1 FROM decision d WHERE d.observation_id=o.id AND d.state<>'excluded') OR EXISTS(SELECT 1 FROM quantity_decision d WHERE d.observation_id=o.id AND d.state<>'excluded'))"]
        parameters: list[str | int] = []
        if query.project_id == 'unassigned':
            predicates.append('f.project_id IS NULL')
        elif query.project_id is not None:
            predicates.append('f.project_id=?'); parameters.append(str(query.project_id))
        included = '1'
        if isinstance(query.range, DateRange):
            start, end = cast(int, micros(query.range.start)), cast(int, micros(query.range.end))
            # Integer bounds originate in validated DateRange values, never SQL text input.
            predicates.append(f"((o.time_kind='point' AND o.at_us>={start} AND o.at_us<{end}) OR (o.time_kind='interval' AND (o.start_us IS NULL OR (o.start_us<{end} AND o.end_us>{start}))) OR o.time_kind='undated')")
            included = f"(o.time_kind='point' OR (o.time_kind='interval' AND o.start_us>={start} AND o.end_us<={end}))"
        db.execute('CREATE TEMP TABLE base AS SELECT o.id,o.session_id,s.harness,f.owner_id,f.project_id,o.provider,o.model,o.time_kind,o.at_us,'
                   + f"COALESCE(json_extract_string(o.safe_facts_json,'$.\"usage.outputFinality\"')='{VSCODE_OUTPUT_LOWER_BOUND}',false) AS output_lower_bound,"
                   + f'COALESCE({included},0) AS included,'
                   + "(EXISTS(SELECT 1 FROM decision d WHERE d.observation_id=o.id AND d.state='selected') OR EXISTS(SELECT 1 FROM quantity_decision d WHERE d.observation_id=o.id AND d.state='selected')) AS selected "
                   + 'FROM observation o JOIN session_view s ON s.id=o.session_id JOIN owners f ON f.session_id=o.session_id WHERE ' + ' AND '.join(predicates), parameters)
        db.execute('CREATE UNIQUE INDEX base_id ON base(id)')
        diagnostics = report_diagnostics(db, 'base')
        model_conflicts = [identity for identity, codes in diagnostics.items() if 'codex_model_conflict' in codes]
        if model_conflicts:
            db.execute('UPDATE base SET provider=NULL,model=NULL WHERE id IN (' + ','.join('?' for _ in model_conflicts) + ')', model_conflicts)
        endpoints = db.execute("SELECT MIN(at_us),MAX(at_us) FROM base WHERE included AND time_kind='point'").one()
        axis = time_axis(from_micros(endpoints[0]), from_micros(endpoints[1]), query)
        db.execute('CREATE TEMP TABLE axis(start_us BIGINT PRIMARY KEY,end_us BIGINT,bucket BIGINT)')
        db.executemany('INSERT INTO axis VALUES(?,?,?)', ((micros(start), micros(end), i) for i, (start, end) in enumerate(axis)))
        db.execute('CREATE TEMP TABLE tiers(provider TEXT,model TEXT,boundary BIGINT)')
        tiers: list[tuple[str | None, str, int]] = []
        for recorded_provider, model_name in db.execute('SELECT DISTINCT provider,model FROM base'):
            key = catalog.pricing_key(ModelIdentity(recorded_provider, model_name))
            rates = catalog.models.get(key) if key is not None else None
            if rates is not None and not rates.invalid_tier:
                tiers.extend((recorded_provider, model_name, boundary) for boundary, _ in rates.tiers)
        db.executemany('INSERT INTO tiers VALUES(?,?,?)', tiers)
        columns: list[str] = []
        joins = []
        for index, measure in enumerate((*MEASURES[:4], 'reported_total', 'cache_write_1h')):
            alias = f'v{index}'
            joins.append(f"LEFT JOIN token_value {alias} ON {alias}.observation_id=b.id AND {alias}.measure='{measure}'")
            columns.extend(f'{alias}.{field} AS {measure}_{field}' for field in ('amount', 'state', 'reason'))
        for index, measure in enumerate(MEASURES):
            alias = f'd{index}'
            joins.append(f"LEFT JOIN decision {alias} ON {alias}.observation_id=b.id AND {alias}.measure='{measure}'")
            columns.append(f"COALESCE({alias}.state,'excluded') AS {measure}_decision")
        values_sql = ('SELECT b.*,' + ','.join(columns)
                      + ' FROM base b ' + ' '.join(joins) + ' WHERE b.included')
        subtotal = '+'.join(f'COALESCE(CAST({m}_amount AS HUGEINT),0)' for m in MEASURES[:4])
        all_known = ' AND '.join(f"{m}_state='known'" for m in MEASURES[:4])
        invalid = ' OR '.join(f"{m}_reason='invalid_count'" for m in MEASURES[:4])
        total = f"CASE WHEN ({invalid}) OR ({subtotal})>=9223372036854775808 THEN NULL WHEN reported_total_state='known' THEN CASE WHEN reported_total_amount<({subtotal}) OR (({all_known}) AND reported_total_amount<>({subtotal})) THEN NULL ELSE reported_total_amount END WHEN reported_total_reason='invalid_count' THEN NULL WHEN {all_known} THEN ({subtotal}) ELSE NULL END"
        prompt = '+'.join(f'COALESCE(CAST({m}_amount AS HUGEINT),0)' for m in ('input', 'cache_read', 'cache_write'))
        unknown_prompt = ' OR '.join(f"{m}_state NOT IN ('known','not_applicable')" for m in ('input', 'cache_read', 'cache_write'))
        matching = 't.provider IS NOT DISTINCT FROM v.provider AND t.model=v.model'
        classified_sql = ('SELECT v.*,' + total + ' AS total_amount,'
                   + f"CASE WHEN EXISTS(SELECT 1 FROM tiers t WHERE {matching}) THEN CASE WHEN time_kind<>'point' THEN 'aggregate_context' WHEN {unknown_prompt} THEN 'unknown_context_size' END END AS context_reason,"
                   + f"CASE WHEN time_kind='point' AND NOT ({unknown_prompt}) THEN (SELECT MAX(boundary) FROM tiers t WHERE {matching} AND boundary<({prompt})) END AS context_threshold,"
                   + "CASE WHEN cache_write_1h_state='known' AND cache_write_1h_amount>0 THEN 1 ELSE 0 END AS long_cache,"
                   + "(SELECT bucket FROM axis WHERE start_us<=v.at_us AND end_us>v.at_us ORDER BY start_us DESC LIMIT 1) AS bucket FROM values_by_observation v")
        dimensions = ['owner_id', 'project_id', 'harness', 'provider', 'model', 'bucket', 'context_reason', 'context_threshold', 'long_cache']
        group_by = ['owner_id', 'project_id', 'harness', 'provider', 'model', 'bucket', 'context_reason', 'context_threshold', 'long_cache']
        aggregates = ['COUNT(*) AS observations', 'MAX(selected) AS selected']
        for m in MEASURES:
            state = f'{m}_state' if m != 'total' else "CASE WHEN total_amount IS NULL THEN 'unknown' ELSE 'known' END"
            aggregates.extend((f"SUM(CAST(CASE WHEN {m}_decision='selected' AND ({state})='known' THEN {m}_amount ELSE 0 END AS BIGNUM)) AS {m}_known",
                               f"SUM({m}_decision='unresolved' OR ({m}_decision='selected' AND ({state})='unknown')) AS {m}_unknown",
                               f"SUM({m}_decision='selected' AND ({state})='not_applicable') AS {m}_na",
                               f"SUM({m}_decision='selected' AND ({state})='known' AND output_lower_bound) AS {m}_lower" if m == 'output' else f'0 AS {m}_lower'))
            if m != 'total':
                dimensions.extend((f'{m}_state', f'{m}_decision', f'COALESCE({m}_amount=0,0) AS {m}_zero'))
                group_by.extend((f'{m}_state', f'{m}_decision', f'{m}_zero'))
                aggregates.append(f'SUM(CAST({m}_amount AS BIGNUM)) AS {m}_price_count')
        # Ordinary CTEs let DuckDB combine joins, classification and aggregation
        # without writing and rereading per-observation intermediate tables.
        grouped_sql = ('WITH values_by_observation AS (' + values_sql + '), classified AS (' + classified_sql + ') '
                       + 'SELECT ' + ','.join((*dimensions, *aggregates)) + ' FROM classified GROUP BY ' + ','.join(group_by))
        grouped = db.execute(grouped_sql).fetchall()
        groups = []
        for row in grouped:
            model = ModelIdentity(row['provider'], row['model'])
            counts = tuple(int(row[f'{m}_price_count']) if row[f'{m}_state'] == 'known' else 0 if row[f'{m}_state'] == 'not_applicable' else None for m in MEASURES[:4])
            price = catalog.price_group(model, counts, tuple(row[f'{m}_decision'] for m in MEASURES[:4]), context_threshold=row['context_threshold'], context_reason=row['context_reason'], cache_write_1h=bool(row['long_cache']))
            tokens = TokenSums(*(MetricSum(int(row[f'{m}_known']), row[f'{m}_unknown'], row[f'{m}_na'], row[f'{m}_lower']) for m in MEASURES))
            groups.append(_Group(row['owner_id'], ProjectId(row['project_id']) if row['project_id'] is not None else None, row['harness'], model, row['bucket'], tokens, price, row['observations'] if price.unpriced else 0, bool(row['selected'])))
        quantities = quantity_rows((row['harness'],
            MeasuredQuantity(row['measure'], row['state'], Decimal(row['amount_decimal']) if row['amount_decimal'] is not None else None,
                             row['reason'], bool(row['lower_bound']), row['source_ref']), row['decision'])
            for row in db.execute('SELECT b.harness,q.*,d.state AS decision FROM base b JOIN quantity_value q ON q.observation_id=b.id JOIN quantity_decision d ON d.observation_id=b.id AND d.measure=q.measure WHERE b.included'))
        subagents: dict[str, int] = dict(db.execute('SELECT owner_id,COUNT(DISTINCT CASE WHEN session_id<>owner_id THEN session_id END) FROM base WHERE included GROUP BY owner_id'))
        # Transfer only scoped category/identity facts, never raw diagnostics or payloads.
        facts: dict[str, list[tuple[str, ObservationId]]] = defaultdict(list)
        all_facts: list[tuple[str, ObservationId]] = []
        conditions = {
            'unresolved_evidence': "EXISTS(SELECT 1 FROM decision d WHERE d.observation_id=b.id AND d.state='unresolved') OR EXISTS(SELECT 1 FROM quantity_decision d WHERE d.observation_id=b.id AND d.state='unresolved')",
            'interval_only': "b.time_kind='interval'",
            'timing_unavailable': "b.time_kind='undated' OR NOT b.included",
            'unknown_model_identity': 'b.provider IS NULL OR b.model IS NULL',
            'unassigned': 'b.project_id IS NULL',
            'lower_bound': "EXISTS(SELECT 1 FROM quantity_value q JOIN quantity_decision d ON d.observation_id=q.observation_id AND d.measure=q.measure WHERE q.observation_id=b.id AND d.state='selected' AND q.state='known' AND q.lower_bound) OR (b.output_lower_bound AND EXISTS(SELECT 1 FROM token_value v JOIN decision d ON d.observation_id=v.observation_id AND d.measure=v.measure WHERE v.observation_id=b.id AND v.measure='output' AND v.state='known' AND d.state='selected'))",
        }
        for state, code in (('known', '_known_usage'), ('unknown', 'usage_partial'), ('not_applicable', 'not_applicable')):
            conditions[code] = ("EXISTS(SELECT 1 FROM token_value v JOIN decision d ON d.observation_id=v.observation_id AND d.measure=v.measure "
                                f"WHERE v.observation_id=b.id AND d.state='selected' AND v.measure IN ('input','output','cache_read','cache_write') AND v.state='{state}') OR "
                                "EXISTS(SELECT 1 FROM quantity_value q JOIN quantity_decision d ON d.observation_id=q.observation_id AND d.measure=q.measure "
                                f"WHERE q.observation_id=b.id AND d.state='selected' AND q.state='{state}')")
        coverage_sql = ' UNION ALL '.join(f"SELECT b.id,b.owner_id,b.included,b.harness,'{code}' AS code FROM base b WHERE {predicate}" for code, predicate in conditions.items())
        # The normalized total can be unknown even when individual buckets are known.
        coverage_sql += (" UNION ALL SELECT id,owner_id,included,harness,CASE WHEN normalized_total IS NULL THEN 'usage_partial' ELSE '_known_usage' END "
                         + 'FROM (SELECT v.*,' + total + ' AS normalized_total FROM ('
                         + values_sql.removesuffix(' WHERE b.included') + ") v) totals WHERE total_decision='selected'")
        harnesses = set()
        for row in db.execute(coverage_sql):
            fact = (row['code'], ObservationId(row['id']))
            if row['code'] in OVERLAP_USAGE_CODES:
                harnesses.add(row['harness'])
            all_facts.append(fact)
            if row['included']:
                facts[row['owner_id']].append(fact)
        # Map known decision reasons without transferring arbitrary private reason strings.
        reason_sql = ('SELECT DISTINCT b.id,b.owner_id,b.included,d.reason FROM base b JOIN decision d ON d.observation_id=b.id '
                      + "WHERE d.state<>'selected' AND d.reason IN (" + ','.join('?' for _ in DIAGNOSTIC_CODES) + ')')
        for row in db.execute(reason_sql, DIAGNOSTIC_CODES):
            diagnostics.setdefault(row['id'], set()).add(row['reason'])
        for row in db.execute('SELECT id,owner_id,included FROM base WHERE id IN (' + ','.join('?' for _ in diagnostics) + ')', tuple(diagnostics)) if diagnostics else ():
            for code in diagnostics[row['id']]:
                fact = (code, ObservationId(row['id']))
                all_facts.append(fact)
                if row['included']:
                    facts[row['owner_id']].append(fact)
        coverage = coverage_rows(all_facts)
        if {'copilot-vscode', 'copilot-cli'} <= harnesses:
            coverage = tuple(sorted((*coverage, Coverage('copilot_cross_source_join_unavailable', ())), key=lambda value: value.code))
    by_owner: dict[str, list[_Group]] = defaultdict(list)
    by_project: dict[ProjectId | None, list[_Group]] = defaultdict(list)
    by_model: dict[tuple[str, ModelIdentity], list[_Group]] = defaultdict(list)
    for group in groups:
        by_owner[group.owner].append(group); by_project[group.project].append(group); by_model[group.harness, group.model].append(group)
    def buckets(members: Sequence[_Group]) -> tuple[Bucket, ...]:
        by_bucket: dict[int | None, list[_Group]] = defaultdict(list)
        for group in members:
            by_bucket[group.bucket].append(group)
        return tuple(Bucket(start, end, *_sums(by_bucket[i], catalog)) for i, (start, end) in enumerate(axis))
    def first(owner: str) -> datetime:
        return from_micros(sessions[owner]['started_us']) or datetime.max.replace(tzinfo=UTC)
    owner_ids = sorted(by_owner, key=lambda owner: (first(owner), owner)) if include_sessions else []
    owner_sums = {owner: _sums(members, catalog) for owner, members in by_owner.items()} if include_sessions else {}
    if session_sort != 'started':
        def cost_key(owner: str) -> tuple[bool, Decimal]:
            money = owner_sums[owner][1]
            known = money.missing_observations == 0 or any(line.subtotal is not None for line in money.calculations)
            return not known, money.known.copy_negate() if session_sort == 'cost_desc' else money.known
        owner_ids.sort(key=cost_key)
    if session_page is not None:
        session_page = min(session_page, max(1, (len(owner_ids) + page_size - 1) // page_size))
        owner_ids = owner_ids[(session_page - 1) * page_size:session_page * page_size]
    elif session_id is not None:
        owner_ids = [session_id] if session_id in by_owner else []
    session_rows = []
    for owner in owner_ids:
        row = sessions[owner]; family = families.get(owner)
        session_start = from_micros(row['started_us']); session_end = family.last_observed if family else from_micros(row['last_seen_us'])
        elapsed = ElapsedSpan(session_start, session_end, True) if session_start is not None and session_end is not None and session_start <= session_end else UnknownElapsed('missing_endpoints')
        session_rows.append(SessionRow(SessionId(owner), row['display_name'] or row['title_excerpt'] or 'Session ' + owner.partition(':')[2][:8], attributions[owner], elapsed,
                           *owner_sums[owner], buckets(by_owner[owner]), coverage_rows(facts[owner]), session_start,
                           tuple(sorted({g.model for g in by_owner[owner]}, key=lambda m: (m.provider or '', m.model or ''))), subagents[owner], row['harness']))
    projects = tuple(ProjectRow(project, project_label(project), *_sums(members, catalog), len({g.owner for g in members if g.selected})) for project, members in sorted(by_project.items(), key=lambda pair: str(pair[0] or '')))
    models = tuple(ModelRow(model, *_sums(members, catalog), harness) for (harness, model), members in sorted(by_model.items(), key=lambda pair: (pair[0][0], pair[0][1].provider or '', pair[0][1].model or '')))
    tokens, money = _sums(groups, catalog)
    return Report(revision, query, projects, tuple(session_rows), tokens, money, buckets(groups), _sums([g for g in groups if g.bucket is None], catalog)[0],
                  coverage, models, len(by_owner), session_page, page_size, session_sort, quantities)
