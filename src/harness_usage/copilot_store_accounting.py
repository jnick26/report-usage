"""Per-call index projection and non-additive reconciliation with native CLI controls."""
from collections import Counter, defaultdict
from decimal import Decimal
import json
from typing import TYPE_CHECKING, Any, cast

from .accounting import MEASURES
from .copilot_cli_accounting import CopilotCLICandidate, reconcile_copilot_cli
from .database import Connection, Row
from .domain import Known, Unknown, TokenValue, TokenBreakdown, TokenEvidence, Point, Undated, ModelIdentity, MissingEstimate, MeasuredQuantity
from .pi_reader import UsageRecord, EntryIdentity
from .pricing import exact_sum

if TYPE_CHECKING:
    from .copilot_store_reader import CopilotStoreCall

LOWER_BOUND = 'copilot_store_partial_index'
CONFLICT = 'copilot_store_conflict'
FIELDS = ('input_tokens', 'output_tokens', 'cache_read_tokens', 'cache_write_tokens', 'reasoning_tokens')
NATIVE_FIELDS = ('input', 'output', 'cache_read', 'cache_write', 'reasoning', 'requests')


def store_record(call: 'CopilotStoreCall | None', source: str, ordinal: int, profile: str) -> UsageRecord:
    def token(name: str) -> TokenValue:
        if call and name in call.invalid_fields:
            return Unknown('invalid_count')
        value = getattr(call, name) if call is not None else None
        return Known(value) if value is not None else Unknown('not_reported')

    fresh = token('input_tokens')
    if call is None or call.cache_read_tokens != 0 or call.cache_write_tokens != 0:
        fresh = Unknown('cache_inclusive_input')
    reasoning = token('reasoning_tokens')
    if call and call.reasoning_tokens is not None and call.output_tokens is not None and call.reasoning_tokens > call.output_tokens:
        reasoning = Unknown('invalid_count')
    tokens = TokenEvidence(TokenBreakdown(fresh, token('output_tokens'), token('cache_read_tokens'), token('cache_write_tokens')),
                           Unknown('not_reported'), reasoning)
    quantities = (
        MeasuredQuantity('request_count', 'known', Decimal(1), None, False, profile) if call else
        MeasuredQuantity('request_count', 'unknown', None, 'not_reported', False, None),
        MeasuredQuantity('nano_aiu', 'known', call.total_nano_aiu, None, False, profile) if call and call.total_nano_aiu is not None and 'total_nano_aiu' not in call.invalid_fields else
        MeasuredQuantity('nano_aiu', 'unknown', None, 'not_reported', False, None),
    )
    return UsageRecord(EntryIdentity('store:' + source + ':' + (str(call.row_id) if call else 'unavailable'), None, ordinal),
                       'assistant' if call else 'request_summary', Point(call.created_at) if call and call.created_at else Undated('not_reported'),
                       ModelIdentity(None, call.model if call else None), tokens, MissingEstimate('not_recorded'), None, None, (), quantities)


def _relation(rows: list[Row], native: dict[str, Any]) -> str:
    raw = [json.loads(row['counters_json']) for row in rows]
    if any(item.get('unavailable') for item in raw):
        return 'conflict'
    values = [sum(item[field] for item in raw) if all(item.get(field) is not None and field not in item.get('invalid_fields', ()) for item in raw) else None for field in FIELDS]
    values.append(len(raw))
    other = [native.get(field) for field in NATIVE_FIELDS]
    if any(value is None for value in (*values, *other)):
        return 'conflict'
    if values == other:
        return 'equal'
    return 'covered' if all(cast(int, a) <= cast(int, b) for a, b in zip(values, other)) else 'conflict'


def reconcile_store(db: Connection) -> None:
    """Overlay index evidence only after native event-chain decisions are settled."""
    rows = tuple(db.execute(
        'SELECT o.*,e.source_id,e.row_id,e.counters_json,e.compatibility,g.availability '
        'FROM copilot_store_evidence e JOIN observation o ON o.id=e.observation_id '
        'JOIN source_generation g ON g.id=e.source_id '
        'WHERE o.session_id IN (SELECT id FROM reconcile_sessions) AND NOT EXISTS '
        '(SELECT 1 FROM source_generation newer WHERE newer.locator=g.locator AND newer.generation>g.generation)'))
    if not rows:
        return
    native = tuple(db.execute(
        'SELECT o.*,e.source_id,e.line,e.event_id,e.parent_event_id,e.counter_epoch,e.session_start_us,'
        'e.source_kind,e.state AS evidence_state,e.counters_json,g.availability '
        'FROM copilot_cli_evidence e JOIN observation o ON o.id=e.observation_id '
        'JOIN source_generation g ON g.id=e.source_id WHERE o.session_id IN (SELECT id FROM reconcile_sessions) '
        'AND NOT EXISTS(SELECT 1 FROM source_generation newer WHERE newer.locator=g.locator AND newer.generation>g.generation)'))
    by_session: dict[str, list[Row]] = defaultdict(list)
    native_by_session: dict[str, list[Row]] = defaultdict(list)
    for row in rows:
        by_session[row['session_id']].append(row)
    for row in native:
        native_by_session[row['session_id']].append(row)

    quantity_keys = {(row[0], row[1]) for row in db.execute(
        'SELECT q.observation_id,q.measure FROM quantity_value q JOIN observation o ON o.id=q.observation_id '
        'WHERE o.session_id IN (SELECT id FROM reconcile_sessions)')}
    selected_output = {row[0] for row in db.execute(
        "SELECT observation_id FROM decision WHERE measure='output' AND state='selected'")}
    pending: dict[str, dict[tuple[str, str], tuple[object, ...]]] = {'decision': {}, 'quantity_decision': {}}

    def decide(items: list[Row], state: str, reason: str, *, measures: tuple[str, ...] = MEASURES,
               quantities: tuple[str, ...] = ('request_count', 'nano_aiu'), canonical: str | None = None) -> None:
        for row in items:
            owner = row['session_id'] if state == 'selected' else None
            for table, names in (('decision', measures), ('quantity_decision', quantities)):
                for measure in names:
                    if table == 'quantity_decision' and (row['id'], measure) not in quantity_keys:
                        continue
                    pending[table][row['id'], measure] = (row['id'], measure, state, owner, canonical if state == 'excluded' else None, reason, 'copilot-store-1')

    for sid, session_rows in by_session.items():
        # Available copies supersede saved unavailable copies. Old generations
        # are absent from this population and never resurrect deleted calls.
        live = any(row['availability'] == 'available' for row in session_rows)
        active = [row for row in session_rows if row['availability'] == 'available' or not live]
        copies: dict[str, list[Row]] = defaultdict(list)
        for row in active:
            copies[row['source_id']].append(row)
        decide(session_rows, 'excluded', 'copilot_store_superseded')
        chosen = copies[min(copies)]
        signatures = [Counter(row['compatibility'] for row in copy) for copy in copies.values()]
        cli = native_by_session.get(sid, [])
        if any(row['availability'] == 'available' for row in cli):
            cli = [row for row in cli if row['availability'] == 'available']
        if any(signature != signatures[0] for signature in signatures[1:]):
            decide(active, 'unresolved', CONFLICT)
            decide(cli, 'unresolved', CONFLICT)
            continue
        # A canonical representative links excluded copies to the chosen scope;
        # retained multisets prove multiplicity, not the rebuild-local row IDs.
        for copy in copies.values():
            if copy is not chosen:
                decide(copy, 'excluded', 'copilot_store_duplicate_copy', canonical=chosen[0]['id'])
        raw_cli = [(row, json.loads(row['counters_json'])) for row in cli if row['counters_json']]
        controls = [(row, raw) for row, raw in raw_cli if raw.get('scope') == 'session']
        model_cli = [(row, raw) for row, raw in raw_cli if raw.get('scope') == 'model']
        if not controls and not model_cli:
            # Metadata-only native history cannot supply or contradict counts.
            decide(cli, 'excluded', 'copilot_store_usage_available', canonical=chosen[0]['id'])
            # Projection already keeps invalid fields Unknown. Valid independent
            # counters remain usable lower bounds, not whole-row conflicts.
            decide(chosen, 'selected', LOWER_BOUND)
            continue
        shutdowns = [(row, raw) for row, raw in controls if row['source_kind'] == 'shutdown']
        candidates = tuple(CopilotCLICandidate(
            row['id'], sid, row['event_id'], row['line'], ('session', None), 0,
            row['evidence_state'] == 'usable' and not raw.get('invalid'), CONFLICT,
            row['source_id'], row['parent_event_id'], row['counter_epoch'], row['counters_json'])
            for row, raw in shutdowns)
        selected = {decision.observation_id for decision in reconcile_copilot_cli(candidates) if decision.state == 'selected'}
        final = list({row['id']: (row, raw) for row, raw in shutdowns if row['id'] in selected}.values())
        epochs = {row['counter_epoch'] for row in cli if row['counter_epoch'] is not None}
        # Bounds corroborate a native epoch, never establish it by themselves.
        qualified = len(epochs) == 1 and len(final) == 1
        control, control_raw = final[0] if final else (None, {})
        if control is not None:
            start, end = control['session_start_us'], control['at_us']
            qualified = qualified and start is not None and end is not None and all(
                row['at_us'] is not None and start <= row['at_us'] <= end for row in chosen)
        if not qualified:
            decide(chosen, 'unresolved', CONFLICT)
            decide(cli, 'unresolved', CONFLICT)
            continue
        assert control is not None
        by_model: dict[str | None, list[Row]] = defaultdict(list)
        for row in chosen:
            by_model[row['model']].append(row)
        for model, calls in by_model.items():
            matches = [(row, raw) for row, raw in model_cli if row['event_id'] == control['event_id'] and row['model'] == model]
            # Equal event copies can reference the same observation; native
            # reconciliation has already qualified their compatibility.
            selected_matches = [(row, raw) for row, raw in matches if row['id'] in selected_output]
            relation = _relation(calls, selected_matches[0][1]) if selected_matches and model in control_raw.get('models', []) else 'conflict'
            if relation == 'equal':
                decide(calls, 'selected', 'copilot_store_exact_detail', quantities=('request_count',))
                decide([row for row, _ in matches], 'excluded', 'copilot_store_duplicate_control', quantities=('request_count',), canonical=calls[0]['id'])
            elif relation == 'covered':
                decide(calls, 'excluded', 'copilot_store_covered', quantities=('request_count',), canonical=selected_matches[0][0]['id'])
            else:
                decide(calls, 'unresolved', CONFLICT, quantities=('request_count',))
                decide([row for row, _ in model_cli if row['model'] == model], 'unresolved', CONFLICT, quantities=('request_count',))
        # Nano-AIU is session-wide; model detail and agent breakdowns must not
        # create another session total. Premium requests have no DB equivalent.
        nano_rows = list(db.execute(
            "SELECT o.*,q.amount_decimal,q.state AS quantity_state FROM observation o JOIN quantity_value q ON q.observation_id=o.id "
            "JOIN quantity_decision d ON d.observation_id=o.id AND d.measure=q.measure "
            "JOIN copilot_cli_evidence e ON e.observation_id=o.id WHERE o.session_id=? AND q.measure='nano_aiu' AND d.state='selected'",
            (sid,)))
        raw_amounts = [json.loads(row['counters_json']) for row in chosen]
        amounts = [raw.get('total_nano_aiu') if 'total_nano_aiu' not in raw.get('invalid_fields', ()) else None for raw in raw_amounts]
        # Token-vector conflicts do not invalidate independently qualified native
        # nano-AIU; the shared identity, epoch and time gates above still apply.
        if any(value is None for value in amounts):
            decide(chosen, 'unresolved', CONFLICT, measures=(), quantities=('nano_aiu',))
            decide(nano_rows, 'unresolved', CONFLICT, measures=(), quantities=('nano_aiu',))
        elif not nano_rows or any(row['quantity_state'] != 'known' for row in nano_rows):
            decide(chosen, 'selected', LOWER_BOUND, measures=(), quantities=('nano_aiu',))
            decide(nano_rows, 'excluded', 'copilot_store_usage_available', measures=(), quantities=('nano_aiu',), canonical=chosen[0]['id'])
        else:
            amount = exact_sum([Decimal(cast(str, value)) for value in amounts])
            native_amount = Decimal(nano_rows[0]['amount_decimal'])
            state = 'selected' if amount == native_amount else 'excluded' if amount < native_amount else 'unresolved'
            decide(chosen, state, CONFLICT if state == 'unresolved' else 'copilot_store_exact_detail' if state == 'selected' else 'copilot_store_covered', measures=(), quantities=('nano_aiu',), canonical=nano_rows[0]['id'])
            if state != 'excluded':
                decide(nano_rows, 'excluded' if state == 'selected' else 'unresolved', 'copilot_store_duplicate_control' if state == 'selected' else CONFLICT, measures=(), quantities=('nano_aiu',), canonical=chosen[0]['id'])
    for table, changes in pending.items():
        for measure in {key[1] for key in changes}:
            identities = [identity for identity, name in changes if name == measure]
            db.execute(f'DELETE FROM {table} WHERE measure=? AND observation_id IN (SELECT UNNEST(?))', (measure, identities))
        db.executemany(f'INSERT INTO {table} VALUES(?,?,?,?,?,?,?)',
                       (row for row in changes.values() if row[2] != 'excluded' or row[4] is not None))
