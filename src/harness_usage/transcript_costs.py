"""Pi branch costs joined only to the exact imported native source snapshot."""
from collections import defaultdict
from decimal import Decimal

from .domain import Known, ModelIdentity, NotApplicable, TokenBreakdown, TokenEvidence, TokenValue, Unknown
from .database import Row
from .pricing import CATEGORIES, Catalog, CostLine, exact_sum
from .source_input import SourcePayload
from .storage import PROFILE, Storage
from .transcript import Message, Transcript, TranscriptCostPoint


def pi_transcript_costs(storage: Storage, transcript: Transcript, locator: str,
                        payload: bytes, catalog: Catalog) -> tuple[TranscriptCostPoint, ...]:
    """Keep source qualification, decisions and token values in one read transaction."""
    messages = tuple(entry for entry in transcript.entries
                     if isinstance(entry, Message) and entry.role == 'assistant' and entry.usage_id is not None)
    if not messages or transcript.harness != 'pi':
        return ()
    amounts: dict[str, tuple[Decimal | None, bool, tuple[CostLine, ...]]] = {}
    with storage.connect() as db:
        source = db.execute('SELECT * FROM source_generation WHERE locator=? ORDER BY generation DESC LIMIT 1', (locator,)).fetchone()
        if (source is not None and source['session_id'] == transcript.session_id
                and source['profile'] == PROFILE and source['availability'] == 'available'
                and source['sha256'] == SourcePayload(locator, payload).fingerprint()):
            observations = tuple(db.execute(
                "SELECT DISTINCT o.id,o.native_entry_id,o.provider,o.model FROM appearance a "
                "JOIN observation o ON o.id=a.observation_id WHERE a.source_id=? AND o.session_id=? "
                "AND o.kind='assistant' AND o.native_entry_id IN (SELECT UNNEST(?))",
                (source['id'], transcript.session_id, [entry.usage_id for entry in messages])))
            identities = [row['id'] for row in observations]
            values: dict[str, dict[str, TokenValue]] = defaultdict(dict)
            for row in db.execute('SELECT * FROM token_value WHERE observation_id IN (SELECT UNNEST(?))', (identities,)):
                values[row['observation_id']][row['measure']] = (Known(row['amount']) if row['state'] == 'known' else
                    NotApplicable(row['reason']) if row['state'] == 'not_applicable' else Unknown(row['reason']))
            decisions: dict[str, list[tuple[str, str]]] = defaultdict(list)
            for row in db.execute('SELECT observation_id,measure,state FROM decision WHERE observation_id IN (SELECT UNNEST(?))', (identities,)):
                decisions[row['observation_id']].append((row['measure'], row['state']))
            by_entry: dict[str, list[Row]] = defaultdict(list)
            for row in observations:
                by_entry[row['native_entry_id']].append(row)
            for usage_id, matching in by_entry.items():
                if len(matching) != 1:
                    continue
                row = matching[0]
                fields = values[row['id']]
                tokens = TokenEvidence(TokenBreakdown(*(fields.get(name, Unknown('not_reported')) for name in CATEGORIES)),
                    fields.get('reported_total', Unknown('not_reported')), fields.get('reasoning', Unknown('not_reported')),
                    fields.get('cache_write_1h', Unknown('not_reported')))
                selected = decisions[row['id']]
                price = catalog.price(ModelIdentity(row['provider'], row['model']), tokens, selected)
                states = dict(selected)
                known_zero = all(states.get(name) == 'selected' and (isinstance(value, NotApplicable)
                                 or isinstance(value, Known) and value.value == 0)
                                 for name, value in zip(CATEGORIES, tokens.buckets.values))
                amount = price.known if known_zero or any(line.subtotal is not None for line in price.calculations) else None
                incomplete = amount is None or price.unpriced or any(states.get(name) != 'selected' for name in CATEGORIES)
                amounts[usage_id] = amount, incomplete, price.calculations
    points = []
    cumulative = Decimal(0)
    cumulative_incomplete = False
    for message in messages:
        amount, incomplete, calculations = amounts.get(message.usage_id or '', (None, True, ()))
        if amount is not None:
            cumulative = exact_sum((cumulative, amount))
        cumulative_incomplete |= incomplete
        points.append(TranscriptCostPoint(message.id, amount, cumulative, incomplete, cumulative_incomplete, calculations))
    return tuple(points)
