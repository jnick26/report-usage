"""Check synthetic oracle arithmetic and schema constraints, not an importer."""
import json
import sqlite3
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).parent
REPO = ROOT.parents[2]


def fixture_oracles():
    manifest = json.loads((ROOT / 'cases.json').read_text())
    assert manifest['synthetic'] is True
    records = {}
    malformed = []
    for path in (ROOT / 'source').glob('*.jsonl'):
        for number, line in enumerate(path.read_text().splitlines(), 1):
            try:
                record = json.loads(line, parse_float=Decimal)
            except json.JSONDecodeError:
                malformed.append((path.name, number))
                continue
            assert not {'details', 'errorMessage', 'arguments'} & record.keys()
            if record['type'] == 'session':
                assert record['version'] == 3
                assert record['cwd'].startswith('/fixtures/')
            else:
                assert record.get('summary', '') == ''
                assert record.get('message', {}).get('content', []) == []
                records[path.name + ':' + record['id']] = record
    assert sorted(malformed) == [('malformed-middle.jsonl', 3), ('tail-before.jsonl', 3)]
    fields = {'input': 'input', 'output': 'output', 'cache_read': 'cacheRead',
              'cache_write': 'cacheWrite', 'total': 'totalTokens'}
    for case in manifest['cases']:
        for filename in case['files']:
            assert (ROOT / 'source' / filename).is_file()
        # Selection is hand-specified by the oracle; this checker never decides overlap.
        selected = [records[ref].get('message', records[ref])['usage']
                    for ref in case['selected']]
        for field, raw in fields.items():
            if case['expected'][field] is None:
                assert any(raw not in usage for usage in selected)
            else:
                assert sum(usage.get(raw, 0) for usage in selected) == case['expected'][field], (case['name'], field)
        amount = sum((Decimal(str(u['cost']['total'])) for u in selected), Decimal(0))
        assert amount == Decimal(case['expected']['recorded_usd']), case['name']
        for ref in case.get('unresolved', []) + case.get('excluded_duplicate', []):
            assert ref in records and ref not in case['selected']
    return len(manifest['cases'])


def schema_constraints(db=None):
    if db is None:
        db = sqlite3.connect(':memory:')
        db.executescript((REPO / 'docs/design/pi-ledger-schema.sql').read_text())
    db.execute("INSERT INTO session(id,harness,native_id,attribution_reason) VALUES('pi:s','pi','s','unknown')")
    db.execute("INSERT INTO observation(id,session_id,native_entry_id,fingerprint,kind,time_kind,at_us,time_reason,safe_facts_json) VALUES('o','pi:s','e','digest','assistant','point',1,'response_recorded_at','{}')")
    checks = 0

    def rejects(sql, params=()):
        nonlocal checks
        db.execute('SAVEPOINT forbidden')
        try:
            db.execute(sql, params)
        except sqlite3.IntegrityError:
            checks += 1
        else:
            raise AssertionError('Schema accepted forbidden combination: ' + sql)
        finally:
            db.execute('ROLLBACK TO forbidden')
            db.execute('RELEASE forbidden')

    rejects("INSERT INTO token_value VALUES('o','output','known',-1,NULL)")
    rejects("INSERT INTO token_value VALUES('o','output','unknown',0,'missing')")
    rejects("INSERT INTO token_value VALUES('o','output','not_applicable',0,'unsupported')")
    rejects("INSERT INTO token_value VALUES('o','output','known',NULL,NULL)")
    rejects("INSERT INTO token_value VALUES('missing','output','known',0,NULL)")
    rejects("INSERT INTO token_value VALUES('o','output','known',1.5,NULL)")
    db.execute("INSERT INTO token_value VALUES('o','output','known',0,NULL)")
    rejects("INSERT INTO token_value VALUES('o','output','known',0,NULL)")
    rejects("INSERT INTO recorded_estimate VALUES('o','missing','0','USD','{}','source','missing')")
    rejects("INSERT INTO recorded_estimate VALUES('o','known','0',NULL,'{}','source',NULL)")
    db.execute("INSERT INTO recorded_estimate VALUES('o','known','0','USD','{}','source',NULL)")
    rejects("INSERT INTO decision VALUES('o','output','selected',NULL,NULL,'rule','1')")
    rejects("INSERT INTO decision VALUES('o','output','unresolved','pi:s',NULL,'reason','1')")
    rejects("INSERT INTO decision VALUES('o','output','excluded',NULL,'o','rule','1')")
    rejects("INSERT INTO observation(id,session_id,native_entry_id,fingerprint,kind,time_kind,at_us,safe_facts_json) VALUES('bad','pi:s','bad','x','assistant','point',1,'{}')")
    rejects("INSERT INTO observation(id,session_id,native_entry_id,fingerprint,kind,time_kind,start_us,end_us,safe_facts_json) VALUES('bad','pi:s','bad','x','compaction','interval',5,4,'{}')")
    db.execute("INSERT INTO import_run VALUES('run','running',1,NULL,0,0,NULL)")
    rejects("INSERT INTO import_run VALUES('run2','running',1,NULL,0,0,NULL)")
    rejects("INSERT INTO import_run VALUES('run3','failed',1,2,0,0,NULL)")
    # A failed transaction cannot move the revision ahead of evidence.
    before = db.execute('SELECT revision FROM ledger_meta').fetchone()[0]
    db.execute('SAVEPOINT rollback_example')
    db.execute('UPDATE ledger_meta SET revision=revision+1')
    db.execute('ROLLBACK TO rollback_example')
    db.execute('RELEASE rollback_example')
    assert db.execute('SELECT revision FROM ledger_meta').fetchone()[0] == before
    assert db.execute('PRAGMA foreign_key_check').fetchall() == []
    db.close()
    return checks


if __name__ == '__main__':
    print(f'{fixture_oracles()} fixture oracles checked; {schema_constraints()} forbidden SQL combinations rejected.')
    print('No importer, domain implementation, or real-corpus reconciliation was exercised.')
