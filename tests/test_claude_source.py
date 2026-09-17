from dataclasses import replace
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest

from harness_usage.claude_reader import PROFILE, ClaudeReadBatch, read_claude
from harness_usage.claude_accounting import ClaudeCandidate, reconcile_claude
from harness_usage.claude_transcript import parse_claude_transcript
from harness_usage.domain import Known, MissingEstimate, Undated, Unknown
from harness_usage.pi_reader import RejectedSource
from harness_usage.reporting import AllTime, ReportQuery, build_report
from harness_usage.storage import Storage
from harness_usage.transcript import Message, Notice, ReasoningBlock, ToolBlock, TranscriptUnavailable
from harness_usage.transcript_access import read_transcript_page


FIXTURES = Path(__file__).parent / 'fixtures' / 'claude'
MAIN = '11111111-1111-4111-8111-111111111111'
BRANCH = '22222222-2222-4222-8222-222222222222'
THIRD = '33333333-3333-4333-8333-333333333333'
_FROZEN_RELATIVE = Path('.superpowers/sdd/2026-09-16-three-source-usage')


def _frozen_root(source_file: Path) -> Path:
    for ancestor in (source_file.parent, *source_file.parents):
        candidate = ancestor / _FROZEN_RELATIVE
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError('frozen_archive_root_missing')


def _versioned_source(*rows: tuple[object, object, str, str]) -> bytes:
    records = []
    for index, (version, output, message_id, request_id) in enumerate(rows, 1):
        usage: dict[str, object] = {'input_tokens': 3}
        if output is not ...:
            usage['output_tokens'] = output
        record: dict[str, object] = {
            'type': 'assistant', 'uuid': f'entry-{index}', 'parentUuid': None,
            'sessionId': MAIN, 'cwd': '/fixture/repo',
            'timestamp': f'2026-09-16T09:00:{index:02d}Z', 'requestId': request_id,
            'message': {'id': message_id, 'role': 'assistant', 'model': 'claude-test',
                        'usage': usage, 'content': [{'type': 'text', 'text': 'PRIVATE_CANARY'}]},
        }
        if version is not None:
            record['version'] = version
        records.append(record)
    return ''.join(json.dumps(record) + '\n' for record in records).encode()


def _frozen_import(snapshot: str, ledger: Path, sources: tuple[tuple[Path, bytes], ...]) -> None:
    try:
        archive = _frozen_root(Path(__file__))
    except FileNotFoundError:
        pytest.skip('historical upgrade test requires private frozen archive (not distributed)')
    snapshot_src = archive / snapshot / 'src'
    if not snapshot_src.is_dir():
        raise FileNotFoundError('frozen_snapshot_src_missing')
    for locator, data in sources:
        locator.parent.mkdir(parents=True, exist_ok=True)
        locator.write_bytes(data)
    script = (
        "import sys\n"
        "from pathlib import Path\n"
        "import harness_usage\n"
        "from harness_usage.storage import Storage\n"
        "expected=Path(sys.argv[2]).resolve()\n"
        "loaded=Path(harness_usage.__file__).resolve()\n"
        "assert loaded.is_relative_to(expected),'frozen_snapshot_import_mismatch'\n"
        "storage=Storage(Path(sys.argv[1]))\n"
        "for index in range(3,len(sys.argv),2):\n"
        " storage.import_source(sys.argv[index],Path(sys.argv[index+1]).read_bytes())\n"
        "storage.close()\n")
    arguments = [value for locator, _ in sources for value in (str(locator), str(locator))]
    environment = dict(os.environ)
    environment['PYTHONPATH'] = str(snapshot_src)
    environment['PYTHONDONTWRITEBYTECODE'] = '1'
    subprocess.run([sys.executable, '-B', '-c', script, str(ledger), str(snapshot_src), *arguments],
                   check=True, capture_output=True, text=True, env=environment)


def _identityless_source(data: bytes) -> bytes:
    record = json.loads(data)
    record.pop('requestId')
    record['message'].pop('id')
    return (json.dumps(record) + '\n').encode()


def _zero_usage_source() -> bytes:
    return (json.dumps({
        'type': 'user', 'uuid': 'trigger', 'parentUuid': None, 'sessionId': MAIN,
        'cwd': '/fixture/repo', 'timestamp': '2026-09-16T09:01:00Z',
        'message': {'role': 'user', 'content': 'trigger'},
    }) + '\n').encode()


def _insert_projection_marker(storage: Storage, locator: Path, data: bytes, *,
                              accepted: bool, profile: str = PROFILE) -> str:
    source_id = sha256(f'{locator}:{profile}:{accepted}'.encode()).hexdigest()
    with storage.connect(write=True) as db:
        generation = int(db.execute(
            'SELECT MAX(generation)+1 FROM source_generation WHERE locator=?',
            (str(locator),)).one()[0])
        db.execute(
            'INSERT INTO source_generation VALUES(?,?,?,?,?,?,?,?,?)',
            (source_id, str(locator), generation, sha256(data).hexdigest(),
             f'claude:{MAIN}' if accepted else None, profile,
             len(data) if accepted else 0, 0, 'available'))
        if accepted:
            db.execute('INSERT INTO source_metadata VALUES(?,?)',
                       (source_id, sha256(data).hexdigest()))
        else:
            db.execute('INSERT INTO diagnostic(source_id,code) VALUES(?,?)',
                       (source_id, 'conflicting_session_identity'))
    return source_id


@pytest.mark.parametrize(
    ('version', 'reason'),
    [('2.1.96', 'stream_start_placeholder'),
     ('2.1.97-rc.1', 'stream_start_placeholder'),
     (None, 'writer_version_unavailable'),
     (7, 'writer_version_unavailable'),
     (True, 'writer_version_unavailable'),
     ('', 'writer_version_unavailable'),
     ([], 'writer_version_unavailable'),
     ({}, 'writer_version_unavailable'),
     ('claude-export-import', 'writer_version_unavailable'),
     ('2.1.9', 'stream_start_placeholder'),
     ('02.1.97', 'writer_version_unavailable'),
     ('v2.1.97', 'writer_version_unavailable'),
     ('2.1.97-01', 'writer_version_unavailable'),
     (' 2.1.97', 'writer_version_unavailable')],
)
def test_claude_output_correction_unqualified_versions_stay_unknown(
        version: object, reason: str) -> None:
    read = read_claude(_versioned_source((version, 37, 'message', 'request')),
                       locator=f'/fixture/{MAIN}.jsonl')
    assert isinstance(read, ClaudeReadBatch)
    assert read.usage[0].tokens.buckets.output == Unknown(reason)


@pytest.mark.parametrize('version', ['2.1.97', '2.1.97+build.1', '2.1.98-rc.1',
                                     '2.1.98', '2.1.100', '2.1.273'])
def test_claude_output_correction_semver_releases_are_final(version: str) -> None:
    read = read_claude(_versioned_source((version, 37, 'message', 'request')),
                       locator=f'/fixture/{MAIN}.jsonl')
    assert isinstance(read, ClaudeReadBatch)
    assert read.usage[0].tokens.buckets.output == Known(37)


@pytest.mark.parametrize(('output', 'expected'), [(0, Known(0)), (2**63 - 1, Known(2**63 - 1)),
                                                   (..., Unknown('not_reported'))])
def test_claude_output_correction_accepts_exact_bounds_and_absence(
        output: object, expected: Known | Unknown) -> None:
    read = read_claude(_versioned_source(('2.1.97', output, 'message', 'request')),
                       locator=f'/fixture/{MAIN}.jsonl')
    assert isinstance(read, ClaudeReadBatch)
    assert read.usage[0].tokens.buckets.output == expected


@pytest.mark.parametrize('output', [True, 1.5, -1, 2**63], ids=['bool', 'fraction', 'negative', 'overflow'])
def test_claude_output_correction_rejects_invalid_modern_finals(output: object) -> None:
    read = read_claude(_versioned_source(('2.1.97', output, 'message', 'request')),
                       locator=f'/fixture/{MAIN}.jsonl')
    assert isinstance(read, ClaudeReadBatch)
    assert read.usage[0].tokens.buckets.output == Unknown('invalid_count')
    assert read.evidence[0].state == 'unresolved'
    assert 'invalid_count' in {item.code for item in read.diagnostics}


def test_claude_output_correction_evaluates_mixed_calls_per_entry() -> None:
    read = read_claude(_versioned_source(
        ('2.1.96', 5, 'old-message', 'old-request'),
        ('2.1.97', 7, 'new-message', 'new-request'),
        (None, 9, 'unknown-message', 'unknown-request')),
        locator=f'/fixture/{MAIN}.jsonl')
    assert isinstance(read, ClaudeReadBatch)
    assert tuple(record.tokens.buckets.output for record in read.usage) == (
        Unknown('stream_start_placeholder'), Known(7), Unknown('writer_version_unavailable'))


def test_claude_output_correction_equal_modern_blocks_deduplicate_in_reports(tmp_path: Path) -> None:
    storage = Storage(tmp_path / 'ledger.duckdb')
    storage.import_source(f'/fixture/{MAIN}.jsonl', (FIXTURES / 'versioned.jsonl').read_bytes())
    report = build_report(storage.report_input(ReportQuery(None, AllTime())).contributions,
                          revision=storage.snapshot().revision, query=ReportQuery(None, AllTime()))
    assert report.tokens.input.known == 15
    assert report.tokens.output.known == 37
    assert report.tokens.output.unknown_observations == 1
    assert report.tokens.total.known == 54
    assert report.tokens.total.unknown_observations == 2
    assert report.money.missing_observations == 3


def test_claude_output_correction_unequal_modern_finals_unresolve_whole_call(tmp_path: Path) -> None:
    data = _versioned_source(
        ('2.1.97', 37, 'message', 'request'), ('2.1.273', 38, 'message', 'request'))
    storage = Storage(tmp_path / 'ledger.duckdb')
    storage.import_source(f'/fixture/{MAIN}.jsonl', data)
    observations = storage.snapshot().observations
    assert len(observations) == 2
    assert all(set(item.decisions.values()) == {'unresolved'} for item in observations)
    assert all(set(item.reasons) == {'claude_output_conflict'} for item in observations)


@pytest.mark.parametrize(('versions', 'same_entry'), [
    (('2.1.97', '2.1.273'), False),
    (('2.1.273', '2.1.97'), False),
    (('2.1.97', '2.1.273'), True),
    (('2.1.96', '2.1.95'), False),
])
def test_claude_output_fix1_qualified_writer_mismatch_unresolves_one_call(
        tmp_path: Path, versions: tuple[str, str], same_entry: bool) -> None:
    data = _versioned_source(
        (versions[0], 37, 'message', 'request'),
        (versions[1], 37, 'message', 'request'))
    if same_entry:
        data = data.replace(b'entry-2', b'entry-1')
    path = tmp_path / 'ledger.duckdb'
    storage = Storage(path)
    storage.import_source(f'/fixture/{MAIN}.jsonl', data)
    storage.close()
    reopened = Storage(path)
    observations = reopened.snapshot().observations
    assert len(observations) == 2
    assert all(set(item.decisions.values()) == {'unresolved'} for item in observations)
    assert all(set(item.reasons) == {'claude_output_conflict'} for item in observations)


@pytest.mark.parametrize('reverse', [False, True])
@pytest.mark.parametrize('batch', [False, True], ids=['incremental', 'batch'])
def test_claude_output_fix1_writer_conflict_is_source_order_independent(
        tmp_path: Path, reverse: bool, batch: bool) -> None:
    first = (f'/a/{MAIN}.jsonl',
             _versioned_source(('2.1.97', 37, 'message', 'request')))
    second_data = _versioned_source(('2.1.273', 37, 'message', 'request')).replace(
        MAIN.encode(), BRANCH.encode()).replace(b'entry-1', b'other-entry')
    second = (f'/b/{BRANCH}.jsonl', second_data)
    sources = (second, first) if reverse else (first, second)
    path = tmp_path / f'ledger-{reverse}-{batch}.duckdb'
    storage = Storage(path)
    if batch:
        storage.import_sources(sources)
    else:
        for locator, data in sources:
            storage.import_source(locator, data)
    storage.close()
    decisions = Storage(path).snapshot().observations
    assert len(decisions) == 2
    assert all(set(item.decisions.values()) == {'unresolved'} for item in decisions)


def test_claude_output_fix1_mixed_writers_on_distinct_calls_remain_independent(tmp_path: Path) -> None:
    data = _versioned_source(
        ('2.1.96', 5, 'old-message', 'old-request'),
        ('2.1.273', 7, 'new-message', 'new-request'))
    storage = Storage(tmp_path / 'ledger.duckdb')
    storage.import_source(f'/fixture/{MAIN}.jsonl', data)
    selected = [row for row in storage.snapshot().observations
                if row.decisions['output'] == 'selected']
    assert len(selected) == 2
    assert {row.record.tokens.buckets.output for row in selected} == {
        Unknown('stream_start_placeholder'), Known(7)}


def test_claude_output_fix1_writer_conflict_crosses_typed_identity_bridge(tmp_path: Path) -> None:
    data = _versioned_source(
        ('2.1.97', 37, 'message', ''),
        ('2.1.97', 37, 'message', 'request'),
        ('2.1.273', 37, '', 'request'))
    storage = Storage(tmp_path / 'ledger.duckdb')
    storage.import_source(f'/fixture/{MAIN}.jsonl', data)
    observations = storage.snapshot().observations
    assert len(observations) == 3
    assert all(set(item.decisions.values()) == {'unresolved'} for item in observations)


def test_claude_output_fix1_writer_token_is_fixed_private_metadata(tmp_path: Path) -> None:
    version = '2.1.273+PRIVATE-CANARY'
    data = _versioned_source(
        (version, 37, 'message', 'request'),
        (version, 37, 'message', 'request'))
    storage = Storage(tmp_path / 'ledger.duckdb')
    storage.import_source(f'/fixture/{MAIN}.jsonl', data)
    with storage.connect() as db:
        facts = tuple(json.loads(row[0]) for row in db.execute(
            "SELECT safe_facts_json FROM observation WHERE session_id LIKE 'claude:%'"))
        stored = '\n'.join(str(tuple(row)) for table in
                           ('observation', 'claude_evidence', 'diagnostic', 'source_generation')
                           for row in db.execute(f'SELECT * FROM {table}'))
    assert facts and all('usage.writerVersionToken' in fact for fact in facts)
    tokens = {fact['usage.writerVersionToken'] for fact in facts}
    assert len(tokens) == 1
    assert all(len(token) == 64 and token.isascii() and token.isalnum() for token in tokens)
    assert 'PRIVATE-CANARY' not in stored
    assert all('usage.writerVersionToken' not in dict(row.record.safe_facts)
               for row in storage.snapshot().observations)
    assert sorted(row.decisions['output'] for row in storage.snapshot().observations) == [
        'excluded', 'selected']


def _refinement_source(legacy_version: object = '2.1.96') -> bytes:
    return _versioned_source(
        (legacy_version, 1, 'message', 'request'),
        ('2.1.97', 37, 'message', 'request'))


def _selected_output(storage: Storage) -> list[Known | Unknown]:
    return [row.record.tokens.buckets.output for row in storage.snapshot().observations
            if row.decisions['output'] == 'selected']


@pytest.mark.parametrize('legacy_version', ['2.1.96', None], ids=['qualified-old', 'unversioned'])
def test_claude_output_fix2_real_shape2_upgrade_matches_fresh_current_projection(
        tmp_path: Path, legacy_version: object) -> None:
    data = _refinement_source(legacy_version)
    locator = tmp_path / 'sources' / f'{MAIN}.jsonl'
    upgraded_path = tmp_path / 'shape2.duckdb'
    _frozen_import('claude-output-fix-head-clean', upgraded_path, ((locator, data),))
    upgraded = Storage(upgraded_path)
    assert _selected_output(upgraded) == [Known(37)]
    upgraded.import_source(str(locator), data)

    fresh = Storage(tmp_path / 'fresh.duckdb')
    fresh.import_source(str(locator), data)
    assert _selected_output(upgraded) == _selected_output(fresh) == [Known(37)]
    upgraded_report = build_report(
        upgraded.report_input(ReportQuery(None, AllTime())).contributions,
        revision=upgraded.snapshot().revision, query=ReportQuery(None, AllTime()))
    fresh_report = build_report(
        fresh.report_input(ReportQuery(None, AllTime())).contributions,
        revision=fresh.snapshot().revision, query=ReportQuery(None, AllTime()))
    assert upgraded_report.tokens == fresh_report.tokens
    with upgraded.connect() as db:
        rows = tuple(db.execute(
            "SELECT g.profile,d.state,d.canonical,o.id FROM source_generation g "
            "JOIN appearance a ON a.source_id=g.id JOIN observation o ON o.id=a.observation_id "
            "JOIN decision d ON d.observation_id=o.id AND d.measure='output' "
            "ORDER BY g.generation,a.line"))
    current_canonical = next(row['id'] for row in rows
                             if row['profile'] == PROFILE and row['state'] == 'selected')
    assert all(row['state'] == 'excluded' and row['canonical'] == current_canonical
               for row in rows if row['profile'] != PROFILE)
    revision = upgraded.snapshot().revision
    upgraded.close()
    reopened = Storage(upgraded_path)
    reopened.import_source(str(locator), data)
    assert reopened.snapshot().revision == revision
    assert _selected_output(reopened) == [Known(37)]


def test_claude_output_fix2_repairs_faulty_shape3_ledger_on_unchanged_import(
        tmp_path: Path) -> None:
    data = _refinement_source()
    locator = tmp_path / 'sources' / f'{MAIN}.jsonl'
    path = tmp_path / 'faulty-shape3.duckdb'
    sources = ((locator, data),)
    _frozen_import('claude-output-fix-head-clean', path, sources)
    _frozen_import('claude-output-fix1-head-clean', path, sources)
    storage = Storage(path)
    assert _selected_output(storage) == []
    before_revision = storage.snapshot().revision
    storage.import_source(str(locator), data)
    assert storage.snapshot().revision == before_revision + 1
    assert _selected_output(storage) == [Known(37)]
    repaired_revision = storage.snapshot().revision
    storage.import_source(str(locator), data)
    assert storage.snapshot().revision == repaired_revision


def test_claude_output_fix2_tokenless_other_locator_is_active_conflicting_evidence(
        tmp_path: Path) -> None:
    older_data = _versioned_source(('2.1.97', 37, 'message', 'request'))
    old_locator = tmp_path / 'old' / f'{MAIN}.jsonl'
    path = tmp_path / 'ledger.duckdb'
    _frozen_import('claude-output-fix-head-clean', path, ((old_locator, older_data),))

    current_data = _versioned_source(('2.1.97', 37, 'message', 'request')).replace(
        MAIN.encode(), BRANCH.encode()).replace(b'entry-1', b'current-entry')
    current_locator = tmp_path / 'current' / f'{BRANCH}.jsonl'
    storage = Storage(path)
    storage.import_source(str(current_locator), current_data)
    observations = storage.snapshot().observations
    assert len(observations) == 2
    assert all(set(row.decisions.values()) == {'unresolved'} for row in observations)
    assert all(set(row.reasons) == {'claude_output_conflict'} for row in observations)


def _shape2_identity_pair(tmp_path: Path) -> tuple[Storage, Path, bytes]:
    data = _versioned_source(('2.1.97', 37, 'message', 'request'))
    identified = tmp_path / 'identified' / f'{MAIN}.jsonl'
    identityless = tmp_path / 'identityless' / f'{MAIN}.jsonl'
    path = tmp_path / 'shape2-identity-pair.duckdb'
    _frozen_import(
        'claude-output-fix-head-clean', path,
        ((identified, data), (identityless, _identityless_source(data))))
    storage = Storage(path)
    assert _selected_output(storage) == [Known(37)]
    return storage, identified, data


def test_claude_output_fix3_rejected_projection_does_not_supersede_prior_usage(
        tmp_path: Path) -> None:
    storage, locator, data = _shape2_identity_pair(tmp_path)
    rejected_source = _insert_projection_marker(storage, locator, data, accepted=False)

    trigger = tmp_path / 'trigger' / f'{MAIN}.jsonl'
    storage.import_source(str(trigger), _zero_usage_source())

    assert _selected_output(storage) == [Known(37)]
    with storage.connect() as db:
        assert db.execute(
            "SELECT session_id FROM source_generation WHERE id=?",
            (rejected_source,)).one()[0] is None
        assert db.execute(
            "SELECT code FROM diagnostic WHERE source_id=?",
            (rejected_source,)).one()[0] == 'conflicting_session_identity'


def test_claude_output_fix3_accepted_zero_observation_projection_can_supersede(
        tmp_path: Path) -> None:
    storage, locator, data = _shape2_identity_pair(tmp_path)
    _insert_projection_marker(storage, locator, data, accepted=True)

    trigger = tmp_path / 'trigger' / f'{MAIN}.jsonl'
    storage.import_source(str(trigger), _zero_usage_source())

    assert _selected_output(storage) == []
    assert {reason for row in storage.snapshot().observations for reason in row.reasons} == {
        'missing_claude_identity'}


def test_claude_output_fix3_repairs_poisoned_shape4_decision_once(tmp_path: Path) -> None:
    data = _versioned_source(('2.1.97', 37, 'message', 'request'))
    locator = tmp_path / 'source' / f'{MAIN}.jsonl'
    path = tmp_path / 'poisoned-shape4.duckdb'
    _frozen_import('claude-output-fix2-head-clean', path, ((locator, data),))
    storage = Storage(path)
    with storage.connect(write=True) as db:
        db.execute(
            "UPDATE decision SET state='unresolved',owner_session=NULL,canonical=NULL,"
            "reason='missing_claude_identity' WHERE observation_id IN "
            "(SELECT id FROM observation WHERE session_id=?)",
            (f'claude:{MAIN}',))
    assert _selected_output(storage) == []
    before = storage.snapshot().revision

    storage.import_source(str(locator), data)

    assert storage.snapshot().revision == before + 1
    assert _selected_output(storage) == [Known(37)]
    repaired = storage.snapshot().revision
    storage.import_source(str(locator), data)
    assert storage.snapshot().revision == repaired
    storage.close()
    reopened = Storage(path)
    reopened.import_source(str(locator), data)
    assert reopened.snapshot().revision == repaired
    assert _selected_output(reopened) == [Known(37)]


def test_claude_output_frozen_archive_resolves_from_copied_test_layout(tmp_path: Path) -> None:
    archive = tmp_path / _FROZEN_RELATIVE
    archive.mkdir(parents=True)
    copied_test = archive / 'claude-output-fix2-head-clean' / 'tests' / 'test_claude_source.py'
    assert _frozen_root(copied_test) == archive


def test_claude_output_frozen_import_fails_closed_for_missing_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    archive = tmp_path / 'archive'
    archive.mkdir()
    monkeypatch.setattr(sys.modules[__name__], '_FROZEN_RELATIVE', archive)
    locator = tmp_path / 'source' / f'{MAIN}.jsonl'
    ledger = tmp_path / 'must-not-exist.duckdb'
    with pytest.raises(FileNotFoundError, match='frozen_snapshot_src_missing'):
        _frozen_import('missing-frozen-snapshot', ledger, ((locator, _zero_usage_source()),))
    assert not ledger.exists()
    assert not locator.exists()


def test_claude_output_frozen_import_skips_only_when_archive_absent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys.modules[__name__], '_FROZEN_RELATIVE', tmp_path / 'absent-archive')
    locator = tmp_path / 'source' / f'{MAIN}.jsonl'
    ledger = tmp_path / 'must-not-exist.duckdb'
    with pytest.raises(pytest.skip.Exception, match='historical upgrade test requires private frozen archive'):
        _frozen_import('final-accepted', ledger, ((locator, _zero_usage_source()),))
    assert not ledger.exists()
    assert not locator.exists()


@pytest.mark.parametrize(('other_version', 'other_output'),
                         [('2.1.96', 1), (None, 1), ('2.1.273', ...)])
def test_claude_output_correction_same_entry_writer_or_finality_contradiction_is_unresolved(
        tmp_path: Path, other_version: object, other_output: object) -> None:
    left = _versioned_source((other_version, other_output, 'message', 'request'))
    right = _versioned_source(('2.1.273', 37, 'message', 'request'))
    storage = Storage(tmp_path / 'ledger.duckdb')
    storage.import_sources(((f'/left/{MAIN}.jsonl', left), (f'/right/{MAIN}.jsonl', right)))
    observations = storage.snapshot().observations
    assert len(observations) == 2
    assert all(set(item.decisions.values()) == {'unresolved'} for item in observations)
    assert all(set(item.reasons) == {'claude_output_conflict'} for item in observations)


@pytest.mark.parametrize('reverse', [False, True])
def test_claude_output_correction_modern_final_refines_old_copy_and_survives_reopen(
        tmp_path: Path, reverse: bool) -> None:
    old = (f'/old/{MAIN}.jsonl', _versioned_source(('2.1.96', 1, 'message', 'request')))
    modern = (f'/modern/{BRANCH}.jsonl', _versioned_source(('2.1.273', 37, 'message', 'request')).replace(
        MAIN.encode(), BRANCH.encode()).replace(b'entry-1', b'modern-entry'))
    path = tmp_path / f'ledger-{reverse}.duckdb'
    storage = Storage(path)
    storage.import_sources((modern, old) if reverse else (old, modern))
    storage.close()
    reopened = Storage(path)
    selected = [row for row in reopened.snapshot().observations if row.decisions['output'] == 'selected']
    assert len(selected) == 1
    assert selected[0].record.tokens.buckets.output == Known(37)
    revision = reopened.snapshot().revision
    reopened.import_sources((old, modern))
    assert reopened.snapshot().revision == revision


def test_claude_output_correction_removal_reappearance_is_stable(tmp_path: Path) -> None:
    locator = f'/fixture/{MAIN}.jsonl'
    data = _versioned_source(('2.1.273', 37, 'message', 'request'))
    storage = Storage(tmp_path / 'ledger.duckdb')
    storage.import_source(locator, data)
    before = [(row.record.entry.native_id, row.decisions) for row in storage.snapshot().observations]
    storage.mark_missing((locator,))
    storage.import_source(locator, data)
    after = [(row.record.entry.native_id, row.decisions) for row in storage.snapshot().observations]
    assert after == before


def test_claude_output_correction_persists_only_bounded_content_free_facts(tmp_path: Path) -> None:
    storage = Storage(tmp_path / 'ledger.duckdb')
    storage.import_source(f'/fixture/{MAIN}.jsonl', _versioned_source(
        ('2.1.273+PRIVATE-CANARY', 37, 'message', 'request')))
    with storage.connect() as db:
        stored = '\n'.join(str(tuple(row)) for table in
                           ('observation', 'claude_evidence', 'diagnostic', 'source_generation')
                           for row in db.execute(f'SELECT * FROM {table}'))
    assert 'PRIVATE-CANARY' not in stored
    assert storage.snapshot().observations[0].record.tokens.buckets.output == Known(37)


def _seed_prior_claude_reader_ledger(path: Path, locator: str, data: bytes) -> None:
    storage = Storage(path)
    source_id, observation_id = 'prior-source', 'prior-observation'
    session_id = f'claude:{MAIN}'
    with storage.connect(write=True) as db:
        db.execute('INSERT INTO session(id,harness,native_id,cwd) VALUES(?,?,?,?)',
                   (session_id, 'claude', MAIN, '/fixture/repo'))
        db.execute("INSERT INTO session_attribution VALUES(?,NULL,NULL,'not_resolved')", (session_id,))
        db.execute('INSERT INTO source_generation VALUES(?,?,?,?,?,?,?,?,?)',
                   (source_id, locator, 0, sha256(data).hexdigest(), session_id,
                    'claude-code/2.1.273-shape-1', len(data), 0, 'available'))
        db.execute('INSERT INTO observation VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                   (observation_id, session_id, 'entry-1', 'prior-fingerprint', 'assistant',
                    'point', 1789549201000000, None, None, 'response_recorded_at', None,
                    'claude-test', None, None, json.dumps({'usage.output': 37})))
        db.execute('INSERT INTO appearance(source_id,line,observation_id) VALUES(?,?,?)',
                   (source_id, 1, observation_id))
        db.execute('INSERT INTO entry_edge VALUES(?,?,?,?)', (source_id, 1, 'entry-1', None))
        token_rows = (
            ('input', 'known', 3, None), ('output', 'unknown', None, 'stream_start_only'),
            ('cache_read', 'unknown', None, 'not_reported'),
            ('cache_write', 'unknown', None, 'not_reported'),
            ('reported_total', 'unknown', None, 'partial_total'),
            ('reasoning', 'unknown', None, 'not_reported'),
            ('cache_write_1h', 'unknown', None, 'not_reported'))
        db.executemany('INSERT INTO token_value VALUES(?,?,?,?,?)',
                       ((observation_id, *row) for row in token_rows))
        db.execute('INSERT INTO recorded_estimate VALUES(?,?,?,?,?,?,?)',
                   (observation_id, 'missing', None, None, None, None, 'not_recorded'))
        db.execute('INSERT INTO claude_evidence(source_id,line,observation_id,entry_id,message_id,'
                   'request_id,entry_uuid,agent_id,state,reason) VALUES(?,?,?,?,?,?,?,?,?,?)',
                   (source_id, 1, observation_id, observation_id, 'message', 'request',
                    'entry-1', None, 'usable', None))
        db.execute('INSERT INTO source_metadata VALUES(?,?)', (source_id, sha256(data).hexdigest()))
        db.execute("INSERT INTO diagnostic(source_id,code) VALUES(?,'writer_version_unavailable')",
                   (source_id,))
        db.executemany('INSERT INTO decision VALUES(?,?,?,?,?,?,?)',
                       ((observation_id, measure, 'selected', session_id, None, 'claude_call', 'claude-1')
                        for measure in ('input', 'output', 'cache_read', 'cache_write', 'total',
                                        'recorded_usd')))
        db.execute('UPDATE ledger_meta SET revision=1')


def test_claude_output_correction_unchanged_prior_reader_ledger_reprojects_once(tmp_path: Path) -> None:
    locator = f'/fixture/{MAIN}.jsonl'
    data = _versioned_source(('2.1.273', 37, 'message', 'request'))
    path = tmp_path / 'ledger.duckdb'
    _seed_prior_claude_reader_ledger(path, locator, data)
    storage = Storage(path)
    before = storage.snapshot()
    assert before.observations[0].record.tokens.buckets.output == Unknown('stream_start_only')
    assert 'writer_version_unavailable' in before.diagnostics
    storage.import_source(locator, data)
    selected = [row for row in storage.snapshot().observations if row.decisions['output'] == 'selected']
    assert len(selected) == 1
    assert selected[0].record.tokens.buckets.output == Known(37)
    assert 'writer_version_unavailable' not in storage.snapshot().diagnostics
    assert 'writer_version_unavailable' not in storage.report_input(
        ReportQuery(None, AllTime())).diagnostics
    with storage.connect() as db:
        assert tuple(row[0] for row in db.execute(
            'SELECT profile FROM source_generation WHERE locator=? ORDER BY generation', (locator,))) == (
                'claude-code/2.1.273-shape-1', PROFILE)
    revision = storage.snapshot().revision
    storage.close()
    reopened = Storage(path)
    reopened.import_source(locator, data)
    assert reopened.snapshot().revision == revision
    assert 'writer_version_unavailable' not in reopened.snapshot().diagnostics


def test_claude_output_fix1_reprojection_keeps_current_unknown_writer_diagnostic(
        tmp_path: Path) -> None:
    locator = f'/fixture/{MAIN}.jsonl'
    data = _versioned_source((None, 37, 'message', 'request'))
    path = tmp_path / 'ledger.duckdb'
    _seed_prior_claude_reader_ledger(path, locator, data)
    storage = Storage(path)
    storage.import_source(locator, data)
    assert 'writer_version_unavailable' in storage.snapshot().diagnostics
    assert 'writer_version_unavailable' in storage.report_input(
        ReportQuery(None, AllTime())).diagnostics
    with storage.connect() as db:
        retained = tuple(db.execute(
            "SELECT g.profile,d.line FROM diagnostic d JOIN source_generation g ON g.id=d.source_id "
            "WHERE d.code='writer_version_unavailable' ORDER BY g.generation,d.line"))
    assert retained == ((PROFILE, 1),)


@pytest.mark.parametrize('changed_bytes', [False, True], ids=['unavailable', 'changed-bytes'])
def test_claude_output_fix1_does_not_retire_stale_diagnostic_without_unchanged_available_upgrade(
        tmp_path: Path, changed_bytes: bool) -> None:
    locator = f'/fixture/{MAIN}.jsonl'
    data = _versioned_source(('2.1.273', 37, 'message', 'request'))
    path = tmp_path / 'ledger.duckdb'
    _seed_prior_claude_reader_ledger(path, locator, data)
    storage = Storage(path)
    if changed_bytes:
        imported = _versioned_source(('2.1.273', 38, 'message', 'request'))
    else:
        with storage.connect(write=True) as db:
            db.execute("UPDATE source_generation SET availability='missing' WHERE locator=?", (locator,))
        imported = data
    storage.import_source(locator, imported)
    assert 'writer_version_unavailable' in storage.snapshot().diagnostics


def test_claude_output_correction_family_rollup_and_transcript_content_are_independent(
        tmp_path: Path) -> None:
    main_locator = tmp_path / f'{MAIN}.jsonl'
    child_locator = tmp_path / MAIN / 'subagents' / 'agent-reviewer.jsonl'
    main = _versioned_source(('2.1.273', 37, 'message-main', 'request-main'))
    child = _versioned_source(('2.1.273', 5, 'message-child', 'request-child')).replace(
        b'entry-1', b'child-entry')
    storage = Storage(tmp_path / 'ledger.duckdb')
    storage.import_sources(((str(main_locator), main), (str(child_locator), child)))
    report = build_report(storage.report_input(ReportQuery(None, AllTime())).contributions,
                          revision=storage.snapshot().revision, query=ReportQuery(None, AllTime()))
    assert report.total_session_count == 1
    assert report.sessions[0].subagent_count == 1
    assert report.tokens.output.known == 42
    transcript = parse_claude_transcript(main, MAIN)
    assert 'PRIVATE_CANARY' in repr(transcript)


def test_claude_output_correction_preserves_unrelated_harness_decision_rows(tmp_path: Path) -> None:
    storage = Storage(tmp_path / 'ledger.duckdb')
    storage.import_source('/pi', (Path(__file__).parent / 'fixtures/pi/source/ordinary.jsonl').read_bytes())
    with storage.connect() as db:
        before = tuple(db.execute(
            "SELECT d.rowid,d.* FROM decision d JOIN observation o ON o.id=d.observation_id "
            "WHERE o.session_id LIKE 'pi:%' ORDER BY d.observation_id,d.measure"))
    storage.import_source(f'/fixture/{MAIN}.jsonl',
                          _versioned_source(('2.1.273', 37, 'message', 'request')))
    with storage.connect() as db:
        after = tuple(db.execute(
            "SELECT d.rowid,d.* FROM decision d JOIN observation o ON o.id=d.observation_id "
            "WHERE o.session_id LIKE 'pi:%' ORDER BY d.observation_id,d.measure"))
    assert after == before


def test_reader_preserves_partial_counts_and_no_content() -> None:
    read = read_claude((FIXTURES / 'main.jsonl').read_bytes(), locator=f'/fixture/{MAIN}.jsonl')
    assert isinstance(read, ClaudeReadBatch)
    assert read.session.id == f'claude:{MAIN}'
    assert read.session.cwd == '/fixture/repo'
    assert len(read.usage) == 3
    first, repeated, zero = read.usage
    assert first.model.provider is None and first.model.model == 'claude-sonnet-4-5'
    assert first.tokens.buckets.input == Known(10)
    assert first.tokens.buckets.cache_read == Known(4)
    assert first.tokens.buckets.cache_write == Known(3)
    assert first.tokens.buckets.output == Unknown('writer_version_unavailable')
    assert first.tokens.reported_total == Unknown('partial_total')
    assert first.money == MissingEstimate('not_recorded')
    assert repeated.entry.native_id != first.entry.native_id
    assert repeated.tokens.buckets.output == Unknown('writer_version_unavailable')
    assert zero.tokens.buckets.input == Known(0)
    assert zero.tokens.buckets.cache_read == Known(0)
    assert zero.tokens.buckets.cache_write == Known(0)
    assert [record.safe_facts for record in read.usage] == [
        (('usage.output', 1), ('usage.outputFinality', 'unqualified')),
        (('usage.output', 2), ('usage.outputFinality', 'unqualified')),
        (('usage.output', 0), ('usage.outputFinality', 'unqualified'))]
    assert not any('CANARY_' in repr(record.safe_facts) for record in read.usage)
    assert 'writer_version_unavailable' in {item.code for item in read.diagnostics}


def test_reader_validates_locator_identity_and_subagent_parent() -> None:
    data = (FIXTURES / 'subagent.jsonl').read_bytes()
    locator = f'/fixture/{MAIN}/subagents/nested/agent-reviewer.jsonl'
    read = read_claude(data, locator=locator, profile=PROFILE)
    assert isinstance(read, ClaudeReadBatch)
    assert read.session.id == f'claude:{MAIN}:agent:reviewer'
    assert read.session.parent_locator == f'/fixture/{MAIN}.jsonl'
    assert read.parent_session_id == f'claude:{MAIN}'
    assert read.evidence[0].agent_id == 'reviewer'
    rejected = read_claude(data, locator=f'/fixture/{BRANCH}.jsonl')
    assert isinstance(rejected, RejectedSource)
    assert rejected.diagnostics[0].code == 'conflicting_session_identity'
    missing = data.replace((b'"sessionId":"' + MAIN.encode() + b'",'), b'')
    assert isinstance(read_claude(missing, locator=locator), RejectedSource)


def test_reader_rejects_bad_counts_and_uses_fixed_undated_reason() -> None:
    data = (FIXTURES / 'main.jsonl').read_text().splitlines()[1]
    bad = data.replace('"input_tokens":10', '"input_tokens":1.5').replace('"timestamp":"2026-09-16T08:00:00Z",', '')
    read = read_claude((bad + '\n').encode(), locator=f'/fixture/{MAIN}.jsonl')
    assert isinstance(read, ClaudeReadBatch)
    assert read.usage == ()
    assert 'invalid_count' in {item.code for item in read.diagnostics}
    valid = read_claude((data.replace('"timestamp":"2026-09-16T08:00:00Z",', '') + '\n').encode(), locator=f'/fixture/{MAIN}.jsonl')
    assert isinstance(valid, ClaudeReadBatch)
    assert isinstance(valid.usage[0].time, Undated)
    assert valid.usage[0].time.reason == 'timestamp_unavailable'


def test_reader_reports_malformed_middle_and_holds_only_final_fragment() -> None:
    line = (FIXTURES / 'branch.jsonl').read_bytes().splitlines()[0]
    read = read_claude(line + b'\n{"bad":\n' + line[:20], locator=f'/fixture/{BRANCH}.jsonl')
    assert isinstance(read, ClaudeReadBatch)
    assert read.pending_tail is True
    assert 'malformed_json' in {item.code for item in read.diagnostics}
    assert read.complete_bytes == len(line) + 1 + len(b'{"bad":\n')


def test_reader_accepts_cwd_change_without_changing_identity() -> None:
    line = (FIXTURES / 'branch.jsonl').read_text().splitlines()[0]
    second = line.replace('/fixture/branch', '/fixture/elsewhere').replace('000000000012', '000000000099')
    read = read_claude((line + '\n' + second + '\n').encode(), locator=f'/fixture/{BRANCH}.jsonl')
    assert isinstance(read, ClaudeReadBatch)
    assert read.session.id == f'claude:{BRANCH}'


@pytest.mark.parametrize('second_cwd', ['relative/path', '/fixture/other'])
def test_reader_uses_one_cwd_validity_decision_for_metadata_and_diagnostic(second_cwd: str) -> None:
    line = (FIXTURES / 'branch.jsonl').read_text().splitlines()[0]
    second = line.replace('/fixture/branch', second_cwd).replace('000000000012', '000000000099')
    read = read_claude((line + '\n' + second + '\n').encode(), locator=f'/fixture/{BRANCH}.jsonl')
    assert isinstance(read, ClaudeReadBatch)
    assert read.session.cwd is None
    assert 'cwd_attribution_unavailable' in {item.code for item in read.diagnostics}
    assert len(read.usage) == 2


def _claude_cwd_source(session_id: str, entry: str, cwd: object = ...) -> bytes:
    record: dict[str, object] = {
        'type': 'assistant', 'uuid': entry, 'parentUuid': None,
        'sessionId': session_id, 'timestamp': '2026-09-16T08:00:00Z',
        'message': {'id': 'message-' + entry, 'role': 'assistant', 'model': 'claude-test',
                    'usage': {'input_tokens': 3}, 'content': []},
    }
    if cwd is not ...:
        record['cwd'] = cwd
    return (json.dumps(record) + '\n').encode()


@pytest.mark.parametrize('invalid', [None, '', ' ', 'relative/path', 7, True, 1.5, [], {}],
                         ids=['null', 'empty', 'whitespace', 'relative', 'integer', 'bool',
                              'fraction', 'list', 'object'])
@pytest.mark.parametrize('invalid_first', [False, True], ids=['valid-first', 'invalid-first'])
def test_reader_distinguishes_present_invalid_cwd_from_omission(
        invalid: object, invalid_first: bool) -> None:
    valid = _claude_cwd_source(BRANCH, 'valid', '/repo')
    bad = _claude_cwd_source(BRANCH, 'invalid', invalid)
    read = read_claude((bad + valid) if invalid_first else (valid + bad),
                       locator=f'/fixture/{BRANCH}.jsonl')
    assert isinstance(read, ClaudeReadBatch)
    assert read.session.cwd is None
    assert 'cwd_attribution_unavailable' in {item.code for item in read.diagnostics}
    assert len(read.usage) == 2


@pytest.mark.parametrize('missing_first', [False, True])
def test_reader_allows_cwd_omission_beside_one_valid_value(missing_first: bool) -> None:
    valid = _claude_cwd_source(BRANCH, 'valid', '/repo')
    missing = _claude_cwd_source(BRANCH, 'missing')
    read = read_claude((missing + valid) if missing_first else (valid + missing),
                       locator=f'/fixture/{BRANCH}.jsonl')
    assert isinstance(read, ClaudeReadBatch)
    assert read.session.cwd == '/repo'
    assert 'cwd_attribution_unavailable' not in {item.code for item in read.diagnostics}


@pytest.mark.parametrize('missing_first', [False, True])
def test_storage_missing_and_valid_cwd_merge_to_valid_in_both_orders_and_reopen(
        tmp_path: Path, missing_first: bool) -> None:
    missing = (f'/a/{MAIN}.jsonl', _claude_cwd_source(MAIN, 'missing'))
    valid = (f'/b/{MAIN}.jsonl', _claude_cwd_source(MAIN, 'valid', '/repo'))
    ordered = (missing, valid) if missing_first else (valid, missing)
    path = tmp_path / f'missing-valid-{missing_first}.duckdb'
    storage = Storage(path)
    storage.import_sources(ordered)
    storage.close()
    reopened = Storage(path)
    assert reopened.snapshot().sessions[0].cwd == '/repo'
    assert _selected_input(reopened) == 6


@pytest.mark.parametrize('invalid', [None, '', 7, 'relative/path'],
                         ids=['null', 'empty', 'integer', 'relative'])
@pytest.mark.parametrize('invalid_first', [False, True])
def test_storage_invalid_cwd_poison_is_sticky_across_import_order_and_reopen(
        tmp_path: Path, invalid: object, invalid_first: bool) -> None:
    invalid_source = (f'/a/{MAIN}.jsonl', _claude_cwd_source(MAIN, 'invalid', invalid))
    valid = (f'/b/{MAIN}.jsonl', _claude_cwd_source(MAIN, 'valid', '/repo'))
    ordered = (invalid_source, valid) if invalid_first else (valid, invalid_source)
    path = tmp_path / f'invalid-valid-{invalid_first}-{type(invalid).__name__}.duckdb'
    storage = Storage(path)
    storage.import_sources(ordered)
    storage.import_source(f'/c/{MAIN}.jsonl', _claude_cwd_source(MAIN, 'later', '/repo'))
    storage.close()
    reopened = Storage(path)
    assert reopened.snapshot().sessions[0].cwd is None
    assert _selected_input(reopened) == 9
    with reopened.connect() as db:
        assert db.execute(
            "SELECT 1 FROM diagnostic d JOIN source_generation g ON g.id=d.source_id "
            "WHERE g.session_id=? AND d.code='cwd_attribution_unavailable' AND d.line IS NOT NULL",
            (f'claude:{MAIN}',)).fetchone() is not None


def test_storage_cwd_state_is_conservative_across_same_locator_generations(
        tmp_path: Path) -> None:
    locator = f'/fixture/{MAIN}.jsonl'
    preserving = Storage(tmp_path / 'preserving.duckdb')
    preserving.import_source(locator, _claude_cwd_source(MAIN, 'missing-one'))
    preserving.import_source(locator, _claude_cwd_source(MAIN, 'valid', '/repo'))
    preserving.import_source(locator, _claude_cwd_source(MAIN, 'missing-two'))
    assert preserving.snapshot().sessions[0].cwd == '/repo'

    poisoned = Storage(tmp_path / 'poisoned.duckdb')
    poisoned.import_source(locator, _claude_cwd_source(MAIN, 'valid-one', '/repo'))
    poisoned.import_source(locator, _claude_cwd_source(MAIN, 'invalid', None))
    poisoned.import_source(locator, _claude_cwd_source(MAIN, 'valid-two', '/repo'))
    assert poisoned.snapshot().sessions[0].cwd is None


@pytest.mark.parametrize('reverse', [False, True])
def test_storage_conflicting_valid_cwds_remain_unavailable_after_later_valid(
        tmp_path: Path, reverse: bool) -> None:
    first = (f'/a/{MAIN}.jsonl', _claude_cwd_source(MAIN, 'first', '/repo-a'))
    second = (f'/b/{MAIN}.jsonl', _claude_cwd_source(MAIN, 'second', '/repo-b'))
    storage = Storage(tmp_path / f'conflicting-{reverse}.duckdb')
    storage.import_sources((second, first) if reverse else (first, second))
    storage.import_source(f'/c/{MAIN}.jsonl', _claude_cwd_source(MAIN, 'later', '/repo-a'))
    assert storage.snapshot().sessions[0].cwd is None


def _selected_input(storage: Storage) -> int:
    return sum(record.record.tokens.buckets.input.value for record in storage.snapshot().observations
               if record.decisions.get('input') == 'selected' and isinstance(record.record.tokens.buckets.input, Known))


def test_storage_deduplicates_typed_message_request_graph_in_both_orders(tmp_path: Path) -> None:
    main = (FIXTURES / 'main.jsonl').read_bytes()
    branch = (FIXTURES / 'branch.jsonl').read_bytes()
    for reverse in (False, True):
        storage = Storage(tmp_path / f'order-{reverse}.duckdb')
        sources = [(f'/fixture/{MAIN}.jsonl', main), (f'/fixture/{BRANCH}.jsonl', branch)]
        storage.import_sources(reversed(sources) if reverse else sources)
        assert _selected_input(storage) == 15
        copied = [row for row in storage.snapshot().observations if row.record.entry.native_id in {
            '00000000-0000-4000-8000-000000000002',
            '00000000-0000-4000-8000-000000000003',
            '00000000-0000-4000-8000-000000000012'}]
        assert sum(row.decisions['input'] == 'selected' for row in copied) == 1
        assert sum(row.decisions['input'] == 'excluded' for row in copied) == 2


def test_storage_whole_call_conflict_and_identityless_are_unresolved(tmp_path: Path) -> None:
    storage = Storage(tmp_path / 'ledger.duckdb')
    main = (FIXTURES / 'main.jsonl').read_bytes()
    conflict = (FIXTURES / 'branch.jsonl').read_bytes().replace(b'"input_tokens":10', b'"input_tokens":11', 1)
    storage.import_sources(((f'/fixture/{MAIN}.jsonl', main), (f'/fixture/{BRANCH}.jsonl', conflict)))
    copied = [row for row in storage.snapshot().observations if row.record.entry.native_id in {
        '00000000-0000-4000-8000-000000000002',
        '00000000-0000-4000-8000-000000000003',
        '00000000-0000-4000-8000-000000000012'}]
    assert copied and all(set(row.decisions.values()) == {'unresolved'} for row in copied)
    line = (FIXTURES / 'branch.jsonl').read_text().splitlines()[1]
    no_ids = line.replace(',"requestId":"request-branch"', '').replace('"id":"message-branch",', '')
    no_ids = no_ids.replace('00000000-0000-4000-8000-000000000013',
                            '00000000-0000-4000-8000-000000000099')
    no_ids = no_ids.replace(BRANCH, '33333333-3333-4333-8333-333333333333')
    storage.import_source('/fixture/33333333-3333-4333-8333-333333333333.jsonl', (no_ids + '\n').encode())
    record = next(row for row in storage.snapshot().observations if row.session_id == 'claude:33333333-3333-4333-8333-333333333333')
    assert set(record.decisions.values()) == {'unresolved'}


@pytest.mark.parametrize('field,first,second', [
    ('message', 'message-one', 'message-two'),
    ('request', 'request-one', 'request-two'),
])
def test_changed_present_identity_on_exact_entry_quarantines_component(
        tmp_path: Path, field: str, first: str, second: str) -> None:
    def source(identity: str) -> bytes:
        message = {'role': 'assistant', 'model': 'claude-test',
                   'usage': {'input_tokens': 3}, 'content': []}
        record: dict[str, object] = {
            'type': 'assistant', 'uuid': 'same-entry', 'parentUuid': None,
            'sessionId': MAIN, 'cwd': '/fixture/repo',
            'timestamp': '2026-09-16T08:00:00Z', 'message': message,
        }
        if field == 'message':
            message['id'] = identity
        else:
            record['requestId'] = identity
        return (json.dumps(record) + '\n').encode()
    storage = Storage(tmp_path / f'{field}.duckdb')
    storage.import_sources(((f'/a/{MAIN}.jsonl', source(first)),
                            (f'/b/{MAIN}.jsonl', source(second))))
    observation = storage.snapshot().observations[0]
    assert set(observation.decisions.values()) == {'unresolved'}
    assert observation.reasons == ('claude_identity_conflict',)


def test_neither_id_exact_entry_redelivery_is_excluded_not_conflicting(tmp_path: Path) -> None:
    def source(session_id: str, *, identified: bool) -> bytes:
        message: dict[str, object] = {'role': 'assistant', 'model': 'claude-test',
                                      'usage': {'input_tokens': 3}, 'content': []}
        if identified:
            message['id'] = 'message-one'
        return (json.dumps({
            'type': 'assistant', 'uuid': 'same-entry', 'parentUuid': None,
            'sessionId': session_id, 'cwd': '/fixture/repo',
            'timestamp': '2026-09-16T08:00:00Z', 'message': message,
        }) + '\n').encode()
    storage = Storage(tmp_path / 'redelivery.duckdb')
    storage.import_sources(((f'/a/{MAIN}.jsonl', source(MAIN, identified=True)),
                            (f'/b/{BRANCH}.jsonl', source(BRANCH, identified=False))))
    observations = {row.session_id: row for row in storage.snapshot().observations}
    assert set(observations[f'claude:{MAIN}'].decisions.values()) == {'selected'}
    assert set(observations[f'claude:{BRANCH}'].decisions.values()) == {'excluded'}


@pytest.mark.parametrize('redelivery_time', ['2026-09-16T08:00:00Z', '2026-09-16T07:00:00Z'],
                         ids=['earlier-locator', 'earlier-time'])
def test_identified_exact_entry_peer_remains_canonical_after_reopen(
        tmp_path: Path, redelivery_time: str) -> None:
    def source(session_id: str, *, identified: bool, timestamp: str) -> bytes:
        message: dict[str, object] = {'role': 'assistant', 'model': 'claude-test',
                                      'usage': {'input_tokens': 10,
                                                'cache_read_input_tokens': 4,
                                                'cache_creation_input_tokens': 3},
                                      'content': []}
        if identified:
            message['id'] = 'message-one'
        return (json.dumps({
            'type': 'assistant', 'uuid': 'same-entry', 'parentUuid': None,
            'sessionId': session_id, 'cwd': '/fixture/repo', 'timestamp': timestamp,
            'message': message,
        }) + '\n').encode()
    identified = (f'/z/{MAIN}.jsonl', source(MAIN, identified=True,
                                             timestamp='2026-09-16T08:00:00Z'))
    redelivery = (f'/a/{BRANCH}.jsonl', source(BRANCH, identified=False,
                                               timestamp=redelivery_time))
    results = []
    for name, sources in (('forward', (identified, redelivery)),
                          ('reverse', (redelivery, identified))):
        path = tmp_path / f'{name}-{redelivery_time[11:13]}.duckdb'
        storage = Storage(path)
        storage.import_sources(sources)
        storage.close()
        reopened = Storage(path)
        observations = {row.session_id: row for row in reopened.snapshot().observations}
        results.append(tuple((session_id, tuple(sorted(row.decisions.items())))
                             for session_id, row in sorted(observations.items())))
        assert set(observations[f'claude:{MAIN}'].decisions.values()) == {'selected'}
        assert set(observations[f'claude:{BRANCH}'].decisions.values()) == {'excluded'}
    assert results[0] == results[1]


def test_canonical_sort_uses_identified_appearance_before_observation_reduction(
        tmp_path: Path) -> None:
    def source(session_id: str, entry: str, message_id: str | None) -> bytes:
        message: dict[str, object] = {'role': 'assistant', 'model': 'claude-test',
                                      'usage': {'input_tokens': 3}, 'content': []}
        if message_id is not None:
            message['id'] = message_id
        return (json.dumps({
            'type': 'assistant', 'uuid': entry, 'parentUuid': None,
            'sessionId': session_id, 'cwd': '/fixture/repo',
            'timestamp': '2026-09-16T08:00:00Z', 'message': message,
        }) + '\n').encode()
    sources = (
        (f'/z/{MAIN}.jsonl', source(MAIN, 'entry-a', 'message-one')),
        (f'/a/{MAIN}.jsonl', source(MAIN, 'entry-a', None)),
        (f'/m/{BRANCH}.jsonl', source(BRANCH, 'entry-b', 'message-one')),
    )
    results = []
    for name, ordered in (('forward', sources), ('reverse', tuple(reversed(sources)))):
        path = tmp_path / f'trap-{name}.duckdb'
        storage = Storage(path)
        storage.import_sources(ordered)
        storage.close()
        reopened = Storage(path)
        with reopened.connect() as db:
            decisions = tuple(db.execute(
                "SELECT o.native_entry_id,o.session_id,d.state,d.owner_session,d.canonical "
                "FROM decision d JOIN observation o ON o.id=d.observation_id "
                "WHERE d.measure='input' AND o.session_id LIKE 'claude:%' "
                "ORDER BY o.native_entry_id"))
        results.append(decisions)
        selected = next(row for row in decisions if row['state'] == 'selected')
        assert selected['native_entry_id'] == 'entry-b'
        assert selected['owner_session'] == f'claude:{BRANCH}'
    assert results[0] == results[1]


def test_repeated_observation_uses_minimum_appearance_key_in_all_import_orders(tmp_path: Path) -> None:
    def source(session_id: str, entry_uuid: str) -> bytes:
        return (json.dumps({
            'type': 'assistant', 'uuid': entry_uuid, 'parentUuid': None,
            'sessionId': session_id, 'cwd': '/fixture/repo', 'requestId': 'request-one',
            'timestamp': '2026-09-16T08:00:00Z',
            'message': {'id': 'message-one', 'role': 'assistant', 'model': 'claude-test',
                        'usage': {'input_tokens': 3}, 'content': []},
        }) + '\n').encode()
    a = source(MAIN, 'entry-a')
    b = source(BRANCH, 'entry-b')
    sources = ((f'/a/{MAIN}.jsonl', a), (f'/m/{BRANCH}.jsonl', b), (f'/z/{MAIN}.jsonl', a))
    decisions = []
    for name, ordered in (('forward', sources), ('reverse', tuple(reversed(sources)))):
        storage = Storage(tmp_path / f'{name}.duckdb')
        storage.import_sources(ordered)
        with storage.connect() as db:
            decisions.append(tuple(db.execute(
                "SELECT o.native_entry_id,d.state,d.owner_session,d.canonical FROM decision d "
                "JOIN observation o ON o.id=d.observation_id WHERE d.measure='input' "
                "AND o.session_id LIKE 'claude:%' ORDER BY o.native_entry_id")))
    assert decisions[0] == decisions[1]
    assert decisions[0][0]['native_entry_id'] == 'entry-a'
    assert decisions[0][0]['state'] == 'selected'
    assert decisions[0][0]['owner_session'] == f'claude:{MAIN}'


def test_subagent_parent_locator_survives_reopen_and_rolls_up(tmp_path: Path) -> None:
    path = tmp_path / 'ledger.duckdb'
    main_locator = tmp_path / f'{MAIN}.jsonl'
    agent_locator = tmp_path / MAIN / 'subagents' / 'nested' / 'agent-reviewer.jsonl'
    storage = Storage(path)
    storage.import_sources(((str(main_locator), (FIXTURES / 'main.jsonl').read_bytes()),
                            (str(agent_locator), (FIXTURES / 'subagent.jsonl').read_bytes())))
    storage.close()
    reopened = Storage(path)
    report = build_report(reopened.report_input(ReportQuery(None, AllTime())).contributions,
                          revision=reopened.snapshot().revision, query=ReportQuery(None, AllTime()))
    assert report.total_session_count == 1
    assert report.sessions[0].id == f'claude:{MAIN}'
    assert report.sessions[0].subagent_count == 1
    assert report.sessions[0].tokens.input.known == 17


def test_persistence_keeps_only_allowlisted_scalar_accounting_metadata(tmp_path: Path) -> None:
    storage = Storage(tmp_path / 'ledger.duckdb')
    storage.import_source(f'/fixture/{MAIN}.jsonl', (FIXTURES / 'main.jsonl').read_bytes())
    canaries = ('CANARY_PROMPT_7f3a', 'CANARY_RESPONSE_9d2c', 'CANARY_REASONING_4b1e', 'CANARY_TOOL_ARGS_6a8f')
    with storage.connect() as db:
        stored = '\n'.join(str(tuple(row)) for table in ('session_view', 'source_generation', 'observation', 'claude_evidence', 'diagnostic')
                           for row in db.execute(f'SELECT * FROM {table}'))
    assert not any(canary in stored for canary in canaries)


def test_reconciliation_uses_typed_bridges_and_exact_entry_redelivery() -> None:
    def candidate(oid: str, message: str | None, request: str | None, entry: str, order: int) -> ClaudeCandidate:
        return ClaudeCandidate(oid, 'session', message, request, entry, 'same', (0, order, '/source', order, entry), 'usable', None)
    decisions = {item.observation_id: item for item in reconcile_claude((
        candidate('message', 'm', None, 'one', 1),
        candidate('bridge', 'm', 'r', 'two', 2),
        candidate('request', None, 'r', 'three', 3),
        candidate('redelivery', None, None, 'one', 4),
        candidate('neither', None, None, 'four', 5),
    ))}
    assert decisions['message'].state == 'selected'
    assert decisions['bridge'].canonical == 'message'
    assert decisions['request'].canonical == 'message'
    assert decisions['redelivery'].canonical == 'message'
    assert decisions['neither'].state == 'unresolved'


def test_storage_uses_observation_id_for_evidence_entry_id_and_warns_on_output_variation(tmp_path: Path) -> None:
    storage = Storage(tmp_path / 'ledger.duckdb')
    storage.import_source(f'/fixture/{MAIN}.jsonl', (FIXTURES / 'main.jsonl').read_bytes())
    with storage.connect() as db:
        evidence = tuple(db.execute('SELECT entry_id,observation_id,entry_uuid FROM claude_evidence ORDER BY line'))
        warnings = {row[0] for row in db.execute("SELECT code FROM diagnostic WHERE code='claude_output_variation'")}
    assert evidence and all(row['entry_id'] == row['observation_id'] for row in evidence)
    assert evidence[0]['entry_uuid'] == '00000000-0000-4000-8000-000000000002'
    assert warnings == {'claude_output_variation'}


@pytest.mark.parametrize('last_output', [9, None], ids=['value-value', 'absent-present'])
def test_output_variation_is_computed_across_typed_bridge_component(
        tmp_path: Path, last_output: int | None) -> None:
    def source(session_id: str, entry: str, message_id: str | None,
               request_id: str | None, output: int | None) -> bytes:
        usage: dict[str, object] = {'input_tokens': 3}
        if output is not None:
            usage['output_tokens'] = output
        message: dict[str, object] = {'role': 'assistant', 'model': 'claude-test',
                                      'usage': usage, 'content': []}
        if message_id is not None:
            message['id'] = message_id
        record: dict[str, object] = {
            'type': 'assistant', 'uuid': entry, 'parentUuid': None,
            'sessionId': session_id, 'cwd': '/fixture/repo',
            'timestamp': '2026-09-16T08:00:00Z', 'message': message,
        }
        if request_id is not None:
            record['requestId'] = request_id
        return (json.dumps(record) + '\n').encode()
    storage = Storage(tmp_path / f'bridge-{last_output}.duckdb')
    storage.import_sources((
        (f'/a/{MAIN}.jsonl', source(MAIN, 'entry-message', 'message', None, 0)),
        (f'/b/{BRANCH}.jsonl', source(BRANCH, 'entry-bridge', 'message', 'request', 0)),
        (f'/c/{THIRD}.jsonl', source(THIRD, 'entry-request', None, 'request', last_output)),
    ))
    snapshot = storage.snapshot()
    assert sum(row.record.tokens.buckets.input.value for row in snapshot.observations
               if row.decisions['input'] == 'selected' and isinstance(row.record.tokens.buckets.input, Known)) == 3
    assert all(isinstance(row.record.tokens.buckets.output, Unknown) for row in snapshot.observations)
    with storage.connect() as db:
        assert {row[0] for row in db.execute("SELECT code FROM diagnostic WHERE code='claude_output_variation'")} == {
            'claude_output_variation'}


def test_claude_scoped_reconciliation_matches_full_and_preserves_unrelated_pi_rows(tmp_path: Path) -> None:
    storage = Storage(tmp_path / 'ledger.duckdb')
    storage.import_source('/pi', (Path(__file__).parent / 'fixtures/pi/source/ordinary.jsonl').read_bytes())
    with storage.connect() as db:
        before_pi = tuple(db.execute("SELECT d.* FROM decision d JOIN observation o ON o.id=d.observation_id WHERE o.session_id LIKE 'pi:%' ORDER BY d.observation_id,d.measure"))
    main_locator, branch_locator = f'/fixture/{MAIN}.jsonl', f'/fixture/{BRANCH}.jsonl'
    storage.import_sources(((main_locator, (FIXTURES / 'main.jsonl').read_bytes()),
                            (branch_locator, (FIXTURES / 'branch.jsonl').read_bytes())))
    conflict = (FIXTURES / 'branch.jsonl').read_bytes().replace(b'"input_tokens":10', b'"input_tokens":11', 1)
    storage.import_source(branch_locator, conflict)
    with storage.connect() as db:
        scoped = tuple(db.execute("SELECT d.* FROM decision d JOIN observation o ON o.id=d.observation_id WHERE o.session_id LIKE 'claude:%' ORDER BY d.observation_id,d.measure"))
        after_pi = tuple(db.execute("SELECT d.* FROM decision d JOIN observation o ON o.id=d.observation_id WHERE o.session_id LIKE 'pi:%' ORDER BY d.observation_id,d.measure"))
    with storage.connect(write=True) as db:
        storage._reconcile(db, None)
    with storage.connect() as db:
        full = tuple(db.execute("SELECT d.* FROM decision d JOIN observation o ON o.id=d.observation_id WHERE o.session_id LIKE 'claude:%' ORDER BY d.observation_id,d.measure"))
    assert scoped == full
    assert before_pi == after_pi


def test_claude_subagent_missing_source_keeps_saved_usage(tmp_path: Path) -> None:
    root = tmp_path / 'sources'; root.mkdir()
    main = root / f'{MAIN}.jsonl'; main.write_bytes((FIXTURES / 'main.jsonl').read_bytes())
    child = root / MAIN / 'subagents' / 'agent-reviewer.jsonl'
    child.parent.mkdir(parents=True); child.write_bytes((FIXTURES / 'subagent.jsonl').read_bytes())
    storage = Storage(tmp_path / 'ledger.duckdb')
    storage.import_sources(((str(main), main.read_bytes()), (str(child), child.read_bytes())))
    query = ReportQuery(None, AllTime())
    before = build_report(storage.report_input(query).contributions, revision=storage.snapshot().revision, query=query)
    child_page = read_transcript_page(storage, (str(root),), f'claude:{MAIN}:agent:reviewer')
    assert child_page.parent is not None and child_page.parent.session_id == f'claude:{MAIN}'
    child.unlink(); storage.mark_missing((str(child),))
    after = build_report(storage.report_input(query).contributions, revision=storage.snapshot().revision, query=query)
    assert after.tokens == before.tokens
    assert 'saved_history' in storage.report_input(query).diagnostics
    with pytest.raises(TranscriptUnavailable) as unavailable:
        read_transcript_page(storage, (str(root),), f'claude:{MAIN}:agent:reviewer')
    assert unavailable.value.kind == 'missing'


def test_transcript_selects_parent_chain_and_pairs_supported_blocks() -> None:
    transcript = parse_claude_transcript((FIXTURES / 'main.jsonl').read_bytes(), MAIN)
    assert transcript.harness == 'claude' and transcript.native_id == MAIN
    messages = [entry for entry in transcript.entries if isinstance(entry, Message)]
    assert messages[0].role == 'user'
    assert messages[0].blocks[0].text.startswith('CANARY_PROMPT_7f3a')
    assistant = messages[1]
    assert any(isinstance(block, ReasoningBlock) for block in assistant.blocks)
    tool = next(block for block in assistant.blocks if isinstance(block, ToolBlock))
    assert tool.name == 'Read' and tool.output is not None
    assert any(message.blocks[0].text == 'done' for message in messages if message.blocks)


def test_transcript_uses_fixed_notices_and_rejects_identity_or_branch() -> None:
    internal = ('{"type":"system","uuid":"sys","parentUuid":null,"sessionId":"' + MAIN
                + '","secret":"CANARY_INTERNAL"}\n')
    user = ('{"type":"user","uuid":"user","parentUuid":"sys","sessionId":"' + MAIN
            + '","isCompactSummary":true,"message":{"content":[{"type":"image","source":{"url":"https://invalid.example/CANARY"}}]}}\n')
    transcript = parse_claude_transcript((internal + user).encode(), MAIN)
    assert any(isinstance(entry, Notice) and entry.label == 'Recorded internal entry' for entry in transcript.entries)
    assert 'CANARY_INTERNAL' not in repr(transcript)
    assert 'https://invalid.example' not in repr(transcript)
    assert any(isinstance(entry, Message) and entry.phase == 'compact summary' for entry in transcript.entries)
    with pytest.raises(TranscriptUnavailable) as mismatch:
        parse_claude_transcript((FIXTURES / 'main.jsonl').read_bytes(), BRANCH)
    assert mismatch.value.kind == 'changed'
    with pytest.raises(TranscriptUnavailable) as invalid_branch:
        parse_claude_transcript((FIXTURES / 'main.jsonl').read_bytes(), MAIN, branch='other')
    assert invalid_branch.value.kind == 'invalid_branch'


@pytest.mark.parametrize('connector', ['system', 'progress', 'attachment'])
def test_transcript_unknown_child_cannot_hide_supported_leaf_and_known_connectors_link(
        connector: str) -> None:
    records = [
        {'type': 'user', 'uuid': 'root', 'parentUuid': None, 'sessionId': MAIN,
         'message': {'content': [{'type': 'text', 'text': 'root'}]}},
        {'type': 'assistant', 'uuid': 'old', 'parentUuid': 'root', 'sessionId': MAIN,
         'message': {'content': [{'type': 'text', 'text': 'old'}]}},
        {'type': connector, 'uuid': 'connector', 'parentUuid': 'root', 'sessionId': MAIN,
         'message': {'content': [{'type': 'text', 'text': 'INTERNAL_CONNECTOR'}]}},
        {'type': 'assistant', 'uuid': 'latest', 'parentUuid': 'connector', 'sessionId': MAIN,
         'message': {'content': [{'type': 'text', 'text': 'latest'}]}},
        {'type': 'unknown-kind', 'uuid': 'unknown', 'parentUuid': 'latest', 'sessionId': MAIN,
         'message': {'content': [{'type': 'text', 'text': 'UNKNOWN_CHILD'}]}},
    ]
    transcript = parse_claude_transcript(
        ''.join(json.dumps(record) + '\n' for record in records).encode(), MAIN)
    rendered = repr(transcript)
    assert "TextBlock(text='latest')" in rendered
    assert "TextBlock(text='old')" not in rendered
    assert 'INTERNAL_CONNECTOR' not in rendered
    assert 'UNKNOWN_CHILD' not in rendered
    assert any(isinstance(entry, Notice) and entry.label == 'Recorded internal entry'
               for entry in transcript.entries)


@pytest.mark.parametrize('record_type', [[], {}, None], ids=['list', 'object', 'null'])
def test_transcript_treats_non_string_record_type_as_unsupported_fixed_notice(
        record_type: object) -> None:
    records = [
        {'type': 'user', 'uuid': 'root', 'parentUuid': None, 'sessionId': MAIN,
         'message': {'content': [{'type': 'text', 'text': 'root'}]}},
        {'type': 'assistant', 'uuid': 'latest', 'parentUuid': 'root', 'sessionId': MAIN,
         'message': {'content': [{'type': 'text', 'text': 'latest'}]}},
        {'type': record_type, 'uuid': 'unsupported', 'parentUuid': 'latest',
         'sessionId': MAIN,
         'message': {'content': [{'type': 'tool_result', 'tool_use_id': 'call',
                                  'content': 'NON_STRING_TYPE_CANARY'}]}},
    ]
    transcript = parse_claude_transcript(
        ''.join(json.dumps(record) + '\n' for record in records).encode(), MAIN)
    assert "TextBlock(text='latest')" in repr(transcript)
    assert 'NON_STRING_TYPE_CANARY' not in repr(transcript)
    assert any(isinstance(entry, Notice) and entry.label == 'Unsupported Claude entry'
               for entry in transcript.entries)


def test_registered_transcript_access_bounds_non_string_record_type(tmp_path: Path) -> None:
    root = tmp_path / 'sources'
    root.mkdir()
    locator = root / f'{MAIN}.jsonl'
    records = [
        {'type': 'user', 'uuid': 'root', 'parentUuid': None, 'sessionId': MAIN,
         'cwd': '/repo', 'message': {'content': [{'type': 'text', 'text': 'root'}]}},
        {'type': 'assistant', 'uuid': 'latest', 'parentUuid': 'root', 'sessionId': MAIN,
         'cwd': '/repo', 'timestamp': '2026-09-16T08:00:00Z',
         'message': {'id': 'message', 'model': 'claude-test', 'usage': {'input_tokens': 3},
                     'content': [{'type': 'text', 'text': 'latest'}]}},
    ]
    initial = ''.join(json.dumps(record) + '\n' for record in records).encode()
    locator.write_bytes(initial)
    storage = Storage(tmp_path / 'ledger.duckdb')
    storage.import_source(str(locator), initial)
    unsupported = {'type': [], 'uuid': 'unsupported', 'parentUuid': 'latest',
                   'sessionId': MAIN,
                   'message': {'content': [{'type': 'text', 'text': 'ACCESS_CANARY'}]}}
    locator.write_bytes(initial + (json.dumps(unsupported) + '\n').encode())
    page = read_transcript_page(storage, (str(root),), f'claude:{MAIN}')
    assert "TextBlock(text='latest')" in repr(page.transcript)
    assert 'ACCESS_CANARY' not in repr(page.transcript)
    assert any(isinstance(entry, Notice) and entry.label == 'Unsupported Claude entry'
               for entry in page.transcript.entries)


def test_internal_connector_cannot_supply_visible_tool_output() -> None:
    records = [
        {'type': 'assistant', 'uuid': 'tool', 'parentUuid': None, 'sessionId': MAIN,
         'message': {'content': [{'type': 'tool_use', 'id': 'call', 'name': 'Read', 'input': {}}]}},
        {'type': 'system', 'uuid': 'internal', 'parentUuid': 'tool', 'sessionId': MAIN,
         'message': {'content': [{'type': 'tool_result', 'tool_use_id': 'call',
                                  'content': 'INTERNAL_TOOL_SECRET'}]}},
        {'type': 'user', 'uuid': 'leaf', 'parentUuid': 'internal', 'sessionId': MAIN,
         'message': {'content': [{'type': 'text', 'text': 'continue'}]}},
    ]
    transcript = parse_claude_transcript(
        ''.join(json.dumps(record) + '\n' for record in records).encode(), MAIN)
    tool = next(block for entry in transcript.entries if isinstance(entry, Message)
                for block in entry.blocks if isinstance(block, ToolBlock))
    assert tool.output is None
    assert 'INTERNAL_TOOL_SECRET' not in repr(transcript)
