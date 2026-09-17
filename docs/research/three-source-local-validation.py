"""Aggregate-only local qualification; review the adjacent contract before use."""
from __future__ import annotations

import argparse
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
import tempfile
import time

PROFILES = ('vscode-stable', 'vscode-insiders', 'copilot-cli')
HARNESSES = ('copilot-vscode', 'copilot-cli')
STATES = ('known', 'unknown', 'not_applicable')
TOKENS = ('input', 'output', 'cache_read', 'cache_write', 'reported_total', 'reasoning', 'cache_write_1h')
UNITS = ('ai_credits', 'nano_aiu', 'premium_requests', 'request_count')
DIAGNOSTICS = ('malformed_json', 'unsupported_profile', 'events_unavailable', 'lossy_turn_summary',
               'invalid_cli_event', 'invalid_cli_counter', 'invalid_event_chain', 'event_chain_gap',
               'duplicate_event_id', 'cli_partial_lifetime', 'workspace_changed_cumulative_scope', 'other')
FIELDS = ('promptTokens', 'completionTokens', 'copilotCredits', 'sessionCopilotCredits',
          'inputTokens', 'outputTokens', 'cachedTokens', 'cacheReadTokens', 'cacheWriteTokens',
          'reasoningTokens', 'totalNanoAiu', 'totalPremiumRequests')
COUNTS = ('files', 'sidecars', 'json_files', 'jsonl_files', 'metadata_only_files', 'malformed_records',
          'schema_v3_flat_files', 'flat_requests', 'flat_copilot_requests', 'flat_model_rows',
          'operation_records', 'initial_v3_records', 'cli_start_records', 'cli_schema1_producer_records',
          'cli_shutdown_records', 'cli_checkpoint_records', 'cli_model_rows')
ERRORS = ('invalid_arguments', 'unsafe_output', 'unsafe_source', 'read_failed', 'limit_exceeded',
          'import_failed', 'validation_failed')
QUALIFICATIONS = ('unavailable', 'not_locally_qualified', 'passed', 'failed', 'inconclusive', 'not_compared')
MAX_FILE = 64 * 1024 * 1024
MAX_FILES = 100_000


class ValidationError(Exception):
    pass


def empty_census():
    return {**dict.fromkeys(COUNTS, 0),
            'flat_fields': dict.fromkeys(FIELDS, 0),
            'operation_fields': dict.fromkeys(FIELDS, 0),
            'cli_fields': dict.fromkeys(FIELDS, 0)}


def empty_observed():
    return {'revision': 0, 'files_processed': 0, 'source_generations': 0, 'observations': 0,
            'sessions': dict.fromkeys(HARNESSES, 0),
            'tokens': {h: {m: dict.fromkeys(STATES, 0) for m in TOKENS} for h in HARNESSES},
            'quantities': {h: {m: {**dict.fromkeys(STATES, 0), 'lower_bound': 0} for m in UNITS} for h in HARNESSES},
            'decisions': dict.fromkeys(('selected', 'excluded', 'unresolved'), 0),
            'quantity_decisions': dict.fromkeys(('selected', 'excluded', 'unresolved'), 0),
            'evidence': dict.fromkeys(('model_totals', 'turn_summary', 'session_control', 'unavailable',
                                       'shutdown', 'usage_checkpoint', 'metadata'), 0),
            'diagnostics': dict.fromkeys(DIAGNOSTICS, 0),
            'metadata_sessions': 0, 'metadata_known_tokens': 0, 'metadata_known_quantities': 0}


def empty_result():
    return {'schema': 2, 'error': None, 'roots': dict.fromkeys(PROFILES, None),
            'versions': {'vscode-stable': {'app': None, 'extension': None},
                         'vscode-insiders': {'app': None, 'extension': None}, 'copilot-cli': {'cli': None}},
            'expected': {p: empty_census() for p in PROFILES},
            'observed': empty_observed(),
            'observed_diagnostics': {
                'oracle': 'production_replay', 'independence': 'not_independent',
                'comparison': 'not_compared', 'status': 'unavailable',
                'profiles': {p: {representation: {
                    'complete_files': 0, 'partial_files': 0, 'replay_failed_files': 0,
                    'participants': {participant: {
                        'final_requests': 0, 'direct_fields': dict.fromkeys(FIELDS, 0),
                        'nested_fields': dict.fromkeys(FIELDS, 0)}
                        for participant in ('recognized_copilot', 'other_or_missing_participant')}}
                    for representation in ('flat', 'operation_log')}
                    for p in ('vscode-stable', 'vscode-insiders')}},
            'qualification': {'stable': 'unavailable', 'restart': 'unavailable',
                              'unchanged_revision': 'unavailable', 'comparison': 'unavailable',
                              'file_census': 'unavailable',
                              'metadata_only_cli': 'unavailable', 'claude': 'not_locally_qualified'}}


def check_output(result):
    """Reject unexpected output keys/types; no input-derived strings are admitted."""
    def check(value, template, path=()):
        if isinstance(template, dict):
            if not isinstance(value, dict) or value.keys() != template.keys():
                raise ValidationError('validation_failed')
            for key in template:
                check(value[key], template[key], (*path, key))
        elif path == ('schema',):
            if type(value) is not int or value != 2:
                raise ValidationError('validation_failed')
        elif path == ('error',):
            if value is not None and value not in ERRORS:
                raise ValidationError('validation_failed')
        elif path[0] == 'roots':
            if value is not None and type(value) is not bool:
                raise ValidationError('validation_failed')
        elif path[0] == 'versions':
            if value is not None and (path[-1] != 'app' or not valid_version(value)):
                raise ValidationError('validation_failed')
        elif isinstance(template, str):
            allowed = (('unavailable', 'complete', 'partial', 'inconclusive') if path == ('observed_diagnostics', 'status')
                       else (template,) if path[0] == 'observed_diagnostics' else QUALIFICATIONS)
            if value not in allowed:
                raise ValidationError('validation_failed')
        elif type(value) is not int or value < 0:
            raise ValidationError('validation_failed')
    check(result, empty_result())


@contextmanager
def directory(path):
    """Pin every ancestor with openat and O_NOFOLLOW, including the root itself."""
    path = Path(os.path.abspath(path))
    descriptor = os.open('/', os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in path.parts[1:]:
            next_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_fd
        yield descriptor
    finally:
        os.close(descriptor)


def read_regular(path, limit=MAX_FILE):
    with directory(path.parent) as parent:
        descriptor = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode):
                raise ValidationError('unsafe_source')
            if before.st_size > limit:
                raise ValidationError('limit_exceeded')
            with os.fdopen(os.dup(descriptor), 'rb') as stream:
                data = stream.read(limit + 1)
            if len(data) > limit:
                raise ValidationError('limit_exceeded')
            after = os.fstat(descriptor)
            identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
            if identity(before) != identity(after):
                raise ValidationError('read_failed')
            return data, (*identity(after), hashlib.sha256(data).digest())
        finally:
            os.close(descriptor)


def names(path):
    with directory(path) as fd:
        result = []
        with os.scandir(fd) as entries:
            for entry in entries:
                if len(result) >= MAX_FILES:
                    raise ValidationError('limit_exceeded')
                metadata = entry.stat(follow_symlinks=False)
                if stat.S_ISLNK(metadata.st_mode):
                    raise ValidationError('unsafe_source')
                result.append((entry.name, stat.S_ISDIR(metadata.st_mode)))
        return sorted(result)


def read_sidecar(path):
    from harness_usage.source_input import MAX_PROBE_BYTES
    return read_regular(path, limit=MAX_PROBE_BYTES)


def candidates(roots):
    """Return primaries and optional sidecars; no recursion outside named shapes."""
    found, availability = [], {}
    for profile, root in roots.items():
        try:
            children = names(root)
        except FileNotFoundError:
            availability[profile] = False
            continue
        availability[profile] = True
        for child, is_dir in children:
            if not is_dir:
                continue
            folder = root / child
            contents = dict(names(folder))
            if profile == 'copilot-cli':
                sidecar = folder / 'workspace.yaml' if 'workspace.yaml' in contents else None
                if 'events.jsonl' in contents:
                    found.append((profile, folder / 'events.jsonl', sidecar))
                elif sidecar is not None:
                    found.append((profile, sidecar, None))
            elif contents.get('chatSessions'):
                sidecar = folder / 'workspace.json' if 'workspace.json' in contents else None
                for name, child_is_dir in names(folder / 'chatSessions'):
                    if not child_is_dir and Path(name).suffix in ('.json', '.jsonl'):
                        found.append((profile, folder / 'chatSessions' / name, sidecar))
            if len(found) > MAX_FILES:
                raise ValidationError('limit_exceeded')
    return found, availability


def snapshot(roots, *, census=None, observed_diagnostics=None):
    manifest, availability = candidates(roots)
    fingerprints = {}
    for profile, path, sidecar in manifest:
        data, fingerprints[path] = read_regular(path)
        if census is not None:
            raw_counts(data, path, census[profile])
        if observed_diagnostics is not None and profile in ('vscode-stable', 'vscode-insiders'):
            observed_request_fields(data, path, profile, observed_diagnostics)
        if sidecar is not None and sidecar not in fingerprints:
            _, fingerprints[sidecar] = read_sidecar(sidecar)
            if census is not None:
                census[profile]['sidecars'] += 1
    return manifest, availability, fingerprints


def observed_request_fields(data, path, profile, diagnostics):
    """Production-replayed field presence only; never an independent comparison."""
    from harness_usage.copilot_vscode_reader import replay_chat_v3
    representation = 'flat' if path.suffix == '.json' else 'operation_log'
    counts = diagnostics['profiles'][profile][representation]
    try:
        replayed = replay_chat_v3(data, representation=representation)
    except ValueError:
        counts['replay_failed_files'] += 1
        diagnostics['status'] = 'partial'
        return
    counts['partial_files' if replayed.pending_tail else 'complete_files'] += 1
    if replayed.pending_tail:
        diagnostics['status'] = 'partial'
    elif diagnostics['status'] == 'unavailable':
        diagnostics['status'] = 'complete'
    requests = replayed.value.get('requests')
    for request in requests if isinstance(requests, list) else []:
        if not isinstance(request, dict):
            continue
        agent = request.get('agent')
        extension = agent.get('extensionId') if isinstance(agent, dict) else None
        extension_id = extension.get('value') if isinstance(extension, dict) else None
        recognized = isinstance(extension_id, str) and extension_id.casefold() == 'github.copilot-chat'
        bucket = counts['participants']['recognized_copilot' if recognized else 'other_or_missing_participant']
        bucket['final_requests'] += 1
        count_fields(request, bucket['direct_fields'])
        stack = list(request.values())
        while stack:
            value = stack.pop()
            if isinstance(value, dict):
                count_fields(value, bucket['nested_fields'])
                stack.extend(value.values())
            elif isinstance(value, list):
                stack.extend(value)


def raw_counts(data, path, count):
    """Shape census only: never call production detection, replay, or readers."""
    count['files'] += 1
    if path.name == 'workspace.yaml':
        count['metadata_only_files'] += 1
        return
    flat = path.suffix == '.json'
    count['json_files' if flat else 'jsonl_files'] += 1
    records = [data] if flat else data.splitlines()
    for record in records:
        if not record.strip():
            continue
        try:
            row = json.loads(record)
        except (ValueError, UnicodeError, RecursionError):
            count['malformed_records'] += 1
            continue
        if not isinstance(row, dict):
            count['malformed_records'] += 1
            continue
        if flat:
            if type(row.get('version')) is int and row['version'] == 3:
                count['schema_v3_flat_files'] += 1
            for request in row.get('requests', []) if isinstance(row.get('requests'), list) else []:
                if not isinstance(request, dict):
                    continue
                count['flat_requests'] += 1
                agent = request.get('agent')
                ext = agent.get('extensionId') if isinstance(agent, dict) else None
                ext = ext.get('value') if isinstance(ext, dict) else ext
                count['flat_copilot_requests'] += int(isinstance(ext, str) and ext.lower() == 'github.copilot-chat')
                count_fields(request, count['flat_fields'])
                models = request.get('modelTotals', [])
                if isinstance(models, list):
                    for model in models:
                        if isinstance(model, dict):
                            count['flat_model_rows'] += 1
                            count_fields(model, count['flat_fields'])
        elif path.name == 'events.jsonl':
            payload = row.get('data')
            if not isinstance(payload, dict):
                continue
            kind = row.get('type')
            if kind == 'session.start':
                count['cli_start_records'] += 1
                count['cli_schema1_producer_records'] += int(type(payload.get('version')) is int and payload['version'] == 1 and payload.get('producer') in ('copilot-cli', 'github-copilot-cli'))
            if kind in ('session.shutdown', 'session.usage_checkpoint'):
                count['cli_shutdown_records' if kind == 'session.shutdown' else 'cli_checkpoint_records'] += 1
                count_fields(payload, count['cli_fields'])
                models = payload.get('modelMetrics')
                if isinstance(models, dict):
                    for model in models.values():
                        if isinstance(model, dict):
                            count['cli_model_rows'] += 1
                            if isinstance(model.get('usage'), dict):
                                count_fields(model['usage'], count['cli_fields'])
        else:
            if type(row.get('kind')) is int and row['kind'] in (0, 1, 2, 3):
                count['operation_records'] += 1
                value = row.get('v')
                if row['kind'] == 0 and isinstance(value, dict) and type(value.get('version')) is int and value['version'] == 3:
                    count['initial_v3_records'] += 1
                # Raw named-field occurrences, including set paths, never final replay fields.
                stack = [value]
                while stack:
                    item = stack.pop()
                    if isinstance(item, dict):
                        count_fields(item, count['operation_fields'])
                        stack.extend(item.values())
                    elif isinstance(item, list):
                        stack.extend(item)
                keys = row.get('k')
                if row['kind'] == 1 and isinstance(keys, list) and keys and keys[-1] in FIELDS:
                    count['operation_fields'][keys[-1]] += 1


def count_fields(value, counts):
    for field in FIELDS:
        counts[field] += int(field in value)


def valid_version(value):
    return isinstance(value, str) and re.fullmatch(r'(?:0|[1-9][0-9]{0,3})\.(?:0|[1-9][0-9]{0,3})\.(?:0|[1-9][0-9]{0,3})(?:-insider)?', value) is not None


def installed_versions(paths):
    result = empty_result()['versions']
    for profile in ('vscode-stable', 'vscode-insiders'):
        if profile not in paths:
            continue
        try:
            metadata = json.loads(read_regular(paths[profile], limit=65536)[0])
            version = metadata.get('version') if isinstance(metadata, dict) else None
            if valid_version(version):
                result[profile]['app'] = version
        except (OSError, ValueError, UnicodeError, RecursionError, ValidationError):
            pass
    return result


def fresh_output(requested):
    requested = Path(os.path.abspath(requested))
    try:
        with directory(requested.parent) as parent:
            try:
                existing = os.stat(requested.name, dir_fd=parent, follow_symlinks=False)
            except FileNotFoundError:
                os.mkdir(requested.name, 0o700, dir_fd=parent)
                return requested
            if not stat.S_ISDIR(existing.st_mode) or existing.st_uid != os.getuid() or existing.st_mode & 0o077:
                raise ValidationError('unsafe_output')
            return Path(tempfile.mkdtemp(prefix=requested.name + '-', dir=requested.parent))
    except OSError:
        raise ValidationError('unsafe_output') from None


def check_private(directory_path):
    for folder, children, files in os.walk(directory_path, followlinks=False):
        for path in (Path(folder), *(Path(folder) / name for name in children + files)):
            metadata = path.lstat()
            if stat.S_ISLNK(metadata.st_mode) or metadata.st_uid != os.getuid() or metadata.st_mode & 0o077:
                raise ValidationError('unsafe_output')


def controlled_application(output, roots):
    from harness_usage.application import Application
    from harness_usage.domain import Unassigned
    from harness_usage.source_input import SourcePayload, detect_source

    class BoundedApplication(Application):
        def get_roots(self):
            return tuple(str(root) for root in roots.values())

        def _scan(self, _roots):
            manifest, _ = candidates(roots)
            for profile, path, sidecar in manifest:
                data, _ = read_regular(path)
                context = () if sidecar is None else ((sidecar.name, read_sidecar(sidecar)[0]),)
                payload = SourcePayload(str(path), data, context)
                expected = 'copilot-cli' if profile == 'copilot-cli' else 'copilot-vscode'
                if detect_source(payload) not in (None, expected):
                    raise ValidationError('unsafe_source')
                yield payload

        def _assign_projects(self):
            # Recorded cwd must not cause Git execution or reads beyond approved roots.
            sessions, _ = self.storage.attribution_input()
            self.storage.save_attributions({session.id: Unassigned('local_validation_scope') for session in sessions})

    return BoundedApplication(output, timezone='UTC')


def observe(app):
    from harness_usage.copilot_cli_reader import WORKSPACE_PROFILE
    result = empty_observed()
    with app.storage.connect() as db:
        def count(sql, args=()):
            return int(db.execute(sql, args).one()[0])
        result['revision'] = count('SELECT revision FROM ledger_meta')
        result['files_processed'] = app.status().files_processed
        for key, table in (('source_generations', 'source_generation'), ('observations', 'observation')):
            result[key] = count('SELECT count(*) FROM ' + table)
        for harness in HARNESSES:
            result['sessions'][harness] = count('SELECT count(*) FROM session WHERE harness=?', (harness,))
            for table, measures, key in (('token_value', TOKENS, 'tokens'), ('quantity_value', UNITS, 'quantities')):
                for measure in measures:
                    for state in STATES:
                        result[key][harness][measure][state] = count('SELECT count(*) FROM ' + table + ' v JOIN observation o ON o.id=v.observation_id JOIN session s ON s.id=o.session_id WHERE s.harness=? AND v.measure=? AND v.state=?', (harness, measure, state))
                    if table == 'quantity_value':
                        result[key][harness][measure]['lower_bound'] = count('SELECT count(*) FROM quantity_value v JOIN observation o ON o.id=v.observation_id JOIN session s ON s.id=o.session_id WHERE s.harness=? AND v.measure=? AND v.lower_bound=1', (harness, measure))
        for table, key in (('decision', 'decisions'), ('quantity_decision', 'quantity_decisions')):
            for state in result[key]:
                result[key][state] = count('SELECT count(*) FROM ' + table + ' WHERE state=?', (state,))
        for kind in result['evidence']:
            table, column = ('copilot_cli_evidence', 'source_kind') if kind in ('shutdown', 'usage_checkpoint', 'metadata') else ('copilot_vscode_evidence', 'evidence_kind')
            result['evidence'][kind] = count('SELECT count(*) FROM ' + table + ' WHERE ' + column + '=?', (kind,))
        for code in DIAGNOSTICS[:-1]:
            result['diagnostics'][code] = count('SELECT count(*) FROM diagnostic WHERE code=?', (code,))
        result['diagnostics']['other'] = count('SELECT count(*) FROM diagnostic') - sum(result['diagnostics'].values())
        result['metadata_sessions'] = count("SELECT count(DISTINCT o.session_id) FROM copilot_cli_evidence e JOIN observation o ON o.id=e.observation_id WHERE e.source_kind='metadata' AND e.profile=?", (WORKSPACE_PROFILE,))
        for table, key in (('token_value', 'metadata_known_tokens'), ('quantity_value', 'metadata_known_quantities')):
            result[key] = count('SELECT count(*) FROM ' + table + " v WHERE v.state='known' AND v.observation_id IN (SELECT observation_id FROM copilot_cli_evidence WHERE source_kind='metadata' AND profile=?)", (WORKSPACE_PROFILE,))
    return result


def import_once(app):
    app.start_import()
    deadline = time.monotonic() + 300
    while app.status().state == 'running':
        if time.monotonic() > deadline:
            raise ValidationError('import_failed')
        time.sleep(0.02)
    if app.status().state != 'succeeded':
        raise ValidationError('import_failed')


def validate(requested, roots, *, metadata_paths=None):
    result = empty_result()
    mask = os.umask(0o077)
    app = None
    try:
        if not roots or any(profile not in PROFILES for profile in roots):
            raise ValidationError('invalid_arguments')
        output = fresh_output(requested)
        result['versions'] = installed_versions(metadata_paths or {})
        before = snapshot(roots, census=result['expected'], observed_diagnostics=result['observed_diagnostics'])
        result['roots'].update(before[1])
        app = controlled_application(output, roots)
        import_once(app)
        first = observe(app)
        app.close()
        app = controlled_application(output, roots)
        reopened = observe(app)
        import_once(app)
        repeated = observe(app)
        app.close()
        app = None
        after = snapshot(roots)
        check_private(output)
        result['observed'] = first
        stable = before == after
        if not stable:
            result['observed_diagnostics']['status'] = 'inconclusive'
        populated = any(c['files'] for c in result['expected'].values())
        qualified = result['qualification']
        qualified['stable'] = 'passed' if stable else 'inconclusive'
        qualified['restart'] = ('passed' if first == reopened else 'failed') if populated else 'unavailable'
        qualified['unchanged_revision'] = (('passed' if first == repeated else 'failed') if populated else 'unavailable') if stable else 'inconclusive'
        qualified['comparison'] = ('not_compared' if populated else 'unavailable') if stable else 'inconclusive'
        qualified['file_census'] = (('passed' if first['files_processed'] == sum(c['files'] for c in result['expected'].values()) else 'failed') if populated else 'unavailable') if stable else 'inconclusive'
        metadata = result['expected']['copilot-cli']['metadata_only_files']
        if metadata:
            qualified['metadata_only_cli'] = ('passed' if first['metadata_sessions'] == metadata and first['metadata_known_tokens'] == first['metadata_known_quantities'] == 0 else 'failed') if stable else 'inconclusive'
    except ValidationError as error:
        result['error'] = str(error) if str(error) in ERRORS else 'validation_failed'
    except OSError as error:
        result['error'] = 'unsafe_source' if error.errno in (20, 40, 62) else 'read_failed'
    except Exception:
        result['error'] = 'validation_failed'
    finally:
        if app is not None:
            try:
                app.close()
            except Exception:
                result['error'] = 'validation_failed'
        if 'output' in locals():
            try:
                check_private(output)
            except Exception:
                result['error'] = 'unsafe_output'
        if result['error'] is not None and 'before' in locals():
            for key in ('stable', 'comparison', 'file_census', 'unchanged_revision'):
                result['qualification'][key] = 'inconclusive'
        if result['error'] is not None:
            result['observed_diagnostics']['status'] = 'inconclusive'
        os.umask(mask)
    check_output(result)
    return result


def main(argv=None):
    class Parser(argparse.ArgumentParser):
        def error(self, _message):
            raise ValidationError('invalid_arguments')
    result = empty_result()
    try:
        parser = Parser(add_help=False)
        parser.add_argument('--data-dir', type=Path, required=True)
        parser.add_argument('--root-profile', action='append', choices=PROFILES, required=True)
        args = parser.parse_args(argv)
        home = Path.home()
        approved = {'vscode-stable': home / 'Library/Application Support/Code/User/workspaceStorage',
                    'vscode-insiders': home / 'Library/Application Support/Code - Insiders/User/workspaceStorage',
                    'copilot-cli': home / '.copilot/session-state'}
        # Suppress incidental library output without retaining private messages in memory/files.
        with open(os.devnull, 'w') as sink, redirect_stdout(sink), redirect_stderr(sink):
            metadata = {'vscode-stable': Path('/Applications/Visual Studio Code.app/Contents/Resources/app/package.json'),
                        'vscode-insiders': Path('/Applications/Visual Studio Code - Insiders.app/Contents/Resources/app/package.json')}
            result = validate(args.data_dir, {profile: approved[profile] for profile in args.root_profile},
                              metadata_paths={p: metadata[p] for p in args.root_profile if p in metadata})
    except ValidationError:
        result['error'] = 'invalid_arguments'
    except Exception:
        result['error'] = 'validation_failed'
    check_output(result)
    print(json.dumps(result, sort_keys=True, separators=(',', ':')))
    return 2 if result['error'] is not None else 0


if __name__ == '__main__':
    sys.exit(main())
