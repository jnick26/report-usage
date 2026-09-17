-- Application ledger schema v4.
PRAGMA foreign_keys = ON;
CREATE TABLE ledger_meta (
 singleton INTEGER PRIMARY KEY CHECK(singleton=1),
 schema_version INTEGER NOT NULL CHECK(schema_version=4),
 revision INTEGER NOT NULL CHECK(revision>=0)
) STRICT;
INSERT INTO ledger_meta VALUES(1,4,0);
CREATE TABLE import_run (
 id TEXT PRIMARY KEY NOT NULL,
 state TEXT NOT NULL CHECK(state IN ('running','succeeded','failed','interrupted')),
 started_us INTEGER NOT NULL, finished_us INTEGER,
 files_processed INTEGER NOT NULL DEFAULT 0 CHECK(files_processed>=0),
 revision INTEGER NOT NULL CHECK(revision>=0), error_code TEXT,
 CHECK((state='running' AND finished_us IS NULL AND error_code IS NULL)
    OR (state='succeeded' AND finished_us IS NOT NULL AND error_code IS NULL)
    OR (state IN ('failed','interrupted') AND finished_us IS NOT NULL AND error_code IS NOT NULL)),
 CHECK(finished_us IS NULL OR finished_us>=started_us)
) STRICT;
CREATE UNIQUE INDEX one_running_import ON import_run(state) WHERE state='running';
CREATE TABLE project (
 id TEXT PRIMARY KEY NOT NULL, identity TEXT NOT NULL UNIQUE,
 kind TEXT NOT NULL CHECK(kind IN ('repository','directory')), label TEXT NOT NULL
) STRICT;
CREATE TABLE session (
 id TEXT PRIMARY KEY NOT NULL, harness TEXT NOT NULL CHECK(harness IN ('pi','codex')),
 native_id TEXT NOT NULL, display_name TEXT, started_us INTEGER, last_seen_us INTEGER,
 title_excerpt TEXT CHECK(title_excerpt IS NULL OR length(title_excerpt) BETWEEN 1 AND 160),
 cwd TEXT, parent_locator TEXT, project_id TEXT REFERENCES project(id), worktree TEXT,
 attribution_reason TEXT NOT NULL,
 UNIQUE(harness,native_id),
 CHECK(project_id IS NULL OR worktree IS NOT NULL),
 CHECK(started_us IS NULL OR last_seen_us IS NULL OR last_seen_us>=started_us)
) STRICT;
CREATE TABLE source_generation (
 id TEXT PRIMARY KEY NOT NULL, locator TEXT NOT NULL,
 generation INTEGER NOT NULL CHECK(generation>=0), sha256 TEXT NOT NULL CHECK(length(sha256)=64),
 session_id TEXT REFERENCES session(id), profile TEXT NOT NULL,
 complete_bytes INTEGER NOT NULL CHECK(complete_bytes>=0),
 pending_tail INTEGER NOT NULL CHECK(pending_tail IN (0,1)),
 availability TEXT NOT NULL CHECK(availability IN ('available','missing','unreadable')),
 UNIQUE(locator,generation)
) STRICT;
CREATE TABLE observation (
 id TEXT PRIMARY KEY NOT NULL, session_id TEXT NOT NULL REFERENCES session(id),
 native_entry_id TEXT NOT NULL, fingerprint TEXT NOT NULL,
 kind TEXT NOT NULL CHECK(kind IN ('assistant','compaction','branch_summary','tool_result')),
 time_kind TEXT NOT NULL CHECK(time_kind IN ('point','interval','undated')),
 at_us INTEGER, start_us INTEGER, end_us INTEGER, time_reason TEXT,
 provider TEXT, model TEXT, stop_reason TEXT, tool_call_id TEXT,
 safe_facts_json TEXT NOT NULL CHECK(json_valid(safe_facts_json)),
 UNIQUE(session_id,native_entry_id,fingerprint),
 CHECK((time_kind='point' AND at_us IS NOT NULL AND start_us IS NULL AND end_us IS NULL AND time_reason IS NOT NULL AND time_reason='response_recorded_at')
    OR (time_kind='interval' AND at_us IS NULL AND end_us IS NOT NULL AND time_reason IS NULL AND (start_us IS NULL OR start_us<end_us))
    OR (time_kind='undated' AND at_us IS NULL AND start_us IS NULL AND end_us IS NULL AND time_reason IS NOT NULL))
) STRICT;
CREATE TABLE appearance (
 source_id TEXT NOT NULL REFERENCES source_generation(id),
 line INTEGER NOT NULL CHECK(line>0), observation_id TEXT NOT NULL REFERENCES observation(id),
 PRIMARY KEY(source_id,line,observation_id)
) STRICT;
CREATE TABLE entry_edge (
 source_id TEXT NOT NULL REFERENCES source_generation(id), line INTEGER NOT NULL CHECK(line>0),
 native_entry_id TEXT NOT NULL, parent_entry_id TEXT,
 PRIMARY KEY(source_id,line)
) STRICT;
CREATE TABLE token_value (
 observation_id TEXT NOT NULL REFERENCES observation(id),
 measure TEXT NOT NULL CHECK(measure IN ('input','output','cache_read','cache_write','reported_total','reasoning','cache_write_1h')),
 state TEXT NOT NULL CHECK(state IN ('known','unknown','not_applicable')),
 amount INTEGER, reason TEXT,
 PRIMARY KEY(observation_id,measure),
 CHECK((state='known' AND amount IS NOT NULL AND amount>=0 AND reason IS NULL)
    OR (state IN ('unknown','not_applicable') AND amount IS NULL AND reason IS NOT NULL))
) STRICT;
CREATE TABLE recorded_estimate (
 observation_id TEXT PRIMARY KEY NOT NULL REFERENCES observation(id),
 state TEXT NOT NULL CHECK(state IN ('known','missing')),
 amount_decimal TEXT, currency TEXT, components_json TEXT, source_ref TEXT, reason TEXT,
 CHECK((state='known' AND amount_decimal IS NOT NULL AND currency IS NOT NULL AND currency='USD' AND components_json IS NOT NULL AND json_valid(components_json) AND source_ref IS NOT NULL AND reason IS NULL)
    OR (state='missing' AND amount_decimal IS NULL AND currency IS NULL AND components_json IS NULL AND source_ref IS NULL AND reason IS NOT NULL))
) STRICT;
CREATE TABLE decision (
 observation_id TEXT NOT NULL REFERENCES observation(id),
 measure TEXT NOT NULL CHECK(measure IN ('input','output','cache_read','cache_write','total','recorded_usd')),
 state TEXT NOT NULL CHECK(state IN ('selected','excluded','unresolved')),
 owner_session TEXT REFERENCES session(id), canonical TEXT REFERENCES observation(id),
 reason TEXT NOT NULL, rule_version TEXT NOT NULL,
 PRIMARY KEY(observation_id,measure),
 CHECK((state='selected' AND owner_session IS NOT NULL AND canonical IS NULL)
    OR (state='excluded' AND owner_session IS NULL AND canonical IS NOT NULL AND canonical<>observation_id)
    OR (state='unresolved' AND owner_session IS NULL AND canonical IS NULL))
) STRICT;
CREATE TABLE diagnostic (
 id INTEGER PRIMARY KEY, source_id TEXT NOT NULL REFERENCES source_generation(id),
 observation_id TEXT REFERENCES observation(id), line INTEGER CHECK(line>0),
 code TEXT NOT NULL, measure TEXT
) STRICT;
CREATE INDEX decision_nonexcluded ON decision(observation_id) WHERE state<>'excluded';
CREATE INDEX observation_session ON observation(session_id);
CREATE INDEX observation_time ON observation(at_us) WHERE time_kind='point';
CREATE INDEX source_session ON source_generation(session_id);
CREATE TABLE source_metadata (
 source_id TEXT PRIMARY KEY NOT NULL REFERENCES source_generation(id),
 complete_sha256 TEXT NOT NULL CHECK(length(complete_sha256)=64)
) STRICT;
CREATE INDEX appearance_observation ON appearance(observation_id);

-- Delegation ownership metadata, independent of accounting observations.
CREATE TABLE delegation_ref (
 source_id TEXT NOT NULL REFERENCES source_generation(id), entry_id TEXT NOT NULL,
 kind TEXT NOT NULL CHECK(kind IN ('child_path','child_id','child_path_hash','child_name','agent_id')),
 value TEXT NOT NULL CHECK(length(value)>0), owner_path TEXT NOT NULL,
 PRIMARY KEY(source_id,entry_id,kind,value,owner_path)
) STRICT;
CREATE TABLE delegation_scan (
 source_id TEXT PRIMARY KEY NOT NULL REFERENCES source_generation(id), profile TEXT NOT NULL,
 path_hash TEXT NOT NULL CHECK(length(path_hash)=64)
) STRICT;

-- Codex source accounting evidence.
CREATE TABLE codex_source (
 source_id TEXT PRIMARY KEY NOT NULL REFERENCES source_generation(id),
 parent_thread_id TEXT, forked_from_id TEXT, root_session_id TEXT
) STRICT;
CREATE TABLE codex_evidence (
 source_id TEXT NOT NULL REFERENCES source_generation(id),
 line INTEGER NOT NULL CHECK(line>0),
 observation_id TEXT NOT NULL REFERENCES observation(id),
 source_kind TEXT NOT NULL CHECK(source_kind IN ('response','legacy')),
 response_id TEXT, thread_id TEXT, turn_id TEXT, mirror_response_id TEXT,
 cumulative_json TEXT CHECK(cumulative_json IS NULL OR json_valid(cumulative_json)),
 last_json TEXT CHECK(last_json IS NULL OR json_valid(last_json)),
 state TEXT NOT NULL CHECK(state IN ('usable','unresolved')), reason TEXT,
 PRIMARY KEY(source_id,line,observation_id),
 CHECK((state='usable' AND reason IS NULL) OR (state='unresolved' AND reason IS NOT NULL))
) STRICT;
CREATE INDEX codex_response ON codex_evidence(response_id) WHERE response_id IS NOT NULL;
CREATE INDEX codex_observation ON codex_evidence(observation_id);
CREATE TABLE codex_title (
 session_id TEXT PRIMARY KEY NOT NULL REFERENCES session(id),
 title TEXT NOT NULL CHECK(length(title) BETWEEN 1 AND 160)
) STRICT;
