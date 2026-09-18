-- Application DuckDB ledger schema v7.
CREATE SEQUENCE ledger_order START 1;

CREATE TABLE ledger_meta (
 singleton BIGINT PRIMARY KEY CHECK(singleton=1),
 schema_version BIGINT NOT NULL CHECK(schema_version=7),
 revision BIGINT NOT NULL CHECK(revision>=0)
);
INSERT INTO ledger_meta VALUES(1,7,0);
CREATE TABLE import_run (
 id TEXT PRIMARY KEY NOT NULL,
 state TEXT NOT NULL CHECK(state IN ('running','succeeded','failed','interrupted')),
 started_us BIGINT NOT NULL, finished_us BIGINT,
 files_processed BIGINT NOT NULL DEFAULT 0 CHECK(files_processed>=0),
 revision BIGINT NOT NULL CHECK(revision>=0), error_code TEXT,
 CHECK((state='running' AND finished_us IS NULL AND error_code IS NULL)
    OR (state='succeeded' AND finished_us IS NOT NULL AND error_code IS NULL)
    OR (state IN ('failed','interrupted') AND finished_us IS NOT NULL AND error_code IS NOT NULL)),
 CHECK(finished_us IS NULL OR finished_us>=started_us),
 ordinal BIGINT NOT NULL DEFAULT nextval('ledger_order')
);
CREATE UNIQUE INDEX one_running_import ON import_run ((CASE WHEN state='running' THEN 1 ELSE NULL END));
CREATE TABLE project (
 id TEXT PRIMARY KEY NOT NULL, identity TEXT NOT NULL UNIQUE,
 kind TEXT NOT NULL CHECK(kind IN ('repository','directory')), label TEXT NOT NULL
);
CREATE TABLE session (
 id TEXT PRIMARY KEY NOT NULL, harness TEXT NOT NULL CHECK(harness IN ('pi','codex','claude','copilot-vscode','copilot-cli')),
 native_id TEXT NOT NULL, display_name TEXT, started_us BIGINT, last_seen_us BIGINT,
 title_excerpt TEXT CHECK(title_excerpt IS NULL OR length(title_excerpt) BETWEEN 1 AND 160),
 cwd TEXT, parent_locator TEXT,
 UNIQUE(harness,native_id),
 CHECK(started_us IS NULL OR last_seen_us IS NULL OR last_seen_us>=started_us)
);
CREATE TABLE session_attribution (
 session_id TEXT PRIMARY KEY REFERENCES session(id), project_id TEXT REFERENCES project(id), worktree TEXT,
 attribution_reason TEXT NOT NULL,
 CHECK(project_id IS NULL OR worktree IS NOT NULL)
);
CREATE VIEW session_view AS SELECT s.*,a.project_id,a.worktree,a.attribution_reason
 FROM session s JOIN session_attribution a ON a.session_id=s.id;
CREATE TABLE source_generation (
 id TEXT PRIMARY KEY NOT NULL, locator TEXT NOT NULL,
 generation BIGINT NOT NULL CHECK(generation>=0), sha256 TEXT NOT NULL CHECK(length(sha256)=64),
 session_id TEXT REFERENCES session(id), profile TEXT NOT NULL,
 complete_bytes BIGINT NOT NULL CHECK(complete_bytes>=0),
 pending_tail BIGINT NOT NULL CHECK(pending_tail IN (0,1)),
 availability TEXT NOT NULL CHECK(availability IN ('available','missing','unreadable')),
 UNIQUE(locator,generation)
);
CREATE TABLE observation (
 id TEXT PRIMARY KEY NOT NULL, session_id TEXT NOT NULL REFERENCES session(id),
 native_entry_id TEXT NOT NULL, fingerprint TEXT NOT NULL,
 kind TEXT NOT NULL CHECK(kind IN ('assistant','compaction','branch_summary','tool_result','request_summary','usage_checkpoint')),
 time_kind TEXT NOT NULL CHECK(time_kind IN ('point','interval','undated')),
 at_us BIGINT, start_us BIGINT, end_us BIGINT, time_reason TEXT,
 provider TEXT, model TEXT, stop_reason TEXT, tool_call_id TEXT,
 safe_facts_json TEXT NOT NULL CHECK(json_valid(safe_facts_json)),
 UNIQUE(session_id,native_entry_id,fingerprint),
 CHECK((time_kind='point' AND at_us IS NOT NULL AND start_us IS NULL AND end_us IS NULL AND time_reason IS NOT NULL AND time_reason='response_recorded_at')
    OR (time_kind='interval' AND at_us IS NULL AND end_us IS NOT NULL AND time_reason IS NULL AND (start_us IS NULL OR start_us<end_us))
    OR (time_kind='undated' AND at_us IS NULL AND start_us IS NULL AND end_us IS NULL AND time_reason IS NOT NULL))
);
CREATE TABLE appearance (
 source_id TEXT NOT NULL REFERENCES source_generation(id),
 line BIGINT NOT NULL CHECK(line>0), observation_id TEXT NOT NULL REFERENCES observation(id),
 PRIMARY KEY(source_id,line,observation_id),
 ordinal BIGINT NOT NULL DEFAULT nextval('ledger_order')
);
CREATE TABLE entry_edge (
 source_id TEXT NOT NULL REFERENCES source_generation(id), line BIGINT NOT NULL CHECK(line>0),
 native_entry_id TEXT NOT NULL, parent_entry_id TEXT,
 PRIMARY KEY(source_id,line)
);
CREATE TABLE token_value (
 observation_id TEXT NOT NULL REFERENCES observation(id),
 measure TEXT NOT NULL CHECK(measure IN ('input','output','cache_read','cache_write','reported_total','reasoning','cache_write_1h')),
 state TEXT NOT NULL CHECK(state IN ('known','unknown','not_applicable')),
 amount BIGINT, reason TEXT,
 PRIMARY KEY(observation_id,measure),
 CHECK((state='known' AND amount IS NOT NULL AND amount>=0 AND reason IS NULL)
    OR (state IN ('unknown','not_applicable') AND amount IS NULL AND reason IS NOT NULL))
);
CREATE TABLE recorded_estimate (
 observation_id TEXT PRIMARY KEY NOT NULL REFERENCES observation(id),
 state TEXT NOT NULL CHECK(state IN ('known','missing')),
 amount_decimal TEXT, currency TEXT, components_json TEXT, source_ref TEXT, reason TEXT,
 CHECK((state='known' AND amount_decimal IS NOT NULL AND currency IS NOT NULL AND currency='USD' AND components_json IS NOT NULL AND json_valid(components_json) AND source_ref IS NOT NULL AND reason IS NULL)
    OR (state='missing' AND amount_decimal IS NULL AND currency IS NULL AND components_json IS NULL AND source_ref IS NULL AND reason IS NOT NULL))
);
CREATE TABLE quantity_value (
 observation_id TEXT NOT NULL REFERENCES observation(id),
 measure TEXT NOT NULL CHECK(measure IN ('ai_credits','nano_aiu','premium_requests','request_count')),
 state TEXT NOT NULL CHECK(state IN ('known','unknown','not_applicable')),
 amount_decimal TEXT, reason TEXT,
 lower_bound BIGINT NOT NULL CHECK(lower_bound IN (0,1)), source_ref TEXT,
 PRIMARY KEY(observation_id,measure),
 CHECK((state='known' AND amount_decimal IS NOT NULL
        AND regexp_full_match(amount_decimal,'(?:0|[1-9][0-9]*)(?:[.][0-9]+)?(?:[eE][+-]?[0-9]+)?')
        AND reason IS NULL AND source_ref IS NOT NULL AND length(source_ref)>0)
    OR (state IN ('unknown','not_applicable') AND amount_decimal IS NULL AND reason IS NOT NULL
        AND length(reason)>0 AND lower_bound=0 AND source_ref IS NULL))
);
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
);
CREATE TABLE quantity_decision (
 observation_id TEXT NOT NULL REFERENCES observation(id),
 measure TEXT NOT NULL CHECK(measure IN ('ai_credits','nano_aiu','premium_requests','request_count')),
 state TEXT NOT NULL CHECK(state IN ('selected','excluded','unresolved')),
 owner_session TEXT REFERENCES session(id), canonical TEXT REFERENCES observation(id),
 reason TEXT NOT NULL, rule_version TEXT NOT NULL,
 PRIMARY KEY(observation_id,measure),
 CHECK((state='selected' AND owner_session IS NOT NULL AND canonical IS NULL)
    OR (state='excluded' AND owner_session IS NULL AND canonical IS NOT NULL AND canonical<>observation_id)
    OR (state='unresolved' AND owner_session IS NULL AND canonical IS NULL))
);
CREATE TABLE diagnostic (
 id BIGINT PRIMARY KEY DEFAULT nextval('ledger_order'), source_id TEXT NOT NULL REFERENCES source_generation(id),
 observation_id TEXT REFERENCES observation(id), line BIGINT CHECK(line>0),
 code TEXT NOT NULL, measure TEXT
);
CREATE INDEX decision_nonexcluded ON decision(observation_id);
CREATE INDEX observation_session ON observation(session_id);
CREATE INDEX observation_time ON observation(at_us);
CREATE INDEX source_session ON source_generation(session_id);
CREATE TABLE source_metadata (
 source_id TEXT PRIMARY KEY NOT NULL REFERENCES source_generation(id),
 complete_sha256 TEXT NOT NULL CHECK(length(complete_sha256)=64)
);
CREATE INDEX appearance_observation ON appearance(observation_id);

-- Delegation ownership metadata, independent of accounting observations.
CREATE TABLE delegation_ref (
 source_id TEXT NOT NULL REFERENCES source_generation(id), entry_id TEXT NOT NULL,
 kind TEXT NOT NULL CHECK(kind IN ('child_path','child_id','child_path_hash','child_name','agent_id')),
 value TEXT NOT NULL CHECK(length(value)>0), owner_path TEXT NOT NULL,
 PRIMARY KEY(source_id,entry_id,kind,value,owner_path)
);
CREATE TABLE delegation_scan (
 source_id TEXT PRIMARY KEY NOT NULL REFERENCES source_generation(id), profile TEXT NOT NULL,
 path_hash TEXT NOT NULL CHECK(length(path_hash)=64)
);

-- Codex source accounting evidence.
CREATE TABLE codex_source (
 source_id TEXT PRIMARY KEY NOT NULL REFERENCES source_generation(id),
 parent_thread_id TEXT, forked_from_id TEXT, root_session_id TEXT
);
CREATE TABLE codex_evidence (
 source_id TEXT NOT NULL REFERENCES source_generation(id),
 line BIGINT NOT NULL CHECK(line>0),
 observation_id TEXT NOT NULL REFERENCES observation(id),
 source_kind TEXT NOT NULL CHECK(source_kind IN ('response','legacy')),
 response_id TEXT, thread_id TEXT, turn_id TEXT, mirror_response_id TEXT,
 cumulative_json TEXT CHECK(cumulative_json IS NULL OR json_valid(cumulative_json)),
 last_json TEXT CHECK(last_json IS NULL OR json_valid(last_json)),
 state TEXT NOT NULL CHECK(state IN ('usable','unresolved')), reason TEXT,
 PRIMARY KEY(source_id,line,observation_id),
 CHECK((state='usable' AND reason IS NULL) OR (state='unresolved' AND reason IS NOT NULL)),
 ordinal BIGINT NOT NULL DEFAULT nextval('ledger_order')
);
CREATE INDEX codex_response ON codex_evidence(response_id);
CREATE INDEX codex_observation ON codex_evidence(observation_id);
CREATE TABLE codex_title (
 session_id TEXT PRIMARY KEY NOT NULL REFERENCES session(id),
 title TEXT NOT NULL CHECK(length(title) BETWEEN 1 AND 160)
);

-- Privacy-bounded source evidence for schema-v6 harnesses.
CREATE TABLE claude_evidence (
 source_id TEXT NOT NULL REFERENCES source_generation(id), line BIGINT NOT NULL CHECK(line>0),
 observation_id TEXT NOT NULL REFERENCES observation(id), entry_id TEXT NOT NULL,
 message_id TEXT, request_id TEXT, entry_uuid TEXT NOT NULL, agent_id TEXT,
 state TEXT NOT NULL CHECK(state IN ('usable','unresolved')), reason TEXT,
 PRIMARY KEY(source_id,line,observation_id),
 CHECK((state='usable' AND reason IS NULL) OR (state='unresolved' AND reason IS NOT NULL)),
 ordinal BIGINT NOT NULL DEFAULT nextval('ledger_order')
);
CREATE INDEX claude_message ON claude_evidence(message_id);
CREATE INDEX claude_request ON claude_evidence(request_id);
CREATE TABLE copilot_vscode_evidence (
 source_id TEXT NOT NULL REFERENCES source_generation(id), line BIGINT NOT NULL CHECK(line>0),
 observation_id TEXT NOT NULL REFERENCES observation(id), profile TEXT NOT NULL,
 session_native_id TEXT NOT NULL, request_id TEXT, response_id TEXT, selected_model TEXT,
 representation TEXT NOT NULL CHECK(representation IN ('flat','operation_log')),
 request_ordinal BIGINT NOT NULL CHECK(request_ordinal>=0),
 evidence_kind TEXT NOT NULL CHECK(evidence_kind IN ('model_totals','turn_summary','session_control','unavailable')),
 raw_input BIGINT CHECK(raw_input IS NULL OR raw_input>=0), raw_output BIGINT CHECK(raw_output IS NULL OR raw_output>=0),
 raw_cache_read BIGINT CHECK(raw_cache_read IS NULL OR raw_cache_read>=0), raw_cache_write BIGINT CHECK(raw_cache_write IS NULL OR raw_cache_write>=0),
 state TEXT NOT NULL CHECK(state IN ('usable','unresolved')), reason TEXT,
 PRIMARY KEY(source_id,line,observation_id),
 CHECK((state='usable' AND reason IS NULL) OR (state='unresolved' AND reason IS NOT NULL)),
 ordinal BIGINT NOT NULL DEFAULT nextval('ledger_order')
);
CREATE INDEX copilot_vscode_request ON copilot_vscode_evidence(session_native_id,request_id,response_id);
CREATE TABLE copilot_cli_evidence (
 source_id TEXT NOT NULL REFERENCES source_generation(id), line BIGINT NOT NULL CHECK(line>0),
 observation_id TEXT NOT NULL REFERENCES observation(id), profile TEXT NOT NULL,
 source_kind TEXT NOT NULL CHECK(source_kind IN ('shutdown','usage_checkpoint','metadata')),
 event_id TEXT, parent_event_id TEXT, event_schema_version TEXT, writer_version TEXT,
 session_start_us BIGINT, current_model TEXT, agent_id TEXT, counter_epoch TEXT,
 counters_json TEXT CHECK(counters_json IS NULL OR json_valid(counters_json)),
 state TEXT NOT NULL CHECK(state IN ('usable','unresolved')), reason TEXT,
 PRIMARY KEY(source_id,line,observation_id),
 CHECK((state='usable' AND reason IS NULL) OR (state='unresolved' AND reason IS NOT NULL)),
 ordinal BIGINT NOT NULL DEFAULT nextval('ledger_order')
);
CREATE INDEX copilot_cli_event ON copilot_cli_evidence(event_id);
CREATE INDEX copilot_cli_epoch ON copilot_cli_evidence(counter_epoch);

-- Copilot session-store provenance.
CREATE TABLE copilot_store_session (
 source_id TEXT NOT NULL REFERENCES source_generation(id),
 session_id TEXT NOT NULL REFERENCES session(id),
 created_us BIGINT, updated_us BIGINT, host_type TEXT,
 PRIMARY KEY(source_id,session_id)
);
CREATE TABLE copilot_store_evidence (
 source_id TEXT NOT NULL REFERENCES source_generation(id),
 row_id BIGINT, observation_id TEXT NOT NULL REFERENCES observation(id),
 turn_index BIGINT, agent_id TEXT, parent_tool_call_id TEXT,
 counters_json TEXT NOT NULL CHECK(json_valid(counters_json)),
 compatibility TEXT NOT NULL,
 PRIMARY KEY(source_id,observation_id)
);
