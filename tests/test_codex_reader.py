from dataclasses import replace
from datetime import UTC, datetime
import json

from harness_usage.codex_reader import CodexReadBatch, read_codex
from harness_usage.domain import Known, Unknown, ModelIdentity, Point, MissingEstimate
from harness_usage.pi_reader import RejectedSource

HEADER={'timestamp':'2026-09-12T00:00:00Z','type':'session_meta','payload':{'id':'main','session_id':'root','timestamp':'2026-09-12T00:00:00Z','cwd':'/project','model_provider':'openai','cli_version':'0.153.4'}}


def source(*rows, header=HEADER):
    return ('\n'.join(json.dumps(row) for row in (header,*rows))+'\n').encode()


def usage(input=100,cached=30,write=10,output=20,reasoning=5):
    return dict(input_tokens=input,cached_input_tokens=cached,cache_write_input_tokens=write,output_tokens=output,reasoning_output_tokens=reasoning,total_tokens=input+output)


def context(model='gpt-5.6-sol',turn='turn'):
    return {'timestamp':'2026-09-12T01:00:00Z','type':'turn_context','payload':{'turn_id':turn,'model':model,'cwd':'/project'}}


def modern(identity='response',vector=None,total=None,thread='main'):
    vector=usage() if vector is None else vector
    return {'timestamp':'2026-09-12T02:00:00Z','type':'token_usage_record','payload':{'thread_id':thread,'turn_id':'turn','response_id':identity,'session_id':'root','root_turn_id':'root-turn','usage':vector,'thread_token_usage':total or vector,'turn_token_usage':usage(input=9999)}}


def legacy(last=None,total=None):
    last=usage() if last is None else last
    return {'timestamp':'2026-09-12T02:00:01Z','type':'event_msg','payload':{'type':'token_count','info':{'last_token_usage':last,'total_token_usage':total or last}}}


def read(data):
    result=read_codex(data,locator='/active/session.jsonl')
    assert isinstance(result,CodexReadBatch)
    return result


def test_modern_delta_identity_context_and_reasoning_are_not_double_counted():
    result=read(source(context(),modern()))
    record=result.usage[0]
    assert result.session.id=='codex:main' and result.root_session_id=='root'
    assert record.entry.native_id=='codex-response:response'
    assert record.model==ModelIdentity('openai','gpt-5.6-sol')
    assert record.tokens.buckets.values==(Known(60),Known(20),Known(30),Known(10))
    assert record.tokens.reasoning==Known(5) and record.tokens.total==Known(120)
    assert isinstance(record.money,MissingEstimate)
    assert result.evidence[0].thread_id=='main' and result.evidence[0].source_kind=='response'
    archived=read_codex(source(context(),modern()),locator='/archived/copy.jsonl')
    assert archived.usage==result.usage


def test_missing_fields_stay_unknown_and_invalid_subsets_do_not_crash():
    fields=usage();del fields['cache_write_input_tokens']
    result=read(source(context(),modern(vector=fields)))
    assert isinstance(result.usage[0].tokens.buckets.input,Unknown)
    assert isinstance(result.usage[0].tokens.buckets.cache_write,Unknown)
    assert 'cache_write_input_tokens' not in dict(result.evidence[0].last)
    invalid=usage(cached=110,reasoning=21)
    result=read(source(modern(vector=invalid)))
    assert isinstance(result.usage[0].tokens.buckets.input,Unknown)
    assert isinstance(result.usage[0].tokens.reasoning,Unknown)
    assert result.diagnostics


def test_legacy_cumulative_advancement_repeated_snapshots_and_replay():
    one=usage();two={key:value*2 for key,value in one.items()};three={key:value*3 for key,value in one.items()}
    result=read(source(context(),legacy(one,one),legacy(one,one),legacy(one,two),legacy(one,one),legacy(one,three)))
    assert len(result.usage)==3 and all(item.state=='usable' for item in result.evidence)
    assert sum(record.tokens.total.value for record in result.usage)==360
    reordered=read(source(context(),legacy(one,one),legacy(one,two),legacy(one,three)))
    assert [row.entry.native_id for row in reordered.usage]==[row.entry.native_id for row in result.usage]


def test_missing_regressing_or_inconsistent_legacy_counters_remain_unresolved():
    first=usage();regressed=usage(input=50,cached=10,write=0,output=5,reasoning=0)
    result=read(source(context(),legacy(first,first),legacy(regressed,regressed)))
    assert result.evidence[-1].state=='unresolved'
    broken=legacy();broken['payload']['info'].pop('total_token_usage')
    result=read(source(broken))
    assert len(result.usage)==1 and result.evidence[0].state=='unresolved'


def test_modern_legacy_mirror_is_retained_as_proven_overlap_and_earlier_usage_survives():
    vector=usage();twice={key:value*2 for key,value in vector.items()}
    result=read(source(context(),legacy(vector,vector),modern(total=twice),legacy(vector,twice)))
    assert len(result.usage)==3
    assert result.evidence[0].source_kind=='legacy' and result.evidence[0].state=='usable'
    assert result.evidence[-1].reason=='modern_legacy_overlap'
    assert result.evidence[-1].mirror_response_id=='response'


def test_context_only_events_and_rate_limit_status_are_not_usage():
    fill=usage(0,0,0,0,0);fill['total_tokens']=80000
    empty={'timestamp':'2026-09-12T01:00:00Z','type':'event_msg','payload':{'type':'token_count','info':None,'rate_limits':{'PRIVATE':'PRIVATE'}}}
    result=read(source(legacy(fill,fill),empty))
    assert result.usage==() and result.evidence==()
    assert 'PRIVATE' not in repr(result)


def test_first_header_is_canonical_and_privacy_bounded_excerpt_only():
    copied={**HEADER,'payload':{**HEADER['payload'],'id':'copied','cwd':'/different'}}
    user={'timestamp':'2026-09-12T01:00:00Z','type':'event_msg','payload':{'type':'user_message','message':'A user request '+('x'*200)}}
    hidden={'type':'response_item','payload':{'type':'message','role':'assistant','content':[{'type':'output_text','text':'PRIVATE RESPONSE'}]}}
    result=read(source(copied,user,hidden,context(),modern()))
    assert result.session.native_id=='main' and result.session.cwd=='/project'
    assert len(result.session.title_excerpt)==160 and result.session.title_excerpt.endswith('…')
    assert 'PRIVATE RESPONSE' not in repr(result)


def test_valid_final_record_incomplete_tail_and_bad_header():
    complete=source(context(),modern())
    assert len(read(complete[:-1]).usage)==1
    pending=read(complete+b'{"type":"token_usage_record"')
    assert pending.pending_tail and pending.complete_bytes==len(complete)
    assert isinstance(read_codex(b'{"type":"session_meta","payload":{"id":null}}',locator='x'),RejectedSource)


def test_applied_settings_are_scoped_to_the_explicit_thread_and_update_model():
    settings={'timestamp':'2026-09-12T01:30:00Z','type':'event_msg','payload':{'type':'thread_settings_applied','thread_id':'main','thread_settings':{'model':'gpt-5.6-terra','model_provider_id':'openai'}}}
    result=read(source(context(),settings,modern()))
    assert result.usage[0].model==ModelIdentity('openai','gpt-5.6-terra')
    foreign=read(source(context(),modern(thread='parent')))
    assert foreign.usage[0].model.provider is None
    assert foreign.usage[0].model.model=='gpt-5.6-sol'


def test_explicit_user_message_excerpt_precedes_model_input_scaffolding():
    scaffold={'type':'response_item','payload':{'type':'message','role':'user','content':[{'type':'input_text','text':'Environment context scaffolding'}]}}
    request={'type':'event_msg','payload':{'type':'user_message','message':'Actual user request'}}
    result=read(source(scaffold,request))
    assert result.session.title_excerpt=='Actual user request'


def test_legacy_fallback_skips_known_metadata_only_user_messages():
    environment='<environment_context>\nWorking directory: /project\n</environment_context>'
    plugins='<recommended_plugins>\nAvailable plugins\n</recommended_plugins>'
    agents='# AGENTS.md instructions for /project\n\n<INSTRUCTIONS>\nProject guidance\n</INSTRUCTIONS>'
    request='Review the parser '+('x'*200)
    def message(text):
        return {'type':'response_item','payload':{'type':'message','role':'user','content':[{'type':'input_text','text':text}]}}
    for scaffold in (environment,plugins,agents,plugins+'\n'+environment,plugins+'\n'+environment+'\n'+agents):
        result=read(source(message(scaffold),message(request)))
        assert result.session.title_excerpt==request[:159]+'…'
    for scaffold in (environment,agents):
        mixed=scaffold+'\nReview the parser'
        assert read(source(message(mixed))).session.title_excerpt==' '.join(mixed.split())


def test_duplicate_response_variants_survive_for_conflict_reconciliation():
    result=read(source(context(),modern(),context(model='gpt-5.6-terra',turn='other'),modern()))
    assert len(result.usage)==len(result.evidence)==2
    assert result.usage[0].entry.native_id==result.usage[1].entry.native_id


def test_delayed_response_uses_its_turn_and_explicit_owner_provider():
    settings={'type':'event_msg','payload':{'type':'thread_settings_applied','thread_id':'parent','thread_settings':{'model':'gpt-5.6-sol','model_provider_id':'custom-provider'}}}
    result=read(source(settings,context(),context(model='gpt-5.6-terra',turn='next'),modern(thread='parent')))
    assert result.usage[0].model==ModelIdentity('custom-provider','gpt-5.6-sol')
    assert read(source(modern(),context())).usage[0].model==ModelIdentity('openai','gpt-5.6-sol')


def test_sanitized_mixed_fixture_matches_independent_expected_values():
    from pathlib import Path
    folder=Path(__file__).parent/'fixtures'/'codex'
    expected=json.loads((folder/'mixed.expected.json').read_text())
    result=read((folder/'mixed.jsonl').read_bytes())
    assert result.parent_thread_id==expected['parent_thread_id']
    assert result.forked_from_id==expected['forked_from_id']
    assert [tuple(value.value for value in row.tokens.buckets.values) for row in result.usage]==[tuple(values) for values in expected['buckets']]
    assert [item.reason for item in result.evidence]==expected['reasons']
