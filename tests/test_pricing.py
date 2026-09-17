from decimal import Decimal
import json
from dataclasses import replace

import pytest

from harness_usage.domain import Known, ModelIdentity, NotApplicable, TokenBreakdown, TokenEvidence, Unknown
from harness_usage.pricing import Catalog, aggregate


def catalog(cost, provider='openai', model='example'):
    return Catalog.from_bytes(json.dumps({provider:{'models':{model:{'cost':cost}}}}).encode(),snapshot_date='2026-09-12',sha256='fixture')


def tokens(input=100,output=20,cache_read=0,cache_write=0):
    return TokenEvidence(TokenBreakdown(*(Known(value) if isinstance(value,int) else value for value in (input,output,cache_read,cache_write))),Unknown('not_needed'))


SELECTED=tuple((category,'selected') for category in ('input','output','cache_read','cache_write'))


def test_exact_math_alias_and_selected_categories():
    rates=catalog({'input':0.1,'output':2,'cache_read':0.01})
    result=rates.price(ModelIdentity('openai-codex','example'),tokens(cache_read=300),SELECTED)
    assert result.known==Decimal('0.000053') and not result.unpriced
    assert {line.source_provider for line in result.calculations}=={'openai'}
    assert {(line.category,line.tokens,line.rate,line.subtotal) for line in result.calculations}=={
        ('input',100,Decimal('0.1'),Decimal('0.00001')),
        ('output',20,Decimal('2'),Decimal('0.00004')),
        ('cache_read',300,Decimal('0.01'),Decimal('0.000003'))}
    partial=rates.price(ModelIdentity('openai','example'),tokens(),(('input','excluded'),('output','selected'),('recorded_usd','excluded')))
    assert partial.known==Decimal('0.00004') and len(partial.calculations)==1


def test_missing_model_rate_and_unknown_tokens_remain_unpriced():
    rates=catalog({'input':1,'output':2})
    missing=rates.price(ModelIdentity('openai','typo'),tokens(),SELECTED)
    assert missing.known==0 and missing.unpriced
    assert {line.reason for line in missing.calculations}=={'model_not_in_catalog'}
    partial=rates.price(ModelIdentity('openai','example'),tokens(input=Unknown('not_recorded'),cache_write=10),SELECTED)
    assert partial.known==Decimal('0.00004') and partial.unpriced
    assert {line.reason for line in partial.calculations if line.subtotal is None}=={'unknown_tokens','missing_category_rate'}
    unresolved=rates.price(ModelIdentity('openai','example'),tokens(),(('input','unresolved'),('output','selected')))
    assert unresolved.known==Decimal('0.00004') and unresolved.unpriced


def test_modern_context_tiers_override_legacy_per_observation_and_at_boundary():
    rates=catalog({'input':1,'output':2,'cache_read':0.1,'cache_write':1.2,
                   'tiers':[{'tier':{'type':'context','size':200},'input':3,'output':4,'cache_read':0.3,'cache_write':3.6}],
                   'context_over_200k':{'input':999,'output':999}})
    base=rates.price(ModelIdentity('openai','example'),tokens(input=100,cache_read=100),SELECTED)
    high=rates.price(ModelIdentity('openai','example'),tokens(input=100,cache_read=101),SELECTED)
    assert {line.context_threshold for line in base.calculations}=={None}
    assert {line.context_threshold for line in high.calculations}=={200}
    assert next(line.rate for line in base.calculations if line.category=='output')==2
    assert next(line.rate for line in high.calculations if line.category=='output')==4
    assert all(line.rate!=999 for line in high.calculations)
    output_only=rates.price(ModelIdentity('openai','example'),tokens(input=201), (('input','excluded'),('output','selected')))
    assert output_only.calculations[0].rate==4
    unknown=rates.price(ModelIdentity('openai','example'),tokens(input=Unknown('missing')),SELECTED)
    assert unknown.known==0 and unknown.unpriced
    assert next(line.reason for line in unknown.calculations if line.category=='output')=='unknown_context_size'
    assert next(line.rate for line in unknown.calculations if line.category=='output') is None


def test_rate_precision_is_not_rounded_by_decimal_default_context():
    raw=b'{"openai":{"models":{"example":{"cost":{"input":0.123456789012345678901234567891}}}}}'
    rates=Catalog.from_bytes(raw,snapshot_date='2026-09-12',sha256='fixture')
    result=rates.price(ModelIdentity('openai','example'),tokens(input=1000000,output=0),SELECTED)
    assert result.known==Decimal('0.123456789012345678901234567891')


def test_legacy_tier_and_unknown_tier_dimensions_are_conservative():
    legacy=catalog({'input':1,'output':2,'context_over_200k':{'input':3,'output':4}})
    assert next(line.rate for line in legacy.price(ModelIdentity('openai','example'),tokens(input=200001),SELECTED).calculations if line.category=='input')==3
    invalid=catalog({'input':1,'output':2,'tiers':[{'tier':{'type':'region','size':100},'input':3}]})
    result=invalid.price(ModelIdentity('openai','example'),tokens(),SELECTED)
    assert result.known==0 and result.unpriced
    assert {line.reason for line in result.calculations}=={'unsupported_pricing_tier'}


def test_bundled_catalog_uses_current_thresholds_and_exact_model_ids():
    from harness_usage.pricing import load_bundled_catalog
    rates=load_bundled_catalog()
    result=rates.price(ModelIdentity('openai-codex','gpt-5.6-sol'),tokens(input=200000,output=1000,cache_read=100000),SELECTED)
    assert result.known==Decimal('1.71') and not result.unpriced
    assert {line.context_threshold for line in result.calculations}=={272000}
    below=rates.price(ModelIdentity('openai-codex','gpt-5.5'),tokens(input=200001,output=1000),SELECTED)
    assert below.known==Decimal('1.030005') and not below.unpriced
    minimax=rates.price(ModelIdentity('opencode-go','minimax-m3'),tokens(input=300000,output=1000),SELECTED)
    assert minimax.known==Decimal('0.0912') and not minimax.unpriced
    typo=rates.price(ModelIdentity('openai-codex','gpt-5.6-lunz'),tokens(),SELECTED)
    assert typo.unpriced and typo.known==0


def test_aggregate_usage_only_prices_context_invariant_categories():
    rates=catalog({'input':1,'output':2,'tiers':[{'tier':{'type':'context','size':200},'input':3,'output':2}]})
    result=rates.price(ModelIdentity('openai','example'),tokens(input=201),SELECTED,single_response=False)
    assert result.known==Decimal('0.00004') and result.unpriced
    assert next(line.reason for line in result.calculations if line.category=='input')=='aggregate_context'
    assert next(line.rate for line in result.calculations if line.category=='input') is None
    assert next(line.rate for line in result.calculations if line.category=='output')==2
    flat=catalog({'input':1,'output':2}).price(ModelIdentity('openai','example'),tokens(),SELECTED,single_response=False)
    assert flat.known==Decimal('0.00014') and not flat.unpriced


def test_positive_one_hour_cache_subset_is_unpriced_and_not_added_to_prompt_size():
    from dataclasses import replace
    rates=catalog({'input':1,'output':2,'cache_write':1.2,'tiers':[{'tier':{'type':'context','size':125},'input':3,'output':4,'cache_write':3.6}]})
    usage=replace(tokens(input=100,output=20,cache_write=20),cache_write_1h=Known(10))
    result=rates.price(ModelIdentity('openai','example'),usage,SELECTED)
    assert result.known==Decimal('0.00014') and result.unpriced
    writes=next(line for line in result.calculations if line.category=='cache_write')
    assert writes.subtotal is None and writes.rate is None and writes.reason=='unsupported_cache_write_duration'
    assert {line.context_threshold for line in result.calculations}=={None}


@pytest.mark.parametrize(('input_count','threshold','known'), [
    (150,None,'0.00038'), (201,200,'0.001286'), (401,400,'0.00417')])
def test_group_pricing_keeps_original_response_tier_when_sum_crosses_boundary(input_count,threshold,known):
    rates=catalog({'input':1,'output':2,'tiers':[
        {'tier':{'type':'context','size':200},'input':3,'output':2},
        {'tier':{'type':'context','size':400},'input':5,'output':4}]})
    model=ModelIdentity('openai-codex','example')
    individual=rates.price(model,tokens(input=input_count),SELECTED)
    grouped=rates.price_group(model,(input_count*2,40,0,0),('selected',)*4,context_threshold=threshold)
    assert grouped.known==Decimal(known) and not grouped.unpriced
    expected=aggregate([individual,individual])
    assert (grouped.known,2*grouped.unpriced,aggregate([grouped])[2])==expected


@pytest.mark.parametrize(('case','known','reasons'), [
    ('unknown_context','0.00008',{'unknown_tokens'}),
    ('aggregate_context','0.00008',{'aggregate_context'}),
    ('cache_write_1h','0.00028',{'unsupported_cache_write_duration'}),
    ('missing_rate','0.00028',{'missing_category_rate'}),
    ('missing_model','0',{'model_not_in_catalog'}),
    ('invalid_tier','0',{'unsupported_pricing_tier'}),
    ('unresolved_zero','0.00028',{'unresolved_tokens'})])
def test_group_pricing_preserves_unpriced_categories_and_observation_weight(case,known,reasons):
    cost={'input':1,'output':2,'tiers':[{'tier':{'type':'context','size':200},'input':3,'output':2}]}
    model=ModelIdentity('openai','missing' if case=='missing_model' else 'example')
    usage=tokens(cache_write=10 if case in {'cache_write_1h','missing_rate'} else 0)
    counts=(200,40,0,20 if case in {'cache_write_1h','missing_rate'} else 0)
    states=('selected',)*4
    context_reason=None
    if case=='unknown_context':
        usage=tokens(input=Unknown('missing'))
        counts=(None,40,0,0)
        context_reason='unknown_context_size'
    elif case=='aggregate_context':
        context_reason='aggregate_context'
    elif case=='cache_write_1h':
        usage=replace(usage,cache_write_1h=Known(5))
    elif case=='invalid_tier':
        cost['tiers']=[{'tier':{'type':'region','size':200},'input':3}]
    elif case=='unresolved_zero':
        states=('selected','selected','unresolved','excluded')
        counts=(200,40,None,0)
    rates=catalog(cost)
    individual=rates.price(model,usage,tuple(zip(('input','output','cache_read','cache_write'),states)),single_response=case!='aggregate_context')
    grouped=rates.price_group(model,counts,states,context_reason=context_reason,cache_write_1h=case=='cache_write_1h')
    assert grouped.known==Decimal(known) and grouped.unpriced
    assert {line.reason for line in grouped.calculations if line.reason}==reasons
    assert (grouped.known,2*grouped.unpriced,aggregate([grouped])[2])==aggregate([individual,individual])


def test_group_pricing_skips_selected_not_applicable_and_keeps_large_exact_sums():
    rates=catalog({'input':1})
    model=ModelIdentity('openai','example')
    usage=tokens(input=2**62,output=NotApplicable('absent'),cache_read=Unknown('missing'))
    decisions=(('input','selected'),('output','selected'),('cache_read','excluded'),('cache_write','selected'))
    individual=rates.price(model,usage,decisions)
    grouped=rates.price_group(model,(2**64,0,None,0),tuple(state for _,state in decisions))
    assert grouped.known==Decimal('18446744073709.551616') and not grouped.unpriced
    assert len(grouped.calculations)==1
    assert (grouped.known,4*grouped.unpriced,aggregate([grouped])[2])==aggregate([individual]*4)


@pytest.mark.parametrize(('provider', 'official', 'name'), [
    ('claude-bridge', 'anthropic', 'claude-opus-5'),
    ('claude-bridge', 'anthropic', 'claude-sonnet-5'),
    ('claude-bridge', 'anthropic', 'claude-fable-5'),
    ('claude-bridge', 'anthropic', 'claude-sonnet-4-6'),
    (None, 'anthropic', 'claude-sonnet-4-6'),
    ('github-copilot', 'anthropic', 'claude-sonnet-4-6'),
    (None, 'openai', 'gpt-5.4'),
    ('github-copilot', 'openai', 'gpt-5.4'),
    (None, 'google', 'gemini-2.5-pro'),
    ('github-copilot', 'google', 'gemini-2.5-pro'),
])
def test_exact_official_fallback_preserves_identity_and_context_tier(provider, official, name):
    rates = catalog({'input': 1, 'output': 2, 'tiers': [
        {'tier': {'type': 'context', 'size': 200}, 'input': 3, 'output': 4}]}, official, name)
    model = ModelIdentity(provider, name)
    result = rates.price(model, tokens(input=201), SELECTED)
    assert result.known == Decimal('0.000683') and not result.unpriced
    assert {line.model for line in result.calculations} == {model}
    assert {line.source_provider for line in result.calculations} == {official}
    assert {line.context_threshold for line in result.calculations} == {200}
    assert rates.price_group(model, (201, 20, 0, 0), ('selected',) * 4, context_threshold=200) == result
    partial = rates.price(model, tokens(input=Unknown('not_reported')), SELECTED)
    assert partial.known == 0 and partial.unpriced
    assert {line.reason for line in partial.calculations} == {'unknown_tokens', 'unknown_context_size'}
    zero = rates.price(model, tokens(input=0, output=0), SELECTED)
    assert zero.known == 0 and not zero.unpriced


def test_official_fallback_is_exact_unique_and_does_not_override_direct_catalog():
    cost = {'input': 1, 'output': 2}
    raw = {provider: {'models': {'claude-sonnet-4-6': {'cost': cost}}}
           for provider in ('anthropic', 'openai')}
    raw['github-copilot'] = {'models': {'claude-sonnet-4-6': {'cost': {'input': 5, 'output': 10}}}}
    rates = Catalog.from_bytes(json.dumps(raw).encode(), snapshot_date='fixture', sha256='fixture')
    direct = rates.price(ModelIdentity('github-copilot', 'claude-sonnet-4-6'), tokens(), SELECTED)
    assert direct.known == Decimal('0.0007')
    assert {line.source_provider for line in direct.calculations} == {'github-copilot'}
    for model in (ModelIdentity(None, 'claude-sonnet-4-6'),
                  ModelIdentity('litellm', 'claude-sonnet-4-6'),
                  ModelIdentity('claude-bridge', 'claude-sonnet-4.6'),
                  ModelIdentity(None, None)):
        result = rates.price(model, tokens(), SELECTED)
        assert result.known == 0 and result.unpriced
        assert {line.reason for line in result.calculations} == {'model_not_in_catalog'}
    # A bridge explicitly names the Anthropic catalog; generic missing-provider evidence does not.
    bridge = rates.price(ModelIdentity('claude-bridge', 'claude-sonnet-4-6'), tokens(), SELECTED)
    assert bridge.known == Decimal('0.00014')
    raw['github-copilot']['models']['claude-sonnet-4-6']['cost'] = {}
    incomplete = Catalog.from_bytes(json.dumps(raw).encode(), snapshot_date='fixture', sha256='fixture')
    result = incomplete.price(ModelIdentity('github-copilot', 'claude-sonnet-4-6'), tokens(), SELECTED)
    assert result.unpriced and result.known == 0
    assert {line.reason for line in result.calculations} == {'missing_category_rate'}


@pytest.mark.parametrize(('recorded', 'canonical'), [
    ('claude-haiku-4.5', 'claude-haiku-4-5'),
    ('claude-opus-4.6', 'claude-opus-4-6'),
    ('claude-opus-4.7', 'claude-opus-4-7'),
    ('claude-opus-4.8', 'claude-opus-4-8'),
    ('claude-sonnet-4.5', 'claude-sonnet-4-5'),
    ('claude-sonnet-4.6', 'claude-sonnet-4-6'),
])
@pytest.mark.parametrize('provider', [None, 'github-copilot'])
def test_retained_copilot_punctuation_aliases_are_explicit_and_keep_identity(recorded, canonical, provider):
    rates = catalog({'input': 1, 'output': 2}, 'anthropic', canonical)
    model = ModelIdentity(provider, recorded)
    result = rates.price(model, tokens(), SELECTED)
    assert result.known == Decimal('0.00014') and not result.unpriced
    assert {line.model for line in result.calculations} == {model}
    assert {line.source_provider for line in result.calculations} == {'anthropic'}
    assert rates.price(ModelIdentity('litellm', recorded), tokens(), SELECTED).unpriced


def test_fast_or_unknown_punctuation_variants_do_not_borrow_other_model_prices():
    rates = catalog({'input': 1, 'output': 2}, 'anthropic', 'claude-opus-4-8')
    for name in ('claude-opus-4.8-fast', 'claude-opus-4.9', 'Claude-opus-4.8'):
        result = rates.price(ModelIdentity(None, name), tokens(), SELECTED)
        assert result.known == 0 and result.unpriced
        assert {line.reason for line in result.calculations} == {'model_not_in_catalog'}
