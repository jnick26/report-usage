from decimal import Decimal
from hashlib import sha256
from html import unescape
import json
from pathlib import Path
import re
import subprocess

import pytest

from harness_usage.domain import ModelIdentity
from harness_usage.pricing import CostLine
from harness_usage.transcript import Message, Notice, TextBlock, ToolBlock, Transcript, TranscriptPage
from harness_usage.transcript_rendering import render_transcript, transcript_csp


def cost_page():
    from harness_usage.transcript import TranscriptCostPoint

    entries = (Message('request/one', 'assistant', (TextBlock('Answer <script>preview</script>'),), model='model-one'),
               Message('request/two', 'assistant', (ToolBlock('tool', 'Read', '{}'),), model='model-two'),
               Message('request/three', 'assistant', (TextBlock('Unpriced answer'),)),
               Message('request/four', 'assistant', (TextBlock('Known zero'),)))
    line = CostLine(ModelIdentity('vendor', 'model-one'), 'output', 123, Decimal('1'), Decimal('0.000123'), 'vendor')
    return TranscriptPage(Transcript('pi:session', 'session', 'Cost test', 'pi', entries), costs=(
        TranscriptCostPoint('request/one', Decimal('0.000123'), Decimal('0.000123'), False, False, (line,)),
        TranscriptCostPoint('request/two', Decimal('0.002'), Decimal('0.002123'), True, True),
        TranscriptCostPoint('request/three', None, Decimal('0.002123'), True, True),
        TranscriptCostPoint('request/four', Decimal(0), Decimal('0.002123'), False, True),
    ))


@pytest.mark.parametrize('standalone', [False, True])
def test_cost_chart_links_requests_and_preserves_exact_unknown_and_partial_labels(standalone):
    html = render_transcript(cost_page(), standalone=standalone)
    assert 'id="cost-timeline"' in html
    assert 'Branch total (estimated)' in html and 'API-equivalent estimate' in html
    assert '≥$0.002123' in html and '$0.000123' in html
    assert '3 of 4 requests have a priced amount' in html
    assert 'Cost unavailable' in html and 'Est. $0' in html
    assert 'id="cost-histogram"' in html and 'role="slider"' in html
    assert 'aria-valuemax="4"' in html and 'type="range"' not in html
    assert 'history-rail' not in html and 'history-tick' not in html
    assert '<script>preview</script>' not in html
    for index, identity in enumerate(('request/one', 'request/two', 'request/three', 'request/four'), 1):
        anchor = 'entry-' + sha256(identity.encode()).hexdigest()[:16]
        assert f'id="{anchor}"' in html
        assert f'Response {index}' in html
        assert anchor in html
    tool_article = html.split('class="activity-entry"', 1)[1].split('</article>', 1)[0]
    assert 'Response 2' in tool_article and '≥$0.002' in tool_article
    assert '123 tokens' in html and '$1 / Mtok' in html
    assert 'Tool execution is not separately priced' in html
    if standalone:
        assert '/static/transcript.' not in html
        assert transcript_csp(standalone=True) in html.replace('&#39;', "'")


def test_extension_metadata_groups_without_losing_anchors_or_other_source_content():
    entries = (Notice('meta-1', 'Extension state', 'one'), Message('user', 'user', (TextBlock('request'),)),
               Notice('meta-2', 'Extension state', 'two'), Notice('branch', 'Branch summary', 'preserved'))
    html = render_transcript(TranscriptPage(Transcript('pi:s', 's', 's', 'pi', entries)))
    assert html.count('class="context-note metadata-group"') == 1
    assert 'Session metadata · 2 records' in html
    assert 'one' in html and 'two' in html and 'Branch summary' in html
    assert 'id="cost-timeline"' not in html
    assert 'id="transcript-search"' in html and 'href="#end"' in html
    for identity in ('meta-1', 'meta-2'):
        assert 'entry-' + sha256(identity.encode()).hexdigest()[:16] in html


def test_cost_browser_behaviors():
    result = subprocess.run(['node', str(Path(__file__).with_name('check_transcript_cost.cjs'))],
                            check=True, capture_output=True, text=True)
    assert 'cost interactions passed' in result.stdout


def test_many_requests_share_one_keyboard_selector_and_keep_last_anchor():
    from harness_usage.transcript import TranscriptCostPoint

    entries = tuple(Message(f'request-{i}', 'assistant', (TextBlock('response'),)) for i in range(1601))
    costs = tuple(TranscriptCostPoint(entry.id, Decimal('0.01'), Decimal(i + 1) / 100, False, False)
                  for i, entry in enumerate(entries))
    html = render_transcript(TranscriptPage(Transcript('pi:s', 's', 'Long', 'pi', entries), costs=costs))
    assert html.count('role="slider"') == 1 and 'type="range"' not in html
    assert 'aria-valuemax="1601"' in html and '1601 of 1601 requests' in html
    assert 'Response 1601' in html
    assert 'entry-' + sha256(b'request-1600').hexdigest()[:16] in html


def test_entirely_unpriced_requests_do_not_present_zero_cost():
    from harness_usage.transcript import TranscriptCostPoint

    page = TranscriptPage(Transcript('pi:s', 's', 'Unknown', 'pi',
                          (Message('missing', 'assistant', (TextBlock('response'),)),)),
                          costs=(TranscriptCostPoint('missing', None, Decimal(0), True, True),))
    html = render_transcript(page)
    assert '>Cost unavailable</strong>' in html
    assert 'Est. $0' not in html and '0 of 1 requests have a priced amount' in html


def test_main_labels_use_readable_cents_but_details_preserve_exact_values():
    from harness_usage.transcript import TranscriptCostPoint

    entries = (Message('one', 'assistant', (TextBlock('response'),)),
               Message('two', 'assistant', (TextBlock('response'),)))
    costs = (TranscriptCostPoint('one', Decimal('0.000123'), Decimal('0.000123'), False, False),
             TranscriptCostPoint('two', Decimal('171.964877'), Decimal('171.965'), False, False))
    html = render_transcript(TranscriptPage(Transcript('pi:s', 's', 'Money', 'pi', entries), costs=costs))
    assert 'Est. &lt;$0.01</summary>' in html
    assert 'Est. $171.96</summary>' in html
    assert 'Exact estimate: $171.964877' in html
    assert 'Exact total so far: $171.965' in html
    assert '≥&lt;' not in html


@pytest.mark.parametrize(('amount', 'axis', 'bound'), [
    ('1.999', '$2.00', '≥$1.99'),
    ('171.96008325', '$172', '≥$171.96'),
])
def test_partial_axis_scale_does_not_claim_rounded_value_as_lower_bound(amount, axis, bound):
    from harness_usage.transcript import TranscriptCostPoint

    value = Decimal(amount)
    page = TranscriptPage(Transcript('pi:s', 's', 'Partial', 'pi',
                          (Message('partial', 'assistant', (TextBlock('response'),)),)),
                          costs=(TranscriptCostPoint('partial', value, value, True, True),))
    html = render_transcript(page)
    assert f'<span class="cost-axis-top">{axis}</span>' in html
    assert f'>{bound}</strong>' in html
    assert f'Exact total so far: ≥${amount}' in html


def test_known_zero_without_calculation_lines_is_not_labeled_unavailable():
    from harness_usage.transcript import TranscriptCostPoint

    page = TranscriptPage(Transcript('pi:s', 's', 'Zero', 'pi',
                          (Message('zero', 'assistant', (TextBlock('response'),)),)),
                          costs=(TranscriptCostPoint('zero', Decimal(0), Decimal(0), False, False),))
    html = render_transcript(page)
    assert 'Est. $0.00</summary>' in html
    assert 'Pricing evidence is unavailable' not in html


def test_histogram_aggregates_request_costs_exactly_and_keeps_unknown_bucket_local():
    from harness_usage.transcript import TranscriptCostPoint

    amounts = ['0.1', '0.2', '1', '2', None, None, '0.05', None, '2', '1'] + ['0'] * 182
    entries = tuple(Message(str(i), 'assistant', (TextBlock('response'),)) for i in range(192))
    total = Decimal(0)
    incomplete = False
    costs = []
    for i, amount in enumerate(amounts):
        value = Decimal(amount) if amount is not None else None
        total += value or Decimal(0)
        incomplete |= value is None
        costs.append(TranscriptCostPoint(str(i), value, total, value is None, incomplete))
    rendered = render_transcript(TranscriptPage(Transcript('pi:s', 's', 'Bars', 'pi', entries), costs=tuple(costs)))
    match = re.search(r'data-cost-buckets="([^"]+)"', rendered)
    assert match is not None
    buckets = json.loads(unescape(match[1]))
    assert len(buckets) == 96
    assert [(bucket['start'], bucket['end']) for bucket in buckets] == [(i, i + 2) for i in range(0, 192, 2)]
    assert buckets[0]['amount'] == '0.3' and buckets[0]['ratio'] == 0.1
    assert buckets[1]['amount'] == '3' and buckets[1]['ratio'] == 1
    assert buckets[2]['amount'] is None and buckets[2]['unknown']
    assert buckets[2]['label'] == 'Cost unavailable'
    assert buckets[3]['amount'] == '0.05' and buckets[3]['incomplete'] and buckets[3]['label'] == '≥$0.05'
    assert buckets[4]['amount'] == '3' and not buckets[4]['incomplete']
    assert buckets[5]['amount'] == '0' and not buckets[5]['unknown']
    assert sum(Decimal(bucket['amount']) for bucket in buckets if bucket['amount'] is not None) == Decimal('6.35')
