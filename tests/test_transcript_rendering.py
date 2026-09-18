from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest

from harness_usage.transcript import (
    AttachmentBlock,
    Branch,
    ConflictingOutput,
    Message,
    Notice,
    ReasoningBlock,
    RecordedOutput,
    TextBlock,
    ToolBlock,
    Transcript,
    TranscriptLink,
    TranscriptPage,
)
from harness_usage.transcript_rendering import render_transcript, transcript_csp


def page_with(*entries: Message | Notice, harness: str = 'pi') -> TranscriptPage:
    return TranscriptPage(
        transcript=Transcript(
            session_id='session / one',
            native_id='native',
            title='Unsafe <script>alert(1)</script>',
            harness=harness,  # type: ignore[arg-type]
            entries=entries,
            cwd='/tmp/<img src=x onerror=alert(1)>',
            model='model&name',
            branches=(Branch('leaf / one', 'Latest <recorded>'), Branch('other', 'Earlier')),
            selected_branch='leaf / one',
            warnings=('Warning <svg onload=alert(1)>',),
        ),
        parent=TranscriptLink('parent / one', 'Parent <session>'),
        children=(TranscriptLink('child / one', 'Child <session>'),),
    )


@pytest.mark.parametrize('standalone', [False, True])
@pytest.mark.parametrize(('harness', 'label'), [
    ('pi', 'Pi'),
    ('codex', 'Codex'),
    ('claude', 'Claude Code'),
    ('copilot-vscode', 'Copilot in VS Code'),
    ('copilot-cli', 'Copilot CLI'),
])
def test_transcript_header_uses_product_labels(harness: str, label: str, standalone: bool) -> None:
    rendered = render_transcript(page_with(harness=harness), standalone=standalone)

    assert f'<span>{label}</span>' in rendered
    assert '<script>alert(1)</script>' not in rendered


def test_unknown_harness_label_does_not_echo_untrusted_input() -> None:
    from harness_usage.web import harness_label

    assert harness_label('<script>alert(1)</script>') == 'Unknown source'


@pytest.mark.parametrize('harness', ['pi', 'codex'])
def test_renderer_escapes_transcript_content_and_keeps_attachments_inert(harness: str) -> None:
    attack = (
        '<script>window.pwned=true</script>\n\n'
        '# Safe heading\n\n- item\n\n'
        '[remote label](https://evil.example/link)\n\n'
        '![remote image](https://evil.example/image)\n\n'
        '```html\n</style><img src=x>\n```'
    )
    page = page_with(
        Message(
            id='message-one',
            role='user',
            blocks=(
                TextBlock(attack),
                AttachmentBlock('https://evil.example/file?<script>'),
                ToolBlock(
                    call_id='call',
                    name='<img src=x onerror=alert(1)>',
                    arguments='</pre><script>alert(2)</script>',
                    output=RecordedOutput((TextBlock('<iframe src=https://evil.example>'),)),
                ),
            ),
        ),
        Notice('notice', 'Unknown <state>', '<a href=https://evil.example>go</a>'),
        harness=harness,
    )

    rendered = render_transcript(page)

    assert '<script>window.pwned' not in rendered
    assert '<img src=x' not in rendered
    assert '<iframe' not in rendered
    assert '&lt;script&gt;window.pwned=true&lt;/script&gt;' in rendered
    assert '<h1>Safe heading</h1>' in rendered
    assert '<li>item</li>' in rendered
    assert 'remote label' in rendered
    assert 'remote image' in rendered
    assert 'https://evil.example/file?' in rendered
    assert not re.search(r'(?:href|src)=["\']https://evil\.example', rendered)
    assert '<pre><code class="language-html">&lt;/style&gt;&lt;img src=x&gt;' in rendered
    assert '<pre class="tool-result"><code>&lt;iframe src=https://evil.example&gt;</code></pre>' in rendered


def test_renderer_groups_only_adjacent_tools_and_exposes_recorded_states() -> None:
    tools = (
        ToolBlock('one', 'Read', '{"path":"src/example.py"}', RecordedOutput((TextBlock('one result'),), status='succeeded')),
        ToolBlock('two', 'Run', 'false', RecordedOutput((TextBlock('failure'),), status='failed')),
        TextBlock('Between groups'),
        ToolBlock('three', 'Search', 'needle'),
        ToolBlock('four', 'Edit', 'patch', ConflictingOutput('duplicate result')),
        ToolBlock('five', 'bash', 'pytest'),
        ToolBlock('six', 'Agent', 'review'),
    )
    rendered = render_transcript(
        page_with(Message('assistant', 'assistant', tools, model='different-model'))
    )

    assert rendered.count('class="activity-group"') == 2
    assert '2 actions' in rendered
    assert 'Includes failed run' in rendered
    assert 'Result not recorded' in rendered
    assert 'Conflicting result' in rendered
    assert 'duplicate result' in rendered
    assert 'one result' in rendered
    assert 'class="message assistant"' in rendered
    assert 'class="nf"' in rendered
    assert re.search(r'class="tool-label"[^>]*>bash</span>', rendered) and '>Run</span>' in rendered
    assert re.search(r'class="tool-label"[^>]*>Agent</span>', rendered) and '>Subagent</span>' in rendered
    assert re.search(r'class="tool-label"[^>]*>src/example\.py</span>', rendered)
    rail = re.findall(r'href="#([^"]+)" class="history-tick', rendered)
    assert rail == []


def test_live_renderer_has_encoded_navigation_selected_branch_and_external_assets() -> None:
    rendered = render_transcript(page_with(Message('m', 'assistant', (TextBlock('hello'),))))

    assert 'href="/sessions/parent%20%2F%20one/transcript"' in rendered
    assert 'href="/sessions/child%20%2F%20one/transcript"' in rendered
    assert 'action="/sessions/session%20%2F%20one/transcript"' in rendered
    assert '<option value="leaf / one" selected>' in rendered
    assert 'href="/sessions/session%20%2F%20one/transcript.html?branch=leaf%20%2F%20one"' in rendered
    assert '<link rel="stylesheet" href="/static/transcript.css">' in rendered
    assert '<script src="/static/transcript.js" defer></script>' in rendered
    assert "'unsafe-inline'" not in transcript_csp()
    assert "style-src 'self'" in transcript_csp()


def test_subagent_links_keep_full_escaped_titles_for_hover() -> None:
    from dataclasses import replace
    from html import unescape

    title = 'Inspect "long path" <notes> & ' + '/workspace/' + 'component' * 60
    page = replace(page_with(), children=(TranscriptLink('child / one', title),))
    rendered = render_transcript(page)

    hover = re.search(r'title="([^"<>]*)"', rendered)
    assert hover and unescape(hover[1]) == title
    label = re.search(r'<span class="child-title">([^<>]*)</span>', rendered)
    assert label and unescape(label[1]) == title
    assert 'href="/sessions/child%20%2F%20one/transcript"' in rendered


def test_background_context_roles_are_searchable_disclosures_not_chat_bubbles() -> None:
    rendered = render_transcript(
        page_with(
            Message('system', 'system', (TextBlock('system context'),)),
            Message('developer', 'developer', (TextBlock('developer context'),)),
            Message('context', 'context', (TextBlock('saved context'),)),
            Message('assistant', 'assistant', (TextBlock('answer'),)),
            Message('later-context', 'context', (TextBlock('later context'),)),
        )
    )

    assert rendered.count('class="context-note context-group"') == 2
    assert rendered.count('<summary>Session context</summary>') == 2
    assert rendered.count('class="context-message"') == 4
    assert '>System</span>' in rendered
    assert '>Developer</span>' in rendered
    assert '>Context</span>' in rendered
    assert 'Context context' not in rendered
    assert 'class="message system"' not in rendered
    assert 'class="message developer"' not in rendered
    assert 'class="message context"' not in rendered
    assert '<p>system context</p>' in rendered
    assert '<p>later context</p>' in rendered
    assert re.search(r'class="context-message" id="entry-[0-9a-f]{16}"', rendered)


def test_consecutive_tool_and_reasoning_only_messages_share_compact_activity_rows() -> None:
    rendered = render_transcript(
        page_with(
            Message('tool-one', 'assistant', (ToolBlock('one', 'Run', 'one'),)),
            Message('reason-one', 'assistant', (ReasoningBlock('thinking one'),)),
            Message(
                'tools-two-three',
                'assistant',
                (ToolBlock('two', 'Read', 'two'), ToolBlock('three', 'Run', 'three')),
            ),
            Message('answer', 'assistant', (TextBlock('visible answer'),)),
            Message('reason-two', 'assistant', (ReasoningBlock('thinking two'),)),
            Message('user', 'user', (TextBlock('next request'),)),
            Message('tool-four', 'assistant', (ToolBlock('four', 'Search', 'four'),)),
        )
    )

    assert rendered.count('class="activity-batch"') == 3
    assert '<span>3 actions · 1 reasoning entry</span>' in rendered
    assert '<span>1 reasoning entry</span>' in rendered
    assert '<span>1 action</span>' in rendered
    assert rendered.count('class="activity-entry"') == 5
    assert 'class="message assistant"' in rendered
    assert '<p>visible answer</p>' in rendered
    assert 'class="message user"' in rendered
    assert re.search(r'class="activity-entry" id="entry-[0-9a-f]{16}"', rendered)
    css = Path('src/harness_usage/static/transcript.css').read_text()
    assert '.activity-entry>.activity-group,.activity-entry>.reasoning{margin:0}' in css
    assert 'line-height:1.4' in css


def test_long_markdown_and_activity_content_has_explicit_shrink_boundaries() -> None:
    css = Path('src/harness_usage/static/transcript.css').read_text()

    assert re.search(r'\.markdown\{[^}]*overflow-wrap:anywhere', css)
    assert re.search(r'\.markdown :not\(pre\)>code\{[^}]*overflow-wrap:anywhere', css)
    assert re.search(r'\.activity-batch\{[^}]*min-width:0', css)
    assert re.search(r'\.activity-entry\{[^}]*min-width:0', css)


def test_safe_markdown_tables_render_readably_without_active_links() -> None:
    markdown = (
        '| Area | Verified finding |\n'
        '| --- | --- |\n'
        '| Parser | **Works** |\n'
        '| Safety | <script>alert(1)</script> [remote](https://evil.example) |'
    )

    rendered = render_transcript(page_with(Message('table', 'assistant', (TextBlock(markdown),))))
    css = Path('src/harness_usage/static/transcript.css').read_text()

    assert '<table>' in rendered and '<th>Area</th>' in rendered
    assert '<strong>Works</strong>' in rendered
    assert '<script>alert(1)</script>' not in rendered
    assert '&lt;script&gt;alert(1)&lt;/script&gt;' in rendered
    assert 'href="https://evil.example"' not in rendered
    assert re.search(r'\.markdown\{[^}]*overflow-x:auto', css)
    assert re.search(r'\.markdown table\{[^}]*border-collapse:collapse', css)


def test_history_rail_omits_activity_but_preserves_all_its_content() -> None:
    entries = tuple(
        Message(f'reason-{number}', 'assistant', (ReasoningBlock(f'thinking {number}'),))
        for number in range(2_000)
    )

    rendered = render_transcript(page_with(*entries))

    assert rendered.count('class="activity-batch"') == 1
    assert rendered.count('class="activity-entry"') == 2_000
    assert rendered.count('class="history-tick') == 0
    assert '<span>2000 reasoning entries</span>' in rendered


def test_standalone_renderer_embeds_trusted_assets_and_matching_csp() -> None:
    text = 'Full original content: <b>still text</b>'
    rendered = render_transcript(page_with(Message('m', 'assistant', (TextBlock(text),))), standalone=True)
    policy = transcript_csp(standalone=True)

    assert '/static/transcript.' not in rendered
    assert 'data:font/ttf;base64,' in rendered
    assert 'data:font/woff;base64,' in rendered
    assert text.replace('<', '&lt;').replace('>', '&gt;') in rendered
    assert rendered.count('<div class="text markdown"><p>Full original content:') == 1
    assert '<meta http-equiv="Content-Security-Policy"' in rendered
    assert 'href="/"' not in rendered
    assert '/sessions/' not in rendered
    assert "'unsafe-inline'" not in policy
    assert policy in rendered.replace('&#39;', "'")
    for digest in re.findall(r"'sha256-[^']+'", policy):
        assert digest.replace("'", '&#39;') in rendered or digest in rendered


def test_many_source_warnings_are_collapsed_but_all_remain_inspectable() -> None:
    page = page_with(Message('m', 'assistant', (TextBlock('hello'),)))
    transcript = page.transcript
    page = TranscriptPage(
        Transcript(
            transcript.session_id,
            transcript.native_id,
            transcript.title,
            transcript.harness,
            transcript.entries,
            warnings=tuple(f'Warning {number}' for number in range(1_001)),
        )
    )

    rendered = render_transcript(page)

    assert '<details class="warnings">' in rendered
    assert '<summary>1001 source notices</summary>' in rendered
    assert 'Warning 0' in rendered and 'Warning 1000' in rendered


def test_entry_anchor_is_stable_when_an_earlier_entry_is_inserted() -> None:
    message = Message('source-local-id', 'assistant', (TextBlock('hello'),))
    first = render_transcript(page_with(message))
    shifted = render_transcript(page_with(Notice('new', 'Checkpoint', 'Earlier'), message))

    first_anchor = re.search(r'<article class="message assistant" id="([^"]+)"', first)
    shifted_anchor = re.search(r'<article class="message assistant" id="([^"]+)"', shifted)
    assert first_anchor and shifted_anchor
    assert first_anchor.group(1) == shifted_anchor.group(1)


def test_large_code_and_empty_recorded_result_remain_bounded_and_explicit() -> None:
    large = 'start\n' + ('x' * 100_000) + '\nend'
    rendered = render_transcript(
        page_with(
            Message(
                'large',
                'assistant',
                (
                    TextBlock(f'```text\n{large}\n```'),
                    ToolBlock('empty', 'Run', large, RecordedOutput(())),
                ),
            )
        )
    )

    assert rendered.count('An empty result was recorded.') == 1
    assert 'start\n' in rendered and '\nend' in rendered
    css = Path('src/harness_usage/static/transcript.css').read_text()
    assert 'max-height:300px' in css
    assert 'overflow:auto' in css


def test_browser_search_uses_unicode_safe_original_string_offsets() -> None:
    javascript = Path('src/harness_usage/static/transcript.js').read_text()
    probe = (
        "const vm=require('node:vm');"
        "const context={document:{addEventListener(){}}};"
        f"vm.runInNewContext({json.dumps(javascript)},context);"
        "process.stdout.write(JSON.stringify(["
        "context.transcriptMatchRanges('İ and [x]','İ'),"
        "context.transcriptMatchRanges('İ and [x]','[x]')]));"
    )

    result = subprocess.run(['node', '-e', probe], check=True, capture_output=True, text=True)

    assert json.loads(result.stdout) == [[[0, 1]], [[6, 3]]]


def test_browser_search_caps_rendered_marks_for_large_transcripts() -> None:
    javascript = Path('src/harness_usage/static/transcript.js').read_text()
    probe = (
        "const vm=require('node:vm');"
        "const context={document:{addEventListener(){}}};"
        f"vm.runInNewContext({json.dumps(javascript)},context);"
        "process.stdout.write(String("
        "context.transcriptMatchRanges('a'.repeat(2500),'a',1000).length));"
    )

    result = subprocess.run(['node', '-e', probe], check=True, capture_output=True, text=True)

    assert result.stdout == '1000'
    assert 'const MAX_SEARCH_MATCHES = 1000' in javascript
    assert "'1000+'" in javascript


def test_section_tracking_for_long_histories_uses_bounded_lookups() -> None:
    javascript = Path('src/harness_usage/static/transcript.js').read_text()
    probe = (
        "const vm=require('node:vm'), assert=require('node:assert/strict');"
        "const context={document:{addEventListener(){}}};"
        f"vm.runInNewContext({json.dumps(javascript)},context);"
        "const tops=[-500,-100,40,800];"
        "assert.equal(context.transcriptSectionIndex(4,i=>tops[i],50),2);"
        "assert.equal(context.transcriptSectionIndex(4,i=>tops[i],-600),-1);"
        "assert.equal(context.transcriptSectionIndex(4,i=>tops[i],900),3);"
        "let reads=0; const index=context.transcriptSectionIndex(100000,i=>{reads++;return i*20;},100001);"
        "assert.equal(index,5000);assert.ok(reads<=17);"
    )
    subprocess.run(['node', '-e', probe], check=True, capture_output=True, text=True)


def test_removed_history_rail_preserves_all_content_in_order() -> None:
    rendered = render_transcript(page_with(
        Message('system', 'system', (TextBlock('background system'),)),
        Message('first', 'user', (TextBlock('first request'),)),
        Message('answer', 'assistant', (TextBlock('answer retained'),)),
        Message('inherited', 'context', (TextBlock('inherited request'),)),
        Message('second', 'user', (TextBlock('second request'),)),
        Message('tool', 'assistant', (ToolBlock('call', 'Run', 'arguments retained'),)),
    ))
    assert 'history-rail' not in rendered
    assert rendered.index('first request') < rendered.index('second request')
    assert all(text in rendered for text in ('background system', 'answer retained', 'inherited request', 'arguments retained'))
