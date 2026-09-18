// Run with node tests/check_frontend_events.cjs; no browser or dependencies.
const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { runInNewContext } = require('node:vm');
let status, inserted;
let reloads = 0;
let finishReload;
let retry;
let hidden = false;
const report = { dataset: { revision: '1', canonical: '/' } };
const progress = { hidden: true, removeAttribute(name) { delete this[name]; } };
const span = {}, button = {};
const line = { dataset: {}, querySelector(selector) {
  return selector === 'span' ? span : selector === 'button' ? button : progress;
} };
class EventSource {
  addEventListener(name, callback) { status = callback; }
}
const up = {
  on(name, callback) { if (name === 'up:fragment:inserted') inserted = callback; },
  reload(target, options) {
    assert.equal(target, '#report');
    assert.equal(options.url, report.dataset.canonical);
    assert.equal(options.scroll, false);
    assert.equal(options.focus, 'keep');
    reloads++;
    return new Promise(resolve => { finishReload = resolve; });
  },
};
runInNewContext(readFileSync('src/harness_usage/static/app.js', 'utf8'), {
  window: { up, EventSource, addEventListener() {} }, up, EventSource,
  document: { querySelector: selector => hidden ? null : selector === '#report' ? report : line },
  setTimeout(callback) { retry = callback; return 1; }, clearTimeout() { retry = null; },
});
const emit = (state, revision, phase = null, checked = 0, total = null) => status({
  data: JSON.stringify({ state, revision, phase, files_checked: checked, files_total: total, run_id: 'run' }),
});
emit('running', 2, 'discovering');
assert.equal(progress.hidden, false);
assert.equal(progress.value, undefined);
assert.match(span.textContent, /Discovering files/);
emit('running', 3, 'checking', 1, 3);
assert.equal(progress.value, 1);
assert.equal(progress.max, 3);
assert.match(span.textContent, /1 \/ 3 files checked · 33%/);
assert.equal(reloads, 1, 'Refresh committed data during import, with only one request in flight');
assert.match(span.textContent, /partial results/i);
emit('running', 3, 'finalizing', 3, 3);
assert.equal(progress.value, undefined, 'Do not claim completion before finalization');
assert.match(span.textContent, /Finalizing/);
assert.equal(reloads, 1);
// Navigation can replace the status line between events; reapply the latest state.
hidden = true;
emit('running', 3, 'checking', 2, 3);
hidden = false;
inserted();
assert.equal(progress.value, 2);
assert.match(span.textContent, /66%/);
emit('succeeded', 3, null, 3, 3);
assert.equal(progress.hidden, true);
assert.equal(line.hidden, true);
assert.equal(reloads, 1, 'Refresh once when the new import is ready');
emit('failed', 3, null, 3, 3);
assert.equal(progress.hidden, true);
assert.equal(progress.value, undefined);
assert.equal(line.hidden, false);
assert.match(span.textContent, /Import failed/);
assert.equal(button.textContent, 'Retry import');
emit('interrupted', 3);
assert.equal(progress.hidden, true);
assert.match(span.textContent, /interrupted/);
emit('running', 3, 'finalizing', 0, 0);
assert.equal(progress.value, undefined);
assert.match(span.textContent, /Finalizing/);
assert.doesNotMatch(span.textContent, /NaN|100%/);
async function checkCatchup() {
  report.dataset.revision = '2';
  finishReload();
  await new Promise(setImmediate);
  assert.equal(typeof retry, 'function', 'Catch up when a newer revision arrived during the request');
  retry();
  assert.equal(reloads, 2);
  report.dataset.revision = '3';
  finishReload();
  await new Promise(setImmediate);
  retry();
  assert.equal(reloads, 2, 'Unchanged data must not reload');
  console.log('Frontend import progress and refresh checks passed');
}
checkCatchup().catch(error => { console.error(error); process.exitCode = 1; });
