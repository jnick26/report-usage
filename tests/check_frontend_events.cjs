// Run with node tests/check_frontend_events.cjs; no browser or dependencies.
const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { runInNewContext } = require('node:vm');
let status;
let reloads = 0;
const report = { dataset: { revision: '1', canonical: '/' } };
class EventSource {
  addEventListener(name, callback) { status = callback; }
}
const up = { on() {}, reload() { reloads++; return Promise.resolve(); } };
runInNewContext(readFileSync('src/harness_usage/static/app.js', 'utf8'), {
  window: { up, EventSource, addEventListener() {} }, up, EventSource,
  document: { querySelector: selector => selector === '#report' ? report : null },
  setTimeout, clearTimeout,
});
const emit = (state, revision) => status({ data: JSON.stringify({ state, revision }) });
emit('running', 2);
emit('running', 3);
assert.equal(reloads, 0, 'Keep the committed report while import progresses');
emit('succeeded', 3);
assert.equal(reloads, 1, 'Refresh once when the new import is ready');
console.log('Frontend import refresh check passed');
