const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const context = {document: {addEventListener() {}}};
vm.runInNewContext(fs.readFileSync(path.join(__dirname, '../src/harness_usage/static/transcript.js'), 'utf8'), context);
assert.equal(context.transcriptCostIndex(4, 0), 0);
assert.equal(context.transcriptCostIndex(4, 1), 3);
assert.equal(context.transcriptCostIndex(10000, .5), 4999);
assert.equal(context.transcriptCostIndex(4, -1), 0);
assert.equal(context.transcriptCostIndex(4, 2), 3);
const points = [{ratio:.25, unknown:false, incomplete:false}, {ratio:.25, unknown:true, incomplete:true}, {ratio:1, unknown:false, incomplete:true}];
const paths = context.transcriptCostPaths(points);
assert.ok(paths.known.includes('V'));
assert.ok(paths.partial.includes('V'));
assert.ok(paths.gaps.includes('M'));
assert.equal(context.transcriptSectionIndex(10000, i => i * 20, 100001), 5000);
// Histogram widths use the same response coordinate as the curve, including the short final bucket.
assert.equal(context.transcriptCostIndex(1601, 1598.01 / 1601), 1598);
assert.equal(context.transcriptCostIndex(1601, 1), 1600);

// Execute the actual listeners with only the native DOM operations they use.
function element(tagName = 'DIV', top = 0) {
  const classes = new Set();
  return {tagName, top, value: '', dataset: {}, handlers: {}, attributes: {}, parentElement: null,
    classList: {add: name => classes.add(name), remove: name => classes.delete(name),
      contains: name => classes.has(name), toggle: (name, on) => on ? classes.add(name) : classes.delete(name)},
    addEventListener(name, callback) { this.handlers[name] = callback; },
    setAttribute(name, value) { this.attributes[name] = value; },
    getBoundingClientRect() { return {top: this.top, height: 80, left: 0, width: 1000}; },
    scrollIntoView(options) { this.scrolled = options; }, querySelectorAll() { return []; }};
}
const ids = Object.fromEntries(['thread', 'transcript-search', 'result-count', 'previous', 'next',
  'expand-tools', 'cost-timeline', 'cost-histogram', 'cost-selection', 'cost-jump', 'cost-preview',
  'cost-cursor', 'cost-plot', 'cost-known', 'cost-partial', 'cost-gaps', 'one', 'two', 'three']
  .map(id => [id, element()]));
const toolbar = element();
const collapsed = element('DETAILS', 900);
collapsed.open = false; collapsed.parentElement = ids.thread;
ids.one.top = 300; ids.two.top = 600; ids.three.top = 900;
ids.one.parentElement = ids.two.parentElement = ids.thread;
ids.three.parentElement = collapsed;
ids['cost-timeline'].dataset.costPoints = JSON.stringify(points.map((point, i) => ({...point,
  anchor: ['one', 'two', 'three'][i], model: 'model', amount_label: i === 1 ? 'Cost unavailable' : 'Est. $1',
  total_label: i ? '≥$1' : '$1', excerpt: `Preview ${i + 1}`})));
ids['cost-timeline'].dataset.costBuckets = JSON.stringify([
  {start:0, end:2, label:'≥$1.00', unknown:false, incomplete:true},
  {start:2, end:3, label:'$3.00', unknown:false, incomplete:false}]);
const bars = [element('RECT'), element('RECT')];
ids['cost-histogram'].querySelectorAll = () => bars;
const events = {}, windowEvents = {};
const document = {activeElement: null, documentElement: {style: {setProperty() {}}, scrollHeight: 10000},
  getElementById: id => ids[id] || null, querySelector: () => toolbar,
  addEventListener: (name, callback) => { events[name] = callback; }};
let now = 0;
const browser = {document, window: {addEventListener: (name, callback) => { windowEvents[name] = callback; }},
  location: {hash: ''}, history: {replaceState: (_state, _title, hash) => { browser.location.hash = hash; }},
  matchMedia: () => ({matches: true}), requestAnimationFrame: callback => callback(),
  Date: {now: () => now}, scrollY: 0, innerHeight: 600};
vm.runInNewContext(fs.readFileSync(path.join(__dirname, '../src/harness_usage/static/transcript.js'), 'utf8'), browser);
events.DOMContentLoaded();
ids['cost-plot'].handlers.pointermove({clientX: 1});
assert.ok(ids['cost-preview'].textContent.includes('Preview 1'));
ids['cost-plot'].handlers.click({clientX: 1000});
assert.equal(collapsed.open, true);
assert.equal(ids.three.scrolled.behavior, 'auto'); // reduced motion
assert.equal(ids.three.classList.contains('cost-selected'), true);
assert.equal(browser.location.hash, '#three');
ids['cost-histogram'].handlers.keydown({key:'ArrowLeft', preventDefault() {}});
assert.equal(browser.location.hash, '#two');
assert.ok(ids['cost-selection'].textContent.includes('Cost unavailable'));
assert.equal(ids.three.classList.contains('cost-selected'), false);
assert.equal(bars[0].classList.contains('selected'), true);
ids['cost-histogram'].handlers.pointermove({clientX: 400});
assert.ok(ids['cost-preview'].textContent.includes('Responses 1–2 · Group cost ≥$1.00'));
ids['cost-histogram'].handlers.click({clientX: 750});
assert.equal(browser.location.hash, '#three');
ids['cost-histogram'].handlers.keydown({key:'Home', preventDefault() {}});
assert.equal(browser.location.hash, '#one');
ids['cost-histogram'].handlers.keydown({key:'End', preventDefault() {}});
assert.equal(browser.location.hash, '#three');
ids['cost-histogram'].handlers.keydown({key:'ArrowRight', preventDefault() {}});
assert.equal(browser.location.hash, '#three');
ids['cost-histogram'].handlers.keydown({key:'Home', preventDefault() {}});
document.activeElement = ids['cost-histogram'];
now = 2000;
ids.one.top = -300; ids.two.top = -100; ids.three.top = 50;
windowEvents.scroll();
assert.equal(ids['cost-histogram'].attributes['aria-valuenow'], '1'); // delayed programmatic scroll cannot replace explicit selection
windowEvents.wheel();
windowEvents.scroll();
assert.equal(ids['cost-histogram'].attributes['aria-valuenow'], '3');
assert.ok(ids['cost-selection'].textContent.includes('≥$1'));
console.log('cost interactions passed');
