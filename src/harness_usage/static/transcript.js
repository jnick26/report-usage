const MAX_SEARCH_MATCHES = 1000;

function transcriptMatchRanges(value, term, limit = Infinity) {
  const escaped = term.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const ranges = [];
  for (const match of value.matchAll(new RegExp(escaped, 'giu'))) {
    ranges.push([match.index, match[0].length]);
    if (ranges.length === limit) break;
  }
  return ranges;
}

// Keep ticks readable; very long histories remain independently scrollable.
function transcriptTickHeight(count, available, coarse = false) {
  return coarse ? 44 : Math.max(10, Math.min(18, available / Math.max(1, count)));
}

function transcriptSectionIndex(count, topAt, readingLine) {
  let low = 0, high = count;
  while (low < high) {
    const middle = (low + high) >>> 1;
    if (topAt(middle) <= readingLine) low = middle + 1;
    else high = middle;
  }
  return low - 1;
}

document.addEventListener('DOMContentLoaded', () => {
  const thread = document.getElementById('thread');
  const search = document.getElementById('transcript-search');
  const count = document.getElementById('result-count');
  const previous = document.getElementById('previous');
  const next = document.getElementById('next');
  let matches = [];
  let current = -1;
  let capped = false;

  function updateCount() {
    const total = capped ? '1000+' : matches.length;
    count.textContent = search.value.trim() ? (matches.length ? `${current + 1} / ${total}` : 'No matches') : '';
    previous.disabled = next.disabled = !matches.length;
  }

  function show(index) {
    if (!matches.length) return;
    matches[current]?.classList.remove('current');
    current = (index + matches.length) % matches.length;
    const mark = matches[current];
    mark.classList.add('current');
    for (let parent = mark.parentElement; parent && parent !== thread; parent = parent.parentElement) {
      if (parent instanceof HTMLDetailsElement) parent.open = true;
    }
    mark.scrollIntoView({block: 'center'});
    updateCount();
  }

  function find() {
    thread.querySelectorAll('mark').forEach(mark => mark.replaceWith(document.createTextNode(mark.textContent)));
    thread.normalize();
    matches = [];
    current = -1;
    capped = false;
    const term = search.value.trim();
    if (!term) return updateCount();
    const walker = document.createTreeWalker(thread, NodeFilter.SHOW_TEXT, {
      acceptNode: node => node.parentElement.closest('button,.sr-only') ? NodeFilter.FILTER_REJECT : NodeFilter.FILTER_ACCEPT
    });
    const nodes = [];
    while (walker.nextNode()) nodes.push(walker.currentNode);
    for (const node of nodes) {
      const value = node.textContent;
      let start = 0;
      const ranges = transcriptMatchRanges(value, term, MAX_SEARCH_MATCHES - matches.length);
      if (!ranges.length) continue;
      const fragment = document.createDocumentFragment();
      for (const [position, length] of ranges) {
        fragment.append(value.slice(start, position));
        const mark = document.createElement('mark');
        mark.textContent = value.slice(position, position + length);
        fragment.append(mark);
        matches.push(mark);
        start = position + length;
      }
      fragment.append(value.slice(start));
      node.replaceWith(fragment);
      if (matches.length === MAX_SEARCH_MATCHES) {
        capped = true;
        break;
      }
    }
    matches.length ? show(0) : updateCount();
  }

  search.addEventListener('input', find);
  search.addEventListener('keydown', event => {
    if (event.key === 'Enter') {
      event.preventDefault();
      show(current + (event.shiftKey ? -1 : 1));
    }
  });
  previous.addEventListener('click', () => show(current - 1));
  next.addEventListener('click', () => show(current + 1));
  document.getElementById('expand-tools').addEventListener('click', event => {
    const open = event.currentTarget.textContent === 'Expand tools';
    thread.querySelectorAll('.activity-batch,.activity-group,.tool').forEach(tool => { tool.open = open; });
    event.currentTarget.textContent = open ? 'Collapse tools' : 'Expand tools';
  });
  document.getElementById('branch-selector')?.addEventListener('change', event => event.currentTarget.form.requestSubmit());
  function openTarget(target, scroll = false) {
    if (!target) return;
    for (let parent = target; parent && parent !== thread; parent = parent.parentElement) {
      if (parent instanceof HTMLDetailsElement) parent.open = true;
    }
    if (scroll) target.scrollIntoView({block: 'start'});
  }

  function openHash() {
    let id;
    try { id = decodeURIComponent(location.hash.slice(1)); } catch { return; }
    if (id) openTarget(document.getElementById(id), true);
  }

  document.addEventListener('click', async event => {
    const copy = event.target.closest('.copy');
    if (copy) {
      try {
        await navigator.clipboard.writeText(document.getElementById(copy.dataset.copyTarget).textContent);
        copy.textContent = 'Copied';
        setTimeout(() => { copy.textContent = 'Copy'; }, 1200);
      } catch {
        copy.textContent = 'Select text to copy';
      }
    }
    const tick = event.target.closest('.history-tick');
    if (tick) openTarget(document.getElementById(tick.hash.slice(1)));
  });
  window.addEventListener('hashchange', openHash);
  const rail = document.querySelector('.history-rail');
  const toolbar = document.querySelector('.toolbar-shell');
  const ticks = [...rail.querySelectorAll('.history-tick')];
  const sections = ticks.map(tick => {
    let section = document.getElementById(tick.hash.slice(1));
    while (section && section.parentElement !== thread) section = section.parentElement;
    return section;
  });
  const preview = document.createElement('div');
  preview.className = 'rail-preview';
  preview.setAttribute('role', 'tooltip');
  preview.hidden = true;
  document.body.append(preview);
  let activeTick = null;
  let scheduled = false;

  function revealTick(tick) {
    if (!tick || rail.matches(':hover') || rail.contains(document.activeElement)) return;
    const box = rail.getBoundingClientRect();
    const target = tick.getBoundingClientRect();
    if (target.top < box.top || target.bottom > box.bottom) {
      rail.scrollTop += (target.top + target.bottom - box.top - box.bottom) / 2;
    }
  }

  function updateRail() {
    scheduled = false;
    const top = toolbar.getBoundingClientRect().height + 16;
    document.documentElement.style.setProperty('--rail-top', `${top}px`);
    const bottom = Math.min(innerHeight - 16, thread.getBoundingClientRect().bottom);
    const available = Math.max(0, bottom - Math.max(top, rail.getBoundingClientRect().top));
    rail.style.maxHeight = `${available}px`;
    rail.style.setProperty('--tick-height', `${transcriptTickHeight(ticks.length,
      available, matchMedia('(pointer:coarse)').matches)}px`);
    if (!ticks.length) return;
    const atEnd = scrollY + innerHeight >= document.documentElement.scrollHeight - 2;
    const index = atEnd ? ticks.length - 1 : transcriptSectionIndex(sections.length,
      i => sections[i].getBoundingClientRect().top, top + 16);
    const tick = ticks[index] ?? null;
    if (activeTick !== tick) {
      activeTick?.removeAttribute('aria-current');
      tick?.setAttribute('aria-current', 'location');
      activeTick = tick;
    }
    revealTick(tick);
  }

  function scheduleRail() {
    if (!scheduled) {
      scheduled = true;
      requestAnimationFrame(updateRail);
    }
  }

  function showPreview(event) {
    const tick = event.target.closest('.history-tick');
    if (!tick) return;
    const source = tick.querySelector('.history-preview');
    preview.replaceChildren(...[...source.children].map(child => child.cloneNode(true)));
    preview.hidden = false;
    const box = tick.getBoundingClientRect();
    preview.style.left = `${Math.max(8, Math.min(box.right + 12, innerWidth - preview.offsetWidth - 8))}px`;
    preview.style.top = `${Math.max(8, Math.min(box.top, innerHeight - preview.offsetHeight - 8))}px`;
  }
  rail.addEventListener('pointerover', showPreview);
  rail.addEventListener('focusin', showPreview);
  rail.addEventListener('pointerleave', () => { preview.hidden = true; scheduleRail(); });
  rail.addEventListener('focusout', () => { preview.hidden = true; });
  rail.addEventListener('scroll', () => { preview.hidden = true; }, {passive: true});
  window.addEventListener('scroll', () => { preview.hidden = true; scheduleRail(); }, {passive: true});
  window.addEventListener('resize', scheduleRail);
  thread.addEventListener('toggle', scheduleRail, true);
  thread.addEventListener('load', scheduleRail, true);
  if ('ResizeObserver' in window) {
    const resize = new ResizeObserver(scheduleRail);
    resize.observe(toolbar);
    resize.observe(thread);
  }
  updateRail();
  openHash();
});
