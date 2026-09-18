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

function transcriptCostIndex(count, fraction) {
  return Math.max(0, Math.min(count - 1, Math.ceil(fraction * count) - 1));
}

function transcriptCostPaths(points) {
  const paths = {known: '', partial: '', gaps: ''};
  let x = 0, y = 100;
  points.forEach((point, index) => {
    const nextX = (index + 1) / points.length * 1000;
    const nextY = 100 - point.ratio * 100;
    if (point.unknown) paths.gaps += `M${nextX},0V100`;
    else paths[point.incomplete ? 'partial' : 'known'] += `M${x},${y}H${nextX}V${nextY}`;
    x = nextX; y = nextY;
  });
  return paths;
}

function transcriptReveal(target, thread, scroll = false) {
  if (!target) return;
  for (let parent = target; parent && parent !== thread; parent = parent.parentElement) {
    if (parent.tagName === 'DETAILS') parent.open = true;
  }
  if (scroll) target.scrollIntoView({block: 'start',
    behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth'});
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
    transcriptReveal(target, thread, scroll);
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
  });
  window.addEventListener('hashchange', openHash);
  const toolbar = document.querySelector('.toolbar-shell');
  const timeline = document.getElementById('cost-timeline');
  const points = timeline ? JSON.parse(timeline.dataset.costPoints) : [];
  const targets = points.map(point => document.getElementById(point.anchor));
  const selector = document.getElementById('cost-histogram');
  const buckets = timeline ? JSON.parse(timeline.dataset.costBuckets) : [];
  const bars = selector ? [...selector.querySelectorAll('.cost-bar')] : [];
  const selection = document.getElementById('cost-selection');
  const jump = document.getElementById('cost-jump');
  const preview = document.getElementById('cost-preview');
  const cursor = document.getElementById('cost-cursor');
  const svg = document.getElementById('cost-plot');
  let selected = -1;
  let explicitSelection = false;
  let scheduled = false;

  function describe(index) {
    const point = points[index];
    return `Response ${index + 1} of ${points.length} · This response ${point.amount_label} · Total so far ${point.total_label}`;
  }

  function select(index, scroll = false) {
    if (!points[index]) return;
    targets[selected]?.classList.remove('cost-selected');
    selected = index;
    targets[index]?.classList.add('cost-selected');
    selector.setAttribute('aria-valuenow', String(index + 1));
    selector.setAttribute('aria-valuetext', describe(index));
    bars.forEach((bar, i) => bar.classList.toggle('selected', index >= buckets[i].start && index < buckets[i].end));
    selection.textContent = describe(index);
    jump.href = '#' + points[index].anchor;
    cursor.setAttribute('cx', String((index + 1) / points.length * 1000));
    cursor.setAttribute('cy', String(100 - points[index].ratio * 100));
    if (scroll) {
      explicitSelection = true;
      openTarget(targets[index], true);
      history.replaceState(null, '', '#' + points[index].anchor);
    }
  }

  function updatePosition() {
    scheduled = false;
    timeline?.classList.toggle('compact', toolbar.getBoundingClientRect().top <= 0 && scrollY > 0);
    const top = toolbar.getBoundingClientRect().height + 16;
    document.documentElement.style.setProperty('--toolbar-height', `${top}px`);
    if (!points.length || explicitSelection) return;
    const atEnd = scrollY + innerHeight >= document.documentElement.scrollHeight - 2;
    const index = atEnd ? points.length - 1 : transcriptSectionIndex(targets.length, i => {
      let target = targets[i];
      for (let parent = target.parentElement; parent && parent !== thread; parent = parent.parentElement) {
        if (parent.tagName === 'DETAILS' && !parent.open) target = parent;
      }
      return target.getBoundingClientRect().top;
    }, top + 16);
    if (index >= 0 && index !== selected) select(index);
  }

  function schedulePosition() {
    if (!scheduled) {
      scheduled = true;
      requestAnimationFrame(updatePosition);
    }
  }

  if (points.length) {
    const paths = transcriptCostPaths(points);
    for (const name of ['known', 'partial', 'gaps']) document.getElementById('cost-' + name).setAttribute('d', paths[name]);
    function pointerIndex(event, surface = svg) {
      const box = surface.getBoundingClientRect();
      const fraction = (event.clientX - box.left) / box.width;
      return transcriptCostIndex(points.length, fraction);
    }
    svg.addEventListener('pointermove', event => {
      const index = pointerIndex(event);
      preview.textContent = describe(index) + ' · ' + points[index].model + ' — ' + points[index].excerpt;
      preview.hidden = false;
    });
    svg.addEventListener('pointerleave', () => { preview.hidden = true; });
    svg.addEventListener('click', event => { preview.hidden = true; select(pointerIndex(event), true); });
    selector.addEventListener('pointermove', event => {
      const index = pointerIndex(event, selector);
      const bucket = buckets.find(item => index >= item.start && index < item.end);
      preview.textContent = `Responses ${bucket.start + 1}–${bucket.end} · Group cost ${bucket.label} · ` + describe(index);
      preview.hidden = false;
    });
    selector.addEventListener('pointerleave', () => { preview.hidden = true; });
    selector.addEventListener('click', event => { preview.hidden = true; select(pointerIndex(event, selector), true); });
    selector.addEventListener('keydown', event => {
      let index = selected;
      if (event.key === 'ArrowRight' || event.key === 'ArrowUp') index++;
      else if (event.key === 'ArrowLeft' || event.key === 'ArrowDown') index--;
      else if (event.key === 'Home') index = 0;
      else if (event.key === 'End') index = points.length - 1;
      else return;
      event.preventDefault();
      select(Math.max(0, Math.min(points.length - 1, index)), true);
    });
    jump.addEventListener('click', event => { event.preventDefault(); select(selected, true); });
    select(0);
  }
  // Programmatic scrolling can settle late or at a clamped position. Keep the
  // chosen response until the reader starts scrolling, not for an arbitrary timer.
  const resumeFollowing = () => { explicitSelection = false; };
  window.addEventListener('wheel', resumeFollowing, {passive: true});
  window.addEventListener('touchstart', resumeFollowing, {passive: true});
  window.addEventListener('pointerdown', event => {
    if (!timeline?.contains(event.target)) resumeFollowing();
  }, {passive: true});
  window.addEventListener('keydown', event => {
    if (!event.defaultPrevented && ['ArrowUp', 'ArrowDown', 'PageUp', 'PageDown', 'Home', 'End', ' '].includes(event.key)) resumeFollowing();
  });
  window.addEventListener('scroll', schedulePosition, {passive: true});
  window.addEventListener('resize', schedulePosition);
  thread.addEventListener('toggle', schedulePosition, true);
  thread.addEventListener('load', schedulePosition, true);
  if ('ResizeObserver' in window) {
    const resize = new ResizeObserver(schedulePosition);
    resize.observe(toolbar);
    resize.observe(thread);
  }
  updatePosition();
  openHash();
});
