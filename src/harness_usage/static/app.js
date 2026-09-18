/* The server renders every total; events only request a fresh committed report. */
if (window.up && window.EventSource) {
  up.on('up:fragment:loaded', event => {
    const shown = document.querySelector('#report');
    const revision = event.response.header('X-Report-Revision');
    if (shown && revision !== undefined && revision !== null && Number(revision) < Number(shown.dataset.revision)) event.skip();
  });
  let stream;
  let retryTimer;
  let refreshing = false;
  let latestRevision = 0;
  let latestStatus;
  function refreshReport() {
    const report = document.querySelector('#report');
    if (!report || refreshing || latestRevision <= Number(report.dataset.revision)) return;
    refreshing = true;
    up.reload('#report', {
      url: report.dataset.canonical, history: false, cache: false,
      focus: 'keep', scroll: false, abort: 'target',
      onLoaded(event) {
        if (document.querySelector('#report') !== report) event.skip();
      }
    }).catch(() => {}).finally(() => {
      refreshing = false;
      clearTimeout(retryTimer);
      retryTimer = setTimeout(refreshReport, 3000);
    });
  }
  function renderStatus(state) {
    if (!state) return;
    const source = /^source_(\d+)_(unavailable|read_failed)$/.exec(state.error || '');
    const failure = source ? `Source ${source[1]} ${source[2] === 'unavailable' ? 'is unavailable' : 'could not be fully read'}. Showing saved data. Check Sources and retry.` : 'Import failed. Showing saved data. Check Sources and retry.';
    const line = document.querySelector('#import-status');
    if (line) {
      line.dataset.state = state.state;
      line.hidden = !['running', 'failed', 'interrupted'].includes(state.state);
      const checking = state.phase === 'checking' && state.files_total > 0;
      const progressText = state.phase === 'finalizing' ? 'Finalizing'
        : checking ? `${state.files_checked} / ${state.files_total} files checked · ${Math.floor(100 * state.files_checked / state.files_total)}%`
        : 'Discovering files';
      const text = state.state === 'running'
        ? `${progressText}. Showing partial results; updating automatically.`
        : state.state === 'failed'
          ? failure
          : state.state === 'interrupted'
            ? 'The previous import was interrupted. Saved history remains available.'
            : 'Saved local history';
      line.querySelector('span').textContent = text;
      const progress = line.querySelector('progress');
      progress.hidden = state.state !== 'running';
      if (state.state === 'running' && checking) {
        progress.max = state.files_total;
        progress.value = state.files_checked;
      } else {
        progress.removeAttribute('value');
      }
      line.querySelector('button').textContent = ['failed', 'interrupted'].includes(state.state) ? 'Retry import' : 'Refresh';
    }
  }
  function connect() {
    stream = new EventSource('/events');
    stream.addEventListener('status', event => {
      latestStatus = JSON.parse(event.data);
      latestRevision = latestStatus.revision;
      renderStatus(latestStatus);
      refreshReport();
    });
  }
  connect();
  up.on('up:fragment:inserted', () => { renderStatus(latestStatus); refreshReport(); });
  window.addEventListener('pagehide', () => { stream.close(); clearTimeout(retryTimer); });
  window.addEventListener('pageshow', event => { if (event.persisted) connect(); });
}
