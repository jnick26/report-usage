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
  let importing = false;
  let latestRevision = 0;
  function refreshReport() {
    const report = document.querySelector('#report');
    if (!report || refreshing || importing || latestRevision <= Number(report.dataset.revision)) return;
    refreshing = true;
    up.reload('#report', {
      url: report.dataset.canonical, history: false, cache: false,
      focus: 'keep', scroll: false, abort: 'target',
      onLoaded(event) {
        if (document.querySelector('#report') !== report) event.skip();
      }
    }).catch(() => {
      clearTimeout(retryTimer);
      retryTimer = setTimeout(refreshReport, 3000);
    }).finally(() => { refreshing = false; });
  }
  function connect() {
    stream = new EventSource('/events');
    stream.addEventListener('status', event => {
    const state = JSON.parse(event.data);
    importing = state.state === 'running';
    const source = /^source_(\d+)_(unavailable|read_failed)$/.exec(state.error || '');
    const failure = source ? `Source ${source[1]} ${source[2] === 'unavailable' ? 'is unavailable' : 'could not be fully read'}. Showing saved data. Check Sources and retry.` : 'Import failed. Showing saved data. Check Sources and retry.';
    latestRevision = state.revision;
    const line = document.querySelector('#import-status');
    if (line) {
      line.dataset.state = state.state;
      line.hidden = !['running', 'failed', 'interrupted'].includes(state.state);
      const text = state.state === 'running'
        ? `Importing local history · ${state.files_processed} files processed. Showing saved data.`
        : state.state === 'failed'
          ? failure
          : state.state === 'interrupted'
            ? 'The previous import was interrupted. Saved history remains available.'
            : 'Saved local history';
      line.querySelector('span').textContent = text;
      line.querySelector('button').textContent = ['failed', 'interrupted'].includes(state.state) ? 'Retry import' : 'Refresh';
    }
    refreshReport();
  });
  }
  connect();
  up.on('up:fragment:inserted', refreshReport);
  window.addEventListener('pagehide', () => { stream.close(); clearTimeout(retryTimer); });
  window.addEventListener('pageshow', event => { if (event.persisted) connect(); });
}
