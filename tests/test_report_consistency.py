"""Range projection must read one committed population across all SQL reads."""
from contextlib import contextmanager

from harness_usage.application import Application
from harness_usage.reporting import ReportQuery, parse_range, build_report
from harness_usage.storage import Storage
from test_report_storage import import_intervals


def test_report_keeps_interval_edges_and_one_revision_during_a_commit(tmp_path, monkeypatch):
    application = Application(tmp_path / 'app', timezone='UTC')
    storage = application.storage
    writer = Storage(storage.path)
    import_intervals(storage, monkeypatch, {'compaction': (1789200000000000, 1789207200000000)})

    # The known interval is 08:00–10:00. Touching either edge is not overlap.
    for start, end, known, unbucketed in [
        ('06:00', '08:00', 0, 0),
        ('10:00', '12:00', 0, 0),
        ('08:00', '10:00', 1005, 35),
    ]:
        query = ReportQuery(None, parse_range(f'2026-09-12T{start}Z', f'2026-09-12T{end}Z', 'UTC'))
        report = application.report(query)
        assert report.tokens.total.known == known
        assert report.unbucketed.total.known == unbucketed
        data = storage.report_input(query)
        assert report == build_report(data.contributions, revision=data.revision, query=query, catalog=application.catalog)

    before = application.report(query)
    original_connect = storage.connect
    reads = 0
    committed = False

    @contextmanager
    def interleaved_connect():
        with original_connect() as db:
            def after_first_read(statement):
                nonlocal reads, committed
                if statement.lstrip().upper().startswith(('SELECT', 'WITH')):
                    reads += 1
                    if reads == 2:
                        writer.mark_missing(('/auxiliary',))
                        committed = True
            db.set_trace_callback(after_first_read)
            yield db

    with monkeypatch.context() as patch:
        patch.setattr(storage, 'connect', interleaved_connect)
        during = application.report(query)
    after = application.report(query)
    assert committed
    assert during == before
    assert after.revision == before.revision + 1
    assert 'saved_history' not in {entry.code for entry in during.coverage}
    assert 'saved_history' in {entry.code for entry in after.coverage}
    assert 'saved_history' in storage.report_input(query).diagnostics
