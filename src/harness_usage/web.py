"""Loopback-only HTML application; Unpoly enhances ordinary GET/POST forms."""
from __future__ import annotations

import asyncio
import json
import re
import secrets
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from pathlib import Path
from typing import TYPE_CHECKING, AsyncIterator, Literal, cast
from urllib.parse import quote, urlencode, urlsplit

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.datastructures import FormData
from starlette.middleware.base import RequestResponseEndpoint
from starlette.responses import Response
from starlette.concurrency import run_in_threadpool

if TYPE_CHECKING:
    from .application import Application

from .application import SourceChangeDuringImport
from .domain import ProjectId
from .presentation import harness_label
from .transcript import TranscriptUnavailable
from .transcript_rendering import render_transcript, transcript_csp
from .reporting import AllTime, Bucket, DateRange, MetricSum, Report, ReportQuery, SessionSort, format_mtok, parse_range, project_label

ASSETS = Path(__file__).parent
def preset_range(preset: str, timezone: str, now: datetime | None = None) -> DateRange | AllTime:
    zone = ZoneInfo(timezone)
    local = (now or datetime.now(UTC)).astimezone(zone)
    if preset == 'all':
        return AllTime()
    start = local.replace(hour=0, minute=0, second=0, microsecond=0)
    if preset == 'month':
        start = start.replace(day=1)
        end = (start.replace(day=28) + timedelta(days=4)).replace(day=1)
    elif preset == 'week':
        start -= timedelta(days=start.weekday())
        end = start + timedelta(days=7)
    elif preset == 'day':
        end = start + timedelta(days=1)
    elif preset == 'hour':
        start = local.replace(minute=0, second=0, microsecond=0)
        end = (start.astimezone(UTC) + timedelta(hours=1)).astimezone(zone)
    else:
        raise ValueError('Choose Month, Week, Day, Hour or All time.')
    return DateRange(start, end, timezone)


def range_query(project: str | None, selected: DateRange | AllTime) -> ReportQuery:
    bucket: Literal['day', 'hour', 'five_minutes'] = 'day'
    if isinstance(selected, DateRange):
        seconds = (selected.end - selected.start).total_seconds()
        bucket = 'five_minutes' if seconds <= 3600 else 'hour' if seconds <= 172800 else 'day'
    return ReportQuery(ProjectId(project) if project else None, selected, bucket)



def exact_mtok(value: int) -> str:
    return f'{value // 1_000_000}.{value % 1_000_000:06d}'.rstrip('0').rstrip('.')


def decimal_text(value: Decimal) -> str:
    text = format(value, 'f')
    return text.rstrip('0').rstrip('.') if '.' in text else text


def bucket_tooltip(bucket: Bucket) -> str:
    def amount(value: MetricSum) -> str:
        if value.lower_bound_observations:
            return '≥ ' + exact_mtok(value.known)
        if value.known == 0 and value.unknown_observations:
            return '—'
        if value.known == 0 and value.not_applicable_observations:
            return 'N/A'
        return format_mtok(value.known)
    tokens = bucket.tokens
    money = bucket.money
    cost = '$—' if money is None or money.known == 0 and money.missing_observations else (
        '<$0.01' if 0 < money.known < Decimal('0.01') else f'${money.known:.2f}')
    return (f'Input {amount(tokens.input)} · Output {amount(tokens.output)} Mtok\n'
            f'Cache read {amount(tokens.cache_read)} · Write {amount(tokens.cache_write)} Mtok\n{cost}')


def import_error(code: str | None) -> str:
    match = re.fullmatch(r'source_(\d+)_(unavailable|read_failed)', code or '')
    if match:
        issue = 'is unavailable' if match[2] == 'unavailable' else 'could not be fully read'
        return f'Source {match[1]} {issue}. Showing saved data. Check Sources and retry.'
    return 'Import failed. Showing saved data. Check Sources and retry.'


def create_app(application: Application) -> FastAPI:
    from .reporting import COVERAGE_TEXT
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            await run_in_threadpool(application.close)

    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    token = secrets.token_urlsafe(32)
    templates = Jinja2Templates(directory=ASSETS / 'templates')
    templates.env.globals['transcript_url'] = lambda identity: '/sessions/' + quote(str(identity), safe='') + '/transcript'
    templates.env.filters['import_error'] = import_error
    templates.env.filters['mtok'] = format_mtok
    templates.env.filters['bucket_tooltip'] = bucket_tooltip
    templates.env.filters['usd'] = lambda value: f'{value:.2f}'
    templates.env.filters['exact_mtok'] = exact_mtok
    templates.env.filters['decimal_text'] = decimal_text
    templates.env.filters['localdate'] = lambda value, timezone: value.astimezone(ZoneInfo(timezone)).strftime('%d %b %Y %H:%M') if value else 'Unknown'
    templates.env.filters['harness_label'] = harness_label
    templates.env.filters['coverage_text'] = lambda code: COVERAGE_TEXT.get(code, '')
    templates.env.filters['quantity_label'] = lambda measure: {'ai_credits': 'AI credits', 'nano_aiu': 'nano-AIU', 'premium_requests': 'Premium requests', 'request_count': 'Request count'}.get(measure, 'Unknown measure')
    templates.env.filters['quantity_unit'] = lambda measure: {'ai_credits': 'AI credits', 'nano_aiu': 'nano-AIU', 'premium_requests': 'premium requests', 'request_count': 'requests'}.get(measure, 'Unknown unit')
    app.mount('/static', StaticFiles(directory=ASSETS / 'static'), name='static')

    @app.middleware('http')
    async def local_only(request: Request, call_next: RequestResponseEndpoint) -> Response:
        host = request.headers.get('host', '')
        try:
            parsed = urlsplit('http://' + host)
            allowed = parsed.hostname in {'localhost', '127.0.0.1', '::1'} and parsed.username is None and parsed.path == '' and not parsed.query and not parsed.fragment
            _ = parsed.port
        except ValueError:
            allowed = False
        if not allowed:
            return HTMLResponse('Unrecognized local host.', status_code=400)
        origin = request.headers.get('origin')
        if request.method not in {'GET', 'HEAD', 'OPTIONS'} and origin and origin != f'{request.url.scheme}://{host}':
            return HTMLResponse('Foreign origin rejected.', status_code=403)
        response = await call_next(request)
        response.headers.setdefault('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; font-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Referrer-Policy'] = 'no-referrer'
        response.headers['Cache-Control'] = 'no-store'
        return response

    def render(request: Request, name: str, *, status_code: int = 200, **context: object) -> Response:
        return templates.TemplateResponse(request=request, name=name, context={'csrf': token, 'status': application.status(), 'saved_roots': application.get_roots(), **context}, status_code=status_code)

    async def command_form(request: Request) -> FormData | None:
        form = await request.form(max_fields=20, max_files=0)
        supplied = form.get('csrf')
        return form if isinstance(supplied, str) and supplied.isascii() and secrets.compare_digest(supplied, token) else None

    def transcript_response(request: Request, session_id: str, *, download: bool = False) -> Response:
        try:
            page = application.transcript(session_id, request.query_params.get('branch'))
        except TranscriptUnavailable as error:
            status = 404 if error.kind == 'missing' else 413 if error.kind == 'too_large' else 409 if error.kind == 'changed' else 422
            return templates.TemplateResponse(request=request, name='transcript_unavailable.html',
                context={'reason': error.reason}, status_code=status)
        headers = {'Content-Security-Policy': transcript_csp(standalone=download)}
        if download:
            headers['Content-Disposition'] = 'attachment; filename="session-transcript.html"'
        return HTMLResponse(render_transcript(page, standalone=download), headers=headers)

    @app.get('/sessions/{session_id:path}/transcript', response_class=HTMLResponse)
    def session_transcript(request: Request, session_id: str) -> Response:
        return transcript_response(request, session_id)

    @app.get('/sessions/{session_id:path}/transcript.html', response_class=HTMLResponse)
    def download_transcript(request: Request, session_id: str) -> Response:
        return transcript_response(request, session_id, download=True)

    @app.get('/', response_class=HTMLResponse)
    def overview(request: Request) -> Response:
        if not application.get_roots():
            return RedirectResponse('/sources', status_code=303)
        params = request.query_params
        timezone = params.get('tz', getattr(application, 'timezone', 'UTC'))
        project = params.get('project') or None
        preset = params.get('preset', 'all')
        error = None
        requested_sort = params.get('sort', 'started')
        if requested_sort not in ('started', 'cost_desc', 'cost_asc'):
            error = 'Choose a valid session sort.'
            requested_sort = 'started'
        session_sort = cast(SessionSort, requested_sort)
        try:
            page = int(params.get('page', '1')) if 'preset' not in params else 1
            if page < 1:
                raise ValueError('invalid_page')
        except ValueError:
            page = 1
            error = 'Choose a positive session page number.'

        def load(selected: DateRange | AllTime) -> Report:
            return application.report(range_query(project, selected), include_sessions=bool(project), session_page=page if project else None, session_sort=session_sort)

        try:
            ZoneInfo(timezone)
            selected = (parse_range(params.get('offset_from') or params.get('from', ''), params.get('offset_to') or params.get('to', ''), timezone)
                        if ('from' in params or 'to' in params) and 'preset' not in params
                        else preset_range(preset, timezone))
            report = load(selected)
        except (ValueError, ZoneInfoNotFoundError, OverflowError):
            error = 'Choose valid dates with an end after the start and a manageable range. If a clock change makes that hour unavailable or ambiguous, choose a different hour. The previous report remains visible.'
            timezone = params.get('previous_tz', getattr(application, 'timezone', 'UTC'))
            try:
                selected = parse_range(params.get('previous_from', ''), params.get('previous_to', ''), timezone)
                report = load(selected)
            except (ValueError, ZoneInfoNotFoundError, OverflowError):
                timezone = getattr(application, 'timezone', 'UTC')
                selected = AllTime()
                report = load(selected)
        zone = ZoneInfo(timezone)
        start = selected.start.astimezone(zone).isoformat() if isinstance(selected, DateRange) else ''
        end = selected.end.astimezone(zone).isoformat() if isinstance(selected, DateRange) else ''
        native_start = selected.start.astimezone(zone).replace(tzinfo=None).isoformat(timespec='auto' if selected.start.second or selected.start.microsecond else 'minutes') if isinstance(selected, DateRange) else ''
        native_end = selected.end.astimezone(zone).replace(tzinfo=None).isoformat(timespec='auto' if selected.end.second or selected.end.microsecond else 'minutes') if isinstance(selected, DateRange) else ''
        valid_params = {'tz': timezone, 'sort': session_sort}
        if project:
            valid_params['project'] = project
        if start:
            valid_params.update({'from': start, 'to': end})
        def link(target: str | None) -> str:
            values = {k: v for k, v in valid_params.items() if k not in {'project', 'page'}}
            if target:
                values['project'] = target
            return '/?' + urlencode(values)
        if project and report.session_page and report.session_page > 1:
            valid_params['page'] = str(report.session_page)

        def page_link(target: int) -> str:
            return '/?' + urlencode({**valid_params, 'page': str(target)})

        def sort_link(target: str) -> str:
            return '/?' + urlencode({**{k: v for k, v in valid_params.items() if k != 'page'}, 'sort': target})

        project_path = project.partition(':')[2] if project and project.startswith(('git:', 'directory:')) else None
        if project_path and project and project.startswith('git:') and Path(project_path).name == '.git':
            project_path = str(Path(project_path).parent)
        name = next((p.label for p in report.projects if str(p.id or 'unassigned') == project), project_label(ProjectId(project) if project else None))
        max_bucket = max((b.tokens.total.known for b in report.buckets), default=1) or 1
        response = render(request, 'report.html', report=report, project=project, project_name=name, project_path=project_path, timezone=timezone,
                      start=start, end=end, entered_start=params.get('from', start) if error else start,
                      entered_end=params.get('to', end) if error else end, error=error,
                      preset=preset if 'preset' in params or isinstance(selected, AllTime) else '',
                      max_bucket=max_bucket, project_link=link, canonical='/?' + urlencode(valid_params),
                      page_link=page_link, sort_link=sort_link, session_sort=session_sort,
                      native_start=params.get('from', native_start) if error else native_start, native_end=params.get('to', native_end) if error else native_end, status_code=422 if error else 200)
        response.headers['X-Report-Revision'] = str(report.revision)
        return response

    @app.get('/sources', response_class=HTMLResponse)
    def sources(request: Request) -> Response:
        return render(request, 'sources.html', roots='\n'.join(application.get_roots()), error=None)

    @app.post('/sources')
    async def save_sources(request: Request) -> Response:
        form = await command_form(request)
        if form is None:
            return HTMLResponse('Invalid form token. Reload and try again.', status_code=403)
        entered = str(form.get('roots', ''))
        roots = tuple(dict.fromkeys(line.strip() for line in entered.splitlines() if line.strip()))
        if not roots or len(roots) > 32 or any(not Path(p).expanduser().is_dir() for p in roots):
            return render(request, 'sources.html', roots=entered, error='Enter an existing, readable directory on each line.', status_code=422)
        try:
            await run_in_threadpool(application.set_roots, tuple(str(Path(p).expanduser().resolve()) for p in roots))
            await run_in_threadpool(application.start_import)
        except SourceChangeDuringImport:
            return render(request, 'sources.html', roots=entered, error='Wait for the current import to finish, then save your sources again.', status_code=422)
        except (OSError, ValueError):
            return render(request, 'sources.html', roots=entered, error='Sources could not be saved. Check the directories and try again.', status_code=422)
        return RedirectResponse('/', status_code=303)

    @app.post('/import')
    async def refresh(request: Request) -> Response:
        form = await command_form(request)
        if form is None:
            return HTMLResponse('Invalid form token. Reload and try again.', status_code=403)
        await run_in_threadpool(application.start_import)
        target = str(form.get('return_to', '/'))
        if not target.startswith('/') or target.startswith('//') or '\\' in target:
            target = '/'
        return RedirectResponse(target, status_code=303)

    @app.get('/events')
    async def events(request: Request) -> StreamingResponse:
        async def updates() -> AsyncIterator[str]:
            previous = ''
            while not await request.is_disconnected():
                current = json.dumps(asdict(await run_in_threadpool(application.status)), sort_keys=True)
                if current != previous:
                    yield f'event: status\ndata: {current}\n\n'
                    previous = current
                else:
                    yield ': keepalive\n\n'
                # Pull the latest snapshot: slow readers retain no event queue.
                await asyncio.sleep(1)
        return StreamingResponse(updates(), media_type='text/event-stream', headers={'X-Accel-Buffering': 'no'})

    return app
