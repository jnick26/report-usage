"""Start one loopback server with writable data outside installed assets."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import errno
from pathlib import Path
import socket
from threading import Timer
from typing import Iterator
import webbrowser
from zoneinfo import ZoneInfoNotFoundError


def default_data_dir() -> Path:
    return Path.home() / 'Library' / 'Application Support' / 'Harness Usage'


@contextmanager
def data_lock(directory: Path) -> Iterator[None]:
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'runtime.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('Harness Usage is already running with this data directory.') from error
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def main() -> None:
    parser = argparse.ArgumentParser(description='Read local Pi, Codex, Claude Code, Copilot in VS Code, and Copilot CLI usage history in an offline browser application.')
    parser.add_argument('--data-dir', type=Path, default=default_data_dir())
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--root', action='append', default=[], help='Local session-storage directories for these products; repeat for multiple roots')
    parser.add_argument('--timezone', help='IANA reporting timezone, defaults to the system timezone')
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error('Port must be between 1 and 65535.')
    # Bind before constructing Application, so failed startup cannot disturb run state.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            listener.bind(('127.0.0.1', args.port))
        except OSError as error:
            parser.error('The local port is unavailable. Close the other app or choose --port.' if error.errno == errno.EADDRINUSE else 'Could not open a local listener: ' + str(error.strerror))
        listener.listen(128)
        try:
            with data_lock(args.data_dir.expanduser().resolve()):
                import uvicorn
                from harness_usage.application import Application
                from harness_usage.web import create_app
                application = Application(args.data_dir.expanduser().resolve(), timezone=args.timezone)
                try:
                    if args.root:
                        application.set_roots(tuple(str(Path(root).expanduser().resolve()) for root in args.root))
                    if application.get_roots():
                        application.start_import()
                    if not args.no_browser:
                        timer = Timer(0.7, webbrowser.open, args=(f'http://127.0.0.1:{args.port}/',))
                        timer.daemon = True
                        timer.start()
                    server = uvicorn.Server(uvicorn.Config(create_app(application), host='127.0.0.1', port=args.port, log_level='warning', access_log=False, timeout_graceful_shutdown=3))
                    server.run(sockets=[listener])
                finally:
                    application.close()
        except ZoneInfoNotFoundError:
            parser.error('Unknown IANA timezone. Use a name such as Europe/Kyiv or UTC.')
        except (RuntimeError, ValueError, OSError) as error:
            parser.error(str(error))


if __name__ == '__main__':
    main()
