"""Read-only local project identity; a remote URL never establishes identity."""
import os
from pathlib import Path
import subprocess

from .domain import Assigned, Attribution, ProjectId, Unassigned


def resolve_project(cwd: str | None, *, previous: Attribution | None = None) -> Attribution:
    if not cwd or not Path(cwd).is_absolute():
        return Unassigned('missing_or_relative_cwd')
    path = Path(cwd)
    if not path.is_dir():
        return previous if isinstance(previous, Assigned) else Unassigned('directory_unavailable')
    # Remove ambient Git overrides so identity belongs to the recorded directory.
    environment = {key: value for key, value in os.environ.items() if not key.startswith('GIT_')}
    environment['GIT_OPTIONAL_LOCKS'] = '0'
    try:
        result = subprocess.run(
            ['git', '-C', str(path), 'rev-parse', '--path-format=absolute', '--show-toplevel', '--git-common-dir'],
            capture_output=True, text=True, timeout=5, env=environment, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return previous if isinstance(previous, Assigned) else Unassigned('git_discovery_unavailable')
    if result.returncode == 0:
        lines = result.stdout.splitlines()
        if len(lines) != 2:
            return Unassigned('git_metadata_unavailable')
        worktree, common = (str(Path(value).resolve()) for value in lines)
        return Assigned(ProjectId('git:' + common), worktree, 'git_common_dir')
    if 'not a git repository' not in result.stderr:
        return previous if isinstance(previous, Assigned) else Unassigned('git_metadata_unavailable')
    return Assigned(ProjectId('directory:' + str(path.resolve())), str(path), 'recorded_directory')
