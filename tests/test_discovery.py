from pathlib import Path
import subprocess

from harness_usage.discovery import resolve_project
from harness_usage.domain import Assigned, Unassigned


def git(path: Path, *arguments: str) -> None:
    subprocess.run(['git', '-C', str(path), *arguments], check=True, capture_output=True)


def test_common_git_identity_keeps_worktrees_and_clones_separate(tmp_path: Path) -> None:
    repo = tmp_path / 'repository'
    repo.mkdir()
    git(repo, 'init')
    git(repo, '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid', 'commit', '--allow-empty', '-m', 'Initial')
    linked = tmp_path / 'linked'
    git(repo, 'worktree', 'add', str(linked), '-b', 'linked')
    clone = tmp_path / 'clone'
    git(tmp_path, 'clone', str(repo), str(clone))
    nested = linked / 'nested'
    nested.mkdir()
    original = resolve_project(str(repo))
    worktree = resolve_project(str(nested))
    copied = resolve_project(str(clone))
    assert isinstance(original, Assigned) and isinstance(worktree, Assigned) and isinstance(copied, Assigned)
    assert original.project_id == worktree.project_id
    assert original.worktree != worktree.worktree
    assert original.project_id != copied.project_id


def test_missing_directory_retains_only_previously_verified_attribution(tmp_path: Path) -> None:
    directory = tmp_path / 'notes'
    directory.mkdir()
    known = resolve_project(str(directory))
    assert isinstance(known, Assigned)
    assert known.worktree == str(directory)
    directory.rmdir()
    assert resolve_project(str(directory), previous=known) == known
    assert isinstance(resolve_project(str(directory)), Unassigned)
    assert isinstance(resolve_project(None), Unassigned)
    assert isinstance(resolve_project('relative/path'), Unassigned)
