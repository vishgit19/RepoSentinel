"""Run workspaces.

Every agent run gets its own copy of the repository under
``data/workspaces/<run_id>``. The copy is git-initialised with a single
"baseline" commit, which gives three things for free:

* ``git diff`` produces a real unified diff of whatever the agent changed,
* ``git checkout`` can roll a failed patch attempt back cleanly,
* the original repository on disk is never modified.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from reposentinel.benchmarks import get_benchmark
from reposentinel.config import Settings, get_settings

IGNORED_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    "node_modules",
    ".idea",
    ".vscode",
    "dist",
    "build",
    ".eggs",
}
IGNORED_SUFFIXES = {".pyc", ".pyo", ".so", ".dll", ".exe", ".zip", ".tar", ".gz"}

# Tool reports (junit xml, scan output) are written here and hidden from git.
ARTIFACTS_DIRNAME = ".reposentinel"

GIT_IDENTITY = (
    "-c",
    "user.email=agent@reposentinel.local",
    "-c",
    "user.name=RepoSentinel",
    "-c",
    "commit.gpgsign=false",
    "-c",
    "core.autocrlf=false",
)


class WorkspaceError(RuntimeError):
    pass


@dataclass
class GitResult:
    exit_code: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.exit_code == 0


def _run_git(repo: Path, args: list[str], timeout: int = 120) -> GitResult:
    """Run git in the workspace.

    This is RepoSentinel's own trusted plumbing, not an agent tool: the agent's
    git access goes through the sandboxed ``git_*`` tools which forbid pushes
    and remote mutation.
    """
    completed = subprocess.run(  # noqa: S603
        ["git", *GIT_IDENTITY, *args],
        cwd=str(repo),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        shell=False,
        check=False,
    )
    return GitResult(completed.returncode, completed.stdout or "", completed.stderr or "")


def _on_remove_error(func, target, _exc) -> None:  # noqa: ANN001
    """Clear the read-only bit and retry a failed deletion.

    Git marks objects and pack files read-only, which makes ``shutil.rmtree``
    fail on Windows. Swallowing that error (``ignore_errors=True``) is worse
    than useless here: it leaves a partial ``.git`` behind, and a partial
    ``.git`` is not a valid repository, so git's directory discovery walks
    *upwards* and quietly operates on RepoSentinel's own checkout instead.
    """
    try:
        os.chmod(target, stat.S_IWRITE)
        func(target)
    except OSError:
        pass


def remove_tree(path: Path) -> bool:
    """Delete a directory tree as thoroughly as the platform allows."""
    if not path.exists():
        return True
    handler = {"onexc": _on_remove_error} if sys.version_info >= (3, 12) else {"onerror": _on_remove_error}
    for attempt in range(3):
        shutil.rmtree(path, **handler)
        if not path.exists():
            return True
        # A virus scanner or editor may hold a brief lock; give it a moment.
        time.sleep(0.1 * (attempt + 1))
    return not path.exists()


def _copy_tree(source: Path, destination: Path) -> int:
    """Copy a repository tree, skipping build artefacts and VCS metadata."""
    copied = 0
    for item in source.rglob("*"):
        relative = item.relative_to(source)
        if any(part in IGNORED_DIRS for part in relative.parts):
            continue
        if item.is_dir():
            (destination / relative).mkdir(parents=True, exist_ok=True)
            continue
        if item.suffix.lower() in IGNORED_SUFFIXES:
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, target)
        copied += 1
    return copied


class Workspace:
    """An isolated, git-tracked copy of a repository for one run."""

    def __init__(self, root: Path, source_label: str, settings: Settings | None = None) -> None:
        self.root = root.resolve()
        self.source_label = source_label
        self.settings = settings or get_settings()
        self.baseline_commit: str = ""

    # -- construction ----------------------------------------------------
    @classmethod
    def prepare(
        cls,
        source: str,
        run_id: str,
        settings: Settings | None = None,
    ) -> Workspace:
        """Materialise a workspace from a benchmark id, local path or Git URL."""
        settings = settings or get_settings()
        destination = settings.workspaces_dir / run_id
        if destination.exists() and not remove_tree(destination):
            raise WorkspaceError(
                f"could not clear the existing workspace at {destination}; "
                "remove it manually before re-running"
            )
        destination.mkdir(parents=True, exist_ok=True)

        benchmark = get_benchmark(source)
        if benchmark is not None:
            _copy_tree(benchmark.repo_path, destination)
            label = f"benchmark:{benchmark.id}"
        elif source.startswith(("http://", "https://", "git@")):
            result = _run_git(
                destination.parent,
                ["clone", "--depth", "50", source, str(destination)],
                timeout=300,
            )
            if not result.ok:
                raise WorkspaceError(f"git clone failed: {result.stderr.strip()[:400]}")
            label = source
        else:
            local = Path(source).expanduser()
            if not local.is_dir():
                raise WorkspaceError(
                    f"'{source}' is not a known benchmark id, an existing directory, or a Git URL"
                )
            _copy_tree(local.resolve(), destination)
            label = str(local.resolve())

        workspace = cls(destination, label, settings=settings)
        workspace._init_baseline()
        return workspace

    @staticmethod
    def validate_source(source: str) -> str:
        """Check a source is loadable, without copying anything yet.

        ``prepare`` runs on a worker thread, so without this an unusable repo
        would only surface as a failed run several seconds later. Callers use it
        to reject the request while the user is still looking at the form.
        """
        cleaned = (source or "").strip()
        if not cleaned:
            raise WorkspaceError("a benchmark id, local path or Git URL is required")
        if get_benchmark(cleaned) is not None:
            return f"benchmark:{cleaned}"
        if cleaned.startswith(("http://", "https://", "git@")):
            return cleaned
        local = Path(cleaned).expanduser()
        if local.is_dir():
            return str(local.resolve())
        raise WorkspaceError(
            f"'{source}' is not a known benchmark id, an existing directory, or a Git URL"
        )

    @property
    def artifacts_dir(self) -> Path:
        """Scratch space for tool reports, excluded from git so diffs stay clean."""
        path = self.root / ARTIFACTS_DIRNAME
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _own_repo_root(self) -> Path | None:
        """The root of the git repository that git commands here resolve to."""
        result = _run_git(self.root, ["rev-parse", "--show-toplevel"])
        if not result.ok or not result.stdout.strip():
            return None
        return Path(result.stdout.strip()).resolve()

    def _init_baseline(self) -> None:
        # The presence of a .git directory is not proof of a usable repository:
        # a half-deleted one makes git search upwards and target the parent
        # checkout instead. Ask git what it actually resolves to.
        if self._own_repo_root() != self.root:
            init = _run_git(self.root, ["init", "--quiet"])
            if not init.ok:
                raise WorkspaceError(f"git init failed: {init.stderr.strip()[:400]}")
        # Keep RepoSentinel's own scratch files out of every diff without
        # touching a .gitignore that belongs to the repository under repair.
        exclude_file = self.root / ".git" / "info" / "exclude"
        exclude_file.parent.mkdir(parents=True, exist_ok=True)
        exclude_file.write_text(
            f"{ARTIFACTS_DIRNAME}/\n.pytest_cache/\n__pycache__/\n*.pyc\n", encoding="utf-8"
        )
        _run_git(self.root, ["add", "-A"])
        commit = _run_git(
            self.root, ["commit", "--quiet", "--allow-empty", "-m", "reposentinel: baseline"]
        )
        if not commit.ok and "nothing to commit" not in commit.stdout.lower():
            raise WorkspaceError(f"baseline commit failed: {commit.stderr.strip()[:400]}")
        head = _run_git(self.root, ["rev-parse", "HEAD"])
        self.baseline_commit = head.stdout.strip() if head.ok else ""

        # Refuse to hand back a workspace whose git commands would touch
        # anything other than the workspace itself. Every mutating call below
        # (add, commit, checkout, clean) trusts this invariant.
        resolved = self._own_repo_root()
        if resolved != self.root:
            raise WorkspaceError(
                f"workspace git isolation failed: commands in {self.root} resolve to "
                f"{resolved or '<no repository>'}"
            )

    # -- inspection ------------------------------------------------------
    def diff(self, staged_too: bool = True) -> str:
        """Unified diff of every uncommitted change."""
        _run_git(self.root, ["add", "-A", "-N"])  # intent-to-add so new files appear
        args = ["diff", "--no-color", "--unified=3"]
        result = _run_git(self.root, args)
        return result.stdout if result.ok else ""

    def diff_stat(self) -> tuple[list[str], int, int]:
        """(changed files, lines added, lines removed) versus the baseline."""
        _run_git(self.root, ["add", "-A", "-N"])
        result = _run_git(self.root, ["diff", "--numstat"])
        files: list[str] = []
        added = removed = 0
        for line in result.stdout.splitlines():
            parts = line.split("\t")
            if len(parts) != 3:
                continue
            plus, minus, path = parts
            files.append(path.strip())
            added += int(plus) if plus.isdigit() else 0
            removed += int(minus) if minus.isdigit() else 0
        return files, added, removed

    def changed_files(self) -> list[str]:
        return self.diff_stat()[0]

    def file_history(self, relative_path: str, limit: int = 10) -> list[dict[str, str]]:
        """Commits touching a path (empty for benchmark copies with one commit)."""
        result = _run_git(
            self.root,
            [
                "log",
                f"-{limit}",
                "--pretty=format:%h%x1f%an%x1f%ar%x1f%s",
                "--",
                relative_path,
            ],
        )
        commits: list[dict[str, str]] = []
        for line in result.stdout.splitlines():
            fields = line.split("\x1f")
            if len(fields) == 4:
                commits.append(
                    {
                        "commit": fields[0],
                        "author": fields[1],
                        "when": fields[2],
                        "subject": fields[3],
                    }
                )
        return commits

    # -- mutation --------------------------------------------------------
    def snapshot(self, label: str) -> str:
        """Commit the current tree so it can be restored later."""
        _run_git(self.root, ["add", "-A"])
        _run_git(self.root, ["commit", "--quiet", "--allow-empty", "-m", f"reposentinel: {label}"])
        head = _run_git(self.root, ["rev-parse", "HEAD"])
        return head.stdout.strip() if head.ok else ""

    def restore_baseline(self) -> None:
        """Discard every change since the baseline commit."""
        if not self.baseline_commit:
            return
        _run_git(self.root, ["checkout", "--force", self.baseline_commit, "--", "."])
        _run_git(self.root, ["clean", "-fdq"])

    def cleanup(self) -> bool:
        """Delete the workspace. Returns False if anything survived."""
        return remove_tree(self.root)

    # -- helpers ---------------------------------------------------------
    def relative_files(self, extensions: tuple[str, ...] = (".py",)) -> list[str]:
        results: list[str] = []
        for item in sorted(self.root.rglob("*")):
            if not item.is_file():
                continue
            relative = item.relative_to(self.root)
            if any(part in IGNORED_DIRS or part == ARTIFACTS_DIRNAME for part in relative.parts):
                continue
            if extensions and item.suffix.lower() not in extensions:
                continue
            results.append(relative.as_posix())
        return results

    def describe(self) -> dict[str, object]:
        return {
            "root": str(self.root),
            "source": self.source_label,
            "baseline_commit": self.baseline_commit[:8],
            "python_files": len(self.relative_files()),
            "created_at": time.time(),
        }
