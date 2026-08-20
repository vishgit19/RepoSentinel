"""Workspace isolation and diff generation."""

from __future__ import annotations

import pytest
from reposentinel.benchmarks import get_benchmark, list_benchmarks
from reposentinel.workspace import Workspace, WorkspaceError


class TestBenchmarkRegistry:
    def test_registry_is_not_empty(self):
        assert list_benchmarks(), "no benchmark manifests were discovered"

    def test_logic_bug_manifest_is_well_formed(self):
        manifest = get_benchmark("logic_bug")
        assert manifest is not None
        assert manifest.repo_path.is_dir()
        assert manifest.gold_files == ["app/auth/token.py"]
        assert manifest.expected_failing_tests
        # Gold files must actually exist in the fixture repository.
        for relative in manifest.relevant_files:
            assert (manifest.repo_path / relative).is_file(), relative

    def test_unknown_benchmark_returns_none(self):
        assert get_benchmark("does_not_exist") is None


class TestWorkspace:
    def test_prepare_from_benchmark_copies_and_initialises_git(self):
        workspace = Workspace.prepare("logic_bug", "test_ws_prepare")
        try:
            assert (workspace.root / "app" / "auth" / "token.py").is_file()
            assert (workspace.root / ".git").is_dir()
            assert len(workspace.baseline_commit) == 40
            assert workspace.diff() == ""
            assert workspace.changed_files() == []
        finally:
            workspace.cleanup()

    def test_source_repository_is_never_modified(self):
        manifest = get_benchmark("logic_bug")
        original = (manifest.repo_path / "app" / "auth" / "token.py").read_text(encoding="utf-8")
        workspace = Workspace.prepare("logic_bug", "test_ws_isolation")
        try:
            target = workspace.root / "app" / "auth" / "token.py"
            target.write_text("# clobbered\n", encoding="utf-8")
            after = (manifest.repo_path / "app" / "auth" / "token.py").read_text(encoding="utf-8")
            assert after == original
        finally:
            workspace.cleanup()

    def test_diff_reflects_edits(self):
        workspace = Workspace.prepare("logic_bug", "test_ws_diff")
        try:
            target = workspace.root / "app" / "auth" / "token.py"
            content = target.read_text(encoding="utf-8")
            target.write_text(
                content.replace(
                    "return current > self.expires_at + SESSION_TTL_SECONDS",
                    "return current > self.expires_at",
                ),
                encoding="utf-8",
            )
            diff = workspace.diff()
            assert "app/auth/token.py" in diff.replace("\\", "/")
            assert "-        return current > self.expires_at + SESSION_TTL_SECONDS" in diff
            assert "+        return current > self.expires_at" in diff

            files, added, removed = workspace.diff_stat()
            assert [f.replace("\\", "/") for f in files] == ["app/auth/token.py"]
            assert added == 1
            assert removed == 1
        finally:
            workspace.cleanup()

    def test_new_files_appear_in_diff(self):
        workspace = Workspace.prepare("logic_bug", "test_ws_newfile")
        try:
            (workspace.root / "tests" / "test_repro.py").write_text(
                "def test_repro():\n    assert True\n", encoding="utf-8"
            )
            assert "test_repro.py" in workspace.diff().replace("\\", "/")
        finally:
            workspace.cleanup()

    def test_restore_baseline_discards_changes(self):
        workspace = Workspace.prepare("logic_bug", "test_ws_restore")
        try:
            target = workspace.root / "app" / "auth" / "token.py"
            target.write_text("# broken\n", encoding="utf-8")
            (workspace.root / "junk.py").write_text("x = 1\n", encoding="utf-8")
            workspace.restore_baseline()
            assert workspace.diff() == ""
            assert "SESSION_TTL_SECONDS" in target.read_text(encoding="utf-8")
            assert not (workspace.root / "junk.py").exists()
        finally:
            workspace.cleanup()

    def test_snapshot_then_further_edits_are_isolated(self):
        workspace = Workspace.prepare("logic_bug", "test_ws_snapshot")
        try:
            target = workspace.root / "app" / "auth" / "token.py"
            target.write_text(target.read_text(encoding="utf-8") + "\n# attempt 1\n", encoding="utf-8")
            commit = workspace.snapshot("attempt-1")
            assert len(commit) == 40
            assert workspace.diff() == ""
        finally:
            workspace.cleanup()

    def test_relative_files_lists_python_sources(self):
        workspace = Workspace.prepare("logic_bug", "test_ws_listing")
        try:
            files = workspace.relative_files()
            assert "app/auth/token.py" in files
            assert "tests/test_auth.py" in files
            assert all(not f.startswith(".git") for f in files)
        finally:
            workspace.cleanup()

    def test_unknown_source_raises(self):
        with pytest.raises(WorkspaceError):
            Workspace.prepare("not-a-benchmark-or-path", "test_ws_bad")


class TestGitIsolation:
    """Workspaces live under ``data/`` inside RepoSentinel's own checkout.

    A workspace that fails to own its git directory makes git search upwards,
    so ``git add -A`` would stage the workspace into RepoSentinel's real index
    and ``git checkout`` would rewrite RepoSentinel's own files. These tests
    pin the invariant that prevents that.
    """

    def test_git_resolves_to_the_workspace_not_the_parent(self):
        workspace = Workspace.prepare("logic_bug", "test_ws_isolated_git")
        try:
            assert workspace._own_repo_root() == workspace.root
        finally:
            workspace.cleanup()

    def test_cleanup_removes_the_read_only_git_objects(self):
        workspace = Workspace.prepare("logic_bug", "test_ws_cleanup")
        root = workspace.root
        assert (root / ".git").is_dir()
        assert workspace.cleanup() is True
        assert not root.exists(), "git's read-only object files defeated cleanup"

    def test_reprepare_over_a_damaged_git_directory_reinitialises(self):
        first = Workspace.prepare("logic_bug", "test_ws_damaged")
        # Simulate an interrupted cleanup: the tree survives but the repository
        # is no longer valid, which is what made git walk up to the parent.
        (first.root / ".git" / "HEAD").unlink()

        second = Workspace.prepare("logic_bug", "test_ws_damaged")
        try:
            assert second._own_repo_root() == second.root
            target = second.root / "app" / "auth" / "token.py"
            target.write_text(
                target.read_text(encoding="utf-8").replace(
                    "return current > self.expires_at + SESSION_TTL_SECONDS",
                    "return current > self.expires_at",
                ),
                encoding="utf-8",
            )
            files, _, _ = second.diff_stat()
            assert [f.replace("\\", "/") for f in files] == ["app/auth/token.py"]
        finally:
            second.cleanup()
