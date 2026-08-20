"""GitHub PR creation is gated and refuses to pretend."""

from __future__ import annotations

from pathlib import Path

from reposentinel.config import Settings
from reposentinel.github import github_origin, maybe_open_pull_request


def test_no_remote_means_no_origin(tmp_path: Path):
    (tmp_path / ".git").mkdir()
    assert github_origin(tmp_path) is None


def test_push_flag_off_is_a_recorded_no_op(tmp_path: Path):
    settings = Settings(allow_github_push=False, github_token="tok")
    result = maybe_open_pull_request(
        workspace_root=tmp_path,
        settings=settings,
        title="fix",
        body="body",
        head="branch",
    )
    assert result.opened is False
    assert "ALLOW_GITHUB_PUSH" in result.reason


def test_missing_token_is_a_recorded_no_op(tmp_path: Path):
    settings = Settings(allow_github_push=True, github_token=None)
    result = maybe_open_pull_request(
        workspace_root=tmp_path,
        settings=settings,
        title="fix",
        body="body",
        head="branch",
    )
    assert result.opened is False
    assert "GITHUB_TOKEN" in result.reason
