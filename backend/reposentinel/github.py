"""Optional GitHub pull-request creation.

A patch never leaves the sandbox unless three things are all true:

* a human approved this run,
* ``REPOSENTINEL_ALLOW_GITHUB_PUSH`` is set,
* a ``GITHUB_TOKEN`` with ``repo`` scope is configured,
* the workspace has a GitHub remote.

Anything else - including every bundled benchmark, which has no remote - is a
deliberate no-op with a recorded reason, not a silent skip.
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from reposentinel.config import Settings

_GITHUB_SSH = re.compile(r"git@github\.com:(?P<owner>[^/]+)/(?P<repo>[^/.]+)")
_GITHUB_HTTPS = re.compile(
    r"https?://github\.com/(?P<owner>[^/]+)/(?P<repo>[^/.]+)"
)


@dataclass(frozen=True)
class PullRequestResult:
    opened: bool
    reason: str = ""
    url: str = ""
    number: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "opened": self.opened,
            "reason": self.reason,
            "url": self.url,
            "number": self.number,
        }


def github_origin(workspace_root) -> tuple[str, str] | None:
    """Return (owner, repo) when *workspace_root* has a GitHub remote."""
    completed = subprocess.run(
        ["git", "remote", "get-url", "origin"],
        cwd=str(workspace_root),
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    if completed.returncode != 0:
        return None
    url = completed.stdout.strip()
    for pattern in (_GITHUB_HTTPS, _GITHUB_SSH):
        match = pattern.search(url)
        if match:
            return match.group("owner"), match.group("repo")
    return None


def maybe_open_pull_request(
    *,
    workspace_root,
    settings: Settings,
    title: str,
    body: str,
    head: str,
    base: str = "main",
) -> PullRequestResult:
    if not settings.allow_github_push:
        return PullRequestResult(
            False,
            "GitHub push is disabled (REPOSENTINEL_ALLOW_GITHUB_PUSH is not set).",
        )
    if not settings.github_token:
        return PullRequestResult(False, "No GITHUB_TOKEN is configured.")
    origin = github_origin(workspace_root)
    if origin is None:
        return PullRequestResult(
            False,
            "The workspace has no GitHub remote; the patch stays in the sandbox.",
        )
    owner, repo = origin
    payload = json.dumps(
        {"title": title[:80], "body": body[:4000], "head": head, "base": base}
    ).encode("utf-8")
    request = Request(
        f"https://api.github.com/repos/{owner}/{repo}/pulls",
        data=payload,
        method="POST",
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {settings.github_token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "Content-Type": "application/json",
            "User-Agent": "RepoSentinel",
        },
    )
    try:
        with urlopen(request, timeout=30) as response:  # noqa: S310 - https only
            data = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:400]
        return PullRequestResult(False, f"GitHub API {exc.code}: {detail}")
    except URLError as exc:
        return PullRequestResult(False, f"GitHub API unreachable: {exc.reason}")
    return PullRequestResult(
        True,
        "opened",
        url=str(data.get("html_url") or ""),
        number=data.get("number"),
    )
