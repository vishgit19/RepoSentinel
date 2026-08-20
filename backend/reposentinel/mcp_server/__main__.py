"""Entry point for the MCP stdio server.

    python -m reposentinel.mcp_server --repo logic_bug
    python -m reposentinel.mcp_server --repo C:\\path\\to\\repo

Register it with an MCP client by pointing the client's command at exactly
that, with ``cwd`` set to the RepoSentinel checkout.
"""

from __future__ import annotations

import argparse
import sys

from reposentinel.mcp_server.server import serve
from reposentinel.workspace import WorkspaceError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m reposentinel.mcp_server",
        description="Expose RepoSentinel's read-only repository tools over MCP.",
    )
    parser.add_argument(
        "--repo",
        required=True,
        help="Benchmark id, local repository path, or Git URL.",
    )
    arguments = parser.parse_args(argv)

    try:
        from reposentinel.workspace import Workspace

        Workspace.validate_source(arguments.repo)
    except WorkspaceError as exc:
        # stderr, because stdout is the JSON-RPC channel.
        print(f"reposentinel-mcp: {exc}", file=sys.stderr)
        return 2

    print(f"reposentinel-mcp: serving '{arguments.repo}' over stdio", file=sys.stderr)
    return serve(arguments.repo)


if __name__ == "__main__":
    raise SystemExit(main())
