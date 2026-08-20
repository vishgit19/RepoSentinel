"""Print the tool registry: category, MCP exposure and mutation flag."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

import reposentinel.tools  # noqa: F401  (registers every tool)
from reposentinel.tools.base import registry


def main() -> int:
    for spec in registry.all():
        flags = []
        if spec.expose_via_mcp:
            flags.append("mcp")
        if spec.mutating:
            flags.append("mutating")
        print(f"{spec.category:<11} {spec.name:<22} {','.join(flags)}")
    print(f"\n{len(registry.all())} tools, {len(registry.mcp_tools())} exposed via MCP")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
