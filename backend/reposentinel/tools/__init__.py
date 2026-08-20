"""Tool registry.

Importing this package registers every tool exactly once. The same registry is
consumed by the agent's tool-calling loop, by the MCP server and by the API's
``/api/tools`` introspection endpoint.
"""

from __future__ import annotations

from reposentinel.tools import exec_tools, repo_tools, retrieval_tools  # noqa: F401
from reposentinel.tools.base import (
    ToolContext,
    ToolRegistry,
    ToolResult,
    ToolSpec,
    execute_tool,
    registry,
)

__all__ = [
    "ToolContext",
    "ToolRegistry",
    "ToolResult",
    "ToolSpec",
    "execute_tool",
    "registry",
]
