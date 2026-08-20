"""MCP adapter exposing RepoSentinel's repository tools to any MCP client."""

from reposentinel.mcp_server.server import (
    DEFAULT_PROTOCOL_VERSION,
    SUPPORTED_PROTOCOL_VERSIONS,
    MCPServer,
    ProtocolError,
    RepositorySession,
    serve,
)

__all__ = [
    "DEFAULT_PROTOCOL_VERSION",
    "SUPPORTED_PROTOCOL_VERSIONS",
    "MCPServer",
    "ProtocolError",
    "RepositorySession",
    "serve",
]
