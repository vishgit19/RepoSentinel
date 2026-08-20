"""An MCP server exposing RepoSentinel's repository tools.

The point of this module is interoperability: the tools the agent uses
internally are the *same* tools any MCP client (an IDE, another agent) can
call, described by the same JSON schemas. Nothing is reimplemented here - this
is a protocol adapter over :mod:`reposentinel.tools`.

Transport is newline-delimited JSON-RPC 2.0 on stdin/stdout, which is the MCP
stdio transport. It is implemented directly rather than via an SDK so the
server has no dependency beyond the standard library, and so the framing stays
inspectable.

Two safety decisions are deliberate:

* Only non-mutating tools are exposed. ``apply_patch`` and ``write_file`` are
  registered with ``expose_via_mcp=False`` and are unreachable from here, so a
  remote client cannot edit code through this server.
* The server works on an isolated copy of the repository. Tools that execute
  commands (pytest, ruff) therefore cannot touch the caller's working tree.
"""

from __future__ import annotations

import json
import sys
import threading
from dataclasses import dataclass, field
from typing import Any, TextIO

from reposentinel.config import Settings, get_settings
from reposentinel.models.schemas import new_id
from reposentinel.retrieval.indexer import CodeIndexer
from reposentinel.sandbox import build_sandbox
from reposentinel.tools.base import ToolContext, execute_tool, registry
from reposentinel.workspace import Workspace

# Versions of the MCP spec this adapter implements. The client's requested
# version is echoed back when we support it, per the negotiation rules.
SUPPORTED_PROTOCOL_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")
DEFAULT_PROTOCOL_VERSION = SUPPORTED_PROTOCOL_VERSIONS[-1]

SERVER_INFO = {"name": "reposentinel", "version": "1.0.0"}

# JSON-RPC error codes (the reserved range from the spec).
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603


class ProtocolError(Exception):
    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass
class RepositorySession:
    """The repository this server answers questions about.

    Indexing is cheap (AST only) and happens up front. The retriever is built
    lazily because it needs an embedding backend, which may require network
    access that a read-only client never asks for.
    """

    source: str
    settings: Settings
    session_id: str = field(default_factory=lambda: new_id("mcp"))
    _context: ToolContext | None = None
    _retriever_ready: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def context(self) -> ToolContext:
        with self._lock:
            if self._context is None:
                workspace = Workspace.prepare(
                    self.source, self.session_id, settings=self.settings
                )
                index = CodeIndexer(self.session_id, workspace.root).index(
                    workspace.relative_files(extensions=(".py",))
                )
                self._context = ToolContext(
                    workspace=workspace,
                    sandbox=build_sandbox(workspace.root, settings=self.settings),
                    settings=self.settings,
                    index=index,
                    retriever=None,
                    run_id=self.session_id,
                )
            return self._context

    def ensure_retriever(self) -> None:
        """Attach a retriever on first use of a retrieval tool."""
        context = self.context()
        with self._lock:
            if self._retriever_ready or context.retriever is not None:
                return
            self._retriever_ready = True
        # Imported here so a client that only reads files never pays for the
        # embedding stack.
        from reposentinel.retrieval.embeddings import (
            HashingEmbeddings,
            build_embedding_backend,
        )
        from reposentinel.retrieval.pipeline import HybridRetriever
        from reposentinel.retrieval.reranker import LexicalReranker
        from reposentinel.retrieval.vector_store import build_vector_store

        try:
            embeddings = build_embedding_backend(self.settings)
            store = build_vector_store(self.settings, dimensions=embeddings.dimensions)
        except Exception:  # noqa: BLE001 - degrade to offline embeddings
            embeddings = HashingEmbeddings()
            store = build_vector_store(self.settings, dimensions=embeddings.dimensions)

        retriever = HybridRetriever(
            repo_id=self.session_id,
            index=context.index,
            embeddings=embeddings,
            vector_store=store,
            # Lexical reranking keeps this server free of LLM calls: an MCP
            # client pays for its own model, not for ours.
            reranker=LexicalReranker(),
            settings=self.settings,
        )
        retriever.build()
        context.retriever = retriever

    def describe(self) -> dict[str, Any]:
        context = self.context()
        return {
            "session": self.session_id,
            "source": self.source,
            "workspace": str(context.workspace.root),
            "indexed": context.index.stats(),
            "sandbox": context.sandbox.backend_name,
        }

    def close(self) -> None:
        if self._context is not None:
            self._context.workspace.cleanup()


class MCPServer:
    """JSON-RPC dispatcher for the MCP methods this server implements."""

    def __init__(self, source: str, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.session = RepositorySession(source=source, settings=self.settings)
        self.initialized = False
        self.protocol_version = DEFAULT_PROTOCOL_VERSION

    # -- tool descriptions ------------------------------------------------
    def tool_descriptors(self) -> list[dict[str, Any]]:
        descriptors = []
        for spec in registry.mcp_tools():
            descriptors.append(
                {
                    "name": spec.name,
                    "description": spec.description,
                    "inputSchema": spec.parameters,
                    # Read-only hints let a client decide what needs consent.
                    "annotations": {
                        "readOnlyHint": not spec.mutating,
                        "destructiveHint": False,
                        "category": spec.category,
                    },
                }
            )
        return descriptors

    # -- method handlers --------------------------------------------------
    def handle(self, message: dict[str, Any]) -> dict[str, Any] | None:
        """Dispatch one JSON-RPC message; returns None for notifications."""
        if message.get("jsonrpc") != "2.0":
            raise ProtocolError(INVALID_REQUEST, "jsonrpc must be '2.0'")

        method = message.get("method")
        if not isinstance(method, str):
            raise ProtocolError(INVALID_REQUEST, "missing method")

        params = message.get("params") or {}
        message_id = message.get("id")
        is_notification = "id" not in message

        if method.startswith("notifications/"):
            if method == "notifications/initialized":
                self.initialized = True
            return None

        handlers = {
            "initialize": self._initialize,
            "ping": lambda _params: {},
            "tools/list": self._tools_list,
            "tools/call": self._tools_call,
            "resources/list": lambda _params: {"resources": self._resources()},
            "resources/read": self._resources_read,
            "prompts/list": lambda _params: {"prompts": []},
            "shutdown": lambda _params: {},
        }
        handler = handlers.get(method)
        if handler is None:
            raise ProtocolError(METHOD_NOT_FOUND, f"unknown method '{method}'")

        result = handler(params)
        if is_notification:
            return None
        return {"jsonrpc": "2.0", "id": message_id, "result": result}

    def _initialize(self, params: dict[str, Any]) -> dict[str, Any]:
        requested = params.get("protocolVersion")
        self.protocol_version = (
            requested if requested in SUPPORTED_PROTOCOL_VERSIONS else DEFAULT_PROTOCOL_VERSION
        )
        return {
            "protocolVersion": self.protocol_version,
            "capabilities": {
                "tools": {"listChanged": False},
                "resources": {"subscribe": False, "listChanged": False},
            },
            "serverInfo": SERVER_INFO,
            "instructions": (
                "Repository inspection tools from RepoSentinel. All tools are "
                "read-only and operate on an isolated copy of the repository; "
                "they cannot modify your working tree."
            ),
        }

    def _tools_list(self, _params: dict[str, Any]) -> dict[str, Any]:
        return {"tools": self.tool_descriptors()}

    def _tools_call(self, params: dict[str, Any]) -> dict[str, Any]:
        name = params.get("name")
        if not isinstance(name, str):
            raise ProtocolError(INVALID_PARAMS, "'name' is required")

        spec = registry.get(name)
        if spec is None or not spec.expose_via_mcp:
            # A tool that exists but is not exposed must not be distinguishable
            # from one that does not: an unexposed tool is simply unavailable.
            raise ProtocolError(INVALID_PARAMS, f"tool '{name}' is not available")

        arguments = params.get("arguments") or {}
        if not isinstance(arguments, dict):
            raise ProtocolError(INVALID_PARAMS, "'arguments' must be an object")

        if spec.category == "retrieval":
            self.session.ensure_retriever()

        record = execute_tool(name, arguments, self.session.context(), via_mcp=True)
        text = record.output or record.summary or ""
        if record.error:
            text = f"{text}\n{record.error}".strip()

        return {
            "content": [{"type": "text", "text": text or "(no output)"}],
            # `isError` marks a failed *execution*. A tool that ran correctly
            # and reported a negative finding - failing tests, for instance -
            # is not a protocol-level error.
            "isError": not record.executed,
            "structuredContent": {
                "tool": record.name,
                "ok": record.ok,
                "executed": record.executed,
                "summary": record.summary,
                "duration_ms": record.duration_ms,
                "blocked": record.blocked,
                "block_reason": record.block_reason,
            },
        }

    def _resources(self) -> list[dict[str, Any]]:
        return [
            {
                "uri": "reposentinel://repository",
                "name": "Repository index",
                "description": "Files, symbols and graph edges discovered in the repository.",
                "mimeType": "application/json",
            }
        ]

    def _resources_read(self, params: dict[str, Any]) -> dict[str, Any]:
        uri = params.get("uri")
        if uri != "reposentinel://repository":
            raise ProtocolError(INVALID_PARAMS, f"unknown resource '{uri}'")
        return {
            "contents": [
                {
                    "uri": uri,
                    "mimeType": "application/json",
                    "text": json.dumps(self.session.describe(), indent=2),
                }
            ]
        }


def serve(
    source: str,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
    settings: Settings | None = None,
) -> int:
    """Read newline-delimited JSON-RPC from *stdin*, write replies to *stdout*."""
    reader = stdin or sys.stdin
    writer = stdout or sys.stdout
    server = MCPServer(source, settings=settings)

    def reply(payload: dict[str, Any]) -> None:
        writer.write(json.dumps(payload) + "\n")
        writer.flush()

    try:
        for line in reader:
            line = line.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError as exc:
                reply(
                    {
                        "jsonrpc": "2.0",
                        "id": None,
                        "error": {"code": PARSE_ERROR, "message": f"invalid JSON: {exc}"},
                    }
                )
                continue

            try:
                response = server.handle(message)
            except ProtocolError as exc:
                reply(
                    {
                        "jsonrpc": "2.0",
                        "id": message.get("id"),
                        "error": {"code": exc.code, "message": exc.message},
                    }
                )
                continue
            except Exception as exc:  # noqa: BLE001 - a bad call must not kill the server
                reply(
                    {
                        "jsonrpc": "2.0",
                        "id": message.get("id"),
                        "error": {
                            "code": INTERNAL_ERROR,
                            "message": f"{type(exc).__name__}: {exc}",
                        },
                    }
                )
                continue

            if response is not None:
                reply(response)
    finally:
        server.session.close()
    return 0
