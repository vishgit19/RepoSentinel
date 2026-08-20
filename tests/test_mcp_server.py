"""The MCP adapter, driven the way a real client drives it.

These tests speak JSON-RPC to the dispatcher directly, which is what an MCP
client does over stdio, and additionally run the process end to end to confirm
the stdio framing is right.
"""

from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path

import pytest
from reposentinel.mcp_server.server import (
    DEFAULT_PROTOCOL_VERSION,
    INVALID_PARAMS,
    METHOD_NOT_FOUND,
    MCPServer,
    ProtocolError,
    serve,
)
from reposentinel.tools.base import registry

BACKEND = Path(__file__).resolve().parents[1] / "backend"


def rpc(method: str, params: dict | None = None, message_id: int = 1) -> dict:
    return {"jsonrpc": "2.0", "id": message_id, "method": method, "params": params or {}}


@pytest.fixture
def server():
    instance = MCPServer("logic_bug")
    try:
        yield instance
    finally:
        instance.session.close()


class TestHandshake:
    def test_initialize_echoes_a_supported_protocol_version(self, server):
        response = server.handle(rpc("initialize", {"protocolVersion": "2024-11-05"}))
        result = response["result"]
        assert result["protocolVersion"] == "2024-11-05"
        assert result["serverInfo"]["name"] == "reposentinel"
        assert result["capabilities"]["tools"] == {"listChanged": False}

    def test_unknown_protocol_version_falls_back(self, server):
        response = server.handle(rpc("initialize", {"protocolVersion": "1999-01-01"}))
        assert response["result"]["protocolVersion"] == DEFAULT_PROTOCOL_VERSION

    def test_initialized_notification_produces_no_response(self, server):
        assert server.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
        assert server.initialized is True

    def test_wrong_jsonrpc_version_is_rejected(self, server):
        with pytest.raises(ProtocolError) as exception:
            server.handle({"jsonrpc": "1.0", "id": 1, "method": "ping"})
        assert exception.value.code == -32600

    def test_unknown_method_is_rejected(self, server):
        with pytest.raises(ProtocolError) as exception:
            server.handle(rpc("does/not/exist"))
        assert exception.value.code == METHOD_NOT_FOUND

    def test_ping_is_answered(self, server):
        assert server.handle(rpc("ping"))["result"] == {}


class TestToolListing:
    def test_every_exposed_tool_is_advertised_with_its_schema(self, server):
        tools = server.handle(rpc("tools/list"))["result"]["tools"]
        assert len(tools) == len(registry.mcp_tools())
        for tool in tools:
            assert tool["description"].strip()
            assert tool["inputSchema"]["type"] == "object"
            assert tool["annotations"]["readOnlyHint"] is True

    def test_mutating_tools_are_not_advertised(self, server):
        names = {tool["name"] for tool in server.handle(rpc("tools/list"))["result"]["tools"]}
        # A remote client must not be able to edit code through this server.
        assert "apply_patch" not in names
        assert "write_file" not in names


class TestToolCalls:
    def test_read_file_returns_repository_content(self, server):
        response = server.handle(
            rpc("tools/call", {"name": "read_file", "arguments": {"path": "app/auth/token.py"}})
        )
        result = response["result"]
        assert result["isError"] is False
        assert "is_expired" in result["content"][0]["text"]
        assert result["structuredContent"]["executed"] is True

    def test_search_symbols_uses_the_ast_index(self, server):
        response = server.handle(
            rpc("tools/call", {"name": "search_symbols", "arguments": {"name": "is_expired"}})
        )
        assert "token.py" in response["result"]["content"][0]["text"]

    def test_a_reproduced_test_failure_is_not_a_protocol_error(self, server):
        response = server.handle(
            rpc(
                "tools/call",
                {
                    "name": "run_targeted_tests",
                    "arguments": {"targets": ["tests/test_auth.py"]},
                },
            )
        )
        result = response["result"]
        # The seeded bug makes these tests fail. The tool ran correctly, so the
        # call succeeded even though the finding is negative.
        assert result["isError"] is False
        assert result["structuredContent"]["executed"] is True
        assert result["structuredContent"]["ok"] is False

    def test_reading_outside_the_repository_is_refused(self, server):
        response = server.handle(
            rpc("tools/call", {"name": "read_file", "arguments": {"path": "../../secrets.txt"}})
        )
        assert response["result"]["isError"] is True

    def test_unexposed_tool_is_unavailable(self, server):
        with pytest.raises(ProtocolError) as exception:
            server.handle(
                rpc(
                    "tools/call",
                    {"name": "apply_patch", "arguments": {"edits": []}},
                )
            )
        assert exception.value.code == INVALID_PARAMS
        assert "not available" in exception.value.message

    def test_missing_name_is_rejected(self, server):
        with pytest.raises(ProtocolError) as exception:
            server.handle(rpc("tools/call", {"arguments": {}}))
        assert exception.value.code == INVALID_PARAMS

    def test_retrieval_tool_builds_its_index_on_demand(self, server):
        response = server.handle(
            rpc(
                "tools/call",
                {
                    "name": "bm25_search",
                    "arguments": {"query": "session token expiry is_expired"},
                },
            )
        )
        result = response["result"]
        assert result["isError"] is False
        assert "token.py" in result["content"][0]["text"]


class TestResources:
    def test_repository_index_is_exposed_as_a_resource(self, server):
        listing = server.handle(rpc("resources/list"))["result"]["resources"]
        assert listing[0]["uri"] == "reposentinel://repository"

        read = server.handle(
            rpc("resources/read", {"uri": "reposentinel://repository"})
        )["result"]
        payload = json.loads(read["contents"][0]["text"])
        assert payload["indexed"]["files"] > 0

    def test_unknown_resource_is_rejected(self, server):
        with pytest.raises(ProtocolError):
            server.handle(rpc("resources/read", {"uri": "reposentinel://nope"}))


class TestStdioFraming:
    def test_serve_reads_and_writes_newline_delimited_json(self):
        requests = "\n".join(
            [
                json.dumps(rpc("initialize", {"protocolVersion": "2024-11-05"}, 1)),
                json.dumps(rpc("tools/list", {}, 2)),
                "not json at all",
            ]
        )
        stdout = io.StringIO()
        serve("logic_bug", stdin=io.StringIO(requests), stdout=stdout)

        lines = [json.loads(line) for line in stdout.getvalue().splitlines() if line.strip()]
        assert lines[0]["id"] == 1
        assert lines[1]["id"] == 2
        # Malformed input is answered with a parse error, not a crash.
        assert lines[2]["error"]["code"] == -32700

    def test_process_serves_over_real_stdio(self):
        """The documented invocation must actually work as a subprocess."""
        payload = (
            json.dumps(rpc("initialize", {"protocolVersion": "2024-11-05"}, 1))
            + "\n"
            + json.dumps(rpc("tools/list", {}, 2))
            + "\n"
        )
        completed = subprocess.run(
            [sys.executable, "-m", "reposentinel.mcp_server", "--repo", "logic_bug"],
            input=payload,
            capture_output=True,
            text=True,
            timeout=180,
            cwd=str(BACKEND),
        )
        assert completed.returncode == 0, completed.stderr
        lines = [
            json.loads(line) for line in completed.stdout.splitlines() if line.strip()
        ]
        assert lines[0]["result"]["serverInfo"]["name"] == "reposentinel"
        assert len(lines[1]["result"]["tools"]) == len(registry.mcp_tools())

    def test_unknown_repository_exits_with_an_error(self):
        completed = subprocess.run(
            [sys.executable, "-m", "reposentinel.mcp_server", "--repo", "nope-not-real"],
            input="",
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(BACKEND),
        )
        assert completed.returncode == 2
        assert "not a known benchmark" in completed.stderr
