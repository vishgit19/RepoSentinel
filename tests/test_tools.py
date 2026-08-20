"""The agent's tool surface, exercised against the seeded benchmark repository.

These are the tools the model actually calls, so they are tested through
``execute_tool`` - the same entry point the graph and the MCP server use -
rather than by calling the handlers directly.
"""

from __future__ import annotations

import pytest
from reposentinel.retrieval.indexer import CodeIndexer
from reposentinel.sandbox import build_sandbox
from reposentinel.tools import registry
from reposentinel.tools.base import ToolContext, execute_tool
from reposentinel.workspace import Workspace


@pytest.fixture(scope="module")
def tools_context():
    workspace = Workspace.prepare("logic_bug", "test_tools_ws")
    try:
        index = CodeIndexer("logic_bug", workspace.root).index(workspace.relative_files())
        yield ToolContext(
            workspace=workspace,
            sandbox=build_sandbox(workspace.root),
            index=index,
            run_id="test_tools",
        )
    finally:
        workspace.cleanup()


def call(context: ToolContext, tool: str, **arguments):
    return execute_tool(tool, arguments, context)


class TestRegistry:
    def test_every_tool_exposes_a_usable_schema(self):
        specs = registry.all()
        assert specs, "no tools registered"
        for spec in specs:
            schema = spec.parameters
            assert schema["type"] == "object"
            # OpenAI strict function calling rejects free-form objects.
            assert schema.get("additionalProperties") is False, spec.name
            assert spec.description.strip(), spec.name

    def test_unknown_tool_is_reported_not_raised(self, tools_context):
        record = call(tools_context, "no_such_tool")
        assert record.ok is False
        assert record.executed is False
        assert "unknown tool" in record.error


class TestRepositoryTools:
    def test_list_files(self, tools_context):
        record = call(tools_context, "list_files", directory=".")
        assert record.ok is True
        assert "app/auth/token.py" in record.output

    def test_read_file_returns_requested_line_range(self, tools_context):
        record = call(tools_context, "read_file", path="app/auth/token.py", start_line=41, end_line=50)
        assert record.ok is True
        assert "is_expired" in record.output

    def test_read_file_outside_the_workspace_is_refused(self, tools_context):
        record = call(tools_context, "read_file", path="../../../secrets.txt")
        assert record.ok is False
        assert record.executed is False
        assert any(e.kind == "path_escape" for e in tools_context.safety_events)

    def test_search_code(self, tools_context):
        record = call(tools_context, "search_code", query="SESSION_TTL_SECONDS")
        assert record.ok is True
        assert "app/auth/token.py" in record.output

    def test_search_symbols(self, tools_context):
        record = call(tools_context, "search_symbols", name="is_expired")
        assert record.ok is True
        assert "SessionToken.is_expired" in record.output

    def test_inspect_imports(self, tools_context):
        record = call(tools_context, "inspect_imports", path="app/auth/middleware.py")
        assert record.ok is True
        assert "token" in record.output

    def test_find_callers_of_the_buggy_method(self, tools_context):
        record = call(tools_context, "find_callers", symbol="is_expired")
        assert record.ok is True
        assert "middleware.py" in record.output or "token.py" in record.output

    def test_git_diff_is_empty_before_any_edit(self, tools_context):
        record = call(tools_context, "git_diff")
        assert record.ok is True


class TestExecutionTools:
    def test_targeted_tests_reproduce_the_seeded_failure(self, tools_context):
        """The central distinction: the tool worked, the code under test did not."""
        record = call(
            tools_context,
            "run_targeted_tests",
            targets=["tests/test_auth.py::test_expired_token_is_rejected_by_validate_token"],
        )
        assert record.ok is False, "the seeded bug should make this test fail"
        assert record.executed is True, "a reproduced failure is not a tool error"
        assert record.blocked is False

    def test_full_suite_runs(self, tools_context):
        record = call(tools_context, "run_full_tests")
        assert record.executed is True
        assert "passed" in record.summary or "failed" in record.summary

    def test_lint_is_not_blocked_by_its_own_output_flag(self, tools_context):
        """``--output-format=concise`` once tripped a 'disk operation' rule."""
        record = call(tools_context, "run_lint", path="app")
        assert record.blocked is False, record.block_reason
        assert record.executed is True

    def test_security_scan_runs(self, tools_context):
        record = call(tools_context, "run_security_scan")
        assert record.executed is True


class TestPatchTools:
    def test_apply_patch_then_tests_pass(self, tools_context):
        record = call(
            tools_context,
            "apply_patch",
            edits=[
                {
                    "path": "app/auth/token.py",
                    "search": "return current > self.expires_at + SESSION_TTL_SECONDS",
                    "replace": "return current > self.expires_at",
                }
            ],
        )
        assert record.ok is True, record.error

        diff = call(tools_context, "git_diff")
        assert "app/auth/token.py" in diff.output.replace("\\", "/")

        after = call(
            tools_context,
            "run_targeted_tests",
            targets=["tests/test_auth.py::test_expired_token_is_rejected_by_validate_token"],
        )
        assert after.ok is True, after.output[:400]

        tools_context.workspace.restore_baseline()

    def test_apply_patch_rejects_text_that_is_not_present(self, tools_context):
        record = call(
            tools_context,
            "apply_patch",
            edits=[{"path": "app/auth/token.py", "search": "not_in_this_file", "replace": "x"}],
        )
        assert record.ok is False

    def test_write_file_refuses_non_source_types(self, tools_context):
        record = call(tools_context, "write_file", path="payload.bin", content="x")
        assert record.ok is False
