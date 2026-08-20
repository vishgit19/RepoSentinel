"""Sandbox execution behaviour, verified against real subprocesses."""

from __future__ import annotations

import sys

from reposentinel.sandbox import build_sandbox, describe_backend
from reposentinel.sandbox.local import LocalSandbox


def write_script(workspace, name: str, body: str) -> str:
    (workspace / name).write_text(body, encoding="utf-8")
    return name


class TestLocalSandboxExecution:
    def test_runs_a_workspace_script_and_captures_stdout(self, tmp_path):
        script = write_script(tmp_path, "hello.py", "print('hello from sandbox')\n")
        result = LocalSandbox(tmp_path).run(["python", script])
        assert result.ok, result.combined_output
        assert "hello from sandbox" in result.stdout

    def test_nonzero_exit_is_reported(self, tmp_path):
        script = write_script(tmp_path, "boom.py", "import sys\nsys.exit(3)\n")
        result = LocalSandbox(tmp_path).run(["python", script])
        assert result.exit_code == 3
        assert result.ok is False

    def test_timeout_is_enforced(self, tmp_path):
        script = write_script(tmp_path, "slow.py", "import time\ntime.sleep(30)\n")
        result = LocalSandbox(tmp_path, timeout_seconds=1).run(["python", script])
        assert result.timed_out is True
        assert result.exit_code == 124

    def test_inline_code_is_refused(self, tmp_path):
        result = LocalSandbox(tmp_path).run(["python", "-c", "print(1)"])
        assert result.blocked is True
        assert "inline code" in result.block_reason

    def test_unlisted_module_is_refused(self, tmp_path):
        result = LocalSandbox(tmp_path).run(["python", "-m", "http.server"])
        assert result.blocked is True
        assert "allow-list" in result.block_reason

    def test_blocked_command_is_not_executed(self, tmp_path):
        result = LocalSandbox(tmp_path).run(["curl", "https://example.com"])
        assert result.blocked is True
        assert result.exit_code == 126
        assert "allow-list" in result.block_reason

    def test_working_directory_confinement(self, tmp_path):
        result = LocalSandbox(tmp_path).run(["pytest", "-q"], cwd="../..")
        assert result.blocked is True

    def test_host_secrets_are_not_visible_to_child(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-should-never-be-inherited-000000")
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_shouldneverbeinherited00000")
        script = write_script(
            tmp_path,
            "leak.py",
            "import os, json\n"
            "print(json.dumps({k: v for k, v in os.environ.items() "
            "if 'KEY' in k or 'TOKEN' in k}))\n",
        )
        result = LocalSandbox(tmp_path).run(["python", script])
        assert result.ok, result.combined_output
        assert "should-never-be-inherited" not in result.stdout
        assert "shouldneverbeinherited" not in result.stdout
        assert result.stdout.strip() == "{}"

    def test_sandbox_marker_is_set(self, tmp_path):
        script = write_script(
            tmp_path, "marker.py", "import os\nprint(os.environ.get('REPOSENTINEL_SANDBOX'))\n"
        )
        result = LocalSandbox(tmp_path).run(["python", script])
        assert result.stdout.strip() == "1"

    def test_secret_redaction_on_output(self, tmp_path):
        script = write_script(
            tmp_path, "print_secret.py", "print('leaked ghp_aaaaaaaaaaaaaaaaaaaaaaaaaaaa')\n"
        )
        result = LocalSandbox(tmp_path).run(["python", script])
        sanitised = result.sanitised()
        assert "ghp_aaaaaaaaaaaaaaaaaaaaaaaaaaaa" not in sanitised
        assert "REDACTED" in sanitised

    def test_pytest_runs_inside_workspace(self, tmp_path):
        write_script(tmp_path, "test_sample.py", "def test_ok():\n    assert 1 + 1 == 2\n")
        result = LocalSandbox(tmp_path).run(["pytest", "-q", "test_sample.py"])
        assert result.ok, result.combined_output
        assert "1 passed" in result.stdout

    def test_failing_pytest_reports_nonzero(self, tmp_path):
        write_script(tmp_path, "test_bad.py", "def test_bad():\n    assert 1 == 2\n")
        result = LocalSandbox(tmp_path).run(["pytest", "-q", "test_bad.py"])
        assert result.exit_code != 0
        assert "1 failed" in result.stdout

    def test_missing_working_directory(self, tmp_path):
        result = LocalSandbox(tmp_path).run(["pytest", "-q"], cwd="nope")
        assert result.exit_code == 127

    def test_module_style_tools_resolve_to_the_venv(self, tmp_path):
        argv = LocalSandbox(tmp_path)._resolve_executable(["pytest", "-q"])
        assert argv[0] == sys.executable
        assert argv[1:3] == ["-m", "pytest"]


class TestBackendSelection:
    def test_build_sandbox_returns_a_working_backend(self, tmp_path):
        sandbox = build_sandbox(tmp_path)
        assert sandbox.backend_name in {"local", "docker"}
        assert sandbox.workspace == tmp_path.resolve()

    def test_describe_backend_reports_availability(self):
        described = describe_backend()
        assert described["active"] in {"local", "docker"}
        assert isinstance(described["docker_available"], bool)
