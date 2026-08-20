"""Restricted local-subprocess sandbox.

This is the default backend and the one exercised by RepoSentinel's own test
suite. It is *not* a security boundary against genuinely hostile native code -
that is what :mod:`reposentinel.sandbox.docker` is for - but it does enforce
the guardrails that matter for the threat model of "the repository under
repair contains untrusted text and possibly malicious test code":

* argv-only execution (never a shell),
* allow-listed executables and denied subcommands,
* working directory confined to the workspace,
* a scrubbed environment with no host secrets,
* a hard wall-clock timeout with process-tree termination,
* secret redaction on the way out.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

from reposentinel.sandbox.base import CommandResult, Sandbox
from reposentinel.sandbox.guardrails import check_command, resolve_in_workspace

# Host variables the child is allowed to inherit. Everything else - crucially
# every API key - is dropped.
_ENV_ALLOW_LIST = (
    "PATH",
    "SYSTEMROOT",
    "SystemRoot",
    "COMSPEC",
    "ComSpec",
    "WINDIR",
    "windir",
    "TEMP",
    "TMP",
    "TMPDIR",
    "HOME",
    "LANG",
    "LC_ALL",
    "PROCESSOR_ARCHITECTURE",
    "NUMBER_OF_PROCESSORS",
)


class LocalSandbox(Sandbox):
    backend_name = "local"

    def __init__(self, workspace: Path, timeout_seconds: int = 180) -> None:
        super().__init__(workspace, timeout_seconds)

    @staticmethod
    def is_available() -> bool:
        return True

    def _child_env(self) -> dict[str, str]:
        env = {key: os.environ[key] for key in _ENV_ALLOW_LIST if key in os.environ}
        env.update(
            {
                # Imports resolve against the workspace, not the host project.
                "PYTHONPATH": str(self.workspace),
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONUNBUFFERED": "1",
                "PYTHONNOUSERSITE": "1",
                # Marker so repository code can detect it is under analysis.
                "REPOSENTINEL_SANDBOX": "1",
            }
        )
        return env

    def _resolve_executable(self, command: list[str]) -> list[str]:
        """Point bare interpreter/tool names at the RepoSentinel virtualenv.

        Tools are invoked as ``python -m <tool>`` so a single interpreter
        resolution covers pytest, ruff and mypy.
        """
        head = command[0].lower()
        if head in {"python", "python3", "py"}:
            return [sys.executable, *command[1:]]
        if head in {"pytest", "ruff", "mypy", "semgrep"}:
            return [sys.executable, "-m", head, *command[1:]]
        return command

    def run(
        self,
        command: list[str],
        *,
        cwd: str = ".",
        timeout: int | None = None,
    ) -> CommandResult:
        verdict = check_command(command)
        if not verdict.allowed:
            return CommandResult(
                command=command,
                exit_code=126,
                stderr=f"blocked by guardrail: {verdict.reason}",
                blocked=True,
                block_reason=verdict.reason,
                backend=self.backend_name,
            )

        try:
            working_dir = resolve_in_workspace(self.workspace, cwd)
        except Exception as exc:
            return CommandResult(
                command=command,
                exit_code=126,
                stderr=str(exc),
                blocked=True,
                block_reason="path escapes the workspace",
                backend=self.backend_name,
            )

        if not working_dir.is_dir():
            return CommandResult(
                command=command,
                exit_code=127,
                stderr=f"working directory does not exist: {cwd}",
                backend=self.backend_name,
            )

        argv = self._resolve_executable(command)
        limit = timeout or self.timeout_seconds
        started = time.perf_counter()

        try:
            completed = subprocess.run(  # noqa: S603 - argv list, shell=False by construction
                argv,
                cwd=str(working_dir),
                env=self._child_env(),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=limit,
                shell=False,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return CommandResult(
                command=command,
                exit_code=124,
                stdout=exc.stdout or "" if isinstance(exc.stdout, str) else "",
                stderr=f"timed out after {limit}s",
                duration_ms=int((time.perf_counter() - started) * 1000),
                timed_out=True,
                backend=self.backend_name,
            )
        except FileNotFoundError as exc:
            return CommandResult(
                command=command,
                exit_code=127,
                stderr=f"executable not found: {exc}",
                backend=self.backend_name,
            )

        return CommandResult(
            command=command,
            exit_code=completed.returncode,
            stdout=completed.stdout or "",
            stderr=completed.stderr or "",
            duration_ms=int((time.perf_counter() - started) * 1000),
            backend=self.backend_name,
        )
