"""Docker-backed sandbox.

Selected automatically when a Docker daemon is reachable, or forced with
``REPOSENTINEL_SANDBOX_BACKEND=docker``. Every command runs in a throwaway
container with:

* no network at all (``--network none``),
* a read-only root filesystem with only the workspace mount writable,
* dropped capabilities and no privilege escalation,
* memory / CPU / PID ceilings,
* a non-root user.

Build the image with ``docker build -t reposentinel-sandbox:latest -f
docker/Dockerfile.sandbox .``
"""

from __future__ import annotations

import functools
import shutil
import subprocess
import time
from pathlib import Path

from reposentinel.sandbox.base import CommandResult, Sandbox
from reposentinel.sandbox.guardrails import check_command, resolve_in_workspace

CONTAINER_WORKDIR = "/workspace"


@functools.lru_cache(maxsize=1)
def docker_available() -> bool:
    """Whether a usable Docker daemon is reachable (cached per process)."""
    if shutil.which("docker") is None:
        return False
    try:
        completed = subprocess.run(  # noqa: S603
            ["docker", "info", "--format", "{{.ServerVersion}}"],
            capture_output=True,
            text=True,
            timeout=15,
            shell=False,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return completed.returncode == 0


class DockerSandbox(Sandbox):
    backend_name = "docker"

    def __init__(
        self,
        workspace: Path,
        timeout_seconds: int = 180,
        image: str = "reposentinel-sandbox:latest",
        memory: str = "1g",
        cpus: str = "2",
    ) -> None:
        super().__init__(workspace, timeout_seconds)
        self.image = image
        self.memory = memory
        self.cpus = cpus

    @staticmethod
    def is_available() -> bool:
        return docker_available()

    def _container_argv(self, command: list[str], workdir: str, limit: int) -> list[str]:
        return [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--memory",
            self.memory,
            "--memory-swap",
            self.memory,
            "--cpus",
            self.cpus,
            "--pids-limit",
            "256",
            "--security-opt",
            "no-new-privileges",
            "--cap-drop",
            "ALL",
            "--read-only",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=64m",
            "--user",
            "10001:10001",
            "--env",
            "PYTHONDONTWRITEBYTECODE=1",
            "--env",
            f"PYTHONPATH={CONTAINER_WORKDIR}",
            "--env",
            "REPOSENTINEL_SANDBOX=1",
            "--volume",
            f"{self.workspace}:{CONTAINER_WORKDIR}:rw",
            "--workdir",
            workdir,
            "--stop-timeout",
            str(min(limit, 30)),
            self.image,
            *command,
        ]

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
            resolved = resolve_in_workspace(self.workspace, cwd)
        except Exception as exc:
            return CommandResult(
                command=command,
                exit_code=126,
                stderr=str(exc),
                blocked=True,
                block_reason="path escapes the workspace",
                backend=self.backend_name,
            )

        relative = resolved.relative_to(self.workspace).as_posix()
        workdir = CONTAINER_WORKDIR if relative in {"", "."} else f"{CONTAINER_WORKDIR}/{relative}"
        limit = timeout or self.timeout_seconds
        started = time.perf_counter()

        try:
            completed = subprocess.run(  # noqa: S603
                self._container_argv(command, workdir, limit),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                # Allow Docker itself a little headroom beyond the inner limit.
                timeout=limit + 20,
                shell=False,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return CommandResult(
                command=command,
                exit_code=124,
                stderr=f"container timed out after {limit}s",
                duration_ms=int((time.perf_counter() - started) * 1000),
                timed_out=True,
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
