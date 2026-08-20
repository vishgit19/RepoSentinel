"""Sandbox abstraction shared by the local and Docker backends."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from pathlib import Path

from reposentinel.sandbox.guardrails import redact_secrets, truncate


@dataclass
class CommandResult:
    command: list[str]
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    duration_ms: int = 0
    timed_out: bool = False
    blocked: bool = False
    block_reason: str = ""
    backend: str = "local"
    redactions: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.blocked and not self.timed_out

    @property
    def combined_output(self) -> str:
        parts = [p for p in (self.stdout, self.stderr) if p]
        return "\n".join(parts)

    def sanitised(self, limit: int = 20_000) -> str:
        text, labels = redact_secrets(self.combined_output)
        self.redactions.extend(labels)
        return truncate(text, limit, note="command output")


class Sandbox(abc.ABC):
    """Executes commands against a single workspace directory.

    Implementations must honour three invariants:

    1. commands are executed without a shell,
    2. the working directory never leaves ``workspace``,
    3. the child process cannot read the host's secret environment.
    """

    backend_name: str = "abstract"

    def __init__(self, workspace: Path, timeout_seconds: int = 180) -> None:
        self.workspace = workspace.resolve()
        self.timeout_seconds = timeout_seconds

    @abc.abstractmethod
    def run(
        self,
        command: list[str],
        *,
        cwd: str = ".",
        timeout: int | None = None,
    ) -> CommandResult:
        """Execute ``command`` and return its result."""

    @staticmethod
    @abc.abstractmethod
    def is_available() -> bool:
        """Whether this backend can be used on the current host."""
