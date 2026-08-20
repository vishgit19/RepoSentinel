"""Sandbox selection."""

from __future__ import annotations

from pathlib import Path

from reposentinel.config import Settings, get_settings
from reposentinel.sandbox.base import CommandResult, Sandbox
from reposentinel.sandbox.docker import DockerSandbox, docker_available
from reposentinel.sandbox.guardrails import GuardrailViolation
from reposentinel.sandbox.local import LocalSandbox

__all__ = [
    "CommandResult",
    "DockerSandbox",
    "GuardrailViolation",
    "LocalSandbox",
    "Sandbox",
    "build_sandbox",
    "describe_backend",
]


def build_sandbox(workspace: Path, settings: Settings | None = None) -> Sandbox:
    settings = settings or get_settings()
    timeout = settings.limits.sandbox_command_seconds
    choice = settings.sandbox_backend

    if choice == "docker" or (choice == "auto" and docker_available()):
        return DockerSandbox(
            workspace, timeout_seconds=timeout, image=settings.sandbox_docker_image
        )
    return LocalSandbox(workspace, timeout_seconds=timeout)


def describe_backend(settings: Settings | None = None) -> dict[str, object]:
    settings = settings or get_settings()
    available = docker_available()
    if settings.sandbox_backend == "docker" or (settings.sandbox_backend == "auto" and available):
        active = "docker"
    else:
        active = "local"
    return {
        "configured": settings.sandbox_backend,
        "active": active,
        "docker_available": available,
    }
