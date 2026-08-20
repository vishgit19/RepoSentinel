"""Tool framework.

A tool is a plain function plus a JSON schema. The schema is what gets handed
to the model as an OpenAI-style function definition, and the same registry
backs the MCP server, so a tool is defined exactly once regardless of how it
is reached.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from reposentinel.config import Settings, get_settings
from reposentinel.models.schemas import SafetyEvent, ToolCallRecord
from reposentinel.retrieval.indexer import RepoIndex
from reposentinel.sandbox.base import Sandbox
from reposentinel.workspace import Workspace


@dataclass
class ToolResult:
    """What a tool hands back to the agent.

    ``ok`` and ``executed`` mean different things and both are needed. A pytest
    run that reports two failing tests has ``ok=False`` (the agent must see the
    failure) but ``executed=True`` (the tool itself worked perfectly - in fact
    reproducing the failure is the point). Only ``executed=False`` counts as an
    error in the trace, which keeps the observability panel honest.
    """

    ok: bool = True
    executed: bool = True
    summary: str = ""
    output: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    evidence: list[str] = field(default_factory=list)
    error: str | None = None
    safety_events: list[SafetyEvent] = field(default_factory=list)

    @classmethod
    def failure(cls, error: str, summary: str = "") -> ToolResult:
        """A tool that could not do its job at all."""
        return cls(
            ok=False, executed=False, error=error, summary=summary or error, output=error
        )


@dataclass
class ToolContext:
    """Everything a tool is allowed to touch."""

    workspace: Workspace
    sandbox: Sandbox
    settings: Settings = field(default_factory=get_settings)
    index: RepoIndex | None = None
    retriever: Any = None  # HybridRetriever; late-bound to avoid a cycle
    run_id: str = ""
    # Filled by the tools; the graph copies these into run state.
    safety_events: list[SafetyEvent] = field(default_factory=list)
    files_inspected: set[str] = field(default_factory=set)

    def record_safety(self, event: SafetyEvent) -> None:
        self.safety_events.append(event)


ToolHandler = Callable[..., ToolResult]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: ToolHandler
    category: str = "repository"
    mutating: bool = False
    expose_via_mcp: bool = False

    def openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        category: str = "repository",
        mutating: bool = False,
        expose_via_mcp: bool = False,
    ) -> Callable[[ToolHandler], ToolHandler]:
        def decorator(handler: ToolHandler) -> ToolHandler:
            if name in self._tools:
                raise ValueError(f"tool '{name}' is already registered")
            self._tools[name] = ToolSpec(
                name=name,
                description=description,
                parameters=parameters,
                handler=handler,
                category=category,
                mutating=mutating,
                expose_via_mcp=expose_via_mcp,
            )
            return handler

        return decorator

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def all(self) -> list[ToolSpec]:
        return sorted(self._tools.values(), key=lambda t: (t.category, t.name))

    def by_category(self, category: str) -> list[ToolSpec]:
        return [t for t in self.all() if t.category == category]

    def mcp_tools(self) -> list[ToolSpec]:
        return [t for t in self.all() if t.expose_via_mcp]

    def openai_schemas(self, names: list[str] | None = None) -> list[dict[str, Any]]:
        chosen = self.all() if names is None else [t for t in self.all() if t.name in names]
        return [t.openai_schema() for t in chosen]

    def names(self) -> list[str]:
        return [t.name for t in self.all()]


registry = ToolRegistry()


def string_param(description: str) -> dict[str, Any]:
    return {"type": "string", "description": description}


def int_param(description: str, default: int) -> dict[str, Any]:
    return {"type": "integer", "description": description, "default": default}


def schema(
    properties: dict[str, Any], required: list[str] | None = None
) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required or [],
        "additionalProperties": False,
    }


def execute_tool(
    name: str,
    arguments: dict[str, Any],
    context: ToolContext,
    *,
    via_mcp: bool = False,
) -> ToolCallRecord:
    """Run a tool by name, always returning a record (never raising)."""
    spec = registry.get(name)
    started = time.perf_counter()

    if spec is None:
        return ToolCallRecord(
            name=name,
            arguments=arguments,
            ok=False,
            executed=False,
            error=f"unknown tool '{name}'",
            summary=f"unknown tool '{name}'",
            duration_ms=0,
            via_mcp=via_mcp,
        )

    try:
        result = spec.handler(context, **arguments)
    except TypeError as exc:
        result = ToolResult.failure(f"invalid arguments for '{name}': {exc}")
    except Exception as exc:  # noqa: BLE001 - a tool failure must not kill the run
        result = ToolResult.failure(f"{type(exc).__name__}: {exc}")

    duration_ms = int((time.perf_counter() - started) * 1000)
    limit = context.settings.limits.max_tool_output_chars
    output = result.output or ""
    if len(output) > limit:
        output = f"{output[:limit]}\n... [truncated {len(output) - limit} chars]"

    for event in result.safety_events:
        context.record_safety(event)

    blocked = any(e.kind == "blocked_command" for e in result.safety_events)
    return ToolCallRecord(
        name=name,
        arguments=arguments,
        ok=result.ok,
        executed=result.executed and not blocked,
        summary=result.summary,
        output=output,
        error=result.error,
        duration_ms=duration_ms,
        blocked=blocked,
        block_reason=next(
            (e.detail for e in result.safety_events if e.kind == "blocked_command"), None
        ),
        via_mcp=via_mcp,
    )
