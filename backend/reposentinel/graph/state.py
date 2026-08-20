"""LangGraph state and per-run context.

The graph state is a ``TypedDict`` whose *values* are Pydantic models. That
combination is deliberate: append-only channels (``tool_history``,
``timeline``, ``patches``, ...) get LangGraph reducers so a node returns only
what it added, while every value inside remains a validated model that can be
persisted and replayed.

Runtime objects that cannot be serialised - the workspace, the sandbox, the
model provider - live in :class:`RunContext`, which is bound to the nodes when
the graph is compiled rather than stored in state.
"""

from __future__ import annotations

import operator
import time
from dataclasses import dataclass, field
from typing import Annotated, Any, TypedDict

from reposentinel.config import Settings, get_settings
from reposentinel.models.providers.base import ModelProvider
from reposentinel.models.schemas import (
    CodeChunk,
    Diagnosis,
    FinalReport,
    LintReport,
    LLMCallRecord,
    PatchRecord,
    RepairPlan,
    RootCause,
    RunStatus,
    SafetyEvent,
    SecurityReport,
    StepStatus,
    TestReport,
    TimelineEvent,
    ToolCallRecord,
    TriageResult,
    VerificationVerdict,
)


class LimitExceeded(Exception):
    """Raised when a run hits one of its hard ceilings."""

    def __init__(self, which: str, detail: str) -> None:
        super().__init__(f"{which}: {detail}")
        self.which = which
        self.detail = detail


@dataclass
class Budget:
    """Live accounting for one run, checked between and inside nodes."""

    settings: Settings
    started_at: float = field(default_factory=time.time)
    tool_calls: int = 0
    blocked_tool_calls: int = 0
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    retries: int = 0
    retrieval_queries: int = 0
    retrieved_chunks: int = 0
    patches_attempted: int = 0

    @property
    def elapsed_seconds(self) -> float:
        return time.time() - self.started_at

    @property
    def elapsed_ms(self) -> int:
        return int(self.elapsed_seconds * 1000)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def check(self) -> None:
        limits = self.settings.limits
        if self.tool_calls >= limits.max_tool_calls:
            raise LimitExceeded("max_tool_calls", f"{self.tool_calls}/{limits.max_tool_calls}")
        if self.llm_calls >= limits.max_llm_calls:
            raise LimitExceeded("max_llm_calls", f"{self.llm_calls}/{limits.max_llm_calls}")
        if self.total_tokens >= limits.max_tokens:
            raise LimitExceeded("max_tokens", f"{self.total_tokens}/{limits.max_tokens}")
        if self.cost_usd >= limits.max_cost_usd:
            raise LimitExceeded("max_cost_usd", f"${self.cost_usd:.4f}/${limits.max_cost_usd}")
        if self.elapsed_seconds >= limits.wall_clock_seconds:
            raise LimitExceeded(
                "wall_clock_seconds", f"{self.elapsed_seconds:.0f}s/{limits.wall_clock_seconds}s"
            )

    def remaining(self) -> dict[str, Any]:
        limits = self.settings.limits
        return {
            "tool_calls": limits.max_tool_calls - self.tool_calls,
            "llm_calls": limits.max_llm_calls - self.llm_calls,
            "tokens": limits.max_tokens - self.total_tokens,
            "cost_usd": round(limits.max_cost_usd - self.cost_usd, 4),
            "seconds": int(limits.wall_clock_seconds - self.elapsed_seconds),
        }

    def snapshot(self) -> dict[str, Any]:
        return {
            "tool_calls": self.tool_calls,
            "blocked_tool_calls": self.blocked_tool_calls,
            "llm_calls": self.llm_calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "retries": self.retries,
            "retrieval_queries": self.retrieval_queries,
            "retrieved_chunks": self.retrieved_chunks,
            "patches_attempted": self.patches_attempted,
            "latency_ms": self.elapsed_ms,
        }


@dataclass
class RunContext:
    """Non-serialisable machinery for one run."""

    run_id: str
    workspace: Any  # Workspace
    sandbox: Any  # Sandbox
    provider: ModelProvider
    settings: Settings = field(default_factory=get_settings)
    index: Any = None  # RepoIndex
    retriever: Any = None  # HybridRetriever
    tool_context: Any = None  # ToolContext
    tracer: Any = None  # RunTracer
    emitter: Any = None  # TimelineEmitter
    memory: Any = None  # RepairMemory
    budget: Budget = None  # type: ignore[assignment]
    benchmark: Any = None  # BenchmarkManifest
    strategy: str = "agentic"
    memory_enabled: bool = True
    auto_approve: bool = False
    approval_gate: Any = None  # ApprovalGate

    def __post_init__(self) -> None:
        if self.budget is None:
            self.budget = Budget(settings=self.settings)


class GraphState(TypedDict, total=False):
    """The channel definitions for the repair graph."""

    # -- inputs
    run_id: str
    issue: str
    issue_id: str
    repo: str
    model: str
    strategy: str

    # -- understanding
    # Channel names must not collide with node names ('triage', 'root_cause'),
    # which LangGraph reserves, hence the '_result'/'_analysis' suffixes.
    triage_result: TriageResult | None
    plan: RepairPlan | None
    current_step: str
    hypotheses: Annotated[list[str], operator.add]
    root_cause_analysis: RootCause | None
    diagnosis: Diagnosis | None

    # -- evidence
    retrieved_context: Annotated[list[CodeChunk], operator.add]
    files_inspected: Annotated[list[str], operator.add]
    tool_history: Annotated[list[ToolCallRecord], operator.add]
    memory_hits: Annotated[list[dict], operator.add]
    evidence: Annotated[list[str], operator.add]

    # -- repair
    patches: Annotated[list[PatchRecord], operator.add]
    test_results: Annotated[list[TestReport], operator.add]
    lint_results: Annotated[list[LintReport], operator.add]
    security_results: Annotated[list[SecurityReport], operator.add]
    retry_count: int

    # -- outcome
    verification: VerificationVerdict | None
    verification_status: str
    status: str
    final_report: FinalReport | None
    failure_reason: str

    # -- observability
    timeline: Annotated[list[TimelineEvent], operator.add]
    llm_calls: Annotated[list[LLMCallRecord], operator.add]
    safety_events: Annotated[list[SafetyEvent], operator.add]
    metrics: dict


def initial_state(
    run_id: str,
    issue: str,
    repo: str,
    model: str,
    strategy: str = "agentic",
    issue_id: str = "",
) -> GraphState:
    return {
        "run_id": run_id,
        "issue": issue,
        "issue_id": issue_id,
        "repo": repo,
        "model": model,
        "strategy": strategy,
        "triage_result": None,
        "plan": None,
        "current_step": "input",
        "hypotheses": [],
        "root_cause_analysis": None,
        "diagnosis": None,
        "retrieved_context": [],
        "files_inspected": [],
        "tool_history": [],
        "memory_hits": [],
        "evidence": [],
        "patches": [],
        "test_results": [],
        "lint_results": [],
        "security_results": [],
        "retry_count": 0,
        "verification": None,
        "verification_status": "pending",
        "status": RunStatus.RUNNING.value,
        "final_report": None,
        "failure_reason": "",
        "timeline": [],
        "llm_calls": [],
        "safety_events": [],
        "metrics": {},
    }


# The node order the UI renders as a fixed vertical timeline. Nodes may repeat
# (patch/checks on retry); the UI collapses repeats into one row with a count.
DISPLAY_STEPS: tuple[tuple[str, str], ...] = (
    ("triage", "Triage"),
    ("memory", "Repair memory"),
    ("planner", "Planning"),
    ("retrieve", "Repository search"),
    ("tools", "Investigation"),
    ("root_cause", "Root cause"),
    ("patch", "Patch generation"),
    ("checks", "Testing & security"),
    ("reflect", "Self-correction"),
    ("verifier", "Verification"),
    ("approval", "Human approval"),
    ("report", "Final report"),
)


def step_labels() -> list[dict[str, str]]:
    return [
        {"node": node, "label": label, "status": StepStatus.PENDING.value}
        for node, label in DISPLAY_STEPS
    ]
