"""Pydantic schemas shared by the graph, the API and the evaluation harness.

Two families live here:

* **Structured LLM outputs** (``TriageResult``, ``RepairPlan``, ``RootCause``,
  ``PatchProposal``, ``Diagnosis``, ``VerificationVerdict``) - these are handed
  to the provider layer as JSON schemas so model replies are parsed, not
  scraped.
* **Run records** (``ToolCallRecord``, ``TestReport``, ``TimelineEvent``, ...)
  which are persisted so a finished run can be reopened and replayed.
"""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def now() -> float:
    return time.time()


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class NodeName(str, Enum):
    INPUT = "input"
    TRIAGE = "triage"
    MEMORY = "memory"
    PLANNER = "planner"
    INVESTIGATE = "investigate"
    RETRIEVE = "retrieve"
    TOOLS = "tools"
    ROOT_CAUSE = "root_cause"
    PATCH = "patch"
    CHECKS = "checks"
    REFLECT = "reflect"
    VERIFIER = "verifier"
    APPROVAL = "approval"
    REPORT = "report"


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILURE = "failure"
    SKIPPED = "skipped"
    BLOCKED = "blocked"


class IssueKind(str, Enum):
    LOGIC_BUG = "logic_bug"
    CROSS_FILE_BUG = "cross_file_bug"
    SECURITY_VULNERABILITY = "security_vulnerability"
    PERFORMANCE = "performance"
    TEST_FAILURE = "test_failure"
    UNKNOWN = "unknown"


class RunStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    AWAITING_APPROVAL = "awaiting_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    ABORTED = "aborted"


# ---------------------------------------------------------------------------
# Structured LLM outputs
# ---------------------------------------------------------------------------


class StrictModel(BaseModel):
    """Base for schemas sent to providers as strict JSON schemas."""

    model_config = ConfigDict(extra="forbid")


class TriageResult(StrictModel):
    issue_kind: IssueKind = Field(description="Best-guess category for the issue.")
    summary: str = Field(description="One sentence restatement of the problem.")
    confidence: float = Field(ge=0.0, le=1.0)
    suspected_areas: list[str] = Field(
        default_factory=list,
        description="Subsystem or directory names likely involved, e.g. 'auth', 'cache'.",
    )
    search_terms: list[str] = Field(
        default_factory=list,
        description="Identifiers or phrases worth searching for in the repository.",
    )
    needs_security_scan: bool = False
    reasoning_summary: str = Field(
        default="",
        description="Short user-facing justification. Not private chain-of-thought.",
    )


class PlanStep(StrictModel):
    index: int
    action: str = Field(description="Imperative description of the investigation step.")
    tool_hint: str = Field(
        default="",
        description="Tool the agent expects to use, e.g. 'search_symbols'.",
    )


class RepairPlan(StrictModel):
    goal: str
    steps: list[PlanStep]
    success_criteria: list[str] = Field(default_factory=list)


class RootCause(StrictModel):
    statement: str = Field(description="The defect, in one or two sentences.")
    file_path: str = Field(description="Primary file containing the defect.")
    symbol: str = Field(default="", description="Function/class containing the defect.")
    evidence: list[str] = Field(
        default_factory=list,
        description="Concrete observations (test output, code lines) supporting this.",
    )
    confidence: float = Field(ge=0.0, le=1.0)


class FileEdit(StrictModel):
    """A single exact-match replacement inside one file.

    Search/replace pairs are used instead of model-authored unified diffs
    because they can be validated deterministically before anything touches
    disk: the ``search`` text must occur exactly once in the target file.
    """

    path: str
    search: str = Field(description="Exact existing snippet, copied verbatim.")
    replace: str = Field(description="Replacement snippet.")
    rationale: str = Field(default="")


class PatchProposal(StrictModel):
    summary: str
    edits: list[FileEdit]
    tests_to_run: list[str] = Field(
        default_factory=list,
        description="Test paths or node ids to run first, cheapest signal first.",
    )
    risk_notes: list[str] = Field(default_factory=list)


class Diagnosis(StrictModel):
    """Produced by REFLECT after a failed check run."""

    failure_summary: str
    revised_hypothesis: str
    what_was_wrong_with_last_patch: str
    next_actions: list[str] = Field(default_factory=list)
    files_to_retrieve: list[str] = Field(
        default_factory=list,
        description="Additional files the agent now believes it needs.",
    )
    should_retry: bool = True


class VerificationVerdict(StrictModel):
    verified: bool
    explanation: str
    remaining_risks: list[str] = Field(default_factory=list)
    behaviour_preserved: bool = True


class RetrievalDecision(StrictModel):
    """Lets the agent decide *whether* to retrieve, and what for."""

    needs_retrieval: bool
    queries: list[str] = Field(
        default_factory=list,
        description="Focused natural-language or identifier queries.",
    )
    reason: str = ""


# ---------------------------------------------------------------------------
# Retrieval records
# ---------------------------------------------------------------------------


class Provenance(StrictModel):
    """Where a chunk came from and why it survived retrieval."""

    retriever: str = Field(description="bm25 | dense | graph | tool | seed")
    query: str = ""
    bm25_score: float | None = None
    dense_score: float | None = None
    fused_score: float | None = None
    rerank_score: float | None = None
    graph_relation: str | None = None
    graph_source: str | None = None
    commit: str | None = None


class CodeChunk(BaseModel):
    chunk_id: str
    repo_id: str
    path: str
    symbol: str = ""
    symbol_kind: str = "file"  # file | class | function | method
    start_line: int = 1
    end_line: int = 1
    content: str = ""
    language: str = "python"
    provenance: Provenance | None = None

    @property
    def location(self) -> str:
        return f"{self.path}:{self.start_line}-{self.end_line}"


class SymbolEdge(BaseModel):
    """One edge of the repository knowledge graph.

    Relations mirror the spec: calls, contains, imports, tests, modified_by.
    """

    source: str
    relation: Literal["calls", "contains", "imports", "tests", "modified_by"]
    target: str
    path: str = ""
    line: int = 0


# ---------------------------------------------------------------------------
# Tool / execution records
# ---------------------------------------------------------------------------


class ToolCallRecord(BaseModel):
    call_id: str = Field(default_factory=lambda: new_id("tc"))
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    # ``ok`` is the finding (did the tests pass?); ``executed`` is whether the
    # tool ran at all. A reproduced test failure is ok=False, executed=True.
    ok: bool = True
    executed: bool = True
    summary: str = ""
    output: str = ""
    error: str | None = None
    duration_ms: int = 0
    blocked: bool = False
    block_reason: str | None = None
    via_mcp: bool = False
    started_at: float = Field(default_factory=now)


class TestCaseResult(BaseModel):
    node_id: str
    outcome: Literal["passed", "failed", "error", "skipped"]
    message: str = ""
    duration_ms: int = 0


class TestReport(BaseModel):
    command: str
    scope: Literal["targeted", "full"] = "targeted"
    passed: int = 0
    failed: int = 0
    errors: int = 0
    skipped: int = 0
    exit_code: int = -1
    duration_ms: int = 0
    failures: list[TestCaseResult] = Field(default_factory=list)
    stdout_tail: str = ""
    timed_out: bool = False

    @property
    def total(self) -> int:
        return self.passed + self.failed + self.errors + self.skipped

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and self.failed == 0 and self.errors == 0


class LintReport(BaseModel):
    tool: str = "ruff"
    ok: bool = True
    issue_count: int = 0
    issues: list[str] = Field(default_factory=list)
    exit_code: int = 0


class SecurityFinding(BaseModel):
    rule_id: str
    severity: Literal["INFO", "WARNING", "ERROR"] = "WARNING"
    message: str
    path: str
    line: int = 0
    snippet: str = ""
    cwe: str = ""


class SecurityReport(BaseModel):
    backend: str = "builtin"
    ok: bool = True
    findings: list[SecurityFinding] = Field(default_factory=list)
    files_scanned: int = 0
    duration_ms: int = 0

    @property
    def blocking_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == "ERROR")


class PatchRecord(BaseModel):
    """An applied (or attempted) patch, with the real diff produced on disk."""

    patch_id: str = Field(default_factory=lambda: new_id("patch"))
    attempt: int = 1
    summary: str = ""
    diff: str = ""
    files_changed: list[str] = Field(default_factory=list)
    lines_added: int = 0
    lines_removed: int = 0
    applied: bool = False
    apply_error: str | None = None
    created_at: float = Field(default_factory=now)


class SafetyEvent(BaseModel):
    kind: Literal[
        "blocked_command",
        "path_escape",
        "prompt_injection",
        "secret_redacted",
        "limit_exceeded",
        "approval_required",
    ]
    detail: str
    source: str = ""
    severity: Literal["info", "warning", "critical"] = "warning"
    at: float = Field(default_factory=now)


# ---------------------------------------------------------------------------
# Observability
# ---------------------------------------------------------------------------


class LLMCallRecord(BaseModel):
    call_id: str = Field(default_factory=lambda: new_id("llm"))
    node: str = ""
    provider: str = ""
    model: str = ""
    purpose: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    duration_ms: int = 0
    ok: bool = True
    error: str | None = None
    input_chars: int = 0
    output_chars: int = 0


class RunMetrics(BaseModel):
    tool_calls: int = 0
    blocked_tool_calls: int = 0
    llm_calls: int = 0
    retrieval_queries: int = 0
    retrieved_chunks: int = 0
    retries: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0
    files_inspected: int = 0
    patches_attempted: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class TimelineEvent(BaseModel):
    """One entry in the user-visible execution timeline.

    Deliberately contains only summarised decisions - never raw model
    reasoning - so the UI can be public-facing.
    """

    event_id: str = Field(default_factory=lambda: new_id("ev"))
    run_id: str = ""
    seq: int = 0
    node: str = ""
    title: str = ""
    status: StepStatus = StepStatus.RUNNING
    detail: str = ""
    lines: list[str] = Field(default_factory=list)
    tool_call: ToolCallRecord | None = None
    evidence: list[str] = Field(default_factory=list)
    # How long this step's own work took, versus wall-clock since the run
    # began. Keeping them separate stops a step with no measurable work of its
    # own (an approval gate, say) from appearing to have taken the whole run.
    duration_ms: int = 0
    run_elapsed_ms: int = 0
    tokens: int = 0
    cost_usd: float = 0.0
    at: float = Field(default_factory=now)


class TraceSpan(BaseModel):
    span_id: str = Field(default_factory=lambda: new_id("span"))
    run_id: str = ""
    parent_id: str | None = None
    name: str = ""
    kind: Literal["run", "node", "llm", "retrieval", "tool", "test", "security", "patch"] = "node"
    started_at: float = Field(default_factory=now)
    ended_at: float | None = None
    duration_ms: int = 0
    ok: bool = True
    error: str | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Requests / reports
# ---------------------------------------------------------------------------


class RunRequest(BaseModel):
    issue: str = Field(min_length=1)
    repo: str = Field(
        description="Benchmark id, local path, or GitHub URL.",
    )
    issue_id: str = ""
    model: str = ""
    strategy: str = Field(
        default="agentic",
        description="agentic | llm_only | vector_rag | hybrid_rag | graph_rag",
    )
    memory_enabled: bool | None = None
    seed_files: list[str] = Field(default_factory=list)
    benchmark_id: str = ""
    auto_approve: bool = False


class FinalReport(BaseModel):
    problem: str = ""
    root_cause: str = ""
    changed_files: list[str] = Field(default_factory=list)
    explanation: str = ""
    evidence: list[str] = Field(default_factory=list)
    validation_performed: list[str] = Field(default_factory=list)
    remaining_risks: list[str] = Field(default_factory=list)
    diff: str = ""
    verified: bool = False
