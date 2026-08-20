"""Run orchestration.

Assembles everything a run needs - workspace, index, retriever, provider,
memory, tracer, emitter, sandbox - drives the LangGraph workflow, streams
progress, and persists the result so the run can be reopened later.

The graph itself is synchronous (its nodes block on subprocesses and HTTP), so
it is executed in a worker thread while the API keeps serving. Events cross the
boundary through :class:`~reposentinel.observability.events.EventBus`.
"""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from reposentinel.benchmarks import get_benchmark
from reposentinel.config import Settings, get_settings
from reposentinel.graph.nodes.finalize import ApprovalGate
from reposentinel.graph.state import Budget, RunContext, initial_state, step_labels
from reposentinel.graph.workflow import build_graph
from reposentinel.memory.store import RepairMemory
from reposentinel.models.providers import build_provider
from reposentinel.models.schemas import (
    RunRequest,
    RunStatus,
    StepStatus,
    new_id,
)
from reposentinel.observability.emitter import TimelineEmitter
from reposentinel.observability.events import EventBus
from reposentinel.observability.events import bus as default_bus
from reposentinel.observability.store import RunStore, get_store
from reposentinel.observability.tracer import RunTracer
from reposentinel.retrieval.embeddings import HashingEmbeddings, build_embedding_backend
from reposentinel.retrieval.indexer import CodeIndexer, attach_commit_edges
from reposentinel.retrieval.pipeline import HybridRetriever
from reposentinel.retrieval.reranker import LexicalReranker, LLMReranker
from reposentinel.retrieval.vector_store import build_vector_store
from reposentinel.sandbox import build_sandbox
from reposentinel.tools.base import ToolContext
from reposentinel.workspace import Workspace

# Recursion headroom: the linear path plus max_repair_attempts loops.
RECURSION_LIMIT = 80


def serialize_state(state: dict[str, Any]) -> dict[str, Any]:
    """Convert graph state into JSON-safe data for persistence and the API."""
    payload: dict[str, Any] = {}
    for key, value in state.items():
        if hasattr(value, "model_dump"):
            payload[key] = value.model_dump(mode="json")
        elif isinstance(value, list):
            payload[key] = [
                item.model_dump(mode="json") if hasattr(item, "model_dump") else item
                for item in value
            ]
        else:
            payload[key] = value
    return payload


@dataclass
class RunHandle:
    run_id: str
    request: RunRequest
    context: RunContext | None = None
    approval_gate: ApprovalGate = field(default_factory=ApprovalGate)
    state: dict[str, Any] = field(default_factory=dict)
    status: str = RunStatus.QUEUED.value
    error: str | None = None
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    thread: threading.Thread | None = None
    workspace_root: str = ""

    def snapshot(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "error": self.error,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "request": self.request.model_dump(),
            "workspace": self.workspace_root,
        }


class Orchestrator:
    def __init__(
        self,
        settings: Settings | None = None,
        bus: EventBus | None = None,
        store: RunStore | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.bus = bus or default_bus
        self.store = store or get_store(self.settings)
        self.runs: dict[str, RunHandle] = {}
        self._lock = threading.Lock()

    # -- lifecycle -------------------------------------------------------
    def create_run(self, request: RunRequest) -> RunHandle:
        # Validated before the run is registered so a bad repository is a
        # rejected request rather than a run that dies on a worker thread.
        Workspace.validate_source(request.benchmark_id or request.repo)

        run_id = new_id("run")
        handle = RunHandle(run_id=run_id, request=request)
        with self._lock:
            self.runs[run_id] = handle

        benchmark = get_benchmark(request.benchmark_id or request.repo)
        self.store.create_run(
            run_id,
            status=RunStatus.QUEUED.value,
            issue=request.issue,
            issue_id=request.issue_id,
            repo=request.repo,
            benchmark_id=benchmark.id if benchmark else "",
            model=request.model or self.settings.default_model,
            provider="openai",
            strategy=request.strategy,
            memory_enabled=(
                self.settings.memory_enabled
                if request.memory_enabled is None
                else request.memory_enabled
            ),
        )
        self.bus.publish(
            run_id,
            {
                "type": "run_created",
                "run_id": run_id,
                "steps": step_labels(),
                "request": request.model_dump(),
            },
        )
        return handle

    def start_background(self, handle: RunHandle) -> None:
        thread = threading.Thread(
            target=self._execute_safely, args=(handle,), name=f"run-{handle.run_id}", daemon=True
        )
        handle.thread = thread
        thread.start()

    async def start(self, request: RunRequest) -> RunHandle:
        loop = asyncio.get_running_loop()
        self.bus.bind_loop(loop)
        handle = self.create_run(request)
        self.start_background(handle)
        return handle

    def run_blocking(self, request: RunRequest) -> RunHandle:
        """Execute a run to completion on the calling thread (evaluation harness)."""
        handle = self.create_run(request)
        self._execute_safely(handle)
        return handle

    # -- execution -------------------------------------------------------
    def _execute_safely(self, handle: RunHandle) -> None:
        try:
            self._execute(handle)
        except Exception as exc:  # noqa: BLE001 - the run thread must never die silently
            handle.status = RunStatus.FAILED.value
            handle.error = f"{type(exc).__name__}: {exc}"
            handle.finished_at = time.time()
            self.store.update_run(
                handle.run_id, status=handle.status, failure_reason=handle.error[:500]
            )
            self.bus.publish(
                handle.run_id,
                {
                    "type": "status",
                    "run_id": handle.run_id,
                    "status": handle.status,
                    "detail": handle.error,
                },
            )
        finally:
            self.bus.mark_finished(handle.run_id)

    def _build_context(self, handle: RunHandle) -> RunContext:
        request = handle.request
        settings = self.settings
        run_id = handle.run_id

        benchmark = get_benchmark(request.benchmark_id or request.repo)
        source = benchmark.id if benchmark else request.repo

        workspace = Workspace.prepare(source, run_id, settings=settings)
        handle.workspace_root = str(workspace.root)
        sandbox = build_sandbox(workspace.root, settings=settings)

        provider = build_provider(request.model or settings.default_model, settings)

        # Index the repository: AST symbols, symbol-boundary chunks, graph edges.
        relative_paths = workspace.relative_files(extensions=(".py",))
        index = CodeIndexer(run_id, workspace.root).index(relative_paths)
        attach_commit_edges(
            index, {path: workspace.file_history(path, limit=3) for path in relative_paths[:40]}
        )

        embeddings = build_embedding_backend(settings)
        try:
            vector_store = build_vector_store(settings, dimensions=embeddings.dimensions)
        except Exception:  # noqa: BLE001 - fall back rather than fail the run
            embeddings = HashingEmbeddings()
            vector_store = build_vector_store(settings, dimensions=embeddings.dimensions)

        reranker = LLMReranker(provider) if settings.openai_api_key else LexicalReranker()

        retriever = HybridRetriever(
            repo_id=run_id,
            index=index,
            embeddings=embeddings,
            vector_store=vector_store,
            reranker=reranker,
            settings=settings,
        )
        build_stats = retriever.build()

        memory_enabled = (
            settings.memory_enabled if request.memory_enabled is None else request.memory_enabled
        )
        memory = RepairMemory(settings.data_dir / "memory.db", embeddings=embeddings)

        budget = Budget(settings=settings)
        emitter = TimelineEmitter(run_id=run_id, bus=self.bus, budget=budget)
        tracer = RunTracer(run_id=run_id)
        tracer.spans[0].attributes.update(
            {
                "model": provider.model,
                "provider": provider.name,
                "strategy": request.strategy,
                "repo": source,
                "sandbox": sandbox.backend_name,
                "index": index.stats(),
                "retrieval_build": {
                    k: v for k, v in build_stats.items() if k in {"chunks_indexed", "embed_ms"}
                },
            }
        )

        tool_context = ToolContext(
            workspace=workspace,
            sandbox=sandbox,
            settings=settings,
            index=index,
            retriever=retriever,
            run_id=run_id,
        )

        return RunContext(
            run_id=run_id,
            workspace=workspace,
            sandbox=sandbox,
            provider=provider,
            settings=settings,
            index=index,
            retriever=retriever,
            tool_context=tool_context,
            tracer=tracer,
            emitter=emitter,
            memory=memory,
            budget=budget,
            benchmark=benchmark,
            strategy=request.strategy,
            memory_enabled=memory_enabled,
            auto_approve=request.auto_approve,
            approval_gate=handle.approval_gate,
        )

    def _execute(self, handle: RunHandle) -> None:
        request = handle.request
        handle.status = RunStatus.RUNNING.value
        self.store.update_run(handle.run_id, status=handle.status)
        self.bus.publish(
            handle.run_id,
            {"type": "status", "run_id": handle.run_id, "status": handle.status, "detail": "preparing workspace"},
        )

        context = self._build_context(handle)
        handle.context = context

        self.bus.publish(
            handle.run_id,
            {
                "type": "run_started",
                "run_id": handle.run_id,
                "workspace": context.workspace.describe(),
                "index": context.index.stats(),
                "retrieval": context.retriever.describe(),
                "model": context.provider.describe(),
                "sandbox": context.sandbox.backend_name,
                "memory_enabled": context.memory_enabled,
                "memory_records": context.memory.count() if context.memory else 0,
            },
        )

        graph = build_graph(context)
        state = initial_state(
            run_id=handle.run_id,
            issue=request.issue,
            repo=request.repo,
            model=context.provider.model,
            strategy=request.strategy,
            issue_id=request.issue_id,
        )

        ok = True
        error: str | None = None
        try:
            final_state = graph.invoke(
                state, config={"recursion_limit": RECURSION_LIMIT}
            )
        except Exception as exc:  # noqa: BLE001
            ok = False
            error = f"{type(exc).__name__}: {exc}"
            final_state = dict(state)
            final_state["status"] = RunStatus.FAILED.value
            final_state["failure_reason"] = error
            context.emitter.emit(
                "report",
                "Run failed",
                status=StepStatus.FAILURE,
                detail=error,
            )
        finally:
            context.tracer.finish(ok=ok, error=error)

        decided = handle.status in {
            RunStatus.APPROVED.value,
            RunStatus.REJECTED.value,
        }
        graph_status = final_state.get("status", RunStatus.FAILED.value)
        if decided and graph_status == RunStatus.AWAITING_APPROVAL.value:
            final_state["status"] = handle.status
        else:
            handle.status = graph_status
        handle.state = serialize_state(final_state)
        handle.error = final_state.get("failure_reason") or error
        handle.finished_at = time.time()

        self._persist(handle, context, final_state)
        self.bus.publish(
            handle.run_id,
            {
                "type": "run_finished",
                "run_id": handle.run_id,
                "status": handle.status,
                "metrics": context.budget.snapshot(),
                "trace": context.tracer.totals(),
            },
        )

    def _persist(self, handle: RunHandle, context: RunContext, state: dict[str, Any]) -> None:
        reports = state.get("test_results") or []
        full = [r for r in reports if r.scope == "full"]
        latest_tests = full[-1] if full else (reports[-1] if reports else None)
        security = state.get("security_results") or []
        patches = state.get("patches") or []
        verification = state.get("verification")
        budget = context.budget

        self.store.update_run(
            handle.run_id,
            status=handle.status,
            verified=int(bool(verification and verification.verified)),
            retries=max(0, (state.get("retry_count", 1) or 1) - 1),
            tool_calls=budget.tool_calls,
            llm_calls=budget.llm_calls,
            total_tokens=budget.total_tokens,
            cost_usd=round(budget.cost_usd, 6),
            latency_ms=budget.elapsed_ms,
            tests_passed=latest_tests.passed if latest_tests else 0,
            tests_failed=(latest_tests.failed + latest_tests.errors) if latest_tests else 0,
            security_ok=int(bool(not security or security[-1].ok)),
            failure_reason=(handle.error or "")[:500],
            diff=patches[-1].diff if patches else "",
        )
        self.store.save_state(handle.run_id, handle.state)
        self.store.append_events(
            handle.run_id,
            [event.model_dump(mode="json") for event in context.emitter.events],
        )
        self.store.save_spans(handle.run_id, context.tracer.tree())

    # -- human in the loop -----------------------------------------------
    def decide(self, run_id: str, approved: bool, note: str = "") -> bool:
        handle = self.runs.get(run_id)
        if handle is None:
            return False
        handle.approval_gate.decide(approved, note)
        handle.status = (
            RunStatus.APPROVED.value if approved else RunStatus.REJECTED.value
        )
        # Publish before persisting so the UI leaves the approval bar even if
        # SQLite is briefly busy. The graph thread is woken by decide() above.
        self.bus.publish(
            run_id,
            {
                "type": "status",
                "run_id": run_id,
                "status": handle.status,
                "detail": "approved by reviewer" if approved else "rejected by reviewer",
            },
        )
        self.store.update_run(run_id, approved=int(approved), status=handle.status)
        return True

    def get(self, run_id: str) -> RunHandle | None:
        return self.runs.get(run_id)

    def cleanup_workspace(self, run_id: str) -> None:
        handle = self.runs.get(run_id)
        if handle and handle.context is not None:
            handle.context.workspace.cleanup()


_orchestrator: Orchestrator | None = None
_orchestrator_lock = threading.Lock()


def get_orchestrator(settings: Settings | None = None) -> Orchestrator:
    global _orchestrator
    with _orchestrator_lock:
        if _orchestrator is None:
            _orchestrator = Orchestrator(settings=settings)
    return _orchestrator
