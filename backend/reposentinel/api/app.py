"""The RepoSentinel HTTP API.

Serves three things:

* a small JSON API for starting runs, reading run history and approving patches,
* a Server-Sent Events stream carrying the live execution timeline,
* the static frontend, so ``uvicorn`` alone is enough to demo the system.

The agent graph is synchronous and blocking (it shells out to pytest and waits
on HTTP), so runs execute on worker threads and reach the event loop through the
:class:`~reposentinel.observability.events.EventBus`.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from reposentinel.benchmarks import list_benchmarks
from reposentinel.config import get_settings
from reposentinel.evaluation.harness import EvaluationHarness
from reposentinel.graph.state import step_labels
from reposentinel.graph.workflow import graph_topology
from reposentinel.memory.store import RepairMemory
from reposentinel.models.providers import available_models
from reposentinel.models.schemas import RunRequest
from reposentinel.observability.events import bus
from reposentinel.observability.store import get_store
from reposentinel.orchestrator import get_orchestrator
from reposentinel.retrieval.embeddings import build_embedding_backend
from reposentinel.sandbox import describe_backend
from reposentinel.security_scan import resolve_backend as resolve_security_backend
from reposentinel.strategies import describe_strategies
from reposentinel.tools.base import registry

FRONTEND_DIR = Path(__file__).resolve().parents[3] / "frontend"


class ApprovalRequest(BaseModel):
    approved: bool
    note: str = ""


class EvaluationRequest(BaseModel):
    benchmark_ids: list[str] = Field(default_factory=list)
    strategies: list[str] = Field(default_factory=list)
    model: str = ""
    memory_enabled: bool | None = None
    suite_label: str = ""


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Runs publish events from worker threads, so the bus needs a handle on the
    # serving loop before the first run starts.
    bus.bind_loop(asyncio.get_running_loop())
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="RepoSentinel",
        version="1.0.0",
        description="Agentic secure code repair and verification.",
        lifespan=lifespan,
    )

    orchestrator = get_orchestrator(settings)
    store = get_store(settings)
    harness = EvaluationHarness(orchestrator=orchestrator, store=store, settings=settings)

    @app.middleware("http")
    async def no_store_frontend(request: Request, call_next):
        response = await call_next(request)
        path = request.url.path
        if path == "/" or path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    # -- capability discovery ------------------------------------------------
    @app.get("/api/health")
    def health() -> dict[str, Any]:
        embeddings = build_embedding_backend(settings)
        return {
            "status": "ok",
            "sandbox": describe_backend(settings),
            "security_backend": resolve_security_backend(settings),
            "vector_store": settings.vector_store,
            "embeddings": embeddings.describe(),
            "providers": settings.provider_availability(),
            "memory_enabled": settings.memory_enabled,
            "memory_records": RepairMemory(
                settings.data_dir / "memory.db", embeddings=embeddings
            ).count(),
            "limits": settings.limits.model_dump(),
            "tools": len(registry.all()),
            "mcp_tools": len(registry.mcp_tools()),
        }

    @app.get("/api/benchmarks")
    def benchmarks() -> list[dict[str, Any]]:
        return [
            {
                "id": manifest.id,
                "title": manifest.title,
                "category": manifest.category,
                "difficulty": manifest.difficulty,
                "issue": manifest.issue,
                "issue_id": manifest.issue_id,
                "gold_files": manifest.gold_files,
                "relevant_files": manifest.relevant_files,
                "expected_failing_tests": manifest.expected_failing_tests,
                "expected_behaviour": manifest.expected_behaviour,
                "expects_injection": manifest.expects_injection,
                "expected_retry": manifest.expected_retry,
                "security_scan_required": manifest.security_scan_required,
            }
            for manifest in list_benchmarks()
        ]

    @app.get("/api/models")
    def models() -> dict[str, Any]:
        return {"default": settings.default_model, "models": available_models(settings)}

    @app.get("/api/strategies")
    def strategies() -> list[dict[str, Any]]:
        return describe_strategies()

    @app.get("/api/topology")
    def topology() -> dict[str, Any]:
        return {"graph": graph_topology(), "steps": step_labels()}

    @app.get("/api/tools")
    def tools() -> list[dict[str, Any]]:
        return [
            {
                "name": spec.name,
                "description": spec.description,
                "category": spec.category,
                "mcp": spec.expose_via_mcp,
                "parameters": spec.parameters,
            }
            for spec in registry.all()
        ]

    # -- runs ----------------------------------------------------------------
    @app.post("/api/runs", status_code=202)
    async def start_run(request: RunRequest) -> dict[str, Any]:
        try:
            handle = await orchestrator.start(request)
        except Exception as exc:  # noqa: BLE001 - surface config errors as 400
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"run_id": handle.run_id, "status": handle.status, "steps": step_labels()}

    @app.get("/api/runs")
    def list_runs(limit: int = 50, benchmark_id: str = "") -> list[dict[str, Any]]:
        return store.list_runs(limit=limit, benchmark_id=benchmark_id or None)

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str) -> dict[str, Any]:
        record = store.get_run(run_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"unknown run '{run_id}'")
        record["events"] = store.get_events(run_id)
        record["spans"] = store.get_spans(run_id)

        # A run still in flight has not been persisted yet; prefer live data.
        handle = orchestrator.get(run_id)
        if handle is not None:
            record["status"] = handle.status
            record["live"] = handle.thread is not None and handle.thread.is_alive()
            if not record["events"]:
                record["events"] = [
                    event
                    for event in bus.replay(run_id)
                    if event.get("type") == "timeline"
                ]
            if handle.state:
                record["state"] = handle.state
        else:
            record["live"] = False
        return record

    @app.get("/api/runs/{run_id}/events")
    async def stream_events(run_id: str, request: Request) -> StreamingResponse:
        async def event_source() -> AsyncIterator[bytes]:
            async for event in bus.subscribe(run_id):
                if await request.is_disconnected():
                    break
                payload = json.dumps(event, default=str)
                yield f"event: {event.get('type', 'message')}\ndata: {payload}\n\n".encode()

        return StreamingResponse(
            event_source(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
                # Defeats nginx response buffering, which would otherwise hold
                # the whole timeline back until the run finished.
                "X-Accel-Buffering": "no",
            },
        )

    @app.post("/api/runs/{run_id}/approval")
    async def decide(run_id: str, decision: ApprovalRequest) -> dict[str, Any]:
        # Must not run on the event loop: the graph thread is blocked on the
        # approval Event, and the SSE stream is already occupying this loop.
        ok = await asyncio.to_thread(
            orchestrator.decide, run_id, decision.approved, decision.note
        )
        if not ok:
            raise HTTPException(
                status_code=404, detail=f"run '{run_id}' is not awaiting a decision"
            )
        return {"run_id": run_id, "approved": decision.approved}

    @app.delete("/api/runs/{run_id}")
    def delete_run(run_id: str) -> dict[str, Any]:
        store.delete_run(run_id)
        bus.forget(run_id)
        return {"deleted": run_id}

    # -- evaluation ----------------------------------------------------------
    @app.post("/api/evaluations", status_code=202)
    async def start_evaluation(request: EvaluationRequest) -> dict[str, Any]:
        bus.bind_loop(asyncio.get_running_loop())
        try:
            suite_id = harness.start_background(
                benchmark_ids=request.benchmark_ids,
                approaches=request.strategies,
                model=request.model,
                memory_enabled=request.memory_enabled,
                label=request.suite_label,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"suite_id": suite_id}

    @app.get("/api/evaluations")
    def list_evaluations() -> dict[str, Any]:
        return {
            "suites": store.list_eval_suites(),
            "active": harness.active_suites(),
        }

    @app.get("/api/evaluations/{suite_id}")
    def get_evaluation(suite_id: str) -> dict[str, Any]:
        results = store.get_eval_results(suite_id)
        if not results and suite_id not in harness.active_suites():
            raise HTTPException(status_code=404, detail=f"unknown suite '{suite_id}'")
        return harness.summarise(suite_id, results)

    @app.get("/api/evaluations/{suite_id}/events")
    async def stream_evaluation(suite_id: str, request: Request) -> StreamingResponse:
        async def event_source() -> AsyncIterator[bytes]:
            async for event in bus.subscribe(suite_id):
                if await request.is_disconnected():
                    break
                payload = json.dumps(event, default=str)
                yield f"event: {event.get('type', 'message')}\ndata: {payload}\n\n".encode()

        return StreamingResponse(
            event_source(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
        )

    # -- frontend ------------------------------------------------------------
    if FRONTEND_DIR.is_dir():
        app.mount(
            "/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static"
        )

        @app.get("/")
        def index() -> FileResponse:
            return FileResponse(FRONTEND_DIR / "index.html")

    else:  # pragma: no cover - only if the frontend directory is missing

        @app.get("/")
        def index_missing() -> JSONResponse:
            return JSONResponse(
                {"detail": f"frontend not found at {FRONTEND_DIR}"}, status_code=500
            )

    return app


app = create_app()
