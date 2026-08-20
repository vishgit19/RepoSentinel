"""The evaluation harness.

Runs the same benchmark problem through several configurations of the same
system (baselines A-D and the full agent E) and records comparable metrics for
each. Suites execute on a worker thread and publish progress on the event bus
under the suite id, so the dashboard can watch a comparison being built the same
way it watches a single run.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

from reposentinel.benchmarks import get_benchmark, list_benchmarks
from reposentinel.config import Settings, get_settings
from reposentinel.evaluation.metrics import (
    agent_metrics,
    repair_metrics,
    retrieval_metrics,
    retrieved_paths_from_state,
    safety_metrics,
)
from reposentinel.models.schemas import RunRequest, new_id
from reposentinel.observability.events import EventBus
from reposentinel.observability.events import bus as default_bus
from reposentinel.observability.store import RunStore, get_store
from reposentinel.orchestrator import Orchestrator, get_orchestrator
from reposentinel.strategies import BY_ID, strategy_ids


@dataclass
class SuiteProgress:
    suite_id: str
    label: str
    total: int
    completed: int = 0
    started_at: float = field(default_factory=time.time)
    finished: bool = False
    current: str = ""
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite_id": self.suite_id,
            "label": self.label,
            "total": self.total,
            "completed": self.completed,
            "started_at": self.started_at,
            "finished": self.finished,
            "current": self.current,
            "errors": self.errors,
        }


class EvaluationHarness:
    def __init__(
        self,
        orchestrator: Orchestrator | None = None,
        store: RunStore | None = None,
        settings: Settings | None = None,
        bus: EventBus | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.orchestrator = orchestrator or get_orchestrator(self.settings)
        self.store = store or get_store(self.settings)
        self.bus = bus or default_bus
        self._suites: dict[str, SuiteProgress] = {}
        self._lock = threading.Lock()

    # -- public API ------------------------------------------------------
    def start_background(
        self,
        benchmark_ids: list[str] | None = None,
        approaches: list[str] | None = None,
        model: str = "",
        memory_enabled: bool | None = None,
        label: str = "",
    ) -> str:
        """Kick off a suite on a worker thread and return its id."""
        benchmarks, strategies = self._resolve(benchmark_ids, approaches or [])

        suite_id = new_id("suite")
        progress = SuiteProgress(
            suite_id=suite_id,
            label=label or f"{len(benchmarks)} problem(s) x {len(strategies)} approach(es)",
            total=len(benchmarks) * len(strategies),
        )
        with self._lock:
            self._suites[suite_id] = progress

        thread = threading.Thread(
            target=self._run_suite,
            args=(suite_id, benchmarks, strategies, model, memory_enabled),
            name=f"eval-{suite_id}",
            daemon=True,
        )
        thread.start()
        return suite_id

    def run_blocking(
        self,
        benchmark_ids: list[str] | None = None,
        approaches: list[str] | None = None,
        model: str = "",
        memory_enabled: bool | None = None,
        label: str = "",
    ) -> str:
        benchmarks, strategies = self._resolve(benchmark_ids, approaches or [])
        suite_id = new_id("suite")
        with self._lock:
            self._suites[suite_id] = SuiteProgress(
                suite_id=suite_id,
                label=label or "blocking suite",
                total=len(benchmarks) * len(strategies),
            )
        self._run_suite(suite_id, benchmarks, strategies, model, memory_enabled)
        return suite_id

    def active_suites(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return {
                suite_id: progress.to_dict()
                for suite_id, progress in self._suites.items()
                if not progress.finished
            }

    def suite_progress(self, suite_id: str) -> dict[str, Any] | None:
        with self._lock:
            progress = self._suites.get(suite_id)
        return progress.to_dict() if progress else None

    # -- execution -------------------------------------------------------
    def _resolve(
        self, benchmark_ids: list[str] | None, chosen_strategies: list[str]
    ) -> tuple[list[str], list[str]]:
        available = [manifest.id for manifest in list_benchmarks()]
        benchmarks = [b for b in (benchmark_ids or available) if b in available]
        if not benchmarks:
            raise ValueError(
                f"no known benchmarks selected (available: {', '.join(available) or 'none'})"
            )
        strategies = [s for s in (chosen_strategies or strategy_ids()) if s in BY_ID]
        if not strategies:
            raise ValueError("no known strategies selected")
        # Always evaluate in baseline order so tables read A -> E.
        strategies.sort(key=lambda s: BY_ID[s].baseline)
        return benchmarks, strategies

    def _publish(self, suite_id: str, payload: dict[str, Any]) -> None:
        self.bus.publish(suite_id, {"suite_id": suite_id, **payload})

    def _run_suite(
        self,
        suite_id: str,
        benchmark_ids: list[str],
        approaches: list[str],
        model: str,
        memory_enabled: bool | None,
    ) -> None:
        progress = self._suites[suite_id]
        self._publish(
            suite_id,
            {
                "type": "suite_started",
                "benchmarks": benchmark_ids,
                "strategies": approaches,
                "total": progress.total,
            },
        )

        try:
            for benchmark_id in benchmark_ids:
                for strategy_id in approaches:
                    progress.current = f"{benchmark_id} / {strategy_id}"
                    self._publish(
                        suite_id,
                        {
                            "type": "case_started",
                            "benchmark_id": benchmark_id,
                            "strategy": strategy_id,
                            "completed": progress.completed,
                            "total": progress.total,
                        },
                    )
                    try:
                        result = self.evaluate_case(
                            suite_id=suite_id,
                            benchmark_id=benchmark_id,
                            strategy_id=strategy_id,
                            model=model,
                            memory_enabled=memory_enabled,
                        )
                    except Exception as exc:  # noqa: BLE001 - one bad case must not kill the suite
                        message = f"{benchmark_id}/{strategy_id}: {type(exc).__name__}: {exc}"
                        progress.errors.append(message)
                        result = {"error": message}
                    progress.completed += 1
                    self._publish(
                        suite_id,
                        {
                            "type": "case_finished",
                            "benchmark_id": benchmark_id,
                            "strategy": strategy_id,
                            "completed": progress.completed,
                            "total": progress.total,
                            "result": result,
                        },
                    )
        finally:
            progress.finished = True
            progress.current = ""
            self._publish(
                suite_id,
                {
                    "type": "suite_finished",
                    "completed": progress.completed,
                    "total": progress.total,
                    "errors": progress.errors,
                },
            )
            self.bus.mark_finished(suite_id)

    def evaluate_case(
        self,
        suite_id: str,
        benchmark_id: str,
        strategy_id: str,
        model: str = "",
        memory_enabled: bool | None = None,
    ) -> dict[str, Any]:
        """Execute one (benchmark, strategy) pair and score it."""
        manifest = get_benchmark(benchmark_id)
        if manifest is None:
            raise ValueError(f"unknown benchmark '{benchmark_id}'")

        request = RunRequest(
            issue=manifest.issue,
            repo=benchmark_id,
            issue_id=manifest.issue_id,
            benchmark_id=benchmark_id,
            model=model or self.settings.default_model,
            strategy=strategy_id,
            memory_enabled=memory_enabled,
            seed_files=manifest.baseline_seed_files,
            # The harness is the human in the loop for an automated suite; no
            # remote repository is touched either way.
            auto_approve=True,
        )
        handle = self.orchestrator.run_blocking(request)
        state = handle.state or {}
        budget = state.get("metrics") or {}

        record = {
            "suite_id": suite_id,
            "benchmark_id": benchmark_id,
            "strategy": strategy_id,
            "baseline": BY_ID[strategy_id].baseline,
            "strategy_label": BY_ID[strategy_id].label,
            "model": handle.request.model,
            "run_id": handle.run_id,
            "status": handle.status,
            "error": handle.error,
            "retrieval": retrieval_metrics(
                retrieved_paths_from_state(state), manifest
            ).to_dict(),
            "repair": repair_metrics(state, manifest),
            "agent": agent_metrics(state, budget),
            "safety": safety_metrics(state, manifest),
            "trajectory": self._trajectory(state),
        }
        self.store.save_eval_result(
            suite_id=suite_id,
            benchmark_id=benchmark_id,
            approach=strategy_id,
            model=record["model"],
            payload=record,
            run_id=handle.run_id,
        )
        # Workspaces are per-run copies; a suite would otherwise leave dozens.
        self.orchestrator.cleanup_workspace(handle.run_id)
        return record

    @staticmethod
    def _trajectory(state: dict[str, Any]) -> list[dict[str, Any]]:
        """A compact step list for the side-by-side trajectory comparison."""
        steps: list[dict[str, Any]] = []
        for event in state.get("timeline") or []:
            node = event.get("node", "")
            status = event.get("status", "")
            if node in {"tools", "checks", "patch", "reflect", "retrieve", "verifier"}:
                steps.append(
                    {
                        "node": node,
                        "title": event.get("title", ""),
                        "status": status,
                    }
                )
        return steps

    # -- aggregation -----------------------------------------------------
    def summarise(
        self, suite_id: str, results: list[dict[str, Any]] | None = None
    ) -> dict[str, Any]:
        """Aggregate a suite into the dashboard's comparison table."""
        rows = results if results is not None else self.store.get_eval_results(suite_id)
        payloads = [row["payload"] for row in rows]

        by_strategy: dict[str, list[dict[str, Any]]] = {}
        for payload in payloads:
            by_strategy.setdefault(payload.get("strategy", "unknown"), []).append(payload)

        comparison = []
        for strategy_id, cases in by_strategy.items():
            strategy = BY_ID.get(strategy_id)
            count = len(cases)
            comparison.append(
                {
                    "strategy": strategy_id,
                    "baseline": strategy.baseline if strategy else "?",
                    "label": strategy.label if strategy else strategy_id,
                    "cases": count,
                    "repair_rate": _rate(cases, lambda c: c["repair"]["verified"]),
                    "patch_applied_rate": _rate(cases, lambda c: c["repair"]["patch_applied"]),
                    "targeted_pass_rate": _rate(
                        cases, lambda c: c["repair"]["targeted_tests_passed"]
                    ),
                    "full_pass_rate": _rate(cases, lambda c: c["repair"]["full_tests_passed"]),
                    "regression_rate": _rate(
                        cases, lambda c: c["repair"]["regression_introduced"]
                    ),
                    "correct_file_rate": _rate(
                        cases, lambda c: c["repair"]["correct_file_targeted"]
                    ),
                    "recovery_rate": _rate(
                        cases, lambda c: c["agent"]["recovered_after_failure"]
                    ),
                    "avg_recall_at_k": _mean(cases, lambda c: c["retrieval"]["recall_at_k"]),
                    "avg_precision_at_k": _mean(
                        cases, lambda c: c["retrieval"]["precision_at_k"]
                    ),
                    "avg_mrr": _mean(cases, lambda c: c["retrieval"]["mrr"]),
                    "avg_gold_recall": _mean(
                        cases, lambda c: c["retrieval"]["gold_file_recall"]
                    ),
                    "avg_tool_calls": _mean(cases, lambda c: c["agent"]["tool_calls"]),
                    "avg_unnecessary_tool_calls": _mean(
                        cases, lambda c: c["agent"]["unnecessary_tool_calls"]
                    ),
                    "avg_llm_calls": _mean(cases, lambda c: c["agent"]["llm_calls"]),
                    "avg_retries": _mean(cases, lambda c: c["agent"]["retries"]),
                    "avg_latency_ms": _mean(cases, lambda c: c["agent"]["latency_ms"]),
                    "avg_tokens": _mean(cases, lambda c: c["agent"]["total_tokens"]),
                    "avg_cost_usd": _mean(cases, lambda c: c["agent"]["cost_usd"], digits=6),
                    "blocked_commands": sum(
                        c["safety"]["blocked_commands"] for c in cases
                    ),
                    "injections_detected": sum(
                        c["safety"]["prompt_injections_detected"] for c in cases
                    ),
                }
            )
        comparison.sort(key=lambda row: row["baseline"])

        return {
            "suite_id": suite_id,
            "progress": self.suite_progress(suite_id),
            "comparison": comparison,
            "cases": payloads,
            "benchmarks": sorted({p.get("benchmark_id", "") for p in payloads}),
        }


def _rate(cases: list[dict[str, Any]], predicate) -> float:
    if not cases:
        return 0.0
    hits = 0
    for case in cases:
        try:
            hits += 1 if predicate(case) else 0
        except (KeyError, TypeError):
            continue
    return round(hits / len(cases), 4)


def _mean(cases: list[dict[str, Any]], selector, digits: int = 2) -> float:
    values = []
    for case in cases:
        try:
            value = selector(case)
        except (KeyError, TypeError):
            continue
        if isinstance(value, int | float):
            values.append(value)
    if not values:
        return 0.0
    return round(sum(values) / len(values), digits)
