"""Evaluation metrics are derived from recorded evidence, not from model claims."""

from __future__ import annotations

from pytest import approx
from reposentinel.benchmarks import get_benchmark, reload_benchmarks
from reposentinel.evaluation.metrics import (
    agent_metrics,
    repair_metrics,
    retrieval_metrics,
    retrieved_paths_from_state,
    safety_metrics,
)

reload_benchmarks()


def test_retrieval_metrics_are_rank_aware():
    manifest = get_benchmark("logic_bug")
    assert manifest is not None
    perfect = retrieval_metrics(
        ["app/auth/token.py", "app/auth/middleware.py", "tests/test_auth.py"],
        manifest,
        k=3,
    )
    assert perfect.mrr == 1.0
    assert perfect.gold_file_recall == 1.0
    assert perfect.recall_at_k == 1.0

    late = retrieval_metrics(
        ["app/api.py", "app/users/repository.py", "app/auth/token.py"],
        manifest,
        k=3,
    )
    assert late.mrr == approx(1 / 3)
    assert late.first_relevant_rank == 3


def test_repair_metrics_require_an_applied_patch_and_passing_tests():
    manifest = get_benchmark("logic_bug")
    assert manifest is not None
    state = {
        "patches": [
            {
                "applied": True,
                "files_changed": ["app/auth/token.py"],
                "diff": "--- a/app/auth/token.py\n+++ b/app/auth/token.py\n",
            }
        ],
        "test_results": [
            {"scope": "targeted", "passed": 3, "failed": 0, "errors": 0, "skipped": 0},
            {"scope": "full", "passed": 13, "failed": 0, "errors": 0, "skipped": 0},
        ],
        "verification": {"verified": True},
    }
    repair = repair_metrics(state, manifest)
    assert repair["patch_applied"] is True
    assert repair["targeted_tests_passed"] is True
    assert repair["full_tests_passed"] is True
    assert repair["correct_file_targeted"] is True
    assert repair["regression_introduced"] is False


def test_a_failing_full_suite_is_a_regression():
    manifest = get_benchmark("logic_bug")
    assert manifest is not None
    repair = repair_metrics(
        {
            "patches": [{"applied": True, "files_changed": ["app/auth/token.py"]}],
            "test_results": [
                {"scope": "full", "passed": 10, "failed": 3, "errors": 0, "skipped": 0}
            ],
            "verification": {"verified": False},
        },
        manifest,
    )
    assert repair["regression_introduced"] is True
    assert repair["full_tests_passed"] is False


def test_agent_metrics_treat_a_reproduced_failure_as_useful():
    metrics = agent_metrics(
        {
            "status": "approved",
            "tool_history": [
                {"executed": True, "ok": False, "name": "run_targeted_tests"},
                {"executed": False, "ok": False, "name": "unknown"},
            ],
            "patches": [{}, {}],
            "test_results": [
                {"scope": "targeted", "passed": 0, "failed": 3, "errors": 0},
                {"scope": "targeted", "passed": 3, "failed": 0, "errors": 0},
            ],
        },
        {"tool_calls": 8, "retries": 1, "latency_ms": 40000, "total_tokens": 12, "cost_usd": 0.01},
    )
    assert metrics["unnecessary_tool_calls"] == 1
    assert metrics["recovered_after_failure"] is True
    assert metrics["completed"] is True


def test_safety_metrics_only_score_injection_resistance_when_it_was_planted():
    injected = get_benchmark("injection")
    ordinary = get_benchmark("logic_bug")
    assert injected is not None and ordinary is not None
    state = {
        "safety_events": [
            {"kind": "prompt_injection", "detail": "override_instructions"},
            {"kind": "blocked_command", "detail": "rm"},
        ]
    }
    planted = safety_metrics(state, injected)
    assert planted["injection_resisted"] is True
    assert planted["blocked_commands"] == 1

    elsewhere = safety_metrics(state, ordinary)
    assert elsewhere["injection_resisted"] is None
    assert elsewhere["injection_expected"] is False


def test_retrieved_paths_preserve_first_seen_order():
    paths = retrieved_paths_from_state(
        {
            "retrieved_context": [
                {"path": "a.py"},
                {"path": "b.py"},
                {"path": "a.py"},
            ]
        }
    )
    assert paths == ["a.py", "b.py", "a.py"]
