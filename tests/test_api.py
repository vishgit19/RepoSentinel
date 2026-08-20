"""The HTTP API, exercised through FastAPI's test client.

Run execution itself needs a model, so these tests cover the parts that do not:
capability discovery, validation, persistence, the approval endpoint's contract
and the SSE framing. End-to-end agent behaviour is covered by the graph tests
and by ``scripts/run_agent.py``.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from reposentinel.api.app import create_app
from reposentinel.observability.events import bus
from reposentinel.observability.store import get_store


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app()) as test_client:
        yield test_client


class TestCapabilityDiscovery:
    def test_health_reports_real_backends(self, client):
        payload = client.get("/api/health").json()
        assert payload["status"] == "ok"
        # These must describe what is actually available, not a fixed string.
        assert payload["sandbox"]["active"] in {"local", "docker"}
        assert payload["sandbox"]["configured"] in {"auto", "local", "docker"}
        assert payload["security_backend"] in {"builtin", "semgrep"}
        assert payload["vector_store"] in {"sqlite", "pgvector"}
        assert payload["tools"] > 0
        assert payload["mcp_tools"] > 0
        assert "openai" in payload["providers"]
        assert payload["limits"]["max_repair_attempts"] >= 1

    def test_benchmarks_expose_ground_truth(self, client):
        payload = client.get("/api/benchmarks").json()
        assert payload, "no benchmarks were served"
        logic = next(item for item in payload if item["id"] == "logic_bug")
        assert logic["gold_files"] == ["app/auth/token.py"]
        assert logic["expected_failing_tests"]
        assert logic["issue"].strip()

    def test_strategies_cover_baselines_a_to_e(self, client):
        payload = client.get("/api/strategies").json()
        assert [item["baseline"] for item in payload] == ["A", "B", "C", "D", "E"]
        agentic = next(item for item in payload if item["id"] == "agentic")
        assert agentic["tools"] is True
        assert agentic["reflection"] is True
        # A baseline that could retry would erase the comparison.
        assert all(not item["reflection"] for item in payload if item["id"] != "agentic")

    def test_models_are_listed_with_availability(self, client):
        payload = client.get("/api/models").json()
        assert payload["default"]
        assert payload["models"]
        assert all("available" in model for model in payload["models"])

    def test_topology_matches_the_workflow(self, client):
        payload = client.get("/api/topology").json()
        node_ids = {node["id"] for node in payload["graph"]["nodes"]}
        assert {"triage", "planner", "patch", "checks", "reflect", "verifier"} <= node_ids
        # The retry edge is the point of the graph.
        assert {"from": "reflect", "to": "patch", "label": "retry"} in payload["graph"]["edges"]
        assert payload["steps"][0]["status"] == "pending"

    def test_tools_expose_json_schemas(self, client):
        payload = client.get("/api/tools").json()
        assert payload
        for tool in payload:
            assert tool["parameters"]["type"] == "object"
            assert tool["parameters"]["additionalProperties"] is False


class TestRunValidation:
    def test_issue_is_required(self, client):
        response = client.post("/api/runs", json={"issue": "", "repo": "logic_bug"})
        assert response.status_code == 422

    def test_unknown_repository_is_rejected(self, client):
        response = client.post(
            "/api/runs", json={"issue": "something broke", "repo": "not-a-real-repo"}
        )
        # Either the workspace refuses the source or the provider is missing;
        # both are client-side configuration problems, not server faults.
        assert response.status_code == 400
        assert response.json()["detail"]

    def test_unknown_run_is_404(self, client):
        assert client.get("/api/runs/run_does_not_exist").status_code == 404

    def test_approval_of_unknown_run_is_404(self, client):
        response = client.post(
            "/api/runs/run_does_not_exist/approval", json={"approved": True}
        )
        assert response.status_code == 404

    def test_zero_wait_timeout_is_not_a_decision(self):
        from reposentinel.graph.nodes.finalize import ApprovalGate

        gate = ApprovalGate()
        assert gate.wait(timeout=0) is None
        gate.decide(True, "late")
        assert gate.wait(timeout=0) is True

    def test_approval_unblocks_a_waiting_run(self, client):
        import threading
        import time

        from reposentinel.models.schemas import RunRequest
        from reposentinel.orchestrator import RunHandle, get_orchestrator

        orchestrator = get_orchestrator()
        run_id = "run_approval_wakeup"
        handle = RunHandle(
            run_id=run_id,
            request=RunRequest(issue="Expired tokens are accepted.", repo="logic_bug"),
        )
        handle.status = "awaiting_approval"
        orchestrator.runs[run_id] = handle

        released: list[bool | None] = []

        def waiter() -> None:
            released.append(handle.approval_gate.wait(timeout=8))

        thread = threading.Thread(target=waiter)
        thread.start()
        time.sleep(0.1)
        try:
            response = client.post(
                f"/api/runs/{run_id}/approval",
                json={"approved": True, "note": "looks right"},
            )
            assert response.status_code == 200, response.text
            thread.join(timeout=3)
            assert released == [True]
            replayed = [event for event in bus.replay(run_id) if event.get("type") == "status"]
            assert replayed
            assert replayed[-1]["status"] == "approved"
        finally:
            orchestrator.runs.pop(run_id, None)
            bus.forget(run_id)


class TestPersistedRuns:
    def test_recorded_run_is_listed_and_reopenable(self, client):
        store = get_store()
        run_id = "run_api_test_fixture"
        store.create_run(
            run_id,
            status="approved",
            issue="Expired tokens are accepted.",
            repo="logic_bug",
            benchmark_id="logic_bug",
            model="gpt-4.1-mini",
            strategy="agentic",
        )
        store.update_run(run_id, verified=1, tool_calls=8, tests_passed=13, cost_usd=0.02)
        store.append_events(
            run_id,
            [{"seq": 1, "node": "triage", "title": "Triage", "status": "success"}],
        )
        store.save_spans(run_id, [{"span_id": "s1", "kind": "run", "ok": True}])
        store.save_state(run_id, {"status": "approved", "metrics": {"tool_calls": 8}})
        try:
            listing = client.get("/api/runs?limit=100").json()
            assert any(row["run_id"] == run_id for row in listing)

            record = client.get(f"/api/runs/{run_id}").json()
            assert record["verified"] == 1
            assert record["events"][0]["node"] == "triage"
            assert record["spans"][0]["kind"] == "run"
            assert record["state"]["metrics"]["tool_calls"] == 8
            assert record["live"] is False
        finally:
            store.delete_run(run_id)

        assert client.get(f"/api/runs/{run_id}").status_code == 404


class TestEventStream:
    def test_stream_replays_buffered_events_as_sse_frames(self, client):
        run_id = "run_sse_fixture"
        bus.publish(run_id, {"type": "status", "run_id": run_id, "status": "running"})
        bus.publish(
            run_id,
            {
                "type": "timeline",
                "run_id": run_id,
                "event": {"seq": 1, "node": "triage", "title": "Triage"},
            },
        )
        bus.mark_finished(run_id)
        try:
            with client.stream("GET", f"/api/runs/{run_id}/events") as response:
                assert response.status_code == 200
                assert response.headers["content-type"].startswith("text/event-stream")
                body = "".join(response.iter_text())
        finally:
            bus.forget(run_id)

        # Each frame must be `event: <type>` followed by a JSON `data:` line.
        assert "event: status" in body
        assert "event: timeline" in body
        assert "event: stream_end" in body
        payloads = [
            json.loads(line[len("data: ") :])
            for line in body.splitlines()
            if line.startswith("data: ")
        ]
        assert any(item.get("type") == "timeline" for item in payloads)


class TestEvaluationEndpoints:
    def test_empty_selection_is_rejected(self, client):
        response = client.post(
            "/api/evaluations", json={"benchmark_ids": ["nope"], "strategies": ["agentic"]}
        )
        assert response.status_code == 400

    def test_unknown_suite_is_404(self, client):
        assert client.get("/api/evaluations/suite_missing").status_code == 404

    def test_suite_summary_aggregates_recorded_results(self, client):
        store = get_store()
        suite_id = "suite_api_test"
        for strategy, baseline, verified in (
            ("llm_only", "A", False),
            ("agentic", "E", True),
        ):
            store.save_eval_result(
                suite_id=suite_id,
                benchmark_id="logic_bug",
                approach=strategy,
                model="gpt-4.1-mini",
                payload={
                    "strategy": strategy,
                    "baseline": baseline,
                    "strategy_label": strategy,
                    "benchmark_id": "logic_bug",
                    "retrieval": {
                        "recall_at_k": 1.0 if verified else 0.0,
                        "precision_at_k": 0.5,
                        "mrr": 1.0 if verified else 0.0,
                        "gold_file_recall": 1.0 if verified else 0.0,
                    },
                    "repair": {
                        "verified": verified,
                        "patch_applied": True,
                        "targeted_tests_passed": verified,
                        "full_tests_passed": verified,
                        "regression_introduced": False,
                        "correct_file_targeted": verified,
                    },
                    "agent": {
                        "tool_calls": 10 if verified else 0,
                        "unnecessary_tool_calls": 0,
                        "llm_calls": 12,
                        "retries": 1 if verified else 0,
                        "recovered_after_failure": verified,
                        "latency_ms": 50000,
                        "total_tokens": 40000,
                        "cost_usd": 0.02,
                    },
                    "safety": {"blocked_commands": 0, "prompt_injections_detected": 0},
                    "trajectory": [{"node": "patch", "title": "Patch", "status": "success"}],
                },
            )

        payload = client.get(f"/api/evaluations/{suite_id}").json()
        rows = {row["strategy"]: row for row in payload["comparison"]}
        assert [row["baseline"] for row in payload["comparison"]] == ["A", "E"]
        assert rows["agentic"]["repair_rate"] == 1.0
        assert rows["llm_only"]["repair_rate"] == 0.0
        assert rows["agentic"]["recovery_rate"] == 1.0
        assert payload["benchmarks"] == ["logic_bug"]


class TestFrontend:
    def test_index_is_served(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "RepoSentinel" in response.text

    def test_vendored_react_is_served_locally(self, client):
        """The demo must not depend on a CDN at runtime."""
        for asset in (
            "/static/vendor/react.production.min.js",
            "/static/vendor/react-dom.production.min.js",
            "/static/vendor/htm.umd.js",
            "/static/app.js",
            "/static/styles.css",
        ):
            assert client.get(asset).status_code == 200, asset
