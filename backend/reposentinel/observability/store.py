"""Run persistence.

Every run, timeline event and trace span is written to SQLite so a finished run
can be reopened and replayed, and so the evaluation dashboard can aggregate
across runs. Queryable fields are real columns; the full state snapshot is kept
as JSON alongside them.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from reposentinel.config import Settings, get_settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id          TEXT PRIMARY KEY,
    created_at      REAL NOT NULL,
    updated_at      REAL NOT NULL,
    status          TEXT NOT NULL,
    issue           TEXT NOT NULL,
    issue_id        TEXT NOT NULL DEFAULT '',
    repo            TEXT NOT NULL,
    benchmark_id    TEXT NOT NULL DEFAULT '',
    model           TEXT NOT NULL DEFAULT '',
    provider        TEXT NOT NULL DEFAULT '',
    strategy        TEXT NOT NULL DEFAULT 'agentic',
    memory_enabled  INTEGER NOT NULL DEFAULT 1,
    verified        INTEGER NOT NULL DEFAULT 0,
    approved        INTEGER,
    retries         INTEGER NOT NULL DEFAULT 0,
    tool_calls      INTEGER NOT NULL DEFAULT 0,
    llm_calls       INTEGER NOT NULL DEFAULT 0,
    total_tokens    INTEGER NOT NULL DEFAULT 0,
    cost_usd        REAL NOT NULL DEFAULT 0,
    latency_ms      INTEGER NOT NULL DEFAULT 0,
    tests_passed    INTEGER NOT NULL DEFAULT 0,
    tests_failed    INTEGER NOT NULL DEFAULT 0,
    security_ok     INTEGER NOT NULL DEFAULT 1,
    failure_reason  TEXT NOT NULL DEFAULT '',
    diff            TEXT NOT NULL DEFAULT '',
    state_json      TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS run_events (
    run_id   TEXT NOT NULL,
    seq      INTEGER NOT NULL,
    payload  TEXT NOT NULL,
    PRIMARY KEY (run_id, seq)
);

CREATE TABLE IF NOT EXISTS run_spans (
    run_id   TEXT NOT NULL,
    span_id  TEXT NOT NULL,
    payload  TEXT NOT NULL,
    PRIMARY KEY (run_id, span_id)
);

CREATE TABLE IF NOT EXISTS eval_results (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    suite_id      TEXT NOT NULL,
    created_at    REAL NOT NULL,
    benchmark_id  TEXT NOT NULL,
    approach      TEXT NOT NULL,
    model         TEXT NOT NULL,
    run_id        TEXT NOT NULL DEFAULT '',
    payload       TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_runs_created ON runs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_runs_benchmark ON runs(benchmark_id);
CREATE INDEX IF NOT EXISTS idx_eval_suite ON eval_results(suite_id);
"""


class RunStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._connect() as connection:
            connection.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        return connection

    # -- runs ------------------------------------------------------------
    def create_run(self, run_id: str, **fields: Any) -> None:
        now = time.time()
        payload = {
            "run_id": run_id,
            "created_at": now,
            "updated_at": now,
            "status": fields.get("status", "queued"),
            "issue": fields.get("issue", ""),
            "issue_id": fields.get("issue_id", ""),
            "repo": fields.get("repo", ""),
            "benchmark_id": fields.get("benchmark_id", ""),
            "model": fields.get("model", ""),
            "provider": fields.get("provider", ""),
            "strategy": fields.get("strategy", "agentic"),
            "memory_enabled": int(bool(fields.get("memory_enabled", True))),
        }
        columns = ", ".join(payload)
        placeholders = ", ".join("?" * len(payload))
        with self._lock, self._connect() as connection:
            connection.execute(
                f"INSERT OR REPLACE INTO runs ({columns}) VALUES ({placeholders})",
                list(payload.values()),
            )

    def update_run(self, run_id: str, **fields: Any) -> None:
        if not fields:
            return
        fields["updated_at"] = time.time()
        assignments = ", ".join(f"{key} = ?" for key in fields)
        with self._lock, self._connect() as connection:
            connection.execute(
                f"UPDATE runs SET {assignments} WHERE run_id = ?",
                [*fields.values(), run_id],
            )

    def save_state(self, run_id: str, state: dict[str, Any]) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                "UPDATE runs SET state_json = ?, updated_at = ? WHERE run_id = ?",
                (json.dumps(state, default=str), time.time(), run_id),
            )

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            return None
        record = dict(row)
        record["state"] = json.loads(record.pop("state_json") or "{}")
        return record

    def list_runs(self, limit: int = 50, benchmark_id: str | None = None) -> list[dict[str, Any]]:
        query = (
            "SELECT run_id, created_at, updated_at, status, issue, issue_id, repo, benchmark_id, "
            "model, provider, strategy, memory_enabled, verified, approved, retries, tool_calls, "
            "llm_calls, total_tokens, cost_usd, latency_ms, tests_passed, tests_failed, "
            "security_ok, failure_reason FROM runs"
        )
        parameters: list[Any] = []
        if benchmark_id:
            query += " WHERE benchmark_id = ?"
            parameters.append(benchmark_id)
        query += " ORDER BY created_at DESC LIMIT ?"
        parameters.append(limit)
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [dict(row) for row in rows]

    def delete_run(self, run_id: str) -> None:
        with self._lock, self._connect() as connection:
            connection.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
            connection.execute("DELETE FROM run_events WHERE run_id = ?", (run_id,))
            connection.execute("DELETE FROM run_spans WHERE run_id = ?", (run_id,))

    # -- timeline / spans -------------------------------------------------
    def append_events(self, run_id: str, events: list[dict[str, Any]]) -> None:
        if not events:
            return
        with self._lock, self._connect() as connection:
            connection.executemany(
                "INSERT OR REPLACE INTO run_events (run_id, seq, payload) VALUES (?, ?, ?)",
                [
                    (run_id, int(event.get("seq", index)), json.dumps(event, default=str))
                    for index, event in enumerate(events)
                ],
            )

    def get_events(self, run_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT payload FROM run_events WHERE run_id = ? ORDER BY seq", (run_id,)
            ).fetchall()
        return [json.loads(row["payload"]) for row in rows]

    def save_spans(self, run_id: str, spans: list[dict[str, Any]]) -> None:
        if not spans:
            return
        with self._lock, self._connect() as connection:
            connection.executemany(
                "INSERT OR REPLACE INTO run_spans (run_id, span_id, payload) VALUES (?, ?, ?)",
                [
                    (run_id, str(span.get("span_id", index)), json.dumps(span, default=str))
                    for index, span in enumerate(spans)
                ],
            )

    def get_spans(self, run_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT payload FROM run_spans WHERE run_id = ?", (run_id,)
            ).fetchall()
        return [json.loads(row["payload"]) for row in rows]

    # -- evaluation ------------------------------------------------------
    def save_eval_result(
        self,
        suite_id: str,
        benchmark_id: str,
        approach: str,
        model: str,
        payload: dict[str, Any],
        run_id: str = "",
    ) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                "INSERT INTO eval_results (suite_id, created_at, benchmark_id, approach, model, "
                "run_id, payload) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    suite_id,
                    time.time(),
                    benchmark_id,
                    approach,
                    model,
                    run_id,
                    json.dumps(payload, default=str),
                ),
            )

    def get_eval_results(self, suite_id: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT suite_id, created_at, benchmark_id, approach, model, run_id, payload FROM eval_results"
        parameters: list[Any] = []
        if suite_id:
            query += " WHERE suite_id = ?"
            parameters.append(suite_id)
        query += " ORDER BY created_at DESC"
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        results = []
        for row in rows:
            record = dict(row)
            record["payload"] = json.loads(record["payload"])
            results.append(record)
        return results

    def list_eval_suites(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT suite_id, COUNT(*) AS results, MIN(created_at) AS created_at "
                "FROM eval_results GROUP BY suite_id ORDER BY created_at DESC"
            ).fetchall()
        return [dict(row) for row in rows]


_store: RunStore | None = None
_store_lock = threading.Lock()


def get_store(settings: Settings | None = None) -> RunStore:
    global _store
    settings = settings or get_settings()
    with _store_lock:
        if _store is None:
            _store = RunStore(settings.db_path)
    return _store
