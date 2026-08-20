"""Repair memory.

Stores what actually happened in previous repairs - issue type, files touched,
which tools helped, which approaches failed, and whether verification passed -
and retrieves the most similar prior repairs before a new run starts.

Retrieval is embedding-based over a compact text summary of each record, with a
category bonus so a security issue prefers prior security repairs. Memory is a
toggle (``memory_enabled``) precisely so the evaluation harness can measure
"agent with repair memory" against "agent without".

Nothing here is conversational: there is no chat history, only outcomes that
change future behaviour.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from reposentinel.retrieval.embeddings import EmbeddingBackend
from reposentinel.retrieval.vector_store import pack, unpack

SCHEMA = """
CREATE TABLE IF NOT EXISTS repair_memory (
    memory_id       TEXT PRIMARY KEY,
    created_at      REAL NOT NULL,
    run_id          TEXT NOT NULL DEFAULT '',
    repo            TEXT NOT NULL DEFAULT '',
    benchmark_id    TEXT NOT NULL DEFAULT '',
    issue           TEXT NOT NULL,
    issue_kind      TEXT NOT NULL DEFAULT 'unknown',
    root_cause      TEXT NOT NULL DEFAULT '',
    files_involved  TEXT NOT NULL DEFAULT '[]',
    successful_tools TEXT NOT NULL DEFAULT '[]',
    failed_approaches TEXT NOT NULL DEFAULT '[]',
    patch_summary   TEXT NOT NULL DEFAULT '',
    patch_diff      TEXT NOT NULL DEFAULT '',
    verified        INTEGER NOT NULL DEFAULT 0,
    attempts        INTEGER NOT NULL DEFAULT 1,
    lesson          TEXT NOT NULL DEFAULT '',
    dimensions      INTEGER NOT NULL DEFAULT 0,
    embedding       BLOB
);
CREATE INDEX IF NOT EXISTS idx_memory_kind ON repair_memory(issue_kind);
"""


@dataclass
class MemoryRecord:
    memory_id: str
    issue: str
    issue_kind: str = "unknown"
    root_cause: str = ""
    files_involved: list[str] = field(default_factory=list)
    successful_tools: list[str] = field(default_factory=list)
    failed_approaches: list[str] = field(default_factory=list)
    patch_summary: str = ""
    patch_diff: str = ""
    verified: bool = False
    attempts: int = 1
    lesson: str = ""
    repo: str = ""
    benchmark_id: str = ""
    run_id: str = ""
    created_at: float = field(default_factory=time.time)
    similarity: float = 0.0

    def to_text(self) -> str:
        """The embedded representation."""
        return "\n".join(
            [
                f"issue: {self.issue}",
                f"kind: {self.issue_kind}",
                f"root cause: {self.root_cause}",
                f"files: {', '.join(self.files_involved)}",
                f"fix: {self.patch_summary}",
                f"lesson: {self.lesson}",
            ]
        )

    def as_prompt_block(self) -> str:
        """How a hit is shown to the model - outcomes, not instructions."""
        lines = [
            f"Similar past repair (similarity {self.similarity:.2f}, "
            f"{'verified' if self.verified else 'NOT verified'}, {self.attempts} attempt(s)):",
            f"  issue: {self.issue[:200]}",
            f"  kind: {self.issue_kind}",
        ]
        if self.root_cause:
            lines.append(f"  root cause found: {self.root_cause[:240]}")
        if self.files_involved:
            lines.append(f"  files that mattered: {', '.join(self.files_involved[:6])}")
        if self.successful_tools:
            lines.append(f"  tools that helped: {', '.join(self.successful_tools[:6])}")
        if self.failed_approaches:
            lines.append(f"  approaches that FAILED: {'; '.join(self.failed_approaches[:3])}")
        if self.lesson:
            lines.append(f"  lesson: {self.lesson}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "issue": self.issue,
            "issue_kind": self.issue_kind,
            "root_cause": self.root_cause,
            "files_involved": self.files_involved,
            "successful_tools": self.successful_tools,
            "failed_approaches": self.failed_approaches,
            "patch_summary": self.patch_summary,
            "verified": self.verified,
            "attempts": self.attempts,
            "lesson": self.lesson,
            "benchmark_id": self.benchmark_id,
            "run_id": self.run_id,
            "similarity": round(self.similarity, 4),
            "created_at": self.created_at,
        }


class RepairMemory:
    def __init__(self, db_path: Path, embeddings: EmbeddingBackend) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.embeddings = embeddings
        self._lock = threading.Lock()
        with self._connect() as connection:
            connection.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        return connection

    # -- writing ---------------------------------------------------------
    def remember(self, record: MemoryRecord) -> MemoryRecord:
        vector = self.embeddings.embed_one(record.to_text())
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO repair_memory (
                    memory_id, created_at, run_id, repo, benchmark_id, issue, issue_kind,
                    root_cause, files_involved, successful_tools, failed_approaches,
                    patch_summary, patch_diff, verified, attempts, lesson, dimensions, embedding
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.memory_id,
                    record.created_at,
                    record.run_id,
                    record.repo,
                    record.benchmark_id,
                    record.issue,
                    record.issue_kind,
                    record.root_cause,
                    json.dumps(record.files_involved),
                    json.dumps(record.successful_tools),
                    json.dumps(record.failed_approaches),
                    record.patch_summary,
                    record.patch_diff[:20_000],
                    int(record.verified),
                    record.attempts,
                    record.lesson,
                    int(vector.shape[0]),
                    pack(vector),
                ),
            )
        return record

    # -- reading ---------------------------------------------------------
    def _row_to_record(self, row: sqlite3.Row) -> MemoryRecord:
        return MemoryRecord(
            memory_id=row["memory_id"],
            issue=row["issue"],
            issue_kind=row["issue_kind"],
            root_cause=row["root_cause"],
            files_involved=json.loads(row["files_involved"] or "[]"),
            successful_tools=json.loads(row["successful_tools"] or "[]"),
            failed_approaches=json.loads(row["failed_approaches"] or "[]"),
            patch_summary=row["patch_summary"],
            patch_diff=row["patch_diff"],
            verified=bool(row["verified"]),
            attempts=row["attempts"],
            lesson=row["lesson"],
            repo=row["repo"],
            benchmark_id=row["benchmark_id"],
            run_id=row["run_id"],
            created_at=row["created_at"],
        )

    def recall(
        self,
        issue: str,
        issue_kind: str = "",
        top_k: int = 3,
        exclude_run_id: str = "",
        min_similarity: float = 0.35,
    ) -> list[MemoryRecord]:
        """Most similar prior repairs, best first."""
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM repair_memory").fetchall()
        if not rows:
            return []

        query_vector = self.embeddings.embed_one(issue)
        dimensions = int(query_vector.shape[0])
        scored: list[MemoryRecord] = []

        for row in rows:
            if exclude_run_id and row["run_id"] == exclude_run_id:
                continue
            if row["embedding"] is None or row["dimensions"] != dimensions:
                continue
            vector = unpack(row["embedding"], row["dimensions"])
            denominator = float(np.linalg.norm(vector) * np.linalg.norm(query_vector)) or 1.0
            similarity = float(np.dot(vector, query_vector) / denominator)
            # A matching category is genuine evidence of relevance.
            if issue_kind and row["issue_kind"] == issue_kind:
                similarity += 0.08
            record = self._row_to_record(row)
            record.similarity = similarity
            scored.append(record)

        scored.sort(key=lambda r: -r.similarity)
        return [r for r in scored if r.similarity >= min_similarity][:top_k]

    def all_records(self, limit: int = 100) -> list[MemoryRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM repair_memory ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def count(self) -> int:
        with self._connect() as connection:
            row = connection.execute("SELECT COUNT(*) AS n FROM repair_memory").fetchone()
        return int(row["n"]) if row else 0

    def clear(self) -> None:
        with self._lock, self._connect() as connection:
            connection.execute("DELETE FROM repair_memory")
