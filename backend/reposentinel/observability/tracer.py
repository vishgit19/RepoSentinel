"""Run tracing.

Produces an OpenTelemetry-shaped span tree per run:

    run
     |- node:triage
     |   `- llm:triage
     |- node:retrieve
     |   `- retrieval:hybrid
     |- node:tools
     |   |- llm:tool_selection
     |   `- tool:search_symbols
     ...

Spans carry latency, model, tokens, cost, input/output sizes and errors.
:class:`RunTracer` is standalone (no vendor SDK required) but
``export_langfuse_batch`` renders the same tree in Langfuse's ingestion shape,
so pointing this at Langfuse or an OTLP collector is a serialisation change
rather than a re-instrumentation.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from reposentinel.models.schemas import TraceSpan


@dataclass
class RunTracer:
    run_id: str
    spans: list[TraceSpan] = field(default_factory=list)
    _stack: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        root = TraceSpan(run_id=self.run_id, name="run", kind="run", parent_id=None)
        self.spans.append(root)
        self.root_id = root.span_id
        self._stack.append(root.span_id)

    @property
    def current_parent(self) -> str | None:
        return self._stack[-1] if self._stack else None

    @contextlib.contextmanager
    def span(
        self,
        name: str,
        kind: str = "node",
        attributes: dict[str, Any] | None = None,
    ) -> Iterator[TraceSpan]:
        span = TraceSpan(
            run_id=self.run_id,
            parent_id=self.current_parent,
            name=name,
            kind=kind,  # type: ignore[arg-type]
            attributes=dict(attributes or {}),
        )
        self.spans.append(span)
        self._stack.append(span.span_id)
        started = time.perf_counter()
        try:
            yield span
        except Exception as exc:
            span.ok = False
            span.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            span.duration_ms = int((time.perf_counter() - started) * 1000)
            span.ended_at = time.time()
            self._stack.pop()

    def finish(self, ok: bool = True, error: str | None = None) -> None:
        root = self.spans[0]
        root.ok = ok
        root.error = error
        root.ended_at = time.time()
        root.duration_ms = int((root.ended_at - root.started_at) * 1000)

    # -- reporting -------------------------------------------------------
    def totals(self) -> dict[str, Any]:
        by_kind: dict[str, int] = {}
        tokens = cost = 0
        for span in self.spans:
            by_kind[span.kind] = by_kind.get(span.kind, 0) + 1
            tokens += int(span.attributes.get("total_tokens", 0) or 0)
            cost += float(span.attributes.get("cost_usd", 0.0) or 0.0)
        return {
            "spans": len(self.spans),
            "by_kind": by_kind,
            "tokens": tokens,
            "cost_usd": round(cost, 6),
            "duration_ms": self.spans[0].duration_ms,
            "errors": sum(1 for s in self.spans if not s.ok),
        }

    def tree(self) -> list[dict[str, Any]]:
        """Flattened depth-annotated tree for the UI."""
        children: dict[str | None, list[TraceSpan]] = {}
        for span in self.spans:
            children.setdefault(span.parent_id, []).append(span)

        rows: list[dict[str, Any]] = []

        def walk(span: TraceSpan, depth: int) -> None:
            rows.append(
                {
                    "span_id": span.span_id,
                    "parent_id": span.parent_id,
                    "depth": depth,
                    "name": span.name,
                    "kind": span.kind,
                    "duration_ms": span.duration_ms,
                    "ok": span.ok,
                    "error": span.error,
                    "attributes": span.attributes,
                }
            )
            for child in children.get(span.span_id, []):
                walk(child, depth + 1)

        if self.spans:
            walk(self.spans[0], 0)
        return rows

    def export_langfuse_batch(self) -> list[dict[str, Any]]:
        """Render the span tree as a Langfuse ingestion batch."""
        batch: list[dict[str, Any]] = []
        for span in self.spans:
            if span.kind == "run":
                batch.append(
                    {
                        "type": "trace-create",
                        "body": {
                            "id": self.run_id,
                            "name": "reposentinel-run",
                            "metadata": span.attributes,
                        },
                    }
                )
                continue
            body = {
                "id": span.span_id,
                "traceId": self.run_id,
                "parentObservationId": (
                    None if span.parent_id == self.root_id else span.parent_id
                ),
                "name": span.name,
                "startTime": span.started_at,
                "endTime": span.ended_at,
                "metadata": span.attributes,
                "level": "DEFAULT" if span.ok else "ERROR",
                "statusMessage": span.error,
            }
            if span.kind == "llm":
                body |= {
                    "model": span.attributes.get("model"),
                    "usage": {
                        "promptTokens": span.attributes.get("prompt_tokens", 0),
                        "completionTokens": span.attributes.get("completion_tokens", 0),
                        "totalCost": span.attributes.get("cost_usd", 0.0),
                    },
                }
                batch.append({"type": "generation-create", "body": body})
            else:
                batch.append({"type": "span-create", "body": body})
        return batch
