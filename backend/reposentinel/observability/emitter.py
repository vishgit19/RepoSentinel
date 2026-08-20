"""Builds the user-visible execution timeline.

The timeline is a summary channel, not a reasoning dump. Nodes emit a title, a
status, a short decision summary, the tool that ran, the evidence retrieved and
the cost so far. Raw model reasoning is never published: structured outputs
carry an explicit ``reasoning_summary``-style field written for humans, and
that is the only narrative that reaches the UI.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

from reposentinel.models.schemas import (
    StepStatus,
    TimelineEvent,
    ToolCallRecord,
)
from reposentinel.observability.events import EventBus


@dataclass
class TimelineEmitter:
    run_id: str
    bus: EventBus
    budget: object = None  # Budget; read for token/cost stamps
    events: list[TimelineEvent] = field(default_factory=list)
    _seq: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def _next_seq(self) -> int:
        with self._lock:
            self._seq += 1
            return self._seq

    def emit(
        self,
        node: str,
        title: str,
        status: StepStatus = StepStatus.SUCCESS,
        detail: str = "",
        lines: list[str] | None = None,
        tool_call: ToolCallRecord | None = None,
        evidence: list[str] | None = None,
        duration_ms: int = 0,
    ) -> TimelineEvent:
        budget = self.budget
        event = TimelineEvent(
            run_id=self.run_id,
            seq=self._next_seq(),
            node=node,
            title=title,
            status=status,
            detail=detail,
            lines=lines or [],
            tool_call=tool_call,
            evidence=evidence or [],
            duration_ms=duration_ms,
            run_elapsed_ms=getattr(budget, "elapsed_ms", 0) if budget else 0,
            tokens=getattr(budget, "total_tokens", 0) if budget else 0,
            cost_usd=round(getattr(budget, "cost_usd", 0.0), 6) if budget else 0.0,
        )
        self.events.append(event)
        self.bus.publish(
            self.run_id,
            {"type": "timeline", "run_id": self.run_id, "event": event.model_dump(mode="json")},
        )
        # Piggyback the running budget so the UI's metrics view is live rather
        # than only populated at the end. The snapshot is a handful of counters.
        if budget is not None and hasattr(budget, "snapshot"):
            self.metrics(budget.snapshot())
        return event

    def node_started(self, node: str, title: str, detail: str = "") -> TimelineEvent:
        return self.emit(node, title, status=StepStatus.RUNNING, detail=detail)

    def step_status(self, node: str, status: StepStatus) -> None:
        """Update the fixed step rail without adding a timeline entry."""
        self.bus.publish(
            self.run_id,
            {
                "type": "step",
                "run_id": self.run_id,
                "node": node,
                "status": status.value,
            },
        )

    def metrics(self, payload: dict) -> None:
        self.bus.publish(
            self.run_id, {"type": "metrics", "run_id": self.run_id, "metrics": payload}
        )

    def state_patch(self, payload: dict) -> None:
        """Push incremental run data (diff, reports, files) to the UI."""
        self.bus.publish(
            self.run_id, {"type": "state", "run_id": self.run_id, "data": payload}
        )

    def status(self, status: str, detail: str = "") -> None:
        self.bus.publish(
            self.run_id,
            {"type": "status", "run_id": self.run_id, "status": status, "detail": detail},
        )
