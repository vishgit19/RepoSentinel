"""Live event streaming from the graph to the browser.

The graph runs in a worker thread (its nodes make blocking subprocess and HTTP
calls) while the API serves SSE on the event loop. :class:`EventBus` bridges
the two with ``loop.call_soon_threadsafe``, and keeps a bounded replay buffer
so a client that connects mid-run still receives the whole timeline before the
live tail.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
from collections import defaultdict, deque
from collections.abc import AsyncIterator
from typing import Any

REPLAY_LIMIT = 1000


class EventBus:
    def __init__(self) -> None:
        self._subscribers: dict[str, list[asyncio.Queue]] = defaultdict(list)
        self._replay: dict[str, deque[dict[str, Any]]] = defaultdict(
            lambda: deque(maxlen=REPLAY_LIMIT)
        )
        self._finished: set[str] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._lock = threading.Lock()

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def publish(self, run_id: str, event: dict[str, Any]) -> None:
        """Publish from any thread."""
        with self._lock:
            self._replay[run_id].append(event)
            queues = list(self._subscribers.get(run_id, []))

        if not queues:
            return
        loop = self._loop
        for queue in queues:
            if loop is not None and not loop.is_closed():
                with contextlib.suppress(RuntimeError):
                    loop.call_soon_threadsafe(queue.put_nowait, event)
            else:  # pragma: no cover - only when no loop is bound (tests)
                with contextlib.suppress(asyncio.QueueFull):
                    queue.put_nowait(event)

    def mark_finished(self, run_id: str) -> None:
        with self._lock:
            self._finished.add(run_id)
        self.publish(run_id, {"type": "stream_end", "run_id": run_id})

    def is_finished(self, run_id: str) -> bool:
        with self._lock:
            return run_id in self._finished

    def replay(self, run_id: str) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._replay.get(run_id, []))

    async def subscribe(self, run_id: str) -> AsyncIterator[dict[str, Any]]:
        """Yield the replay buffer, then live events until the run ends."""
        queue: asyncio.Queue = asyncio.Queue()
        with self._lock:
            backlog = list(self._replay.get(run_id, []))
            already_finished = run_id in self._finished
            self._subscribers[run_id].append(queue)

        try:
            for event in backlog:
                yield event
            if already_finished:
                return

            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=20.0)
                except TimeoutError:
                    # Keep-alive so proxies do not drop an idle connection.
                    yield {"type": "heartbeat", "run_id": run_id}
                    if self.is_finished(run_id):
                        return
                    continue
                yield event
                if event.get("type") == "stream_end":
                    return
        finally:
            with self._lock:
                subscribers = self._subscribers.get(run_id, [])
                if queue in subscribers:
                    subscribers.remove(queue)

    def forget(self, run_id: str) -> None:
        with self._lock:
            self._replay.pop(run_id, None)
            self._subscribers.pop(run_id, None)
            self._finished.discard(run_id)


bus = EventBus()
