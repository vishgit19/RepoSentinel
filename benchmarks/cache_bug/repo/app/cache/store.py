"""An in-memory cache with per-entry TTL and prefix invalidation."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

DEFAULT_TTL_SECONDS = 300.0


@dataclass
class Entry:
    value: Any
    stored_at: float
    ttl: float

    def is_stale(self, now: float) -> bool:
        return now - self.stored_at >= self.ttl


class CacheStore:
    def __init__(self, ttl_seconds: float = DEFAULT_TTL_SECONDS) -> None:
        self.ttl_seconds = ttl_seconds
        self._entries: dict[str, Entry] = {}
        self.hits = 0
        self.misses = 0

    def get(self, key: str, now: float | None = None) -> Any | None:
        current = time.time() if now is None else now
        entry = self._entries.get(key)
        if entry is None:
            self.misses += 1
            return None
        if entry.is_stale(current):
            del self._entries[key]
            self.misses += 1
            return None
        self.hits += 1
        return entry.value

    def set(self, key: str, value: Any, now: float | None = None, ttl: float | None = None) -> None:
        current = time.time() if now is None else now
        self._entries[key] = Entry(
            value=value, stored_at=current, ttl=self.ttl_seconds if ttl is None else ttl
        )

    def invalidate(self, key: str) -> bool:
        """Drop exactly one key. Returns True when something was removed."""
        return self._entries.pop(key, None) is not None

    def invalidate_prefix(self, prefix: str) -> int:
        """Drop every key starting with *prefix*. Returns the number removed."""
        doomed = [key for key in self._entries if key.startswith(prefix)]
        for key in doomed:
            del self._entries[key]
        return len(doomed)

    def keys(self) -> list[str]:
        return sorted(self._entries)

    def clear(self) -> None:
        self._entries.clear()
        self.hits = 0
        self.misses = 0

    def __len__(self) -> int:
        return len(self._entries)
