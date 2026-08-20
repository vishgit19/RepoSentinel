"""The cache store itself, independent of what is being cached.

These pass with the seeded defect in place: the store is correct, which is what
makes the bug interesting to locate.
"""

from __future__ import annotations

from app.cache.store import CacheStore

T0 = 1_700_000_000.0


def test_set_then_get_returns_the_value():
    cache = CacheStore()
    cache.set("k", 42, now=T0)
    assert cache.get("k", now=T0) == 42
    assert cache.hits == 1


def test_missing_key_counts_as_a_miss():
    cache = CacheStore()
    assert cache.get("absent", now=T0) is None
    assert cache.misses == 1


def test_entry_expires_at_its_ttl():
    cache = CacheStore(ttl_seconds=10)
    cache.set("k", "v", now=T0)
    assert cache.get("k", now=T0 + 9) == "v"
    assert cache.get("k", now=T0 + 10) is None


def test_invalidate_removes_one_key_only():
    cache = CacheStore()
    cache.set("a:1", 1, now=T0)
    cache.set("a:2", 2, now=T0)
    assert cache.invalidate("a:1") is True
    assert cache.keys() == ["a:2"]


def test_invalidate_prefix_removes_every_match():
    cache = CacheStore()
    cache.set("article:a1|locale=en", 1, now=T0)
    cache.set("article:a1|locale=de", 2, now=T0)
    cache.set("article:a2|locale=en", 3, now=T0)
    assert cache.invalidate_prefix("article:a1") == 2
    assert cache.keys() == ["article:a2|locale=en"]


def test_clear_resets_counters():
    cache = CacheStore()
    cache.set("k", 1, now=T0)
    cache.get("k", now=T0)
    cache.clear()
    assert len(cache) == 0
    assert cache.hits == 0
