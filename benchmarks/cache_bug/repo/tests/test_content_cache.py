"""Caching contract for the article service.

The rule these tests encode: two lookups share a cache entry only when every
component of their identity matches. Locale is part of that identity.
"""

from __future__ import annotations

from app.cache.keys import CacheKeyParts, make_key
from app.content.repository import Article, ArticleRepository
from app.content.service import ArticleService

T0 = 1_700_000_000.0


def build_service() -> ArticleService:
    repository = ArticleRepository()
    repository.seed(Article("a1", "en", "Hello", "English body"))
    repository.seed(Article("a1", "de", "Hallo", "Deutscher Text"))
    repository.seed(Article("a2", "en", "Second", "Another body"))
    return ArticleService(repository=repository)


def test_cache_key_includes_every_variant():
    english = make_key(CacheKeyParts("article", "a1", {"locale": "en"}))
    german = make_key(CacheKeyParts("article", "a1", {"locale": "de"}))
    assert english != german, "locale must change the cache key"


def test_reading_two_locales_returns_each_locale():
    service = build_service()
    assert service.get_article("a1", "en", now=T0).title == "Hello"
    assert service.get_article("a1", "de", now=T0).title == "Hallo"


def test_second_locale_is_not_served_from_the_first_locales_entry():
    service = build_service()
    service.get_article("a1", "en", now=T0)
    german = service.get_article("a1", "de", now=T0)
    assert german.locale == "de"
    assert german.body == "Deutscher Text"


def test_each_locale_gets_its_own_cache_entry():
    service = build_service()
    service.get_article("a1", "en", now=T0)
    service.get_article("a1", "de", now=T0)
    assert len(service.cache) == 2, service.cache_keys()


def test_repeated_read_is_served_from_cache():
    service = build_service()
    service.get_article("a1", "en", now=T0)
    reads_before = service.repository.reads
    service.get_article("a1", "en", now=T0 + 1)
    assert service.repository.reads == reads_before
    assert service.cache.hits == 1


def test_update_invalidates_every_locale_of_that_article():
    service = build_service()
    service.get_article("a1", "en", now=T0)
    service.get_article("a1", "de", now=T0)
    service.get_article("a2", "en", now=T0)

    service.update_title("a1", "en", "Hello again", now=T0)

    assert service.get_article("a1", "en", now=T0).title == "Hello again"
    # A different article must survive the invalidation.
    assert "article:a2" in " ".join(service.cache_keys())


def test_stale_entry_expires_after_its_ttl():
    service = build_service()
    service.get_article("a1", "en", now=T0)
    service.repository.update_title("a1", "en", "Changed underneath")
    fresh = service.get_article("a1", "en", now=T0 + service.cache.ttl_seconds + 1)
    assert fresh.title == "Changed underneath"


def test_missing_article_is_not_cached():
    service = build_service()
    assert service.get_article("nope", "en", now=T0) is None
    assert len(service.cache) == 0
