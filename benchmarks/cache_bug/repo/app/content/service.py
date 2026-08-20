"""Read-through caching in front of the article repository.

Reads are cached per article *and per locale*: a German reader and an English
reader are asking for different content, so they must not share a cache entry.
Writes invalidate every variant of the article they touched.
"""

from __future__ import annotations

from app.cache.keys import CacheKeyParts, make_key
from app.cache.store import CacheStore
from app.content.repository import Article, ArticleRepository

NAMESPACE = "article"


class ArticleService:
    def __init__(
        self,
        repository: ArticleRepository | None = None,
        cache: CacheStore | None = None,
    ) -> None:
        self.repository = repository or ArticleRepository()
        self.cache = cache or CacheStore()

    def _parts(self, article_id: str, locale: str) -> CacheKeyParts:
        return CacheKeyParts(
            namespace=NAMESPACE,
            identity=article_id,
            variants={"locale": locale},
        )

    def get_article(
        self, article_id: str, locale: str, now: float | None = None
    ) -> Article | None:
        """Return an article, serving from cache when possible."""
        key = make_key(self._parts(article_id, locale))
        cached = self.cache.get(key, now=now)
        if cached is not None:
            return cached

        article = self.repository.get(article_id, locale)
        if article is None:
            return None
        self.cache.set(key, article, now=now)
        return article

    def update_title(
        self, article_id: str, locale: str, title: str, now: float | None = None
    ) -> Article:
        """Update one locale's title and drop the article's cached variants."""
        updated = self.repository.update_title(article_id, locale, title)
        self.invalidate(article_id, now=now)
        return updated

    def invalidate(self, article_id: str, now: float | None = None) -> int:
        """Drop every cached variant of *article_id*."""
        prefix = self._parts(article_id, locale="").prefix()
        return self.cache.invalidate_prefix(prefix)

    def cache_keys(self) -> list[str]:
        return self.cache.keys()
