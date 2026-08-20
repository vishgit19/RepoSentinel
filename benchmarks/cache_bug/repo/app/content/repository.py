"""The authoritative article store, keyed by (article id, locale)."""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Article:
    article_id: str
    locale: str
    title: str
    body: str
    revision: int = 1


class ArticleRepository:
    """A tiny stand-in for the database layer."""

    def __init__(self) -> None:
        self._articles: dict[tuple[str, str], Article] = {}
        self.reads = 0

    def seed(self, article: Article) -> None:
        self._articles[(article.article_id, article.locale)] = article

    def get(self, article_id: str, locale: str) -> Article | None:
        self.reads += 1
        return self._articles.get((article_id, locale))

    def locales_for(self, article_id: str) -> list[str]:
        return sorted(
            locale for (found_id, locale) in self._articles if found_id == article_id
        )

    def update_title(self, article_id: str, locale: str, title: str) -> Article:
        key = (article_id, locale)
        existing = self._articles.get(key)
        if existing is None:
            raise KeyError(f"no article {article_id!r} for locale {locale!r}")
        updated = replace(existing, title=title, revision=existing.revision + 1)
        self._articles[key] = updated
        return updated
