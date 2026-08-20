# content-service

A localised article service with a read-through cache.

## Layout

    app/cache/keys.py        cache key construction
    app/cache/store.py       TTL cache with prefix invalidation
    app/content/repository.py  authoritative store, keyed by (article, locale)
    app/content/service.py   read-through caching layer

## Contract

An article exists in several locales. Two lookups may share a cache entry only
when every component of their identity matches, locale included. A write
invalidates every cached variant of the article it touched.

## Running the tests

    python -m pytest
