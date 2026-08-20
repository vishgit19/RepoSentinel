"""Embedding backends.

``OpenAIEmbeddings``
    Real dense embeddings via the OpenAI embeddings API, with an on-disk cache
    keyed by content hash so re-indexing the same repository is free.

``HashingEmbeddings``
    A deterministic, offline hashing vectoriser (the classic "hashing trick"
    with sub-linear term weighting and L2 normalisation). It is a genuine
    lexical vector space - not a stub - and is used when no API key is present
    and by the test suite, where determinism matters more than semantics.

Both satisfy :class:`EmbeddingBackend`, so retrieval code never branches on
which one is active. ``describe()`` surfaces the active backend in the UI so a
demo never misrepresents which embeddings produced a result.
"""

from __future__ import annotations

import abc
import hashlib
import math
import sqlite3
import struct
from pathlib import Path

import numpy as np

from reposentinel.config import Settings, get_settings
from reposentinel.retrieval.bm25 import tokenize

# Cost per million tokens, used for run cost accounting.
EMBEDDING_PRICES: dict[str, float] = {
    "text-embedding-3-small": 0.02,
    "text-embedding-3-large": 0.13,
    "text-embedding-ada-002": 0.10,
}


class EmbeddingBackend(abc.ABC):
    name: str = "abstract"
    dimensions: int = 0

    @abc.abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an ``(len(texts), dimensions)`` float32 array of unit vectors."""

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]

    def describe(self) -> dict[str, object]:
        return {"backend": self.name, "dimensions": self.dimensions, "semantic": False}


def _l2_normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


class HashingEmbeddings(EmbeddingBackend):
    """Deterministic offline embeddings via feature hashing."""

    name = "hashing"

    def __init__(self, dimensions: int = 512) -> None:
        self.dimensions = dimensions

    def _hash(self, token: str) -> tuple[int, float]:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        bucket = value % self.dimensions
        # Signed hashing reduces collision bias.
        sign = 1.0 if (value >> 63) & 1 else -1.0
        return bucket, sign

    def embed(self, texts: list[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        for row, text in enumerate(texts):
            counts: dict[str, int] = {}
            for token in tokenize(text):
                counts[token] = counts.get(token, 0) + 1
            for token, count in counts.items():
                bucket, sign = self._hash(token)
                # Sub-linear term weighting, as in tf-idf's log-tf.
                matrix[row, bucket] += sign * (1.0 + math.log(count))
        return _l2_normalise(matrix)

    def describe(self) -> dict[str, object]:
        return {
            "backend": self.name,
            "dimensions": self.dimensions,
            "semantic": False,
            "note": "deterministic lexical hashing (no API key required)",
        }


class EmbeddingCache:
    """Content-hash keyed cache so embeddings are paid for at most once."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS embedding_cache (
                    key TEXT PRIMARY KEY,
                    model TEXT NOT NULL,
                    dimensions INTEGER NOT NULL,
                    vector BLOB NOT NULL
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    @staticmethod
    def key(model: str, text: str) -> str:
        return hashlib.sha256(f"{model}\x00{text}".encode()).hexdigest()

    def get_many(self, model: str, texts: list[str]) -> dict[int, np.ndarray]:
        if not texts:
            return {}
        keys = {self.key(model, text): index for index, text in enumerate(texts)}
        found: dict[int, np.ndarray] = {}
        with self._connect() as connection:
            placeholders = ",".join("?" * len(keys))
            rows = connection.execute(
                f"SELECT key, dimensions, vector FROM embedding_cache WHERE key IN ({placeholders})",
                list(keys),
            ).fetchall()
        for key, dimensions, blob in rows:
            vector = np.array(struct.unpack(f"{dimensions}f", blob), dtype=np.float32)
            found[keys[key]] = vector
        return found

    def put_many(self, model: str, texts: list[str], vectors: np.ndarray) -> None:
        with self._connect() as connection:
            connection.executemany(
                "INSERT OR REPLACE INTO embedding_cache (key, model, dimensions, vector) "
                "VALUES (?, ?, ?, ?)",
                [
                    (
                        self.key(model, text),
                        model,
                        int(vector.shape[0]),
                        struct.pack(f"{vector.shape[0]}f", *vector.tolist()),
                    )
                    for text, vector in zip(texts, vectors, strict=False)
                ],
            )


class OpenAIEmbeddings(EmbeddingBackend):
    """Dense embeddings from the OpenAI embeddings API."""

    name = "openai"

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        base_url: str | None = None,
        cache: EmbeddingCache | None = None,
        batch_size: int = 128,
    ) -> None:
        from openai import OpenAI

        self.model = model
        self.batch_size = batch_size
        self.cache = cache
        self._client = OpenAI(api_key=api_key, base_url=base_url) if api_key else OpenAI(base_url=base_url)
        self.dimensions = 1536 if "small" in model or "ada" in model else 3072
        self.tokens_used = 0

    @property
    def cost_usd(self) -> float:
        price = EMBEDDING_PRICES.get(self.model, 0.02)
        return self.tokens_used / 1_000_000 * price

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimensions), dtype=np.float32)

        cached = self.cache.get_many(self.model, texts) if self.cache else {}
        missing = [index for index in range(len(texts)) if index not in cached]
        vectors: dict[int, np.ndarray] = dict(cached)

        for start in range(0, len(missing), self.batch_size):
            batch_indices = missing[start : start + self.batch_size]
            # The API rejects empty strings.
            batch = [texts[i] if texts[i].strip() else "(empty)" for i in batch_indices]
            response = self._client.embeddings.create(model=self.model, input=batch)
            self.tokens_used += getattr(response.usage, "total_tokens", 0) or 0
            for offset, item in enumerate(response.data):
                vectors[batch_indices[offset]] = np.array(item.embedding, dtype=np.float32)
            if self.cache:
                self.cache.put_many(
                    self.model,
                    batch,
                    np.stack([vectors[i] for i in batch_indices]),
                )

        self.dimensions = int(next(iter(vectors.values())).shape[0])
        matrix = np.stack([vectors[index] for index in range(len(texts))])
        return _l2_normalise(matrix)

    def describe(self) -> dict[str, object]:
        return {
            "backend": self.name,
            "model": self.model,
            "dimensions": self.dimensions,
            "semantic": True,
            "tokens_used": self.tokens_used,
            "cost_usd": round(self.cost_usd, 6),
        }


def build_embedding_backend(settings: Settings | None = None) -> EmbeddingBackend:
    """Pick the best available embedding backend."""
    settings = settings or get_settings()
    api_key = settings.openai_api_key
    if api_key:
        try:
            return OpenAIEmbeddings(
                model=settings.embedding_model,
                api_key=api_key,
                base_url=settings.openai_base_url,
                cache=EmbeddingCache(settings.data_dir / "embedding_cache.db"),
            )
        except Exception:  # noqa: BLE001 - never let indexing die over a client error
            pass
    return HashingEmbeddings()
