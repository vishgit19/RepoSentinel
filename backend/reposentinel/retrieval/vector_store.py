"""Vector stores.

``SqliteVectorStore`` is the default: vectors are stored as float32 blobs and
searched with a NumPy matrix product. At the scale of a single repository index
(hundreds to low thousands of chunks) an exact search is both faster and more
accurate than an approximate index, and it needs no server.

``PgVectorStore`` implements the same interface against PostgreSQL + pgvector
for anyone running the Docker Compose stack. Selected with
``REPOSENTINEL_VECTOR_STORE=pgvector``.
"""

from __future__ import annotations

import abc
import sqlite3
import struct
from pathlib import Path

import numpy as np

from reposentinel.config import Settings, get_settings
from reposentinel.models.schemas import CodeChunk


def pack(vector: np.ndarray) -> bytes:
    array = np.asarray(vector, dtype=np.float32).ravel()
    return struct.pack(f"{array.shape[0]}f", *array.tolist())


def unpack(blob: bytes, dimensions: int) -> np.ndarray:
    return np.array(struct.unpack(f"{dimensions}f", blob), dtype=np.float32)


class VectorStore(abc.ABC):
    backend: str = "abstract"

    @abc.abstractmethod
    def upsert(self, repo_id: str, chunks: list[CodeChunk], vectors: np.ndarray) -> int: ...

    @abc.abstractmethod
    def search(
        self, repo_id: str, query_vector: np.ndarray, top_k: int = 20
    ) -> list[tuple[str, float]]: ...

    @abc.abstractmethod
    def get_chunks(self, repo_id: str, chunk_ids: list[str]) -> dict[str, CodeChunk]: ...

    @abc.abstractmethod
    def clear(self, repo_id: str) -> None: ...

    @abc.abstractmethod
    def count(self, repo_id: str) -> int: ...


class SqliteVectorStore(VectorStore):
    backend = "sqlite"

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._create_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _create_schema(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS code_chunks (
                    chunk_id    TEXT NOT NULL,
                    repo_id     TEXT NOT NULL,
                    path        TEXT NOT NULL,
                    symbol      TEXT NOT NULL DEFAULT '',
                    symbol_kind TEXT NOT NULL DEFAULT 'file',
                    start_line  INTEGER NOT NULL DEFAULT 1,
                    end_line    INTEGER NOT NULL DEFAULT 1,
                    language    TEXT NOT NULL DEFAULT 'python',
                    content     TEXT NOT NULL,
                    dimensions  INTEGER NOT NULL,
                    embedding   BLOB NOT NULL,
                    PRIMARY KEY (repo_id, chunk_id)
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_chunks_repo ON code_chunks(repo_id)"
            )

    def upsert(self, repo_id: str, chunks: list[CodeChunk], vectors: np.ndarray) -> int:
        if not chunks:
            return 0
        if len(chunks) != vectors.shape[0]:
            raise ValueError(
                f"chunk/vector count mismatch: {len(chunks)} chunks vs {vectors.shape[0]} vectors"
            )
        rows = [
            (
                chunk.chunk_id,
                repo_id,
                chunk.path,
                chunk.symbol,
                chunk.symbol_kind,
                chunk.start_line,
                chunk.end_line,
                chunk.language,
                chunk.content,
                int(vectors.shape[1]),
                pack(vectors[index]),
            )
            for index, chunk in enumerate(chunks)
        ]
        with self._connect() as connection:
            connection.executemany(
                """
                INSERT OR REPLACE INTO code_chunks
                    (chunk_id, repo_id, path, symbol, symbol_kind, start_line, end_line,
                     language, content, dimensions, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
        return len(rows)

    def search(
        self, repo_id: str, query_vector: np.ndarray, top_k: int = 20
    ) -> list[tuple[str, float]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT chunk_id, dimensions, embedding FROM code_chunks WHERE repo_id = ?",
                (repo_id,),
            ).fetchall()
        if not rows:
            return []

        query = np.asarray(query_vector, dtype=np.float32).ravel()
        norm = float(np.linalg.norm(query)) or 1.0
        query = query / norm

        chunk_ids: list[str] = []
        matrix = np.zeros((len(rows), query.shape[0]), dtype=np.float32)
        kept = 0
        for row in rows:
            if row["dimensions"] != query.shape[0]:
                # Dimension mismatch means the index was built with a different
                # embedding backend; skip rather than return nonsense scores.
                continue
            matrix[kept] = unpack(row["embedding"], row["dimensions"])
            chunk_ids.append(row["chunk_id"])
            kept += 1
        if not kept:
            return []

        scores = matrix[:kept] @ query
        order = np.argsort(-scores)[:top_k]
        return [(chunk_ids[int(i)], float(scores[int(i)])) for i in order]

    def get_chunks(self, repo_id: str, chunk_ids: list[str]) -> dict[str, CodeChunk]:
        if not chunk_ids:
            return {}
        placeholders = ",".join("?" * len(chunk_ids))
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT chunk_id, repo_id, path, symbol, symbol_kind, start_line, end_line,
                       language, content
                FROM code_chunks WHERE repo_id = ? AND chunk_id IN ({placeholders})
                """,
                (repo_id, *chunk_ids),
            ).fetchall()
        return {
            row["chunk_id"]: CodeChunk(
                chunk_id=row["chunk_id"],
                repo_id=row["repo_id"],
                path=row["path"],
                symbol=row["symbol"],
                symbol_kind=row["symbol_kind"],
                start_line=row["start_line"],
                end_line=row["end_line"],
                language=row["language"],
                content=row["content"],
            )
            for row in rows
        }

    def clear(self, repo_id: str) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM code_chunks WHERE repo_id = ?", (repo_id,))

    def count(self, repo_id: str) -> int:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS n FROM code_chunks WHERE repo_id = ?", (repo_id,)
            ).fetchone()
        return int(row["n"]) if row else 0


class PgVectorStore(VectorStore):
    """PostgreSQL + pgvector implementation of the same contract."""

    backend = "pgvector"

    def __init__(self, database_url: str, dimensions: int = 1536) -> None:
        import psycopg  # imported lazily: optional dependency
        from pgvector.psycopg import register_vector

        self._psycopg = psycopg
        self._register_vector = register_vector
        self.database_url = database_url
        self.dimensions = dimensions
        self._create_schema()

    def _connect(self):
        connection = self._psycopg.connect(self.database_url, autocommit=True)
        self._register_vector(connection)
        return connection

    def _create_schema(self) -> None:
        with self._connect() as connection:
            connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
            connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS code_chunks (
                    chunk_id    TEXT NOT NULL,
                    repo_id     TEXT NOT NULL,
                    path        TEXT NOT NULL,
                    symbol      TEXT NOT NULL DEFAULT '',
                    symbol_kind TEXT NOT NULL DEFAULT 'file',
                    start_line  INTEGER NOT NULL DEFAULT 1,
                    end_line    INTEGER NOT NULL DEFAULT 1,
                    language    TEXT NOT NULL DEFAULT 'python',
                    content     TEXT NOT NULL,
                    embedding   vector({self.dimensions}) NOT NULL,
                    PRIMARY KEY (repo_id, chunk_id)
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON code_chunks "
                "USING hnsw (embedding vector_cosine_ops)"
            )

    def upsert(self, repo_id: str, chunks: list[CodeChunk], vectors: np.ndarray) -> int:
        if not chunks:
            return 0
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.executemany(
                """
                INSERT INTO code_chunks
                    (chunk_id, repo_id, path, symbol, symbol_kind, start_line, end_line,
                     language, content, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (repo_id, chunk_id) DO UPDATE SET
                    content = EXCLUDED.content, embedding = EXCLUDED.embedding
                """,
                [
                    (
                        chunk.chunk_id,
                        repo_id,
                        chunk.path,
                        chunk.symbol,
                        chunk.symbol_kind,
                        chunk.start_line,
                        chunk.end_line,
                        chunk.language,
                        chunk.content,
                        np.asarray(vectors[index], dtype=np.float32),
                    )
                    for index, chunk in enumerate(chunks)
                ],
            )
        return len(chunks)

    def search(
        self, repo_id: str, query_vector: np.ndarray, top_k: int = 20
    ) -> list[tuple[str, float]]:
        with self._connect() as connection, connection.cursor() as cursor:
            rows = cursor.execute(
                """
                SELECT chunk_id, 1 - (embedding <=> %s) AS score
                FROM code_chunks WHERE repo_id = %s
                ORDER BY embedding <=> %s LIMIT %s
                """,
                (
                    np.asarray(query_vector, dtype=np.float32),
                    repo_id,
                    np.asarray(query_vector, dtype=np.float32),
                    top_k,
                ),
            ).fetchall()
        return [(row[0], float(row[1])) for row in rows]

    def get_chunks(self, repo_id: str, chunk_ids: list[str]) -> dict[str, CodeChunk]:
        if not chunk_ids:
            return {}
        with self._connect() as connection, connection.cursor() as cursor:
            rows = cursor.execute(
                """
                SELECT chunk_id, repo_id, path, symbol, symbol_kind, start_line, end_line,
                       language, content
                FROM code_chunks WHERE repo_id = %s AND chunk_id = ANY(%s)
                """,
                (repo_id, chunk_ids),
            ).fetchall()
        return {
            row[0]: CodeChunk(
                chunk_id=row[0],
                repo_id=row[1],
                path=row[2],
                symbol=row[3],
                symbol_kind=row[4],
                start_line=row[5],
                end_line=row[6],
                language=row[7],
                content=row[8],
            )
            for row in rows
        }

    def clear(self, repo_id: str) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM code_chunks WHERE repo_id = %s", (repo_id,))

    def count(self, repo_id: str) -> int:
        with self._connect() as connection, connection.cursor() as cursor:
            row = cursor.execute(
                "SELECT COUNT(*) FROM code_chunks WHERE repo_id = %s", (repo_id,)
            ).fetchone()
        return int(row[0]) if row else 0


def build_vector_store(settings: Settings | None = None, dimensions: int = 1536) -> VectorStore:
    settings = settings or get_settings()
    if settings.vector_store == "pgvector":
        if not settings.database_url:
            raise ValueError(
                "REPOSENTINEL_VECTOR_STORE=pgvector requires REPOSENTINEL_DATABASE_URL"
            )
        try:
            return PgVectorStore(settings.database_url, dimensions=dimensions)
        except ImportError as exc:
            raise ImportError(
                "pgvector backend needs the optional extras: "
                "pip install -r requirements-optional.txt"
            ) from exc
    return SqliteVectorStore(settings.data_dir / "vectors.db")


def describe_vector_store(settings: Settings | None = None) -> dict[str, object]:
    settings = settings or get_settings()
    return {
        "configured": settings.vector_store,
        "database_url_present": bool(settings.database_url),
    }
