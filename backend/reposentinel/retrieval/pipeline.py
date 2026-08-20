"""The retrieval pipeline.

    Query
      -> BM25 + dense retrieval (in parallel over the same chunk set)
      -> candidate merge (weighted reciprocal-rank fusion)
      -> rerank
      -> dependency / symbol-graph expansion
      -> final context

Every returned chunk carries :class:`Provenance` recording which retrievers
found it, its score at each stage, and - for graph-expanded chunks - which
relation and which source symbol pulled it in. Nothing reaches a prompt
without that trail.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from reposentinel.config import Settings, get_settings
from reposentinel.models.providers.base import Usage
from reposentinel.models.schemas import CodeChunk, Provenance
from reposentinel.retrieval.bm25 import BM25Index, build_bm25
from reposentinel.retrieval.embeddings import EmbeddingBackend
from reposentinel.retrieval.indexer import RepoIndex
from reposentinel.retrieval.reranker import Reranker
from reposentinel.retrieval.vector_store import VectorStore

RetrievalMode = Literal["bm25", "dense", "hybrid", "graph"]

# Reciprocal-rank-fusion damping. 60 is the value from the original RRF paper
# and behaves well when merging two retrievers of differing quality.
RRF_K = 60


@dataclass
class RetrievalStats:
    bm25_candidates: int = 0
    dense_candidates: int = 0
    merged_candidates: int = 0
    reranked: int = 0
    graph_expanded: int = 0
    final: int = 0
    embed_ms: int = 0
    bm25_ms: int = 0
    dense_ms: int = 0
    rerank_ms: int = 0
    expand_ms: int = 0
    total_ms: int = 0
    rerank_backend: str = ""
    embedding_backend: str = ""
    llm_calls: int = 0
    usage: Usage = field(default_factory=Usage)


@dataclass
class RetrievalResult:
    query: str
    mode: RetrievalMode
    chunks: list[CodeChunk] = field(default_factory=list)
    stats: RetrievalStats = field(default_factory=RetrievalStats)

    @property
    def paths(self) -> list[str]:
        seen: list[str] = []
        for chunk in self.chunks:
            if chunk.path not in seen:
                seen.append(chunk.path)
        return seen

    def as_context(self, max_chars: int = 12_000) -> str:
        """Render the retrieved chunks for a prompt, with provenance headers."""
        blocks: list[str] = []
        budget = max_chars
        for chunk in self.chunks:
            provenance = chunk.provenance
            trail = provenance.retriever if provenance else "unknown"
            if provenance and provenance.rerank_score is not None:
                trail += f", rerank={provenance.rerank_score:.1f}"
            if provenance and provenance.graph_relation:
                trail += f", via {provenance.graph_relation} of {provenance.graph_source}"
            label = f"{chunk.path}::{chunk.symbol}" if chunk.symbol else chunk.path
            header = f"--- {label}  (lines {chunk.start_line}-{chunk.end_line}) [{trail}] ---"
            body = chunk.content
            block = f"{header}\n{body}"
            if len(block) > budget:
                break
            blocks.append(block)
            budget -= len(block)
        return "\n\n".join(blocks)

    def evidence(self) -> list[str]:
        return [c.location for c in self.chunks]


class HybridRetriever:
    """Owns the index, the vector store and the retrievers for one repository."""

    def __init__(
        self,
        repo_id: str,
        index: RepoIndex,
        embeddings: EmbeddingBackend,
        vector_store: VectorStore,
        reranker: Reranker,
        settings: Settings | None = None,
    ) -> None:
        self.repo_id = repo_id
        self.index = index
        self.embeddings = embeddings
        self.vector_store = vector_store
        self.reranker = reranker
        self.settings = settings or get_settings()
        self._bm25: BM25Index | None = None
        self._chunks: dict[str, CodeChunk] = {}
        self.build_stats: dict[str, object] = {}

    # -- build -----------------------------------------------------------
    def build(self) -> dict[str, object]:
        """Embed every chunk, populate the vector store and build BM25."""
        started = time.perf_counter()
        chunks = self.index.chunks
        self._chunks = {c.chunk_id: c for c in chunks}

        documents = [(c.chunk_id, self._bm25_document(c)) for c in chunks]
        self._bm25 = build_bm25(
            documents, k1=self.settings.bm25_k1, b=self.settings.bm25_b
        )

        embed_started = time.perf_counter()
        self.vector_store.clear(self.repo_id)
        vectors = self.embeddings.embed([self._embedding_document(c) for c in chunks])
        stored = self.vector_store.upsert(self.repo_id, chunks, vectors)
        embed_ms = int((time.perf_counter() - embed_started) * 1000)

        self.build_stats = {
            "chunks_indexed": stored,
            "bm25_documents": self._bm25.size,
            "vocabulary": len(self._bm25.doc_frequency),
            "embedding": self.embeddings.describe(),
            "vector_store": self.vector_store.backend,
            "embed_ms": embed_ms,
            "total_ms": int((time.perf_counter() - started) * 1000),
            **self.index.stats(),
        }
        return self.build_stats

    def _bm25_document(self, chunk: CodeChunk) -> str:
        """Weight the identifier-bearing parts by repeating them."""
        label = chunk.symbol or chunk.path
        return f"{chunk.path} {label} {label} {chunk.content}"

    def _embedding_document(self, chunk: CodeChunk) -> str:
        label = f"{chunk.path}::{chunk.symbol}" if chunk.symbol else chunk.path
        return f"# {label} ({chunk.symbol_kind}, lines {chunk.start_line}-{chunk.end_line})\n{chunk.content}"

    # -- retrieval -------------------------------------------------------
    def retrieve(
        self,
        query: str,
        mode: RetrievalMode = "graph",
        top_k: int | None = None,
        final_k: int | None = None,
    ) -> RetrievalResult:
        started = time.perf_counter()
        top_k = top_k or self.settings.retrieval_top_k
        final_k = final_k or self.settings.retrieval_final_k
        stats = RetrievalStats(embedding_backend=str(self.embeddings.describe().get("backend", "")))

        bm25_hits: list[tuple[str, float]] = []
        dense_hits: list[tuple[str, float]] = []

        if mode in {"bm25", "hybrid", "graph"}:
            bm25_started = time.perf_counter()
            bm25_hits = self._bm25.search(query, top_k=top_k) if self._bm25 else []
            stats.bm25_ms = int((time.perf_counter() - bm25_started) * 1000)
            stats.bm25_candidates = len(bm25_hits)

        if mode in {"dense", "hybrid", "graph"}:
            embed_started = time.perf_counter()
            query_vector = self.embeddings.embed_one(query)
            stats.embed_ms = int((time.perf_counter() - embed_started) * 1000)
            dense_started = time.perf_counter()
            dense_hits = self.vector_store.search(self.repo_id, query_vector, top_k=top_k)
            stats.dense_ms = int((time.perf_counter() - dense_started) * 1000)
            stats.dense_candidates = len(dense_hits)

        merged = self._merge(query, bm25_hits, dense_hits, mode)
        stats.merged_candidates = len(merged)

        if not merged:
            stats.total_ms = int((time.perf_counter() - started) * 1000)
            return RetrievalResult(query=query, mode=mode, chunks=[], stats=stats)

        outcome = self.reranker.rerank(query, merged, top_k=final_k)
        stats.rerank_ms = outcome.duration_ms
        stats.rerank_backend = outcome.backend
        stats.reranked = len(outcome.ranked)
        stats.llm_calls = outcome.llm_calls
        stats.usage = outcome.usage
        selected = [chunk for chunk, _ in outcome.ranked]

        if mode == "graph":
            expand_started = time.perf_counter()
            expanded = self._expand_graph(selected, limit=max(2, final_k // 2))
            stats.expand_ms = int((time.perf_counter() - expand_started) * 1000)
            stats.graph_expanded = len(expanded)
            selected = selected + expanded

        stats.final = len(selected)
        stats.total_ms = int((time.perf_counter() - started) * 1000)
        return RetrievalResult(query=query, mode=mode, chunks=selected, stats=stats)

    # -- internals -------------------------------------------------------
    def _merge(
        self,
        query: str,
        bm25_hits: list[tuple[str, float]],
        dense_hits: list[tuple[str, float]],
        mode: RetrievalMode,
    ) -> list[CodeChunk]:
        """Weighted reciprocal-rank fusion over both candidate lists."""
        dense_weight = self.settings.hybrid_dense_weight
        sparse_weight = 1.0 - dense_weight
        if mode == "bm25":
            dense_weight, sparse_weight = 0.0, 1.0
        elif mode == "dense":
            dense_weight, sparse_weight = 1.0, 0.0

        fused: dict[str, float] = {}
        provenance: dict[str, Provenance] = {}

        for rank, (chunk_id, score) in enumerate(bm25_hits):
            fused[chunk_id] = fused.get(chunk_id, 0.0) + sparse_weight / (RRF_K + rank + 1)
            provenance[chunk_id] = Provenance(retriever="bm25", query=query, bm25_score=round(score, 4))

        for rank, (chunk_id, score) in enumerate(dense_hits):
            fused[chunk_id] = fused.get(chunk_id, 0.0) + dense_weight / (RRF_K + rank + 1)
            existing = provenance.get(chunk_id)
            if existing is None:
                provenance[chunk_id] = Provenance(
                    retriever="dense", query=query, dense_score=round(score, 4)
                )
            else:
                existing.retriever = "bm25+dense"
                existing.dense_score = round(score, 4)

        ordered = sorted(fused.items(), key=lambda item: -item[1])
        resolved = self._resolve_chunks([chunk_id for chunk_id, _ in ordered])

        merged: list[CodeChunk] = []
        for chunk_id, fused_score in ordered:
            chunk = resolved.get(chunk_id)
            if chunk is None:
                continue
            trail = provenance[chunk_id]
            trail.fused_score = round(fused_score, 6)
            # Copy so repeated queries never share mutable provenance.
            merged.append(chunk.model_copy(update={"provenance": trail}))
        return merged

    def _resolve_chunks(self, chunk_ids: list[str]) -> dict[str, CodeChunk]:
        """Prefer the in-memory index; fall back to the vector store."""
        found = {cid: self._chunks[cid] for cid in chunk_ids if cid in self._chunks}
        missing = [cid for cid in chunk_ids if cid not in found]
        if missing:
            found.update(self.vector_store.get_chunks(self.repo_id, missing))
        return found

    def _expand_graph(self, selected: list[CodeChunk], limit: int) -> list[CodeChunk]:
        """Pull in symbols related to the winners via the symbol graph."""
        hops = self.settings.retrieval_graph_hops
        already = {chunk.chunk_id for chunk in selected}
        additions: list[CodeChunk] = []

        for chunk in selected:
            if len(additions) >= limit:
                break
            for edge, target in self.index.neighbours(chunk, hops=hops):
                if len(additions) >= limit:
                    break
                symbol = self.index.symbols.get(target)
                if symbol is None:
                    continue
                neighbour = next(
                    (
                        candidate
                        for candidate in self.index.chunks
                        if candidate.path == symbol.path
                        and candidate.symbol == symbol.dotted
                    ),
                    None,
                )
                if neighbour is None or neighbour.chunk_id in already:
                    continue
                already.add(neighbour.chunk_id)
                source_label = (
                    f"{chunk.path}::{chunk.symbol}" if chunk.symbol else chunk.path
                )
                additions.append(
                    neighbour.model_copy(
                        update={
                            "provenance": Provenance(
                                retriever="graph",
                                query=f"expansion of {source_label}",
                                graph_relation=edge.relation,
                                graph_source=source_label,
                            )
                        }
                    )
                )
        return additions

    # -- direct lookups used by tools ------------------------------------
    def chunks_for_paths(self, paths: list[str], query: str = "") -> list[CodeChunk]:
        results: list[CodeChunk] = []
        for path in paths:
            for chunk in self.index.chunks_for_file(path):
                results.append(
                    chunk.model_copy(
                        update={"provenance": Provenance(retriever="seed", query=query)}
                    )
                )
        return results

    def describe(self) -> dict[str, object]:
        return {
            "repo_id": self.repo_id,
            "chunks": len(self.index.chunks),
            "bm25_documents": self._bm25.size if self._bm25 else 0,
            "embedding": self.embeddings.describe(),
            "vector_store": self.vector_store.backend,
            "reranker": self.reranker.name,
        }


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
    return float(np.dot(a, b) / denominator)
