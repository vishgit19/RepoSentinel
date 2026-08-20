"""Retrieval pipeline: BM25, fusion, reranking and graph expansion.

These tests use the deterministic hashing embedder so they run offline and
give identical results on every machine. Real OpenAI embeddings are exercised
separately by ``scripts/probe_retrieval.py``.
"""

from __future__ import annotations

import numpy as np
import pytest
from reposentinel.config import get_settings
from reposentinel.retrieval.bm25 import build_bm25, tokenize
from reposentinel.retrieval.embeddings import HashingEmbeddings
from reposentinel.retrieval.indexer import CodeIndexer
from reposentinel.retrieval.pipeline import HybridRetriever
from reposentinel.retrieval.reranker import LexicalReranker
from reposentinel.retrieval.vector_store import SqliteVectorStore
from reposentinel.workspace import Workspace


class TestTokenizer:
    def test_snake_case_is_split_and_kept(self):
        tokens = tokenize("validate_token")
        assert "validate_token" in tokens
        assert "validate" in tokens
        assert "token" in tokens

    def test_camel_case_is_split_and_kept(self):
        tokens = tokenize("SessionToken")
        assert "sessiontoken" in tokens
        assert "session" in tokens
        assert "token" in tokens

    def test_acronym_boundary(self):
        assert "http" in tokenize("HTTPResponse")
        assert "response" in tokenize("HTTPResponse")

    def test_stopwords_and_single_chars_dropped(self):
        assert tokenize("the a def self return") == []

    def test_natural_language_query(self):
        tokens = tokenize("expired session token comparison")
        assert {"expired", "session", "token", "comparison"} <= set(tokens)


class TestBM25:
    @pytest.fixture()
    def index(self):
        return build_bm25(
            [
                ("a", "def is_expired(self, now): return current > self.expires_at"),
                ("b", "def seconds_remaining(self, now): return max(0.0, self.expires_at - current)"),
                ("c", "class UserRepository: def update_email(self, user_id, email)"),
                ("d", "def test_update_email(): repo.update_email('u9', 'new@example.com')"),
            ]
        )

    def test_identifier_query_ranks_exact_match_first(self, index):
        results = index.search("is_expired", top_k=3)
        assert results
        assert results[0][0] == "a"

    def test_natural_language_query(self, index):
        results = index.search("update a user's email", top_k=3)
        assert {doc for doc, _ in results} & {"c", "d"}

    def test_unknown_term_returns_nothing(self, index):
        assert index.search("kubernetes_operator_reconcile", top_k=5) == []

    def test_scores_are_positive_and_sorted(self, index):
        results = index.search("expires_at", top_k=4)
        scores = [score for _, score in results]
        assert all(s > 0 for s in scores)
        assert scores == sorted(scores, reverse=True)

    def test_idf_penalises_ubiquitous_terms(self, index):
        # 'def' is a stopword; 'expires_at' appears in 2 of 4 docs, 'email' in 2.
        assert index._idf("expires_at") > 0
        assert index._idf("nonexistent") == 0.0


class TestHashingEmbeddings:
    def test_deterministic(self):
        backend = HashingEmbeddings(dimensions=128)
        first = backend.embed(["def is_expired(self): pass"])
        second = backend.embed(["def is_expired(self): pass"])
        assert np.allclose(first, second)

    def test_unit_norm(self):
        backend = HashingEmbeddings(dimensions=128)
        vectors = backend.embed(["alpha beta", "gamma delta epsilon"])
        norms = np.linalg.norm(vectors, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)

    def test_similar_text_scores_higher_than_unrelated(self):
        backend = HashingEmbeddings(dimensions=256)
        vectors = backend.embed(
            [
                "session token expiry validation",
                "token expiry check for a session",
                "matrix multiplication kernel tuning",
            ]
        )
        related = float(vectors[0] @ vectors[1])
        unrelated = float(vectors[0] @ vectors[2])
        assert related > unrelated

    def test_empty_input(self):
        assert HashingEmbeddings(dimensions=32).embed([]).shape == (0, 32)


class TestVectorStore:
    def test_roundtrip_and_search(self, tmp_path):
        from reposentinel.models.schemas import CodeChunk

        store = SqliteVectorStore(tmp_path / "v.db")
        chunks = [
            CodeChunk(chunk_id="c1", repo_id="r", path="a.py", content="alpha"),
            CodeChunk(chunk_id="c2", repo_id="r", path="b.py", content="beta"),
        ]
        vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        assert store.upsert("r", chunks, vectors) == 2
        assert store.count("r") == 2

        hits = store.search("r", np.array([1.0, 0.05], dtype=np.float32), top_k=2)
        assert hits[0][0] == "c1"
        assert hits[0][1] > hits[1][1]

        fetched = store.get_chunks("r", ["c1", "c2"])
        assert fetched["c1"].content == "alpha"

        store.clear("r")
        assert store.count("r") == 0

    def test_repos_are_isolated(self, tmp_path):
        from reposentinel.models.schemas import CodeChunk

        store = SqliteVectorStore(tmp_path / "v.db")
        vector = np.array([[1.0, 0.0]], dtype=np.float32)
        store.upsert("r1", [CodeChunk(chunk_id="x", repo_id="r1", path="a.py")], vector)
        store.upsert("r2", [CodeChunk(chunk_id="y", repo_id="r2", path="b.py")], vector)
        assert [cid for cid, _ in store.search("r1", vector[0], top_k=5)] == ["x"]

    def test_mismatched_counts_raise(self, tmp_path):
        from reposentinel.models.schemas import CodeChunk

        store = SqliteVectorStore(tmp_path / "v.db")
        with pytest.raises(ValueError, match="mismatch"):
            store.upsert(
                "r",
                [CodeChunk(chunk_id="a", repo_id="r", path="a.py")],
                np.zeros((2, 4), dtype=np.float32),
            )


@pytest.fixture(scope="module")
def retriever(tmp_path_factory):
    workspace = Workspace.prepare("logic_bug", "test_retrieval_ws")
    settings = get_settings()
    index = CodeIndexer("logic_bug_test", workspace.root).index(workspace.relative_files())
    store = SqliteVectorStore(tmp_path_factory.mktemp("vectors") / "vectors.db")
    built = HybridRetriever(
        repo_id="logic_bug_test",
        index=index,
        embeddings=HashingEmbeddings(dimensions=384),
        vector_store=store,
        reranker=LexicalReranker(),
        settings=settings,
    )
    built.build()
    try:
        yield built
    finally:
        workspace.cleanup()


class TestHybridPipeline:
    def test_build_reports_stats(self, retriever):
        stats = retriever.build_stats
        assert stats["chunks_indexed"] > 20
        assert stats["bm25_documents"] == stats["chunks_indexed"]
        assert stats["vocabulary"] > 50

    def test_bm25_mode_finds_the_gold_symbol(self, retriever):
        result = retriever.retrieve("is_expired token expiry comparison", mode="bm25", final_k=5)
        assert "app/auth/token.py" in result.paths
        assert result.stats.dense_candidates == 0
        assert result.stats.bm25_candidates > 0

    def test_dense_mode_runs_without_bm25(self, retriever):
        result = retriever.retrieve("session expiry validation", mode="dense", final_k=5)
        assert result.stats.bm25_candidates == 0
        assert result.stats.dense_candidates > 0
        assert result.chunks

    def test_hybrid_mode_uses_both_retrievers(self, retriever):
        result = retriever.retrieve("expired session token accepted", mode="hybrid", final_k=6)
        assert result.stats.bm25_candidates > 0
        assert result.stats.dense_candidates > 0
        assert result.stats.merged_candidates > 0
        retrievers = {c.provenance.retriever for c in result.chunks if c.provenance}
        assert retrievers, "provenance was not attached"

    def test_graph_mode_adds_neighbours_with_provenance(self, retriever):
        result = retriever.retrieve("is_expired", mode="graph", final_k=4)
        graph_chunks = [c for c in result.chunks if c.provenance and c.provenance.retriever == "graph"]
        assert graph_chunks, "graph expansion produced nothing"
        for chunk in graph_chunks:
            assert chunk.provenance.graph_relation in {
                "calls",
                "contains",
                "imports",
                "tests",
                "modified_by",
            }
            assert chunk.provenance.graph_source

    def test_gold_file_is_retrieved_for_the_real_issue(self, retriever):
        issue = (
            "Users with an expired session token are still being authenticated. "
            "A token whose expires_at timestamp is in the past is accepted."
        )
        result = retriever.retrieve(issue, mode="graph", final_k=8)
        assert "app/auth/token.py" in result.paths, result.paths

    def test_provenance_records_scores(self, retriever):
        result = retriever.retrieve("validate_token", mode="hybrid", final_k=5)
        for chunk in result.chunks:
            provenance = chunk.provenance
            assert provenance is not None
            assert provenance.fused_score is not None or provenance.retriever == "graph"
            assert provenance.rerank_score is not None or provenance.retriever == "graph"

    def test_context_rendering_includes_provenance_and_respects_budget(self, retriever):
        result = retriever.retrieve("token expiry", mode="hybrid", final_k=6)
        context = result.as_context(max_chars=1500)
        assert len(context) <= 1500
        assert "---" in context
        assert "lines" in context

    def test_no_results_for_nonsense_query(self, retriever):
        result = retriever.retrieve("zzzz_kubernetes_operator_xyzzy", mode="bm25", final_k=5)
        assert result.chunks == []

    def test_stats_timings_are_recorded(self, retriever):
        result = retriever.retrieve("token", mode="graph", final_k=4)
        assert result.stats.total_ms >= 0
        assert result.stats.rerank_backend == "lexical"


class TestLexicalReranker:
    def test_reranker_prefers_the_matching_symbol(self, retriever):
        candidates = retriever.index.chunks
        outcome = LexicalReranker().rerank("is_expired token expiry", candidates, top_k=3)
        assert outcome.ranked
        best = outcome.ranked[0][0]
        assert "is_expired" in best.symbol or "expires" in best.content

    def test_scores_are_bounded_and_ordered(self, retriever):
        outcome = LexicalReranker().rerank("validate token", retriever.index.chunks, top_k=10)
        scores = [score for _, score in outcome.ranked]
        assert all(0.0 <= s <= 10.0 for s in scores)
        assert scores == sorted(scores, reverse=True)

    def test_reranking_actually_reorders(self, retriever):
        candidates = list(retriever.index.chunks)
        outcome = LexicalReranker().rerank("update_email user repository", candidates, top_k=len(candidates))
        before = [c.chunk_id for c in candidates]
        after = [c.chunk_id for c, _ in outcome.ranked]
        assert before != after
