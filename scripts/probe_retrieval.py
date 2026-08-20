"""Manual probe: hybrid retrieval with real OpenAI embeddings and LLM reranking.

Run with:  python scripts/probe_retrieval.py
Compares BM25-only, dense-only, hybrid and graph modes on benchmark problem 1.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from reposentinel.benchmarks import get_benchmark  # noqa: E402
from reposentinel.config import get_settings  # noqa: E402
from reposentinel.models.providers import build_provider  # noqa: E402
from reposentinel.retrieval.embeddings import build_embedding_backend  # noqa: E402
from reposentinel.retrieval.indexer import CodeIndexer  # noqa: E402
from reposentinel.retrieval.pipeline import HybridRetriever  # noqa: E402
from reposentinel.retrieval.reranker import LLMReranker  # noqa: E402
from reposentinel.retrieval.vector_store import SqliteVectorStore  # noqa: E402
from reposentinel.workspace import Workspace  # noqa: E402


def main() -> int:
    settings = get_settings()
    manifest = get_benchmark("logic_bug")
    workspace = Workspace.prepare("logic_bug", "probe_retrieval")

    try:
        index = CodeIndexer("probe_retrieval", workspace.root).index(workspace.relative_files())
        print(f"index: {index.stats()}")

        embeddings = build_embedding_backend(settings)
        print(f"embeddings: {embeddings.describe()}")

        provider = build_provider("gpt-4.1-mini", settings)
        retriever = HybridRetriever(
            repo_id="probe_retrieval",
            index=index,
            embeddings=embeddings,
            vector_store=SqliteVectorStore(settings.data_dir / "probe_vectors.db"),
            reranker=LLMReranker(provider),
            settings=settings,
        )
        build_stats = retriever.build()
        print(
            f"build: {build_stats['chunks_indexed']} chunks embedded in "
            f"{build_stats['embed_ms']}ms via {build_stats['embedding']['backend']}"
        )

        gold = set(manifest.gold_files)
        for mode in ("bm25", "dense", "hybrid", "graph"):
            result = retriever.retrieve(manifest.issue, mode=mode, final_k=6)
            hit = "HIT " if gold & set(result.paths) else "MISS"
            rank = next(
                (i + 1 for i, c in enumerate(result.chunks) if c.path in gold), None
            )
            print(
                f"\n[{mode:6s}] {hit} gold_rank={rank} "
                f"bm25={result.stats.bm25_candidates} dense={result.stats.dense_candidates} "
                f"merged={result.stats.merged_candidates} graph+{result.stats.graph_expanded} "
                f"rerank={result.stats.rerank_backend} {result.stats.total_ms}ms "
                f"cost=${result.stats.usage.cost_usd:.6f}"
            )
            for chunk in result.chunks[:6]:
                provenance = chunk.provenance
                score = provenance.rerank_score if provenance else None
                extra = ""
                if provenance and provenance.graph_relation:
                    extra = f" via {provenance.graph_relation}({provenance.graph_source})"
                label = f"{chunk.path}::{chunk.symbol}" if chunk.symbol else chunk.path
                print(
                    f"    {label:58s} L{chunk.start_line}-{chunk.end_line} "
                    f"[{provenance.retriever if provenance else '?'}"
                    f"{f' score={score:.1f}' if score is not None else ''}]{extra}"
                )
        print("\nOK")
        return 0
    finally:
        workspace.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
