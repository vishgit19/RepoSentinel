"""The comparable approaches a run can be executed with.

Baselines A-D exist so the agentic system (E) can be measured against
progressively weaker configurations of the *same* code path rather than against
a strawman. Only three things vary between them: how context is gathered,
whether the model may call tools, and whether the reflect-and-retry loop is
allowed to run.

That last flag matters for the demo. A baseline must stop at its first failed
patch - "patch, tests fail, stop" - because recovering from a failed attempt is
precisely the capability the full agent is meant to demonstrate.
"""

from __future__ import annotations

from dataclasses import dataclass

# Retrieval mode passed to the retrieval pipeline. ``None`` means no retrieval
# at all: the run is handed seed files directly.
RetrievalMode = str | None


@dataclass(frozen=True)
class Strategy:
    id: str
    label: str
    baseline: str
    description: str
    retrieval: RetrievalMode
    tools: bool
    reflection: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "label": self.label,
            "baseline": self.baseline,
            "description": self.description,
            "retrieval": self.retrieval or "none",
            "tools": self.tools,
            "reflection": self.reflection,
        }


STRATEGIES: tuple[Strategy, ...] = (
    Strategy(
        id="llm_only",
        label="LLM only",
        baseline="A",
        description="Issue plus the manifest's seed files handed straight to the model.",
        retrieval=None,
        tools=False,
        reflection=False,
    ),
    Strategy(
        id="vector_rag",
        label="Vector RAG",
        baseline="B",
        description="Dense embedding retrieval over symbol-boundary chunks.",
        retrieval="dense",
        tools=False,
        reflection=False,
    ),
    Strategy(
        id="hybrid_rag",
        label="Hybrid RAG + rerank",
        baseline="C",
        description="BM25 and dense candidates fused, then reranked.",
        retrieval="hybrid",
        tools=False,
        reflection=False,
    ),
    Strategy(
        id="graph_rag",
        label="Graph-enhanced RAG",
        baseline="D",
        description="Hybrid retrieval expanded along the symbol/dependency graph.",
        retrieval="graph",
        tools=False,
        reflection=False,
    ),
    Strategy(
        id="agentic",
        label="RepoSentinel (full agent)",
        baseline="E",
        description=(
            "Graph retrieval plus tool-using investigation, test execution and the "
            "reflect-and-retry loop."
        ),
        retrieval="graph",
        tools=True,
        reflection=True,
    ),
)

DEFAULT_STRATEGY = "agentic"

BY_ID: dict[str, Strategy] = {strategy.id: strategy for strategy in STRATEGIES}


def get_strategy(strategy_id: str) -> Strategy:
    return BY_ID.get(strategy_id, BY_ID[DEFAULT_STRATEGY])


def strategy_ids() -> list[str]:
    return [strategy.id for strategy in STRATEGIES]


def describe_strategies() -> list[dict[str, object]]:
    return [strategy.to_dict() for strategy in STRATEGIES]
