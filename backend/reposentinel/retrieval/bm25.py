"""BM25 sparse retrieval with a code-aware tokeniser.

Off-the-shelf tokenisers destroy the most useful signal in source code:
``validate_token`` becomes one opaque term and ``SessionToken`` never matches a
query for "session token". The tokeniser here emits the whole identifier *and*
its sub-tokens, so both the exact-identifier query and the natural-language
query hit the same chunk.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")

# Python keywords and boilerplate carry no discriminative value in a Python
# corpus, so they are dropped to keep idf meaningful.
STOPWORDS = frozenset(
    {
        "the", "a", "an", "and", "or", "of", "to", "in", "is", "it", "for", "on", "with",
        "as", "by", "at", "from", "that", "this", "be", "are", "was", "were", "not",
        "self", "cls", "def", "class", "return", "import", "if", "else", "elif", "none",
        "true", "false", "pass", "raise", "try", "except", "finally", "while", "lambda", "yield", "assert", "del", "global", "nonlocal", "async", "await",
        "str", "int", "float", "bool", "list", "dict", "set", "tuple", "object", "type",
    }
)


def tokenize(text: str) -> list[str]:
    """Split text into search tokens, expanding identifier casing conventions."""
    tokens: list[str] = []
    for raw in _IDENTIFIER_RE.findall(text or ""):
        lowered = raw.lower()
        pieces = [p for p in _CAMEL_BOUNDARY.split(raw) if p]
        subtokens = []
        for piece in pieces:
            subtokens.extend(part for part in piece.split("_") if part)

        if lowered not in STOPWORDS and len(lowered) > 1:
            tokens.append(lowered)
        # Emit sub-tokens too, but only when they add something new.
        if len(subtokens) > 1:
            for sub in subtokens:
                low = sub.lower()
                if len(low) > 1 and low not in STOPWORDS:
                    tokens.append(low)
    return tokens


@dataclass
class BM25Index:
    """A BM25-Okapi index over pre-tokenised documents."""

    k1: float = 1.5
    b: float = 0.75
    doc_ids: list[str] = field(default_factory=list)
    doc_tokens: list[Counter] = field(default_factory=list)
    doc_lengths: list[int] = field(default_factory=list)
    doc_frequency: Counter = field(default_factory=Counter)
    average_length: float = 0.0

    @property
    def size(self) -> int:
        return len(self.doc_ids)

    def add(self, doc_id: str, text: str) -> None:
        tokens = tokenize(text)
        counts = Counter(tokens)
        self.doc_ids.append(doc_id)
        self.doc_tokens.append(counts)
        self.doc_lengths.append(len(tokens))
        for term in counts:
            self.doc_frequency[term] += 1

    def finalise(self) -> None:
        total = sum(self.doc_lengths)
        self.average_length = (total / len(self.doc_lengths)) if self.doc_lengths else 0.0

    def _idf(self, term: str) -> float:
        n = self.size
        df = self.doc_frequency.get(term, 0)
        if df == 0:
            return 0.0
        # Okapi idf with the +1 smoothing that keeps values non-negative.
        return math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    def search(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        query_terms = [t for t in tokenize(query) if self.doc_frequency.get(t)]
        if not query_terms or not self.size:
            return []
        if self.average_length == 0.0:
            self.finalise()

        query_counts = Counter(query_terms)
        scores: list[tuple[str, float]] = []
        for position, doc_id in enumerate(self.doc_ids):
            counts = self.doc_tokens[position]
            length = self.doc_lengths[position] or 1
            score = 0.0
            for term, query_tf in query_counts.items():
                tf = counts.get(term, 0)
                if not tf:
                    continue
                idf = self._idf(term)
                denominator = tf + self.k1 * (1 - self.b + self.b * length / (self.average_length or 1))
                score += idf * (tf * (self.k1 + 1)) / denominator * (1 + math.log(query_tf))
            if score > 0:
                scores.append((doc_id, score))

        scores.sort(key=lambda item: (-item[1], item[0]))
        return scores[:top_k]


def build_bm25(
    documents: list[tuple[str, str]], k1: float = 1.5, b: float = 0.75
) -> BM25Index:
    index = BM25Index(k1=k1, b=b)
    for doc_id, text in documents:
        index.add(doc_id, text)
    index.finalise()
    return index
