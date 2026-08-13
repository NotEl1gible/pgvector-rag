"""Cross-encoder reranking, and the comparison that decides whether it is worth the latency.

A cross-encoder reads the query and the passage together, so it can weigh them against each
other in a way two independent embeddings cannot. It is also two orders of magnitude more
expensive per candidate, which is why it only ever runs over a shortlist.

The question the eval suite asks is not "does reranking improve the top 5" -- it usually does.
It is whether reranking a shortlist of 50 beats simply RETURNING more of the fused list, which
costs nothing. Plenty of pipelines pay for a reranker to recover precision they could have had
by raising k, and nobody checks, because the reranked number is compared against the same k
rather than against the same budget.

Lazy import and lazy construction: nothing here loads a model until a caller asks for one, so
importing this module costs nothing and the hermetic tests never touch ONNX.
"""
from __future__ import annotations

from .config import RERANK_MODEL


class CrossEncoder:
    name = "cross-encoder"

    def __init__(self, model: str = RERANK_MODEL, threads: int = 2):
        self.model_name = model
        self.threads = threads
        self._m = None

    def _load(self):
        if self._m is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder
            self._m = TextCrossEncoder(model_name=self.model_name, threads=self.threads)
        return self._m

    def rerank(self, query: str, docs: list[str]) -> list[float]:
        if not docs:
            return []
        return [float(s) for s in self._load().rerank(query, docs)]

    def top(self, query: str, candidates: list[int], texts: list[str],
            k: int) -> list[int]:
        scores = self.rerank(query, texts)
        order = sorted(range(len(candidates)), key=lambda i: -scores[i])
        return [candidates[i] for i in order[:k]]


class IdentityReranker:
    """Keeps the fused order. Used by the hermetic tests and by `--no-rerank`, and it is the
    honest control for the reranker: the arm that pays nothing and changes nothing."""

    name = "identity"

    def rerank(self, query: str, docs: list[str]) -> list[float]:
        return [float(len(docs) - i) for i in range(len(docs))]

    def top(self, query: str, candidates: list[int], texts: list[str],
            k: int) -> list[int]:
        return candidates[:k]


def build_reranker(enabled: bool):
    return CrossEncoder() if enabled else IdentityReranker()
