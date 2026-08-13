"""Reciprocal Rank Fusion.

RRF combines rankings using positions only, never scores, and that restraint is the point.
BM25 returns unbounded scores whose scale depends on corpus statistics; cosine similarity is
bounded in a range where a "good" match might be 0.75. Normalising those onto a common scale
requires a choice -- min-max over the returned window, z-scores, a fitted mapping -- and every
such choice is a hyperparameter that shifts when the corpus does. Rank position is already
comparable, so there is nothing to tune and nothing to drift.

The cost is real and worth naming: RRF cannot tell a runaway winner from a narrow one. A
passage that BM25 scores at ten times the runner-up contributes exactly what a passage
scraping ahead by a hair contributes. That is the trade, and it is why `k` is the only knob.
"""
from __future__ import annotations


def rrf(rank_lists: dict[str, list[int]], k: int = 60,
        weights: dict[str, float] | None = None) -> list[tuple[int, float]]:
    """`rank_lists` maps a retriever name to its ranked document indices, best first."""
    weights = weights or {}
    acc: dict[int, float] = {}
    for name, ranked in rank_lists.items():
        w = weights.get(name, 1.0)
        for pos, doc in enumerate(ranked):
            acc[doc] = acc.get(doc, 0.0) + w / (k + pos + 1)
    # Ties broken by document index so the output is deterministic. Without this, two runs
    # over the same data can disagree on rank order, and a measured difference between arms
    # becomes partly dictionary ordering.
    return sorted(acc.items(), key=lambda kv: (-kv[1], kv[0]))


def fuse_top(rank_lists: dict[str, list[int]], k: int, rrf_k: int = 60,
             weights: dict[str, float] | None = None) -> list[int]:
    return [doc for doc, _ in rrf(rank_lists, rrf_k, weights)[:k]]
