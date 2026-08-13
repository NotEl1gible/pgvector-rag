"""Exact brute-force search: the answer key every approximate number is scored against.

This is the quiet foundation of the whole repository. Approximate indexes are normally
evaluated against human relevance labels, which conflates two different questions -- did the
index find the nearest vectors, and were the nearest vectors the right passages. Those have to
be separated, because only the first one is the index's fault. An HNSW index can have terrible
recall against exact search while the end-to-end numbers look fine, and it can have perfect
recall while the answers are wrong; treating them as one number hides both.

At this corpus size a full scan is one matrix multiply over 12k by 384 floats: milliseconds,
and exactly right. So index recall is measured with no labels at all, and the qrels are kept
for the separate question of whether the pipeline answers correctly.

That freedom is also the project's honest limit, stated in the README rather than implied: a
corpus small enough for exact search is a corpus that does not need ANN. The METHOD transfers;
these particular numbers do not.
"""
from __future__ import annotations

import numpy as np


class Exact:
    def __init__(self, vectors: np.ndarray):
        # Vectors arrive L2-normalised, so a dot product IS cosine similarity.
        self.v = np.ascontiguousarray(vectors, dtype=np.float32)
        self.n = self.v.shape[0]

    def scores(self, q: np.ndarray) -> np.ndarray:
        return self.v @ np.asarray(q, dtype=np.float32).ravel()

    def top(self, q: np.ndarray, k: int, allowed: np.ndarray | None = None) -> list[int]:
        s = self.scores(q)
        if allowed is not None:
            s = np.where(allowed, s, -np.inf)
        k = min(k, self.n)
        idx = np.argpartition(-s, k - 1)[:k]
        return [int(i) for i in idx[np.argsort(-s[idx])] if np.isfinite(s[int(i)])]


def mask_for(chunks: list[dict], flt: dict[str, str] | None) -> np.ndarray | None:
    """A boolean mask over the corpus.

    `None` means unfiltered, which is deliberately NOT the same as a mask that happens to be
    all True: the difference decides whether an index takes its filtered code path at all, and
    the filtered-ANN instrument exists to measure exactly that path.
    """
    if not flt:
        return None
    m = np.ones(len(chunks), dtype=bool)
    for key, val in flt.items():
        m &= np.array([c.get(key) == val for c in chunks], dtype=bool)
    return m
