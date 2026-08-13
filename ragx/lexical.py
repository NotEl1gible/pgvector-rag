"""BM25, and a tokenizer that does not destroy the thing BM25 is best at.

Lexical retrieval earns its place in a hybrid system by matching exact strings: error codes,
configuration keys, version numbers. A tokenizer that splits GATEWAY_RETRY_MAX_ATTEMPTS into
four common words throws that away and leaves BM25 doing a worse job of what dense retrieval
already does well -- at which point the hybrid has two halves that fail the same way, and RRF
is fusing two opinions from the same viewpoint.

So identifiers are emitted WHOLE and also split into parts. Whole gives the exact match its
full weight; the parts let a query saying "retry attempts" still reach the passage. This is
what a production analyzer's word-delimiter filter does, and it is the difference between the
identifier queries being a real test and being a formality.
"""
from __future__ import annotations

import re

import numpy as np

_TOKEN = re.compile(r"[A-Za-z0-9]+(?:[_\-.][A-Za-z0-9]+)*")
_SPLIT = re.compile(r"[_\-.]")


def tokenize(text: str) -> list[str]:
    out: list[str] = []
    for m in _TOKEN.findall(text.lower()):
        out.append(m)
        if _SPLIT.search(m):
            out.extend(p for p in _SPLIT.split(m) if p)
    return out


class Lexical:
    def __init__(self, chunks: list[dict]):
        from rank_bm25 import BM25Okapi
        self.n = len(chunks)
        self._bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])

    def scores(self, query: str) -> np.ndarray:
        return np.asarray(self._bm25.get_scores(tokenize(query)), dtype=np.float32)

    def top(self, query: str, k: int, allowed: np.ndarray | None = None) -> list[int]:
        s = self.scores(query)
        if allowed is not None:
            s = np.where(allowed, s, -np.inf)
        k = min(k, self.n)
        idx = np.argpartition(-s, k - 1)[:k]
        return [int(i) for i in idx[np.argsort(-s[idx])] if np.isfinite(s[int(i)])]
