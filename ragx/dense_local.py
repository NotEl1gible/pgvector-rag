"""HNSW in-process, via hnswlib.

This exists so the measurement can happen on a laptop with no Docker and no database. The
same sweep drives pgvector in CI, and the integration job asserts the two agree on recall --
if they ever disagree, one of them is not building the index it claims to build.

Filtering is where HNSW gets interesting, so both real strategies are implemented rather than
one:

- **pre-filter** hands hnswlib a predicate it applies DURING graph traversal. The walk still
  visits forbidden nodes, it just refuses to return them, so as the filter gets narrower the
  search burns more of its budget on candidates it must discard.
- **post-filter** searches normally and drops the misses afterwards. It is faster and it
  silently returns FEWER THAN k results, which is the failure mode that reaches production as
  "the answer was missing a source" rather than as an error.

Exact filtered search sits underneath both as the answer key.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


class LocalHnsw:
    name = "local"

    def __init__(self, dim: int, m: int = 16, ef_construction: int = 200,
                 space: str = "ip", seed: int = 100):
        # Inner product, because the vectors are L2-normalised on the way in -- so this IS
        # cosine, computed with one fewer operation. `test_embeddings_are_normalised` is what
        # keeps that substitution honest.
        self.dim, self.m, self.ef_construction, self.space = dim, m, ef_construction, space
        # Graph construction is randomised, so two indexes over identical vectors are not the
        # same graph. The A/A instrument rebuilds with a different seed to find out how much
        # of any recall difference is construction luck rather than a setting.
        self.seed = seed
        self._idx = None
        self.n = 0

    def build(self, vectors: np.ndarray, verbose: bool = False) -> LocalHnsw:
        import hnswlib
        v = np.ascontiguousarray(vectors, dtype=np.float32)
        self.n = v.shape[0]
        idx = hnswlib.Index(space=self.space, dim=self.dim)
        idx.init_index(max_elements=self.n, ef_construction=self.ef_construction,
                       M=self.m, random_seed=self.seed)
        idx.set_num_threads(2)                  # a laptop is not a build cluster
        idx.add_items(v, np.arange(self.n))
        self._idx = idx
        if verbose:
            print(f"  hnsw built: n={self.n} m={self.m} efc={self.ef_construction}")
        return self

    def set_ef(self, ef: int) -> None:
        # hnswlib requires ef >= k; the caller's k is small, but a sweep that starts below it
        # would silently clamp and flatten the interesting end of the curve.
        self._idx.set_ef(max(1, ef))

    def top(self, q: np.ndarray, k: int, allowed: np.ndarray | None = None,
            mode: str = "pre", oversample: int = 4) -> list[int]:
        q = np.asarray(q, dtype=np.float32).reshape(1, -1)
        if allowed is None:
            labels, _ = self._idx.knn_query(q, k=min(k, self.n))
            return [int(x) for x in labels[0]]
        if mode == "pre":
            mask = allowed
            labels, _ = self._idx.knn_query(q, k=min(k, int(mask.sum()) or 1),
                                            filter=lambda label: bool(mask[label]))
            return [int(x) for x in labels[0]]
        # post: search wide, then discard. Returns fewer than k whenever the filter is
        # narrow -- which is the point of measuring it.
        wide = min(self.n, k * oversample)
        labels, _ = self._idx.knn_query(q, k=wide)
        return [int(x) for x in labels[0] if allowed[int(x)]][:k]

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._idx.save_index(path)

    def load(self, path: str, max_elements: int) -> LocalHnsw:
        import hnswlib
        idx = hnswlib.Index(space=self.space, dim=self.dim)
        idx.load_index(path, max_elements=max_elements)
        idx.set_num_threads(2)
        self._idx, self.n = idx, max_elements
        return self
