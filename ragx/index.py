"""The retriever: the product object the CLI and the API both drive.

The one thing here that other retrieval services do not do is `recall_estimate`. A search API
that returns passages and nothing else is asking the caller to assume the index found what was
there. This one answers with what it gave up, and says where that number came from:

- `recall_measured_now=True` -- exact search was run for this query and the recall is that
  query's actual recall. Affordable here because the corpus is small, and the README is clear
  that this is a property of the corpus size, not a clever trick.
- `recall_measured_now=False` -- the number is read off the committed frontier for the
  `ef_search` in use. It is the mean over the eval queries, so it describes the setting, not
  this query.

The distinction is in the response rather than in a footnote, because a caller that cannot
tell a measurement from an average will eventually treat one as the other.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import Settings
from .dense_local import LocalHnsw
from .embed import Embedder, build_embedder, embed_corpus
from .exact import Exact, mask_for
from .fuse import fuse_top
from .lexical import Lexical
from .rerank import build_reranker
from .schemas import Hit, SearchResponse
from .tracing import StageClock, Tracing

FRONTIER_PATH = "artifacts/frontier.json"


def _choose_ef(rows: list[dict], target: float, fallback: int) -> tuple[int, float | None]:
    ok = [r for r in rows if r["recall"] >= target]
    if ok:
        best = min(ok, key=lambda r: r["ef_search"])
        return best["ef_search"], best["recall"]
    if rows:
        best = max(rows, key=lambda r: r["recall"])
        return best["ef_search"], best["recall"]
    return fallback, None


class Retriever:
    def __init__(self, s: Settings, chunks: list[dict], vectors: np.ndarray,
                 embedder: Embedder, tracing: Tracing | None = None,
                 frontier: list[dict] | None = None):
        self.s = s
        self.chunks = chunks
        self.vectors = vectors
        self.embedder = embedder
        self.tracing = tracing
        self.frontier = frontier or []
        self.exact = Exact(vectors)
        self.lexical = Lexical(chunks)
        self.dense = LocalHnsw(vectors.shape[1], s.hnsw_m, s.hnsw_ef_construction)
        self.dense.build(vectors)
        self.reranker = build_reranker(s.rerank)

    # ------------------------------------------------------------------
    def search(self, query: str, k: int | None = None,
               recall_target: float | None = None,
               flt: dict[str, str] | None = None,
               rerank: bool | None = None,
               verify: bool = False) -> SearchResponse:
        s = self.s
        k = k or s.top_k
        want_rerank = s.rerank if rerank is None else rerank
        clock = StageClock(self.tracing)

        target = recall_target if recall_target is not None else s.recall_target
        ef, curve_recall = _choose_ef(self.frontier, target, s.ef_search)

        with clock.stage("embed_query"):
            qv = self.embedder.encode([query])[0]

        mask = mask_for(self.chunks, flt)
        self.dense.set_ef(max(ef, s.depth))
        with clock.stage("dense", ef_search=ef, filtered=mask is not None):
            dense_ids = self.dense.top(qv, s.depth, mask, mode="pre")
        with clock.stage("lexical"):
            lex_ids = self.lexical.top(query, s.depth, mask)
        with clock.stage("fuse"):
            fused = fuse_top({"dense": dense_ids, "lexical": lex_ids},
                             s.rerank_depth if want_rerank else k, s.rrf_k)

        if want_rerank:
            with clock.stage("rerank", candidates=len(fused)):
                fused = self.reranker.top(query, fused,
                                          [self.chunks[i]["text"] for i in fused], k)
        final = fused[:k]

        recall, measured = curve_recall, False
        if verify:
            # Honest about its own cost: this doubles the work by running the thing the index
            # exists to avoid. It is offered because at this corpus size it is milliseconds,
            # and refused at scale rather than quietly approximated.
            with clock.stage("verify_recall"):
                truth = self.exact.top(qv, s.depth, mask)
                hit = len(set(truth) & set(dense_ids))
                recall = hit / max(1, len(truth))
                measured = True

        hits = [Hit(chunk_id=self.chunks[i]["id"], rank=r + 1, score=0.0,
                    source="rerank" if want_rerank else "rrf",
                    title=self.chunks[i]["title"], text=self.chunks[i]["text"])
                for r, i in enumerate(final)]
        return SearchResponse(query=query, hits=hits, backend="local", ef_search=ef,
                              recall_estimate=recall, recall_measured_now=measured,
                              filtered=mask is not None, timings=clock.stages,
                              total_ms=clock.total_ms)


# ----------------------------------------------------------------------------
def load_frontier(path: str = FRONTIER_PATH) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    d = json.loads(p.read_text(encoding="utf-8"))
    return d.get("rows", d) if isinstance(d, dict) else d


def build_retriever(s: Settings, chunks: list[dict], tracing: Tracing | None = None,
                    verbose: bool = True) -> Retriever:
    emb = build_embedder(s)
    cache = f"{s.index_dir}/vectors-{emb.name}.npy"
    v = embed_corpus(emb, chunks, cache=cache, batch=s.embed_batch, verbose=verbose,
                     throttle=s.embed_throttle_s)
    return Retriever(s, chunks, v, emb, tracing, load_frontier())
