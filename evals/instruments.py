"""The instruments. The first two need no semantics and no labels at all.

That is worth stating plainly because it decides where they can run. "Did the index return the
same neighbours brute force would have" is a question about vector geometry and graph
traversal, not about meaning -- so `frontier` and `filtered` are exactly as valid on cheap
deterministic vectors as on a 130 MB transformer, and they run on any machine in seconds.

Only `rerank`, `querytype` and `endtoend` need real embeddings, because only those ask whether
the nearest neighbours were the RIGHT passages. Those run in CI, on a known machine, and the
README says which numbers came from where.
"""
from __future__ import annotations

import time

import numpy as np

from evals import metrics as M
from ragx.dense_local import LocalHnsw
from ragx.exact import Exact, mask_for


def _timed(fn, *a, **kw):
    t0 = time.perf_counter()
    out = fn(*a, **kw)
    return out, (time.perf_counter() - t0) * 1000


def frontier(vectors: np.ndarray, queries: np.ndarray, ef_sweep: list[int], k: int = 10,
             m: int = 16, ef_construction: int = 200,
             verbose: bool = True) -> list[dict]:
    """Recall against exact search across ef_search, with the latency each setting costs.

    Recall is the machine-independent axis and leads the table; latency is reported beside it
    with the machine named, because a number measured on a shared vCPU is a number about that
    vCPU as much as about the index.
    """
    ex = Exact(vectors)
    truth = [ex.top(q, k) for q in queries]

    idx, build_ms = _timed(lambda: LocalHnsw(vectors.shape[1], m, ef_construction)
                           .build(vectors))
    rows = []
    for ef in ef_sweep:
        idx.set_ef(max(ef, k))
        recalls, times = [], []
        for q, want in zip(queries, truth, strict=True):
            got, ms = _timed(idx.top, q, k)
            recalls.append(M.recall_at_k(got, want, k))
            times.append(ms)
        rows.append({
            "ef_search": ef, "k": k, "m": m, "ef_construction": ef_construction,
            "recall": sum(recalls) / len(recalls),
            "recall_min": min(recalls),
            "perfect_share": sum(1 for r in recalls if r >= 1.0) / len(recalls),
            "p50_ms": M.percentile(times, 0.50), "p95_ms": M.percentile(times, 0.95),
            "build_ms": build_ms, "n": vectors.shape[0], "queries": len(queries),
        })
        if verbose:
            r = rows[-1]
            print(f"  ef_search {ef:>4}  recall@{k} {r['recall']:.4f}  "
                  f"worst {r['recall_min']:.2f}  perfect {r['perfect_share']:.2f}  "
                  f"p50 {r['p50_ms']:.3f} ms  p95 {r['p95_ms']:.3f} ms")
    return rows


def exact_baseline_ms(vectors: np.ndarray, queries: np.ndarray, k: int = 10) -> float:
    """What the approximation is being bought instead of. Without this column the frontier
    shows a cost with nothing to compare it to -- and at this corpus size brute force is often
    the honest answer, which the README says out loud."""
    ex = Exact(vectors)
    times = [_timed(ex.top, q, k)[1] for q in queries]
    return M.percentile(times, 0.50)


def choose_ef(rows: list[dict], target: float) -> int:
    """The smallest ef_search whose measured recall meets the target.

    Smallest, not safest. Past the point where the curve flattens, a larger ef buys recall
    that is already there and pays latency for it every query forever -- and that plateau is
    invisible in any single number, which is why the product reads this off the curve instead
    of shipping a default somebody liked the look of.
    """
    ok = [r for r in rows if r["recall"] >= target]
    return min((r["ef_search"] for r in ok), default=max(r["ef_search"] for r in rows))


def filtered(vectors: np.ndarray, queries: np.ndarray, chunks: list[dict],
             filters: list[dict | None], k: int = 10, ef: int = 64,
             m: int = 16, ef_construction: int = 200, verbose: bool = True) -> list[dict]:
    """Pre-filter against post-filter against exact filtered search, by selectivity.

    The failure this looks for is not slowness. Post-filtering returns fewer than k results
    whenever the filter is narrow, and a pipeline that asked for five passages and silently
    got two produces a confident answer built on less evidence than it was designed for.
    """
    ex = Exact(vectors)
    idx = LocalHnsw(vectors.shape[1], m, ef_construction).build(vectors)
    idx.set_ef(max(ef, k))
    rows = []
    for mode in ("pre", "post"):
        # Scored over the FILTERED queries only. An unfiltered query takes the same code path
        # in both modes, so folding those in dilutes the effect by whatever share of the set
        # happens to carry no filter -- which would have reported a 30% short-return rate as
        # 10% purely because two thirds of the queries were not being filtered at all.
        recalls, times, short, kept = [], [], 0, []
        for q, flt in zip(queries, filters, strict=True):
            if not flt:
                continue
            mask = mask_for(chunks, flt)
            want = ex.top(q, k, mask)
            got, ms = _timed(idx.top, q, k, mask, mode)
            recalls.append(M.recall_at_k(got, want, k))
            times.append(ms)
            short += 1 if len(got) < min(k, len(want)) else 0
            kept.append(float(mask.mean()))
        n = len(recalls)
        rows.append({
            "mode": mode, "k": k, "ef_search": ef,
            "recall": sum(recalls) / n if n else 1.0,
            "returned_short": short, "n_filtered": n,
            "short_rate": short / n if n else 0.0,
            "p50_ms": M.percentile(times, 0.50), "p95_ms": M.percentile(times, 0.95),
            "mean_selectivity": sum(kept) / n if n else 1.0,
        })
        if verbose:
            r = rows[-1]
            print(f"  {mode:<5} recall {r['recall']:.4f}   returned fewer than k "
                  f"{M.rate(r['returned_short'], r['n_filtered'])}   "
                  f"p50 {r['p50_ms']:.3f} ms   selectivity "
                  f"{r['mean_selectivity']:.3f}")
    return rows
