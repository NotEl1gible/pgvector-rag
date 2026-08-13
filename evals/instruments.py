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
from ragx.fuse import fuse_top
from ragx.lexical import Lexical


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
        # hnswlib needs ef >= k to return k results, so a sweep value below k is silently
        # raised. Recorded rather than hidden: two rows of a table that ran at the same
        # effective setting must not look like two measurements.
        eff = max(ef, k)
        idx.set_ef(eff)
        recalls, times = [], []
        for q, want in zip(queries, truth, strict=True):
            got, ms = _timed(idx.top, q, k)
            recalls.append(M.recall_at_k(got, want, k))
            times.append(ms)
        rows.append({
            "ef_search": ef, "ef_effective": eff, "k": k, "m": m,
            "ef_construction": ef_construction,
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


# ----------------------------------------------------------------------------
# The A/A instrument -- the noise floor every other number has to clear
# ----------------------------------------------------------------------------
def aa_index(vectors: np.ndarray, queries: np.ndarray, ef: int, k: int = 10,
             m: int = 16, ef_construction: int = 200,
             seeds: tuple[int, int] = (100, 271)) -> dict:
    """Build the same index twice with different construction seeds and compare.

    HNSW construction is randomised, so two indexes over identical vectors are not the same
    graph. Nothing distinguishes these two runs except luck, which makes the gap between them
    the yardstick every tuning result has to clear. A repository that reports "m=32 gained 0.4
    points of recall" without this number has not shown that m did anything.
    """
    ex = Exact(vectors)
    truth = [ex.top(q, k) for q in queries]
    out = {}
    for tag, seed in zip(("a", "b"), seeds, strict=True):
        idx = LocalHnsw(vectors.shape[1], m, ef_construction, seed=seed).build(vectors)
        idx.set_ef(max(ef, k))
        rec = [M.recall_at_k(idx.top(q, k), want, k)
               for q, want in zip(queries, truth, strict=True)]
        out[tag] = rec
    d = M.paired_bootstrap(out["a"], out["b"])
    return {"recall_a": sum(out["a"]) / len(out["a"]),
            "recall_b": sum(out["b"]) / len(out["b"]),
            "gap": abs(sum(out["a"]) / len(out["a"]) - sum(out["b"]) / len(out["b"])),
            "test": d, "seeds": list(seeds), "ef_search": ef}


# ----------------------------------------------------------------------------
# Instruments that need real embeddings -- these run in CI
# ----------------------------------------------------------------------------
class Pipeline:
    """The hybrid pipeline, with ef_search left as a dial so the ablation can turn it."""

    def __init__(self, chunks: list[dict], vectors: np.ndarray, m: int = 16,
                 ef_construction: int = 200, rrf_k: int = 60, depth: int = 50):
        self.chunks, self.rrf_k, self.depth = chunks, rrf_k, depth
        self.lex = Lexical(chunks)
        self.exact = Exact(vectors)
        self.idx = LocalHnsw(vectors.shape[1], m, ef_construction).build(vectors)

    def rank(self, qv: np.ndarray, qtext: str, ef: int, k: int,
             flt: dict | None = None, dense_only: bool = False,
             lexical_only: bool = False, depth: int | None = None) -> list[str]:
        depth = depth or self.depth
        mask = mask_for(self.chunks, flt)
        if lexical_only:
            ids = self.lex.top(qtext, k, mask)
        elif dense_only:
            self.idx.set_ef(max(ef, depth))
            ids = self.idx.top(qv, k, mask, mode="pre")
        else:
            self.idx.set_ef(max(ef, depth))
            ids = fuse_top({"dense": self.idx.top(qv, depth, mask, mode="pre"),
                            "lexical": self.lex.top(qtext, depth, mask)}, k, self.rrf_k)
        return [self.chunks[i]["id"] for i in ids]


def _score(ranked: list[str], q: dict, k: int) -> dict:
    return {"hit": M.hit_at_k(ranked, q["relevant"], k),
            "rr": M.rr(ranked, q["relevant"]),
            "ndcg": M.ndcg_at_k(ranked, q["relevant"], k)}


def endtoend(pipe: Pipeline, Q: np.ndarray, queries: list[dict], ef_sweep: list[int],
             k: int = 5, depth: int | None = None, verbose: bool = True) -> list[dict]:
    """Does index recall loss reach the answer?

    The expected shape is that end-to-end quality is FLAT while index recall falls, and that
    is the argument the repository is built on: the metric everyone reports cannot see the
    thing that is degrading. Fusion and a top-k of five give the pipeline a great deal of room
    to absorb a missing neighbour, right up until the point where it cannot.
    """
    # hnswlib cannot return `depth` candidates with an ef below `depth`, so an ef sweep run at
    # the product's depth of 50 silently clamps every value under 50 to 50 -- which is exactly
    # what happened here first: ef 8, 16 and 32 all reported the SAME index recall because
    # they were all running at 50. The dial has to be able to move, so this instrument uses a
    # shallower depth and records the EFFECTIVE ef alongside the requested one.
    depth = depth or min(pipe.depth, 20)

    ex_ranked = {}
    for i, q in enumerate(queries):
        mask = mask_for(pipe.chunks, q.get("filter"))
        ids = pipe.exact.top(Q[i], depth, mask)
        fused = fuse_top({"dense": ids, "lexical": pipe.lex.top(q["text"], depth, mask)},
                         k, pipe.rrf_k)
        ex_ranked[q["id"]] = [pipe.chunks[j]["id"] for j in fused]

    rows = []
    base = [_score(ex_ranked[q["id"]], q, k) for q in queries]
    for ef in ef_sweep:
        eff = max(ef, depth)
        scored, recalls = [], []
        for i, q in enumerate(queries):
            mask = mask_for(pipe.chunks, q.get("filter"))
            pipe.idx.set_ef(eff)
            approx = pipe.idx.top(Q[i], depth, mask, mode="pre")
            recalls.append(M.recall_at_k(approx, pipe.exact.top(Q[i], depth, mask), depth))
            ranked = pipe.rank(Q[i], q["text"], ef, k, q.get("filter"), depth=depth)
            scored.append(_score(ranked, q, k))
        row = {"ef_search": ef, "ef_effective": eff, "depth": depth,
               "index_recall": sum(recalls) / len(recalls),
               "hit": sum(s["hit"] for s in scored) / len(scored),
               "mrr": sum(s["rr"] for s in scored) / len(scored),
               "ndcg": sum(s["ndcg"] for s in scored) / len(scored),
               "vs_exact": M.paired_bootstrap([s["rr"] for s in scored],
                                              [b["rr"] for b in base])}
        rows.append(row)
        if verbose:
            clamp = "" if eff == ef else f" (clamped to {eff} by depth {depth})"
            print(f"  ef {ef:>4}{clamp}  index recall {row['index_recall']:.4f}   "
                  f"hit@{k} {row['hit']:.3f}  mrr {row['mrr']:.3f}  "
                  f"ndcg {row['ndcg']:.3f}   vs exact pipeline "
                  f"{M.fmt_delta(row['vs_exact'])}")
    return rows


def rerank_compare(pipe: Pipeline, Q: np.ndarray, queries: list[dict], reranker,
                   ef: int, k: int = 5, shortlist: int = 50,
                   verbose: bool = True) -> list[dict]:
    """Cross-encoder over a shortlist, against simply returning more of the fused list.

    The comparison usually made is reranked-top-5 against fused-top-5, which flatters the
    reranker by giving it a larger candidate pool for free. The arm that matters is
    fused-top-{shortlist}: the same evidence, no model, no latency. If a reranker cannot beat
    that, it is buying precision that raising k already gave away.
    """
    # Every scored arm is scored AT THE SAME k. The first version scored the wide arm at
    # k=shortlist, so hit@25 was being compared against hit@5 and the wide arm "won" by
    # answering a different question. The shortlist is reported as a CEILING -- the best the
    # reranker could possibly reach given what it was handed -- and is deliberately kept out
    # of the significance family, because it is not a product anybody would ship.
    arms: dict[str, list[dict]] = {"fused@k": [], "reranked": []}
    times: dict[str, list[float]] = {"fused@k": [], "reranked": []}
    ceiling: list[float] = []
    by_id = {c["id"]: c for c in pipe.chunks}

    for i, q in enumerate(queries):
        flt = q.get("filter")
        ranked_k, t1 = _timed(pipe.rank, Q[i], q["text"], ef, k, flt)
        arms["fused@k"].append(_score(ranked_k, q, k))
        times["fused@k"].append(t1)

        wide, t2 = _timed(pipe.rank, Q[i], q["text"], ef, shortlist, flt)
        ceiling.append(M.hit_at_k(wide, q["relevant"], shortlist))

        rr_ids, t3 = _timed(reranker.top, q["text"], wide,
                            [by_id[x]["text"] for x in wide], k)
        arms["reranked"].append(_score(rr_ids, q, k))
        times["reranked"].append(t2 + t3)

    rows, tests = [], []
    base = [s["rr"] for s in arms["fused@k"]]
    for name, scored in arms.items():
        row = {"arm": name,
               "hit": sum(s["hit"] for s in scored) / len(scored),
               "mrr": sum(s["rr"] for s in scored) / len(scored),
               "ndcg": sum(s["ndcg"] for s in scored) / len(scored),
               "p50_ms": M.percentile(times[name], 0.50)}
        if name != "fused@k":
            row["test"] = M.paired_bootstrap([s["rr"] for s in scored], base)
            tests.append(row["test"]["p"])
        rows.append(row)
    rows.append({"arm": f"ceiling@{shortlist}", "hit": sum(ceiling) / len(ceiling),
                 "mrr": float("nan"), "ndcg": float("nan"), "p50_ms": 0.0,
                 "note": "upper bound: share of queries whose answer is in the shortlist "
                         "at all. The reranker cannot exceed this."})
    survives = M.holm(tests) if tests else []
    j = 0
    for row in rows:
        if "test" in row:
            row["survives"] = survives[j]
            j += 1
    if verbose:
        for row in rows:
            if "note" in row:
                print(f"  {row['arm']:<14} hit {row['hit']:.3f}   <- {row['note']}")
                continue
            extra = ("   " + M.fmt_delta(row["test"], row.get("survives"))
                     if "test" in row else "")
            print(f"  {row['arm']:<14} hit {row['hit']:.3f}  mrr {row['mrr']:.3f}  "
                  f"ndcg {row['ndcg']:.3f}  p50 {row['p50_ms']:.1f} ms{extra}")
    return rows


def query_type(pipe: Pipeline, Q: np.ndarray, queries: list[dict], ef: int,
               k: int = 5, verbose: bool = True) -> list[dict]:
    """Dense against lexical against hybrid, on the identifier/paraphrase TWINS.

    The twins share their target chunks and differ only in phrasing, so this is the claim
    "dense and lexical fail in opposite ways" measured on identical targets rather than
    asserted. If the two styles rank the same way, the hybrid is fusing two opinions from the
    same viewpoint and RRF is paying for nothing.
    """
    rows = []
    for style in ("identifier", "paraphrase", "versioned"):
        sub = [(i, q) for i, q in enumerate(queries) if q["style"] == style]
        if not sub:
            continue
        got: dict[str, list[float]] = {"dense": [], "lexical": [], "hybrid": []}
        for i, q in sub:
            flt = q.get("filter")
            got["dense"].append(M.rr(pipe.rank(Q[i], q["text"], ef, k, flt,
                                               dense_only=True), q["relevant"]))
            got["lexical"].append(M.rr(pipe.rank(Q[i], q["text"], ef, k, flt,
                                                 lexical_only=True), q["relevant"]))
            got["hybrid"].append(M.rr(pipe.rank(Q[i], q["text"], ef, k, flt),
                                      q["relevant"]))
        row = {"style": style, "n": len(sub),
               **{f"mrr_{key}": sum(v) / len(v) for key, v in got.items()},
               "dense_vs_lexical": M.paired_bootstrap(got["dense"], got["lexical"]),
               "hybrid_vs_best": M.paired_bootstrap(
                   got["hybrid"],
                   got["dense"] if sum(got["dense"]) >= sum(got["lexical"])
                   else got["lexical"])}
        rows.append(row)
        if verbose:
            print(f"  {style:<12} n={row['n']:<4} dense {row['mrr_dense']:.3f}  "
                  f"lexical {row['mrr_lexical']:.3f}  hybrid {row['mrr_hybrid']:.3f}")
            print(f"               dense vs lexical  {M.fmt_delta(row['dense_vs_lexical'])}")
            print(f"               hybrid vs better  {M.fmt_delta(row['hybrid_vs_best'])}")
    return rows
