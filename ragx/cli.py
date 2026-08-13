"""Command line for the retriever and its instruments.

`corpus` comes first and must exist before any retriever does, so no query can have been
chosen after seeing a score.

One rule runs through the rest of it: **nothing starts a multi-minute CPU job without being
asked twice.** The ONNX embedder is the only expensive step in the repository, and a retrieval
demo that pins someone's laptop for minutes is a demo they cancel. It is cached after the
first pass, and the first pass requires `--yes`.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from evals import instruments as I

from . import corpus as C
from .config import get_settings
from .embed import build_embedder, embed_corpus
from .index import FRONTIER_PATH, Retriever, load_frontier
from .tracing import build as build_tracing


def _settings(a, **over):
    kw = {}
    if getattr(a, "embedder", None):
        kw["embedder"] = a.embedder
    kw.update(over)
    return get_settings(**kw)


def _vectors(a, chunks, s, allow_heavy: bool = False):
    """Load or compute the corpus vectors, refusing to start the slow path unasked."""
    emb = build_embedder(s)
    cache = Path(f"{s.index_dir}/vectors-{emb.name}.npy")
    if emb.name == "onnx" and not cache.exists() and not allow_heavy:
        print(f"embedding {len(chunks)} chunks with {emb.name} would take a few minutes of "
              f"CPU\nand has not been cached yet. Re-run with --yes to allow it, or use "
              f"--embedder hash\nfor the index instruments, which need no semantics at all.",
              file=sys.stderr)
        raise SystemExit(3)
    return emb, embed_corpus(emb, chunks, cache=str(cache), batch=s.embed_batch,
                             throttle=s.embed_throttle_s)


# ----------------------------------------------------------------------------
def cmd_corpus(a) -> int:
    s = get_settings()
    chunks = C.build_corpus(n_facts_per_section=a.facts, seed=s.seed, n_services=a.services)
    queries = C.build_queries(chunks, n=a.queries, seed=s.seed + 4)

    # The invariant the corpus lives or dies by. A fact key repeated inside one version means
    # two chunks state different values for the same thing, and then no retrieval result can
    # be called wrong -- whichever comes back, another chunk contradicts it.
    inside = Counter((c["service"], c["version"], c["fact"]["key"]) for c in chunks)
    worst = max(inside.values())
    across = Counter(len({x["version"] for x in chunks if x["service"] == c["service"]
                          and x["fact"]["key"] == c["fact"]["key"]}) for c in chunks)

    print(f"chunks   {len(chunks)}   mean {sum(len(c['text']) for c in chunks) // len(chunks)}"
          f" chars   services {len({c['service'] for c in chunks})} of {len(C.SERVICES)}  "
          f"versions {len(C.VERSIONS)}  sections {len(C.SECTIONS)}")
    print(f"queries  {len(queries)}   " + "  ".join(
        f"{k} {v}" for k, v in sorted(Counter(q['style'] for q in queries).items())))
    print(f"\nkey unique within a version: {'YES' if worst == 1 else 'NO'} "
          f"(max {worst} chunks share one)")
    print(f"versions per key (the designed duplication): "
          f"{dict(sorted(Counter(across.elements()).items()))}")
    sizes = {st: dict(sorted(Counter(len(q["relevant"]) for q in queries
                                     if q["style"] == st).items()))
             for st in ("identifier", "paraphrase", "versioned")}
    print(f"relevant-set sizes: {sizes}")
    if worst != 1:
        print("GATE FAILED: the corpus contradicts itself", file=sys.stderr)
        return 1
    if a.write:
        cp, qp = C.write(chunks, queries)
        print(f"\nwrote {cp} and {qp}")
    return 0


def cmd_frontier(a) -> int:
    """Recall against exact search across ef_search. Needs no labels and no semantics."""
    s = _settings(a)
    chunks, queries = C.load()
    emb, V = _vectors(a, chunks, s, a.yes)
    Q = emb.encode([q["text"] for q in queries[:a.limit or len(queries)]])
    print(f"n={V.shape[0]} vectors, dim={V.shape[1]}, {Q.shape[0]} queries, k={a.k}, "
          f"embedder={emb.name}, m={s.hnsw_m}, ef_construction={s.hnsw_ef_construction}\n")
    rows = I.frontier(V, Q, s.ef_sweep, k=a.k, m=s.hnsw_m,
                      ef_construction=s.hnsw_ef_construction)
    base = I.exact_baseline_ms(V, Q, k=a.k)
    chosen = I.choose_ef(rows, s.recall_target)
    best = max(r["recall"] for r in rows)
    print(f"\nexact brute force p50 {base:.3f} ms   index build "
          f"{rows[0]['build_ms']:.0f} ms")
    print(f"smallest ef_search meeting recall {s.recall_target}: {chosen}")
    print("\nThe curve is the product: past the point where it flattens, a larger ef buys")
    print("recall that is already there and pays for it on every query forever. And at this")
    print("corpus size exact search is milliseconds, so the honest reading is that ANN is")
    print("not yet necessary here -- the METHOD transfers, these particular numbers do not.")

    Path(s.index_dir).mkdir(parents=True, exist_ok=True)
    Path(FRONTIER_PATH).write_text(json.dumps(
        {"embedder": emb.name, "n": int(V.shape[0]), "k": a.k, "rows": rows,
         "exact_p50_ms": base}, indent=1), encoding="utf-8")
    print(f"\nwrote {FRONTIER_PATH}")

    if best >= 1.0 - 1e-9 and min(r["recall"] for r in rows) >= 1.0 - 1e-9:
        print("GATE FAILED: recall is 1.000 at every ef_search -- the index is not actually "
              "approximate at this size, so the sweep measures nothing", file=sys.stderr)
        return 1
    return 0


def cmd_filtered(a) -> int:
    """Pre-filter against post-filter against exact filtered search."""
    s = _settings(a)
    chunks, queries = C.load()
    emb, V = _vectors(a, chunks, s, a.yes)
    sub = queries[:a.limit or len(queries)]
    Q = emb.encode([q["text"] for q in sub])
    n_f = sum(1 for q in sub if q.get("filter"))
    print(f"{n_f} of {len(sub)} queries carry a metadata filter, ef_search={s.ef_search}\n")
    rows = I.filtered(V, Q, chunks, [q.get("filter") for q in sub], k=a.k, ef=s.ef_search,
                      m=s.hnsw_m, ef_construction=s.hnsw_ef_construction)
    print("\npre-filter applies the predicate DURING traversal: it never returns short, and")
    print("as the filter narrows it spends more of its budget on candidates it must discard.")
    print("post-filter searches normally and drops the misses, so it silently returns FEWER")
    print("THAN K -- which reaches production as a missing source, not as an error.")
    by = {r["mode"]: r for r in rows}
    if by["pre"]["short_rate"] > 0:
        print("GATE FAILED: pre-filtering returned short; it is not supposed to be able to",
              file=sys.stderr)
        return 1
    return 0


def cmd_search(a) -> int:
    s = _settings(a, rerank=a.rerank)
    chunks, _ = C.load()
    emb, V = _vectors(a, chunks, s, a.yes)
    tracing = build_tracing(s) if s.otlp_endpoint else None
    r = Retriever(s, chunks, V, emb, tracing, load_frontier())
    flt = None
    if a.filter:
        flt = dict(kv.split("=", 1) for kv in a.filter.split(","))
    out = r.search(a.query, k=a.k, recall_target=a.recall_target, flt=flt,
                   rerank=a.rerank, verify=a.verify)
    print(f"\n{a.query}\n")
    for h in out.hits:
        print(f"  {h.rank}. {h.chunk_id:<28} {h.title}")
    src = "measured for this query" if out.recall_measured_now else "from the committed curve"
    rec = "n/a" if out.recall_estimate is None else f"{out.recall_estimate:.3f}"
    print(f"\n  ef_search {out.ef_search}   recall {rec} ({src})   "
          f"{'filtered' if out.filtered else 'unfiltered'}")
    print("  " + "  ".join(f"{t.stage} {t.ms:.2f}ms" for t in out.timings)
          + f"   total {out.total_ms:.2f}ms")
    return 0


def cmd_build(a) -> int:
    """Compute and cache the corpus vectors. The only expensive command in the repository."""
    s = _settings(a)
    chunks, _ = C.load()
    emb, V = _vectors(a, chunks, s, a.yes)
    print(f"vectors ready: {V.shape} embedder={emb.name} "
          f"cached at {s.index_dir}/vectors-{emb.name}.npy")
    print(f"norms {float(np.linalg.norm(V, axis=1).min()):.4f} .. "
          f"{float(np.linalg.norm(V, axis=1).max()):.4f}   (must be 1.0: exact search, "
          f"hnswlib inner product and pgvector all assume it)")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(prog="ragx", description=__doc__.split("\n")[0])
    ap.add_argument("--embedder", choices=["hash", "onnx"], default="")
    ap.add_argument("--yes", action="store_true",
                    help="allow the slow ONNX embedding pass on first run")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("corpus", help="generate the corpus and its queries")
    p.add_argument("--facts", type=int, default=14)
    p.add_argument("--services", type=int, default=C.DEFAULT_SERVICES)
    p.add_argument("--queries", type=int, default=180)
    p.add_argument("--write", action="store_true")
    p.set_defaults(fn=cmd_corpus)

    p = sub.add_parser("build", help="compute and cache the corpus vectors")
    p.set_defaults(fn=cmd_build)

    p = sub.add_parser("frontier", help="HNSW recall against exact, across ef_search")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--limit", type=int, default=0)
    p.set_defaults(fn=cmd_frontier)

    p = sub.add_parser("filtered", help="pre-filter vs post-filter vs exact filtered")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--limit", type=int, default=0)
    p.set_defaults(fn=cmd_filtered)

    p = sub.add_parser("search", help="one query through the product")
    p.add_argument("query")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--recall-target", dest="recall_target", type=float, default=None)
    p.add_argument("--filter", default="", help="service=gateway,version=2.4")
    p.add_argument("--rerank", action="store_true")
    p.add_argument("--verify", action="store_true",
                   help="run exact search too and report this query's real recall")
    p.set_defaults(fn=cmd_search)

    a = ap.parse_args()
    sys.exit(a.fn(a) or 0)


if __name__ == "__main__":
    main()
