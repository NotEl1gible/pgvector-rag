"""Command line. Grows one subcommand per build step; `corpus` is the first and must exist
before any retriever does, so that no query can have been chosen after seeing a score."""
from __future__ import annotations

import argparse
import sys
from collections import Counter

from . import corpus as C
from .config import get_settings


def cmd_corpus(a) -> int:
    s = get_settings()
    chunks = C.build_corpus(n_facts_per_section=a.facts, seed=s.seed)
    queries = C.build_queries(chunks, n=a.queries, seed=s.seed + 4)

    # The invariant the corpus lives or dies by. A fact key repeated inside one version means
    # two chunks state different values for the same thing, and then no retrieval result can
    # be called wrong -- whichever one comes back, another chunk contradicts it.
    inside = Counter((c["service"], c["version"], c["fact"]["key"]) for c in chunks)
    worst = max(inside.values())
    across = Counter(len({x["version"] for x in chunks
                          if x["service"] == c["service"]
                          and x["fact"]["key"] == c["fact"]["key"]}) for c in chunks)

    print(f"chunks   {len(chunks)}   mean {sum(len(c['text']) for c in chunks) // len(chunks)}"
          f" chars   services {len(C.SERVICES)}  versions {len(C.VERSIONS)}  "
          f"sections {len(C.SECTIONS)}")
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
    print("  identifier and paraphrase are TWINS on the same targets, so the lexical/dense")
    print("  comparison changes only the phrasing. versioned always has exactly one answer,")
    print("  and its two near-identical siblings differ only in the number.")

    if worst != 1:
        print("GATE FAILED: the corpus contradicts itself", file=sys.stderr)
        return 1
    if a.write:
        cp, qp = C.write(chunks, queries)
        print(f"\nwrote {cp} and {qp}")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(prog="ragx", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("corpus", help="generate the corpus and its queries")
    p.add_argument("--facts", type=int, default=28)
    p.add_argument("--queries", type=int, default=300)
    p.add_argument("--write", action="store_true")
    p.set_defaults(fn=cmd_corpus)

    a = ap.parse_args()
    sys.exit(a.fn(a) or 0)


if __name__ == "__main__":
    main()
