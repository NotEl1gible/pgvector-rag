"""Retrieval scoring, and the intervals that keep it honest.

Two families of number live here and they must not be confused, because they answer different
questions and only one of them is the index's responsibility:

- **Index recall** compares an approximate result against EXACT search on the same vectors.
  It needs no labels and asks only "did the index find the nearest neighbours".
- **Retrieval quality** compares a result against the qrels. It needs labels and asks "were
  the nearest neighbours the right passages".

An index can score badly on the first and fine on the second, which is exactly the finding the
`endtoend` instrument goes looking for. Reporting one number for both would hide it.

The comparison discipline is carried over from the sibling projects: arms run over the same
queries, so every interval is a bootstrap of the DIFFERENCE, a family of comparisons gets a
Holm correction, and an interval containing zero prints INCONCLUSIVE rather than a winner.
"""
from __future__ import annotations

import math
import random


# ---------------------------------------------------------------- index recall (no labels)
def recall_at_k(approx: list[int], exact: list[int], k: int | None = None) -> float:
    """Fraction of the exact top-k that the approximate search also returned.

    Order within the top-k is deliberately ignored. An index that returns the right k
    documents in a different order has lost nothing a reranker or a fusion step cannot fix;
    an index that returns different documents has lost information nothing downstream can
    recover.
    """
    k = k or len(exact)
    if k == 0:
        return 1.0
    want = set(exact[:k])
    return len(want & set(approx[:k])) / len(want)


# ---------------------------------------------------------------- retrieval quality (labels)
def hit_at_k(ranked_ids: list[str], relevant: list[str], k: int) -> float:
    """Did anything relevant make the top k. For a query whose relevant set spans three
    versions this asks 'did you find the topic'."""
    return 1.0 if set(ranked_ids[:k]) & set(relevant) else 0.0


def recall_labelled(ranked_ids: list[str], relevant: list[str], k: int) -> float:
    if not relevant:
        return 1.0
    return len(set(ranked_ids[:k]) & set(relevant)) / len(relevant)


def rr(ranked_ids: list[str], relevant: list[str]) -> float:
    rel = set(relevant)
    for i, doc in enumerate(ranked_ids):
        if doc in rel:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(ranked_ids: list[str], relevant: list[str], k: int) -> float:
    rel = set(relevant)
    dcg = sum(1.0 / math.log2(i + 2) for i, d in enumerate(ranked_ids[:k]) if d in rel)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(rel), k)))
    return dcg / ideal if ideal else 0.0


# ---------------------------------------------------------------- rates and intervals
def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    z, p = 1.96, k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def rate(k: int, n: int) -> str:
    lo, hi = wilson(k, n)
    return f"{(k / n if n else 0):.2f} ({k}/{n}) [{lo:.2f}, {hi:.2f}]"


def paired_bootstrap(a: list[float], b: list[float], trials: int = 4000,
                     seed: int = 0) -> dict:
    """Bootstrap the mean per-query DIFFERENCE, a minus b. Positive means a is better.

    The arms answer the same queries, so they are not independent samples. An independent
    interval per arm is far wider than the evidence warrants and is the standard way a
    retrieval ablation ends in "the intervals overlap, we cannot say".
    """
    n = len(a)
    if n == 0 or len(b) != n:
        raise ValueError("paired bootstrap needs two equal-length sequences")
    rng = random.Random(seed)
    point = sum(x - y for x, y in zip(a, b, strict=True)) / n
    deltas = []
    for _ in range(trials):
        idx = [rng.randrange(n) for _ in range(n)]
        deltas.append(sum(a[i] - b[i] for i in idx) / n)
    deltas.sort()
    lo = deltas[int(0.025 * trials)]
    hi = deltas[min(trials - 1, int(0.975 * trials))]
    below = sum(1 for d in deltas if d <= 0.0) / trials
    above = sum(1 for d in deltas if d >= 0.0) / trials
    return {"delta": point, "lo": lo, "hi": hi, "n": n,
            "p": min(1.0, 2 * min(below, above)),
            "verdict": "INCONCLUSIVE" if lo <= 0.0 <= hi
                       else ("a better" if point > 0 else "b better")}


def holm(pvalues: list[float], alpha: float = 0.05) -> list[bool]:
    """Holm-Bonferroni across a FAMILY of comparisons.

    Not a formality. Comparing several arms against one baseline at 95% produces roughly one
    spurious winner per run by design, and in a sibling project one duly appeared in an arm
    that was configurationally identical to the baseline. Holm rather than plain Bonferroni
    because it is uniformly more powerful at the same family-wise error rate.
    """
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    m = len(pvalues)
    out = [False] * m
    for rank, i in enumerate(order):
        if pvalues[i] <= alpha / (m - rank):
            out[i] = True
        else:
            break
    return out


def fmt_delta(d: dict, survives: bool | None = None) -> str:
    s = f"{d['delta']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}]  p={d['p']:.3f}  "
    if survives is None or d["verdict"] == "INCONCLUSIVE":
        return s + d["verdict"]
    return s + (d["verdict"] if survives else "n.s. after Holm")


def percentile(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    i = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[i]
