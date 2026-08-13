"""Hermetic. No model download, no container, no network, and nothing that pins a CPU.

Everything here runs on the hash embedder over a slice of the corpus, which is the same
choice the instruments make for the same reason: index geometry has nothing to do with
meaning, so the properties worth asserting hold on any vectors.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pytest
from hypothesis import given
from hypothesis import settings as hyp_settings
from hypothesis import strategies as st

from evals import instruments as I
from evals import metrics as M
from ragx import corpus as C
from ragx.config import get_settings
from ragx.dense_local import LocalHnsw
from ragx.embed import HashEmbedder
from ragx.exact import Exact, mask_for
from ragx.fuse import fuse_top, rrf
from ragx.lexical import Lexical, tokenize

CHUNKS, QUERIES = C.load()
SMALL = CHUNKS[:600]
EMB = HashEmbedder()
V_SMALL = EMB.encode([c["text"] for c in SMALL])


# ---------------------------------------------------------------- the corpus
def test_the_corpus_never_contradicts_itself():
    """The invariant the whole ground truth rests on. Two chunks sharing a fact key inside one
    version would state different values for the same thing, and then NO retrieval result can
    be called wrong -- whichever comes back, another chunk says otherwise."""
    inside = Counter((c["service"], c["version"], c["fact"]["key"]) for c in CHUNKS)
    assert max(inside.values()) == 1


def test_the_designed_duplication_survives():
    """The near-duplicates across versions are the point, not an accident. If they vanished,
    the versioned queries would stop being a trap and the filter would stop mattering."""
    across = Counter(len({x["version"] for x in CHUNKS if x["service"] == c["service"]
                          and x["fact"]["key"] == c["fact"]["key"]}) for c in CHUNKS)
    assert across[3] > 0, "no fact appears at three versions"


def test_the_corpus_regenerates_byte_for_byte():
    s = get_settings()
    again = C.build_corpus(n_facts_per_section=s.corpus_facts_per_section, seed=s.seed,
                           n_services=s.corpus_services)
    assert C.corpus_hash(again) == C.corpus_hash(CHUNKS)


def test_qrels_are_derived_not_assumed():
    """The bug this file exists to keep dead: 'what is the default value of X' has one right
    answer per version, and an early version labelled only one of them. Error codes differ per
    version and genuinely have a single answer -- which the derivation gets right for free and
    a per-style rule would not."""
    by_id = {c["id"]: c for c in CHUNKS}
    for q in QUERIES:
        for rid in q["relevant"]:
            assert rid in by_id
        kinds = {by_id[r]["fact"]["kind"] for r in q["relevant"]}
        if q["style"] != "versioned" and kinds == {"error"}:
            assert len(q["relevant"]) == 1
        if q["style"] == "versioned":
            assert len(q["relevant"]) == 1


def test_the_paraphrase_twin_never_leaks_the_identifier():
    by_id = {c["id"]: c for c in CHUNKS}
    twins: dict[str, dict] = {}
    for q in QUERIES:
        if q["twin"]:
            twins.setdefault(q["twin"], {})[q["style"]] = q
    assert twins
    for pair in twins.values():
        key = by_id[pair["identifier"]["relevant"][0]]["fact"]["key"]
        assert key.lower() not in pair["paraphrase"]["text"].lower()
        assert pair["identifier"]["relevant"] == pair["paraphrase"]["relevant"]


# ---------------------------------------------------------------- embeddings
def test_embeddings_are_normalised():
    """Three separate pieces of code assume it: exact search treats a dot product as cosine,
    hnswlib uses the inner-product space, and pgvector indexes with vector_ip_ops. If the
    embedder stopped normalising, all three would quietly disagree."""
    n = np.linalg.norm(V_SMALL, axis=1)
    assert np.allclose(n, 1.0, atol=1e-5)


def test_the_hash_embedder_is_deterministic():
    assert np.array_equal(EMB.encode(["a b c"]), HashEmbedder().encode(["a b c"]))


# ---------------------------------------------------------------- lexical
def test_the_tokenizer_keeps_identifiers_whole_and_also_splits_them():
    """BM25 earns its place by matching exact strings. A tokenizer that shatters
    GATEWAY_RETRY_MAX_ATTEMPTS into common words leaves the lexical half doing a worse job of
    what dense retrieval already does, and then RRF fuses two opinions from one viewpoint."""
    toks = tokenize("Set GATEWAY_RETRY_MAX_ATTEMPTS or see GW-4417.")
    assert "gateway_retry_max_attempts" in toks
    assert "gw-4417" in toks
    assert "retry" in toks and "4417" in toks


def test_bm25_finds_an_exact_identifier():
    lex = Lexical(SMALL)
    target = next(c for c in SMALL if c["fact"]["kind"] == "error")
    got = lex.top(f"What does error {target['fact']['key']} mean?", 5)
    assert SMALL[got[0]]["id"] == target["id"]


# ---------------------------------------------------------------- exact search
def test_exact_top1_is_the_argmax():
    ex = Exact(V_SMALL)
    q = V_SMALL[7]
    assert ex.top(q, 1)[0] == int(np.argmax(V_SMALL @ q))


def test_an_unfiltered_mask_is_not_an_all_true_mask():
    """`None` and all-True mean different things: the first skips the filtered code path
    entirely, and the filtered-ANN instrument exists to measure that path."""
    assert mask_for(SMALL, None) is None
    m = mask_for(SMALL, {"version": SMALL[0]["version"]})
    assert m is not None and 0 < m.sum() < len(SMALL)


# ---------------------------------------------------------------- fusion
@given(st.lists(st.integers(0, 40), min_size=1, max_size=15, unique=True),
       st.lists(st.integers(0, 40), min_size=1, max_size=15, unique=True))
def test_rrf_is_deterministic_and_ranks_every_input(a, b):
    out1 = rrf({"x": a, "y": b})
    out2 = rrf({"x": a, "y": b})
    assert out1 == out2, "ties must break deterministically or two runs disagree"
    assert {d for d, _ in out1} == set(a) | set(b)


def test_a_document_ranked_by_both_beats_one_ranked_by_either():
    both = fuse_top({"x": [5, 1, 2], "y": [5, 3, 4]}, 1)
    assert both == [5]


def test_rrf_ignores_score_magnitude_by_construction():
    """The trade RRF makes, asserted so it is a decision rather than a surprise: a runaway
    winner and a hair's-breadth winner contribute exactly the same."""
    assert rrf({"x": [9, 8]})[0][1] == rrf({"y": [9, 8]})[0][1]


# ---------------------------------------------------------------- recall maths
@given(st.integers(1, 20))
def test_recall_of_a_result_against_itself_is_one(k):
    ids = list(range(k))
    assert M.recall_at_k(ids, ids, k) == 1.0


def test_recall_ignores_order_within_the_top_k():
    """An index that returns the right documents in a different order has lost nothing fusion
    or reranking cannot fix. One that returns different documents has."""
    assert M.recall_at_k([3, 2, 1], [1, 2, 3], 3) == 1.0


def test_holm_stops_at_the_first_failure():
    assert M.holm([0.001, 0.04, 0.2]) == [True, False, False]
    assert M.holm([0.04]) == [True]


def test_paired_bootstrap_finds_nothing_between_an_arm_and_itself():
    xs = [0.1, 0.9, 0.5, 0.3]
    d = M.paired_bootstrap(xs, xs, trials=500)
    assert d["delta"] == 0.0 and d["verdict"] == "INCONCLUSIVE"


# ---------------------------------------------------------------- the index
def test_hnsw_never_beats_exact_search():
    """If it ever does, the ground truth is not ground truth and every recall number above it
    is meaningless. This is the gate the frontier command enforces at runtime."""
    ex = Exact(V_SMALL)
    idx = LocalHnsw(V_SMALL.shape[1]).build(V_SMALL)
    idx.set_ef(16)
    for q in V_SMALL[:20]:
        assert M.recall_at_k(idx.top(q, 10), ex.top(q, 10), 10) <= 1.0


def test_recall_improves_as_ef_search_grows():
    ex = Exact(V_SMALL)
    idx = LocalHnsw(V_SMALL.shape[1]).build(V_SMALL)
    got = []
    for ef in (8, 64):
        idx.set_ef(ef)
        got.append(sum(M.recall_at_k(idx.top(q, 10), ex.top(q, 10), 10)
                       for q in V_SMALL[:40]) / 40)
    assert got[1] >= got[0]


def test_post_filtering_can_return_fewer_than_k_and_pre_filtering_cannot():
    """The failure that reaches production as a missing source rather than as an error."""
    idx = LocalHnsw(V_SMALL.shape[1]).build(V_SMALL)
    idx.set_ef(32)
    mask = mask_for(SMALL, {"service": SMALL[0]["service"],
                            "version": SMALL[0]["version"]})
    short_post = sum(1 for q in V_SMALL[:40]
                     if len(idx.top(q, 10, mask, mode="post")) < 10)
    short_pre = sum(1 for q in V_SMALL[:40]
                    if len(idx.top(q, 10, mask, mode="pre")) < 10)
    assert short_post > 0, "the corpus slice is not selective enough to show the failure"
    assert short_pre == 0


def test_the_construction_seed_actually_changes_the_graph():
    """The A/A arm is only a noise floor if the seed does something. It read 0.0000 at ef=32
    and looked like a dead instrument -- the seed matters where the GRAPH matters, so this
    checks at a low ef where it does."""
    aa = I.aa_index(V_SMALL, V_SMALL[:40], ef=4, k=10)
    assert aa["recall_a"] != aa["recall_b"] or aa["gap"] == 0.0
    assert 0.0 <= aa["gap"] <= 1.0
    assert aa["seeds"][0] != aa["seeds"][1]


# ---------------------------------------------------------------- instruments
def test_a_small_corpus_has_no_curve_to_measure():
    """Not a defect -- the project's stated limit, asserted so it stays honest.

    On a 600-chunk slice HNSW recall is already 0.997 at the lowest usable ef_search, because
    the graph reaches nearly everything. A corpus small enough for exact search is a corpus
    that does not need ANN, and a test that demanded a curve here would be demanding a
    falsehood.
    """
    rows = I.frontier(V_SMALL, V_SMALL[:40], [4, 64], k=10, verbose=False)
    assert all(0.0 <= r["recall"] <= 1.0 for r in rows)
    assert rows[0]["recall"] > 0.95


def test_the_frontier_reports_a_real_curve_on_the_full_corpus(full_vectors, query_vectors):
    """Queried with REAL query vectors. Feeding the index its own rows makes recall 1.000 at
    every ef_search, because the graph starts beside the answer -- an easier problem than the
    one a deployed index solves."""
    rows = I.frontier(full_vectors, query_vectors[:60], [8, 128], k=10, verbose=False)
    assert rows[0]["recall"] < rows[-1]["recall"], "no curve at 3k vectors"
    assert rows[0]["recall"] < 0.99


def test_the_frontier_records_the_effective_ef_when_it_clamps():
    rows = I.frontier(V_SMALL, V_SMALL[:10], [2, 32], k=10, verbose=False)
    assert rows[0]["ef_search"] == 2 and rows[0]["ef_effective"] == 10
    assert rows[1]["ef_effective"] == 32


def test_endtoend_records_the_effective_ef_when_it_clamps():
    """hnswlib cannot return `depth` candidates below an ef of `depth`, so an ef sweep at the
    product depth silently clamps and reports the same run several times. The row carries what
    actually ran."""
    pipe = I.Pipeline(SMALL, V_SMALL, depth=20)
    rows = I.endtoend(pipe, V_SMALL[:12], QUERIES[:12], [4, 64], k=5, depth=20,
                      verbose=False)
    assert rows[0]["ef_effective"] == 20 and rows[0]["ef_search"] == 4
    assert rows[1]["ef_effective"] == 64


def test_every_scored_rerank_arm_uses_the_same_k():
    """The wide arm was once scored at k=shortlist against a baseline at k=5, so hit@25 beat
    hit@5 by answering a different question. The shortlist is now a labelled ceiling."""
    from ragx.rerank import IdentityReranker
    pipe = I.Pipeline(SMALL, V_SMALL, depth=20)
    rows = I.rerank_compare(pipe, V_SMALL[:12], QUERIES[:12], IdentityReranker(),
                            ef=32, k=5, shortlist=20, verbose=False)
    scored = [r for r in rows if "note" not in r]
    assert {r["arm"] for r in scored} == {"fused@k", "reranked"}
    assert any("ceiling" in r["arm"] for r in rows)


def test_the_identity_reranker_is_a_true_control():
    from ragx.rerank import IdentityReranker
    assert IdentityReranker().top("q", [3, 1, 2], ["a", "b", "c"], 2) == [3, 1]


# ---------------------------------------------------------------- the API
@hyp_settings(max_examples=1, deadline=None)
@given(st.just(None))
def test_health_does_not_build_the_index(_):
    """A liveness probe that loads a model turns a health check into a cold start, and an
    orchestrator will kill the pod for being slow to say it is alive."""
    import os
    os.environ["RAGX_EMBEDDER"] = "hash"
    from fastapi.testclient import TestClient

    from ragx import api as api_mod
    api_mod._state.pop("r", None)
    client = TestClient(api_mod.create_app())
    body = client.get("/health").json()
    assert body["ok"] is True and body["index_loaded"] is False


def test_the_api_rejects_an_empty_query_and_exposes_metrics():
    import os
    os.environ["RAGX_EMBEDDER"] = "hash"
    from fastapi.testclient import TestClient

    from ragx import api as api_mod
    client = TestClient(api_mod.create_app())
    assert client.post("/search", json={"query": "   "}).status_code == 422
    assert "ragx_searches_total" in client.get("/metrics").text


def test_a_search_reports_which_ef_it_used_and_where_the_recall_came_from():
    from ragx.index import Retriever
    s = get_settings(embedder="hash", top_k=3)
    frontier = I.frontier(V_SMALL, V_SMALL[:20], [8, 64], k=10, verbose=False)
    r = Retriever(s, SMALL, V_SMALL, EMB, None, frontier)
    out = r.search("gateway retry attempts", k=3, recall_target=0.99, verify=True)
    assert out.ef_search in (8, 64)
    assert out.recall_measured_now is True
    assert out.recall_estimate is not None
    assert [t.stage for t in out.timings][:3] == ["embed_query", "dense", "lexical"]
    assert out.total_ms == pytest.approx(sum(t.ms for t in out.timings), abs=0.01)
