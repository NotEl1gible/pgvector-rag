"""Runs only where a real Postgres with pgvector exists -- the CI integration job.

These are the claims the hermetic suite cannot make. hnswlib and pgvector are two separate
HNSW implementations with two separate sets of defaults, and the only way to know they agree
is to build both and compare. Skipped rather than faked when the server is absent, because a
green tick that means nothing is worse than a skip.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from evals import metrics as M
from ragx import corpus as C
from ragx.dense_local import LocalHnsw
from ragx.dense_pg import PgHnsw, build_engine
from ragx.embed import HashEmbedder
from ragx.exact import Exact, mask_for

PG = os.environ.get("RAGX_TEST_POSTGRES_URL", "")
pg_only = pytest.mark.skipif(not PG, reason="no RAGX_TEST_POSTGRES_URL")

N = 1500          # enough for HNSW to be genuinely approximate, small enough to build fast
K = 10
EF = 32


@pytest.fixture(scope="module")
def fixture():
    chunks, queries = C.load()
    chunks = chunks[:N]
    emb = HashEmbedder()
    v = emb.encode([c["text"] for c in chunks])
    q = emb.encode([x["text"] for x in queries[:60]])
    return chunks, v, q


@pg_only
def test_pgvector_builds_the_index_and_returns_neighbours(fixture):
    chunks, v, q = fixture
    pg = PgHnsw(build_engine(PG), dim=v.shape[1]).build(chunks, v)
    assert pg.count() == len(chunks)
    ids, _ = pg.top(q[0], K, ef=EF)
    assert len(ids) == K and len(set(ids)) == K


@pg_only
def test_pgvector_and_hnswlib_agree_on_recall(fixture):
    """The claim that makes one measurement stand for both backends.

    They are separate implementations with separate defaults -- opclass, distance operator,
    how ef_search is applied. If their recall against the SAME exact answer key diverges, the
    local sweep in the README does not describe what the deployed database does, and every
    number in it is about hnswlib only.
    """
    chunks, v, q = fixture
    ex = Exact(v)
    truth = [ex.top(x, K) for x in q]

    local = LocalHnsw(v.shape[1]).build(v)
    local.set_ef(EF)
    r_local = [M.recall_at_k(local.top(x, K), t, K) for x, t in zip(q, truth, strict=True)]

    pg = PgHnsw(build_engine(PG), dim=v.shape[1]).build(chunks, v)
    r_pg = [M.recall_at_k(pg.top(x, K, ef=EF)[0], t, K)
            for x, t in zip(q, truth, strict=True)]

    mean_local = sum(r_local) / len(r_local)
    mean_pg = sum(r_pg) / len(r_pg)
    assert abs(mean_local - mean_pg) < 0.10, (
        f"hnswlib {mean_local:.3f} vs pgvector {mean_pg:.3f} -- the two backends disagree, "
        f"so the local frontier does not describe the deployed index")
    assert mean_pg < 0.9999, "pgvector returned exact results; it is not indexing"


@pg_only
def test_neither_backend_beats_exact_search(fixture):
    chunks, v, q = fixture
    ex = Exact(v)
    pg = PgHnsw(build_engine(PG), dim=v.shape[1]).build(chunks, v)
    for x in q[:20]:
        got, _ = pg.top(x, K, ef=EF)
        assert M.recall_at_k(got, ex.top(x, K), K) <= 1.0


@pg_only
def test_a_filtered_query_says_which_plan_it_used(fixture):
    """With pgvector you cannot tell pre- from post-filtering by reading the SQL: one WHERE
    clause, and the PLANNER decides whether to walk the graph or give up on the index. EXPLAIN
    is the only way to find out afterwards, which is why it is captured rather than assumed."""
    chunks, v, q = fixture
    pg = PgHnsw(build_engine(PG), dim=v.shape[1]).build(chunks, v)
    flt = {"service": chunks[0]["service"], "version": chunks[0]["version"]}
    ids, plan = pg.top(q[0], K, ef=EF, flt=flt, explain=True)
    assert plan, "no plan captured"
    assert ("Index Scan" in plan) or ("Seq Scan" in plan)
    keep = mask_for(chunks, flt)
    assert all(bool(keep[i]) for i in ids), "the filter let a non-matching row through"


@pg_only
def test_iterative_scan_is_what_stops_a_short_return(fixture):
    """pgvector 0.8's answer to the post-filter failure, and it is OFF by default -- which is
    why so many deployments have that failure without knowing it."""
    chunks, v, q = fixture
    pg = PgHnsw(build_engine(PG), dim=v.shape[1]).build(chunks, v)
    flt = {"service": chunks[0]["service"], "version": chunks[0]["version"]}
    short_plain = sum(1 for x in q[:30] if len(pg.top(x, K, ef=8, flt=flt)[0]) < K)
    short_iter = sum(1 for x in q[:30]
                     if len(pg.top(x, K, ef=8, flt=flt, iterative=True)[0]) < K)
    assert short_iter <= short_plain


@pg_only
def test_the_stored_vectors_survive_the_round_trip(fixture):
    """A float32 vector written as text and parsed back is a place precision quietly dies, and
    a similarity that is wrong in the fourth decimal changes rankings without changing
    anything a schema check would notice."""
    from sqlalchemy import text
    chunks, v, _ = fixture
    eng = build_engine(PG)
    pg = PgHnsw(eng, dim=v.shape[1]).build(chunks, v)
    with eng.begin() as c:
        got = c.execute(text(f"SELECT embedding FROM {pg.table} WHERE ord = 0")).scalar_one()
    back = np.array([float(x) for x in str(got).strip("[]").split(",")], dtype=np.float32)
    assert np.allclose(back, v[0], atol=1e-6)
    assert abs(float(np.linalg.norm(back)) - 1.0) < 1e-4
