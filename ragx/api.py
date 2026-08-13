"""The product surface.

`POST /search` is the endpoint the whole repository exists to justify. It accepts a
`recall_target` and returns the `ef_search` it chose to meet it, together with what that
setting actually recalls -- so a caller can tell a fast answer from a complete one instead of
assuming they are the same thing.

`/metrics` is hand-rolled Prometheus text rather than a client library: one fewer dependency,
and the counters here are the ones this service actually has.
"""
from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException, Response

from . import corpus as C
from .config import get_settings
from .index import build_retriever
from .schemas import SearchRequest, SearchResponse

COUNTERS: dict[str, float] = {
    "ragx_searches_total": 0.0,
    "ragx_filtered_searches_total": 0.0,
    "ragx_reranked_searches_total": 0.0,
    "ragx_verified_searches_total": 0.0,
    "ragx_search_ms_total": 0.0,
}

_state: dict[str, Any] = {}


def _retriever():
    if "r" not in _state:
        s = get_settings()
        chunks, _ = C.load()
        _state["r"] = build_retriever(s, chunks, verbose=False)
    return _state["r"]


def create_app() -> FastAPI:
    s = get_settings()
    api = FastAPI(title="pgvector-rag", version="0.1.0",
                  description="Hybrid retrieval that reports what its approximate index "
                              "gave up.")

    @api.get("/health")
    def health() -> dict[str, Any]:
        # Deliberately does NOT build the index. A health check that loads a model turns a
        # liveness probe into a cold start, and an orchestrator will kill the pod for being
        # slow to answer a question about whether it is alive.
        return {"ok": True, "backend": s.backend, "embedder": s.embedder,
                "ef_search": s.ef_search, "recall_target": s.recall_target,
                "index_loaded": "r" in _state}

    @api.post("/search", response_model=SearchResponse)
    def search(req: SearchRequest) -> SearchResponse:
        if not req.query.strip():
            raise HTTPException(status_code=422, detail="empty query")
        r = _retriever()
        out = r.search(req.query, k=req.k, recall_target=req.recall_target,
                       flt=req.filter, rerank=req.rerank,
                       verify=bool(req.recall_target))
        COUNTERS["ragx_searches_total"] += 1
        COUNTERS["ragx_filtered_searches_total"] += 1 if out.filtered else 0
        COUNTERS["ragx_reranked_searches_total"] += 1 if req.rerank else 0
        COUNTERS["ragx_verified_searches_total"] += 1 if out.recall_measured_now else 0
        COUNTERS["ragx_search_ms_total"] += out.total_ms
        return out

    @api.get("/metrics")
    def metrics() -> Response:
        lines = []
        for name, value in COUNTERS.items():
            lines.append(f"# TYPE {name} counter")
            lines.append(f"{name} {value}")
        return Response("\n".join(lines) + "\n", media_type="text/plain; version=0.0.4")

    return api


app = create_app()
