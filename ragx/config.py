"""One settings object. Every knob is here or it does not exist.

The defaults are the product's defaults, and several of them are chosen by a measurement in
`evals/` rather than by taste -- `ef_search` most of all. Anything read from the environment is
read once, here, so that "what was this run configured with" is answerable from a single dump.
"""
from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# 384 dims, L2-normalised by the model, so cosine and inner product are the same thing and
# exact search is one matrix multiply. Verified, not assumed: `test_embeddings_are_normalised`.
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_DIM = 384
RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"

PRICES: dict[str, tuple[float, float]] = {      # USD per 1M tokens, answer layer only
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-sonnet-5": (3.0, 15.0),
    "mock": (0.0, 0.0),
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAGX_", env_file=".env", extra="ignore",
                                      protected_namespaces=())

    # --- retrieval ---------------------------------------------------------
    backend: str = "local"                     # local (hnswlib) | pg (pgvector)
    embedder: str = "onnx"                     # onnx (fastembed) | hash (tests only)
    top_k: int = 5
    depth: int = 50                            # candidates each retriever contributes to RRF
    rrf_k: int = 60
    rerank: bool = False
    rerank_depth: int = 50

    # --- the index under measurement ---------------------------------------
    hnsw_m: int = 16
    hnsw_ef_construction: int = 200
    # The product default, and it is not a guess: `frontier` prints recall against exact at
    # every setting, and this is the smallest value meeting the recall target on that curve.
    ef_search: int = 64
    recall_target: float = 0.95

    # --- infrastructure ----------------------------------------------------
    database_url: str = "postgresql+psycopg://ragx:ragx@localhost:5432/ragx"
    index_dir: str = "artifacts"
    otlp_endpoint: str = ""
    service_name: str = "pgvector-rag"

    # --- answer layer (optional; no headline number depends on it) ---------
    provider: str = "mock"                     # mock | anthropic
    answer_model: str = "claude-haiku-4-5"
    anthropic_api_key: str = ""

    seed: int = 7
    corpus_facts_per_section: int = 28
    n_queries: int = 300

    ef_sweep: list[int] = Field(
        default_factory=lambda: [8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256])


def get_settings(**overrides) -> Settings:
    return Settings(**overrides)
