"""Typed contracts for everything crossing a boundary.

The one that matters is `SearchResponse.recall_estimate`. A retrieval API that returns
passages and nothing else is asking the caller to assume the index found what was there. This
one returns what it actually measured, and `exact` says whether that number came from a
comparison run now or from the committed frontier.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Backend = Literal["local", "pg", "exact"]


class Chunk(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    service: str
    version: str
    section: str
    title: str
    text: str
    fact: dict = Field(default_factory=dict)


class Hit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    rank: int
    score: float
    source: str                     # dense | lexical | rrf | rerank
    title: str = ""
    text: str = ""


class StageTiming(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stage: str
    ms: float


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    k: int | None = None
    recall_target: float | None = Field(default=None, ge=0.0, le=1.0)
    filter: dict[str, str] | None = None
    rerank: bool | None = None


class SearchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    hits: list[Hit]
    backend: Backend
    ef_search: int
    # What the index gave up, not what it hopes it gave up. `exact` is True when this run
    # actually compared against brute force; False when it is read off the committed curve.
    recall_estimate: float | None = None
    recall_measured_now: bool = False
    filtered: bool = False
    timings: list[StageTiming] = Field(default_factory=list)
    total_ms: float = 0.0


class AnswerResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    answer: str
    citations: list[str] = Field(default_factory=list)
    hits: list[Hit] = Field(default_factory=list)
    grounded: bool = True
    model: str = "mock"
    usd: float = 0.0
    total_ms: float = 0.0
