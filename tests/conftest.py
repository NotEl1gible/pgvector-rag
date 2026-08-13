"""Shared fixtures.

The full-corpus vectors are session-scoped and hash-embedded: about a second and a half of
NumPy, computed once for the whole suite. Only the tests that genuinely need 3,000 vectors --
the ones asserting that an ANN curve exists at all -- ask for them.
"""
from __future__ import annotations

import pytest

from ragx import corpus as C
from ragx.embed import HashEmbedder


@pytest.fixture(scope="session")
def full_vectors():
    chunks, _ = C.load()
    return HashEmbedder().encode([c["text"] for c in chunks])


@pytest.fixture(scope="session")
def query_vectors():
    """Real query vectors, NOT rows of the corpus.

    Querying an HNSW index with a vector that is already in it is a different and much easier
    problem: the graph entry point is at or beside the answer, so recall comes out at 1.000
    even at the lowest ef_search. Measuring the frontier that way would report an index far
    better than the one anybody deploys.
    """
    _, queries = C.load()
    return HashEmbedder().encode([q["text"] for q in queries])
