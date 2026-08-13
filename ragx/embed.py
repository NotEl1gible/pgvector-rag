"""Embeddings, and the one property everything downstream assumes.

BGE-small returns vectors that are already L2-normalised, so cosine similarity and inner
product are the same operation. That is not a detail: exact search becomes a single matrix
multiply, hnswlib can use the cheaper inner-product space, and pgvector's cosine and inner
product operators agree. If the model ever stopped normalising, all three would quietly
disagree with each other -- so `test_embeddings_are_normalised` asserts it rather than
trusting a dependency's README.

The hash embedder is not a stub for convenience. Tests that download a 130 MB ONNX model are
tests nobody runs, and CI that downloads one is CI that fails on a bad day at HuggingFace. It
carries real lexical signal, so a harness test can still tell a working retriever from a
broken one.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Protocol

import numpy as np

from .config import EMBED_DIM, EMBED_MODEL, Settings

_WORD = re.compile(r"[A-Za-z0-9_.\-]+")


class Embedder(Protocol):
    dim: int
    name: str

    def encode(self, texts: list[str]) -> np.ndarray: ...


def l2(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.maximum(n, 1e-12)


class HashEmbedder:
    """Deterministic bag-of-words hashing. Offline, instant, reproducible everywhere."""

    name = "hash"

    def __init__(self, dim: int = EMBED_DIM, seed: int = 7):
        self.dim = dim
        self.seed = seed

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in _WORD.findall(t.lower()):
                h = hashlib.blake2b(f"{self.seed}:{w}".encode(), digest_size=8).digest()
                idx = int.from_bytes(h[:4], "little") % self.dim
                out[i, idx] += 1.0 if h[4] & 1 else -1.0
        return l2(out)


class OnnxEmbedder:
    """fastembed's BGE-small, CPU only. No torch, so `pip install -r requirements.txt` is
    genuinely enough for a reader to reproduce every number in the README."""

    name = "onnx"

    # Two threads, not every core. This runs on a laptop that the person reading the output
    # is also using; a retrieval demo that pins the machine for four minutes is a demo they
    # cancel. The corpus is sized so the whole embedding pass fits in well under a minute at
    # this thread count, and the result is cached so it happens once ever.
    def __init__(self, model: str = EMBED_MODEL, dim: int = EMBED_DIM, threads: int = 2):
        from fastembed import TextEmbedding
        self.dim = dim
        self.model_name = model
        self._m = TextEmbedding(model_name=model, threads=threads)

    def encode(self, texts: list[str]) -> np.ndarray:
        v = np.asarray(list(self._m.embed(texts)), dtype=np.float32)
        # Normalised again rather than assumed. The cost is one division over a matrix that
        # was already unit length; the alternative is three index implementations silently
        # disagreeing the day a model version changes.
        return l2(v)


def build_embedder(s: Settings) -> Embedder:
    return OnnxEmbedder() if s.embedder == "onnx" else HashEmbedder()


def embed_corpus(emb: Embedder, chunks: list[dict], cache: str | None = None,
                 batch: int = 64, verbose: bool = True, throttle: float = 0.15,
                 throttle_every: int = 4) -> np.ndarray:
    """Embed once, cache to disk, and yield the CPU back between batches.

    Two separate protections against a command that makes the machine unusable. The cache
    means a full pass happens once ever rather than once per instrument -- every later command
    reads a .npy. The throttle hands the scheduler a gap every few batches, which costs a few
    seconds on the one run that computes and keeps the laptop responsive while it does.
    """
    import time

    p = Path(cache) if cache else None
    if p and p.exists():
        v = np.load(p)
        if v.shape == (len(chunks), emb.dim):
            if verbose:
                print(f"  vectors cached: {p} {v.shape}")
            return v
    vecs = []
    n_batches = (len(chunks) + batch - 1) // batch
    for bi, i in enumerate(range(0, len(chunks), batch)):
        vecs.append(emb.encode([c["text"] for c in chunks[i:i + batch]]))
        if verbose and bi % 8 == 0:
            print(f"  embedded {min(i + batch, len(chunks))}/{len(chunks)}", flush=True)
        if throttle and bi % throttle_every == throttle_every - 1 and bi < n_batches - 1:
            time.sleep(throttle)
    v = np.vstack(vecs).astype(np.float32)
    if p:
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(p, v)
    return v
