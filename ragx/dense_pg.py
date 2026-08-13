"""The same HNSW index, in Postgres.

Written here, proved in CI: this machine has no Docker, so every line below is exercised
against a real `pgvector/pgvector:pg16` service container rather than locally. The integration
job asserts that pgvector and hnswlib agree on recall at the same `ef_search` -- if they ever
diverge, one of them is not building the index it claims to.

**The reason both implementations exist** is not redundancy. With pgvector you cannot tell
from the SQL whether a filtered query was pre-filtered or post-filtered: you write one `WHERE`
clause and the PLANNER decides whether to walk the HNSW graph and discard non-matching rows,
or to give up on the index and scan. Which one you got is invisible in the result and very
visible in the recall. hnswlib exposes both strategies explicitly, so the in-process numbers
are what the planner's choice is measured against.

`EXPLAIN` is captured alongside the results for the same reason -- it is the only way to know
after the fact which plan actually ran.
"""
from __future__ import annotations

import numpy as np
from sqlalchemy import text

DIM_DEFAULT = 384


class PgHnsw:
    name = "pg"

    def __init__(self, engine, dim: int = DIM_DEFAULT, m: int = 16,
                 ef_construction: int = 200, table: str = "chunks"):
        self.engine, self.dim, self.m = engine, dim, m
        self.ef_construction, self.table = ef_construction, table
        self.n = 0

    # ---------------------------------------------------------------- build
    def create(self) -> None:
        with self.engine.begin() as c:
            c.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            c.execute(text(f"DROP TABLE IF EXISTS {self.table}"))
            c.execute(text(f"""
                CREATE TABLE {self.table} (
                    ord       integer PRIMARY KEY,
                    id        text NOT NULL UNIQUE,
                    service   text NOT NULL,
                    version   text NOT NULL,
                    section   text NOT NULL,
                    title     text NOT NULL,
                    body      text NOT NULL,
                    embedding vector({self.dim}) NOT NULL
                )"""))

    def insert(self, chunks: list[dict], vectors: np.ndarray, batch: int = 500) -> None:
        rows = [{"ord": i, "id": c["id"], "service": c["service"], "version": c["version"],
                 "section": c["section"], "title": c["title"], "body": c["text"],
                 "embedding": "[" + ",".join(f"{x:.7f}" for x in vectors[i]) + "]"}
                for i, c in enumerate(chunks)]
        stmt = text(f"INSERT INTO {self.table} "
                    f"(ord, id, service, version, section, title, body, embedding) "
                    f"VALUES (:ord, :id, :service, :version, :section, :title, :body, "
                    f"CAST(:embedding AS vector))")
        with self.engine.begin() as c:
            for i in range(0, len(rows), batch):
                c.execute(stmt, rows[i:i + batch])
        self.n = len(rows)

    def index(self) -> None:
        """`vector_ip_ops`, because the vectors are L2-normalised and inner product is then
        cosine with one fewer operation. Building the index AFTER the insert is deliberate:
        HNSW built incrementally during a bulk load is both slower and worse connected."""
        with self.engine.begin() as c:
            c.execute(text(f"""
                CREATE INDEX IF NOT EXISTS {self.table}_hnsw
                ON {self.table} USING hnsw (embedding vector_ip_ops)
                WITH (m = {self.m}, ef_construction = {self.ef_construction})"""))
            c.execute(text("ANALYZE " + self.table))

    def build(self, chunks: list[dict], vectors: np.ndarray) -> "PgHnsw":
        self.create()
        self.insert(chunks, vectors)
        self.index()
        return self

    # ---------------------------------------------------------------- search
    @staticmethod
    def _lit(q: np.ndarray) -> str:
        return "[" + ",".join(f"{x:.7f}" for x in np.asarray(q).ravel()) + "]"

    def top(self, q: np.ndarray, k: int, ef: int = 64,
            flt: dict[str, str] | None = None, iterative: bool = False,
            explain: bool = False) -> tuple[list[int], str]:
        where, params = "", {"q": self._lit(q), "k": k}
        if flt:
            clauses = []
            for i, (key, val) in enumerate(flt.items()):
                if key not in ("service", "version", "section"):
                    raise ValueError(f"not a filterable column: {key}")
                clauses.append(f"{key} = :f{i}")
                params[f"f{i}"] = val
            where = "WHERE " + " AND ".join(clauses)
        sql = (f"SELECT ord FROM {self.table} {where} "
               f"ORDER BY embedding <#> CAST(:q AS vector) LIMIT :k")
        plan = ""
        with self.engine.begin() as c:
            c.execute(text(f"SET LOCAL hnsw.ef_search = {max(1, ef)}"))
            if iterative:
                # pgvector 0.8+: keep scanning the graph until k matching rows are found,
                # instead of returning short. It is the fix for the post-filter failure and
                # it is off by default, which is why so many deployments have that failure.
                c.execute(text("SET LOCAL hnsw.iterative_scan = relaxed_order"))
            if explain:
                plan = "\n".join(r[0] for r in c.execute(
                    text("EXPLAIN (ANALYZE, BUFFERS) " + sql), params))
            rows = c.execute(text(sql), params).fetchall()
        return [int(r[0]) for r in rows], plan

    def count(self) -> int:
        with self.engine.begin() as c:
            return int(c.execute(text(f"SELECT count(*) FROM {self.table}")).scalar_one())


def build_engine(url: str, echo: bool = False):
    from sqlalchemy import create_engine
    return create_engine(url, future=True, echo=echo, pool_pre_ping=True)
