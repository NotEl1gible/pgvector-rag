"""A generated documentation corpus, and queries whose answer chunk is known by construction.

Retrieval evaluation normally begins with someone hand-labelling which passage answers which
query. That labour is the reason most retrieval evals are small, and hand labels carry a
failure this corpus cannot have: the label that disagrees with the passage.

Here every chunk is built AROUND one fact, and every query is generated FROM that fact. The
relevant chunk is therefore exact by construction, at any scale, with nobody annotating
anything.

The **version axis** is doing three jobs at once, which is why it is the spine of the design:

1. **Near-duplicate distractors, for free and for a realistic reason.** The same section of the
   same service exists at 2.2, 2.3 and 2.4, differing in one number. A retriever that finds
   "the retry limit for the gateway" without respecting the version has found a passage that
   reads correctly and answers wrongly -- the most dangerous kind of retrieval error, and one
   that never appears in a corpus of unrelated documents.
2. **A metadata filter with genuine selectivity**, which is what the filtered-ANN instrument
   needs. `version = '2.2'` keeps a third of the corpus; `service = 'gateway' AND version =
   '2.2'` keeps about a hundredth. HNSW behaves very differently across that range.
3. **Three query styles from one fact.** The identifier style contains the literal token and
   is where BM25 wins; the paraphrase style shares no content words with the chunk and is where
   dense wins; the versioned style needs the filter. Dense and lexical retrieval fail in
   opposite ways, and this is how that claim gets measured rather than repeated.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

SERVICES: list[tuple[str, str, str]] = [
    ("gateway", "API Gateway", "GW"), ("payments", "Payments", "PY"),
    ("indexer", "Search Indexer", "IX"), ("scheduler", "Job Scheduler", "SC"),
    ("auth", "Identity", "AU"), ("storage", "Object Storage", "ST"),
    ("ledger", "Ledger", "LD"), ("notify", "Notifications", "NT"),
    ("stream", "Event Stream", "EV"), ("catalog", "Product Catalog", "CT"),
    ("billing", "Billing", "BL"), ("shipping", "Shipping", "SH"),
    ("reporting", "Reporting", "RP"), ("registry", "Service Registry", "RG"),
    ("vault", "Secret Vault", "VT"), ("mesh", "Service Mesh", "MS"),
    ("cache", "Edge Cache", "CH"), ("router", "Traffic Router", "TR"),
    ("audit", "Audit Log", "AD"), ("archive", "Cold Archive", "AR"),
    ("workflow", "Workflow Engine", "WF"), ("insights", "Insights", "IN"),
    ("connect", "Partner Connect", "PC"), ("console", "Admin Console", "CN"),
]
VERSIONS = ["2.2", "2.3", "2.4"]
SECTIONS = ["configuration", "errors", "limits", "operations", "migration", "security"]

_CONFIG_THINGS = [
    ("RETRY_MAX_ATTEMPTS", "attempts", "retries a failed upstream call"),
    ("CONNECT_TIMEOUT_MS", "milliseconds", "waits for a connection to be established"),
    ("POOL_MAX_SIZE", "connections", "keeps open to each downstream service"),
    ("BATCH_FLUSH_SIZE", "records", "buffers before writing a batch"),
    ("HEARTBEAT_INTERVAL_S", "seconds", "waits between liveness probes"),
    ("QUEUE_DEPTH_WARN", "messages", "allows to queue before warning"),
    ("SHUTDOWN_GRACE_S", "seconds", "allows in-flight work to finish on shutdown"),
]
_ERROR_CAUSES = [
    ("the upstream connection was reset before the first byte arrived",
     "retry with backoff; the request was never processed"),
    ("the request body exceeded the negotiated frame size",
     "split the payload or raise the frame limit"),
    ("the caller presented a token signed by a retired key",
     "refresh the signing key set and reissue the token"),
    ("a downstream dependency reported a schema it does not publish",
     "pin the dependency version and re-run the contract check"),
    ("the write was accepted but the durable commit did not confirm in time",
     "treat the operation as unknown and reconcile from the ledger"),
    ("the tenant exceeded its concurrency allowance mid-request",
     "shed the request and retry outside the burst window"),
]
_LIMIT_THINGS = [
    ("maximum request body", "MiB"), ("maximum concurrent streams", "streams"),
    ("maximum objects per listing page", "objects"), ("retention of failed jobs", "days"),
    ("maximum tags per resource", "tags"), ("burst allowance per tenant", "requests"),
]
_OPS_THINGS = [
    ("compaction runs", "compaction"), ("nightly settlement runs", "settlement"),
    ("index snapshots are taken", "snapshot"), ("credentials are rotated", "rotation"),
    ("cold data is swept to archive", "sweep"), ("replicas are re-balanced", "rebalance"),
]
# A qualifier dimension, and it exists to fix a real defect rather than to add variety. With
# only the base lists, the fact key repeated every len(list) indices INSIDE one version -- so
# four chunks of storage 2.2 each claimed a different value for "burst allowance per tenant".
# A corpus that contradicts itself has no ground truth at all: whichever chunk a retriever
# returns, some other chunk says otherwise. Qualifiers make the key unique within a version,
# leaving the ONLY duplication the designed one -- the same key across 2.2, 2.3 and 2.4.
_QUALIFIERS = ["", " for streaming traffic", " for batch imports", " on the public edge",
               " for partner tenants", " in the private region"]


def _qual(i: int, base_len: int) -> str:
    return _QUALIFIERS[(i // base_len) % len(_QUALIFIERS)]


_FILLER = [
    "Operators generally leave this at the default; changing it in isolation rarely helps.",
    "The value applies per instance, not per cluster, which surprises people during a scale-out.",
    "It is read once at start-up, so a change requires a rolling restart to take effect.",
    "The dashboard reports the effective value, which may differ from the configured one.",
    "Support will ask for this value first when a latency ticket is opened.",
    "Changing it does not migrate work already in flight.",
]


def _fact_text(rng: random.Random, service: tuple[str, str, str], version: str,
               section: str, i: int) -> tuple[str, str, dict]:
    slug, name, code = service
    vtag = version.replace(".", "")
    if section == "configuration":
        key_base, unit, does = _CONFIG_THINGS[i % len(_CONFIG_THINGS)]
        suffix = ["", "_STREAM", "_BATCH", "_EDGE", "_PARTNER", "_PRIVATE"][
            (i // len(_CONFIG_THINGS)) % 6]
        key = f"{slug.upper()}_{key_base}{suffix}"
        value = 2 + (i * 7 + int(vtag)) % 90
        title = f"{name} {version} configuration: {key}"
        body = (f"{key} controls how many {unit} the {name} service {does}. "
                f"In {name} {version} the default is {value} {unit}. ")
        fact = {"kind": "config", "key": key, "value": f"{value} {unit}", "does": does}
    elif section == "errors":
        cause, remedy = _ERROR_CAUSES[i % len(_ERROR_CAUSES)]
        ecode = f"{code}-{4000 + (i * 13 + int(vtag) * 3) % 5000}"
        title = f"{name} {version} errors: {ecode}"
        body = (f"Error {ecode} is returned when {cause}. The recommended response is to "
                f"{remedy}. In {name} {version} this error is retryable at the edge. ")
        fact = {"kind": "error", "key": ecode, "value": cause, "does": remedy}
    elif section == "limits":
        thing, unit = _LIMIT_THINGS[i % len(_LIMIT_THINGS)]
        thing = thing + _qual(i, len(_LIMIT_THINGS))
        value = 4 + (i * 11 + int(vtag)) % 200
        title = f"{name} {version} limits: {thing}"
        body = (f"The {thing} in {name} {version} is {value} {unit}. Requests beyond the "
                f"limit are rejected before any work is scheduled. ")
        fact = {"kind": "limit", "key": thing, "value": f"{value} {unit}", "does": thing}
    elif section == "operations":
        thing, word = _OPS_THINGS[i % len(_OPS_THINGS)]
        q = _qual(i, len(_OPS_THINGS))
        thing, word = thing + q, (word + q.replace(" ", "-")).strip("-")
        hour = (i * 5 + int(vtag)) % 24
        title = f"{name} {version} operations: {word}"
        body = (f"In {name} {version}, {thing} daily at {hour:02d}:00 UTC. The window is "
                f"chosen to sit outside the regional peak. ")
        fact = {"kind": "ops", "key": word, "value": f"{hour:02d}:00 UTC", "does": thing}
    elif section == "migration":
        subsystem = ["core", "wire", "storage", "auth", "telemetry", "scheduler"][i % 6]
        q = _qual(i, 6).replace(" ", "-").strip("-") or "base"
        old = f"{slug.upper()}_{subsystem.upper()}_LEGACY_MODE"
        new = f"{slug.upper()}_{subsystem.upper()}_COMPAT_LEVEL"
        title = f"{name} {version} migration notes: {subsystem} ({q})"
        body = (f"{name} {version} replaces {old} with {new} for the {subsystem} subsystem. "
                f"Deployments still setting {old} start with a warning and fall back to "
                f"level {1 + i % 4}. ")
        fact = {"kind": "migration", "key": f"{new}:{q}", "value": old,
                "does": f"replaces {old} with {new}"}
    else:
        scope = ["read", "write", "admin", "billing", "audit"][i % 5]
        scope_key = f"{scope} scope{_qual(i, 5)}"
        days = 7 + (i * 3 + int(vtag)) % 80
        title = f"{name} {version} security: {scope_key}"
        body = (f"Tokens carrying the {scope_key} for {name} {version} expire after "
                f"{days} days and cannot be renewed in place. ")
        fact = {"kind": "security", "key": scope_key, "value": f"{days} days",
                "does": f"how long a {scope_key} token for {name} stays valid"}
    body += rng.choice(_FILLER) + " " + rng.choice(_FILLER)
    return title, body, fact


def build_corpus(n_facts_per_section: int = 28, seed: int = 5) -> list[dict]:
    rng = random.Random(seed)
    chunks: list[dict] = []
    for svc in SERVICES:
        for version in VERSIONS:
            for section in SECTIONS:
                for i in range(n_facts_per_section):
                    title, body, fact = _fact_text(rng, svc, version, section, i)
                    chunks.append({
                        "id": f"{svc[0]}-{version}-{section}-{i:02d}",
                        "service": svc[0], "version": version, "section": section,
                        "title": title, "text": f"{title}. {body}".strip(),
                        "fact": fact,
                    })
    return chunks


# ----------------------------------------------------------------------------
# Queries -- three styles from the same fact
# ----------------------------------------------------------------------------
def _service_name(slug: str) -> str:
    return next(s[1] for s in SERVICES if s[0] == slug)


def _identifier_query(c: dict) -> str | None:
    """Contains the literal token. BM25 territory: an exact string a dense model has no
    particular reason to place near anything."""
    f = c["fact"]
    if f["kind"] == "config":
        return f"What is the default value of {f['key']}?"
    if f["kind"] == "error":
        return f"What does error {f['key']} mean?"
    return None


def _paraphrase_query(c: dict) -> str | None:
    """The SAME fact, described without its identifier. Dense territory.

    Deliberately built from the same facts as the identifier style so the two are PAIRED on
    the same target chunks. Comparing lexical and dense retrieval on different query subsets
    measures the subsets as much as the retrievers; here the only thing that changes is how
    the question is phrased.
    """
    f = c["fact"]
    name = _service_name(c["service"])
    if f["kind"] == "config":
        unit = f["value"].split(" ", 1)[1]
        return f"How many {unit} does {name} use when it {f['does']}?"
    if f["kind"] == "error":
        return f"Why does a {name} request fail when {f['value']}?"
    return None


def _versioned_query(c: dict) -> str | None:
    """The trap, and the only style with exactly one right answer.

    The same section exists at three versions with different numbers, so a retriever that
    ignores the constraint returns a passage that reads correctly and answers wrongly -- the
    most dangerous retrieval error there is. Carries a filter, which is what gives the
    filtered-ANN instrument something real to filter on.
    """
    f = c["fact"]
    name = _service_name(c["service"])
    if f["kind"] == "config":
        return f"In {name} {c['version']}, what is {f['key']} set to?"
    if f["kind"] in ("limit", "ops", "security"):
        return f"In {name} {c['version']}, what is the {f['key']}?"
    return None


def _relevant_set(chunks: list[dict], c: dict, versioned: bool) -> list[str]:
    """Every chunk that genuinely answers the query, computed rather than assumed.

    This is the bug the first version of this file shipped: "What is the default value of
    GATEWAY_RETRY_MAX_ATTEMPTS?" has THREE right answers, one per version, and only one was
    labelled. A retriever would have been marked wrong for finding a passage that answers the
    question. Error codes differ per version, so their relevant set is genuinely one chunk --
    which is why the set is derived from the fact key instead of from the style.
    """
    if versioned:
        return [c["id"]]
    key = c["fact"]["key"]
    return sorted(x["id"] for x in chunks
                  if x["service"] == c["service"] and x["fact"]["key"] == key)


def build_queries(chunks: list[dict], n: int = 300, seed: int = 11) -> list[dict]:
    rng = random.Random(seed)
    pool = list(chunks)
    rng.shuffle(pool)
    per = n // 3

    # identifier and paraphrase are drawn from the SAME chunks, in the same order, so every
    # identifier query has a paraphrase twin on the same target.
    twinnable = [c for c in pool
                 if _identifier_query(c) and _paraphrase_query(c)][:per]
    out: list[dict] = []
    for i, c in enumerate(twinnable):
        rel = _relevant_set(chunks, c, versioned=False)
        for style, fn in (("identifier", _identifier_query),
                          ("paraphrase", _paraphrase_query)):
            out.append({"id": f"q-{style[:3]}-{i:03d}", "style": style, "text": fn(c),
                        "relevant": rel, "twin": f"pair-{i:03d}",
                        "service": c["service"], "version": c["version"],
                        "section": c["section"], "filter": None})

    taken = 0
    for c in pool:
        if taken >= per:
            break
        text = _versioned_query(c)
        if not text:
            continue
        out.append({"id": f"q-ver-{taken:03d}", "style": "versioned", "text": text,
                    "relevant": _relevant_set(chunks, c, versioned=True),
                    "twin": None, "service": c["service"], "version": c["version"],
                    "section": c["section"],
                    "filter": {"service": c["service"], "version": c["version"]}})
        taken += 1
    return out


def corpus_hash(chunks: list[dict]) -> str:
    import hashlib
    h = hashlib.sha256()
    for c in chunks:
        h.update(json.dumps(c, sort_keys=True).encode())
    return h.hexdigest()


def write(chunks: list[dict], queries: list[dict], root: str = "evals") -> tuple[Path, Path]:
    """Queries and the corpus HASH are committed; the corpus itself is not.

    The corpus is 8 MB of templated prose and a pure function of this file plus a seed, so
    committing it would add weight without adding evidence -- a reader learns more from the
    generator than from the output. What must be committed is the part that could otherwise
    drift: the qrels, and a hash that fails CI the moment the corpus stops being the one those
    qrels were built against.
    """
    d = Path(root)
    d.mkdir(parents=True, exist_ok=True)
    cp, qp = d / "corpus.jsonl", d / "queries.jsonl"
    cp.write_text("".join(json.dumps(c) + "\n" for c in chunks), encoding="utf-8")
    qp.write_text("".join(json.dumps(q) + "\n" for q in queries), encoding="utf-8")
    (d / "corpus.sha256").write_text(corpus_hash(chunks) + "\n", encoding="utf-8")
    return cp, qp


def load(root: str = "evals") -> tuple[list[dict], list[dict]]:
    d = Path(root)
    chunks = [json.loads(x) for x in (d / "corpus.jsonl").read_text(encoding="utf-8")
              .splitlines() if x]
    queries = [json.loads(x) for x in (d / "queries.jsonl").read_text(encoding="utf-8")
               .splitlines() if x]
    return chunks, queries
