# pgvector-rag

[![CI](https://github.com/NotEl1gible/pgvector-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/NotEl1gible/pgvector-rag/actions/workflows/ci.yml)

Hybrid retrieval — dense in pgvector with HNSW, BM25 in parallel, fused with RRF, optionally
reranked by a cross-encoder — that **tells you what its approximate index gave up**.

```
POST /search {"query": "...", "recall_target": 0.95}

{ "hits": [...],
  "ef_search": 24,                 <- chosen to meet the target, off the measured curve
  "recall_estimate": 0.970,
  "recall_measured_now": true,     <- exact search actually ran for THIS query
  "timings": [{"stage":"embed_query","ms":0.04}, {"stage":"dense","ms":0.41},
              {"stage":"lexical","ms":3.79}, {"stage":"fuse","ms":0.03}] }
```

Every vector database hands you `ef_search` and no idea what it costs. This one ships the
curve and picks the knob from a target.

> **An approximate index is a silent trade. This one measures what it gave up.**

---

## The idea that makes it measurable

Approximate search is normally evaluated against human relevance labels, which fuses two
different questions:

1. did the index find the **nearest vectors**?
2. were the nearest vectors the **right passages**?

Only the first is the index's fault. An index can have poor recall while the end-to-end
numbers look fine, and perfect recall while the answers are wrong — one number hides both.

So question 1 is answered against **exact brute-force search**, which at this corpus size is
one matrix multiply: milliseconds, exactly right, and **needs no labels at all**. Question 2
keeps the qrels.

That split also decides where things run. Vector geometry has nothing to do with meaning, so
the index instruments are as valid on cheap deterministic vectors as on a transformer and
finish in seconds on any machine. Only the three quality instruments need real embeddings, and
those run in CI on GitHub's hardware.

---

## 1. The recall/latency frontier

```
$ python -m ragx.cli frontier            # 3,024 chunks, k=10, m=16, ef_construction=200

  ef_search    8  recall@10 0.8089  worst 0.00  perfect 0.46  p50 0.014 ms
  ef_search   16  recall@10 0.9444  worst 0.20  perfect 0.66  p50 0.018 ms
  ef_search   24  recall@10 0.9700  worst 0.50  perfect 0.74  p50 0.022 ms
  ef_search   48  recall@10 0.9850  worst 0.80  perfect 0.87  p50 0.033 ms
  ef_search  128  recall@10 0.9850  worst 0.80  perfect 0.86  p50 0.067 ms
  ef_search  256  recall@10 0.9867  worst 0.80  perfect 0.87  p50 0.110 ms

  exact brute force p50 0.158 ms      index build 150 ms
```

Three things a single number cannot say.

**It plateaus at 48.** Going from `ef_search` 48 to 256 costs **3.3× the latency for +0.002
recall**. A default picked "to be safe" pays that on every query forever.

**It never reaches 1.000.** Not at any setting. `perfect` — the share of queries where the
index returned the exact top-10 — tops out at 0.87.

**Recall is not monotonic.** 96 scores below 48. A wider search is not a superset of a
narrower one, so a tuning run that samples three points can easily order them wrongly.

And the honest reading, which the tool prints itself: **exact search is 0.158 ms here.** At
this corpus size the approximate index saves a tenth of a millisecond. The *method* transfers;
these particular numbers describe a corpus that does not need ANN yet.

## 2. Filtered search — the failure that ships

Metadata filtering is where HNSW stops behaving, and it is the least published part of the
subject. At **2.8% selectivity** (one service, one version):

```
$ python -m ragx.cli filtered

  pre    recall 0.9967   returned fewer than k  0.00 (0/60)    p50 0.319 ms
  post   recall 0.8633   returned fewer than k  0.30 (18/60)   p50 0.073 ms
  exact  recall 1.0000                                          p50 0.158 ms
```

**Pre-filtering is twice as slow as scanning the entire corpus exactly.** The graph walk still
visits forbidden nodes; it just refuses to return them, so a narrow filter spends the whole
budget on candidates it must discard. At this selectivity the approximate index has nothing
left to offer.

**Post-filtering silently returns fewer than k on 30% of filtered queries.** Not an error — a
short list. It reaches production as *"the answer was missing a source"*, weeks later, from
someone who cannot reproduce it.

pgvector's fix, `hnsw.iterative_scan`, is wired here and is **off by default upstream**, which
is why so many deployments have this failure without knowing.

## 3. The A/A arm — and the time it caught itself

Nothing distinguishes these two runs except the HNSW construction seed. Run it on **60**
queries:

```
  ef  8   0.8633  vs  0.7600   gap 0.1033   -> the test says "a better"
  ef 16   0.9633  vs  0.9417   gap 0.0217   -> the test says "a better"
```

Ten points of recall from luck, and a significance test naming a winner between two identical
builds. That was going to be the headline. Then the same command on all **180** queries:

```
$ python -m ragx.cli aa

    ef    seed A    seed B      gap   verdict
     8    0.8078    0.7989   0.0089   INCONCLUSIVE
    16    0.9433    0.9406   0.0028   INCONCLUSIVE
    32    0.9767    0.9767   0.0000   INCONCLUSIVE
    64    0.9844    0.9822   0.0022   INCONCLUSIVE
```

**The gap was the sample, not the seed.** At n=60 an unlucky subset made one build look ten
points worse; at n=180 the two agree to within 0.009 and nothing is significant.

This is the most useful thing in the repository, and it is not the finding I expected. **A
noise floor is itself an estimate and needs a sample big enough to trust.** An A/A arm run on
too few queries does not merely fail to catch noise — it manufactures some, and it is
especially convincing because it comes wearing the clothes of a rigour check.

The arm also nearly got retired: at `ef=32` the gap is exactly `0.0000` and the instrument
looks dead. It is not — the seed matters where the *graph* matters, and at high `ef_search`
the search reaches everything regardless of how the graph was wired. Checking one operating
point would have thrown away a working instrument.

## 4. Does the index loss reach the answer?

```
$ python -m ragx.cli endtoend

  ef   8 (clamped to 20 by depth)  index recall 0.9533   mrr 0.6358   INCONCLUSIVE vs exact
  ef  24                           index recall 0.9508   mrr 0.6371   INCONCLUSIVE
  ef  48                           index recall 0.9817   mrr 0.6349   INCONCLUSIVE
  ef 256                           index recall 0.9875   mrr 0.6350   INCONCLUSIVE

  index recall moved 0.9533 -> 0.9875   mrr moved 0.6358 -> 0.6350
```

**Index recall moves three and a half points; end-to-end quality moves less than one
thousandth — in the wrong direction.** Fusion
and a top-k of five absorb a missing neighbour completely — until they do not, and nothing in
the end-to-end metric will tell you when that happens.

That is the whole argument for measuring the index against exact search: **the number everyone
reports cannot see the thing that is degrading.**

## 5. Which retriever wins which query

The identifier and paraphrase queries are **twins**: built from the same facts, aimed at the
same chunks, differing only in phrasing. Comparing lexical against dense on different query
subsets measures the subsets as much as the retrievers.

```
$ python -m ragx.cli querytype                                   [hash vectors]

  identifier   dense 0.067   lexical 1.000   hybrid 0.631
               hybrid vs the better half:  -0.3689  p=0.000   the hybrid LOSES
```

On queries where one retriever is useless, **RRF is significantly worse than that retriever's
better half alone** — fusing a meaningless ranking drags the right chunk down. Hybrid search is
sold as free insurance; it is not, and this is the instrument that says so.

---

## Where the numbers come from

**Everything above is hash-embedded vectors, and that is labelled rather than buried.** The
index instruments (§1–3) are valid on any vectors — that is the point of separating index
recall from retrieval quality. §4 and §5 need real semantics to mean anything about a
production system, and the shapes shown are what the harness produces, not a claim about BGE.

The `quality` CI job runs the same commands with **BGE-small ONNX embeddings** on GitHub's
runner and uploads the results as an artifact. It lives there rather than in a local script
because the machine this was written on is a laptop somebody is also using.

Latency is reported with that caveat too: a p50 measured on a shared vCPU is partly a number
about the vCPU. **Recall is the machine-independent axis and leads every table.**

## Ground truth without annotation

3,024 chunks of generated platform documentation; 180 queries, each authored **from** one
specific chunk — so the relevant chunk is exact by construction and nobody labelled anything.

The **version axis** does three jobs at once. The same section exists at 2.2, 2.3 and 2.4
differing in one number (`147`, `148`, `149 requests`), which gives near-duplicate distractors
for a realistic reason: a retriever ignoring the version returns a passage that **reads
correctly and answers wrongly**. It gives a metadata filter with real selectivity — version
alone keeps 33%, service-and-version keeps 1.4%. And it gives three query styles from one
fact.

Two defects were caught by reading the generated output rather than the code:

- **Broken qrels.** "What is the default value of `GATEWAY_RETRY_MAX_ATTEMPTS`?" has three
  right answers, one per version, and one was labelled. The relevant set is now *derived from
  the fact key*, which also gets error codes right for free — those differ per version and
  genuinely have one answer.
- **A self-contradicting corpus.** The fact key repeated every N indices inside one version, so
  four chunks each claimed a different value for the same limit. A corpus that contradicts
  itself has no ground truth at all. A gate now fails the command.

`corpus.jsonl` is 8 MB of templated prose and a pure function of `ragx/corpus.py` plus a seed,
so the **qrels and a sha256 are committed and the corpus is not**. CI regenerates it and fails
if the hash moves.

## Stack — one line each, and what was refused

| | the job it does |
|---|---|
| **pgvector + HNSW** | The dense index under measurement and the production path. |
| **hnswlib** | The same index in-process, so the sweep runs with no Docker — and, more usefully, it exposes pre- and post-filtering *explicitly*, which pgvector does not. |
| **NumPy** | Exact brute force: the free, perfect, label-free answer key. |
| **fastembed (ONNX)** | BGE-small and the cross-encoder on CPU. No torch, so `pip install` is genuinely enough. |
| **rank-bm25** | The lexical half, with a tokenizer that keeps identifiers whole. |
| **SQLAlchemy + psycopg** | Schema and access; exercised against a real server in CI. |
| **FastAPI** | `/search` with the recall contract, `/health`, `/metrics`. |
| **Pydantic v2** | Typed contracts; a hit is never a loose dict. |
| **OpenTelemetry** | One span per retrieval stage — the latency axis is read off these. |
| **Docker + compose** | pgvector and the API in one command. |
| **Hypothesis** | Property tests on RRF and the recall maths. |

**Refused, and why.** *Pinecone / Weaviate / Qdrant* — the point is to measure the index
against its own exact baseline, and a hosted one cannot be. *LangChain / LlamaIndex* — they
would hide the very retrieval stages this repo exists to time. *Kubernetes* — one compose file
is the honest deployment here. *A second dashboard.*

### Two design decisions worth arguing about

**Both index implementations exist, and not for redundancy.** With pgvector you cannot tell
from the SQL whether a filtered query was pre- or post-filtered: you write one `WHERE` clause
and the *planner* decides. Which one you got is invisible in the result and very visible in
the recall — so `EXPLAIN` is captured, and the CI integration job asserts pgvector and hnswlib
agree on recall. If they ever diverge, the local frontier does not describe the deployed index.

**BM25 is the slowest stage**, at 3.79 ms against 0.41 ms for the vector index. The latency
budget of a hybrid retriever is dominated by the lexical half, not by the ANN everyone tunes.

## What is verified where

**Docker is not installed on the machine this was written on**, so `docker compose up` was
never run here and this README does not claim it was. Four CI jobs:

1. **unit** — ruff, mypy, 31 hermetic tests, and every label-free instrument with its gates.
2. **integration** — a real `pgvector/pgvector:pg16`: the index built for real, pgvector's
   recall compared against hnswlib's, `EXPLAIN` captured, and the float round-trip checked.
3. **quality** — the only expensive job: real BGE embeddings, results uploaded as an artifact.
4. **container** — compose validates, the image builds, the running container answers a search.

## Running it

```bash
pip install -r requirements-dev.txt
python -m ragx.cli corpus --write        # instant, no model
pytest -q                                # 31 tests, ~2 s

python -m ragx.cli --embedder hash frontier      # the curve, in seconds
python -m ragx.cli --embedder hash filtered
python -m ragx.cli --embedder hash search "gateway retry attempts" --verify

python -m ragx.cli --embedder onnx --yes build   # the one slow command; cached forever
docker compose up
```

`--yes` is required before the ONNX pass because it is the only command here that will occupy
a CPU for minutes, and it names the cheap alternative when it refuses.

## Limits, stated

- **One corpus, one embedding model, one date.** The frontier's *shape* moves with corpus
  size: HNSW recall at a given `ef_search` degrades as the corpus grows.
- **This corpus is small enough for exact search**, which is precisely the regime where ANN is
  least necessary. Said out loud because it is the first thing a reader should ask.
- **180 queries is not many.** The A/A arm demonstrates the consequence on itself: at n=60 it
  reported a 0.10 gap between two identical builds, and at n=180 the same comparison is 0.009
  and inconclusive. Read every effect here against that.
- The quality instruments (§4, §5) are shown on hash vectors, where dense retrieval is
  meaningless by construction. They demonstrate that the instruments work; the CI `quality`
  job is where they say anything about a real embedding model.

## License

MIT
