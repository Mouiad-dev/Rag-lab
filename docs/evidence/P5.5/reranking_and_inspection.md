Generated 2026-11-09 by `PYTHONIOENCODING=utf-8 uv run python docs/techniques/lessons/demos/p55_rerank_demo.py`; offline fixture; FakeReranker test double and scripted chat; no model; not a benchmark.

# P5.5 reranking and inspection: demo output

Offline fixture. The R19 reranker port is the FakeReranker test double and the R20 chat is a scripted responder that returns fixed JSON. Neither is a model, and the embedder is the test-only LexicalTestEmbedder. Every table value is computed by the package code from those inputs. Token counts are fake usage (whitespace word counts), cost is 0, and no timing values are printed.

Chunk labels: fact_en and fact_ar are the two fact paragraphs, museum is the long paragraph, dist00-dist13 are the distractors in fixture order. R02 dense search uses k=6.

## R19 cross-encoder reranking, mechanics only: input R02 order as retrieved

Reranker port: fake-reranker-v1: Jaccard word-overlap test double; not a cross-encoder; shows mechanics only.

Question: `How often does the Port Sallow observatory record sea levels?`. Input: R02 dense top 6. R19 config: k=5.

| chunk | before rank | after rank | moved | input_rank | rerank_score |
|---|---|---|---|---|---|
| fact_en | 1 | 1 | +0 | 1 | 0.294118 |
| dist06 | 3 | 2 | +1 | 3 | 0.062500 |
| dist03 | 5 | 3 | +2 | 5 | 0.062500 |
| dist04 | 6 | 4 | +2 | 6 | 0.058824 |
| dist01 | 2 | 5 | -3 | 2 | 0.052632 |
| dist07 | 4 | - | - | - | - |

Dropped by `k` (no longer in the output): dist07. Scoring label: `cross-encoder`. Candidates added by the reranker: none (a reranker may only reorder or drop its input; an added candidate raises `rerank_added_candidate`).

Trace: status `applied`, input 6, output 5, reranker-port calls 1. Fact `fact_en` rank 1 -> 1.

## R19 cross-encoder reranking, mechanics only: input R02 top 6 in reverse order (fact_en at rank 6)

Reranker port: fake-reranker-v1: Jaccard word-overlap test double; not a cross-encoder; shows mechanics only.

Question: `How often does the Port Sallow observatory record sea levels?`. Input: R02 dense top 6. R19 config: k=5.

| chunk | before rank | after rank | moved | input_rank | rerank_score |
|---|---|---|---|---|---|
| fact_en | 6 | 1 | +5 | 6 | 0.294118 |
| dist03 | 2 | 2 | +0 | 2 | 0.062500 |
| dist06 | 4 | 3 | +1 | 4 | 0.062500 |
| dist04 | 1 | 4 | -3 | 1 | 0.058824 |
| dist07 | 3 | 5 | -2 | 3 | 0.052632 |
| dist01 | 5 | - | - | - | - |

Dropped by `k` (no longer in the output): dist01. Scoring label: `cross-encoder`. Candidates added by the reranker: none (a reranker may only reorder or drop its input; an added candidate raises `rerank_added_candidate`).

Trace: status `applied`, input 6, output 5, reranker-port calls 1. Fact `fact_en` rank 6 -> 1.

## R20 listwise LLM reranking (scripted chat, not a model)

The applied case reranks the reversed R02 list so that the fact starts at rank 6; the other cases use the R02 order. Reranker: R20, k=5, unranked `append`. Each scripted reply is fixed JSON `{"ranking": [...]}`. In the injection case the scripted reply stands in for a model that obeyed the passage text; the code treats the passage as data and rejects the unknown id.

| case | scripted ranking | status | reason | order after | fact_en rank before -> after | chat calls | rank-position score of top hit |
|---|---|---|---|---|---|---|---|
| applied: fact first, input has fact_en at rank 6 | p6 p1 p2 p3 p4 p5 | applied | - | fact_en, dist04, dist03, dist07, dist06 | 6 -> 1 | 1 | 1.000000 |
| losing: a distractor first, fact pushed down | p2 p3 p4 p5 p6 p1 | applied | - | dist01, dist06, dist07, dist03, dist04 | 1 -> not in top 5 | 1 | 1.000000 |
| rejected: unknown id `p99` | p1 p99 | rejected | unknown_id | fact_en, dist01, dist06, dist07, dist03, dist04 | 1 -> 1 | 1 | - |
| injection: a passage asks for `p99` | p99 | rejected | unknown_id | fact_en, dist01, dist06, dist07, dist03, dist04 | 1 -> 1 | 1 | - |

The rank-position score is `1 / position` from the list order; it is not a model probability. A rejected reply keeps the input order and the input result unchanged. The losing case is applied by the code exactly as scripted: R20 cannot tell that the reply is wrong.

## R20 order probe (scripted responders, not a model)

The probe runs the reranker twice, once on the input order and once on the reversed input order, and compares the output (2 chat calls). `agreement_at_k` is the overlap of the two top-k sets.

| responder | forward status | reversed status | agreement_at_k (k=5) | same_order | chat calls |
|---|---|---|---|---|---|
| position-biased (keeps the order it is given) | applied | applied | 4 | False | 2 |
| content-based (word overlap, ties by text) | applied | applied | 5 | True | 2 |

A responder that follows the input position fails the probe (the order flips with the input); a responder that follows content passes it. A passing probe on one input does not show the order is stable in general.

## Inspection: R03 -> R07 (R02 + R03) -> R20

Read-only stage view built by `inspect_retrieval` (scope and index version are rechecked). Stages are R03 BM25, R07 RRF over the R02 dense and R03 lists, then R20 with a scripted reply that puts the fact first.

| stage | technique | scoring | candidates (rank order) | changes vs previous stage |
|---|---|---|---|---|
| retrieve | R03 | bm25-okapi-lucene-idf | fact_en, dist06, dist07, dist03, dist04, dist02 | - |
| fuse | R07 | rrf-k60 | fact_en, dist06, dist07, dist03, dist04, dist01 | dist02 dropped from 6; dist01 added at 6 |
| rerank | R20 | llm-listwise-rank | fact_en, dist06, dist07, dist03, dist04 | dist01 dropped from 6 |

| forged rerank stage (first hit replaced by a chunk that was not in the previous stage) | outcome |
|---|---|
| rerank stage adds a candidate | raised rerank_added_candidate |

The `fuse` stage may add candidates; the `rerank` stage may not. The HTTP inspection endpoint is deferred until runs are stored.

## Usage (one row per scenario, fake usage, cost 0)

Retrieval calls are `retriever.retrieve` invocations in this script (the shared R02 input and the R03 stage; the R07 fuse is not a retrieval). Reranker-port calls are R19 calls. Rerank chat calls are R20 calls and include the 2 calls of each order probe. Tokens are the usage of the rerank calls only; the retrieval embeds are on the lexical test embedder and report no tokens.

| scenario | retrieval calls | reranker-port calls (R19) | rerank chat calls (R20) | input tokens | output tokens | cost USD |
|---|---|---|---|---|---|---|
| shared inputs (R02 k=6, R03 k=6) | 2 | 0 | 0 | 0 | 0 | 0 |
| R19 fake reranker, R02 order | 0 | 1 | 0 | 66 | 0 | 0 |
| R19 fake reranker, reversed order | 0 | 1 | 0 | 66 | 0 | 0 |
| R20 applied | 0 | 0 | 1 | 123 | 87 | 0 |
| R20 losing | 0 | 0 | 1 | 123 | 87 | 0 |
| R20 rejected | 0 | 0 | 1 | 123 | 87 | 0 |
| R20 injection | 0 | 0 | 1 | 122 | 86 | 0 |
| R20 order probe, position-biased | 0 | 0 | 2 | 246 | 174 | 0 |
| R20 order probe, content-based | 0 | 0 | 2 | 246 | 174 | 0 |
| inspection chain R20 | 0 | 0 | 1 | 123 | 87 | 0 |
| total | 2 | 2 | 9 | 1238 | 782 | 0 |

Cross-check against the doubles themselves: scripted chats received 9 requests (table: 9); the fake reranker received 2 calls (table: 2); embedder query embed calls: 1; dense retrievals in the script: 1; retrievals in the table: 2 (the remaining 1 is R03, which embeds nothing).

## Live run (orchestrator)

Two single smoke runs on 2026-11-09, both local with cost 0. R19 is one run on two queries (EN and AR); R20 is one run on one question; neither is an evaluation or a benchmark. They are separate from the offline demo above and are excluded from its comparison.

### R19 live (owner-approved local model)

Model: `BAAI/bge-reranker-v2-m3`, revision `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`, licence Apache-2.0, Sigmoid activation, CPU. Command: `RAG_LIVE_RERANK=1 RAG_RERANK_MODEL_DIR=var/models/bge-reranker-v2-m3 uv run pytest tests/live/test_cross_encoder_live.py -m live -v -s`. Result: 1 passed in 22.32s. Each query had 15 candidates: the fact (at index 14) and 14 distractors.

| query | fact position | fact score | distractor scores | input_tokens | latency_ms |
|---|---|---|---|---|---|
| EN: "How often does the Port Sallow observatory record sea levels?" | first (input index 14 of 15) | 0.999503 | all 14 at 1.6e-05 | 455 | 954.7 |
| AR: the Arabic query of the live test | first | 0.999745 | all 14 at 1.6e-05 | 414 | 1025.9 |

Limitation: the 14 distractor scores saturate at the same value, so their relative order carries no information.

### R20 live smoke (local Ollama `gemma4:e4b`, owner OK)

Command: `RAG_LIVE_OLLAMA=1 uv run pytest tests/live/test_llm_rerank_live.py -m live -v -s`. Result: 1 passed in 16.81s. Input had the fact at rank 5 (`live-4`). Output order: live-4, live-0, live-1, live-2, live-3 (the fact moved to rank 1). Order probe: forward applied, reversed applied. Three calls (the rerank and the 2 probe calls), each 201 input and 8 output tokens; latency 15203 ms (first call, includes model load), 479 ms, 514 ms.
