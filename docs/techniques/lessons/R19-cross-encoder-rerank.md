# R19: cross-encoder reranking

## Intuition

Retrieval finds candidates cheaply; a cross-encoder then reads the question and each candidate passage together and scores the pair, which is slower but usually sharper than comparing two separate vectors. R19 reorders (and may shorten) a candidate list. It never adds a candidate.

## Mechanism

`CrossEncoderReranker.rerank(question, result)` takes one ordered `RetrievalResult` (for example R02, R06 or R07 output), sends the question and the hit texts to a `Reranker` port with `top_n=k`, and applies the returned order. Every output hit keeps its source anchors and its earlier score components, and gains two components: `input_rank` (its rank in the input) and `rerank_score` (the port score). The result is labelled `scoring="cross-encoder"`, technique `R19`.

The no-add rule: the output may only contain input chunks (same chunk id and same extraction view and offsets). A reply with an unknown or duplicate index, a non-finite score or a non-descending order raises `rerank_invalid_order`; an added or altered candidate raises `rerank_added_candidate`. Chunks left out by `k` are listed in `trace.dropped`. The input is checked first (one corpus result, at most `max_candidates`).

## Config and defaults

`CrossEncoderRerankConfig`: `k=5` (1 to 50), `max_candidates=50` (1 to 100). With no reranker port the plugin raises `TechniqueUnavailableError` ("needs the [sparse] extra and an approved local cross-encoder model"); it never falls back to another scorer under the R19 label. The port must have kind `reranker`.

## Fidelity

Cross-encoder reranking follows the mechanism of Nogueira and Cho, "Passage Re-ranking with BERT" (2019, arXiv 1901.04085): the query and passage are scored jointly by one encoder. The approved model is a different, multilingual one: `BAAI/bge-reranker-v2-m3` (Apache-2.0, revision `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`), run locally on CPU with a Sigmoid activation, so scores are in 0 to 1. The paper's BERT model, fine-tuning data and evaluation are not reproduced here.

The offline demo does not use that model. It uses `fake-reranker-v1`, a Jaccard word-overlap test double that is not a cross-encoder; its rows show the mechanics only and say nothing about cross-encoder quality.

## Metering

Each rerank is one reranker-port call, carried in `trace.usage` (input tokens from the port, no output tokens, cost from the profile). Retrieval calls and any chat calls are counted separately. The usage table of the demo keeps retrieval calls, reranker-port calls and rerank chat calls in separate columns.

```text
| scenario | retrieval calls | reranker-port calls (R19) | rerank chat calls (R20) | input tokens | output tokens | cost USD |
|---|---|---|---|---|---|---|
| R19 fake reranker, R02 order | 0 | 1 | 0 | 66 | 0 | 0 |
| R19 fake reranker, reversed order | 0 | 1 | 0 | 66 | 0 | 0 |
```

## Runnable demo

The offline demo `docs/techniques/lessons/demos/p55_rerank_demo.py` calls the real package code with the FakeReranker test double and the test-only `LexicalTestEmbedder`. It writes [the evidence file](../../evidence/P5.5/reranking_and_inspection.md); every offline row below is quoted from it. The offline part prints no timing values.

```bash
PYTHONIOENCODING=utf-8 uv run python docs/techniques/lessons/demos/p55_rerank_demo.py
```

## Observations (quoted exactly)

Success (mechanics): when the input has the fact last, the reranker moves it to rank 1 and drops the chunk it ranks lowest.

```text
| chunk | before rank | after rank | moved | input_rank | rerank_score |
|---|---|---|---|---|---|
| fact_en | 6 | 1 | +5 | 6 | 0.294118 |
| dist03 | 2 | 2 | +0 | 2 | 0.062500 |
| dist06 | 4 | 3 | +1 | 4 | 0.062500 |
| dist04 | 1 | 4 | -3 | 1 | 0.058824 |
| dist07 | 3 | 5 | -2 | 3 | 0.052632 |
| dist01 | 5 | - | - | - | - |
Dropped by `k` (no longer in the output): dist01. Scoring label: `cross-encoder`. Candidates added by the reranker: none (a reranker may only reorder or drop its input; an added candidate raises `rerank_added_candidate`).
```

Rows are copied from the evidence file; `...` marks skipped lines.

Losing case: when the fact is already first, reranking changes nothing useful. The distractors are reshuffled by word overlap alone (dist01 goes from rank 2 to rank 5, dist07 is dropped), so the move is noise, not relevance.

```text
| chunk | before rank | after rank | moved | input_rank | rerank_score |
|---|---|---|---|---|---|
| fact_en | 1 | 1 | +0 | 1 | 0.294118 |
| dist06 | 3 | 2 | +1 | 3 | 0.062500 |
| dist03 | 5 | 3 | +2 | 5 | 0.062500 |
| dist04 | 6 | 4 | +2 | 6 | 0.058824 |
| dist01 | 2 | 5 | -3 | 2 | 0.052632 |
| dist07 | 4 | - | - | - | - |
...
Trace: status `applied`, input 6, output 5, reranker-port calls 1. Fact `fact_en` rank 1 -> 1.
```

Forged stage: the inspection view refuses a rerank stage that adds a candidate.

```text
| rerank stage adds a candidate | raised rerank_added_candidate |
```

## Live smoke (orchestrator, local model, not part of the offline demo)

One live run on 2026-11-09 with the owner-approved `BAAI/bge-reranker-v2-m3` at revision `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e` (Apache-2.0, Sigmoid activation, CPU, cost 0): `RAG_LIVE_RERANK=1 RAG_RERANK_MODEL_DIR=var/models/bge-reranker-v2-m3 uv run pytest tests/live/test_cross_encoder_live.py -m live -v -s` gave 1 passed in 22.32s. Quoted from the evidence file, live section:

```text
| EN: "How often does the Port Sallow observatory record sea levels?" | first (input index 14 of 15) | 0.999503 | all 14 at 1.6e-05 | 455 | 954.7 |
| AR: the Arabic query of the live test | first | 0.999745 | all 14 at 1.6e-05 | 414 | 1025.9 |
```

These latencies (about 1 s for 15 candidates on CPU) are live figures from one run, not offline numbers. One run on one fixture with a single fact is a smoke test, not an evaluation.

## Limitations

- The 14 distractor scores saturate at the same value (1.6e-05), so their relative order carries no information. Only "the fact is first" is evidenced.
- The top-k is limited by the candidates: a fact the first stage did not retrieve cannot be rescued.
- Latency grows with candidates times passage length; `max_candidates` bounds it.
- No benchmark, no quality claim. Evaluation belongs to P6.1.

## Status

Implemented; the live test passed with the approved local model. Not validated or benchmarked. The offline demo shows mechanics with a test double and says nothing about cross-encoder quality.
