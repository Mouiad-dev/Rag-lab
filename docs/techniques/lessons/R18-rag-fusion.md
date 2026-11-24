# R18: RAG fusion

## Intuition

Like R14, but the per-query rankings are combined with reciprocal rank fusion, so a chunk found by several queries is boosted.

## Mechanism

Same variant generation as R14 (one metered chat call, original first, deduplicated, constraint-lossy variants dropped by default). Each query is retrieved separately, and a chunk's score is the sum of `1 / (rrf_k + rank)` over the queries that returned it. It reuses R07's `rrf_contribution`. Each hit keeps one `q<i>_rrf` score component per query. Ties are broken by best rank, then extraction view and start/end offsets, then chunk id.

## Config and defaults

`RagFusionConfig` extends `MultiQueryConfig` with `rrf_k=60` (1 to 1000); other defaults as R14.

## Fidelity

Adaptation of the RAG-Fusion approach (Rackauckas, 2024, arXiv 2402.03367, "RAG-Fusion: a New Take on Retrieval-Augmented Generation"): generated queries plus reciprocal rank fusion. This implementation fuses retrieval rankings only and does not include the paper's later answer-generation steps.

## Metering

Same as R14: one chat call and one retrieval and embedding per searched query.

```text
| technique | scenario | transform chat calls | transform embed calls | retrieval calls | retrieval embeds | transform input tokens | transform output tokens | cost USD |
|---|---|---|---|---|---|---|---|---|
| R18 | ablation query, 3 variants (1 duplicate) | 1 | 0 | 3 | 3 | 48 | 13 | 0 |
```

## Runnable demo

The offline demo `docs/techniques/lessons/demos/p54_query_transforms_demo.py` calls the real package code with a scripted chat (fixed JSON replies, not a model) and the test-only `LexicalTestEmbedder` (not a retriever quality signal). It writes [the evidence file](../../evidence/P5.4/query_transforms.md); every row below is quoted from it.

```bash
PYTHONIOENCODING=utf-8 uv run python docs/techniques/lessons/demos/p54_query_transforms_demo.py
```

## Observations (quoted exactly)

The ablation separates expansion from fusion on one query with the same scripted variants. Expansion (R14 against R02) moved the fact into the top 5. Fusion (R18 against R14) did not change the fact's rank, and changed the top hit from one distractor (dist06) to another (dist03).

```text
| method | scoring | fact_en rank | top 5 |
|---|---|---|---|
| R02 original only | cosine | not in top 5 | dist06, dist07, dist03, dist04, dist02 |
| R14 union (best rank) | multi-query-union-best-rank | 2 | dist06, fact_en, dist07, fact_ar, dist03 |
| R18 RRF | rag-fusion-rrf-k60 | 2 | dist03, fact_en, fact_ar, museum, dist08 |
- Expansion (R14 vs R02): fact_en rank not in top 5 -> 2.
- Fusion (R18 vs R14, same candidates): fact_en rank 2 -> 2; top hit dist06 -> dist03.
```

Why, from the score components: the distractor `dist03` appears in all three lists and so collects three contributions, while `fact_en` appears in two.

```text
| R18 hit | score | components (query: rrf) |
|---|---|---|
| dist03 | 0.047123 | q0_rrf: 0.015873, q1_rrf: 0.015625, q2_rrf: 0.015625 |
| fact_en | 0.032787 | q1_rrf: 0.016393, q2_rrf: 0.016393 |
```

Losing case: fusion rewarded agreement between queries, and the agreeing chunk was the wrong one. Only one fixture query was run, so this is an illustration, not a rate.

## Live smoke

One live smoke run on 2026-11-09 (owner OK; local `gemma4:e4b` chat and `bge-m3` embeddings, cost 0; `RAG_LIVE_OLLAMA=1 uv run pytest tests/live/test_query_transforms_live.py -m live -v -s`) used one question: "How often does the Port Sallow observatory record sea levels?" It is one run on one question and is not an evaluation. R18 returned `applied` with 3 variants plus the original (84 input, 37 output tokens); the fact was first.

## Limitations

- Ties now follow source position, which is stable across index builds; chunk id is used only for identical spans.

RRF sees ranks only; a chunk that is mediocre in every list can beat one that is excellent in a single list. Variant quality and the number of queries drive cost.

## Status

Implemented; not validated or benchmarked. The demo uses a scripted chat, so it shows how the code handles given replies and says nothing about model quality.
