# R14: Multi-query expansion

## Intuition

One question can be phrased several ways. R14 asks the model for a few alternative queries, searches each, and merges the results.

## Mechanism

One metered chat call returns up to `n_variants` queries. The original question is always searched first. Variants are deduplicated after NFC normalisation and casefolding (`duplicates_dropped`). A variant that drops a hard constraint of the question (a number, quoted phrase or negation) is recorded, and by default not searched; the trace status becomes `constraint_loss`. Each query goes to the same retriever. Results are merged by best rank: a chunk keeps its best rank over all queries, ties go to the earlier query. More than `n_variants` queries raises `too_many_queries` even with `on_invalid="original"`.

## Config and defaults

`MultiQueryConfig`: `n_variants=3` (1 to 5), `k=5`, `max_output_tokens=256`, `temperature=0.0`, `drop_lossy_variants=True`, `on_invalid="fail"`.

## Fidelity

Adaptation. Multi-query expansion is a common pattern without one reference paper. Best-rank union keeps the original question's ordering authority and does not pool scores from different queries.

## Metering

One chat call plus one retrieval (and one query embedding for dense search) per searched query, original included. The row below has three retrieval calls because one of the three scripted variants duplicated the question and was dropped before searching.

```text
| technique | scenario | transform chat calls | transform embed calls | retrieval calls | retrieval embeds | transform input tokens | transform output tokens | cost USD |
|---|---|---|---|---|---|---|---|---|
| R14 | ablation query, 3 variants (1 duplicate) | 1 | 0 | 3 | 3 | 48 | 13 | 0 |
```

## Runnable demo

The offline demo `docs/techniques/lessons/demos/p54_query_transforms_demo.py` calls the real package code with a scripted chat (fixed JSON replies, not a model) and the test-only `LexicalTestEmbedder` (not a retriever quality signal). It writes [the evidence file](../../evidence/P5.4/query_transforms.md); every row below is quoted from it.

```bash
PYTHONIOENCODING=utf-8 uv run python docs/techniques/lessons/demos/p54_query_transforms_demo.py
```

## Observations (quoted exactly)

Same query and the same scripted variants for R02, R14 and R18. Success (expansion, R14 against R02): the fact is not in the top 5 of the original question and is rank 2 after expansion.

```text
| method | scoring | fact_en rank | top 5 |
|---|---|---|---|
| R02 original only | cosine | not in top 5 | dist06, dist07, dist03, dist04, dist02 |
| R14 union (best rank) | multi-query-union-best-rank | 2 | dist06, fact_en, dist07, fact_ar, dist03 |
- Expansion (R14 vs R02): fact_en rank not in top 5 -> 2.
```

Per-variant new chunks and duplicates:

```text
| query | fact_en rank in this query | results | new chunks vs earlier queries |
|---|---|---|---|
| `What is the measurement schedule for the coastal station?` | not in top 5 | dist06, dist07, dist03, dist04, dist02 | 5 (dist06, dist07, dist03, dist04, dist02) |
| `Port Sallow sea levels` | 1 | fact_en, fact_ar, museum, dist03, dist08 | 4 (fact_en, fact_ar, museum, dist08) |
| `tidal observatory records sea levels` | 1 | fact_en, fact_ar, museum, dist03, dist08 | 0 (-) |
`duplicates_dropped`: R14 1, R18 1. Queries searched: R14 3, R18 3 (original first). Trace status: R14 applied, R18 applied.
```

The second variant added no new chunks, so its extra search cost one embedding for no gain.

Losing case: variants that drop the number `17` are dropped and only the original is searched, so the expansion adds nothing.

```text
| losing case: variants without the number 17 | status | lost_constraints | queries searched | lossy variants dropped |
|---|---|---|---|---|
| default drop_lossy_variants | constraint_loss | 17 | 1 | 2 |
```

## Live smoke

One live smoke run on 2026-11-09 (owner OK; local `gemma4:e4b` chat and `bge-m3` embeddings, cost 0; `RAG_LIVE_OLLAMA=1 uv run pytest tests/live/test_query_transforms_live.py -m live -v -s`) used one question: "How often does the Port Sallow observatory record sea levels?" It is one run on one question and is not an evaluation. R14 returned `applied` with 2 variants plus the original (84 input, 26 output tokens); the fact was first.

## Limitations

Extra queries cost extra embeddings and latency. Best-rank union lets one query's top hit beat a chunk found by several queries, which is what R18 changes. Variants are only as good as the model that writes them.

## Status

Implemented; not validated or benchmarked. The demo uses a scripted chat, so it shows how the code handles given replies and says nothing about model quality.
