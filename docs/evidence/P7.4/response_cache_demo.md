# P7.4 R50 and R51 response caches: demo output

Command: `uv run python docs/techniques/lessons/demos/p74_response_cache_demo.py`. Date: 2026-11-23. Offline; scripted answers; test-only lexical embedder; small illustrative pair sets; not a benchmark. Calls are ledger calls (one fresh answer = 1 query embedding + 1 chat). Cold versus warm is shown as calls avoided, never latency.

## 1. R50 exact cache

The R50 key covers scope, question, history, authz revision, model, prompt, index version, retriever and generation settings. Every hit is revalidated (scope, expiry, index version, anchors, source gate). Bypass requests never touch the store.

| request | outcome | reason | not stored because | calls made | calls avoided |
|---|---|---|---|---|---|
| first run | miss | not_found | - | 2 | 0 |
| repeat of the same request | hit | - | - | 0 | 2 |
| changed model | miss | not_found | pins_mismatch | 2 | 0 |
| history turn added | miss | not_found | - | 2 | 0 |
| authz revision changed | miss | not_found | - | 2 | 0 |
| index version in the key | miss | not_found | pins_mismatch | 2 | 0 |
| index rebuilt since the entry was stored | miss | index_version_changed | - | 2 | 0 |
| other tenant | miss | not_found | - | 2 | 0 |
| forced collision (store returns a foreign entry) | miss | scope_mismatch | skipped_scope_mismatch | 2 | 0 |
| source denied by the gate | miss | source_denied | - | 2 | 0 |
| entry expired (ttl 60 s, test clock) | miss | expired | - | 2 | 0 |
| pins mismatch (answer made with k=3, key says k=5) | miss | not_found | pins_mismatch | 2 | 0 |
| bypass: freshness current | bypass | freshness_current | - | 2 | 0 |
| bypass: enriched mode | bypass | enriched_mode | - | 2 | 0 |

A miss always runs a fresh answer; a foreign entry is never returned, deleted or overwritten.

The `changed model` and `index version in the key` rows show `pins_mismatch` because this demo's scripted answer is still produced with the original model and index while the key names another one. The cache refuses to store an answer whose pins differ from its key (see `pins mismatch`); with a matching answer these requests would simply be stored under their own key.

## 2. R51 semantic cache

A hit needs the same R50 partition, similarity at or above the threshold, a margin over the runner-up, equal guarded features and the R50 revalidation. Guard rows run at threshold 0.50 so the guard, not similarity, decides. Similarity is the cosine of the two question vectors computed here (the package trace reports it only on hits). The ambiguous row uses fixed vectors and two cached answers that differ.

| request | threshold | outcome | reason | similarity | calls made | calls avoided |
|---|---|---|---|---|---|---|
| paraphrase | 0.8 | hit | - | 0.96 | 0 | 2 |
| 2025 vs 2026 | 0.5 | miss | features_differ:numbers | 0.93 | 2 | 0 |
| with vs without | 0.5 | miss | features_differ:negation | 0.94 | 2 | 0 |
| AR negation | 0.5 | miss | features_differ:negation | 0.91 | 2 | 0 |
| identifier | 0.5 | miss | features_differ:identifiers | 0.83 | 2 | 0 |
| other tenant | 0.8 | miss | no_candidate | 0.96 | 2 | 0 |
| ambiguous (fixed vectors, 0.71 and 0.71) | 0.6 | miss | ambiguous | 0.71 | 2 | 0 |

## 3. Calibration on `semantic_pairs_v1.json`

24 hand-written pairs (dev and held-out split), lexical test embedder. The threshold is chosen from dev only; the package reports held-out counts only at the chosen threshold. Format: hits/total.

| threshold | dev paraphrase unguarded | dev paraphrase guarded | dev false hits unguarded | dev false hits guarded |
|---|---|---|---|---|
| 0.5 | 5/5 | 5/5 | 6/7 | 1/7 |
| 0.8 | 5/5 | 5/5 | 6/7 | 1/7 |
| 0.9 | 4/5 | 4/5 | 6/7 | 1/7 |
| 0.92 | 4/5 | 4/5 | 5/7 | 1/7 |
| 0.93 | 4/5 | 4/5 | 4/7 | 0/7 |
| 0.95 | 4/5 | 4/5 | 1/7 | 0/7 |
| 0.97 | 4/5 | 4/5 | 0/7 | 0/7 |
| 1.0 | 3/5 | 3/5 | 0/7 | 0/7 |

Chosen threshold: `0.93`.

| held-out at the chosen threshold | unguarded | guarded |
|---|---|---|
| paraphrase hits | 4/5 | 4/5 |
| false hits on adversarial pairs | 4/7 | 2/7 |

| pair | split | kind | similarity | guard differs on | at the chosen threshold |
|---|---|---|---|---|---|
| dev-06 | dev | year | 0.94 | numbers | stopped by guard |
| dev-07 | dev | negation | 0.97 | negation | stopped by guard |
| dev-08 | dev | identifier | 0.93 | identifiers | stopped by guard |
| dev-09 | dev | comparative | 0.90 | comparatives | stopped by similarity |
| dev-10 | dev | number | 0.94 | numbers | stopped by guard |
| dev-11 | dev | entity swap without digits (documented losing case) | 0.92 | - | stopped by similarity |
| dev-12 | dev | cross-language | 0.00 | script | stopped by similarity |
| held-06 | held_out | year (Arabic) | 0.90 | numbers | stopped by similarity |
| held-07 | held_out | negation | 0.96 | negation | stopped by guard |
| held-08 | held_out | identifier | 0.93 | identifiers | stopped by guard |
| held-09 | held_out | number | 0.93 | numbers | stopped by similarity |
| held-10 | held_out | month name (documented losing case) | 0.95 | - | FALSE HIT |
| held-11 | held_out | unit (documented losing case) | 0.94 | - | FALSE HIT |
| held-12 | held_out | entity swap without digits (documented losing case) | 0.93 | - | stopped by similarity |

Losing cases: a month name (held-10) and a unit (held-11) pass both similarity and the guard, so they are false hits. Entity swaps without digits (dev-11, held-12) also pass the guard (it sees no difference) and are stopped only because their similarity sits just under the threshold (held-12 rounds up to the threshold in the table but is below it); with another embedder or threshold they would be false hits too. The guard sees numbers, identifiers, quoted text, negation, comparatives and script only. Not a benchmark: a few pairs, one embedder.

## 4. Usage per stage

Answer ledger = chat and query-embedding calls of the answered run; cache-embed ledger = the R51 lookup.

| run | answer ledger calls | answer input tokens (scripted) | cache-embed calls | calls avoided |
|---|---|---|---|---|
| R50 cold | 2 | 256 | 0 | 0 |
| R50 warm | 0 | 0 | 0 | 2 |
| R51 cold | 2 | 256 | 1 | 0 |
| R51 warm | 0 | 0 | 1 | 2 |

A warm R51 hit still pays one cache-embedding call; it avoids the retrieval embedding and the chat call. R50 needs no embedding at all. Token figures are the scripted model's estimates.

## Live run (orchestrator)

<orchestrator>
One local run, not an evaluation; numbers vary between runs; no quality or speed claim.

- Command: `PYTHONIOENCODING=utf-8 RAG_LIVE_OLLAMA=1 uv run pytest tests/live/test_prefix_cag_live.py tests/live/test_response_cache_live.py -m live -v -s -p no:cacheprovider`
- Date: 2026-11-23. Ollama 0.40.17 (`ollama --version`); models `gemma4:e4b` (chat) and `bge-m3` (embeddings), already pulled, no pull, no paid call.
- Result: 4 passed in 45.85s (whole command, both files). This replaces a first run in which the stream-mode R49 calls repeated the complete-mode prompts and were therefore not cold (found in review; the test now tags each run and mode).
- R50: 3 of 3 warm calls hit with 0 model calls and answers equal to the cold answers. Cold and warm totals are printed below; they are single-run wall times, not a benchmark.
- R51 with real `bge-m3`: chosen threshold 0.87 on 24 illustrative pairs. Held-out at 0.87: paraphrase hits 5/5 (guarded and unguarded); false hits 2/7 guarded, 5/7 unguarded. False hits remain; the pair set is small and this is not a benchmark.

Printed output:

```text
tests/live/test_response_cache_live.py::test_r50_cold_then_warm_live R50 EN cold outcome=miss status=answered total_ms=1354 lookup_ms=0.0 model_calls=2 input_tokens=439 store_reason=None
R50 EN warm outcome=hit reason=None total_ms=4 lookup_ms=3.7 calls_avoided=2 input_tokens_avoided=439 model_calls=0
R50 AR cold outcome=miss status=answered total_ms=1298 lookup_ms=0.0 model_calls=2 input_tokens=421 store_reason=None
R50 AR warm outcome=hit reason=None total_ms=4 lookup_ms=3.7 calls_avoided=2 input_tokens_avoided=421 model_calls=0
R50 EN2 cold outcome=miss status=answered total_ms=1474 lookup_ms=0.0 model_calls=2 input_tokens=439 store_reason=None
R50 EN2 warm outcome=hit reason=None total_ms=4 lookup_ms=3.8 calls_avoided=2 input_tokens_avoided=439 model_calls=0
R50 hits=3 of 3
PASSED
tests/live/test_response_cache_live.py::test_r51_bge_m3_calibration_live R51 pairs=24 embed_calls=24 (separate cache ledger)
threshold  dev paraphrase hits (ug=unguarded g=guarded)  dev false hits (ug g)
0.5        ug=5 g=5 of 5  ug=7 g=1 of 7
0.51       ug=5 g=5 of 5  ug=7 g=1 of 7
0.52       ug=5 g=5 of 5  ug=7 g=1 of 7
0.53       ug=5 g=5 of 5  ug=7 g=1 of 7
0.54       ug=5 g=5 of 5  ug=7 g=1 of 7
0.55       ug=5 g=5 of 5  ug=7 g=1 of 7
0.56       ug=5 g=5 of 5  ug=7 g=1 of 7
0.57       ug=5 g=5 of 5  ug=7 g=1 of 7
0.58       ug=5 g=5 of 5  ug=7 g=1 of 7
0.59       ug=5 g=5 of 5  ug=7 g=1 of 7
0.6        ug=5 g=5 of 5  ug=7 g=1 of 7
0.61       ug=5 g=5 of 5  ug=7 g=1 of 7
0.62       ug=5 g=5 of 5  ug=7 g=1 of 7
0.63       ug=5 g=5 of 5  ug=7 g=1 of 7
0.64       ug=5 g=5 of 5  ug=7 g=1 of 7
0.65       ug=5 g=5 of 5  ug=7 g=1 of 7
0.66       ug=5 g=5 of 5  ug=7 g=1 of 7
0.67       ug=5 g=5 of 5  ug=7 g=1 of 7
0.68       ug=5 g=5 of 5  ug=7 g=1 of 7
0.69       ug=5 g=5 of 5  ug=7 g=1 of 7
0.7        ug=5 g=5 of 5  ug=7 g=1 of 7
0.71       ug=5 g=5 of 5  ug=7 g=1 of 7
0.72       ug=5 g=5 of 5  ug=7 g=1 of 7
0.73       ug=5 g=5 of 5  ug=7 g=1 of 7
0.74       ug=5 g=5 of 5  ug=7 g=1 of 7
0.75       ug=5 g=5 of 5  ug=7 g=1 of 7
0.76       ug=5 g=5 of 5  ug=7 g=1 of 7
0.77       ug=5 g=5 of 5  ug=6 g=1 of 7
0.78       ug=5 g=5 of 5  ug=6 g=1 of 7
0.79       ug=5 g=5 of 5  ug=6 g=1 of 7
0.8        ug=5 g=5 of 5  ug=6 g=1 of 7
0.81       ug=5 g=5 of 5  ug=6 g=1 of 7
0.82       ug=5 g=5 of 5  ug=6 g=1 of 7
0.83       ug=5 g=5 of 5  ug=6 g=1 of 7
0.84       ug=5 g=5 of 5  ug=6 g=1 of 7
0.85       ug=5 g=5 of 5  ug=6 g=1 of 7
0.86       ug=5 g=5 of 5  ug=6 g=1 of 7
0.87       ug=5 g=5 of 5  ug=4 g=0 of 7
0.88       ug=5 g=5 of 5  ug=4 g=0 of 7
0.89       ug=5 g=5 of 5  ug=4 g=0 of 7
0.9        ug=5 g=5 of 5  ug=4 g=0 of 7
0.91       ug=5 g=5 of 5  ug=2 g=0 of 7
0.92       ug=5 g=5 of 5  ug=2 g=0 of 7
0.93       ug=5 g=5 of 5  ug=1 g=0 of 7
0.94       ug=5 g=5 of 5  ug=1 g=0 of 7
0.95       ug=5 g=5 of 5  ug=1 g=0 of 7
0.96       ug=5 g=5 of 5  ug=1 g=0 of 7
0.97       ug=5 g=5 of 5  ug=0 g=0 of 7
0.98       ug=4 g=4 of 5  ug=0 g=0 of 7
0.99       ug=3 g=3 of 5  ug=0 g=0 of 7
1.0        ug=0 g=0 of 5  ug=0 g=0 of 7
R51 chosen_threshold=0.87 reason=None
R51 held_out paraphrase_hits_guarded=5/5 paraphrase_hits_unguarded=5/5 false_hits_guarded=2/7 false_hits_unguarded=5/7
PASSED
```
</orchestrator>

