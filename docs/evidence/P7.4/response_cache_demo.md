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
Live run result goes here (P7.4-11). Live output is not deterministic and is recorded separately.
</orchestrator>

