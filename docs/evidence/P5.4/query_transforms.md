Command: `PYTHONIOENCODING=utf-8 uv run python docs/techniques/lessons/demos/p54_query_transforms_demo.py` | Date: 2026-11-09 | offline fixture; scripted chat, test-only lexical embedder; no model; not a benchmark
# P5.4 query transforms: demo output

Offline fixture. The chat is a scripted responder that returns fixed JSON, and the embedder is the test-only LexicalTestEmbedder. Neither is a model. Every table value is computed by the package code from those replies.

Chunk labels: fact_en and fact_ar are the two fact paragraphs, museum is the long paragraph, dist00-dist12 are the distractors in fixture order. Dense search uses k=5.

## Metering (one row per technique, fake usage, cost 0)

Transform calls come from `trace.usage`. Retrieval calls are `len(trace.retrieval_queries)` (R17: one index search). Retrieval embeds are `embedder.calls` deltas minus the transform embeds. Tokens are the fake usage of the transform calls only.

| technique | scenario | transform chat calls | transform embed calls | retrieval calls | retrieval embeds | transform input tokens | transform output tokens | cost USD |
|---|---|---|---|---|---|---|---|---|
| R13 | applied rewrite | 1 | 0 | 1 | 1 | 44 | 14 | 0 |
| R14 | ablation query, 3 variants (1 duplicate) | 1 | 0 | 3 | 3 | 48 | 13 | 0 |
| R15 | two subquestions | 1 | 0 | 3 | 3 | 54 | 17 | 0 |
| R16 | step-back hit | 1 | 0 | 2 | 2 | 41 | 11 | 0 |
| R17 | success, defaults | 1 | 2 | 1 | 0 | 46 | 13 | 0 |
| R18 | ablation query, 3 variants (1 duplicate) | 1 | 0 | 3 | 3 | 48 | 13 | 0 |

## R13 question rewriting (scripted chat, not a model)

| case | status | lost_constraints | question kept in trace | retrieval queries | usage calls | fact_en rank, original question | fact_en rank, queries used |
|---|---|---|---|---|---|---|---|
| applied | applied | - | yes | `Port Sallow observatory records sea levels` | 1 | 3 | 1 |
| drops 17 (detected) | constraint_loss | 17 | yes | `Does the Port Sallow observatory record sea levels every 17 minutes?` | 1 | 1 | 1 |
| drops a place name (not detected) | applied | - | yes | `What does the tidal observatory record?` | 1 | 1 | 1 |

Heuristic limit: the hard-constraint check covers numbers, quoted phrases and negations only. A dropped place name has empty `lost_constraints` and status `applied`.

## R02 vs R14 vs R18: expansion and fusion (same query, same scripted variants)

Question: `What is the measurement schedule for the coastal station?`

Scripted variants (same reply for R14 and R18): `Port Sallow sea levels`, `tidal observatory records sea levels`, `WHAT IS THE MEASUREMENT SCHEDULE FOR THE COASTAL STATION?`. The third is the question in capitals, a duplicate after casefolding.

| method | scoring | fact_en rank | top 5 |
|---|---|---|---|
| R02 original only | cosine | not in top 5 | dist06, dist07, dist03, dist04, dist02 |
| R14 union (best rank) | multi-query-union-best-rank | 2 | dist06, fact_en, dist07, fact_ar, dist03 |
| R18 RRF | rag-fusion-rrf-k60 | 2 | dist03, fact_en, fact_ar, museum, dist08 |

- Expansion (R14 vs R02): fact_en rank not in top 5 -> 2.
- Fusion (R18 vs R14, same candidates): fact_en rank 2 -> 2; top hit dist06 -> dist03.

| query | fact_en rank in this query | results | new chunks vs earlier queries |
|---|---|---|---|
| `What is the measurement schedule for the coastal station?` | not in top 5 | dist06, dist07, dist03, dist04, dist02 | 5 (dist06, dist07, dist03, dist04, dist02) |
| `Port Sallow sea levels` | 1 | fact_en, fact_ar, museum, dist03, dist08 | 4 (fact_en, fact_ar, museum, dist08) |
| `tidal observatory records sea levels` | 1 | fact_en, fact_ar, museum, dist03, dist08 | 0 (-) |

`duplicates_dropped`: R14 1, R18 1. Queries searched: R14 3, R18 3 (original first). Trace status: R14 applied, R18 applied.

| R18 hit | score | components (query: rrf) |
|---|---|---|
| dist03 | 0.047123 | q0_rrf: 0.015873, q1_rrf: 0.015625, q2_rrf: 0.015625 |
| fact_en | 0.032787 | q1_rrf: 0.016393, q2_rrf: 0.016393 |
| fact_ar | 0.032258 | q1_rrf: 0.016129, q2_rrf: 0.016129 |
| museum | 0.031746 | q1_rrf: 0.015873, q2_rrf: 0.015873 |
| dist08 | 0.030769 | q1_rrf: 0.015385, q2_rrf: 0.015385 |

| losing case: variants without the number 17 | status | lost_constraints | queries searched | lossy variants dropped |
|---|---|---|---|---|
| default drop_lossy_variants | constraint_loss | 17 | 1 | 2 |

## R15 question decomposition (scripted chat, not a model)

Question: `How often does the observatory record sea levels and who repaints the bridge every spring?`

| id | depends_on | text |
|---|---|---|
| s1 | - | How often does the Port Sallow observatory record sea levels? |
| s2 | s1 | Who repaints the wooden bridge every spring? |

| method | status | fact_en rank | dist03 rank (bridge) | top 5 |
|---|---|---|---|---|
| R02 original question | - | 2 | 1 | dist03, fact_en, dist07, dist06, dist05 |
| R15 original + subquestions | applied | 2 | 1 | dist03, fact_en, dist01, dist07, dist06 |

Queries searched: 3 (original first, then s1, s2). Dependencies are recorded and validated; answers are not fed forward.

| scripted reply | on_invalid | outcome | retrieval calls |
|---|---|---|---|
| s1 depends_on s2 (a later id) | fail | raised invalid_dependency | 0 |
| s1 depends_on s2 (a later id) | original | raised invalid_dependency | 0 |

| losing case: subquestions without 17 | status | lost_constraints | queries searched |
|---|---|---|---|
| default on_constraint_loss=original | constraint_loss | 17 | 1 |

## R16 step-back retrieval (scripted chat, not a model)

| case | question | step-back question | fact_en rank, original only | fact_en rank, original + step-back | lost_constraints |
|---|---|---|---|---|---|
| success | What is the measurement schedule for the coastal station? | What does a tidal observatory do? | not in top 5 | 2 | - |
| losing (identifier) | What is the part number AB-1234 of the tide gauge? | What is a tide gauge? | - | - | ab-1234 |

Order searched for the identifier question: `What is the part number AB-1234 of the tide gauge?` then `What is a tide gauge?`; trace status `applied`. The step-back query lost the identifier and is still searched; the original is always searched first.

## R17 HyDE (scripted chat, not a model)

Question: `What is the measurement schedule for the coastal station?`

| case | include_question | passage | status | fact_en rank | top 5 | passage text in result JSON | every hit text is a corpus chunk |
|---|---|---|---|---|---|---|---|
| R02 same question | - | - | - | not in top 5 | dist06, dist07, dist03, dist04, dist02 | - | - |
| success, defaults | yes | `The tidal observatory at Port Sallow records sea levels every few minutes.` | applied | 1 | fact_en, dist06, dist07, dist03, dist04 | no | yes |
| success, passage only | no | `The tidal observatory at Port Sallow records sea levels every few minutes.` | applied | 1 | fact_en, dist07, dist03, fact_ar, dist06 | no | yes |
| losing, hallucinated passage, defaults | yes | `A bakery on Mill Lane bakes rye bread before sunrise.` | applied | not in top 5 | dist00, dist06, dist07, dist03, dist04 | no | yes |
| losing, hallucinated passage, passage only | no | `A bakery on Mill Lane bakes rye bread before sunrise.` | applied | not in top 5 | dist00, dist03, dist11, dist04, dist02 | no | yes |

Adaptation: the passage is embedded with input type `document` and the question with `query`; the paper embeds only the generated text. The hypothetical passage is never evidence: hits are corpus chunks only.

