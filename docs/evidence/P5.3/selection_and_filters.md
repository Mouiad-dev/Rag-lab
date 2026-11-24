# P5.3 selection and filters: demo output

Command: `uv run python docs/techniques/lessons/demos/p53_selection_demo.py` | Date: 2026-11-09 | offline fixture; no model; not a benchmark

The embedder is the test-only LexicalTestEmbedder and the R12 chat is a scripted responder.

## R10 deduplication

### Default config (protect_numbers=True)

| hand list of 6 | input | kept | dropped | redundancy before | redundancy after |
|---|---|---|---|---|---|
| counts | 6 | 3 | 3 | 0.500 | 0.000 |

| dropped | rule | similarity | provenance (view, span) | merged into |
|---|---|---|---|---|
| c2 | exact_text | 1.000 | view-b [0:287] | c1 |
| c3 | span_overlap | 1.000 | view-a [10:50] | c1 |
| c4 | near_text | 0.979 | view-c [0:293] | c1 |

| kept | view | span | text |
|---|---|---|---|
| c1 | view-a | [0:287] | The annual service plan costs 10 dollars per customer and th |
| c5 | view-d | [0:287] | The annual service plan costs 12 dollars per customer and th |
| c6 | view-e | [0:40] | The office closes at five and stays shut at night |

### Number protection off (protect_numbers=False)

| hand list of 6 | input | kept | dropped | redundancy before | redundancy after |
|---|---|---|---|---|---|
| counts | 6 | 2 | 4 | 0.667 | 0.000 |

| dropped | rule | similarity | provenance (view, span) | merged into |
|---|---|---|---|---|
| c2 | exact_text | 1.000 | view-b [0:287] | c1 |
| c3 | span_overlap | 1.000 | view-a [10:50] | c1 |
| c4 | near_text | 0.979 | view-c [0:293] | c1 |
| c5 | near_text | 0.880 | view-d [0:287] | c1 |

| kept | view | span | text |
|---|---|---|---|
| c1 | view-a | [0:287] | The annual service plan costs 10 dollars per customer and th |
| c6 | view-e | [0:40] | The office closes at five and stays shut at night |

### Losing case: a conflict stated in words, not numbers

Two hits say approved and rejected; the default near_text rule sees no number token and merges them.

| approved vs rejected | input | kept | dropped | redundancy before | redundancy after |
|---|---|---|---|---|---|
| counts | 2 | 1 | 1 | 0.500 | 0.000 |

| dropped | rule | similarity | provenance (view, span) | merged into |
|---|---|---|---|---|
| c2 | near_text | 0.812 | view-b [0:203] | c1 |

| kept | view | span | text |
|---|---|---|---|
| c1 | view-a | [0:203] | The audit report for the northern warehouse was reviewed by  |

## R09 MMR (lexical similarity)

| k | lambda | scoring | order | distinct views | distinct texts | in top k at 1.0 but lost |
|---|---|---|---|---|---|---|
| 3 | 1.0 | mmr[R03]:lexical:lambda=1.00 | c1, c2, c3 | 2 | 2 | - |
| 3 | 0.5 | mmr[R03]:lexical:lambda=0.50 | c1, c3, c2 | 2 | 2 | - |
| 3 | 0.0 | mmr[R03]:lexical:lambda=0.00 | c1, c3, c6 | 2 | 3 | c2 |
| 2 | 1.0 | mmr[R03]:lexical:lambda=1.00 | c1, c2 | 2 | 1 | - |
| 2 | 0.5 | mmr[R03]:lexical:lambda=0.50 | c1, c3 | 1 | 2 | c2 |
| 2 | 0.0 | mmr[R03]:lexical:lambda=0.00 | c1, c3 | 1 | 2 | c2 |

| lambda 0.5 hit | relevance | max similarity | mmr score |
|---|---|---|---|
| c1 | 1.000 | 0.000 | 0.500 |
| c3 | 0.600 | 0.287 | 0.156 |
| c2 | 0.800 | 1.000 | -0.100 |

| tie rule check | input scores | output order |
|---|---|---|
| equal scores | 0.5, 0.5 | c1, c2 |

## R11 metadata filter (post-filter)

| filter | hits | inner retriever calls | embedder calls | scoring |
|---|---|---|---|---|
| no filter | 6 | 1 | 1 | filter[R02]:post:3/3views |
| document report-2025 | 2 | 1 | 1 | filter[R02]:post:1/3views |
| media type text/markdown | 2 | 1 | 1 | filter[R02]:post:1/3views |
| stranger-scope document id | 0 | 0 | 0 | - |

| underfill case (filter keeps one document) | inner k | hits returned |
|---|---|---|
| inner top 2 holds 0 hit(s) of that document | 2 | 0 |
| same filter, same query | 50 | 2 |

| capability | status |
|---|---|
| filter fields | created_from, created_to, document_ids, document_version_ids, media_types |
| language / section filter | not available |

## R12 self-query (scripted chat, not a model)

| offered ref | document |
|---|---|
| d1 | report-2024 |
| d2 | report-2025 |
| d3 | notes |

| scripted response | on_invalid | status | reason | filter / result | usage (fake in/out tokens) |
|---|---|---|---|---|---|
| document_refs [d1] | fail | applied | None | document_ids -> report-2024 | 101/37 |

| end to end | status | hits | views in hits |
|---|---|---|---|
| retrieve_with_self_query | applied | 2 | report-2024 |

| scripted response | on_invalid | status | reason | filter / result | usage (fake in/out tokens) |
|---|---|---|---|---|---|
| extra key language | no_filter | rejected | unknown_filter_field | filter=None | 96/32 |
| UUID instead of ref | no_filter | - | - | raised filter_scope_widening | - |
| UUID instead of ref | fail | - | - | raised filter_scope_widening | - |

