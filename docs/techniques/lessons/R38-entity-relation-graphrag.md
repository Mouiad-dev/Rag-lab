# R38: Entity-relation GraphRAG

## Intuition

Some questions ask how two things are connected, and the facts that connect them sit in chunks that are far apart. A text neighbor of the first fact does not reach the last one. R38 builds an entity graph from the corpus once, then follows relations from one entity to another and returns the source sentences on the path.

## Mechanism

1. Build: a model reads each chunk and proposes entities and relations. Each item must carry a quote that (a) occurs verbatim and exactly once in the chunk, (b) contains the names of the entities involved, and (c) has offsets equal to the slice of the extraction view text. Items that fail are rejected and reported (the demo shows `quote_not_found`); they never enter the graph.
2. Store: the graph is derived data. Migration 0011 adds `entity_graphs`, `graph_entities`, `graph_relations` and `graph_spans`. Every row carries tenant, project, index version and graph key; rows are immutable, and a change means a new graph (see [ADR-0019](../../decisions/ADR-0019-graphrag-storage-and-fidelity.md)).
3. Retrieve: entities named in the question are matched by exact normalized name. With two or more matches the retriever runs a breadth-first search for a path (mode `path`); with one match it returns the neighborhood (mode `neighborhood`). Hits are the source spans of the edges on the path. No model call is made at query time.
4. Answer: the hits go through `AnswerService` like any other retriever output, so citations are original source spans, not graph text.

## Fidelity

**Adaptation** of Microsoft GraphRAG local search. The extraction prompt is our own, one call per chunk. Not implemented: gleaning passes; entity description summarization; alias resolution (only an exact normalized name merge); text-unit embeddings and the local-context mix. Retrieval is a BFS path or neighborhood, not the original context builder. Not paper-faithful and not benchmarked.

## Observations (offline demo, scripted chat; plumbing, not quality)

Source: [graphrag_demo.md](../../evidence/P7.2/graphrag_demo.md). Question: How is Lina Haddad connected to Omar Nasser?

- Success: path of 3 edges (founded, supplies sensors to, leads in reverse), 3 hits P0, P7, P14, evidence recall 1.0. A cited answer through `AnswerService` is covered by `tests/answering/test_graph_answer.py` (3 passed, 1 answer call, 15 graph build calls, offline scripted chat).
- Comparison: R21 is adjacent-text expansion, not GraphRAG. From the same 2 seed chunks with fixed seeds and a 5-sentence window it reaches recall 2/3 (0.67); P7 is 7 paragraphs from P0 and is missed.

## Losing cases

- Alias: "Dr. Haddad" is not an entity name. Only Omar Nasser matches, mode `neighborhood`, recall 1/3 (0.33).
- Unknown entity ("Who repaired the lighthouse?"): status `no_entity_match`, no hits, the retriever abstains.
- Hop limit: with `max_hops=2` and a true path of 3 hops: `no_path_within_hops`, no hits.
- One local live smoke run (gemma4:e4b, 2026-11-22, one run, not an evaluation; [graphrag_live_run.md](../../evidence/P7.2/graphrag_live_run.md)): the structure checks passed, but R38 returned `no_path_within_hops` with recall 0.00 on the model-built graph (42 entities, 22 relations). A model-built graph can miss the path.

## Cost (offline scripted chat; token counts are the fake model's estimates)

- Build: 17 chat calls, 2030 input and 279 output tokens (one call per chunk).
- Query: no model call for retrieval. The answer call is counted by the answer service.
- Update: a changed corpus needs a new graph (full rebuild); not measured separately.
- Live run totals (all stages, one run): 44 chat calls, 12557 input and 5965 output tokens, cost 0 (local).

## Limitations

- No alias resolution; spelling variants and pronouns split or lose entities.
- Entity matching is a plain substring search on normalized text with no word boundaries (for example "Sam" can match inside "Samir").
- No incremental update; graph quality depends on the extraction model.
- Nothing here is validated or benchmarked.

## Status

Implemented (`packages/indexing/src/rag_foundry_indexing/graph_retrieve.py`); not validated or benchmarked.
