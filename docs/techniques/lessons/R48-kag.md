# R48: KAG-style schema-constrained retrieval

## Intuition

Some questions need a count, a filter or a hop across facts ("Who leads the organization that Cedar Labs supplies?"). A pile of similar chunks does not compute that. R48 builds a small typed knowledge graph from the corpus under a declared schema, lets a model write a plan in a restricted logical form, and runs the plan with plain code. The hits are the original chunks behind the facts used.

## Mechanism

1. Schema: entity types, typed properties (cardinality one or many) and relations with domain, range and cardinality, versioned and hashed (demo schema `demo` v1).
2. Grounded, schema-constrained extraction: one model call per chunk. Every entity and fact needs a quote that occurs exactly once in the chunk, contains the names involved, and matches the view offsets. Items with an unknown type, predicate or entity, a range violation or a bad literal are rejected with a reason and never enter the graph. Conflicts (two values for a cardinality-one property) are reported, never resolved.
3. Mutual index: facts to chunks and chunks to facts (`facts_for_chunk`, `chunks_for_fact`).
4. Storage: migration 0013 adds `knowledge_graphs`, `kg_entities`, `kg_facts` and `kg_spans`; rows are scoped and immutable (see [ADR-0020](../../decisions/ADR-0020-raptor-tree-and-kag-adaptation.md)).
5. Logical form `rag_foundry.kag_lf/1`: operators `retrieve`, `filter`, `count`, `sort`, `output`. Plan rejections: `invalid_logical_form`, `unsupported_operator`, `schema_violation` (including filter or sort on a `many` property). Step statuses: `ok`, `empty`, `unknown_entity`, `missing`, `conflict`; run statuses: `answered`, `conflict`, `incomplete`.
6. Planning: one plan call (the question is untrusted data), then at most one re-plan, only after a failed plan or an `unknown_entity` incompletion; never after a data problem. Each attempt records its own prompt SHA. There is no fallback to chunk retrieval.
7. Cited answer: the hits go through `AnswerService`; citations are original chunk spans (`tests/answering/test_kag_answer.py`, 4 passed).

## Fidelity

**Adaptation**, not OpenSPG. See [KAG_OPENSPG_FIDELITY.md](../KAG_OPENSPG_FIDELITY.md) for the feature-by-feature comparison. Not benchmarked.

## Observations (offline demo; scripted extraction and plans; plumbing, not quality)

Source: [kag_demo.md](../../evidence/P7.3/kag_demo.md). Fixture: 18 paragraphs; 9 entities and 9 facts kept, 11 items rejected, 1 conflict. Metric: span recall = gold spans inside a hit / gold spans; six hand-written cases.

- R48 ties R02 (k=3) on Q0, Q1, Q2 and Q3.
- Count with filter: R48 1.00 (hits=2) vs R38 0.50 and R02 0.50. R38 and R02 cannot count or filter, so their recall says nothing about the answer value.
- Q3 multi-hop: R48 1.00 vs R38 0.50 (R02 1.00).
- Re-plan success: a misspelled entity is fixed by the second plan (2 plan calls).
- Live (one local run, `gemma4:e4b` + `bge-m3`, not an evaluation): extraction kept 15 entities and 8 facts from 18 chunks (1 `quote_not_found`, 1 conflict). R48 answered 1 of 5 questions (Q2 count after a replan; recall 0.67, R02 0.67). Q0, Q3 and Q4 were `rejected` (`schema_violation`: relations used in the wrong direction) and Q1 `incomplete`; R02 reached 1.00 on those four. The live planner is the weak point; no failure became a silent fallback.

## Losing cases

- Extraction miss (Q4, "Who leads Delta Works?"): `incomplete`, 0 hits; R02 1.00. Abstaining is honest but loses the evidence.
- Partial graph (Q2 with the Gulf Marine fact missing): `answered` with 2 of 3 suppliers and nothing shows it is incomplete.
- Conflict (Blue Reef founded 2012 vs 2014): `conflict`, no answer, no hits.
- Rejections: `unsupported_operator` (`deduce`), `schema_violation` (`owns` not declared), invalid plan; the single re-plan did not help in these demo cases.

## Cost (offline, scripted chat; token counts are the fake model's estimates)

- Build: 18 chat calls (3697 input, 781 output tokens), one per chunk.
- Query: 1 plan call (about 230 input, 50 output tokens), 2 with a re-plan; the executor makes no model call.
- Live run (one run, local, cost 0): extraction 18 calls (6627 input, 1028 output); planning 9 calls (5 first plans + 4 replans; 5429 input, 670 output).
- Update cost = full rebuild; not measured separately.

## Limitations

- The derived R48 answer (for example a count) is not passed to the answer model; only the supporting chunks are.
- A partial answer is invisible: the graph cannot know a fact was rejected at extraction.
- At most one re-plan; no alias or entity alignment; no LLM operators or chunk fallback; no incremental update; no schema induction; not wired to the API or UI.
- Nothing here is validated or benchmarked.

## Status

Implemented (`packages/indexing/src/rag_foundry_indexing/kag_retrieve.py`); not validated or benchmarked.
