# R48 vs OpenSPG/KAG: fidelity note

R48 is an original adaptation of the KAG idea. No OpenSPG code, dependency or service is used. OpenSPG-side facts come from the OpenSPG/KAG README and the paper abstract, checked 2026-11-22; they were not checked against OpenSPG code. "Here" is this repo (see [R48-kag.md](lessons/R48-kag.md), [ADR-0020](../decisions/ADR-0020-raptor-tree-and-kag-adaptation.md)).

Sources (as in [SOURCE_LINK_AUDIT.md](SOURCE_LINK_AUDIT.md), row 26):
- Repository: `github.com/OpenSPG/KAG` (README names release 0.8.0, 2025-06-27).
- Paper: arXiv 2409.13731, "KAG: Boosting LLMs in Professional Domains via Knowledge Augmented Generation" (title confirmed 2026-11-22).

| OpenSPG/KAG feature | here | status |
|---|---|---|
| Pinned release | No dependency, so nothing is pinned; README names 0.8.0 | not implemented |
| SPG schema (types, properties, relations, concepts/taxonomy, events, rules) | Own schema: entity types, typed properties, relations with domain/range/cardinality. No taxonomy, events or rules. README states schema customization and entity/event types only | partial |
| Schema-constrained extraction | One call per chunk; every item needs a unique grounded quote; rejects by reason | implemented |
| Schema-free open extraction | Not built; every item must fit the schema (README: supported) | not implemented |
| Knowledge alignment/normalization | Exact normalized name merge only; no alias or concept alignment (README: supported) | not implemented |
| Mutual index knowledge <-> chunks | `facts_for_chunk`, `chunks_for_fact`, span rows in migration 0013 | implemented |
| Logical-form planning | Own JSON grammar `kag_lf/1`, model plans, code executes | partial |
| Retrieval/filter/sort/count operators | `retrieve`, `filter`, `sort`, `count`, `output`; filter/sort on `many` properties rejected (not comparable from the README: it does not name these operators) | implemented |
| LLM deduce/math operators | `deduce` is rejected as `unsupported_operator` (README mentions numerical calculation and semantic reasoning steps) | not implemented |
| Hybrid KG + chunk retrieval fallback | None by design: a failed plan is `rejected`, never chunk hits (README: supported) | not implemented |
| Iterative planning/reflection | At most one re-plan after a failed plan (README: static and iterative planning; reflection not stated) | partial |
| Conflict handling | Cardinality-one conflicts reported, never resolved (not comparable from the README: it does not state this) | partial |
| Incremental update | Full rebuild; cost not measured (see `CACHE_AND_KNOWLEDGE_LABS.md`, KAG section, line 33) | not implemented |
| Model fine-tuning | None (README: kag-model to be open-sourced later) | not implemented |
| Published performance claims | Abstract: relative F1 gains of 19.6% (2WikiMultiHopQA) and 33.5% (HotpotQA) over SOTA RAG; README claims better than SOTA without numbers. Untested hypotheses here | not implemented |

Measured here (offline scripted, one local live run; not a benchmark): see [kag_demo.md](../evidence/P7.3/kag_demo.md). On the live run the planner answered 1 of 5 questions.
