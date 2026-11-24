# R39: Community/global GraphRAG

## Intuition

Corpus-level questions ("Which organisations work with X?") are not answered by a few top chunks. R39 groups entities into communities, summarizes each community, and answers by asking every summary and combining the results (map-reduce).

## Mechanism

1. Communities: label propagation over the R38 graph, fixed order, no randomness. Communities with fewer than 2 members are skipped.
2. Summaries: a model writes claim-structured summaries. Each claim lists the fact IDs (entities or relations) it rests on, and each fact resolves to original source spans. A claim that cites a fact outside its community is rejected (`untraced_claim`).
3. Map: one model call per community summary, producing helpfulness-scored points that cite claims of that summary only.
4. Reduce: one call combines the most helpful points. Every final sentence must resolve point to claim to source spans; otherwise it is dropped, and if nothing survives the search abstains.

Summaries are synthetic (generated) text and are not persisted in P7.2. "Traced" is not "entailed": tracing shows which source spans a claim points to, not that the spans support the wording. Citations returned to the user are the original source spans.

## Fidelity

**Adaptation** of Edge et al. 2024. Differences: single-level label propagation instead of hierarchical Leiden; claim-structured summaries instead of free community reports; map-reduce with one summary per map call, no report batching and no shuffling. An optional entity-name prefilter (`prefilter_top`) limits map calls by a cheap lexical match; it saves calls and can lose recall. Not paper-faithful and not benchmarked.

## Observations (offline demo, scripted chat; plumbing, not quality)

Source: [graphrag_demo.md](../../evidence/P7.2/graphrag_demo.md). Fixture: 3 communities, one skipped singleton, one invented claim rejected.

- Success: "Which organisations work with the Harbor Authority?" gave status `answered`, 2 map calls, 1 reduce call, 4 sentences, 0 dropped, every source quote equal to the view slice. The scripted reduce echoes each point, so this shows traceability, not wording quality.

## Losing cases

- `no_relevant_community`: "What is the capital of Atlantis?" abstained after 2 map calls, 0 reduce calls.
- `reduce_untraced`: the reduce cited a point that does not exist; the sentence was dropped and the search abstained.
- Prefilter miss: with `prefilter_top=1` and an Arabic question sharing no token with any summary, it abstained with 0 map calls; without the prefilter it answered with 2 map calls.
- Singleton skipped: an entity with no relation (Mira Yassin) forms a community of one and is not summarized, so facts about it are not reachable by R39.
- One local live smoke run (gemma4:e4b, 2026-11-22, one run, not an evaluation; [graphrag_live_run.md](../../evidence/P7.2/graphrag_live_run.md)): R39 answered with 1 traced sentence.

## Cost (offline scripted chat; token counts are the fake model's estimates)

- Build, summaries: 2 chat calls, 317 input and 129 output tokens (on top of R38 extraction: 17 calls).
- Map, per question: 2 calls, 308 input and 68 output tokens.
- Reduce, per question: 1 call, 128 input and 42 output tokens.
- Update cost = full rebuild, the same as build cost; not measured separately.

## Limitations

- One level only; no hierarchy, so very large graphs need the community cap and map cost grows with community count.
- Summaries are not persisted; each question after a restart needs a rebuild of summaries.
- Quality of claims depends on the model; tracing is not entailment.
- Nothing here is validated or benchmarked.

## Status

Implemented (`packages/indexing/src/rag_foundry_indexing/graph_global.py`); not validated or benchmarked.
