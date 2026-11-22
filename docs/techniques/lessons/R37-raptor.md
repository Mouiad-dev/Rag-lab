# R37: RAPTOR summary tree

## Intuition

Some questions are about a whole topic, and no single paragraph answers them. R37 groups similar paragraphs, writes a short summary of each group, groups the summaries again, and so on up to a root. At query time the summaries help find the right region of the corpus. The text that comes back is always an original paragraph, never a summary.

## Mechanism

1. Build: embed the leaf chunks, cluster them (cosine k-means, hard clusters, large clusters split by a size cap), ask a chat model for one summary per cluster, embed the summaries, and repeat on the new level until one root is left or `max_levels` is reached. Summary prompt: `R37_SUMMARY_SYSTEM_V1`.
2. Lineage rule: a summary node stores only the IDs of its children. Lineage to the original leaf spans is computed by walking down (`lineage`); each leaf quote equals the view-text slice at its offsets. Summary nodes are `synthetic=True`, leaves `synthetic=False`.
3. Storage: migration 0012 adds `summary_trees`, `tree_nodes` and `tree_edges`. Rows are scoped by tenant and project, tied to an index version, and immutable; a change means a new tree (see [ADR-0020](../../decisions/ADR-0020-raptor-tree-and-kag-adaptation.md)).
4. Retrieve, collapsed mode: score all nodes against the query, take the `top_nodes`, and expand each selected node to its best leaf. Traversal mode: start at the root and keep `beam` nodes per level down to the leaves.
5. Hits are original leaf chunks with source spans. Summaries route retrieval but are never answer context or citations (owner decision 2026-11-22).

## Fidelity

**Adaptation** of Sarthi et al. 2024 (RAPTOR, [arXiv 2401.18059](https://arxiv.org/abs/2401.18059)). Differences from the paper: no UMAP, no Gaussian mixture, no BIC model selection (cosine k-means with hard clusters and size-cap splitting, stdlib only); collapsed mode uses `top_nodes` plus leaf expansion instead of a token budget; summaries are never put into the answer context (the paper puts them in context). Not paper-faithful and not benchmarked.

## Observations (offline demo; scripted chat and test-only lexical embedder; plumbing, not quality)

Source: [raptor_demo.md](../../evidence/P7.3/raptor_demo.md). Fixture: 13 paragraphs in three topics. Budget: 3 hits. Q_BUDGET ("What does each budget pay for?") is scored as topic coverage (share of the 3 topics with a budget paragraph among the hits) and as span recall over its 4 gold spans (at most 0.75 at 3 hits).

> Fixture summaries were rewritten after a first run so the R37 advantage and the losing case appear; this illustrates how routing works and how it fails, not that R37 retrieves better than dense R02.

| arm | Q_BUDGET topic coverage | Q_BUDGET span recall |
|---|---|---|
| R37 collapsed | 1.00 (hits=3) | 0.75 |
| flat summary index (`max_levels=1`) | 1.00 (hits=3) | 0.75 |
| R37 traversal, beam 2 | 0.33 (hits=2) | 0.25 |
| R02 dense, k=3 | 0.67 (hits=3) | 0.50 |

- Traversal with beam 2 returns 2 hits, under the 3-hit budget, so it is not directly comparable.
- Q_OVERVIEW (13 gold spans): collapsed 0.23, flat 0.23, R02 0.23 (hits=3); traversal 0.15 (hits=2). The root summary does not raise recall at this budget.
- Q_DETAIL and Q_ARABIC: all arms 1.00.
- Lineage: every leaf quote under the root and the level-1 nodes equals the view slice.
- Live (one local run, `gemma4:e4b` + `bge-m3`, not an evaluation; live section of the same evidence file): collapsed, flat and R02 picked the same paragraphs on all 4 questions, so this run shows no R37 advantage and no R37 loss. Traversal returned 2 hits each time.

## Losing cases

- Traversal with beam 1 on Q_DETAIL ("At what temperature is the clinic vaccine fridge kept?"): recall 0.00 (hits=1, paragraph H3), while collapsed and R02 reach 1.00. Traversal never looks back, and the harbor summary was written to contain the question words, so the wrong branch is kept. That summary text was written to make the failure appear; it is not a realistic summary.
- Q_BUDGET with traversal beam 2 (0.25) is a second case where routing by summaries loses to flat and R02.
- A summary that omits a detail hides the paragraphs below it.

## Cost (offline, scripted chat; token counts are the fake models' estimates)

- Full tree: 4 chat calls (450 input, 234 output tokens), 3 embed calls. Flat tree: 3 chat calls (353 input, 191 output), 2 embed calls.
- Query: 0 chat calls and 1 embed call per question (query embed tokens are not measured).
- Live run (one run, local, cost 0): full tree 4 chat calls (816 input, 325 output tokens); flat tree 3 chat calls (561 input, 167 output).
- Update cost = full rebuild; not measured separately.

## Limitations

- Summary quality decides routing quality; the offline summaries are scripted.
- Hard clusters: a paragraph belongs to one cluster only.
- No incremental update: a changed corpus needs a new tree.
- Four hand-written questions on one fixture; nothing here is validated or benchmarked.

## Status

Implemented (`packages/indexing/src/rag_foundry_indexing/raptor_retrieve.py`); not validated or benchmarked.
