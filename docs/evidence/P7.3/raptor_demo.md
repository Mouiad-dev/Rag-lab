# P7.3 R37 summary tree: demo output

Command: `uv run python docs/techniques/lessons/demos/p73_raptor_demo.py`. Date: 2026-11-22. Offline; scripted summaries; shows lineage and plumbing, not summary or retrieval quality.

Scripted chat (fixed JSON replies, not a model), test-only lexical embedder (token hashing, not a model), in-memory store. Summaries are synthetic text and are never evidence; hits are always original leaf chunks. This is an adaptation of RAPTOR (cosine k-means, hard clusters), not the paper's UMAP and GMM. None of this is a benchmark.

## 1. Fixture paragraphs and gold spans

13 paragraphs, one chunk each, in three lexically disjoint topics. C5-ar is an Arabic clinic sentence that also carries the Latin word `clinic`; the lexical embedder shares no token across scripts, so that word, not meaning, groups it with the clinic topic. Offsets are Unicode code points in the view text.

| label | topic | span | text |
|---|---|---|---|
| H1 | harbor | [0, 89) | Harbor crane, harbor quay, harbor berth: the harbor crane lifts containers onto the quay. |
| H2 | harbor | [91, 162) | Harbor budget, harbor budget: dredging, harbor lights and quay repairs. |
| H3 | harbor | [164, 251) | Harbor boats, harbor herring, harbor nets: fishing boats leave the harbor quay at dawn. |
| H4 | harbor | [253, 342) | Harbor pier, harbor tugboats, harbor budget, harbor budget: pier repairs and quay lights. |
| O1 | orchard | [344, 426) | Orchard apples, orchard pruning, orchard rows: apples ripen on every orchard tree. |
| O2 | orchard | [428, 513) | Orchard ladders, orchard crates, orchard apples: the orchard budget pays for pruning. |
| O3 | orchard | [515, 604) | Orchard bees, orchard blossoms, orchard hives: bees pollinate the pear trees each spring. |
| O4 | orchard | [606, 693) | Orchard frost, orchard heaters, orchard plums: frost threatens the plum trees in April. |
| C1 | clinic | [695, 778) | Clinic nurse, clinic desk, clinic patients: the clinic opens at eight each morning. |
| C2 | clinic | [780, 864) | Clinic staff, clinic training, clinic budget: the clinic budget funds nurse courses. |
| C3 | clinic | [866, 959) | Clinic vaccine fridge, clinic nurse: the fridge stays at four degrees and the clinic logs it. |
| C4 | clinic | [961, 1052) | Clinic hall, clinic doctor, clinic patients: patients wait until the doctor calls a number. |
| C5-ar | clinic | [1054, 1111) | تفتح العيادة clinic أبوابها للمرضى كل صباح في الـ clinic. |

| question | text | gold labels | gold spans (denominator) |
|---|---|---|---|
| Q_BUDGET | What does each budget pay for? | H2, H4, O2, C2 | 4 |
| Q_OVERVIEW | Give a municipal overview of shared priorities and finances in the region. | all 13 paragraphs | 13 |
| Q_DETAIL | At what temperature is the clinic vaccine fridge kept? | C3 | 1 |
| Q_ARABIC | متى تفتح العيادة؟ | C5-ar | 1 |

Q_BUDGET is scored as topic coverage (share of the 3 topics with at least one budget paragraph among the hits) and as span recall over its 4 gold spans, which is at most 0.75 at a 3-hit budget.

## 2. Tree

Summary prompt `R37_SUMMARY_SYSTEM_V1`, sha256 `c1298df7d1fe5d9aef0b3e35537d768a03d0120eb149ce299e4639e10ffd4acf`. Nodes are listed as the set of paragraph labels below them; every summary is `synthetic=True`, every leaf `synthetic=False`.

Levels: 3. Nodes: 17. Leaves: 13.

**Level 1** (clustering re-run with the public `cluster_nodes`: iterations 2, converged True, partition equals the built tree: True)

| node | synthetic | summary text (synthetic) |
|---|---|---|
| L1{H1,H2,H3,H4} | True | Harbor topic: what each harbor budget does pay for; what is kept at the harbor, at what temperature. |
| L1{O1,O2,O3,O4} | True | Orchard topic: what each orchard budget does pay for. |
| L1{C1,C2,C3,C4,C5-ar} | True | Clinic topic: what each clinic budget does pay for. |

**Level 2** (clustering re-run with the public `cluster_nodes`: iterations 0, converged True, partition equals the built tree: True)

| node | synthetic | summary text (synthetic) |
|---|---|---|
| L2{H1,H2,H3,H4,O1,O2,O3,O4,C1,C2,C3,C4,C5-ar} | True | Municipal overview: shared priorities, finances, region. |

Flat variant (`max_levels=1`): 2 levels, 16 nodes (the leaves plus the level-1 summaries, no root).

**Second build with `on_invalid="skip"`** (the scripted chat returns an empty summary for the orchard cluster): 1 promoted cluster(s), 4 levels, 18 nodes.

| level | promoted cluster (leaves kept in the pool without a summary) |
|---|---|
| 1 | O1, O2, O3, O4 |

Nodes per level in that tree: [13, 2, 2, 1]. The promoted orchard leaves stay in the pool, so at level 2 they are clustered again together with the 2 level-1 summaries (giving 2 level-2 summaries), then a root is built at level 3; the full build reaches its root a level earlier.

Store round trip (build, store, read again): tree equal = True. JSON round trip (`to_json`, `from_json`): tree equal = True. Tree key and node IDs differ per run (random index version ID), so they are not printed.

## 3. Lineage

Lineage is structural: a summary stores child IDs only, and `lineage` walks down to the original leaf spans. Each quote below is compared with the slice of the view text at its offsets.

**root L2{H1,H2,H3,H4,O1,O2,O3,O4,C1,C2,C3,C4,C5-ar}** (synthetic=True)

| leaf | span | quote | equals view slice |
|---|---|---|---|
| H1 | [0, 89) | Harbor crane, harbor quay, harbor berth: the harbor crane lifts containers onto the quay. | True |
| H2 | [91, 162) | Harbor budget, harbor budget: dredging, harbor lights and quay repairs. | True |
| H3 | [164, 251) | Harbor boats, harbor herring, harbor nets: fishing boats leave the harbor quay at dawn. | True |
| H4 | [253, 342) | Harbor pier, harbor tugboats, harbor budget, harbor budget: pier repairs and quay lights. | True |
| O1 | [344, 426) | Orchard apples, orchard pruning, orchard rows: apples ripen on every orchard tree. | True |
| O2 | [428, 513) | Orchard ladders, orchard crates, orchard apples: the orchard budget pays for pruning. | True |
| O3 | [515, 604) | Orchard bees, orchard blossoms, orchard hives: bees pollinate the pear trees each spring. | True |
| O4 | [606, 693) | Orchard frost, orchard heaters, orchard plums: frost threatens the plum trees in April. | True |
| C1 | [695, 778) | Clinic nurse, clinic desk, clinic patients: the clinic opens at eight each morning. | True |
| C2 | [780, 864) | Clinic staff, clinic training, clinic budget: the clinic budget funds nurse courses. | True |
| C3 | [866, 959) | Clinic vaccine fridge, clinic nurse: the fridge stays at four degrees and the clinic logs it. | True |
| C4 | [961, 1052) | Clinic hall, clinic doctor, clinic patients: patients wait until the doctor calls a number. | True |
| C5-ar | [1054, 1111) | تفتح العيادة clinic أبوابها للمرضى كل صباح في الـ clinic. | True |

**level-1 node L1{H1,H2,H3,H4}** (synthetic=True)

| leaf | span | quote | equals view slice |
|---|---|---|---|
| H1 | [0, 89) | Harbor crane, harbor quay, harbor berth: the harbor crane lifts containers onto the quay. | True |
| H2 | [91, 162) | Harbor budget, harbor budget: dredging, harbor lights and quay repairs. | True |
| H3 | [164, 251) | Harbor boats, harbor herring, harbor nets: fishing boats leave the harbor quay at dawn. | True |
| H4 | [253, 342) | Harbor pier, harbor tugboats, harbor budget, harbor budget: pier repairs and quay lights. | True |

**level-1 node L1{O1,O2,O3,O4}** (synthetic=True)

| leaf | span | quote | equals view slice |
|---|---|---|---|
| O1 | [344, 426) | Orchard apples, orchard pruning, orchard rows: apples ripen on every orchard tree. | True |
| O2 | [428, 513) | Orchard ladders, orchard crates, orchard apples: the orchard budget pays for pruning. | True |
| O3 | [515, 604) | Orchard bees, orchard blossoms, orchard hives: bees pollinate the pear trees each spring. | True |
| O4 | [606, 693) | Orchard frost, orchard heaters, orchard plums: frost threatens the plum trees in April. | True |

**level-1 node L1{C1,C2,C3,C4,C5-ar}** (synthetic=True)

| leaf | span | quote | equals view slice |
|---|---|---|---|
| C1 | [695, 778) | Clinic nurse, clinic desk, clinic patients: the clinic opens at eight each morning. | True |
| C2 | [780, 864) | Clinic staff, clinic training, clinic budget: the clinic budget funds nurse courses. | True |
| C3 | [866, 959) | Clinic vaccine fridge, clinic nurse: the fridge stays at four degrees and the clinic logs it. | True |
| C4 | [961, 1052) | Clinic hall, clinic doctor, clinic patients: patients wait until the doctor calls a number. | True |
| C5-ar | [1054, 1111) | تفتح العيادة clinic أبوابها للمرضى كل صباح في الـ clinic. | True |

## 4. Comparison at a 3-hit budget

Metric: span recall = gold paragraph spans fully inside one hit / number of gold spans, printed as `recall (hits=n)`. Denominators: Q_BUDGET 4, Q_OVERVIEW 13, Q_DETAIL 1, Q_ARABIC 1. Q_BUDGET also shows topic coverage = topics (of 3) with a budget paragraph among the hits. Four hand-written questions on one 13-paragraph fixture, not a benchmark.

> Fixture summaries were rewritten after a first run so the R37 advantage and the losing case appear; this illustrates how routing works and how it fails, not that R37 retrieves better than dense R02.

Traversal (beam 2) returns 2 hits on every question below, under the 3-hit budget, so its recall is capped by that. Q_OVERVIEW has 13 gold spans, so 3 hits cap recall at 3/13 and 2 hits at 2/13.

| question | arm | hit labels | recall (hits=n) | selected nodes (level, members) |
|---|---|---|---|---|
| Q_BUDGET | R37 collapsed | O2, C2, H2 | 0.75 (hits=3); topic coverage 1.00 | L1{O1,O2,O3,O4}; L1{C1,C2,C3,C4,C5-ar}; L1{H1,H2,H3,H4}; L0{O2}; L0{H2} |
| Q_BUDGET | R37 traversal (beam 2) | O2, O3 | 0.25 (hits=2); topic coverage 0.33 | L2{H1,H2,H3,H4,O1,O2,O3,O4,C1,C2,C3,C4,C5-ar}; L1{O1,O2,O3,O4}; L1{C1,C2,C3,C4,C5-ar}; L0{O2}; L0{O3} |
| Q_BUDGET | flat summary index (R37 on `max_levels=1`) | O2, C2, H2 | 0.75 (hits=3); topic coverage 1.00 | L1{O1,O2,O3,O4}; L1{C1,C2,C3,C4,C5-ar}; L1{H1,H2,H3,H4}; L0{O2}; L0{H2} |
| Q_BUDGET | R02 dense (k=3) | O2, H2, O3 | 0.50 (hits=3); topic coverage 0.67 | - |
| Q_OVERVIEW | R37 collapsed | C3, H2, O4 | 0.23 (hits=3) | L2{H1,H2,H3,H4,O1,O2,O3,O4,C1,C2,C3,C4,C5-ar}; L0{C3}; L0{H2}; L0{O4}; L0{C4} |
| Q_OVERVIEW | R37 traversal (beam 2) | H2, O4 | 0.15 (hits=2) | L2{H1,H2,H3,H4,O1,O2,O3,O4,C1,C2,C3,C4,C5-ar}; L1{H1,H2,H3,H4}; L1{O1,O2,O3,O4}; L0{H2}; L0{O4} |
| Q_OVERVIEW | flat summary index (R37 on `max_levels=1`) | C3, H2, O4 | 0.23 (hits=3) | L0{C3}; L0{H2}; L0{O4}; L0{C4}; L0{C1} |
| Q_OVERVIEW | R02 dense (k=3) | C3, H2, O4 | 0.23 (hits=3) | - |
| Q_DETAIL | R37 collapsed | C3, H3, C1 | 1.00 (hits=3) | L0{C3}; L1{H1,H2,H3,H4}; L0{C1}; L0{C2}; L1{C1,C2,C3,C4,C5-ar} |
| Q_DETAIL | R37 traversal (beam 2) | C3, C1 | 1.00 (hits=2) | L2{H1,H2,H3,H4,O1,O2,O3,O4,C1,C2,C3,C4,C5-ar}; L1{H1,H2,H3,H4}; L1{C1,C2,C3,C4,C5-ar}; L0{C3}; L0{C1} |
| Q_DETAIL | flat summary index (R37 on `max_levels=1`) | C3, H3, C1 | 1.00 (hits=3) | L0{C3}; L1{H1,H2,H3,H4}; L0{C1}; L0{C2}; L1{C1,C2,C3,C4,C5-ar} |
| Q_DETAIL | R02 dense (k=3) | C3, C1, C2 | 1.00 (hits=3) | - |
| Q_ARABIC | R37 collapsed | C5-ar, C4, C3 | 1.00 (hits=3) | L0{C5-ar}; L0{C4}; L0{C3}; L2{H1,H2,H3,H4,O1,O2,O3,O4,C1,C2,C3,C4,C5-ar}; L1{O1,O2,O3,O4} |
| Q_ARABIC | R37 traversal (beam 2) | C5-ar, C4 | 1.00 (hits=2) | L2{H1,H2,H3,H4,O1,O2,O3,O4,C1,C2,C3,C4,C5-ar}; L1{O1,O2,O3,O4}; L1{C1,C2,C3,C4,C5-ar}; L0{C5-ar}; L0{C4} |
| Q_ARABIC | flat summary index (R37 on `max_levels=1`) | C5-ar, C4, C3 | 1.00 (hits=3) | L0{C5-ar}; L0{C4}; L0{C3}; L1{O1,O2,O3,O4}; L1{C1,C2,C3,C4,C5-ar} |
| Q_ARABIC | R02 dense (k=3) | C5-ar, C4, C3 | 1.00 (hits=3) | - |

On Q_BUDGET the arms read: R37 collapsed 0.75 (hits=3); topic coverage 1.00; R37 traversal (beam 2) 0.25 (hits=2); topic coverage 0.33; flat summary index (R37 on `max_levels=1`) 0.75 (hits=3); topic coverage 1.00; R02 dense (k=3) 0.50 (hits=3); topic coverage 0.67. Traversal with beam 2 is the weakest arm there; it is a second losing case for routing by summaries.

R02 dense returns raw chunks with no tree, so it has no selected nodes. The root summary does not raise recall on Q_OVERVIEW: collapsed mode takes only the best leaf under each selected node, so at 3 hits the root changes which paragraphs come back, not how many gold ones.

## 5. Losing case: traversal with beam 1 on Q_DETAIL

| arm | hit labels | recall (hits=n), 1 gold span (C3) | selected nodes |
|---|---|---|---|
| R37 traversal, beam 1 | H3 | 0.00 (hits=1) | L2{H1,H2,H3,H4,O1,O2,O3,O4,C1,C2,C3,C4,C5-ar}; L1{H1,H2,H3,H4}; L0{H3} |
| R37 collapsed | C3, H3, C1 | 1.00 (hits=3) | L0{C3}; L1{H1,H2,H3,H4}; L0{C1}; L0{C2}; L1{C1,C2,C3,C4,C5-ar} |
| R02 dense (k=3) | C3, C1, C2 | 1.00 (hits=3) | - |

Why: traversal keeps only `beam` nodes per level and never looks back. The clinic summary omits the vaccine fridge detail, and the harbor summary was written to contain the question's words (what, kept, at, temperature), so the single kept branch is the harbor one and the answer paragraph C3 is never reached. Dense R02 matches the paragraph directly. This is routing text written to make the failure appear, not a realistic summary.

## 6. Cost

Scripted chat and test embedder; token counts are the fake models' estimates, not provider figures.

| build | chat calls | chat input tokens | chat output tokens | embed calls | embed input tokens | embed output tokens |
|---|---|---|---|---|---|---|
| full tree | 4 | 450 | 234 | 3 | 0 | 0 |
| flat tree (`max_levels=1`) | 3 | 353 | 191 | 2 | 0 | 0 |
| full tree, skip build | 6 | 624 | 300 | 4 | 0 | 0 |

Per query, R37 and R02 make 0 chat calls and the embed calls below (counted on the embedder). The retrievers do not expose embed token usage, so tokens per query are not measured.

| question | arm | query embed calls |
|---|---|---|
| Q_BUDGET | R37 collapsed | 1 |
| Q_BUDGET | R37 traversal (beam 2) | 1 |
| Q_BUDGET | flat summary index (R37 on `max_levels=1`) | 1 |
| Q_BUDGET | R02 dense (k=3) | 1 |
| Q_OVERVIEW | R37 collapsed | 1 |
| Q_OVERVIEW | R37 traversal (beam 2) | 1 |
| Q_OVERVIEW | flat summary index (R37 on `max_levels=1`) | 1 |
| Q_OVERVIEW | R02 dense (k=3) | 1 |
| Q_DETAIL | R37 collapsed | 1 |
| Q_DETAIL | R37 traversal (beam 2) | 1 |
| Q_DETAIL | flat summary index (R37 on `max_levels=1`) | 1 |
| Q_DETAIL | R02 dense (k=3) | 1 |
| Q_ARABIC | R37 collapsed | 1 |
| Q_ARABIC | R37 traversal (beam 2) | 1 |
| Q_ARABIC | flat summary index (R37 on `max_levels=1`) | 1 |
| Q_ARABIC | R02 dense (k=3) | 1 |

Update cost = full rebuild; not measured separately.

## Live run (orchestrator)

One run, not an evaluation. Live model output is not bit-identical across runs.

- Command: `PYTHONIOENCODING=utf-8 RAG_LIVE_OLLAMA=1 uv run pytest tests/live/test_raptor_live.py tests/live/test_kag_live.py -m live -v -s -p no:cacheprovider`
- Date: 2026-11-22 (started 14:06 UTC). Local Ollama, cost 0, no model pull.
- Models: chat `gemma4:e4b`, embeddings `bge-m3` (`ollama list` showed both present before the run).
- Result: 2 passed in 85.01 s (structural assertions only: lineage, grounding, tracing, call counts).

What this run shows: with live summaries and live `bge-m3` embeddings, R37 collapsed, the flat summary index and R02 picked the same 3 paragraphs for every question, so this run shows no R37 advantage and no R37 loss; traversal (default beam) returned 2 hits each time. The beam-1 losing case from the offline demo was not run live. Recall here is gold spans inside a hit / gold spans, as in the offline table.

Printed output (test_raptor_live.py):

```text
test_raptor_live_smoke stage=index_embed calls=13 input_tokens=347 output_tokens=0 latency_ms=10312
full_tree levels=3 nodes=17 summaries=4 leaves=13
full_tree promoted_clusters=0 sizes=[]
stage=full_tree_chat calls=4 input_tokens=816 output_tokens=325 latency_ms=19079
stage=full_tree_embed calls=3 input_tokens=767 output_tokens=0 latency_ms=3663
flat_tree levels=2 nodes=16 summaries=3 leaves=13
flat_tree promoted_clusters=0 sizes=[]
stage=flat_tree_chat calls=3 input_tokens=561 output_tokens=167 latency_ms=3110
stage=flat_tree_embed calls=2 input_tokens=557 output_tokens=0 latency_ms=860
question Q_BUDGET: What does each budget pay for?
Q_BUDGET collapsed hits=['C2', 'O2', 'H2'] recall=0.75 (hits=3)
Q_BUDGET traversal hits=['C2', 'O2'] recall=0.50 (hits=2)
Q_BUDGET flat summary index (R37 on max_levels=1) hits=['C2', 'O2', 'H2'] recall=0.75 (hits=3)
Q_BUDGET R02 dense k=3 hits=['C2', 'O2', 'H2'] recall=0.75 (hits=3)
question Q_OVERVIEW: Give a municipal overview of shared priorities and finances in the region.
Q_OVERVIEW collapsed hits=['H2', 'H4', 'O2'] recall=0.23 (hits=3)
Q_OVERVIEW traversal hits=['H2', 'H4'] recall=0.15 (hits=2)
Q_OVERVIEW flat summary index (R37 on max_levels=1) hits=['H2', 'H4', 'O2'] recall=0.23 (hits=3)
Q_OVERVIEW R02 dense k=3 hits=['H2', 'H4', 'O2'] recall=0.23 (hits=3)
question Q_DETAIL: At what temperature is the clinic vaccine fridge kept?
Q_DETAIL collapsed hits=['C3', 'C1', 'C5-ar'] recall=1.00 (hits=3)
Q_DETAIL traversal hits=['C3', 'C1'] recall=1.00 (hits=2)
Q_DETAIL flat summary index (R37 on max_levels=1) hits=['C3', 'C1', 'C5-ar'] recall=1.00 (hits=3)
Q_DETAIL R02 dense k=3 hits=['C3', 'C1', 'C5-ar'] recall=1.00 (hits=3)
question Q_ARABIC: متى تفتح العيادة؟
Q_ARABIC collapsed hits=['C1', 'C5-ar', 'C4'] recall=1.00 (hits=3)
Q_ARABIC traversal hits=['C1', 'C5-ar'] recall=1.00 (hits=2)
Q_ARABIC flat summary index (R37 on max_levels=1) hits=['C1', 'C5-ar', 'C4'] recall=1.00 (hits=3)
Q_ARABIC R02 dense k=3 hits=['C1', 'C5-ar', 'C4'] recall=1.00 (hits=3)
query_embed_calls_raptor=12 (3 arms x 4 questions)
chat ledger calls=7 input_tokens=1377 output_tokens=492
PASSED
```

