# P7.2 R38 entity graph and R39 community summaries: demo output

Command: `uv run python docs/techniques/lessons/demos/p72_graphrag_demo.py`. Date: 2026-11-22. Offline; scripted chat; shows grounding, tracing and plumbing, not extraction or answer quality.

## 1. Fixture

17 paragraphs, one chunk each. Evidence for the 3-hop question is P0, P7 and P14; the others are filler, an Arabic sentence (P15) and a lone entity (P16). Offsets are Unicode code points in the view text.

| paragraph | span | role | text |
|---|---|---|---|
| 0 | [0, 39) | gold | Lina Haddad founded Cedar Labs in 2019. |
| 1 | [41, 115) | - | The weather station logged steady rainfall over the northern valley today. |
| 7 | [502, 559) | gold | Cedar Labs supplies tide sensors to the Harbor Authority. |
| 14 | [1045, 1084) | gold | Omar Nasser leads the Harbor Authority. |
| 15 | [1086, 1109) | - | سامي يعمل في ميناء صور. |
| 16 | [1111, 1144) | - | Mira Yassin tends the lighthouse. |

Paragraphs P1 to P6 and P8 to P13 are filler (17 paragraphs in total).

Question: How is Lina Haddad connected to Omar Nasser?

## 2. Graph (R38 builder, adaptation)

Entities and relations are kept only when their quote occurs exactly once in the chunk.

**Entities** (7)

| entity | type | span | quote |
|---|---|---|---|
| Lina Haddad | person | [0, 11) | Lina Haddad |
| Cedar Labs | organization | [20, 30) | Cedar Labs |
| Harbor Authority | organization | [542, 558) | Harbor Authority |
| Omar Nasser | person | [1045, 1056) | Omar Nasser |
| سامي | person | [1086, 1090) | سامي |
| ميناء صور | place | [1099, 1108) | ميناء صور |
| Mira Yassin | person | [1111, 1122) | Mira Yassin |

**Relations** (4)

| source | label | target | span | quote |
|---|---|---|---|---|
| Lina Haddad | founded | Cedar Labs | [0, 39) | Lina Haddad founded Cedar Labs in 2019. |
| Cedar Labs | supplies sensors to | Harbor Authority | [502, 559) | Cedar Labs supplies tide sensors to the Harbor Authority. |
| Omar Nasser | leads | Harbor Authority | [1045, 1084) | Omar Nasser leads the Harbor Authority. |
| سامي | works in | ميناء صور | [1086, 1108) | سامي يعمل في ميناء صور |

**Rejected items** (1)

| paragraph | kind | item index | reason |
|---|---|---|---|
| 16 | entity | 1 | quote_not_found |

The rejected entity "Ghost" cites the quote "Ghost walked the pier.", which is not in P16, so it never enters the graph.

Graph key: 64 hex characters (sha256 of the canonical generator and index version; the value differs per run because the index version ID is random, so it is not printed).

Store round trip (build, store, read again): graph equal = True.

Every kept span quote equals the view slice: True.

## 3. Relationship question: R38 path vs the R21 baseline

**R38 entity-graph path** (How is Lina Haddad connected to Omar Nasser?)

Status `ok`, mode `path`, matched entities: Lina Haddad, Omar Nasser.

- hop forward: Lina Haddad -[founded]-> Cedar Labs
- hop forward: Cedar Labs -[supplies sensors to]-> Harbor Authority
- hop reverse: Omar Nasser -[leads]-> Harbor Authority

- P0 [0, 39) score 1.00: Lina Haddad founded Cedar Labs in 2019.
- P7 [502, 559) score 1.00: Cedar Labs supplies tide sensors to the Harbor Authority.
- P14 [1045, 1084) score 1.00: Omar Nasser leads the Harbor Authority.

Evidence recall over the 3 gold spans: 1.0.

**R21 neighbor expansion** (adjacent sentences, 5-sentence window; not GraphRAG) from the same 2 seed chunks (chunks whose text names a question entity):

Evidence recall: 2/3 (0.67). P7 missing: True. The window around P0 and P14 stops at distance 5, and P7 is 7 paragraphs from P0, so only the relation path reaches it.

## 4. R38 losing cases

**Alias question** ("Dr. Haddad" is not an entity name). Exact name matching finds one entity only.

Status `ok`, mode `neighborhood`, matched entities: Omar Nasser, path length 0.

- P14 [1045, 1084) score 0.50: Omar Nasser leads the Harbor Authority.

Evidence recall: 0.33 (1/3). The alias is not resolved; the path through Cedar Labs is lost.

**Unknown entity** (Who repaired the lighthouse?)

Status `no_entity_match`, mode `none`, hits 0.

**Hop limit** (`max_hops=2`, the true path has 3 hops)

Status `no_path_within_hops`, mode `path`, matched entities 2, hits 0.

The R38 cited answer (relationship question through the answer service, exact citations) is covered by the P7.2-06 test `tests/answering/test_graph_answer.py`; the demo does not rebuild the ingestion stack.

## 5. Communities and summaries (R39)

Label propagation, fixed order, no randomness. Summaries are generated text and marked `synthetic`.

| community | members | relations | iterations | converged |
|---|---|---|---|---|
| com_1d98f091731a57a2 | Mira Yassin | 0 | 3 | True |
| com_548d542938506d0b | سامي, ميناء صور | 1 | 3 | True |
| com_5d1483beb0552a16 | Cedar Labs, Harbor Authority, Lina Haddad, Omar Nasser | 3 | 3 | True |

Skipped (fewer than 2 members): Mira Yassin.

**Cluster of 2 entities** (`com_548d542938506d0b`, synthetic=True)

- claim `com_548d542938506d0b:0`: ميناء صور (place)
  - fact IDs: `ent_37c345c2336579b4`
  - source [1099, 1108): "ميناء صور"
- claim `com_548d542938506d0b:1`: سامي works in ميناء صور
  - fact IDs: `rel_eba8f690a37ea845`
  - source [1086, 1108): "سامي يعمل في ميناء صور"

`trace_claims` resolves 2 source spans for this summary.

**Cluster of 4 entities** (`com_5d1483beb0552a16`, synthetic=True)

- claim `com_5d1483beb0552a16:0`: Cedar Labs supplies sensors to Harbor Authority
  - fact IDs: `rel_3fc6667454f847b9`
  - source [502, 559): "Cedar Labs supplies tide sensors to the Harbor Authority."
- claim `com_5d1483beb0552a16:1`: Lina Haddad founded Cedar Labs
  - fact IDs: `rel_50033cf0cb990e46`
  - source [0, 39): "Lina Haddad founded Cedar Labs in 2019."

`trace_claims` resolves 2 source spans for this summary.

**Rejected claims** (community, claim index, reason)

| community | claim index | reason |
|---|---|---|
| com_5d1483beb0552a16 | 2 | untraced_claim |

The invented claim cites a fact ID that is not in its community, so it is rejected before R39 search.

## 6. R39 map-reduce global search

Question: Which organisations work with the Harbor Authority?

Status `answered`, map calls 2, reduce calls 1.

| community | map status | helpfulness | points |
|---|---|---|---|
| com_548d542938506d0b | kept | 10 | 2 |
| com_5d1483beb0552a16 | kept | 90 | 2 |

**Answer sentences** (the scripted reduce echoes each point; the demo checks traceability, not wording)

1. Cedar Labs supplies sensors to Harbor Authority (points: `com_5d1483beb0552a16:p0`)
   - source [502, 559): "Cedar Labs supplies tide sensors to the Harbor Authority."
2. Lina Haddad founded Cedar Labs (points: `com_5d1483beb0552a16:p1`)
   - source [0, 39): "Lina Haddad founded Cedar Labs in 2019."
3. ميناء صور (place) (points: `com_548d542938506d0b:p0`)
   - source [1099, 1108): "ميناء صور"
4. سامي works in ميناء صور (points: `com_548d542938506d0b:p1`)
   - source [1086, 1108): "سامي يعمل في ميناء صور"

Dropped sentences: 0. Rejected points: 0. Every source quote equals the view slice: True.

**Losing case: nothing relevant** (What is the capital of Atlantis?)

Status `abstained`, reason `no_relevant_community`, map calls 2, reduce calls 0.

**Losing case: untraced reduce** (the reduce cites a point that does not exist)

Status `abstained`, reason `reduce_untraced`, dropped sentences 1, sentences kept 0, reduce calls 1.

**Losing case: prefilter miss** (`prefilter_top=1`, Arabic question with no token shared with any summary)

With the prefilter: status `abstained`, reason `no_relevant_community`, map calls 0. Without it: status `answered`, map calls 2. The cheap lexical prefilter saves map calls and can lose recall.

## 7. Cost (scripted chat; token counts are the fake model's estimates)

| stage | chat calls | input tokens | output tokens |
|---|---|---|---|
| R38 extraction (build) | 17 | 2030 | 279 |
| R39 community summaries (build) | 2 | 317 | 129 |
| R39 map (per question) | 2 | 308 | 68 |
| R39 reduce (per question) | 1 | 128 | 42 |

R38 retrieval makes no model call. Update cost = full rebuild (same as build); not measured separately.

## Live run

One local live smoke run (gemma4:e4b on Ollama, 2026-11-22, cost 0, not an evaluation) is recorded
separately in [graphrag_live_run.md](graphrag_live_run.md), because live output is not deterministic.
In that run the structural checks passed, but R38 found no path within the hop limit on the
model-built graph (recall 0.00), while R39 answered with one traced sentence.

