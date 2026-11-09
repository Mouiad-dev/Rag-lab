<!-- generated 2026-11-10 by `uv run python docs/techniques/lessons/demos/p62_representations_demo.py`; offline; scripted chat and scripted embedder; no model; not a benchmark -->
# P6.2 R23 and R24 representations: demo output

Offline fixture: the first six `_fixture` paragraphs in an in-memory index (chunk names: fact_en, bakery, railway, chess, bridge, choir). The chat is a scripted responder that returns fixed JSON keyed by the passage; it is not a model. The embedder is a dict-backed ScriptedEmbedder with hand-written 6-dimensional vectors, so similarity is by construction. All generated text below is synthetic and scripted. Every table value is computed by the package code from those inputs. This is not a benchmark.

## R23 and R24 mechanics

### 1. R23 success: a generated question matches

Question: `How frequently are the tide readings taken?`. Right chunk: `fact_en`.

| retriever | order (chunk(rank)) | right chunk rank |
|---|---|---|
| R02 dense on original chunks (k=5) | chess(1), fact_en(2), bakery(3), railway(4), bridge(5) | 2 |
| R23 (k=3, includes original chunks) | fact_en(1), chess(2), bakery(3) | 1 |

| rank | chunk (original text) | matched by | match kind | synthetic | matched synthetic text |
|---|---|---|---|---|---|
| 1 | fact_en | question:0 | question | yes | `How often are sea levels recorded?` |
| 2 | chess | original:0 | original | no | - |
| 3 | bakery | original:0 | original | no | - |

### 2. R23 losing: a vague summary outranks the right chunk

Question: `Which place bakes bread before dawn?`. Right chunk: `bakery`.

| retriever | order (chunk(rank)) | right chunk rank |
|---|---|---|
| R02 dense on original chunks (k=5) | bakery(1), choir(2), fact_en(3), railway(4), chess(5) | 1 |
| R23 (k=3, includes original chunks) | choir(1), bakery(2), fact_en(3) | 2 |

| rank | chunk (original text) | matched by | match kind | synthetic | matched synthetic text |
|---|---|---|---|---|---|
| 1 | choir | summary:0 | summary | yes | `Local activities and routines.` |
| 2 | bakery | summary:0 | summary | yes | `Bread baking routine.` |
| 3 | fact_en | original:0 | original | no | - |

### 3. R24 success: the prefix adds the missing document name

Question: `Harbour Almanac: how often are readings taken?`. Right chunk: `fact_en`.

| retriever | order (chunk(rank)) | right chunk rank |
|---|---|---|
| R02 dense on original chunks (k=5) | chess(1), fact_en(2), bakery(3), railway(4), bridge(5) | 2 |
| R24 (k=3, prefixed text only) | fact_en(1), chess(2), choir(3) | 1 |

| rank | chunk (original text) | matched by | match kind | synthetic | matched synthetic text |
|---|---|---|---|---|---|
| 1 | fact_en | context_prefix:0 | context_prefix | yes | `From the Harbour Almanac, section on tide gauges.` |
| 2 | chess | context_prefix:0 | context_prefix | yes | `From the Mill Lane Handbook, section on clubs.` |
| 3 | choir | context_prefix:0 | context_prefix | yes | `From the chess club newsletter, meeting times.` |

### 4. R24 losing: a wrong prefix pulls a distractor up

Question: `When does the chess club meet?`. Right chunk: `chess`.

| retriever | order (chunk(rank)) | right chunk rank |
|---|---|---|
| R02 dense on original chunks (k=5) | fact_en(1), choir(2), chess(3), bridge(4), bakery(5) | 3 |
| R24 (k=3, prefixed text only) | choir(1), chess(2), bridge(3) | 2 |

| rank | chunk (original text) | matched by | match kind | synthetic | matched synthetic text |
|---|---|---|---|---|---|
| 1 | choir | context_prefix:0 | context_prefix | yes | `From the chess club newsletter, meeting times.` |
| 2 | chess | context_prefix:0 | context_prefix | yes | `From the Mill Lane Handbook, section on clubs.` |
| 3 | bridge | context_prefix:0 | context_prefix | yes | `From the Mill Lane Handbook, section on bridge upkeep.` |

The losing rows are not bugs: the code ranks by the vectors it is given. In the R23 losing row the scripted summary of `choir` is deliberately vague; in the R24 losing row the scripted prefix of `choir` names the wrong topic. A real generator can produce such text, and R23/R24 cannot detect it.

## 5. Synthetic-text check

For every retrieval above: are all hit texts original chunks, and does any generated string appear inside any `hit.text`?

| retrieval | hits | hits equal to an original chunk | generated strings found inside hit texts |
|---|---|---|---|
| R23 success | 3 | 3 | 0 |
| R23 losing | 3 | 3 | 0 |
| R24 success | 3 | 3 | 0 |
| R24 losing | 3 | 3 | 0 |

Generated strings checked: 18 (6 summaries, 6 questions, 6 prefixes). Matches that came from generated text are marked `synthetic: yes` in the tables; hits are always the stored original chunk with its source span.

## 6. Usage (scripted chat and embedder, fake usage, cost 0)

| step | build chat calls | build embed calls | query embed calls | cost USD |
|---|---|---|---|---|
| original chunk embeddings (index build, one call per chunk) | 0 | 6 | 0 | 0 |
| R23 build (summary + question per chunk) | 6 | 1 | 0 | 0 |
| R24 build (one prefix per chunk) | 6 | 1 | 0 | 0 |
| R23 success query (R02 baseline + technique) | 0 | 0 | 2 | 0 |
| R23 losing query (R02 baseline + technique) | 0 | 0 | 2 | 0 |
| R24 success query (R02 baseline + technique) | 0 | 0 | 2 | 0 |
| R24 losing query (R02 baseline + technique) | 0 | 0 | 2 | 0 |
| total | 12 | 8 | 8 | 0 |

Cross-check against the doubles: the scripted chat received 12 requests; the scripted embedder received 16 calls, of which 6 embedded original chunks. Builder reports: R23 6 chat usage rows and 0 rejected, R24 6 chat usage rows and 0 rejected. Each build embeds all its rows in one batched call.

## Live run (orchestrator)

Filled by the orchestrator after the P6.2-05 live test (`tests/live/test_representation_builders_live.py`, local Ollama, cost 0). Not run by this demo.

| item | value |
|---|---|
| date | <orchestrator> |
| command | <orchestrator> |
| chat model | <orchestrator> |
| embedding model | <orchestrator> |
| result | <orchestrator> |
