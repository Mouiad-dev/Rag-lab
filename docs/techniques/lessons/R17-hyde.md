# R17: HyDE (hypothetical document embeddings)

## Intuition

A question and its answer passage often look different. HyDE asks a model to write a hypothetical answer passage, embeds that, and searches with the vector. The passage is only a search key.

## Mechanism

One metered chat call returns `n_passages` short passages. The passages are embedded as `document`, the question as `query`. The vectors are L2-normalised and averaged, and one dense index search runs with the mean. The returned hits are corpus chunks; hypothetical text is never evidence and never appears in the result (the trace marks it `synthetic`). Index readiness, embedding compatibility and scope are checked before the chat call.

## Config and defaults

`HydeConfig`: `n_passages=1`, `include_question=True` (the question vector is part of the mean), `k=5`, `max_output_tokens=256`, `temperature=0.7`, `max_passage_chars=1200`.

## Fidelity

Adaptation of arXiv 2212.10496 (Gao et al., 2022, "Precise Zero-Shot Dense Retrieval without Relevance Labels"). The paper embeds only the generated text with a contrastive encoder. Embedding the question as `query` and the passages as `document`, and including the question vector by default, is an adaptation made for asymmetric embedding models.

## Metering

One chat call plus two embedding calls (passage, question) and one index search, with no query embedding for retrieval. The embeds are in `trace.usage`.

```text
| technique | scenario | transform chat calls | transform embed calls | retrieval calls | retrieval embeds | transform input tokens | transform output tokens | cost USD |
|---|---|---|---|---|---|---|---|---|
| R17 | success, defaults | 1 | 2 | 1 | 0 | 46 | 13 | 0 |
```

## Runnable demo

The offline demo `docs/techniques/lessons/demos/p54_query_transforms_demo.py` calls the real package code with a scripted chat (fixed JSON replies, not a model) and the test-only `LexicalTestEmbedder` (not a retriever quality signal). It writes [the evidence file](../../evidence/P5.4/query_transforms.md); every row below is quoted from it.

```bash
PYTHONIOENCODING=utf-8 uv run python docs/techniques/lessons/demos/p54_query_transforms_demo.py
```

## Observations (quoted exactly)

Success: the passage that matches the fact finds it first, while the plain question misses it. Losing case: a hallucinated bakery passage ranks a distractor first and the fact falls out of the top 5. The last two columns show the passage text is not in the result and all hits are corpus chunks.

```text
| case | include_question | passage | status | fact_en rank | top 5 | passage text in result JSON | every hit text is a corpus chunk |
|---|---|---|---|---|---|---|---|
| R02 same question | - | - | - | not in top 5 | dist06, dist07, dist03, dist04, dist02 | - | - |
| success, defaults | yes | `The tidal observatory at Port Sallow records sea levels every few minutes.` | applied | 1 | fact_en, dist06, dist07, dist03, dist04 | no | yes |
| success, passage only | no | `The tidal observatory at Port Sallow records sea levels every few minutes.` | applied | 1 | fact_en, dist07, dist03, fact_ar, dist06 | no | yes |
| losing, hallucinated passage, defaults | yes | `A bakery on Mill Lane bakes rye bread before sunrise.` | applied | not in top 5 | dist00, dist06, dist07, dist03, dist04 | no | yes |
| losing, hallucinated passage, passage only | no | `A bakery on Mill Lane bakes rye bread before sunrise.` | applied | not in top 5 | dist00, dist03, dist11, dist04, dist02 | no | yes |
```

## Live smoke

One live smoke run on 2026-11-09 (owner OK; local `gemma4:e4b` chat and `bge-m3` embeddings, cost 0; `RAG_LIVE_OLLAMA=1 uv run pytest tests/live/test_query_transforms_live.py -m live -v -s`) used one question: "How often does the Port Sallow observatory record sea levels?" It is one run on one question and is not an evaluation. R17 returned `applied`; the hypothetical passage asserted "records sea levels on a daily basis", an invented detail and a known HyDE risk (81 input, 47 output chat tokens, two bge-m3 embed calls with 53 and 15 input tokens). Because hypothetical text is never evidence, the invented "daily" cannot be cited.

## Limitations

The passage quality sets the ceiling; a wrong passage can push the search away from the answer, as the losing rows show. Averaging with the question vector softens but does not remove this.

## Status

Implemented; not validated or benchmarked. The demo uses a scripted chat, so it shows how the code handles given replies and says nothing about model quality.
