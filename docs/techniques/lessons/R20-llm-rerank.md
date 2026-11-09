# R20: LLM listwise reranking

## Intuition

Instead of a scoring model, ask a chat model to put a short list of passages in order of how well they answer the question. It is flexible and needs no extra model, but the answer is only as good (and as stable) as the chat model, and the order can depend on the order it was shown.

## Mechanism

`LlmListwiseReranker.rerank(question, result, chat=...)` labels the input hits `p1`, `p2`, ... (position in the input), cuts each passage to `max_passage_chars`, and makes one metered JSON chat call. The system prompt (`RERANK_SYSTEM_V1`, hash recorded in `trace.prompt_sha256`) says the question and passages are untrusted data and asks only for `{"ranking": ["p2", "p1", ...]}`. The reply is validated: it must be a non-empty list of known, non-repeated ids. Anything else is rejected with a reason (`invalid_model_output`, `invalid_ranking`, `empty_ranking`, `unknown_id`, `duplicate_id`) and the input is returned unchanged with `status="rejected"`. An applied ranking becomes the output order. Ids the model left out are appended in input order (or dropped, with `unranked="drop"`). Each hit gets `llm_rank_position` (and `llm_unranked_appended` when appended), and its score is `1 / position`, derived from the list position and not a model probability.

The no-add rule holds as in R19: the output contains only input chunks; ids come from the given labels, so passage text cannot introduce a chunk.

## Config and defaults

`LlmRerankConfig`: `k=5` (1 to 20), `max_candidates=10` (1 to 20), `max_passage_chars=600`, `max_output_tokens=256`, `temperature=0.0`, `unranked="append"`. The chat must be a configured chat port; there is no silent fallback.

## Fidelity

Adaptation of listwise LLM reranking as studied in Sun et al., "Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agents" (2023, arXiv 2304.09542). Differences: one window only (no sliding window, so at most `max_candidates` passages), JSON labels `p1..pn` instead of the paper's identifier format, a hard candidate cap, a strict reject-on-invalid reply, and a different prompt. The paper's distillation and benchmark results are not reproduced.

## Metering

One chat call per rerank, in `trace.usage` (a rejected reply is still metered). The order probe makes 2 calls. Retrieval calls, reranker-port calls (R19) and rerank chat calls are separate columns in the demo.

```text
| scenario | retrieval calls | reranker-port calls (R19) | rerank chat calls (R20) | input tokens | output tokens | cost USD |
|---|---|---|---|---|---|---|
| R20 applied | 0 | 0 | 1 | 123 | 87 | 0 |
| R20 losing | 0 | 0 | 1 | 123 | 87 | 0 |
| R20 rejected | 0 | 0 | 1 | 123 | 87 | 0 |
| R20 injection | 0 | 0 | 1 | 122 | 86 | 0 |
| R20 order probe, position-biased | 0 | 0 | 2 | 246 | 174 | 0 |
| R20 order probe, content-based | 0 | 0 | 2 | 246 | 174 | 0 |
```

The tokens are the fake provider's usage (whitespace word counts), not a model's.

## Runnable demo

The offline demo `docs/techniques/lessons/demos/p55_rerank_demo.py` uses a scripted chat (fixed JSON replies, not a model) and the test-only `LexicalTestEmbedder`. It writes [the evidence file](../../evidence/P5.5/reranking_and_inspection.md); every offline row below is quoted from it.

```bash
PYTHONIOENCODING=utf-8 uv run python docs/techniques/lessons/demos/p55_rerank_demo.py
```

## Observations (quoted exactly)

The scripted replies show how the code treats each kind of reply, not how a model behaves.

```text
| case | scripted ranking | status | reason | order after | fact_en rank before -> after | chat calls | rank-position score of top hit |
|---|---|---|---|---|---|---|---|
| applied: fact first, input has fact_en at rank 6 | p6 p1 p2 p3 p4 p5 | applied | - | fact_en, dist04, dist03, dist07, dist06 | 6 -> 1 | 1 | 1.000000 |
| losing: a distractor first, fact pushed down | p2 p3 p4 p5 p6 p1 | applied | - | dist01, dist06, dist07, dist03, dist04 | 1 -> not in top 5 | 1 | 1.000000 |
| rejected: unknown id `p99` | p1 p99 | rejected | unknown_id | fact_en, dist01, dist06, dist07, dist03, dist04 | 1 -> 1 | 1 | - |
| injection: a passage asks for `p99` | p99 | rejected | unknown_id | fact_en, dist01, dist06, dist07, dist03, dist04 | 1 -> 1 | 1 | - |
```

Losing case: a reply that puts a distractor first is applied exactly as scripted, and the fact falls out of the top 5. R20 validates the form of the reply, not its correctness. Injection: the passage text asked for an id that does not exist; the scripted reply obeys it, and the code rejects the unknown id and keeps the input order.

Order probe: the reranker runs on the input order and on the reversed order.

```text
| responder | forward status | reversed status | agreement_at_k (k=5) | same_order | chat calls |
|---|---|---|---|---|---|
| position-biased (keeps the order it is given) | applied | applied | 4 | False | 2 |
| content-based (word overlap, ties by text) | applied | applied | 5 | True | 2 |
```

A passing probe on one input does not show the order is stable in general.

## Live smoke (orchestrator, local model, not part of the offline demo)

One live run on 2026-11-09 with local Ollama `gemma4:e4b` (owner OK, cost 0): `RAG_LIVE_OLLAMA=1 uv run pytest tests/live/test_llm_rerank_live.py -m live -v -s` gave 1 passed in 16.81s. Quoted from the evidence file, live section:

```text
Command: `RAG_LIVE_OLLAMA=1 uv run pytest tests/live/test_llm_rerank_live.py -m live -v -s`. Result: 1 passed in 16.81s. Input had the fact at rank 5 (`live-4`). Output order: live-4, live-0, live-1, live-2, live-3 (the fact moved to rank 1). Order probe: forward applied, reversed applied. Three calls (the rerank and the 2 probe calls), each 201 input and 8 output tokens; latency 15203 ms (first call, includes model load), 479 ms, 514 ms.
```

One question, one run: a smoke test, not an evaluation. The first call includes model load time.

## Limitations

- A single window: at most `max_candidates` (default 10) passages, each cut to `max_passage_chars`.
- The order may depend on input order and wording; the probe measures this for one input only.
- Rank-position scores are derived, not probabilities, and are not comparable across queries.
- Passages are untrusted data, but a model may still follow text inside them; the code limits the damage to a rejected reply or a bad order, never an added chunk.
- No benchmark, no quality claim. Evaluation belongs to P6.1.

## Status

Implemented; not validated or benchmarked. The offline demo uses a scripted chat, so it shows how the code handles given replies and says nothing about model quality.
