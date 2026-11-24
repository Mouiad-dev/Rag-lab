# R13: Question rewriting

## Intuition

A user question can be noisy or vague. R13 asks a chat model to rewrite it once for retrieval, then checks that the rewrite did not drop anything the question insisted on. If it did, the original question is searched instead.

## Mechanism

One metered chat call returns `{"query": "..."}`. The trace always keeps the exact original question. A deterministic check lists hard constraints in the question (numbers including Arabic-Indic digits, quoted phrases, negations) and compares them with the rewrite. On a loss the trace status is `constraint_loss` and the original question is the one searched. A malformed reply raises `invalid_model_output` unless `on_invalid="original"`.

## Config and defaults

`RewriteConfig`: `max_output_tokens=128`, `temperature=0.0`, `on_invalid="fail"`, `on_constraint_loss="original"` (the alternative `"fail"` raises). The prompt is versioned and its SHA-256 is stored in the trace. The user JSON is treated as untrusted data.

## Fidelity

Adaptation. There is no single reference paper for this step: it is one prompt plus a constraint guard that we added so a rewrite cannot silently change the question.

## Metering

One chat call per rewrite, counted in `trace.usage`; the retrieval that follows is counted separately.

```text
| technique | scenario | transform chat calls | transform embed calls | retrieval calls | retrieval embeds | transform input tokens | transform output tokens | cost USD |
|---|---|---|---|---|---|---|---|---|
| R13 | applied rewrite | 1 | 0 | 1 | 1 | 44 | 14 | 0 |
```

## Runnable demo

The offline demo `docs/techniques/lessons/demos/p54_query_transforms_demo.py` calls the real package code with a scripted chat (fixed JSON replies, not a model) and the test-only `LexicalTestEmbedder` (not a retriever quality signal). It writes [the evidence file](../../evidence/P5.4/query_transforms.md); every row below is quoted from it.

```bash
PYTHONIOENCODING=utf-8 uv run python docs/techniques/lessons/demos/p54_query_transforms_demo.py
```

## Observations (quoted exactly)

Success: the noisy question has the fact at rank 3, the rewrite has it at rank 1. Losing cases: a rewrite that drops `17` is caught and the original is searched; a rewrite that drops the place name is not caught.

```text
| case | status | lost_constraints | question kept in trace | retrieval queries | usage calls | fact_en rank, original question | fact_en rank, queries used |
|---|---|---|---|---|---|---|---|
| applied | applied | - | yes | `Port Sallow observatory records sea levels` | 1 | 3 | 1 |
| drops 17 (detected) | constraint_loss | 17 | yes | `Does the Port Sallow observatory record sea levels every 17 minutes?` | 1 | 1 | 1 |
| drops a place name (not detected) | applied | - | yes | `What does the tidal observatory record?` | 1 | 1 | 1 |
```

The detected case ranks the fact first either way because the original question is searched; only the status shows the loss. The undetected case also ranks the fact first on this fixture, so the demo shows that the loss goes unreported, not that it hurt retrieval.

## Live smoke

One live smoke run on 2026-11-09 (owner OK; local `gemma4:e4b` chat and `bge-m3` embeddings, cost 0; `RAG_LIVE_OLLAMA=1 uv run pytest tests/live/test_query_transforms_live.py -m live -v -s`) used one question: "How often does the Port Sallow observatory record sea levels?" It is one run on one question and is not an evaluation. R13 returned status `applied`, but the rewrite was identical to the question (76 input and 18 output tokens), and the English fact ranked first. The model changed nothing, so this run shows no benefit from rewriting.

## Limitations

The constraint check does not cover names, dates written as words, units or meaning. A rewrite can change intent without losing a number and still be marked `applied`.

## Status

Implemented; not validated or benchmarked. The demo uses a scripted chat, so it shows how the code handles given replies and says nothing about model quality.
