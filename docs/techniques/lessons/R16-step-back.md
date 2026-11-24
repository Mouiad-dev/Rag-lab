# R16: Step-back retrieval

## Intuition

A very specific question may miss passages that state the general principle. R16 also searches a more general version of the question.

## Mechanism

One metered chat call returns one step-back question. The original is always searched first, then the step-back question (skipped if it equals the original after normalisation). Results are merged by best rank. Constraints that the step-back question lost are recorded in `lost_constraints` but the step-back query is still searched, because the original is always searched too.

## Config and defaults

`StepBackConfig`: `k=5`, `max_output_tokens=128`, `temperature=0.0`, `on_invalid="fail"`.

## Fidelity

Adaptation of arXiv 2310.06117 (Zheng et al., 2023, "Take a Step Back: Evoking Reasoning via Abstraction in Large Language Models"). The paper prompts with few-shot examples and uses the abstraction to guide reasoning; our zero-shot, retrieval-only prompt adapts that few-shot method to a search step.

## Metering

One chat call plus two retrievals (original and step-back) and two query embeddings.

```text
| technique | scenario | transform chat calls | transform embed calls | retrieval calls | retrieval embeds | transform input tokens | transform output tokens | cost USD |
|---|---|---|---|---|---|---|---|---|
| R16 | step-back hit | 1 | 0 | 2 | 2 | 41 | 11 | 0 |
```

## Runnable demo

The offline demo `docs/techniques/lessons/demos/p54_query_transforms_demo.py` calls the real package code with a scripted chat (fixed JSON replies, not a model) and the test-only `LexicalTestEmbedder` (not a retriever quality signal). It writes [the evidence file](../../evidence/P5.4/query_transforms.md); every row below is quoted from it.

```bash
PYTHONIOENCODING=utf-8 uv run python docs/techniques/lessons/demos/p54_query_transforms_demo.py
```

## Observations (quoted exactly)

```text
| case | question | step-back question | fact_en rank, original only | fact_en rank, original + step-back | lost_constraints |
|---|---|---|---|---|---|
| success | What is the measurement schedule for the coastal station? | What does a tidal observatory do? | not in top 5 | 2 | - |
| losing (identifier) | What is the part number AB-1234 of the tide gauge? | What is a tide gauge? | - | - | ab-1234 |
```

Success: the original question does not find the fact in its top 5; adding the step-back question brings it to rank 2. Losing case: on an identifier question the step-back question lost `ab-1234`; because the original is searched first the identifier query is not replaced, but the extra search carries no identifier.

```text
Order searched for the identifier question: `What is the part number AB-1234 of the tide gauge?` then `What is a tide gauge?`; trace status `applied`. The step-back query lost the identifier and is still searched; the original is always searched first.
```

## Live smoke

One live smoke run on 2026-11-09 (owner OK; local `gemma4:e4b` chat and `bge-m3` embeddings, cost 0; `RAG_LIVE_OLLAMA=1 uv run pytest tests/live/test_query_transforms_live.py -m live -v -s`) used one question: "How often does the Port Sallow observatory record sea levels?" It is one run on one question and is not an evaluation. R16 returned `applied`; the step-back question "What are the general principles governing the frequency of environmental data collection at scientific observatories?" was searched after the original (78 input, 27 output tokens), and the fact was first.

## Limitations

A general question tends to pull in generic passages. The constraint check is the same heuristic as R13 and misses names and meaning.

## Status

Implemented; not validated or benchmarked. The demo uses a scripted chat, so it shows how the code handles given replies and says nothing about model quality.
