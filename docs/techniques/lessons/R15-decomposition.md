# R15: Question decomposition

## Intuition

A composite question ("how often X, and who does Y") is sometimes answered better by searching each part. R15 asks the model to split it.

## Mechanism

One metered chat call returns subquestions with ids `s1`, `s2`, ... and `depends_on` lists. A subquestion may depend only on earlier ones. The original question is searched first, then each subquestion in order, and results are merged by best rank. If the subquestions together lose a hard constraint of the original (number, quoted phrase, negation), R15 searches the original only and reports `constraint_loss`. Dependencies are recorded and validated; earlier answers are not fed into later queries.

## Config and defaults

`DecomposeConfig`: `max_subquestions=4` (2 to 6), `k=5`, `max_output_tokens=384`, `temperature=0.0`, `on_invalid="fail"`, `on_constraint_loss="original"`. `invalid_dependency` and `too_many_queries` raise even with `on_invalid="original"`; a non-list `depends_on` is `invalid_model_output`.

## Fidelity

Adaptation. It is a retrieval-time split with dependency validation only; it does not run multi-step reasoning or use intermediate answers.

## Metering

One chat call plus one retrieval per searched query (original plus subquestions).

```text
| technique | scenario | transform chat calls | transform embed calls | retrieval calls | retrieval embeds | transform input tokens | transform output tokens | cost USD |
|---|---|---|---|---|---|---|---|---|
| R15 | two subquestions | 1 | 0 | 3 | 3 | 54 | 17 | 0 |
```

## Runnable demo

The offline demo `docs/techniques/lessons/demos/p54_query_transforms_demo.py` calls the real package code with a scripted chat (fixed JSON replies, not a model) and the test-only `LexicalTestEmbedder` (not a retriever quality signal). It writes [the evidence file](../../evidence/P5.4/query_transforms.md); every row below is quoted from it.

```bash
PYTHONIOENCODING=utf-8 uv run python docs/techniques/lessons/demos/p54_query_transforms_demo.py
```

## Observations (quoted exactly)

Success (mechanism): a two-part question becomes two subquestions with a recorded dependency.

```text
| id | depends_on | text |
|---|---|---|
| s1 | - | How often does the Port Sallow observatory record sea levels? |
| s2 | s1 | Who repaints the wooden bridge every spring? |
```

On this fixture the plain question already finds both targets, so decomposition does not improve their ranks:

```text
| method | status | fact_en rank | dist03 rank (bridge) | top 5 |
|---|---|---|---|---|
| R02 original question | - | 2 | 1 | dist03, fact_en, dist07, dist06, dist05 |
| R15 original + subquestions | applied | 2 | 1 | dist03, fact_en, dist01, dist07, dist06 |
```

Losing cases: a dependency on a later id raises, in both modes with no retrieval call; subquestions without the number `17` fall back to the original question.

```text
| scripted reply | on_invalid | outcome | retrieval calls |
|---|---|---|---|
| s1 depends_on s2 (a later id) | fail | raised invalid_dependency | 0 |
| s1 depends_on s2 (a later id) | original | raised invalid_dependency | 0 |
| losing case: subquestions without 17 | status | lost_constraints | queries searched |
|---|---|---|---|
| default on_constraint_loss=original | constraint_loss | 17 | 1 |
```

## Live smoke

One live smoke run on 2026-11-09 (owner OK; local `gemma4:e4b` chat and `bge-m3` embeddings, cost 0; `RAG_LIVE_OLLAMA=1 uv run pytest tests/live/test_query_transforms_live.py -m live -v -s`) used one question: "How often does the Port Sallow observatory record sea levels?" It is one run on one question and is not an evaluation. R15 returned `invalid_model_output`: the model's JSON did not fit the subquestion schema, and the error was raised as designed.

## Limitations

Strict schema and dependency rules make a weak model fail often. Subquestions are searched independently; there is no answer-conditioned follow-up.

## Status

Implemented; not validated or benchmarked. The demo uses a scripted chat, so it shows how the code handles given replies and says nothing about model quality.
