# R28: Conversational RAG

## Intuition

A follow-up such as "How often does it record them?" has no topic on its own. R28 rewrites it once into a standalone question using a short window of earlier turns, then checks the rewrite with two guards before retrieval uses it.

## Mechanism

The caller passes the history. The rewriter keeps a bounded window of the most recent turns, by default user turns only, and makes one metered chat call that returns `{"query": "..."}`. The original turn is always kept in the trace (`synthetic=False`); the rewrite is marked `synthetic=True`. With no usable history the status is `no_history` and no chat call is made.

## Config and defaults

`ConversationConfig`: `max_turns=6` (1 to 20), `max_history_chars=4000` (200 to 16000), `include_assistant=False`, `max_output_tokens=128` (32 to 512), `temperature=0.0`, `on_invalid="fail"`, `on_constraint_loss="original"`, `on_unsupported_fact="original"` (the alternative `"fail"` raises). Each turn text is at most 2000 characters. The oldest turns are dropped first, by count and then by total characters.

## Guards

- `constraint_loss`: hard constraints in the follow-up question (numbers including Arabic-Indic digits, quoted phrases, negations, as in R13) must all appear in the rewrite. If any is lost, the status is `constraint_loss` and the original question is searched.
- `unsupported_fact`: a hard constraint that appears in the rewrite but not in the question or in the history turns that were used is an introduced constraint. The status is `unsupported_fact` (reason `unsupported_memory_fact`) and the original question is searched. With the default of user turns only, a fact that exists only in an assistant turn counts as unsupported.
- If both guards fire, `constraint_loss` wins.

A malformed reply raises `invalid_model_output` unless `on_invalid="original"`.

## Scope

History comes from the caller. Every turn must carry the same `conversation_id` as the call, otherwise `history_scope_mismatch` is raised; `turn_index` must be strictly increasing, otherwise `history_order_invalid`. There is no conversation storage yet and no answer-path wiring; both are planned for P9.

## Fidelity

ADAPTATION. This is a query-rewrite form of conversational RAG, with no memory store and no answer-side history.

## Metering

One chat call per rewrite, counted in `trace.usage`; none when status is `no_history`. Retrieval is counted separately.

## Runnable demo

The offline demo `docs/techniques/lessons/demos/p64_conversation_demo.py` calls the real package code with a scripted chat (fixed replies, not a model). It writes [the evidence file](../../evidence/P6.4/conversation_demo.md); the rows below are quoted from it.

```bash
uv run python docs/techniques/lessons/demos/p64_conversation_demo.py
```

## Examples (quoted exactly)

Success, EN follow-up (the assistant turn is not sent, so "17" is not in the prompt):

```text
| field | value |
|---|---|
| history given | user: Tell me about the Port Sallow observatory. / assistant: It records sea levels every 17 minutes. |
| history used (user turns) | 1 of 2 given |
| assistant turns excluded | 1 |
| follow-up question | How often does it record them? |
| scripted reply (stands in for a model) | How often does the Port Sallow observatory record sea levels? |
| status | applied |
| reason | - |
| introduced_constraints | - |
| lost_constraints | - |
| generated (role, synthetic) | original synthetic=False, rewrite synthetic=True |
| retrieval query | How often does the Port Sallow observatory record sea levels? |
| chat calls | 1 |
```

Losing case (a), `unsupported_fact`: the guard blocks the added "17", and retrieval falls back to a weak query with no topic. This is the cost of the guard: "it" and "them" stay unresolved.

```text
| field | value |
|---|---|
| history given | user: Tell me about the Port Sallow observatory. / assistant: It records sea levels every 17 minutes. |
| history used (user turns) | 1 of 2 given |
| assistant turns excluded | 1 |
| follow-up question | How often does it record them? |
| scripted reply (stands in for a model) | How often does the Port Sallow observatory record sea levels every 17 minutes? |
| status | unsupported_fact |
| reason | unsupported_memory_fact |
| introduced_constraints | 17 |
| lost_constraints | - |
| generated (role, synthetic) | original synthetic=False, rewrite synthetic=True |
| retrieval query | How often does it record them? |
| chat calls | 1 |
```

Losing case (b), `constraint_loss`: the reply drops "2025" and "not", so the original question is searched.

```text
| field | value |
|---|---|
| history given | user: Tell me about the reading room schedule. |
| history used (user turns) | 1 of 1 given |
| assistant turns excluded | 0 |
| follow-up question | What about 2024, not 2025? |
| scripted reply (stands in for a model) | What about 2024? |
| status | constraint_loss |
| reason | constraint_loss |
| introduced_constraints | - |
| lost_constraints | 2025, not |
| generated (role, synthetic) | original synthetic=False, rewrite synthetic=True |
| retrieval query | What about 2024, not 2025? |
| chat calls | 1 |
```

## Live result

One run, not an evaluation (quoted from the live section of the evidence file). Local Ollama, cost 0. The rewriter excludes assistant turns by default, so in the EN case "them" stays unresolved (the sea-level fact is only in the assistant turn). The first call's latency includes model load. In the mixed case the model returned the question unchanged; it still counts as `applied` because no guard fired, so it is not a real rewrite.

- Command: `RAG_LIVE_OLLAMA=1 uv run pytest tests/live/test_conversation_live.py -m live -v -s`
- Date: 2026-11-12 (11:52 UTC), working tree on commit 9fc23d5 plus the uncommitted P6.4-04 files
- Model: `gemma4:e4b` (Ollama, local)
- Result: 1 passed in 48.75s; 3 chat calls; all three conversations `applied`

```text
EN status=applied
standalone=How often does the Port Sallow observatory record them?
introduced_constraints=[]
lost_constraints=[]
usage model=gemma4:e4b input_tokens=124 output_tokens=17 latency_ms=46913.29279999627
AR status=applied
standalone=أين تحفظ مخطوطات مكتبة الواحة؟
introduced_constraints=[]
lost_constraints=[]
usage model=gemma4:e4b input_tokens=124 output_tokens=17 latency_ms=616.4951999962796
mixed status=applied
standalone=Is مكتبة الواحة open at ٠٨:٣٠?
introduced_constraints=[]
lost_constraints=[]
usage model=gemma4:e4b input_tokens=130 output_tokens=20 latency_ms=560.2731999970274
```

## Limitations

Offline tests prove the guards and plumbing, not reference-resolution quality. The guards cover numbers, quoted phrases and negations only; a rewrite can attach the wrong entity and still be `applied`. The live run was one run of three conversations.

## Status

Implemented; not validated or benchmarked. The demo uses a scripted chat, so it shows how the code handles given replies and says nothing about model quality.
