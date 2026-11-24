# P6.4 R28 conversational rewrite: demo output

Command: `uv run python docs/techniques/lessons/demos/p64_conversation_demo.py`. Date: 2026-11-12. Offline; scripted chat; no model; shows plumbing and guards, not resolution quality.

## 1. EN follow-up (applied)

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

The assistant turn is not sent to the rewriter by default, so the fact "17" from the assistant answer is not in the prompt. The standalone question is marked synthetic; the original question is not.

## 2. AR follow-up (applied)

| field | value |
|---|---|
| history given | user: كم مخطوطة تحفظ مكتبة الواحة؟ |
| history used (user turns) | 1 of 1 given |
| assistant turns excluded | 0 |
| follow-up question | وأين تحفظها؟ |
| scripted reply (stands in for a model) | أين تحفظ مكتبة الواحة المخطوطات؟ |
| status | applied |
| reason | - |
| introduced_constraints | - |
| lost_constraints | - |
| generated (role, synthetic) | original synthetic=False, rewrite synthetic=True |
| retrieval query | أين تحفظ مكتبة الواحة المخطوطات؟ |
| chat calls | 1 |

## 3. Mixed language with Arabic-Indic digits (applied)

| field | value |
|---|---|
| history given | user: Tell me about the reading room schedule. |
| history used (user turns) | 1 of 1 given |
| assistant turns excluded | 0 |
| follow-up question | Is مكتبة الواحة open at ٠٨:٣٠? |
| scripted reply (stands in for a model) | Is the Oasis Library open at 08:30? |
| status | applied |
| reason | - |
| introduced_constraints | - |
| lost_constraints | - |
| generated (role, synthetic) | original synthetic=False, rewrite synthetic=True |
| retrieval query | Is the Oasis Library open at 08:30? |
| chat calls | 1 |

The Arabic-Indic digits are normalised before the constraint check, so `08:30` in the reply counts as kept.

## 4. Losing case (a): `unsupported_fact`

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

The reply adds "17", which only the assistant turn contains and the user turns do not support. The guard blocks it, so retrieval falls back to the raw follow-up "How often does it record them?", a weak query with no topic. This is the cost of the guard: it prevents an unsupported fact from steering retrieval, and it also loses the resolution of "it" and "them". With `include_assistant=True` the same reply is accepted.

## 5. Losing case (b): `constraint_loss`

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

The reply drops "2025" and "not", so the meaning changed. The code rejects the rewrite and retrieves with the original question.

## 6. History window

| field | value |
|---|---|
| turns given | 8 |
| max_turns | 3 |
| turns used (turn_index) | 5, 6, 7 |
| history_dropped | 5 |
| status | applied |

The oldest turns are dropped first; the most recent `max_turns` user turns are kept.

## 7. Usage (scripted chat; no timing values)

| case | chat calls (trace) | requests the scripted chat received |
|---|---|---|
| EN follow-up | 1 | 1 |
| AR follow-up | 1 | 1 |
| Mixed AR/EN | 1 | 1 |
| Losing (a) unsupported_fact | 1 | 1 |
| Losing (b) constraint_loss | 1 | 1 |
| Window | 1 | 1 |
| no_history (empty history) | 0 | 0 |

A turn with no usable history makes no chat call (status `no_history`).

## Live run (orchestrator)

One run, not an evaluation. Local Ollama, cost 0. The rewriter excludes assistant turns by default, so in the EN case "them" stays unresolved (the sea-level fact is only in the assistant turn). The first call's latency includes model load. In the mixed case the model returned the question unchanged; it still counts as `applied` because no guard fired, so it is not a real rewrite.

- Command: `RAG_LIVE_OLLAMA=1 uv run pytest tests/live/test_conversation_live.py -m live -v -s`
- Date: 2026-11-12 (11:52 UTC), working tree on commit 9fc23d5 plus the uncommitted P6.4-04 files
- Model: `gemma4:e4b` (Ollama, local)
- Result: 1 passed in 48.75s; 3 chat calls; all three conversations `applied`
- Printed output:

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
