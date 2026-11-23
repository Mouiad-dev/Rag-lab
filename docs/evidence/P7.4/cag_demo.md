# P7.4 R47 CAG: demo output

Command: `uv run python docs/techniques/lessons/demos/p74_cag_demo.py`. Date: 2026-11-23. Offline; scripted answers; shows snapshot bounds, prefix stability and citations, not speed or answer quality; prefix reuse needs a real runtime (see live runs). R49 has no offline demo: a scripted chat cannot show prefix reuse (R49 evidence is `prefix_anthropic_live.md` from P7.4-03a and the live section of P7.4-11).

Scripted chat (fixed replies, not a model), test-only lexical embedder, in-memory store. Chunk, snapshot and run IDs, keys and the nonce are never printed. Token figures are the scripted model's estimates, not provider counts. Cold versus warm is shown only as calls avoided in the response-cache demo. None of this is a benchmark.

## 1. Snapshot

| item | value |
|---|---|
| chunks in the snapshot (S1..Sn, fixed order) | 19 |
| estimated prompt tokens (lab-regex-v1, without question) | 1022 |
| context window (tokens) | 200000 |
| estimate margin | 1.25 |
| reserved output tokens | 512 |
| reserved question tokens | 64 |
| max_evidence_chars (cap) | 100000 |
| evidence chars in use | 1730 |

| overflow case | refusal code | numbers (needed/limit) |
|---|---|---|
| window 300 tokens | cag_overflow | prompt_tokens=1870/300 |
| evidence cap 500 chars | cag_overflow | evidence_chars=1730/500 |

Nothing is selected or truncated: a corpus that does not fit is refused at build time, before any chat call. The refusal figure is the full prompt estimate including margin and reserve, so it is larger than the snapshot estimate above; the per-call input tokens in section 2 come from the scripted chat model, a different estimator.

## 2. Arms on three questions

Each arm runs as a block: all `cag`, then all `full_context_uncached`, then R02 with k=3. Calls come from the answer ledger (R02 includes one query-embedding call); input tokens are the scripted chat estimate. The out-of-corpus reply is the scripted `INSUFFICIENT_EVIDENCE`, so abstaining there is scripted. Markers are numbered within each prompt: `S1` in R02 is not `S1` in the CAG snapshot.

| arm | question | status | abstention reason | cited markers | ledger calls | input tokens (scripted) |
|---|---|---|---|---|---|---|
| cag | EN | answered | - | S1 | 1 | 543 |
| cag | AR | answered | - | S2 | 1 | 538 |
| cag | out-of-corpus | abstained | model_declined | - | 1 | 539 |
| full_context_uncached | EN | answered | - | S1 | 1 | 543 |
| full_context_uncached | AR | answered | - | S2 | 1 | 538 |
| full_context_uncached | out-of-corpus | abstained | model_declined | - | 1 | 539 |
| r02 | EN | answered | - | S1 | 2 | 211 |
| r02 | AR | answered | - | S1 | 2 | 204 |
| r02 | out-of-corpus | abstained | model_declined | - | 2 | 211 |

| arm | prefix identical across questions |
|---|---|
| cag | yes |
| full_context_uncached | no |
| r02 | no |

`cag` keeps one snapshot nonce, so everything before the question is byte-identical and a runtime that supports prefix caching could reuse it (not shown here). `full_context_uncached` draws a fresh nonce per call, so its prefix never repeats. R02 sends different evidence per question.

## 3. Losing cases

| case | R47 result | R02 k=3 result |
|---|---|---|
| corpus overflows a 300-token window | refused: cag_overflow | answered, cited S1 |
| out-of-corpus question | abstained | abstained |

Out-of-corpus questions abstain in all three arms (cited markers `-`); that comes from the scripted reply, not from the arms.

| after a corpus update (one paragraph added) | value |
|---|---|
| chunks before / after | 19 / 19 |
| evidence chars before / after | 1730 / 1786 |
| snapshot identity changed | yes |
| prefix identical to the old snapshot | no |

A corpus update means a new snapshot: rebuild plus a cold prefill of the whole prefix. Not measured offline. Snapshot identity includes chunk IDs, so a new index version changes it even before the text differs.

## Live run (orchestrator)

<orchestrator>
Live run result goes here (P7.4-11). Live output is not deterministic and is recorded separately.
</orchestrator>

