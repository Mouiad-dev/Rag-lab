# P6.5 judge and cost: demo output

`uv run python docs/techniques/lessons/demos/p65_judge_cost_demo.py` | 2026-11-12 | offline; scripted chat; no model; not a benchmark

## Items

| item | question | answer | evidence marker |
|---|---|---|---|
| j-01 | How often does the Port Sallow observatory record sea levels? | The observatory records sea levels every 17 minutes [S1]. | S1 |
| j-02 | How often does the Port Sallow observatory record sea levels? | The observatory records sea levels every 15 minutes [S1]. | S1 |
| j-03 | كم مخطوطة تحفظ مكتبة الواحة؟ | تحفظ مكتبة الواحة ثلاثة آلاف ومئتي مخطوطة في قبوها الشمالي. [S1] | S1 |
| j-04 | كم مخطوطة تحفظ مكتبة الواحة؟ | لا تحفظ مكتبة الواحة أي مخطوطات [S1]. | S1 |
| j-05 | When does the Oasis Library open its reading room? | It opens at 08:30 and closes at 20:00 [S1]. | S1 |
| j-06 | Which club meets on Tuesday evenings? | The choir meets on Tuesday evenings [S1]. | S1 |

## Scripted judge

**Scripted replies, identical for every item; not a judgement; shows plumbing only.**

Every item gets `{"label":"supported","reason":"scripted"}`, except `j-05`, which gets the malformed reply `not json` and is recorded as `invalid_output`.

| item | status | label | prompt_sha256[:12] | rubric |
|---|---|---|---|---|
| j-01 | ok | supported | b6d6c59076e1 | answer-support v1 |
| j-02 | ok | supported | b6d6c59076e1 | answer-support v1 |
| j-03 | ok | supported | b6d6c59076e1 | answer-support v1 |
| j-04 | ok | supported | b6d6c59076e1 | answer-support v1 |
| j-05 | invalid_output | - | b6d6c59076e1 | answer-support v1 |
| j-06 | ok | supported | b6d6c59076e1 | answer-support v1 |

## Agreement

Judge agreement: not yet measured (0 human labels).

Judge labels are evaluator output, not ground truth.

## Cost and latency

# Judge cost per stage (scripted chat, fake usage)

Latency is wall-clock time on this machine for this run. It is not reproducible and is not part of results_json. Cold is the first call per stage and model in this process; warm is every later call. No answer or embedding cache exists, so cold versus warm reflects only process, connection and model-load effects. TTFT (time to first token) exists only for streamed calls; complete calls show not_applicable.

| stage | calls | failed | input_tokens | output_tokens | cost_usd | estimated_token_calls | cold_ms | warm_median_ms | total_median_ms | total_max_ms | ttft_median_ms | ttft |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| judge | 6 | - | 863 | 203 | 0 | 6 | - | - | - | - | - | not_measured |

## Notes

- Offline: no call was timed, so latency is not_measured. Token counts are fake usage.

## Live run (orchestrator)

One run, n=6, not an evaluation; agreement not measured (0 human labels). Judge labels are evaluator output, not ground truth. Local Ollama, cost 0. Cold is the first call per stage in this process; whether Ollama already had the model loaded was not controlled (the complete stage ran first and its cold call was the slow one). In this run the stream mode gave the same label as the complete mode for each of the 6 items.

- Command: `RAG_LIVE_OLLAMA=1 uv run pytest tests/live/test_judge_live.py -m live -v -s`
- Date: 2026-11-12 (13:17 UTC), working tree on commit cf19b33 plus the uncommitted P6.5-04 files
- Model: `gemma4:e4b` (Ollama, local), 12 calls (6 complete, 6 stream)
- Result: 1 passed in 24.41s
- Printed output (Markdown headings demoted two levels to fit this file; text otherwise as printed):

| item | mode | status | label | reason |
|---|---|---|---|---|
| j-01 | complete | ok | supported | 'The evidence explicitly states that the tidal observatory at Port Sallow records' |
| j-02 | complete | ok | unsupported | 'The evidence states the observatory records sea levels every 17 minutes, which c' |
| j-03 | complete | ok | supported | 'The evidence S1 explicitly states that the Oasis Library holds three thousand tw' |
| j-04 | complete | ok | unsupported | 'The answer claims the library holds no manuscripts, but the evidence states it h' |
| j-05 | complete | ok | partially_supported | 'The evidence states the reading room opens at 08:30, but it does not mention whe' |
| j-06 | complete | ok | unsupported | 'The answer states that the choir meets on Tuesday evenings, but the evidence sta' |
| j-01 | stream | ok | supported | 'The evidence explicitly states that the tidal observatory at Port Sallow records' |
| j-02 | stream | ok | unsupported | 'The evidence states the observatory records sea levels every 17 minutes, which c' |
| j-03 | stream | ok | supported | 'The evidence S1 explicitly states that the Oasis Library holds three thousand tw' |
| j-04 | stream | ok | unsupported | 'The answer claims the library holds no manuscripts, but the evidence states it h' |
| j-05 | stream | ok | partially_supported | 'The evidence states the reading room opens at 08:30, but it does not mention whe' |
| j-06 | stream | ok | unsupported | 'The answer states that the choir meets on Tuesday evenings, but the evidence sta' |

### Judge cost and latency (live, gemma4:e4b)

Latency is wall-clock time on this machine for this run. It is not reproducible and is not part of results_json. Cold is the first call per stage and model in this process; warm is every later call. No answer or embedding cache exists, so cold versus warm reflects only process, connection and model-load effects. TTFT (time to first token) exists only for streamed calls; complete calls show not_applicable.

| stage | calls | failed | input_tokens | output_tokens | cost_usd | estimated_token_calls | cold_ms | warm_median_ms | total_median_ms | total_max_ms | ttft_median_ms | ttft |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| judge_complete | 6 | 0 | 1446 | 217 | 0 | 0 | 15311.652 | 789.916 | 791.258 | 15311.652 | - | not_applicable |
| judge_stream | 6 | 0 | 1446 | 217 | 0 | 0 | 752.227 | 782.293 | 782.068 | 837.968 | 417.880 | measured |

#### Notes

- one run, n=6 items per mode; not an evaluation

Judge agreement: not yet measured (0 human labels).

Judge labels are evaluator output, not ground truth.
