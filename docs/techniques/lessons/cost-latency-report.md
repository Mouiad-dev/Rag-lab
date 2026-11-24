# Cost and latency report

This is not a catalogue technique. It is a per-stage report in `packages/evaluation` (`cost_latency.py`, `timing.py`).

## Columns

`stage | calls | failed | input_tokens | output_tokens | cost_usd | estimated_token_calls | cold_ms | warm_median_ms | total_median_ms | total_max_ms | ttft_median_ms | ttft`

- `calls` and `failed` come from the timings when timings exist. Failed calls are kept out of the medians.
- The `failed` column shows `-` when there are no timings (the offline demo) and a count when timings exist (the live run).
- `estimated_token_calls` counts calls whose token numbers were estimated, not reported by the provider. It shows `-` when unknown.

## TTFT

Time to first token exists only for streamed calls, measured to the first non-empty chunk. `complete` calls show `not_applicable`. `AnswerService` and retrieval stages also show `not_applicable`, because they have no streaming path today. Other states: `no_tokens` (a stream gave no token) and `not_measured` (nothing was timed).

## Cold and warm

Cold is the first call per stage and model in this process; warm is every later call. Limits: no answer or embedding cache exists, so cold versus warm reflects only process, connection and model-load effects. Whether Ollama already had the model loaded is not controlled.

## Usage by stage

Tokens and cost are kept per stage (judge, rewriting, reranking and so on are not merged into the answer). Each row shows where its token numbers came from (provider-reported or estimated). Cost is `unknown` when a price is not known; it is never silently shown as zero for a paid model. Local Ollama is cost 0.

## Live numbers

Quoted from the live section of `docs/evidence/P6.5/judge_and_cost_demo.md` (one run, n=6 items per mode, not an evaluation; local Ollama `gemma4:e4b`, cost 0):

| stage | calls | failed | input_tokens | output_tokens | cost_usd | estimated_token_calls | cold_ms | warm_median_ms | total_median_ms | total_max_ms | ttft_median_ms | ttft |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| judge_complete | 6 | 0 | 1446 | 217 | 0 | 0 | 15311.652 | 789.916 | 791.258 | 15311.652 | - | not_applicable |
| judge_stream | 6 | 0 | 1446 | 217 | 0 | 0 | 752.227 | 782.293 | 782.068 | 837.968 | 417.880 | measured |

The complete stage ran first and its cold call was the slow one; whether the model was already loaded was not controlled, so do not read 15311.652 ms as a mode difference. The offline demo has no timings, so its latency is `not_measured`.

## Latency is not reproducible

Latency is wall-clock time on this machine for this run. It is not reproducible and is kept out of `results_json`. There is no p95 because n is too small. Answer TTFT needs a streaming answer path, which does not exist yet.
