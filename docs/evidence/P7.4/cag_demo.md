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
One local run, not an evaluation; numbers vary between runs; no quality or speed claim.

- Command: `PYTHONIOENCODING=utf-8 RAG_LIVE_OLLAMA=1 uv run pytest tests/live/test_prefix_cag_live.py tests/live/test_response_cache_live.py -m live -v -s -p no:cacheprovider`
- Date: 2026-11-23. Ollama 0.40.17 (`ollama --version`); models `gemma4:e4b` (chat) and `bge-m3` (embeddings), already pulled, no pull, no paid call.
- Result: 4 passed in 45.85s (whole command, both files). This replaces a first run in which the stream-mode R49 calls repeated the complete-mode prompts and were therefore not cold (found in review; the test now tags each run and mode).
- R49: Ollama reports the cached count (`prompt_eval_cached_count`), so `cache_metric=reported`. In both modes every cold call, the changed-prefix call and the original-prefix call after the eviction attempt read 0 tokens from cache. Warm `q1` read 472 of 477 prompt tokens from cache in both modes; warm `q2` and `q3` read 0. The other-prefix eviction call read 301 (complete) and 341 (stream) tokens, a shared leading part. Stream mode returned TTFT on every call. One run cannot explain why only the first warm question reused the prefix.
- R47: the first `cag` call (prime) cached 0 of 1368 tokens; the following `cag` calls cached 1367 (EN), 1334 (AR) and 1334 (out of corpus). Uncached and R02 k=3 calls cached 0. Answered EN and AR calls resolve their citations `exact` in all arms (asserted by the test; the printed lines show only markers); the out-of-corpus question abstains in all arms.
- Ollama's `prompt_eval_count` already includes cached tokens (warm EN: input 1368, cached 1367), so input + cached is an upper bound that double counts on warm calls. The truncation check runs on the first `cag` call and the uncached calls (cached 0 in this run).
- The snapshot's lab estimate (1080 tokens, 1350 with the 1.25 margin) is below the largest real prompt (1568, uncached AR). Still far below `num_ctx` 8192, but the margin would not catch a tight fit.

Printed output:

```text
tests/live/test_prefix_cag_live.py::test_r49_prefix_probe_live R49 mode=complete cache_metric=reported note=timings are a proxy; a lower warm prefill alone does not prove a cache hit
R49 complete arm=cold label=q1 ttft_ms=n/a total_ms=1297 prefill_ms=364 decode_ms=76 load_ms=10 input_tokens=498 cache_read_input_tokens=0 cache_read_reported=True output_tokens=7
R49 complete arm=cold label=q2 ttft_ms=n/a total_ms=751 prefill_ms=188 decode_ms=170 load_ms=16 input_tokens=496 cache_read_input_tokens=0 cache_read_reported=True output_tokens=13
R49 complete arm=cold label=q3 ttft_ms=n/a total_ms=870 prefill_ms=186 decode_ms=198 load_ms=6 input_tokens=493 cache_read_input_tokens=0 cache_read_reported=True output_tokens=16
R49 complete arm=warm label=prime ttft_ms=n/a total_ms=664 prefill_ms=186 decode_ms=71 load_ms=6 input_tokens=477 cache_read_input_tokens=0 cache_read_reported=True output_tokens=7
R49 complete arm=warm label=q1 ttft_ms=n/a total_ms=618 prefill_ms=69 decode_ms=70 load_ms=16 input_tokens=477 cache_read_input_tokens=472 cache_read_reported=True output_tokens=7
R49 complete arm=warm label=q2 ttft_ms=n/a total_ms=727 prefill_ms=184 decode_ms=169 load_ms=5 input_tokens=475 cache_read_input_tokens=0 cache_read_reported=True output_tokens=13
R49 complete arm=warm label=q3 ttft_ms=n/a total_ms=788 prefill_ms=199 decode_ms=165 load_ms=6 input_tokens=472 cache_read_input_tokens=0 cache_read_reported=True output_tokens=16
R49 complete arm=changed label=q1 ttft_ms=n/a total_ms=792 prefill_ms=183 decode_ms=71 load_ms=6 input_tokens=477 cache_read_input_tokens=0 cache_read_reported=True output_tokens=7
R49 complete arm=eviction_attempt label=other_prefix ttft_ms=n/a total_ms=897 prefill_ms=259 decode_ms=175 load_ms=5 input_tokens=854 cache_read_input_tokens=301 cache_read_reported=True output_tokens=16
R49 complete arm=eviction_attempt label=original_prefix ttft_ms=n/a total_ms=692 prefill_ms=188 decode_ms=69 load_ms=6 input_tokens=477 cache_read_input_tokens=0 cache_read_reported=True output_tokens=7
R49 complete summary arm=cold calls=3 ttft_median=None prefill_median=187.733 decode_median=169.936 cache_read_median=0.0
R49 complete summary arm=warm calls=4 ttft_median=None prefill_median=185.1825 decode_median=117.67500000000001 cache_read_median=0.0
R49 complete summary arm=changed calls=1 ttft_median=None prefill_median=182.638 decode_median=71.312 cache_read_median=0.0
R49 complete summary arm=eviction_attempt calls=2 ttft_median=None prefill_median=223.34500000000003 decode_median=121.94050000000001 cache_read_median=150.5
R49 mode=stream cache_metric=reported note=timings are a proxy; a lower warm prefill alone does not prove a cache hit
R49 stream arm=cold label=q1 ttft_ms=579 total_ms=657 prefill_ms=183 decode_ms=71 load_ms=6 input_tokens=498 cache_read_input_tokens=0 cache_read_reported=True output_tokens=7
R49 stream arm=cold label=q2 ttft_ms=573 total_ms=744 prefill_ms=185 decode_ms=164 load_ms=5 input_tokens=496 cache_read_input_tokens=0 cache_read_reported=True output_tokens=13
R49 stream arm=cold label=q3 ttft_ms=588 total_ms=792 prefill_ms=194 decode_ms=197 load_ms=16 input_tokens=493 cache_read_input_tokens=0 cache_read_reported=True output_tokens=16
R49 stream arm=warm label=prime ttft_ms=556 total_ms=634 prefill_ms=185 decode_ms=71 load_ms=6 input_tokens=477 cache_read_input_tokens=0 cache_read_reported=True output_tokens=7
R49 stream arm=warm label=q1 ttft_ms=431 total_ms=506 prefill_ms=67 decode_ms=69 load_ms=6 input_tokens=477 cache_read_input_tokens=472 cache_read_reported=True output_tokens=7
R49 stream arm=warm label=q2 ttft_ms=604 total_ms=786 prefill_ms=191 decode_ms=174 load_ms=6 input_tokens=475 cache_read_input_tokens=0 cache_read_reported=True output_tokens=13
R49 stream arm=warm label=q3 ttft_ms=544 total_ms=749 prefill_ms=189 decode_ms=198 load_ms=6 input_tokens=472 cache_read_input_tokens=0 cache_read_reported=True output_tokens=16
R49 stream arm=changed label=q1 ttft_ms=549 total_ms=628 prefill_ms=183 decode_ms=72 load_ms=8 input_tokens=477 cache_read_input_tokens=0 cache_read_reported=True output_tokens=7
R49 stream arm=eviction_attempt label=other_prefix ttft_ms=582 total_ms=768 prefill_ms=199 decode_ms=177 load_ms=7 input_tokens=853 cache_read_input_tokens=341 cache_read_reported=True output_tokens=16
R49 stream arm=eviction_attempt label=original_prefix ttft_ms=626 total_ms=706 prefill_ms=188 decode_ms=73 load_ms=6 input_tokens=477 cache_read_input_tokens=0 cache_read_reported=True output_tokens=7
R49 stream summary arm=cold calls=3 ttft_median=578.9779000042472 prefill_median=185.487 decode_median=164.497 cache_read_median=0.0
R49 stream summary arm=warm calls=4 ttft_median=549.8558999970555 prefill_median=187.072 decode_median=122.4445 cache_read_median=0.0
R49 stream summary arm=changed calls=1 ttft_median=549.1497999901185 prefill_median=182.692 decode_median=72.301 cache_read_median=0.0
R49 stream summary arm=eviction_attempt calls=2 ttft_median=604.2680500031565 prefill_median=193.78300000000002 decode_median=125.19399999999999 cache_read_median=170.5
PASSED
tests/live/test_prefix_cag_live.py::test_r47_cag_vs_uncached_vs_r02_live index chunks_embed_calls=1 num_ctx=8192 reserve=512
snapshot items=19 estimated_prompt_tokens=1080 margin=1.25
arm order: (1) cag (prime, EN, AR, OUT_OF_CORPUS) (2) full_context_uncached (3) R02 k=3
R47cmp arm=cag label=prime status=answered markers=['S1'] wall_ms=1089 prefill_ms=491 decode_ms=172 load_ms=6 input_tokens=1368 cache_read_input_tokens=0 cache_read_reported=True output_tokens=20
R47cmp arm=cag label=EN status=answered markers=['S1'] wall_ms=659 prefill_ms=37 decode_ms=173 load_ms=8 input_tokens=1368 cache_read_input_tokens=1367 cache_read_reported=True output_tokens=20
R47cmp arm=cag label=AR status=answered markers=['S2'] wall_ms=915 prefill_ms=119 decode_ms=269 load_ms=8 input_tokens=1366 cache_read_input_tokens=1334 cache_read_reported=True output_tokens=26
R47cmp arm=cag label=OUT_OF_CORPUS status=abstained markers=[] wall_ms=615 prefill_ms=109 decode_ms=72 load_ms=7 input_tokens=1363 cache_read_input_tokens=1334 cache_read_reported=True output_tokens=8
R47cmp arm=full_context_uncached label=EN status=answered markers=['S1'] wall_ms=1436 prefill_ms=565 decode_ms=339 load_ms=6 input_tokens=1568 cache_read_input_tokens=0 cache_read_reported=True output_tokens=37
R47cmp arm=full_context_uncached label=AR status=answered markers=['S2'] wall_ms=1494 prefill_ms=565 decode_ms=412 load_ms=7 input_tokens=1566 cache_read_input_tokens=0 cache_read_reported=True output_tokens=43
R47cmp arm=full_context_uncached label=OUT_OF_CORPUS status=abstained markers=[] wall_ms=1040 prefill_ms=487 decode_ms=77 load_ms=7 input_tokens=1443 cache_read_input_tokens=0 cache_read_reported=True output_tokens=8
R47cmp arm=R02_k3 label=EN status=answered markers=['S1'] wall_ms=1272 prefill_ms=172 decode_ms=338 load_ms=5 input_tokens=408 cache_read_input_tokens=0 cache_read_reported=True output_tokens=35
R47cmp arm=R02_k3 label=AR status=answered markers=['S1'] wall_ms=1344 prefill_ms=175 decode_ms=272 load_ms=6 input_tokens=425 cache_read_input_tokens=0 cache_read_reported=True output_tokens=26
R47cmp arm=R02_k3 label=OUT_OF_CORPUS status=abstained markers=[] wall_ms=1116 prefill_ms=173 decode_ms=71 load_ms=5 input_tokens=421 cache_read_input_tokens=0 cache_read_reported=True output_tokens=8
PASSED
```
</orchestrator>

