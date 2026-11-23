# R49: prefix KV caching probe

## What it is
Serving runtimes can reuse the prefill work of a shared prompt prefix. R49 is a `cache` plugin (`rag_foundry_cache.prefix`) that probes whether a provider does so, and reports what the provider says. It does not implement a cache.

## How it works here
Arms: `cold` (a salted prefix per call), `warm` (a prime call, then questions with a byte-identical prefix), `changed` (first character replaced, same length) and `eviction_attempt` (another prefix of the same length, then the original prefix again). Modes: `complete` (total, load, prefill, decode) and `stream` (adds TTFT at the first non-empty delta). Reports hold prefix hashes, never prompt or question text.

## cache_metric
`reported` only if the provider returned a cached-token count; otherwise `unavailable`; medians are `None`, never 0, when a value is missing.
- Ollama: `prompt_eval_cached_count` (observed in the live run below).
- Anthropic: `cache_read_input_tokens`; `cache_control` with a 5-minute TTL only; 512-token minimum for Haiku 5.5 (from the provider docs, checked 2026-11-22). The one paid run (`claude-haiku-5-5`, owner-run, hard cap USD 0.05) is still pending (the owner runs it in their own shell), so there are no Anthropic figures: `docs/evidence/P7.4/prefix_anthropic_live.md` does not exist, so there is no Anthropic cached count and no paid cost figure to quote.

## Fidelity: adaptation
A measurement probe over provider-reported counts and timings. No vLLM or other server metrics were measured (open item). Prefill is the uncached prompt only; decode is separate; TTFT includes load and prefill.

## What a hit or a win does not prove
Timings are a proxy, not a hit count: a lower warm prefill alone proves nothing. GPU KV caches are not tenant partitions; a cached count says nothing about isolation.

## Success example (live, one run)
Warm `q1` read 472 of 477 prompt tokens from cache in both modes (prefill 69 ms vs cold median 187.7 ms in complete mode and 185.5 ms in stream mode; stream TTFT 431 ms vs 573 to 588 ms cold).

## Losing examples (same run)
Warm `q2` and `q3` read 0 tokens, with the same prefix; one run cannot explain why only the first warm question reused it. The `changed` call and the original-prefix call after the eviction attempt read 0. The eviction attempt's other-prefix call read 301 (complete) and 341 (stream) tokens, a shared leading part. The `changed` arm replaces the first character, so it cannot show partial reuse.

## Cost per stage
Prefill, decode and load in ms per call (see the evidence file); tokens are provider counts; Ollama cost 0. Anthropic: not run.

## Cold vs warm
Complete mode medians: cold prefill 187.7 ms, warm 185.2 ms (the median hides the single reused call, because 3 of 4 warm calls read 0). n=3 cold and 4 warm calls per mode: not a benchmark.

## Open items
Anthropic run and its cost; why warm `q2` and `q3` read 0; vLLM metrics; a changed arm with a late-character change.
