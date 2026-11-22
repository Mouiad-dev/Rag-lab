# ADR-0021: CAG and three cache layers

Date: 2026-11-22
Status: Accepted (owner, 2026-11-22)

## Context
P7.4 adds R47 (CAG), R49 (prefix-cache probe), R50 (exact response cache) and R51 (semantic response cache). They are different mechanisms and stay separate. Response caching is not CAG.

## Decision
- Four IDs, never merged. R47 is a `workflow` plugin in `rag_foundry_answering.cag`. R49, R50 and R51 are `cache` plugins in `rag_foundry_cache` (new package `rag-foundry-cache`, no third-party dependency beyond pydantic).
- R47 snapshot: a bounded, versioned snapshot of all chunks of one ready index version, in fixed order, with explicit overflow and no silent selection.
- R47 nonce (owner approved 2026-11-22): a stable per-snapshot nonce keeps the prefix byte-stable. It is drawn with `secrets.token_hex(8)` and redrawn if it occurs in any chunk text. It is `repr=False` and never logged, dumped, persisted, traced or printed. Every question is checked before any model call. An answer echoing the bare nonce abstains. A rebuilt snapshot draws a new nonce. The `full_context_uncached` arm uses a per-run nonce.
- R47 residual risks: the provider sees the nonce (for example in Ollama debug logs), and runtime KV caches are not tenant-partitioned.
- `RunPins.k` is not meaningful for R47: all items are sent, and the trace holds the real item count.
- R47 on Ollama relies on runtime prefix reuse, which Ollama does not document between requests. There is no KV save, load or truncate. The paper's HF `DynamicCache` path needs torch and a model download; it is not done (open item).
- R49: `cache_metric = "reported"` only when the provider returns a cached-token count (Ollama `prompt_eval_cached_count`, Anthropic `cache_read_input_tokens`), otherwise `unavailable`. Timings are proxies, never a hit claim. Anthropic `cache_control` (5-minute TTL only) is added behind the existing extra. One paid Haiku run, hard cap USD 0.05, run by the owner (approved 2026-11-22).
- R50/R51 key: SHA-256 of canonical JSON of scope, authorization revision (from the caller; open item), question, history, index/corpus/pipeline/prompt/model/generation versions, mode and freshness. No plaintext in keys, traces or stored rows.
- Stored entries hold the response, source document version IDs and the question's guarded features, never the question text. Only `answered` responses with citations whose pins match the key inputs are stored.
- Every hit revalidates scope, expiry, index version, `SourceGate` revocation and anchor resolution. Any failure or store error is a miss.
- Bypass: tools, action runs, enriched or current-information mode. R51 shares the R50 partition and relaxes only the question match.
- Results are `CachedRun(response, trace)`. The `RunResponse` contract is unchanged until v1.2 (open item).
- Storage (owner approved 2026-11-22): port and entry model in `rag_foundry_domain`, an in-memory store, and migration `0014_response_cache` with a SQL store. Rows are scoped, never updated, and removed only by explicit invalidation or expiry purge. A reverse index covers source versions.
- Product source deletion does not exist yet; `SourceGate` models revocation (open item). vLLM metrics: later.
- Error codes are registered once: `RunError.CODES` gains `cag_overflow`, `cag_too_many_chunks`, `cag_nonce_conflict`, `cag_snapshot_mismatch`, `cag_empty`; new `CacheError` holds `cache_key_invalid`, `cache_entry_invalid`, `cache_scope_mismatch`, `cache_unavailable`, `cache_policy_invalid`, `prefix_probe_invalid`, `cache_exists`.

## R47 fidelity
To be filled by P7.4-12.

## R49 fidelity
To be filled by P7.4-12.

## R50/R51 notes
To be filled by P7.4-12.
