# R51: semantic response cache

## What it is
R51 reuses an answer for a question that is close in meaning, not identical. It is a `cache` plugin (`rag_foundry_cache.semantic`). Similarity alone is not answer equivalence, so R51 adds a margin and a guard on top of it.

## How it works here
- Partition: the exact R50 partition (same key fields except the question; same bypass rules: `freshness_current`, `enriched_mode`, `tools`, `action_run`). Other tenants never match (`no_candidate`).
- A hit needs: cosine similarity at or above the threshold; a margin over the runner-up (default 0.05, else `ambiguous`); equal guarded features; and the R50 revalidation in the same order (`scope_mismatch`, `expired`, `index_version_changed`, `source_denied`, `anchor_unresolved`). Pins are checked before store.
- There is no default threshold: it comes from a calibration the caller ran. A failed embed or store call is a miss.
- Guard: numbers, identifiers, quoted text, negation, comparatives and script. Not seen by the guard: month names, units, entity swaps without digits.
- Storage: in-memory and SQL (migration 0014). Entries hold a vector, the answer and guarded feature digests (SHA-256), never question text. Residual risk (orchestrator note 2026-11-23): stored feature hashes of short values (numbers, identifiers, quoted strings) can be guessed by brute force; they are scoped to one tenant partition and are not encrypted.

## Fidelity: adaptation
Semantic response caching in the style of embedding-based caches; the guard, margin and revalidation are lab additions. Not a validated technique.

## What a hit does not prove
Similarity at 0.96 and a passed guard do not mean the same question. A hit still pays one cache-embedding call (it avoids retrieval embedding and chat). No product source deletion exists yet, so `SourceGate` is the only revocation path.

## Success example (offline, lexical embedder)
Paraphrase at threshold 0.8: similarity 0.96, hit, 0 calls, 2 avoided. Guard stops 2025 vs 2026 (`features_differ:numbers`), with vs without (`negation`), the Arabic negation and the identifier pair at threshold 0.5.

## Losing examples
Held-out (offline, lexical): month name (held-10, similarity 0.95) and unit (held-11, 0.94) are false hits that pass similarity and guard. Entity swaps without digits (dev-11, held-12) pass the guard and are stopped only by similarity just under the threshold. With real bge-m3 at 0.87, 2 of 7 held-out adversarial pairs are still false hits (guarded).

## Cost per stage
Offline (scripted): cold 2 answer calls, 1 cache-embed call; warm 0 answer calls, 1 cache-embed call, 2 avoided.

## Calibration (24 illustrative pairs, dev and held-out; not a benchmark)
Lexical test embedder (offline), chosen from dev only: threshold 0.93. Dev false hits at 0.93: 4/7 unguarded, 0/7 guarded; dev paraphrase 4/5. Held-out at 0.93: paraphrase hits 4/5 (both), false hits 4/7 unguarded, 2/7 guarded.

Real `bge-m3` (live, one run, Ollama): chosen threshold 0.87. Dev false hits at 0.87: 4/7 unguarded, 0/7 guarded; dev paraphrase 5/5. Held-out at 0.87: paraphrase hits 5/5 (both), false hits 5/7 unguarded, 2/7 guarded.

The two tables use different embedders and thresholds and are not comparable. Small sets, one run, n=1.

## Open items
Month, unit and entity features; API/UI wiring and `RunResponse` hits (contract v1.2); SQL store wiring and scheduled purge; Redis; encrypting or keying the stored feature digests; product source deletion wired to `SourceGate`.
