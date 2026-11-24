# R50: exact response cache

## What it is
R50 reuses a stored answer and its evidence only when the effective request is identical. It is a `cache` plugin (`rag_foundry_cache.exact`). It is not CAG (R47) and not a prefix cache (R49): a hit skips retrieval and generation entirely.

## How it works here
- Key: SHA-256 of canonical JSON of scope, question, history, authorization revision (from the caller; open item), technique id and version, retriever config hash, index version, embedding key, prompt template and hash, chat profile, provider and model, typed generation parameters (`max_output_tokens`, `temperature`, `k`, `min_score`, `max_evidence_chars`, `packer`, `citation_check`; all required, no defaults), settings hash, mode, freshness, tools, action flag, output language and contract version. No plaintext in keys, traces or rows.
- Partitions: tenant and project scope are part of the key and are rechecked on every hit.
- Bypass (the store is never touched): `freshness_current`, `enriched_mode`, `tools`, `action_run`.
- Store only `answered` responses with citations. Pins check before store: an answer whose pins differ from the key inputs is not stored (`pins_mismatch`).
- Revalidation order on a hit: `scope_mismatch`, `expired`, `index_version_changed`, `source_denied` (via `SourceGate`), `anchor_unresolved`. Any failure or store error is a miss followed by a fresh run; a foreign entry is never returned, deleted or overwritten.
- Storage: in-memory store and SQL store (migration 0014). Rows hold the response, source version IDs and guarded features; no question text. Rows are never updated and are removed by explicit invalidation or expiry purge.

## Fidelity: adaptation
There is no external paper; this is standard keyed response caching, built to the lab's scope, provenance and revalidation rules.

## What a hit does not prove
A hit proves the key matched and the checks passed, not that the answer is still right: freshness depends on the key fields and on `SourceGate`. There is no product source deletion yet, so the default gate (`AllowAllSourceGate`) denies nothing and the hit trace says `revocation_checked=false`.

## Success example (offline, scripted)
The repeat of the same request is a hit: 0 calls, 2 avoided. Changed history, authorization revision or tenant: miss (2 calls). Index rebuilt after storing: `index_version_changed`. Expired entry: `expired`. Source denied by the gate: `source_denied`. See `docs/evidence/P7.4/response_cache_demo.md`.

## Losing examples
A hit is only as fresh as its key: a change of source text is noticed only through the index version, `SourceGate` and anchor resolution. A forced foreign entry gives `scope_mismatch` and is skipped, not served.

## Cost per stage
Offline (scripted): cold run 2 ledger calls (query embedding plus chat), 256 estimated input tokens; warm 0 calls, 2 avoided; R50 needs no embedding for lookup. Live: cold calls 2 model calls (input tokens 439, 421, 439); warm 0 model calls, input tokens avoided 439, 421, 439.

## Cold vs warm (live, one run, n=3 questions, not a benchmark)
Local Ollama, `gemma4:e4b` and `bge-m3`: 3 of 3 warm calls hit with answers equal to the cold answers. Single-run wall times: cold 1354, 1298, 1474 ms; warm 4, 4, 4 ms (lookup 3.7 to 3.8 ms).

## Open items
API/UI wiring and cache hits in `RunResponse` (contract v1.2); product source deletion wired to `SourceGate`; ACL revision source; SQL store wiring and scheduled purge; Redis.
