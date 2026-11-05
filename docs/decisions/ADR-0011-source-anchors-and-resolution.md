# ADR-0011: Source anchors and resolution

Date: 2026-11-06
Status: accepted (decided by owner 2026-11-06)

## Context
P3.3 needs stable source anchors over the immutable extraction views from ADR-0010, and a way for consumers to resolve them. Chunking changes must not break external links. This follows `SOURCE_ANCHORS_AND_MEDIA.md` and `REUSE_AND_INTEGRATION.md` sections 4 and 6. A public contract and schema lineage change needs an ADR (`10-architecture.md`).

## Decision
Decisions D1-D12, all decided by owner 2026-11-06:
- D1 Anchors are computed only. No anchor table. An anchor is derived from a pinned extraction view, block and range, and is verified against the stored payload when resolved.
- D2 `anchor_id` is deterministic, scheme `anchor-id-v1`, derived from the anchor's pinned source and position, never from chunk or index IDs: `anc_` plus the first 32 hex characters of SHA-256 over the scheme, namespace, document, version, view, block and range (exact encoding in P3.3-02).
- D3 `exact` is always set. `prefix` and `suffix` hold up to 32 code points of context, relative to the block text.
- D4 No new hash field on the anchor. Integrity comes from the immutable view payload.
- D5 `SourceAnchor` stays contract version 1.0 and `source-anchor.v1.json` does not change. The resolve request and response are new additive v1.0 contracts.
- D6 PDF regions (geometry) are deferred until before P9.2. For PDFs the `page_index` comes from the block.
- D7 Resolution statuses are `exact`, `mismatch` and `unavailable`, each non-exact result carrying a typed reason.
- D8 Batch limits: 100 anchors, 10 distinct extraction views, 1 MiB request body. An anchor_id longer than 256 characters is rejected at the request (422).
- D9 The resolver module lives in `packages/ingestion`, not in the contracts package.
- D10 A `provider_namespace` setting names this deployment. Anchors from another namespace resolve as `unavailable` with reason `foreign_namespace`.
- D11 The structure route (`GET .../structure`) is deferred.
- D12 One anchor per block; a span across blocks gives one anchor per overlapped block.

## Alternatives
- Persist anchors in a table: adds schema and drift risk before a query need exists; computed anchors are enough (D1).
- Add a content hash to the anchor: duplicates the view's integrity and changes the v1 contract (D4).
- Put the resolver in contracts: would pull server concerns into a framework-neutral package (D9).

## Consequences
- `packages/contracts` gains `resolve.py`, two committed JSON Schemas and fixtures. SDKs stay free of server imports.
- Resolution never guesses: a changed quote, range or page is a visible `mismatch` or `unavailable`, not a repaired result.
- Not yet evidenced: the resolver logic and the HTTP route (later P3.3 tasks), and any PDF geometry.

## Evidence
Offline contract tests in `tests/contracts/test_resolve.py` cover fixtures, schema drift, bounds, version rejection and status/reason consistency. No resolver behavior is tested yet.
