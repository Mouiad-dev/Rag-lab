# ADR-0016: Chunk parent spans

Date: 2026-11-08
Status: accepted

## Context
C18 creates child chunks within parent chunks. Index inspection and retrieval need to expose the parent range so callers can resolve useful parent context against the extraction view. Chunk parent ids are stable derivations from canonical text and offsets through `parent_id_for`.

## Decision
Store nullable `parent_start_cp` and `parent_end_cp` offsets on index chunks. Both are null for chunkers without parent linkage. Derive parent ids with `parent_id_for`; do not store another identifier format. Parent ranges are chunk-level references, not source anchors. Inspection resolves parent anchors from the range. Query-time parent return remains deferred to R22/R25 at P6.2.

## Consequences
The index schema and chunk contracts carry a validated nullable span pair, while parent identity remains deterministic and derived. Inspection and R02/R03 results can expose the parent range without changing source-anchor storage. Chunks from C05 and other unlinked chunkers remain unchanged with null offsets.
