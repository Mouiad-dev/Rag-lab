# ADR-0018: Chunk representations

Date: 2026-11-10
Status: accepted

## Context
R23 (summaries and questions per chunk) and R24 (context prefixes) add generated text next to original chunks. Generated text can be wrong, so it must never be mistaken for source evidence.

## Decision
- Representations are synthetic. Records always carry `synthetic = true`, and they are never returned as evidence or citations.
- A set of representations is keyed by `representation_key`, the sha256 of the canonical generator config (technique id, model, prompt version, parameters).
- They are embedded with the index version's own embedding profile, so vectors are comparable with chunk vectors.
- They are immutable. A duplicate (chunk, key, kind, ordinal) is rejected, and a key reused with a different technique or generator config is a conflict.
- Scope is inherited through the chunk foreign key to one ready index version; every lookup rechecks tenant and project.
- Search runs over representation vectors but returns the original chunk text with the matching representation as a separate field.
- The port is `RepresentationStore`, separate from `IndexRepository`. A known key with no rows fails with `representations_unavailable`.

## Consequences
Original text and synthetic text stay apart end to end, and provenance remains with the original chunk. Changing the generator creates a new key instead of editing rows. P6.2-03b implements the same port on PostgreSQL against the shared conformance check.
