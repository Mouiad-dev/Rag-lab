# ADR-0010: Extraction views and normalization

Date: 2026-11-05
Status: accepted

## Context
P3.2 turns document bytes into text with stable positions. Chunking (P3.3) and source anchors need an owner for that text that does not change when chunking changes. Lineage and schema changes need an ADR (`10-architecture.md`). This follows ADR-0009 (`Document -> DocumentVersion`) and `SOURCE_ANCHORS_AND_MEDIA.md`.

## Decision
- Lineage: `DocumentVersion -> ExtractionView -> SourceBlock`. A view is immutable and owned by exactly one version, never by chunks or index IDs. Rechunking keeps external anchor links.
- Identity key: `(document_version_id, extractor, extractor_version, profile_id, unicode_version)`. The same key returns the existing view (dedup is atomic in the repository). Any change makes a new view. Old views stay valid.
- Text layers: decoded source (or PDF raw page text) -> canonical text (stored, `canonical-v1`) -> search text (derived on demand, never stored or anchored).
- `block_text = canonical[block.start:block.end]` (half-open code points). `SourceAnchor` v1 `text_range` is relative to `block_text`. Anchors target canonical (NFC) text, so `exact` can differ from the original in combining-mark order. `source_map` reaches the decoded source or PDF raw text.
- "Reversible" means positions map back. Text comes from the stored original; a lossy step is never inverted.
- Errors are typed (`ExtractionError` codes) and carry no content or paths.

Decisions D1-D12, all decided by owner 2026-11-05:
- D1 PDF library `pypdfium2` (Apache-2.0/BSD-3, PDFium BSD-3 plus bundled third-party notices), verified and pinned in task 04. Rejected: PyMuPDF (AGPL), poppler tools (GPL).
- D2 HTML uses a third-party parser, chosen in task 03 after verification (candidates selectolax, html5lib, lxml). Offsets come from an alignment layer over the source. Unalignable text raises `alignment_failed`, never a guess. No network, scripts or external entities.
- D3 Markdown: stdlib line scanner, markup kept verbatim.
- D4 No API trigger. A service plus a dev/test CLI in the worker; a queued job at P8.4; never in the API process. Known gap: no subprocess isolation in P3.2, so a native crash or hang in a parser can take down the worker. Subprocess isolation with timeout and memory limit is planned for P8.4.
- D5 Metadata row in Postgres. Payload (canonical text, blocks, source map, IDs) as canonical JSON in the object store. No blocks table yet.
- D6 `canonical-v1` = LF newlines + NFC only.
- D7 `search-ar-en-v1`: NFKC, casefold, NFKC, strip Cf and U+034F, tatweel, harakat U+064B-U+065F and U+0670, NFKC (added by owner 2026-11-05: stripping can bring a base next to a mark it composes with, so this keeps the profile idempotent), alef/yeh/kaf folds, Arabic-Indic digits, whitespace collapse. Not folded: teh marbuta, hamza seats U+0624/U+0626, Quranic marks U+06D6-U+06ED. Alef maksura to yeh trades precision for recall (U+0639 U+0644 U+0649 and U+0639 U+0644 U+064A then match). Search text only, never stored or anchored.
- D8 PDF: one `page` block per page that has text, joined by `\n\n`, no geometry until P3.3. Pages without text get no block and a `page_without_text:<i>` warning, so `page_starts[i]` belongs to block `i`; use the block's `page_index` for the real page number.
- D9 The Arabic PDF test is skipped until the owner supplies a license-clean fixture (known gap).
- D10 Dedup by identity key.
- D11 Limits: 2,000 pages and 5,000,000 code points (worst-case offset map about 5M segments x 40 B = 200 MB; typical far less since identity runs merge). Encrypted PDFs (needing a user password) and PDFs without a text layer are typed errors. Empty pages are warnings.
- D12 Views record `unicodedata.unidata_version`. A Python upgrade creates new views. Tests never hard-code it.

## Alternatives
- Views owned by chunks: anchors would break on rechunking; rejected.
- Store search text: couples storage to a folding profile; derive on demand instead.
- PyMuPDF or poppler: copyleft licenses; rejected (D1).
- Blocks table now: more schema before a query need exists; JSON payload first (D5).

## Consequences
- `packages/domain` gains `ExtractionView`, `SourceBlock`, an `ExtractionRepository` port, an in-memory implementation and reusable `EXTRACTION_REPOSITORY_CHECKS`; the SQL repository (task 05) must pass the same checks.
- Views are append-only. Rollback means ignoring newer views; payload objects are content-addressed and harmless if unreferenced.
- A new extractor, profile or Python Unicode version costs one extra payload per version, not a migration.
- Not yet evidenced: any extractor, the SQL repository and real PDFs (tasks 02-07).

## Evidence
Offline tests: `tests/domain` run the checks on the in-memory repository and on deliberately broken ones. Library pins (D1, D2) are verified in tasks 03 and 04.
