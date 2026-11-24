# Source fixture provenance

- Author: written for RAG Foundry by the coding agent (Claude Sonnet 5.5) on 2026-11-01, to be reviewed by the maintainer.
- Licence: the repository licence.
- Third-party text: none. All text is original, short and on a neutral topic (how a library catalogue works). No quotations from books, scripture, news or websites.
- Arabic: Modern Standard Arabic; pending native-speaker review (grammar, naturalness, diacritics).
- Files: `en_short.txt`, `ar_short.txt`, `mixed_short.txt` (UTF-8, no BOM, LF, final newline) and `anchors.json` (SourceAnchor v1 list).
- Block rule: `blocks = text.rstrip("\n").split("\n\n")`; blocks are paragraphs and `block_id` is `p1`, `p2`, ... in order. In `en_short.txt`, paragraphs 1 and 3 are identical on purpose.
- Offset convention: `text_range` is a zero-based, half-open range of Unicode code points within the block text, not UTF-8 bytes or UTF-16 units. Arabic is stored as written (no normalization), so diacritics count as code points.
- Plain text has no pages: `page_index` is null and there are no regions.
