# R40: Text-table-image RAG (blocked)

Status: blocked. Owner decision: declined D4 on 2026-11-23 (ADR-0022).

## What the method needs
R40 routes extraction by content type and gives a vision-capable chat model the page or figure images next to text, so chart and table questions can be answered from the image, with links back to the source.

## Why we stop
The provider `ChatMessage` schema has no image parts and no adapter has a vision chat path. R41 returns original text chunks as evidence and does not send page images to a chat model, so nothing in the repo is R40. Generated captions would be synthetic text and may not be used as evidence.

## Exact requirements (ADR-0022, D4)
- Image parts in the provider `ChatMessage` schema, with size and format limits.
- A vision chat path in an adapter (`gemma4:e4b` lists vision in `ollama show`, so no pull is needed).
- Synthetic captions kept out of evidence.
- Page-image rendering: exists after P7.5-07.

## What exists
Page rendering and the R41 retriever (see [R41](R41-visual-multivector.md)). Blocked record with the same text: `docs/evidence/P7.5/visual_multivector.md`, section "R40 blocked record".

## Losing and success cases
None run; no model result exists for R40. The catalogue entry stays with status `blocked` and is not removed.

## Open items
The vision path above, and chart/table test questions that include generated-caption errors.
