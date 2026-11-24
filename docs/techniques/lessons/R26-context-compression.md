# R26: Context compression

## Intuition
Keep question-relevant source sentences while retaining qualifiers that carry constraints.

## Mechanism
`lexical-overlap-v1` selects sentences whose content-term overlap with the question meets `min_overlap`; qualifier sentences are retained when enabled. It emits original sub-spans and reports constraints lost from removed text.

## Config and defaults
`ExtractiveCompressionConfig`: `min_overlap=1` (1..10), `keep_qualifiers=True`, `max_total_chars=8000` (20..100000), `max_hits=50` (1..50). Questions are limited to 2000 characters. These limits are enforced by the public API.

## Fidelity
ADAPTATION: extractive lexical sentence selection only. LLM abstractive compression is not implemented (owner decision 2026-11-10); if added later its output must be synthetic and never cited.

## Metering
Zero model calls. Output rows are from a tiny hand-made fixture, not a benchmark.

## Runnable demo
`uv run --no-sync python docs/techniques/lessons/demos/p63_packing_citations_demo.py`

## Observations (quoted exactly)
Success: `| This does not apply to bicycles. | 0 | qualifier | not |`

Losing case: `| keep_qualifiers=False | The bridge opened on 3 March 2019. Tolls are 5 USD per car. | ('not',) |`

## Limitations
Question-term lexical overlap can miss synonyms. Provenance retains exact source sub-spans; no synthetic text is created. This tiny fixture does not establish retrieval recall or context-size behavior.

## Status
Implemented; primary source is null. Evidence: `docs/evidence/P6.3/packing_and_citations.md`.
