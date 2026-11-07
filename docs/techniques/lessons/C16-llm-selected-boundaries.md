# C16: LLM-selected boundaries

## Intuition
A chat model reads the text and says where a new topic starts. The model only points; it never writes the chunks. Every pointer is checked against the original text.

## Mechanism
`LlmBoundaryChunker` is in `packages/chunking/src/rag_foundry_chunking/llm_boundaries.py`. It is async: `chunk_with_models(doc, ModelChunkContext(chat=..., prompt=..., cache=...))`; the plain `chunk(doc)` raises `TechniqueUnavailableError`.

1. Split the canonical text into sentence units (C06 rules) and number them.
2. Send windows of at most `max_units_per_call` units, each inside nonce delimiters, with the active `chunk-boundaries` prompt template (seeded by migration 0006 as `chunk-boundaries-v1`). More windows than `max_calls` raise `too_many_calls` before any call.
3. The model answers `{"breaks": [{"unit": 3, "starts_with": "first words of unit 3"}]}`. JSON is parsed from text unless the profile declares `structured_output=True`, then a JSON schema is sent.
4. Validation: the unit must be an integer inside the window (not the first unit), ascending; the quote (after NFC normalization and edge-whitespace strip) must be the exact start of that unit and at least `min_quote_chars` long. Each bad item is rejected with a reason code (`out_of_range`, `not_ascending`, `quote_mismatch`, ...). A bad whole answer (`invalid_json`, `invalid_shape`, `truncated`) discards that window (`fallback_calls`), or raises with `on_invalid="fail"`.
5. Accepted breaks split the units; segments are packed under `chunk_size`. Offsets come only from the source units, so no model text can enter a chunk.

Raw model answers are cached under a key of source text hash, model profile (provider, model, id), prompt id + sha256, config, plugin version and `sample`. Validation runs again on every cache hit. Budgets, retries and usage recording are the caller's port wrappers; the report carries `usage`, `total_cost_usd`, `model_calls` and all rejections. `boundary_agreement(a, b)` (Jaccard of break offsets) measures repeat variability between `sample` values.

## Worked example
A scripted model (no network) proposes a break at unit 2 with a correct quote and one at unit 4 with an invented quote. Real output of the demo below:

```text
spans: ['The cat sleeps on the warm mat. The cat wakes up at noon.', 'The market opened higher today. Shares of banks rose. The weather turns cold tonight. Rain is expected tomorrow.']
accepted break offsets: [58]
rejected: quote_mismatch {'unit': 4, 'starts_with': 'Snow falls in the city'}
second run: cache_hit = True model_calls = 0
```

## Runnable demo
```bash
uv run python docs/techniques/lessons/demos/c16_llm_boundaries_demo.py
```
It uses the exact `chunk-boundaries-v1` text from migration 0006.

## Failure case
The invented quote "Snow falls in the city" is rejected, so the real topic change before "The weather turns cold tonight." is lost: the second chunk mixes markets and weather. C16 never repairs a wrong answer into a boundary; a rejected boundary is a missed boundary.

## Limitations
- One model call per window: cost and latency grow with document length.
- The same model can choose different boundaries on repeat runs; compare with `boundary_agreement` using `sample` 0, 1, ...
- Some models ignore "JSON only"; those answers are rejected. gemma4 on Ollama is declared `structured_output=False` in this repository, so JSON is parsed from text.
- Breaks between windows are never proposed; only budget packing splits there.
- No live model run has been recorded yet (needs the owner's OK).

## Baseline comparison
Not measured yet; comes with the P4.5 comparison report.

## Status
Draft.
