# C17: Propositions linked to source spans

## Intuition
Split a passage into small, self-contained facts ("propositions") so one retrieved unit holds one fact. The idea comes from Chen et al. 2023, "Dense X Retrieval: What Retrieval Granularity Should We Use?" (arXiv 2312.06648). Here the model must also copy exact quotes that support each fact, so every fact is tied to a source span.

## Mechanism
`PropositionChunker` is in `packages/chunking/src/rag_foundry_chunking/propositions.py`. It is async: `chunk_with_models(doc, ModelChunkContext(chat=..., prompt=..., cache=...))`; `chunk(doc)` raises `TechniqueUnavailableError`.

1. Pack sentence units into passages of at most `passage_size` code points; one model call per passage (more than `max_calls` passages raise before any call).
2. The passage goes inside nonce delimiters with the active `propositions` prompt template (migration 0006 seeds `propositions-v1`). The model answers `{"propositions": [{"text": "...", "quotes": ["..."]}]}`.
3. Every quote (NFC, edge whitespace stripped) must occur exactly in the passage, else `quote_not_found`. The proposition's source span runs from its first quote to its last.
4. Lexical checks (not entailment): a number in the proposition that is not in the passage (`unsupported_number`); more than `max_unsupported_ratio` of content words absent from the passage (`unsupported_claim`, Arabic `ال` prefix tolerated); a negation cue in the quoted span but not in the proposition (`negation_dropped`) or the reverse (`negation_added`); a condition cue in the quoted span but not in the proposition (`condition_dropped`).
5. Output: `propositions` are `PropositionRecord`s with `generated=True`, the model text and the source offsets. The chunk spans are the distinct source spans of accepted propositions.

Generated proposition text is never chunk text and never evidence: citations use the quoted source span. Raw answers are cached by source text hash, model profile, prompt id + sha256, config, plugin version and `sample`; validation runs again on every cache hit.

## Worked example
A scripted model (no network) answers for one passage. Real output of the demo below:

```text
accepted: 'The city library does not open on Sundays.' -> source span 'It does not open on Sundays'
accepted: 'The garden closes early.' -> source span 'the garden closes early'
accepted: 'The city library reading room has twelve large tables.' -> source span 'twelve large tables'
accepted: 'The reading room has twelve large wooden tables.' -> source span 'twelve large tables'
rejected: negation_dropped: 'The city library opens on Sundays.'
rejected: unsupported_number: 'The library opens at 10 am on weekdays.'
rejected: quote_not_found: 'The library has a cafe.'
rejected: unsupported_number: 'The reading room has 12 big desks.'
chunk spans: ['It does not open on Sundays', 'the garden closes early', 'twelve large tables']
```

The first proposition shows decontextualization: "It" becomes "The city library", which is allowed because the words come from the same passage.

## Runnable demo
```bash
uv run python docs/techniques/lessons/demos/c17_propositions_demo.py
```
It uses the exact `propositions-v1` text from migration 0006.

## Failure case
The checks are lexical heuristics, and the demo shows both kinds of error:
- False accept: "The garden closes early." drops the condition "If it rains", but the model quoted only "the garden closes early", so the condition cue is outside the quoted span and `condition_dropped` does not fire. "twelve large wooden tables" adds "wooden", but 1 unsupported word out of 6 is under the 0.2 ratio.
- False reject: "The reading room has 12 big desks." means the same as the source ("twelve large tables"), but `12` is not a passage token, so it is rejected as `unsupported_number`.

## Limitations
- No entailment model: synonyms, number words and paraphrases are not understood; contradictions that keep the same words pass.
- One model call per passage; cost and latency grow with document length. Repeat runs can differ (use `sample`).
- Proposition text is stored only in the result record; indexing propositions (Dense X style retrieval) is not wired yet.
- No live model run has been recorded yet (needs the owner's OK).

## Baseline comparison
Not measured yet; comes with the P4.5 comparison report.

## Status
Draft.
