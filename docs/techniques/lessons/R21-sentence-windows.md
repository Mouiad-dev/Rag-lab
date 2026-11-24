# R21: Sentence windows

## Intuition

A matching sentence can be too small to carry the context needed to understand it. R21 expands a retrieved seed to nearby source text, while retaining the seed's provenance.

## Mechanism

Expansion happens at query time from the canonical view text. Window metadata is not stored at index time. The configured window may remain within a paragraph or cross paragraph boundaries. This is sentence context expansion, not GraphRAG: it follows textual neighbors and builds no entity graph.

## Config and defaults

Code defaults are `window=1` (range 1-5), `max_window_chars=2000` (20-20000), `max_total_chars=8000` (20-100000), and `max_seeds=20` (1-50). `stay_in_paragraph` defaults to true. The seed count, individual window size, and aggregate output are bounded by those caps; an oversized window cannot exceed its character budget.

## Fidelity

Adaptation of sentence-window retrieval. The demo's `R21-cross` setting changes `stay_in_paragraph` to false to allow neighboring paragraphs. The fixture paragraphs are single sentences, so the default R21 setting may return the same context as the seed. No GraphRAG behavior is implemented.

## Metering

Expansion itself adds no model call. The run's query-embedding usage is recorded in [the evidence](../../evidence/P6.2/seed_context_runs.md); returned text is bounded by `max_seeds` and `max_total_chars`.

## Observations (quoted exactly)

Rows below are exact per-case rows from the tiny offline fixture, not a benchmark.

```text
| en-fact | dev | en | - | R21 | 1 | 1.0000 | 1.0000 | 1.0000 | paired-fixture:0-73 | ok |
| paraphrase | dev | en | paraphrase | R21 | 5 | 0.2000 | 1.0000 | 0.3869 | paired-fixture:500-555 | ok |
```

The paraphrase row is the losing example at rank 5. A window cannot guarantee that a weak seed becomes a useful first result.

## Limitations

- Expansion is limited to source neighbors and configured character/seed budgets.
- Canonical view scope and version determine the text being expanded; stale or out-of-scope source data is not a valid substitute.
- The exact-span rule remains: retain the seed's exact source span and provenance; expanded context does not create a new citation span.
- No graph traversal, learned window selection, or quality guarantee is implemented.

## Status

Implemented; not validated or benchmarked. Evidence is a tiny offline fixture using a test-only lexical embedder.
