# R25: Hierarchical auto-merging

## Intuition

Several retrieved children can point to one larger parent. R25 merges siblings only when enough children match, subject to a clear ratio and context budget.

## Mechanism

R25 uses stored C18 child-to-parent relationships. For a parent with `n` children and `m` retrieved children, merge exactly when `m/n >= merge_ratio and m >= min_children`; the result has two levels only. It does not recursively merge to arbitrary hierarchy depth. The fixture settings are `merge_ratio=0.5`, `min_children=2`, `max_parent_chars=4000`, `max_total_chars=8000`, `max_seeds=20`, and `max_index_chunks=5000`. The index-size limit is checked before chunks are listed.

## Config and defaults

Code defaults are `merge_ratio=0.5` (range (0,1]), `min_children=2` (1-10), `max_parent_chars=4000` (20-20000), `max_total_chars=8000` (20-100000), `max_seeds=20` (1-50), and `max_index_chunks=5000` (1-20000). The evidence run records the settings above; `R25-min1` changes only `min_children` to 1. Parent size, number of seeds, total output characters, and index chunk count are bounded. A parent that does not fit remains represented by its child context.

## Fidelity

Adaptation of hierarchical auto-merging over the project's C18 index and stored spans. No library comparison was checked. Scope and version checks follow the ready index version and its tenant/project ownership. The merged result preserves source lineage and exact underlying spans.

## Metering

Merging uses retrieved children and stored parent spans; it adds no generation call. Query embedding is recorded separately. The configured seed, parent, aggregate character, and index-size limits bound work and output.

## Observations (quoted exactly)

Rows below are exact per-case rows from the tiny offline fixture, not a benchmark.

```text
| en-fact | dev | en | - | R25 | 1 | 1.0000 | 1.0000 | 1.0000 | paired-fixture:0-186 | ok |
| paraphrase | dev | en | paraphrase | R25 | - | 0.0000 | 0.0000 | 0.0000 | paired-fixture:500-547 | ok |
```

The paraphrase row is the losing example: no relevant span is covered in its top five.

## Limitations

- Exactly two levels; merging is governed by `m/n >= merge_ratio and m >= min_children`.
- A parent that exceeds the configured limit cannot be returned as an unbounded context.
- Keep exact source spans and lineage when combining siblings; the merge itself does not create a source span.
- No library equivalence or comparison was checked, and no adaptive hierarchy is implemented.

## Status

Implemented; not validated or benchmarked. Evidence is a tiny offline fixture using a test-only lexical embedder.
