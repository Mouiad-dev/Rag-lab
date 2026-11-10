# R22: Parent document retrieval

## Intuition

A small child can locate a fact while its stored C18 parent provides the surrounding context. Returning a parent is useful only when the child-to-parent lineage and access scope remain intact.

## Mechanism

R22 searches indexed children and returns the corresponding stored parent span. It requires a C18 parent-child index. By default, `on_missing_parent=error` raises `parent_unavailable`; `keep_child` keeps the child span with flag `no_parent`. No parent is ever made up. Parent output is capped, and repeated parents are collapsed.

## Config and defaults

Code defaults are `max_parent_chars=4000` (20-20000), `max_total_chars=8000` (20-100000), `max_seeds=20` (1-50), and `on_missing_parent=error` (choices: `error` or `keep_child`). The index note records C18 `parent_size=200` and `child_size=50`. Parent, seed, and total output limits bound returned context.

## Fidelity

Adaptation of parent-document retrieval using the project's C18 stored spans. Scope is inherited from the ready index version and must be checked for each lookup. The implementation returns authorized parent context associated with a hit child; it does not reconstruct an absent parent.

## Metering

No generation call is used for parent expansion. Query embedding is shown separately in the evidence usage table. Output is bounded by `max_parent_chars`, `max_seeds`, and `max_total_chars`.

## Observations (quoted exactly)

Rows below are exact per-case rows from the tiny offline fixture, not a benchmark.

```text
| en-fact | dev | en | - | R22 | 1 | 1.0000 | 1.0000 | 1.0000 | paired-fixture:0-186 | ok |
| paraphrase | dev | en | paraphrase | R22 | 4 | 0.2500 | 1.0000 | 0.4307 | paired-fixture:500-681 | ok |
```

The paraphrase row is the losing example at rank 4. Parent context can provide more text without moving a weak match to the top.

## Limitations

- Requires an index built with C18 parent-child links; a missing parent raises `parent_unavailable` by default, while `keep_child` retains the child span with `no_parent`.
- Parent context can be larger than the child and is limited by configured character and aggregate budgets.
- Scope and version must match the ready index version, tenant, and project on lookup.
- Preserve the child-to-parent lineage and exact source span; the expanded parent is context, not permission to cite an unrelated span.
- No parent is made up when one is missing.

## Status

Implemented; not validated or benchmarked. Evidence is a tiny offline fixture using a test-only lexical embedder.
