# R23: Multiple representations

## Intuition

A source chunk can be indexed alongside summaries or questions that make it easier to match different queries. Generated representations are synthetic search aids; the original chunk remains the evidence.

## Mechanism

R23 searches stored summary/question vectors and original chunk vectors, then returns the original chunk. Matches identify which representation retrieved the chunk. Synthetic text is stored in `chunk_representations` under ADR-0018 and migration `0010_chunk_representations`; it is never returned as evidence or cited.

## Config and defaults

Search defaults are `k=5` (1-50), `candidate_k=20` (1-100 and must be at least `k`), and `include_original=True`. Build defaults are `n_questions=3` (1-8), `max_summary_chars=600` (50-2000), `max_question_chars=200` (20-500), and `max_chunks=200` (1-2000). The scripted demo builds a summary and question for each of six fixture chunks, and retrieves with `k=3` while including original chunks. Its chat and embedder are scripted doubles, not models. The index version's embedding profile is used for representation vectors; retrieval and generation budgets depend on the configured ports and the number of representation rows.

## Fidelity

Adaptation of multi-representation retrieval. The design keeps each synthetic row linked to its original chunk and separates it from evidence, as recorded in [ADR-0018](../../decisions/ADR-0018-chunk-representations.md). Every lookup must remain within the ready index version, tenant, and project scope. No generated representation is treated as a source.

## Metering

In the scripted demo each chunk receives one summary and one question. The evidence reports scripted usage and zero cost; this is not a real provider cost estimate. Real build usage must account for generation and embedding calls and the number of rows.

## Observations (quoted exactly)

Rows below are exact rows from the scripted demo, no model.

```text
| R23 (k=3, includes original chunks) | fact_en(1), chess(2), bakery(3) | 1 |
| R23 (k=3, includes original chunks) | choir(1), bakery(2), fact_en(3) | 2 |
```

The first row is the success case for a generated question match; the second is the losing case where a vague synthetic summary moves the right chunk to rank 2.

## Limitations

- Synthetic summaries/questions may be vague or wrong and cannot establish facts.
- Preserve the exact span and citation of the original chunk that matched.
- Representation records are immutable and keyed by generator configuration; a known key with no rows is unavailable rather than silently regenerated.
- Scope and version are inherited through the chunk foreign key and checked at lookup.
- No automatic correctness checking of generated text is implemented.

## Status

Implemented; not validated or benchmarked. The demo is scripted, no model, and is not a benchmark.
