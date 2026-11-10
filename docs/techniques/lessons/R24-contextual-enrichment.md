# R24: Contextual enrichment

## Intuition

A chunk can be hard to match when it omits the document identity or section context. R24 adds a generated context prefix for indexing, while keeping that text separate from the source.

## Mechanism

The builder lives in `representation_build.py`. It generates one prefix per chunk and stores the synthetic text in `chunk_representations` under ADR-0018 and migration `0010_chunk_representations`. Search may use the prefix vector but returns the original chunk; citations point to the original exact source span. The prefix is synthetic and never evidence.

## Config and defaults

Build defaults are `max_context_chars=400` (50-1500), `max_document_chars=20000` (1000-200000), `max_chunks=200` (1-2000), `max_output_tokens=400` (64-2000), and `on_invalid=fail` (choices: `fail` or `skip`). Embeddings are batched with `embed_batch=32` (1-128). Search inherits `k=5` (1-50) and `candidate_k=20` (1-100 and at least `k`), with `include_original=False`. The scripted demo used a single context prefix per chunk with a scripted responder and embedder. The builder's live usage is one chat call per chunk, counted in usage; embedding calls are batched by `embed_batch`.

## Fidelity

Adaptation of Anthropic's contextual retrieval: https://www.anthropic.com/research/contextual-retrieval. Our prompt is our own wording. The stored synthetic prefix follows ADR-0018's immutable, versioned representation model and is scoped through the original chunk to the ready index version, tenant, and project. The original chunk, not the prefix, is cited.

## Metering

The builder records each chat call's usage; the build cost is one chat call per chunk, counted in usage. The evidence's offline calls are scripted with fake usage and zero cost, not a live cost estimate.

## Observations (quoted exactly)

Rows below are exact rows from the scripted demo, no model.

```text
| R24 (k=3, prefixed text only) | fact_en(1), chess(2), choir(3) | 1 |
| R24 (k=3, prefixed text only) | choir(1), chess(2), bridge(3) | 2 |
```

The first row is the success case where the prefix adds the document name. The second is the losing case where a wrong prefix pulls a distractor ahead of the correct chunk.

## Live smoke

not run

## Limitations

- Context prefixes can be wrong and are not evidence; cite only the original chunk and its exact source span.
- The evidence demo is scripted, no model. The live smoke in the supplied evidence is unfilled, so it is recorded as not run.
- Contextual BM25, prompt caching, and the reranking stage are not implemented.
- No generated-text correctness check is implemented; scope and version follow the original chunk.

## Status

Implemented; not validated or benchmarked. The offline demo uses a scripted chat and embedder, no model.
