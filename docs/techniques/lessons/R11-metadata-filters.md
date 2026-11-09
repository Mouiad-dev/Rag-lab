# R11: Metadata filtering

## Intuition

Restrict retrieval to documents that match structured facts: which document, which media type, which creation period. The filter only narrows what the caller is already allowed to see.

## Mechanism

`MetadataFilteredRetriever` wraps an inner retriever. It rechecks that the index version is ready for the scope, resolves which extraction views match the filter (by document, document version, media type and version creation date), and returns an empty result without calling the inner retriever when no view matches. Otherwise it calls the inner retriever once and keeps only hits from matching views, re-ranked from 1 and truncated to `k`. Each hit keeps its provenance and gets an `input_rank` component.

## Config and defaults

`MetadataFilterConfig`: `k` 5 (1-50), `filter` empty. `MetadataFilter` fields: `document_ids`, `document_version_ids`, `media_types`, `created_from` (inclusive), `created_to` (exclusive). A per-call `extra_filter` is intersected with the configured filter, so it can only narrow.

## Fidelity

Adaptation, post-filter only. The filter is applied after the inner retriever returns, not inside the index search. `language` and `section` filters are not available yet because those are not stored per chunk.

## Runnable demo

```bash
PYTHONIOENCODING=utf-8 uv run python docs/techniques/lessons/demos/p53_selection_demo.py
```

Full output: [selection_and_filters.md](../../evidence/P5.3/selection_and_filters.md). It uses the three-document fixture with the test-only `LexicalTestEmbedder` (not a semantic model). Offline fixture; no model; not a benchmark.

## Success case

```text
| no filter | 6 | 1 | 1 | filter[R02]:post:3/3views |
| document report-2025 | 2 | 1 | 1 | filter[R02]:post:1/3views |
| media type text/markdown | 2 | 1 | 1 | filter[R02]:post:1/3views |
| stranger-scope document id | 0 | 0 | 0 | - |
```

The document and media filters keep only that document's 2 chunks. A document id that belongs to another scope matches no view in this index, returns 0 hits and never calls the inner retriever or the embedder.

## Losing case

```text
| inner top 2 holds 0 hit(s) of that document | 2 | 0 |
| same filter, same query | 50 | 2 |
```

With an inner retriever of `k=2`, the top 2 hits came from other documents, so the post-filter leaves 0 hits although 2 matching chunks exist. With inner `k=50` the same filter returns 2. Build the inner retriever with a large `k` (for example 50).

```text
| language / section filter | not available |
```

## Limitations

Post-filtering can return fewer than `k` hits. `language` and `section` filters need per-chunk storage first. The date filter uses the document version creation time.

## Status

Implemented; not validated or benchmarked.
