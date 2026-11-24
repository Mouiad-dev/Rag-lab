# C15: Embedding-semantic boundaries

## Intuition
Neighbouring sentences about the same thing have similar embeddings. Where the similarity drops, the topic probably changed, so put a chunk boundary there.

## Mechanism
`SemanticChunker` is in `packages/chunking/src/rag_foundry_chunking/semantic.py`. It is async and needs an `EmbeddingModel` port: call `chunk_with_models(doc, ModelChunkContext(embedder=..., cache=...))`. The plain `chunk(doc)` raises `TechniqueUnavailableError`.

1. Split the canonical text into sentence units (C06 rules; an oversized sentence becomes words, then safe cuts).
2. Embed each unit together with `buffer_size` neighbours on each side (default 1).
3. Distance between neighbours = `1 - cosine`.
4. Threshold: `absolute` (a fixed distance), `percentile` (the given percentile of all distances in this document) or `std_dev` (mean + k standard deviations).
5. A break goes after a unit whose distance is strictly above the threshold. Segments are packed under `chunk_size` code points.

Boundaries are always unit boundaries of the source text, so every chunk is an exact source slice. The distances are cached under a key built from the source text hash, the embedding profile (provider, model, id), the config and the plugin version.

## Worked example
A toy embedder (axis 0 counts "cat", axis 1 counts "market", axis 2 is the sentence length / 10), `buffer_size=0`. Real output of the demo below:

```text
topic change absolute   distances=[0.002, 0.297, 0.0] threshold=0.500
    ['The cat sleeps. The cat eats. The market rises. The market falls.']
topic change percentile distances=[0.002, 0.297, 0.0] threshold=0.238
    ['The cat sleeps. The cat eats.', 'The market rises. The market falls.']
cohesive     absolute   distances=[0.002, 0.001, 0.0] threshold=0.500
    ['The cat sleeps. The cat eats. The cat plays. The cat purrs.']
cohesive     percentile distances=[0.002, 0.001, 0.0] threshold=0.002
    ['The cat sleeps.', 'The cat eats. The cat plays. The cat purrs.']
```

## Runnable demo
```bash
uv run python docs/techniques/lessons/demos/c15_semantic_demo.py
```
No network and no keys: the embedder is a small class inside the demo.

## Failure case
Both threshold kinds fail in the example above. The absolute threshold 0.5 misses the real topic change, because the extra length feature makes the topic distance only 0.297. The percentile threshold always cuts the largest distances of the document, so it also cuts the cohesive passage after "The cat sleeps." where the distance is only 0.002. Real embedding models also place Arabic and English sentences differently, so one threshold may not work for both.

## Limitations
- The result depends fully on the embedding model and its languages; the chunker only applies a rule to the vectors.
- `buffer_size > 0` smooths noise but also smears the boundary over neighbouring windows (with buffer 1, the window before a topic change already contains the new topic).
- One embedding call per batch of units; cost and latency grow with document length (`max_units` stops very long inputs before any call).

## Baseline comparison
Not measured yet; comes with the P4.5 comparison report.

## Status
Draft.
