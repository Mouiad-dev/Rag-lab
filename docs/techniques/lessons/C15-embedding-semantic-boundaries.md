# Intuition

Sentence embeddings let a chunker keep nearby sentences together while placing a boundary where neighbouring sentence windows change topic. The resulting boundaries remain on canonical sentence ranges.

# Mechanism

C15 embeds each sentence together with a configurable number of neighbours. It measures cosine distance between adjacent vectors, then applies an absolute, percentile, or mean-plus-standard-deviation threshold. Segments are packed under the configured code-point budget.

# Worked example

For four cat sentences followed by four market sentences, a topic-aware embedding model should make the distance near the topic change larger than the other neighbouring distances. With an absolute threshold of `0.5`, C15 places a break after the fourth unit when that distance passes the threshold.

# Runnable demo

The test helper `tests/chunking/_scripted_models.py` provides `topic_vector`: it counts cat and market terms, including escaped Arabic tokens, and adds deterministic hash-derived perturbations. Pass it to `ScriptedEmbedder` and run `SemanticChunker.chunk_with_models` with an `InMemorySegmentationCache` to inspect the spans and report.

# Failure case: percentile cuts a cohesive passage; embedding models disagree across languages

A percentile threshold always selects a cut relative to the largest distances, even when all sentences discuss one topic. An embedding model can also place Arabic and English sentences differently, so the same threshold may not produce equivalent boundaries across languages.

# Limitations

The distance is only as useful as the selected embedding model and its treatment of the document's language. This chunker does not infer semantic correctness; it applies a deterministic rule to the vectors it receives.

# Baseline comparison: not measured yet (P4.5)

# Status: Draft
