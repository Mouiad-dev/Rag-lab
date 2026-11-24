# ADR-0017: Learned sparse encoder port

Status: accepted (2026-11-08)

R05 uses a provider `SparseEncoder` port so indexing depends on a framework-neutral interface.
The approved `sentence-transformers` 6.2.0 Apache-2.0 sparse extra and CPU torch 2.15.0 are optional.
The multilingual OpenSearch model (Apache-2.0, 160M parameters, 684 MB) is pinned locally at
revision `1e0f096c2b51c234f1d20725c793e1b5b6d556db`; it is never downloaded implicitly.
Its query path is inference-free: `encode_query` uses the tokenizer and static `SparseStaticEmbedding`
weight table; only `encode_document` runs the transformer. Thus query encoding has no model compute.
Sparse vectors are stored per immutable chunk and encoder key in `chunk_sparse_vectors`.
Missing optional package or approved local model files makes R05 unavailable with a typed error.
There is no random, hash-based, or hand-built substitute for the learned representation.
