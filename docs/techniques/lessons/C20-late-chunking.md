# C20: Late chunking

Fidelity: ADAPTATION (ADR-0022). Source: arXiv 2409.04701, "Late Chunking: Contextual Chunk Embeddings Using Long-Context Embedding Models".

## What it is
Late chunking encodes a whole text once at token level and only then pools the token vectors inside each chunk span, so every chunk vector has seen the text around it. Per-chunk embedding (each chunk encoded alone) is the early baseline; it is never called late chunking.

## How it works here
- Code: `packages/indexing/src/rag_foundry_indexing/late_chunking.py`, registered as the built-in C20 plugin. Token vectors come from the `TokenEncoder` port (`[tokens]` extra, local files only).
- Late mode: one encode per window of the extraction view (macro windows if the view is long, never one per chunk), then mean pooling of the tokens of each chunk span. Windows own disjoint ranges, so overlap is pooled once.
- Early mode: same model, same pooling, each chunk encoded alone.
- Search is in-memory cosine, bound to one tenant and project; another scope gets NotFound before any encoder call. Vectors are not persisted.
- Model: `BAAI/bge-m3` at commit 5617a9f61b028005a4858fdac845db406aefb181, converted locally to safetensors (hashes in the evidence file). It is trained with a CLS head; mean pooling of its hidden states is therefore an adaptation of the paper's setting.

## Fidelity
Not the paper's model (the paper uses a long-context embedding model trained for it); no jina models. Token windows, pooling and the early baseline follow the idea; the retrieval quality of this adaptation is not established.

## Runnable demo
```bash
PYTHONIOENCODING=utf-8 uv run python docs/techniques/lessons/demos/c20_r45_demo.py
```
Output and live run: [evidence](../../evidence/P7.5/late_chunking_and_late_interaction.md). The demo uses a test-only fake encoder.

## Simulation (fake encoder)
The demo prints ranks for 3 documents (EN, AR, mixed) with a fake encoder that has no language understanding. It shows the mechanics (early vs late vectors differ, the losing distractor case), not a model result.

## Live (one local CPU run, 2026-11-23, not an evaluation)
- bge-m3, AMD Ryzen 7 5800H, CPU only, torch 2.15.0+cpu. Each document fit in one window; no truncation; per-chunk token counts equal in early and late mode.
- Rank of the relevant chunk among the 3 or 4 chunks of its own document, 5 answerable questions (n=5): early 2, 3, 3, 2, 1 (EN1, EN2, AR1, MIX1, MIX2); late 2, 3, 2, 2, 1. Only AR1 changed (3 to 2). This is one tiny fixture; no quality claim.
- Late cosine scores are higher than early for the answerable questions, but the unanswerable question also scores about 0.63 in both modes, so absolute cosine is not a quality signal.

## Success
The early and late vectors differ for the same chunk, spans stay aligned to the source offsets, and the scope check holds. One question (AR1) moved up one rank in the live run.

## Losing case
EN2: the relevant chunk (2) is rank 3 in early and late mode; keyword-heavy distractor chunk 3 is top-1 in both. Late chunking does not fix it. The unanswerable question still gets a top-1 chunk (there is no abstention threshold).

## Cost per stage
Build: late mode runs one encode per window; early mode runs one encode per chunk. Live build times (single wall-clock values, first early build includes the model load): EN early 12.92 s, late 0.23 s; AR 0.36 and 0.16 s; MIXED 0.38 and 0.16 s. These are n=1 timings and are not a speed comparison. Query: one encode, about 0.11 to 0.15 s.

## Hardware
CPU only, bge-m3 dimension 1024. Memory is the in-memory token vectors (estimate for the multivector arm: EN 286,720 bytes for 70 tokens); mean-pooled chunk vectors are 1024 floats per chunk.

## Open items
Vector persistence; a long-context model trained for late chunking; documents longer than one window; evaluation on a real corpus.

## Status
Implemented (adaptation); not validated or benchmarked.
