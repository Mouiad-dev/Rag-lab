# R45: Text late-interaction retrieval

Fidelity: ADAPTATION (ADR-0022). Source: arXiv 2004.12832, "ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT".

## What it is
Instead of one vector per chunk, keep one vector per token and score a query against a chunk with MaxSim: for each query token take its best match among the chunk's tokens, then sum over query tokens. R45 is not C20 (C20 pools tokens into one chunk vector) and not a single-vector retriever.

## How it works here
- Code: `packages/indexing/src/rag_foundry_indexing/late_interaction.py`, registered as the built-in R45 plugin. Token vectors come from the `TokenEncoder` port with the ColBERT head of `BAAI/bge-m3`.
- Index: in-memory, L2-normalised token vectors, bound to one tenant and project (other scopes get NotFound). Not persisted.
- Search: exhaustive MaxSim over all chunks (no PLAID, no compression). Each hit carries up to 5 query-token to document-token matches. Memory size is reported as an estimate.
- Same model: bge-m3 at commit 5617a9f61b028005a4858fdac845db406aefb181 (original and converted file hashes in the evidence file). The comparison arm is the mean-pooled single vector from the same encoder.

## Fidelity
The scorer is ColBERT-style, but the checkpoint is bge-m3's ColBERT head, not the original ColBERT model. The port drops special tokens, while official BGE-M3 ColBERT keeps the final `</s>`. No PLAID, no pruning. MaxSim scores are sums over query tokens, so they are not on the cosine scale.

## Runnable demo
```bash
PYTHONIOENCODING=utf-8 uv run python docs/techniques/lessons/demos/c20_r45_demo.py
```
Output and live run: [evidence](../../evidence/P7.5/late_chunking_and_late_interaction.md).

## Simulation (fake encoder)
The offline demo uses a test-only fake encoder; its ranks (for example AR1 MaxSim rank 1 against single-vector rank 2) show mechanics only, not a model result.

## Live (one local CPU run, 2026-11-23, not an evaluation)
Rank of the relevant chunk within its own 3 or 4 chunk document, 5 answerable questions (n=5), order EN1, EN2, AR1, MIX1, MIX2: single-vector 2, 3, 2, 1, 1; MaxSim 2, 3, 2, 2, 2. They differ on MIX1 and MIX2 only, where single-vector ranks the relevant chunk first and MaxSim ranks it second. This tiny fixture does not show an advantage for either arm.

## Success
Token-level matches are inspectable (query token to document token), so a score can be explained. On AR1 both arms rank the relevant chunk second in a 3-chunk document.

## Losing case
EN2: the relevant chunk (2) is rank 3 in both arms and the keyword-heavy distractor chunk 3 is top-1. MIX1 and MIX2 show MaxSim ranking the relevant chunk below the single-vector arm. The unanswerable question still gets a top-1 chunk; there is no abstention threshold.

## Cost per stage
Index: one token-level encode per document view; the stored multivector arm keeps one 1024-float vector per token (live estimate: EN 70 tokens, 286,720 bytes; AR 29 tokens, 118,784 bytes; MIXED 36 tokens, 147,456 bytes). The single-vector arm keeps one vector per chunk. Live multivector build: 0.54, 0.33, 0.37 s (single wall-clock values). Query: one encode plus MaxSim over every chunk, 0.12 to 0.23 s.

## Hardware
AMD Ryzen 7 5800H, CPU only, torch 2.15.0+cpu, transformers 5.22.2, Python 3.13.5.

## Open items
Persistence of token vectors; PLAID or compression for larger corpora; keeping the final `</s>` token as official BGE-M3 does; an evaluation on a real corpus.

## Status
Implemented (adaptation); not validated or benchmarked.
