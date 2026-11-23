# ADR-0022: Research and visual tracks (C20, R45, R41, R31; R40 and R46 blocked)

Date: 2026-11-23
Status: Accepted (owner, 2026-11-23)

## Context
P7.5 covers C20 (late chunking), R45 (text late interaction), R41 (visual multivector), R31 (Self-RAG), R40 (text-table-image RAG) and R46 (CRAG). Several need local model weights and have paper-fidelity limits. This ADR fixes labels, decisions and the exact `blocked` requirements before code lands. Owner decisions D1-D7 are recorded here.

## Decision
### Labels and fakes
- Fidelity labels: `faithful`, `adaptation`, `simulation`. Every P7.5 evidence file and lesson states one.
- Technique fakes (test encoders, scripted models) live only in `tests/` and are never registered. The provider fake (`rag_foundry_providers/fake.py`) is unchanged.
- `completion_policy`: adaptations do not complete research entries (R31, R46). Owner exception, 2026-11-23: R31 may be `implemented` as an adaptation if its live run (P7.5-15) passes; it is never `validated`; if the run fails, R31 is `blocked`. R46 stays `blocked`.

### Local models and scope
- Models are local only: `local_files_only=True`, directories from `RAG_*_MODEL_DIR`, never downloaded at run time. Weights load only from safetensors (no unpickling). Revision and sha256 are recorded in evidence.
- Missing extra or files raise `TechniqueUnavailableError`; there is no fallback.
- Tenant scope: every in-memory index (C20, R45, R41) stores the tenant and project of its chunk records and refuses any other caller scope with NotFound.
- D7 (hardware, checked 2026-11-23): NVIDIA RTX 3060 Laptop, 6 GB VRAM, 15.3 GB RAM, 51 GB free disk. Torch stays the CPU build; GPU is used only through the Ollama runtime (R31).

### D1/D2: C20 and R45 (approved)
- Model: BAAI/bge-m3 (MIT) at commit 5617a9f61b028005a4858fdac845db406aefb181. That commit has only `pytorch_model.bin` and `colbert_linear.pt`, so the orchestrator converts them locally to safetensors (weights_only load) and records all sha256 values. One download serves C20 and R45. No jina models.
- C20: token-level encoding of the view (macro windows if long), then mean pooling per chunk span. The early baseline uses the same model. A CLS-trained model with mean pooling is an adaptation.
- R45: MaxSim on bge-m3 ColBERT vectors is an adaptation. No PLAID. The port drops special tokens, while official BGE-M3 ColBERT keeps the final `</s>` from `last_hidden_state[:, 1:]`.
- Per-chunk embedding is never called late chunking.

### D3: R41 (approved)
- Only vidore/colSmol-256M (about 0.5 GB). `colpali-engine` and Pillow sit behind the `[visual]` extra; CPU only. Any further checkpoint needs owner OK first.
- Page hits are evidence only through text chunks on that page, after the rendered PDF's sha256 matches the scoped document version. Pages without text are reported, never cited.

### D4: R40 declined, `blocked`
Requirements: image parts in the provider `ChatMessage` schema with size and format limits; a vision chat path in an adapter (`gemma4:e4b` lists vision in `ollama show`, so no pull is needed); synthetic captions kept out of evidence; page-image rendering (exists after P7.5-07).

### D5: R31 explored
- Source: a community quantized GGUF of `selfrag_llama2_7b` (Llama 2 Community Licence, about 4 GB, must fit 6 GB VRAM) on Ollama or llama.cpp, gated by P7.5-13. The P7.5-13 header read and the GGUF download are shown to the owner (source and bytes) before they run.
- Label is always `adaptation`: quantized, unofficial conversion, top-k-only logprobs, our retriever.
- Missing single added reflection tokens (`[Retrieve]`, `[IsRel]`, `[IsSup]`, `[IsUse]` families) or token logprobs: `blocked` with the exact gap. Prompt self-checking is never Self-RAG.

### D6: R46 `blocked`
Requirements: the authors' fine-tuned T5-large retrieval evaluator (link and licence from the official repo); knowledge refinement (decompose, filter, recompose); web search through an approved, budgeted provider in enriched mode. A reranker or LLM grader is R29 (P8.1), not R46.

### Error codes (registered once)
- `IndexingError.CODES`: `token_encoding_count_mismatch`, `token_offsets_invalid`, `late_chunk_span_empty`, `late_chunk_window_invalid`, `multivector_index_missing`, `multivector_dim_mismatch`, `visual_index_missing`, `multivector_count_mismatch`, `multivector_page_empty`.
- `ExtractionError.CODES`: `page_render_limit`, `page_image_invalid` (ingestion rendering and R41 provenance checks on source bytes).
- `RunError.CODES`: `selfrag_tokens_missing`, `selfrag_logprobs_unavailable`, `selfrag_output_invalid`.

## Consequences
No measured claims are made here. Evidence files (P7.5-12 and the live-run specs) carry the numbers. The catalogue keeps all entries; R40 and R46 are `blocked`, not removed.
