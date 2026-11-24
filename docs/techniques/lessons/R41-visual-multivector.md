# R41: Visual multivector retrieval

Fidelity: ADAPTATION (ADR-0022). Source: arXiv 2407.01449, "ColPali: Efficient Document Retrieval with Vision Language Models".

## What it is
Render each PDF page to an image, encode it into many patch vectors, encode the query into token vectors, and score pages with MaxSim. This can rank a page whose answer is in a chart or table image that has no text layer. R41 is not R40: R41 never sends page images to a chat model.

## How it works here
- Code: `packages/indexing/src/rag_foundry_indexing/visual.py`, built-in R41 plugin; `PageImageEncoder` port and `ColPaliPageEncoder` (`[visual]` extra, local files only); page rendering in `packages/ingestion/src/rag_foundry_ingestion/page_images.py` (bounded, PDF to PNG).
- Before any rendering the build checks that the PDF bytes match the scoped document version (sha256), that the version is in a READY index, and that re-extraction reproduces the view hash.
- Search: exhaustive MaxSim over all page patch vectors; no token pooling; vectors in memory only; bound to one tenant and project.
- Evidence: a page is evidence only through the original text chunks on that page. Pages without text are scored and reported, never cited.
- Model: `vidore/colSmol-256M` commit c79b633e17e060cc109b11aa2bf92a1517d3bd6f (LoRA adapter, sha256 41673fb85f448ff15356e4dfcf9e039d7c6e3cffa1936bc3513f55453005e63e) on base `vidore/ColSmolVLM-Instruct-256M` commit 99ca96f1f6b95b3a69e6abef74a2416cb738fed0 (sha256 45d026bad7c9d27edaeaf8ba4f2f0cd28406179819e06fa52693c051838c9069), loaded with peft 0.22.0 and sentence-transformers 6.2.0 (`colpali-engine` is not used: it needs torch < 2.14).

## Fidelity
A small checkpoint (about 0.5 GB), not ColPali itself; exhaustive scoring, no pooling, nothing persisted. Page hits become citations only through text anchors.

## Runnable demo
```bash
PYTHONIOENCODING=utf-8 uv run python docs/techniques/lessons/demos/r41_demo.py
```
Output and live run: [evidence](../../evidence/P7.5/visual_multivector.md).

## Simulation (fake encoder)
The offline demo uses a fake page encoder (hash-based patch vectors). It shows R41 ranking the expected page below top-1 for q-chart and q-table. These are simulation results, not model results.

## Live (one local CPU run, 2026-11-24, not an evaluation)
Fixture `chart_report.pdf`, 4 pages (0 text, 1 bar chart, 2 table, 3 no text layer), 3 answerable questions (n=3) plus one unanswerable. R41 and the R02 text arm (bge-m3 on Ollama) both rank the expected page first for all 3 questions. This tiny fixture shows no ranking loss and no advantage for R41.

## Success
Question-dependent real scores: for q-chart R41 scores page 1 at 20.272 against 14.888 for the next page. Provenance holds: only pages with text chunks can be cited.

## Losing case
q-unanswerable: both arms still return a top-1 (page 1), because there is no abstention threshold. The text-less page 3 outranks text page 2 for q-chart and q-text but can never be a hit.

## Cost per stage
Render and encode: 1135 patches per page, dimension 128; 19.0 s per page in the encoder test and 15.1 s per page in the R41 build (each includes its own one-time model load, n=1). Index: 4 pages, 4540 patches, estimated 2,324,480 bytes. Query: 0.91 to 1.14 s for MaxSim over the 4 pages. R02 memory and latency were not measured, so no comparison is made.

## Hardware
AMD Ryzen 7 5800H, 15.3 GB RAM, CPU only; torch 2.15.0+cpu, transformers 5.22.2, Pillow 13.0.0; RSS at the end 1563 MB (not a true peak).

## Open items
Persistence; PLAID or compression; region geometry (page hits do not locate a region); Arabic chart pages; OCR; loading PDF bytes from the object store (the caller passes them in); R40 vision path (blocked, see R40).

## Status
Implemented (adaptation); not validated or benchmarked.
