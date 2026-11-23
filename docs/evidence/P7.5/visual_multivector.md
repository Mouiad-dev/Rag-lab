# P7.5 R41 visual multivector retrieval: demo output and live evidence

Fidelity: ADAPTATION (ADR-0022): ColPali-style patch multivectors with MaxSim, local colSmol-256M, no token pooling, exhaustive scoring, vectors in memory only. A page is evidence only through original text chunks on that page; pages without text are scored and reported, never cited.

## Simulation (offline)

Command: `PYTHONIOENCODING=utf-8 uv run python docs/techniques/lessons/demos/r41_demo.py`. Date: 2026-11-24. Output is byte-identical between runs (two runs compared with `cmp`).

This section is SIMULATION. The page encoder is a test-only fake (patch vectors from a hash of the PNG bytes, one vector per query word from a hash of the word) and the R02 arm uses a test-only lexical hash embedder. Neither sees the chart or understands language, so the ranks show the mechanics (render, encode, MaxSim, text anchoring, ranking per page), not a model result, and nothing here is a quality or speed claim. Ranks are within the one 4-page fixture document.

```text
SIMULATION: fake page encoder, not a model result
Fixture: chart_report.pdf, 4 pages (0 text, 1 bar chart, 2 table, 3 no text layer), one text chunk per PDF block. R41 ranks all 4 pages by MaxSim; R02 (lexical test embedder) ranks the text chunks, shown per page (best chunk per page) and per chunk (c<ordinal>@p<page>). Both rankings are within this one document. MaxSim scores are sums over query tokens, so they are not on the cosine scale of R02.

## Index size
pages=4 patches=12 dim=4 estimated_bytes=192

## Relevant-page rank and score per question
| question | arm | result |
|---|---|---|
| q-chart | r41_visual | rank=3 score=6.781 top1_page=0 ranking(page:score)=p0:7.319 p2:7.046 p1:6.781 p3:6.669(no text) hit_pages(k=3)=[0, 2, 1] |
| q-chart | r02_text | rank=1 score=0.447 top1_page=1 page_ranking(best chunk per page)=p1:0.447 p0:0.126 p2:0.000 chunk_ranking=c1@p1:0.447 c0@p0:0.126 c2@p2:0.000 |
| q-table | r41_visual | rank=2 score=6.071 top1_page=0 ranking(page:score)=p0:6.372 p2:6.071 p1:5.761 p3:5.741(no text) hit_pages(k=3)=[0, 2, 1] |
| q-table | r02_text | rank=1 score=0.334 top1_page=2 page_ranking(best chunk per page)=p2:0.334 p0:0.268 p1:0.000 chunk_ranking=c2@p2:0.334 c0@p0:0.268 c1@p1:0.000 |
| q-text | r41_visual | rank=1 score=5.795 top1_page=0 ranking(page:score)=p0:5.795 p2:5.457 p1:5.286 p3:4.955(no text) hit_pages(k=3)=[0, 2, 1] |
| q-text | r02_text | rank=1 score=0.434 top1_page=0 page_ranking(best chunk per page)=p0:0.434 p1:0.000 p2:0.000 chunk_ranking=c0@p0:0.434 c1@p1:0.000 c2@p2:0.000 |
| q-unanswerable | r41_visual | top1=p2 score=6.285 top1_page=2 ranking(page:score)=p2:6.285 p0:6.158 p1:5.646 p3:5.498(no text) hit_pages(k=3)=[2, 0, 1] |
| q-unanswerable | r02_text | top1=p1 score=0.239 top1_page=1 page_ranking(best chunk per page)=p1:0.239 p0:0.201 p2:0.000 chunk_ranking=c1@p1:0.239 c0@p0:0.201 c2@p2:0.000 |

Pages scored: 4. Page 3 has no text chunk: it is scored and reported as (no text), never a hit.
q-unanswerable has no relevant page: only the top-1 is shown, and it still gets a top-1 in both arms; there is no abstention threshold here.
Losing case: R41 top1_page differs from the expected page for q-chart, q-table.
```

## Live run (orchestrator)

Model facts recorded in P7.5-09 (not output of this run): `vidore/colSmol-256M` commit c79b633e17e060cc109b11aa2bf92a1517d3bd6f (LoRA adapter, `adapter_model.safetensors` 39,135,840 bytes, sha256 41673fb85f448ff15356e4dfcf9e039d7c6e3cffa1936bc3513f55453005e63e; `1_Dense` safetensors 295,584 bytes, sha256 ce211361ebcef24a459e172f5683999ea6323b04fea0171c9caf6cd5e2cf4579); base `vidore/ColSmolVLM-Instruct-256M` commit 99ca96f1f6b95b3a69e6abef74a2416cb738fed0 (`model.safetensors` 456,405,944 bytes, sha256 45d026bad7c9d27edaeaf8ba4f2f0cd28406179819e06fa52693c051838c9069). Installed: torch 2.15.0+cpu, transformers 5.22.2, sentence-transformers 6.2.0, peft 0.22.0, Pillow 13.0.0. `colpali-engine` is not used (0.3.18 needs torch<2.14). R02 arm: bge-m3 on local Ollama.

<orchestrator>
One local CPU run, not an evaluation; no quality or speed claim. Timings are single wall-clock values on this machine and include the one-time model load.

- Command: `PYTHONIOENCODING=utf-8 RAG_LIVE_VISUAL=1 RAG_LIVE_OLLAMA=1 RAG_VISUAL_MODEL_DIR=var/models/colSmol-256M uv run pytest tests/live/test_page_encoder_live.py tests/live/test_visual_live.py -m live -v -s -p no:cacheprovider`
- Date: 2026-11-24. Result: 2 passed in 155.06s (coder run; log checked by the orchestrator).
- Models: R41 `vidore/colSmol-256M` at c79b633e17e060cc109b11aa2bf92a1517d3bd6f (LoRA adapter, `adapter_model.safetensors` sha256 41673fb85f448ff15356e4dfcf9e039d7c6e3cffa1936bc3513f55453005e63e) on base `vidore/ColSmolVLM-Instruct-256M` at 99ca96f1f6b95b3a69e6abef74a2416cb738fed0 (`model.safetensors` 456,405,944 bytes, sha256 45d026bad7c9d27edaeaf8ba4f2f0cd28406179819e06fa52693c051838c9069), loaded with peft 0.22.0 and sentence-transformers 6.2.0 (colpali-engine not used: it needs torch < 2.14). The load report lists the base `linear.weight`/`linear.bias` as unexpected; they are byte-identical to colSmol's `1_Dense` weights, which are applied. R02 uses local `bge-m3` on Ollama.
- Machine: AMD Ryzen 7 5800H, 15.3 GB RAM, CPU only; torch 2.15.0+cpu, transformers 5.22.2, Python 3.13.5. Page encoder: 1135 patches per page; 19.0 s per page in `test_page_encoder_live` and 15.1 s per page in the R41 build (each includes its own one-time model load); RSS at the end 1563 MB (not a true peak).
- Ranks are within this one 4-page document (R02 ranks its 3 text chunks, pages 0 to 2). R41 scores are MaxSim sums over query tokens; R02 scores are cosine: the scales differ.
- Both arms put the expected page at rank 1 for all three answerable questions, so this run shows no ranking loss for R41 or R02 on this tiny fixture. The losing case in this run is q-unanswerable: both arms still return a top-1 (page 1) because there is no abstention threshold. The text-less page 3 is scored by R41 and ranks above text page 2 for q-chart and q-text, but it is never a hit (it has no text chunk to cite). The offline simulation shows R41 ranking losses (q-chart, q-table) with the fake encoder; those are simulation results, not model results.

Printed output (progress bars and load tables removed):

```text
revision=c79b633e17e060cc109b11aa2bf92a1517d3bd6f dim=128 patches_per_page=[1135, 1135, 1135, 1135] query_tokens=[16, 18] seconds_per_page=19.0 peak_rss=1563 MB (current RSS at end)
PASSED
model=vidore/colSmol-256M@c79b633e17e060cc109b11aa2bf92a1517d3bd6f revision=c79b633e17e060cc109b11aa2bf92a1517d3bd6f dim=128 pages=4 patches=4540 estimated_bytes=2324480
build_s_total=60.5 build_s_per_page=15.1 (includes the one-time model load)
Rankings are within this one 4-page document. R41 scores are MaxSim sums over query tokens; R02 scores are cosine: the scales differ. Page 3 has no text and can never be a hit.
R41 q-chart expected=1 rank=1 score=20.272 top1_page=1 ranking(page:score)=p1:20.272 p0:14.888 p3:9.655(no text) p2:9.524 hit_pages(k=3)=[1, 0, 2] query_s=1.14
R02 q-chart expected=1 rank=1 score=0.687 top1_page=1 page_ranking(best chunk per page)=p1:0.687 p0:0.450 p2:0.239 chunk_ranking=c1@p1:0.687 c0@p0:0.450 c2@p2:0.239
R41 q-table expected=2 rank=1 score=18.022 top1_page=2 ranking(page:score)=p2:18.022 p1:9.881 p0:8.735 p3:6.131(no text) hit_pages(k=3)=[2, 1, 0] query_s=0.91
R02 q-table expected=2 rank=1 score=0.723 top1_page=2 page_ranking(best chunk per page)=p2:0.723 p1:0.290 p0:0.284 chunk_ranking=c2@p2:0.723 c1@p1:0.290 c0@p0:0.284
R41 q-text expected=0 rank=1 score=17.358 top1_page=0 ranking(page:score)=p0:17.358 p1:13.514 p3:9.984(no text) p2:8.713 hit_pages(k=3)=[0, 1, 2] query_s=0.91
R02 q-text expected=0 rank=1 score=0.629 top1_page=0 page_ranking(best chunk per page)=p0:0.629 p1:0.420 p2:0.184 chunk_ranking=c0@p0:0.629 c1@p1:0.420 c2@p2:0.184
R41 q-unanswerable expected=None top1=p1 score=15.094 top1_page=1 ranking(page:score)=p1:15.094 p0:10.708 p2:10.195 p3:8.872(no text) hit_pages(k=3)=[1, 0, 2] query_s=1.10
R02 q-unanswerable expected=None top1=p1 score=0.459 top1_page=1 page_ranking(best chunk per page)=p1:0.459 p0:0.314 p2:0.285 chunk_ranking=c1@p1:0.459 c0@p0:0.314 c2@p2:0.285
q-unanswerable has no relevant page: its top-1 is shown and there is no abstention threshold, so both arms still return a top-1.
PASSED
```
</orchestrator>

## R40 blocked record (text-table-image RAG)

- Status: blocked.
- Decision: owner declined D4 on 2026-11-23 (ADR-0022).
- Requirements (copied from ADR-0022, D4): image parts in the provider `ChatMessage` schema with size and format limits; a vision chat path in an adapter (`gemma4:e4b` lists vision in `ollama show`, so no pull is needed); synthetic captions kept out of evidence; page-image rendering (exists after P7.5-07).
- Not implemented: image parts in `ChatMessage` and a vision chat path. R41 returns text chunks as evidence and does not send page images to a chat model. Nothing here is R40.
- Catalogue: the entry stays, status `blocked`; it is not removed.
