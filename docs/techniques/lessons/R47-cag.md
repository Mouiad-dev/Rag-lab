# R47: cache-augmented generation (CAG)

## What it is
CAG puts a whole bounded knowledge set into the prompt prefix once and asks many questions after it, instead of retrieving a few chunks per question. R47 is a `workflow` plugin (`rag_foundry_answering.cag`). It is a different mechanism from R49 (prefix cache probe) and from R50/R51 (response caches).

## How it works here
- Snapshot: all chunks of one ready, in-scope index version, in a fixed order, labelled S1..Sn. Nothing is selected or truncated. Bounds: context window, 1.25 estimate margin, reserved output (512) and question (64) tokens, and an evidence character cap. A corpus that does not fit is refused at build time with `cag_overflow` (before any chat call).
- Nonce (owner approved 2026-11-22): one stable per-snapshot nonce keeps the prefix byte-identical across questions. It is drawn with `secrets.token_hex(8)`, redrawn if it occurs in any chunk, never logged, dumped, persisted or traced, checked in every question, and an answer echoing it abstains. A rebuilt snapshot draws a new nonce. Residual risks: the provider sees the nonce (for example in Ollama debug logs), and runtime KV caches are not tenant-partitioned.
- Arms run as blocks: all `cag`, then all `full_context_uncached`, then R02 with k=3. The uncached arm uses a fresh nonce per call, but it still shares the text before the first nonce with the other arms.
- `RunPins.k` is not meaningful for R47: every item is sent; the trace holds the real item count.

## Fidelity: adaptation
Follows the idea of Chan et al. 2024, "Don't Do RAG: When Cache-Augmented Generation is All You Need for Knowledge Tasks" (arXiv 2412.15605, title checked on arxiv.org 2026-11-23; repo hhhuang/CAG is not linked from the abstract page, unverified). Not reproduced: no KV precompute, save or truncate (the paper's Hugging Face `DynamicCache` path needs torch and a model download). Here prefix reuse is whatever the runtime does between requests, which Ollama does not document. Size checks use the lab-regex estimate, not a model tokenizer.

## What a hit or a win does not prove
Cache reuse does not remove generation: every question still decodes its answer. A cached count shows prefill reuse in one run, not answer quality, and not that the runtime keeps it for another tenant or later.

## Success example (offline demo, scripted answers)
19 chunks, estimate 1022 tokens (without question), prefix identical across questions for `cag`, not for the other arms. Answers cite markers (S1, S2). See `docs/evidence/P7.4/cag_demo.md`.

## Losing examples
- Overflow: at a 300-token window R47 refuses (`cag_overflow`, prompt_tokens=1870/300) while R02 k=3 answers and cites S1. An evidence cap of 500 chars refuses at 1730/500.
- Update: adding one paragraph changed the snapshot identity (evidence chars 1730 to 1786) and the prefix. An update means a rebuild plus a cold prefill of the whole prefix.
- Live (one run): the lab estimate (1080 tokens, 1350 with margin) was below the largest real prompt (1568, uncached AR). Far below `num_ctx` 8192, but the margin would not catch a tight fit.

## Cost per stage
Offline: one chat call per question in the CAG and uncached arms, two (query embedding plus chat) in R02; token figures there are scripted estimates. Live input tokens (provider counts): `cag` 1368, 1366, 1363; uncached 1568, 1566, 1443; R02 408, 425, 421. Output tokens 8 to 43.

## Cold vs warm (live, one run, n=1, not an evaluation)
Local Ollama 0.40.17, `gemma4:e4b`, `bge-m3`. The first `cag` call (prime) cached 0 of 1368 tokens; later `cag` calls cached 1367 (EN), 1334 (AR), 1334 (out of corpus). Uncached and R02 calls cached 0. Prefill_ms of the `cag` calls were 491 (prime), 37, 119, 109; uncached 565, 565, 487. One run on one machine, with a single prime: no speed claim. Ollama's `prompt_eval_count` already includes cached tokens (warm EN: input 1368, cached 1367).

## Open items
KV preload with Hugging Face (torch); API/UI wiring; memory and residency; a model tokenizer for the estimate; per-tenant isolation of runtime KV caches.
