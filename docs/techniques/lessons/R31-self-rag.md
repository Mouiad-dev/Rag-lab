# R31: Self-RAG inference (short-form)

Fidelity: ADAPTATION (ADR-0022); never `validated`. Source: arXiv 2310.11511, "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection". Prompt self-checking is not Self-RAG: this entry needs a model trained to emit reflection tokens, and only that model is used.

## What it is
The model itself emits reflection tokens (`[Retrieval]`, `[Relevant]`, `[Fully supported]`, `[Utility:5]` and others). Retrieval is decided by the model, each retrieved passage is critiqued, and the best-scored candidate answer is returned.

## How it works here
- Code: `packages/answering/src/rag_foundry_answering/self_rag.py` (`SelfRagWorkflow`), on the `RawCompletionModel` port (raw prompt, temperature 0, top_p 1, top_k 0, token logprobs). Retriever: R02 dense (bge-m3 on Ollama), k 5; evidence prompt title is empty (no title field).
- Retrieval decision: a separate 1-token call at position 0. The rule is normalized p([Retrieval]) / (p([Retrieval]) + p([No Retrieval])) above 0.2. Deviation: upstream divides raw logprobs. Fallback `undetermined_topk` (generated-token rule) when both tokens are outside the returned alternatives.
- Reflection tokens are matched by token id (32000-32014) when the runtime returns ids, else by text. A token outside the returned alternatives gets the official -100 logprob.
- Per candidate: total = w_rel*relevance + w_sup*ground + w_use*utility + exp(mean token logprob); ground = p(Fully) + 0.5*p(Partially); utility = sum of [-1, -0.5, 0, 0.5, 1] times normalized p(Utility:1..5); all weights 1.0. The best total is chosen (official selection).
- Citation guard (owner decision 2026-11-24, citation only, selection unchanged): the chosen chunk is cited only if the model generated `[Relevant]` at position 0 and a `[Fully supported]` or `[Partially supported]` token; otherwise the answer is returned ungrounded with no citation.

## Fidelity and deviations
Quantized (Q4_K_M) unofficial GGUF conversion; our retriever instead of the paper's; short-form only; top-k floor and normalized threshold as above; label adaptation. Runtime matters: the method passes only on llama.cpp (full-vocabulary logprobs with ids). On Ollama (top 20 only) it fails rule (d) of `docs/evidence/P7.5/r31_gguf_check.md`: 7 scored positions, 4 missing their group's tokens, share 0.57 > 0.20. R31 is `implemented` because the llama.cpp run passed rules (d) and (e), the ADR-0022 owner exception.

## Runtimes, model and hardware
- Model: `selfrag_llama2_7b-q4_k_m.gguf` (Ribben7 repo commit 2253cc23d265343a963cc84e20d698e7e89ed75b), 4,081,095,936 bytes, sha256 9b05791653cf30a4237040a3c1c7d42f41334c90e74989978445c10e5143abd7. Licence: stated MIT on the repo; the Llama 2 terms of the base model are expected to apply too (owner to confirm before redistribution).
- llama.cpp server b12220 (commit 4b46d965385c9b3b0e912f4fde73e57cd1751d0d), Vulkan, `-ngl 99`, context 2048, bound to 127.0.0.1. Archive `llama-b12220-bin-win-vulkan-x64.zip` sha256 5df033b72421e4c423438f3f5007bb0e439ad0c3e9d364c5c6b3407bf94c1e45. The extracted exe hash was not recorded.
- Ollama 0.40.17 for the Ollama run and for bge-m3 embeddings.
- Hardware: Windows 11, RTX 3060 Laptop 6 GB; nvidia-smi 5763 MiB used with both models loaded.

## Live runs (2026-11-24, 4 questions, n=4, not an evaluation)
Evidence: `docs/evidence/P7.5/self_rag_live.md`. llama.cpp: 1 passed in 58.51s (49.60s after review fixes, identical); 11 scored positions, 0 missing; decisions all by probability threshold with 32016 alternatives and id matching; p_ret 0.9931 (retrieval), 0.0100 (greeting), 0.9074 (unanswerable), 0.3075 (Arabic). The retrieval question answered "06:00 and 18:00 daily" with candidate scores relevance 0.999, support 0.978, utility 0.998, sequence 0.963, total 3.938. Pre-sampler check equal x4. Citation guard rerun: 1 passed in 109.54s, same totals; retrieval question cited, unanswerable and Arabic answers uncited.

## Success
The retrieval question is retrieved, answered from the passage, cited, and every reflection group is read from the full vocabulary.

## Losing cases
- Arabic: on Ollama the answer was unrelated Arabic text about creating an account; on llama.cpp it was an English "not sure what you are asking" reply (p_ret 0.3075, chosen candidate `[Irrelevant]`).
- Unanswerable question: retrieval was triggered (p_ret 0.9074) and the candidate was labelled `[Irrelevant]` with no support token (total 1.342); before the guard the answer cited the only chunk although the model rejected it.
- One-chunk fixture: only one candidate per question, so ranking among candidates was not exercised live (unit tests cover a 3-hit hand-computed choice).
- Runtimes disagree for the same GGUF and prompt: "The" logprob -1.72 on llama.cpp vs -1.01 on Ollama (cause not established). On Ollama, cold vs warm prompt cache differs by up to 0.067 logprob, so scores are not bit-reproducible there.

## Cost per stage
Per question: one 1-token decision call, then if retrieving one evidence generation per passage (up to 5), plus embedding of the query. Live llama.cpp latencies per question: 15.50, 1.30, 11.31, 3.59 s (n=1 each; the guard rerun ran slower while other work was on the machine). Cost 0 (local).

## Open items
Full-precision checkpoint; long-form beam search; a multi-chunk live fixture; the Ollama path (needs more than 20 alternatives).

## Status
Implemented (adaptation); not validated or benchmarked.
