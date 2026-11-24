# R31 Self-RAG short-form live evidence

Status: one local live run on 2026-11-24. Result: **R31 blocked** by rule (d) from
[r31_gguf_check.md](r31_gguf_check.md), section 6. Execution evidence only, not a quality or
speed evaluation.

Implementation defaults follow the README runs documented in `r31_gguf_check.md`, section 2:
threshold 0.2, relevance/support/utility weights 1.0, sequence score on, k (ndocs) 5,
max_new_tokens 100. The retrieval threshold uses normalized probabilities (a documented
adaptation; upstream divides raw log probabilities). Reflection tokens outside Ollama's top 20
get the official -100 logprob fallback.

## Live run

- Date and environment: 2026-11-24, Windows 11, RTX 3060 Laptop 6 GB; `ollama ps`: 5.3 GB,
  20%/80% CPU/GPU, context 2048.
- Ollama: 0.40.17.
- Model: `selfrag-llama2-7b:q4_k_m` from `selfrag_llama2_7b-q4_k_m.gguf` (Ribben7, sha256
  `9b05791653cf30a4237040a3c1c7d42f41334c90e74989978445c10e5143abd7`, P7.5-14).
- Embeddings: `bge-m3` on Ollama (R02 dense retriever, k 5); the fixture is one short English
  document that yields one chunk, so only one candidate was scored and candidate ranking was
  not exercised.
- Command:

```bash
PYTHONIOENCODING=utf-8 RAG_LIVE_OLLAMA=1 RAG_SELFRAG_MODEL=selfrag-llama2-7b:q4_k_m uv run pytest tests/live/test_self_rag_live.py -m live -v -s -p no:cacheprovider
```

  -> 1 passed in 57.08s (run 3; see "Sampler check" for why runs 1 and 2 were replaced). Run 4,
  after review round 1 fixes (position 0 always scored, `###` lines neutralised): 1 passed in
  60.01s with identical decisions, answers, candidate scores, totals (7 / 4 / 0.57) and
  `pre_sampler_check` equal x4; latencies 28.4, 0.69, 1.68, 4.02 s. The table shows run 3.

| Question | Position-0 top 5 | Retrieval rule | Retrieve | Answer (trimmed) | Citation | Latency s | Output tokens |
|---|---|---|---|---|---|---|---|
| retrieval: "At what times does the Northbridge observatory record the tide?" | The, I, Hi, N, S | undetermined_topk_generated_token | yes (`[Retrieval]` generated mid-text) | "The Northbridge Observatory records the tide at 06:00 and 18:00 daily." | the one fixture chunk | 28.0 | 129 |
| no_retrieval: "Write a friendly one-sentence greeting." | Hello, Hi, ", H, Good | undetermined_topk_generated_token | no (`[No Retrieval]` generated) | "Hello!How are you?" (ungrounded) | none | 0.61 | 9 |
| unanswerable: "What is the observatory director's favorite food?" | I, That, As, It, Hi | undetermined_topk_generated_token | no | "I'm sorry, but I don't have enough information ..." (ungrounded) | none | 1.32 | 31 |
| arabic_losing_case (Arabic tide question) | Arabic letters, 1, S, Thank | undetermined_topk_generated_token | no | unrelated Arabic text about creating an account (ungrounded; `language_outside_model` true) | none | 3.73 | 100 |

Chosen candidate (retrieval question): relevance 0.9989, support 0.9806 (support scored),
utility 0.9987 (utility scored), sequence 0.9543, total 3.9325.

- Reflection token strings printed literally: yes (`[No Retrieval]`, `[Retrieval]`,
  `<paragraph>`, `[Utility:4]`, `[Utility:5]`, `[Relevant]`, support tokens seen as whole tokens).
- Scored positions: 7. Positions with none of their group's tokens in the top 20: 4 (share 0.57).
  All 4 are the retrieval decision at position 0 of the no-retrieval generation; the 3
  relevance/support/utility positions had their group's tokens in the top 20.
- `pre_sampler_check`: equal for all 4 questions (max abs diff 0).

## Sampler check

Runs 1 and 2 printed `pre_sampler_check=different` (max abs diff 0.03-0.09, top-20 order
changed). A standalone probe (scratch script, same model, prompt
"At what times does the Northbridge observatory record the tide?") showed the cause is Ollama's
prompt cache, not the sampler: cold vs warm call with the same `top_k` differ by 0.0667; `top_k`
0 vs 1 in the same cache state are identical (diff 0.0, same order). The live test now warms the
cache before comparing. Consequence: logprobs (and so Self-RAG scores) are not bit-reproducible
across cache state; differences up to about 0.07 logprob were seen.

## Decision

R31 is **blocked** (rule (d): 57% > 20% of scored positions lack their group's tokens in the
top 20). The retrieval decision is the failing part: this model rarely starts its answer with a
retrieval token, and the official code reads `[Retrieval]`/`[No Retrieval]` at position 0 from a
full-vocabulary distribution (vLLM `logprobs=32016`). Ollama returns at most 20 alternatives, so
the probability threshold never applied; every decision fell back to the generated-token rule,
which is the official no-threshold mode, not the threshold mode the defaults describe.
Rules (literal tokens) and (e) (pre-sampler) passed.

```text
Date/environment: 2026-11-24, Windows 11, RTX 3060 Laptop 6 GB, Ollama 0.40.17
Command: see "Live run"
Model: selfrag-llama2-7b:q4_k_m (Ribben7 Q4_K_M GGUF, sha256 9b0579...43abd7)
Observed condition: [Retrieval]/[No Retrieval] absent from the position-0 top 20 for 4/4 questions
Evidence/output: scored_positions=7 positions_missing_group_tokens=4 share=0.57
Blocked rule: >20% missing group tokens
R31 status: blocked
Next action: owner decision. Option: llama.cpp server with a larger n_probs (returns token ids;
binary download needs owner OK, plus a new adapter spec), then rerun this test unchanged.
```
