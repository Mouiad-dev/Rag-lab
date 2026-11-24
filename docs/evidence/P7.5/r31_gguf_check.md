# R31 check before coding: GGUF source, reflection tokens, raw-prompt logprobs

Status: filled by the orchestrator on 2026-11-24 (all network data retrieved 2026-11-24).
Sizes and sha256 values come from the Hugging Face API (`/api/models/<repo>?blobs=true`,
`lfs.sha256`), not from memory. Licence names are what each repo states; this is not legal advice.

## 1. Official reflection tokens and ids

Source: `added_tokens.json` of `selfrag/selfrag_llama2_7b` at commit
`190261383b0779ff66d2f95a73c7ad267d94b820`
(https://huggingface.co/selfrag/selfrag_llama2_7b/resolve/190261383b0779ff66d2f95a73c7ad267d94b820/added_tokens.json),
retrieved 2026-11-24. `config.json` at the same commit: `vocab_size` 32016.
Token names used by the inference code: `retrieval_lm/utils.py` lines 43-53 at
AkariAsai/self-rag commit `1fcdc420e48f50a7d7ab1ece5494221b93252e99`.

| Token | Original id |
|---|---|
| `[No Retrieval]` | 32000 |
| `[Retrieval]` | 32001 |
| `[Continue to Use Evidence]` | 32002 |
| `[Irrelevant]` | 32003 |
| `[Relevant]` | 32004 |
| `<paragraph>` | 32005 |
| `</paragraph>` | 32006 |
| `[Utility:1]` .. `[Utility:5]` | 32007 .. 32011 |
| `[Fully supported]` | 32012 |
| `[Partially supported]` | 32013 |
| `[No support / Contradictory]` | 32014 |
| `<pad>` (not a reflection token) | 32015 |

The list in the spec was correct; all 15 reflection tokens plus `<pad>` are added tokens.

## 2. Official short-form inference parameters

Repo: https://github.com/AkariAsai/self-rag, commit `1fcdc420e48f50a7d7ab1ece5494221b93252e99`
(head of `main` on 2026-11-24), files `retrieval_lm/run_short_form.py`, `retrieval_lm/utils.py`, `README.md`.

| Item | Value | Code reference |
|---|---|---|
| Prompt format | `### Instruction:\n{instruction}\n\n### Response:\n` (no input); with evidence the code appends `[Retrieval]<paragraph>{title}\n{text}</paragraph>` to the prompt, one passage per generation | `utils.py` 6-12; `run_short_form.py` 86 |
| Retrieval threshold | CLI default `None` (then retrieve iff the generated text contains `[Retrieval]`); README commands use `--threshold 0.2`; rule: `score[Retrieval] / (score[Retrieval] + score[No Retrieval]) > threshold` | `run_short_form.py` 73-83, 270-271; `README.md` 251, 270 |
| Relevance weight | `w_rel` 1.0 | `run_short_form.py` 54, 281 |
| Support weight | `w_sup` 1.0; ground score = p(Fully) + 0.5 * p(Partially), normalised over the 3 support tokens | `run_short_form.py` 54, 137-141, 283 |
| Utility weight | function default 0.5, CLI default 1.0 (the CLI value is what runs); utility score = sum of [-1, -0.5, 0, 0.5, 1] * normalised p(Utility:1..5) | `run_short_form.py` 54, 144-148, 285 |
| How token probabilities are read | vLLM `logprobs=32016` (full vocabulary) on the no-retrieval pass and `logprobs=5000` on each evidence pass; relevance read at position 0; support and utility at the first position where such a token was generated; a token absent from the returned set gets logprob -100 (relevance/support/utility use `exp`). README runs add `--use_seqscore` (adds `exp(mean token logprob)`) | `run_short_form.py` 58, 89, 104-147, 152-157; `README.md` 253, 272 |
| Top-k passages | `--ndocs` CLI default 10; README: 10 (PopQA) and 5 (ARC-Challenge) | `run_short_form.py` 263; `README.md` 253, 272 |

Upstream quirks (observed, not fixed here): the threshold branch divides raw log probabilities
(no `exp`), and when a retrieval token is missing it sets -100 and then still indexes the missing
id (`run_short_form.py` 76-79). An adaptation must state how it handles both.

## 3. Candidate GGUF table

Base model: `selfrag/selfrag_llama2_7b`. Its model card states `license: mit`; it is a fine-tune of
Llama 2 7B, so the Llama 2 Community Licence terms of the base weights are expected to apply as
well. The owner should confirm the licence reading before any redistribution. Retrieved 2026-11-24.

| Repo (commit) | File | Quant | Bytes | sha256 | Stated licence | Converter and source |
|---|---|---|---|---|---|---|
| Ribben7/selfrag_llama2_7b-Q4_K_M-GGUF (`2253cc23d265343a963cc84e20d698e7e89ed75b`, created 2026-01-16) | `selfrag_llama2_7b-q4_k_m.gguf` | Q4_K_M | 4,081,095,936 | `9b05791653cf30a4237040a3c1c7d42f41334c90e74989978445c10e5143abd7` | MIT | llama.cpp via ggml-org GGUF-my-repo space; `base_model: selfrag/selfrag_llama2_7b` |
| QuantFactory/selfrag_llama2_7b-GGUF (`79baa8c210b9b314c7445e586a1bb5d17e844f95`, 2024-11-30) | `selfrag_llama2_7b.Q4_K_M.gguf` | Q4_K_M | 4,081,095,776 | `fb39ef3293bffcac2c44415123ec14d7b62c5cdc46a5dbfab9cf3830afe7dc03` | MIT | "created using llama.cpp" (no version) |
| kun432/selfrag_llama2_7b-Q4_K_M-GGUF (`7d9335a47713f78a311d3cfe8b7a9f376dfa3fcd`, 2024-04-05) | `selfrag_llama2_7b.Q4_K_M.gguf` | Q4_K_M | 4,081,095,424 | `a7bc8d80fa9c924818fad85b724317ad288a4c25abdcbfffebaa8b560f2bd93b` | MIT | llama.cpp via GGUF-my-repo space |
| m4r1/selfrag_llama2_7b-GGUF (`c45522dfaa84f092be965de0ad028e0ffd2a9b40`, 2024-01-25) | `selfrag_llama2_7b.q4_k_m.gguf` | Q4_K_M | 4,081,095,392 | `c4b1ad43c26754faae3e97e9ae1d7c6613fa251f7533be19a6d3446da3143d05` | MIT | not stated |
| devilteo911/selfrag_llama2_7b-q8_0 (`955dd7916419a9af387e39e9d6645ed5450d0d58`, 2023-10-20) | `selfrag_llama2_7b-q8_0.gguf` | Q8_0 | 7,161,229,408 | `75af6dc901c13e4c991c04b68c27ff2d3d68c37bf28aadb77a751c3e6ab689af` | MIT | not stated; too large for 6 GB VRAM |

URLs: `https://huggingface.co/<repo>` and `https://huggingface.co/api/models/<repo>?blobs=true`.
QuantFactory also lists Q2_K..Q8_0 (17 files); only Q4_K_M is shown. Found with
`https://huggingface.co/api/models?search=selfrag_llama2_7b`.

Pick: Ribben7 Q4_K_M (documented converter, newest llama.cpp, explicit base model).
Fit on 6 GB VRAM (RTX 3060 Laptop, 6144 MiB) at context 2048, estimate only:
weights 4,081,095,936 bytes (3.80 GiB); KV cache fp16 = 2 x 32 layers x 32 KV heads x 128 head dim
x 2 bytes = 524,288 bytes per token, so 1.00 GiB at 2048 (2.00 GiB at 4096; `config.json`: 32 layers,
32 heads, 32 KV heads, hidden 4096). About 4.8 GiB plus compute buffers: expected to fit fully on
GPU at 2048; at 4096 partial CPU offload is likely. Not measured.

## 4. Header check of the pick

Range read: first 4,194,304 bytes (HTTP Range `bytes=0-4194303`) of
`https://huggingface.co/Ribben7/selfrag_llama2_7b-Q4_K_M-GGUF/resolve/2253cc23d265343a963cc84e20d698e7e89ed75b/selfrag_llama2_7b-q4_k_m.gguf`,
owner OK in chat 2026-11-24, saved only in the session scratch dir (prefix sha256
`2b1281b6ea8af0a3d3a40b39fc01b0afc8b668870147826fe189bdd9de6aed48`). No full download.

```bash
python -I scripts/gguf_tokens.py <scratch>/ribben7_q4km_head.gguf --tokens "[No Retrieval]" "[Retrieval]" ...
```

Output (metadata): GGUF v3, tensor_count 291, kv_count 34, `general.architecture` llama,
`general.file_type` 15 (Q4_K_M), `general.license` mit, `general.name` "Llama 2 7b Hf",
`general.organization` "Meta Llama", `general.sampling.temp` 0.6, `general.sampling.top_p` 0.9,
`tokenizer.ggml.model` llama, vocab_size 32016. (`general.name`/`organization` come from the base
config; they do not name Self-RAG.) In Git Bash, `</paragraph>` needs `MSYS_NO_PATHCONV=1`,
otherwise the shell rewrites it as a path.

| Token | GGUF id | Original id | Occurrences | Type |
|---|---|---|---|---|
| `[No Retrieval]` | 32000 | 32000 | 1 | 4 user_defined |
| `[Retrieval]` | 32001 | 32001 | 1 | 4 user_defined |
| `[Continue to Use Evidence]` | 32002 | 32002 | 1 | 4 user_defined |
| `[Irrelevant]` | 32003 | 32003 | 1 | 4 user_defined |
| `[Relevant]` | 32004 | 32004 | 1 | 4 user_defined |
| `<paragraph>` | 32005 | 32005 | 1 | 4 user_defined |
| `</paragraph>` | 32006 | 32006 | 1 | 4 user_defined |
| `[Utility:1]` .. `[Utility:5]` | 32007 .. 32011 | 32007 .. 32011 | 1 each | 4 user_defined |
| `[Fully supported]` | 32012 | 32012 | 1 | 4 user_defined |
| `[Partially supported]` | 32013 | 32013 | 1 | 4 user_defined |
| `[No support / Contradictory]` | 32014 | 32014 | 1 | 4 user_defined |
| `<pad>` | 32015 | 32015 | 1 | 4 user_defined |

Result: every reflection token is a single token with its original id: yes. Type user_defined
(not control), so runtimes are expected to print them as literal text; to be confirmed in P7.5-15.

## 5. Runtime

- `ollama --version`: `ollama version is 0.40.17` (local, 2026-11-24; latest release v0.40.18).
- Sources: Ollama source at tag v0.40.17, `api/types.go` and `server/routes.go`
  (https://github.com/ollama/ollama/tree/v0.40.17), and https://docs.ollama.com/api/generate,
  retrieved 2026-11-24.
- `/api/generate` supports `raw: true`: yes (`api/types.go` 92-93: "no formatting will be applied to the prompt").
- Logprobs: request fields `logprobs` (bool) and `top_logprobs` (int, 0-20; the server returns
  HTTP 400 above 20, `server/routes.go` 331-332). Response: `logprobs` array, one entry per
  generated token with `token` (text), `bytes`, `logprob` and `top_logprobs` (list of
  `{token, bytes, logprob}`); no token ids (`api/types.go` 623-641). Not verified: whether the
  values are before or after the sampler chain.
- llama.cpp server alternative (not installed; a further download needing owner OK): `n_probs`
  returns the top N tokens per generated token with token ids; `post_sampling_probs` switches to
  post-sampler probabilities (`tools/server/README.md` at ggml-org/llama.cpp commit
  `2a487782f2f218bc6c01e1f366a7e73f002faf08`, lines 595-673). No upper bound is stated there.
- Gap with Ollama: at most the top 20 alternatives per position, matched by token text, not id.
  The official code asks for the full vocabulary (32016) or 5000 alternatives and reads specific
  token probabilities. A reflection token outside the top 20 gets no probability; using the
  official fallback (logprob -100) the formulas stay the same, but scores differ from the
  original whenever a scored token falls outside the top 20. Impact for R31: an adaptation, never
  `validated`; P7.5-15 must report, per position, how many scored reflection tokens were missing
  from the top 20.

## 6. Decision

GO (Ollama) as an adaptation, per ADR-0022 D5. Conditions: (a) the full GGUF download
(Ribben7 Q4_K_M, 4,081,095,936 bytes, sha256
`9b05791653cf30a4237040a3c1c7d42f41334c90e74989978445c10e5143abd7`) is shown to the owner first in
P7.5-14 and its sha256 is checked after download; (b) requests use `raw: true`, `logprobs: true`,
`top_logprobs: 20`; (c) the top-20 gap and the missing-token fallback are recorded in the run
evidence; (d) R31 is `blocked` if P7.5-15 shows reflection tokens are not printed as literal
text, or if more than 20% of scored positions (retrieval decision, relevance, support, utility)
have none of that group's reflection tokens in the top 20; the P7.5-15 spec may lower, not raise,
this 20% before the run; (e) the GGUF carries `general.sampling.temp` 0.6 and `top_p` 0.9, so
every request pins sampler options (`temperature: 0`, `top_k: 0` or the widest allowed, `top_p: 1`),
and P7.5-15 checks before reporting any score whether Ollama logprobs are pre- or post-sampler
(for example: the top-20 logprobs at one position must not change between `top_k: 1` and
`top_k: 0`); if they are post-sampler and cannot be made pre-sampler, R31 is `blocked`.
llama.cpp (`n_probs` with ids) stays a fallback that needs owner OK for the binary.
