# Free models

Checked against the Ollama library on 2026-10-26. Model lists change often: re-check at P2.2 and let the Tech Scout flag new ones. All models below are free to download and run locally; check each model's own license before redistributing it or using it commercially.

## Verified for P2.2

Retrieved 2026-11-02 from the official Ollama pages https://ollama.com/library/gemma4 , https://ollama.com/library/gemma4:e4b , https://ollama.com/library/gemma4/tags , https://ollama.com/library/bge-m3 , https://ollama.com/library/bge-m3/tags , the Ollama release notes https://github.com/ollama/ollama/releases and the BAAI model card https://huggingface.co/BAAI/bge-m3 . Other rows below are not re-checked.

| Field | `gemma4:e4b` | `bge-m3` |
|---|---|---|
| Tag exists on ollama.com/library | yes (`e4b`, same entry as `latest`; digest `009acb0d7fe1` on the tags page) | yes (`latest` = `567m`, digest `790764642607`) |
| Download size | 6.6 GB (llama.cpp build, digest `dc35e8d9c606`, same as `e4b-it-q4_K_M`) to 9.5 GB (MLX build for Apple Silicon); the page shows the range "6.6GB - 9.5GB". Other quantizations: q8_0 9.0 GB, bf16 16 GB | 1.2 GB (F16, 567M parameters) |
| Context length | 128K tokens | 8K on the Ollama page; 8192 tokens on the model card |
| Embedding dimension | not applicable | 1024 (BAAI model card table and `1_Pooling/config.json`; the Ollama page does not state it) |
| Tool calling | yes (page tag "tools") | not applicable (embedding model) |
| Vision | yes, text and image input (page tag "vision"; the model card also lists audio for E4B) | no (text input) |
| Structured output | unknown (not stated on the model page; release `v0.21.1` notes a fix for Gemma 4 structured outputs with `think=false`) | not applicable |
| Thinking mode | yes (page tag "thinking") | not applicable |
| License | Apache License 2.0 (license blob on the model page) | MIT (license blob on the model page; model card says `mit`) |
| Ollama minimum version | at least `0.20.0` (the release notes of v0.20.0, 2026-04-02, introduce Gemma 4); the model page states no minimum | unknown (the page states none; `/api/embed` is documented in v0.40.13 `docs/api.md`) |
| Works with `ollama/ollama:0.40.13`? | yes (0.40.13 is newer than 0.20.0) | yes by the same check, no stated minimum |

Open note: the e4b size on the page is a range because the tag has a llama.cpp and an MLX build; the Linux/Docker figure to plan for is 6.6 GB.

## Local chat models (Ollama)

| Model | Sizes | Good for | Notes |
|---|---|---|---|
| `gemma4` | e2b, e4b, 12b, 26b, 31b | **Default (`e4b`)**: low RAM, multilingual incl. Arabic | e2b/e4b run on most laptops |
| `qwen3` / `qwen3.5` / `qwen3.6` / `qwen3.8` | small to 27b+ | Strong multilingual + tool calling | Good alternative default on 16 GB+ RAM |
| `llama3.1` / `llama3.2` / `llama3.3` | 1b to 70b | Widely supported baseline | Weaker Arabic than Qwen/Gemma |
| `mistral`, `mistral-small3.2`, `mistral-nemo` | 7b to 24b | General chat, function calling | |
| `phi4` | 14b | Reasoning on modest hardware | English-focused |
| `deepseek-r1`, `qwq` | various | Step-by-step reasoning (slower) | Use for judge/agent experiments |
| `gpt-oss` | 20b, 120b | Open-weight reasoning + tools | Needs more RAM |
| `qwen3-vl`, `gemma3n`, `llama3.2-vision`, `minicpm-v` | various | Images / scanned pages | Multimodal phase (P7) |
| `glm-ocr` | — | OCR of documents | Ingestion experiments (P3) |

## Local embedding models (Ollama)

| Model | Notes |
|---|---|
| `bge-m3` | **Default**: multilingual (Arabic/English), 1024 dims, long inputs |
| `qwen3-embedding` (0.6b / 4b / 8b) | Strong multilingual; 0.6b is light |
| `embeddinggemma` | Small, multilingual |
| `nomic-embed-text` (v1, v2-moe) | Popular, fast; v2-moe is multilingual |
| `snowflake-arctic-embed2`, `paraphrase-multilingual`, `granite-embedding` | Alternatives to benchmark |
| `mxbai-embed-large`, `all-minilm` | English-focused baselines |

## Rerankers

Ollama has no native rerank endpoint. Plan for P5: run `BAAI/bge-reranker-v2-m3` (multilingual) or a Qwen3 reranker locally through sentence-transformers, or use an LLM-as-reranker with a local chat model.

## Other free local servers
LM Studio and vLLM (OpenAI-compatible) run the same open-weight models from Hugging Face. They are already in `config/providers.example.json`.

## Free hosted tiers (need an account and API key)
Several providers offer free tiers or free models with rate limits, for example Google Gemini API, Groq, Mistral, and free models on OpenRouter. Limits and terms change often and data may be used for training on some free tiers, so the app treats them as normal paid providers with cost 0 and shows a warning. Verify current terms before adding a default.
