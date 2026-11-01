# Free models

Checked against the Ollama library on 2026-10-26. Model lists change often: re-check at P2.2 and let the Tech Scout flag new ones. All models below are free to download and run locally; check each model's own license before redistributing it or using it commercially.

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
