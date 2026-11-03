# packages/providers

Provider-agnostic model layer (ADR-0001, see `docs/decisions/ADR-0001-provider-agnostic-layer.md`).

## What P2.1 provides

- Async ports: `ChatModel`, `EmbeddingModel`, `Reranker`, with frozen request/result types.
- Model profiles and capability flags. A flag of `None` means "not declared" and is never treated as supported.
- Usage records with provider, model, operation, tokens, token source (`provider` or `estimated`), latency and
  estimated cost. A cost of `None` means unknown; it is never invented.
- `ProviderRegistry`: profiles, adapter bindings and pre-job capability validation (`StageRequirement`).
- Deterministic fake chat, embedding and reranker (`rag_foundry_providers.fake`) for tests and offline use.

## Reusing the contract checks

`rag_foundry_providers.contract_checks` has plain async functions with no pytest import, grouped in
`CHAT_CHECKS`, `EMBEDDING_CHECKS`, `RERANK_CHECKS` and `REGISTRY_CHECKS` (called as `check(registry, profile_id)`). `DETERMINISTIC_CHAT_CHECKS` adds a stream-equals-`complete()` check for deterministic adapters such as the fake; real models sample, so it does not apply to them.
An adapter test runs each one against its model:

```python
@pytest.mark.parametrize("check", CHAT_CHECKS, ids=lambda c: c.__name__)
def test_my_adapter(check):
    asyncio.run(check(my_chat_model))
```

Capabilities a profile does not declare as `True` are checked on their refusal path (`MissingCapabilityError`).

## Local Ollama (P2.2)

Direct HTTP adapters (`rag_foundry_providers.adapters.ollama`), no API keys. This is a direct adapter; whether
LiteLLM replaces it is re-decided at P2.3/P2.4.

- Install the extra: `uv sync --locked --extra ollama` (or add `rag-foundry-providers[ollama]`); it pulls in `httpx`.
- Default profiles: `ollama_chat_profile()` is `gemma4:e4b` (effective `num_ctx` 8192); `ollama_embedding_profile()`
  is `bge-m3` (1024 dims). Both have zero cost rates (hardware cost is not modelled).
- Run Ollama natively (0.40.13 verified), or with Docker: `docker compose --profile ollama up -d ollama`.
- Pull the models (sizes from `docs/plan/FREE_MODELS.md`): `ollama pull gemma4:e4b` (6.6 GB) and
  `ollama pull bge-m3` (1.2 GB).
- Token counts are `provider` when Ollama returns them and `estimated` when it omits them.
- Opt-in live test (skipped by default, CI stays offline; the first call can be slow while the model loads):

```powershell
$env:RAG_LIVE_OLLAMA = "1"; uv run pytest tests/live -m live -v -s
```

The base URL comes from `RAG_FOUNDRY_OLLAMA_BASE_URL` (default `http://127.0.0.1:11434`).

## Anthropic (P2.3)

Direct HTTP chat adapter (`rag_foundry_providers.adapters.anthropic`, `AnthropicChatModel`). Chat only: Anthropic has no
embeddings API, so there is no embedding adapter.

- Install the extra: `uv sync --locked --extra anthropic` (or add `rag-foundry-providers[anthropic]`); it pulls in `httpx`.
- The key comes only from the `ANTHROPIC_API_KEY` environment variable. Never put it in settings, config, fixtures or logs.
- Profiles: `anthropic_haiku_profile()` is `claude-haiku-5-5` (thinking disabled, default `max_tokens` 1024);
  `anthropic_sonnet_profile()` is `claude-sonnet-5-5` (thinking cannot be disabled, default `max_tokens` 16384).
  Rates in USD per MTok, verified 2026-11-02 (ADR-0001): Haiku in 0.10, out 0.50, cache read 0.01, 5-minute cache write
  0.125 (prompts up to 100,000 tokens; above that the cost is unknown, `None`); Sonnet in 2, out 10, cache read 0.10,
  5-minute cache write 2.50. 1-hour cache writes have unknown cost.
- Tools, structured output and vision are refused until P2.4. `temperature` is refused (the API rejects it).
- No retries until P2.5: every call is a single attempt.
- Usage is provider-sourced and includes cache read and write tokens. `UsageLedger` (append-only, in memory) and
  `RecordingChatModel` (`rag_foundry_providers.ledger`) record each call's tokens and cost; totals report cost as `None`
  when any call has unknown cost.
- Opt-in PAID live test (skipped by default). It uses Haiku, `max_output_tokens` 64 (32 for the two prompt tests), about
  5 to 8 calls, and a hard cap of USD 0.05 per run checked before each call against the worst-case cost:

```powershell
$k = Read-Host -AsSecureString 'Anthropic key'; $env:ANTHROPIC_API_KEY = [System.Net.NetworkCredential]::new('', $k).Password; $env:RAG_LIVE_ANTHROPIC = '1'; uv run pytest tests/live/test_anthropic_live.py -m live -v -s; Remove-Item Env:ANTHROPIC_API_KEY, Env:RAG_LIVE_ANTHROPIC; Remove-Variable k
```

## Config and per-stage selection (P2.4)

- File format: JSON, `schema_version` 1 (`config/providers.example.json` is loadable; copy it to `config/providers.json`).
  `load_providers_config(path)` validates strictly (no unknown keys, no duplicate keys, at most 1 MiB) and its errors never
  echo input values. The file has `endpoints` (id, adapter, `base_url`, `api_key_env`, `local`), `profiles` (model profiles;
  rates are JSON strings such as `"0.10"`) and `stages`.
- Adapters: `fake`, `ollama`, `anthropic`, `openai_compatible`. `api_key_env` holds only the NAME of an environment
  variable; the key is read when the adapter is built, and only for profiles bound to a stage the pipeline uses. A key
  requires an https `base_url` unless the host is loopback.
- OpenAI-compatible endpoints (LM Studio, vLLM, Moonshot, OpenAI) use the same adapter (`/chat/completions`,
  `/embeddings`). The key is optional for local servers and required for paid ones.
- Stages are a closed set: `generator`, `embedder`, `reranker`, `query_transform`, `judge`, `chunk_llm`, `scout`.
- `resolve_pipeline(config, {"generator": StageNeed("chat", {"tool_calling"})})` rejects a pipeline before any adapter is
  built or key is read: unbound stages raise `UnboundStageError`; wrong kind or missing capabilities raise one
  `MissingCapabilityError` listing every gap. `build_registry(config, stages=[...])` then binds adapters for those
  stages only. There is no fallback to another profile.
- Not implemented yet: tool calling (P8.3), structured output (first step that needs it, P4.4 or P5.4) and vision. A profile
  cannot declare them `True`. Retries and budgets come in P2.5.
- OpenAI-compatible endpoint settings (on the endpoint): `max_tokens_field` (`max_tokens` for LM Studio, default
  `max_completion_tokens`), `supports_temperature` (unset means true on a local endpoint and false on a non-local one; a
  request that sets `temperature` against a false endpoint raises `UnsupportedRequestError` before any call) and
  `default_max_tokens` (output cap when a request gives none). A non-local `openai_compatible` endpoint must set
  `default_max_tokens` (owner decision: 4096 for Moonshot).
