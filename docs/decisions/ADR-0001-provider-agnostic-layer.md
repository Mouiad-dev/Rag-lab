# ADR-0001: Provider-agnostic model layer

Date: 2026-10-26
Status: accepted (library choice to be re-verified at P2.1)

## Context
Users must be able to pick any LLM, embedding or reranking provider: paid APIs (Anthropic, OpenAI, Google, Mistral, Cohere and others), free or open-weight models, and local runtimes (Ollama, LM Studio, vLLM). Kimi models from Moonshot are also wanted. The request said "km3"; we assume this means the Kimi K-series served by Moonshot's OpenAI-compatible API or as open weights through vLLM/Ollama. Confirm the exact model. A public install must work with zero paid keys. The old plan tied model access to LangChain integrations.

## Decision
1. **Own ports** in `packages/providers`: `ChatModel`, `EmbeddingModel`, `Reranker`, each with sync/async calls, streaming for chat, structured output where supported, and a `Usage` record (provider, model, tokens in/out, latency, estimated cost).
2. **Registry and model profiles.** A profile = provider + model ID + base URL (optional) + capability flags: `tool_calling`, `structured_output`, `vision`, `streaming`, `context_window`, `max_output_tokens`, `embedding_dim`, `token_embeddings`, `multivector`, `local`, `cost_per_mtok_in`, `cost_per_mtok_out`. Flags are declared in config and verified by an opt-in probe.
3. **Per-pipeline choice.** Each stage (generator, embedder, reranker, query_transform, judge, chunk_llm, scout) references a profile ID. The resolver rejects a pipeline whose stage needs a capability its profile lacks, before a job starts.
4. **Configuration.** Secrets only from env vars (for example `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`); profiles from `config/providers.yaml` (example: `config/providers.example.json`) and later the settings UI, which shows key presence, never values.
5. **Default adapter: LiteLLM**, wrapped by our ports. Native SDK adapters (for example the Anthropic SDK) may implement the same ports when a feature (prompt caching, extended thinking, batch) is not exposed well enough.
6. **Defaults.** Local: Ollama chat + Ollama embeddings. Paid example: Anthropic. Rerankers: a local cross-encoder plugin plus API rerankers through the port.
7. **Embedding identity** = provider + model + revision + preprocessing + dimension; changing any of these creates a new index (see PROJECT_ARCH.md §9).

## Alternatives
- **LangChain chat-model integrations only.** Rich ecosystem and already used for LangGraph, but one package per provider, uneven feature parity, frequent API churn, and framework types leaking into domain code. Kept as a consumer of our ports (a thin LangChain wrapper around `ChatModel`) for LangGraph workflows.
- **LiteLLM as default (chosen).** One dependency covers 100+ providers including Ollama, any OpenAI-compatible server (LM Studio, vLLM, Moonshot), Anthropic, Gemini, Mistral; unified chat, embedding and rerank calls; a maintained model cost map; optional proxy mode for teams. Risks: a broad, fast-moving dependency surface and occasional lag on new provider features. Mitigation: pin exact versions with hashes, wrap behind ports, keep native adapters possible, run a provider contract suite.
- **Native SDK per provider only.** Maximum fidelity but the most code to maintain; used selectively.
- **OpenAI-compatible HTTP only.** Simple and covers many local servers, but not every provider's native features or embeddings/rerank APIs.

## Consequences
- Domain and technique code never imports a vendor SDK (rule in `.claude/rules/10-architecture.md`).
- Cost and usage are measured uniformly, enabling budgets for experiments and the Tech Scout.
- Contract tests run against a fake provider in CI; live tests are opt-in and skipped without keys.
- Capability flags can be wrong; the probe and explicit errors are the guard. Cost estimates are estimates, not invoices.

## Evidence
To verify at P2.1 against current official docs: LiteLLM provider list, embedding/rerank support, cost map, async/streaming APIs; Ollama and LM Studio OpenAI-compatible endpoints; Moonshot API compatibility.
