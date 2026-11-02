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
`CHAT_CHECKS`, `EMBEDDING_CHECKS`, `RERANK_CHECKS` and `REGISTRY_CHECKS` (called as `check(registry, profile_id)`).
An adapter test runs each one against its model:

```python
@pytest.mark.parametrize("check", CHAT_CHECKS, ids=lambda c: c.__name__)
def test_my_adapter(check):
    asyncio.run(check(my_chat_model))
```

Capabilities a profile does not declare as `True` are checked on their refusal path (`MissingCapabilityError`).

## Not yet

There are no real adapters, no keys and no network access in this package. The local Ollama default arrives in P2.2.
