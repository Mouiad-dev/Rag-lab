"""Per-stage model selection: validate a pipeline's needs against the config, then bind adapters.

No global state, no default profiles and no fallback to another profile. `resolve_pipeline` reads no
environment variable and constructs no adapter; `build_registry` reads a key only for a bound profile.
"""

import asyncio
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from rag_foundry_providers.budget import BudgetedChatModel, BudgetedEmbeddingModel, RunBudget
from rag_foundry_providers.capabilities import Capability, ModelKind, ModelProfile
from rag_foundry_providers.config import Endpoint, ProvidersConfig
from rag_foundry_providers.errors import BudgetRequiredError, ProviderUnavailableError, UnboundStageError
from rag_foundry_providers.ledger import RecordingChatModel, RecordingEmbeddingModel, UsageSink
from rag_foundry_providers.registry import ProviderRegistry, StageRequirement
from rag_foundry_providers.resilience import ResilientChatModel, ResilientEmbeddingModel, RetryPolicy, Sleep
from rag_foundry_providers.usage import Clock

_KIND_CAPABILITY: dict[ModelKind, Capability] = {
    ModelKind.CHAT: Capability.CHAT,
    ModelKind.EMBEDDING: Capability.EMBEDDINGS,
    ModelKind.RERANKER: Capability.RERANKING,
}


@dataclass(frozen=True)
class StageNeed:
    """What a pipeline stage needs: a model kind plus extra capabilities."""

    kind: ModelKind
    required: frozenset[Capability] = frozenset()

    def __post_init__(self) -> None:
        # Accept config strings such as "tool_calling" and "chat".
        object.__setattr__(self, "kind", ModelKind(self.kind))
        object.__setattr__(self, "required", frozenset(Capability(c) for c in self.required))


def _profiles_registry(config: ProvidersConfig) -> ProviderRegistry:
    registry = ProviderRegistry()
    for profile in config.profiles:
        registry.add_profile(profile)
    return registry


def resolve_pipeline(config: ProvidersConfig, needs: Mapping[str, StageNeed]) -> dict[str, StageRequirement]:
    """Reject a pipeline before any adapter is built or key is read.

    Raises UnboundStageError for stages the config does not bind, then MissingCapabilityError listing
    every kind and capability gap together.
    """
    bound: Mapping[str, str] = config.stages
    unbound = tuple(stage for stage in needs if stage not in bound)
    if unbound:
        raise UnboundStageError(unbound)
    reqs = {
        stage: StageRequirement(
            profile_id=bound[stage],
            required=need.required | {_KIND_CAPABILITY[need.kind]},
        )
        for stage, need in needs.items()
    }
    _profiles_registry(config).validate_requirements(reqs)
    return reqs


def _construct(profile: ModelProfile, endpoint: Endpoint, transport: Any, clock: Clock | None) -> object:
    kind = profile.kind
    if endpoint.adapter == "fake":
        from rag_foundry_providers.fake import FakeChatModel, FakeEmbeddingModel, FakeReranker

        fakes = {
            ModelKind.CHAT: FakeChatModel,
            ModelKind.EMBEDDING: FakeEmbeddingModel,
            ModelKind.RERANKER: FakeReranker,
        }
        return fakes[kind](profile, clock=clock)
    if endpoint.adapter == "ollama" and endpoint.base_url is not None:
        from rag_foundry_providers.adapters.ollama import OllamaChatModel, OllamaEmbeddingModel

        if kind is ModelKind.CHAT:
            return OllamaChatModel(profile, api_base=endpoint.base_url, transport=transport, **_clock(clock))
        if kind is ModelKind.EMBEDDING:
            return OllamaEmbeddingModel(profile, api_base=endpoint.base_url, transport=transport, **_clock(clock))
    if endpoint.adapter == "anthropic" and kind is ModelKind.CHAT and endpoint.api_key_env is not None:
        from rag_foundry_providers.adapters.anthropic import AnthropicChatModel

        extra: dict[str, Any] = {} if endpoint.base_url is None else {"api_base": endpoint.base_url}
        return AnthropicChatModel(
            profile, api_key_env=endpoint.api_key_env, transport=transport, **extra, **_clock(clock)
        )
    if endpoint.adapter == "openai_compatible" and endpoint.base_url is not None:
        from rag_foundry_providers.adapters.openai_compat import (
            OpenAICompatibleChatModel,
            OpenAICompatibleEmbeddingModel,
        )

        if kind is ModelKind.CHAT:
            return OpenAICompatibleChatModel(
                profile,
                api_base=endpoint.base_url,
                api_key_env=endpoint.api_key_env,
                default_max_tokens=endpoint.default_max_tokens,
                max_tokens_field=endpoint.max_tokens_field or "max_completion_tokens",
                supports_temperature=(
                    endpoint.local if endpoint.supports_temperature is None else endpoint.supports_temperature
                ),
                transport=transport,
                **_clock(clock),
            )
        if kind is ModelKind.EMBEDDING:
            return OpenAICompatibleEmbeddingModel(
                profile,
                api_base=endpoint.base_url,
                api_key_env=endpoint.api_key_env,
                transport=transport,
                **_clock(clock),
            )
    # No adapter for this adapter/kind pair (for example Anthropic embeddings): fail, never fall back.
    raise ProviderUnavailableError(profile.id)


def _clock(clock: Clock | None) -> dict[str, Any]:
    return {} if clock is None else {"clock": clock}


def build_registry(
    config: ProvidersConfig,
    *,
    stages: Iterable[str],
    transports: Mapping[str, Any] | None = None,
    clock: Clock | None = None,
    ledger: UsageSink | None = None,
    budget: RunBudget | None = None,
    retry: RetryPolicy | None = None,
    retry_sleep: Sleep = asyncio.sleep,
) -> ProviderRegistry:
    """Register every profile; construct and bind adapters only for profiles bound to ``stages``.

    Chat and embedding adapters are bound as Resilient(Budgeted(Recording(adapter))); Recording needs a
    ledger and Budgeted needs a budget. A run that binds any non-local profile without a budget raises
    BudgetRequiredError before any adapter is built or key is read (ADR-0001 P2.5 D5). Rerankers are
    bound unwrapped. A missing key raises MissingCredentialError (variable name only). Unbound profiles
    need no key.
    """
    wanted = tuple(stages)
    bound: Mapping[str, str] = config.stages
    unbound = tuple(stage for stage in wanted if stage not in bound)
    if unbound:
        raise UnboundStageError(unbound)
    registry = _profiles_registry(config)
    endpoints = {endpoint.id: endpoint for endpoint in config.endpoints}
    profiles = {profile.id: profile for profile in config.profiles}
    transports = transports or {}
    pids = sorted({bound[stage] for stage in wanted})
    for pid in pids:
        # Rerankers are bound unwrapped, so a paid one would bypass the budget. Refused until P5.5.
        if profiles[pid].kind is ModelKind.RERANKER and profiles[pid].capabilities.local is not True:
            raise ProviderUnavailableError(pid)
    paid = tuple(pid for pid in pids if profiles[pid].capabilities.local is not True)
    if paid and budget is None:
        raise BudgetRequiredError(paid)
    policy = retry or RetryPolicy()
    for pid in pids:
        profile = profiles[pid]
        endpoint = endpoints[profile.provider]
        adapter = _construct(profile, endpoint, transports.get(endpoint.id), clock)
        registry.bind(pid, _wrap(adapter, profile.kind, ledger, budget, policy, retry_sleep))
    return registry


def _wrap(
    adapter: Any, kind: ModelKind, ledger: UsageSink | None, budget: RunBudget | None, policy: RetryPolicy, sleep: Sleep
) -> Any:
    if kind is ModelKind.CHAT:
        chat: Any = adapter
        if ledger is not None:
            chat = RecordingChatModel(chat, ledger)
        if budget is not None:
            chat = BudgetedChatModel(
                chat, budget, default_max_output_tokens=getattr(adapter, "default_max_tokens", None)
            )
        return ResilientChatModel(chat, policy, sleep=sleep)
    if kind is ModelKind.EMBEDDING:
        embed: Any = adapter
        if ledger is not None:
            embed = RecordingEmbeddingModel(embed, ledger)
        if budget is not None:
            embed = BudgetedEmbeddingModel(embed, budget)
        return ResilientEmbeddingModel(embed, policy, sleep=sleep)
    return adapter
