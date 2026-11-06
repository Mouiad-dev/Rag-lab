"""Explicit, lazy construction of provider adapters."""

from typing import Any, Protocol

from rag_foundry_providers.capabilities import ModelKind, ModelProfile
from rag_foundry_providers.config import Endpoint
from rag_foundry_providers.errors import (
    DuplicateAdapterError,
    ProviderUnavailableError,
    UnknownAdapterError,
    UnsupportedModelKindError,
)
from rag_foundry_providers.usage import Clock


class AdapterFactory(Protocol):
    name: str
    kinds: frozenset[ModelKind]
    requires_base_url: bool
    requires_api_key: bool

    def build(self, profile: ModelProfile, endpoint: Endpoint, *, transport: Any, clock: Clock | None) -> object: ...


def _clock(clock: Clock | None) -> dict[str, Clock]:
    return {} if clock is None else {"clock": clock}


class FakeFactory:
    name = "fake"
    kinds = frozenset({ModelKind.CHAT, ModelKind.EMBEDDING, ModelKind.RERANKER})
    requires_base_url = False
    requires_api_key = False

    def build(self, profile: ModelProfile, endpoint: Endpoint, *, transport: Any, clock: Clock | None) -> object:
        from rag_foundry_providers.fake import FakeChatModel, FakeEmbeddingModel, FakeReranker

        constructors = {
            ModelKind.CHAT: FakeChatModel,
            ModelKind.EMBEDDING: FakeEmbeddingModel,
            ModelKind.RERANKER: FakeReranker,
        }
        return constructors[profile.kind](profile, clock=clock)


class OllamaFactory:
    name = "ollama"
    kinds = frozenset({ModelKind.CHAT, ModelKind.EMBEDDING})
    requires_base_url = True
    requires_api_key = False

    def build(self, profile: ModelProfile, endpoint: Endpoint, *, transport: Any, clock: Clock | None) -> object:
        if endpoint.base_url is None:
            raise ProviderUnavailableError(profile.id)
        from rag_foundry_providers.adapters.ollama import OllamaChatModel, OllamaEmbeddingModel

        constructor = OllamaChatModel if profile.kind is ModelKind.CHAT else OllamaEmbeddingModel
        return constructor(profile, api_base=endpoint.base_url, transport=transport, **_clock(clock))


class AnthropicFactory:
    name = "anthropic"
    kinds = frozenset({ModelKind.CHAT})
    requires_base_url = False
    requires_api_key = True

    def build(self, profile: ModelProfile, endpoint: Endpoint, *, transport: Any, clock: Clock | None) -> object:
        if endpoint.api_key_env is None:
            raise ProviderUnavailableError(profile.id)
        from rag_foundry_providers.adapters.anthropic import AnthropicChatModel

        extra = {} if endpoint.base_url is None else {"api_base": endpoint.base_url}
        return AnthropicChatModel(
            profile, api_key_env=endpoint.api_key_env, transport=transport, **extra, **_clock(clock)
        )


class OpenAICompatibleFactory:
    name = "openai_compatible"
    kinds = frozenset({ModelKind.CHAT, ModelKind.EMBEDDING})
    requires_base_url = True
    requires_api_key = False

    def build(self, profile: ModelProfile, endpoint: Endpoint, *, transport: Any, clock: Clock | None) -> object:
        if endpoint.base_url is None:
            raise ProviderUnavailableError(profile.id)
        from rag_foundry_providers.adapters.openai_compat import (
            OpenAICompatibleChatModel,
            OpenAICompatibleEmbeddingModel,
        )

        if profile.kind is ModelKind.CHAT:
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
        return OpenAICompatibleEmbeddingModel(
            profile,
            api_base=endpoint.base_url,
            api_key_env=endpoint.api_key_env,
            transport=transport,
            **_clock(clock),
        )


class AdapterRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, AdapterFactory] = {}

    def register(self, factory: AdapterFactory) -> None:
        if factory.name in self._factories:
            raise DuplicateAdapterError(factory.name)
        self._factories[factory.name] = factory

    def get(self, name: str) -> AdapterFactory:
        try:
            return self._factories[name]
        except KeyError:
            error = UnknownAdapterError("", name)
            error.args = (f"unknown adapter '{name}'",)
            raise error from None

    def names(self) -> frozenset[str]:
        return frozenset(self._factories)

    def build(
        self,
        profile: ModelProfile,
        endpoint: Endpoint,
        *,
        transport: Any = None,
        clock: Clock | None = None,
    ) -> object:
        try:
            factory = self.get(endpoint.adapter)
        except UnknownAdapterError:
            raise UnknownAdapterError(profile.id, endpoint.adapter) from None
        if profile.kind not in factory.kinds:
            raise UnsupportedModelKindError(profile.id, endpoint.adapter, profile.kind.value)
        if factory.requires_base_url and endpoint.base_url is None:
            raise ProviderUnavailableError(profile.id)
        if factory.requires_api_key and endpoint.api_key_env is None:
            raise ProviderUnavailableError(profile.id)
        return factory.build(profile, endpoint, transport=transport, clock=clock)


def default_adapter_registry() -> AdapterRegistry:
    registry = AdapterRegistry()
    for factory in (FakeFactory(), OllamaFactory(), AnthropicFactory(), OpenAICompatibleFactory()):
        registry.register(factory)
    return registry
