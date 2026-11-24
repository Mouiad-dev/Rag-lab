"""Adapter factory behavior, registry errors, and lazy imports."""

import subprocess
import sys
from typing import Any, get_args

import pytest
from rag_foundry_providers.capabilities import Capabilities, ModelKind, ModelProfile
from rag_foundry_providers.config import AdapterName, Endpoint, ProvidersConfig
from rag_foundry_providers.errors import (
    DuplicateAdapterError,
    ProviderUnavailableError,
    UnknownAdapterError,
    UnsupportedModelKindError,
)
from rag_foundry_providers.factories import AdapterRegistry, default_adapter_registry
from rag_foundry_providers.pipeline import build_registry


def _profile(kind: ModelKind, *, profile_id: str = "test") -> ModelProfile:
    caps = Capabilities(embedding_dim=8) if kind is ModelKind.EMBEDDING else Capabilities(local=True)
    return ModelProfile(id=profile_id, provider="test", model="m", kind=kind, capabilities=caps)


@pytest.mark.parametrize(
    ("adapter", "kind", "expected"),
    [
        ("fake", ModelKind.CHAT, "FakeChatModel"),
        ("fake", ModelKind.EMBEDDING, "FakeEmbeddingModel"),
        ("fake", ModelKind.RERANKER, "FakeReranker"),
        ("ollama", ModelKind.CHAT, "OllamaChatModel"),
        ("ollama", ModelKind.EMBEDDING, "OllamaEmbeddingModel"),
        ("anthropic", ModelKind.CHAT, "AnthropicChatModel"),
        ("openai_compatible", ModelKind.CHAT, "OpenAICompatibleChatModel"),
        ("openai_compatible", ModelKind.EMBEDDING, "OpenAICompatibleEmbeddingModel"),
    ],
)
def test_factories_build_existing_adapter_classes(
    adapter: str, kind: ModelKind, expected: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    endpoint_args: dict[str, Any] = {"id": "ep", "adapter": adapter, "local": adapter != "anthropic"}
    if adapter in {"ollama", "openai_compatible"}:
        endpoint_args["base_url"] = "http://localhost:11434"
    if adapter == "anthropic":
        endpoint_args["api_key_env"] = "TEST_ANTHROPIC_KEY"
        monkeypatch.setenv("TEST_ANTHROPIC_KEY", "test-secret-value")
    endpoint = Endpoint(**endpoint_args)
    model = default_adapter_registry().build(_profile(kind), endpoint)
    assert type(model).__name__ == expected


def test_fake_factory_passes_clock_even_when_none(monkeypatch: pytest.MonkeyPatch) -> None:
    from rag_foundry_providers import fake

    profile = _profile(ModelKind.CHAT)
    endpoint = Endpoint(id="fake", adapter="fake", local=True)
    seen: list[object] = []

    class Capture:
        def __init__(self, _profile, *, clock):
            seen.append(clock)

    monkeypatch.setattr(fake, "FakeChatModel", Capture)
    marker = object()
    default_adapter_registry().build(profile, endpoint, clock=marker)
    default_adapter_registry().build(profile, endpoint)
    assert seen == [marker, None]


@pytest.mark.parametrize(
    ("adapter", "kind", "module", "class_name", "endpoint_args", "clock_given", "expected"),
    [
        (
            "ollama",
            ModelKind.CHAT,
            "ollama",
            "OllamaChatModel",
            {"base_url": "http://local"},
            False,
            {"api_base": "http://local", "transport": "transport"},
        ),
        (
            "ollama",
            ModelKind.EMBEDDING,
            "ollama",
            "OllamaEmbeddingModel",
            {"base_url": "http://local"},
            True,
            {"api_base": "http://local", "transport": "transport", "clock": "clock"},
        ),
        (
            "anthropic",
            ModelKind.CHAT,
            "anthropic",
            "AnthropicChatModel",
            {"api_key_env": "KEY"},
            False,
            {"api_key_env": "KEY", "transport": "transport"},
        ),
        (
            "anthropic",
            ModelKind.CHAT,
            "anthropic",
            "AnthropicChatModel",
            {"api_key_env": "KEY", "base_url": "https://local"},
            True,
            {"api_key_env": "KEY", "api_base": "https://local", "transport": "transport", "clock": "clock"},
        ),
        (
            "openai_compatible",
            ModelKind.CHAT,
            "openai_compat",
            "OpenAICompatibleChatModel",
            {"base_url": "http://localhost", "local": True, "default_max_tokens": 99},
            False,
            {
                "api_base": "http://localhost",
                "api_key_env": None,
                "default_max_tokens": 99,
                "max_tokens_field": "max_completion_tokens",
                "supports_temperature": True,
                "transport": "transport",
            },
        ),
        (
            "openai_compatible",
            ModelKind.CHAT,
            "openai_compat",
            "OpenAICompatibleChatModel",
            {"base_url": "http://localhost", "local": False, "default_max_tokens": 99},
            False,
            {
                "api_base": "http://localhost",
                "api_key_env": None,
                "default_max_tokens": 99,
                "max_tokens_field": "max_completion_tokens",
                "supports_temperature": False,
                "transport": "transport",
            },
        ),
        (
            "openai_compatible",
            ModelKind.CHAT,
            "openai_compat",
            "OpenAICompatibleChatModel",
            {
                "base_url": "http://localhost",
                "local": False,
                "default_max_tokens": 99,
                "supports_temperature": True,
                "max_tokens_field": "max_tokens",
                "api_key_env": "KEY",
            },
            False,
            {
                "api_base": "http://localhost",
                "api_key_env": "KEY",
                "default_max_tokens": 99,
                "max_tokens_field": "max_tokens",
                "supports_temperature": True,
                "transport": "transport",
            },
        ),
        (
            "openai_compatible",
            ModelKind.EMBEDDING,
            "openai_compat",
            "OpenAICompatibleEmbeddingModel",
            {"base_url": "http://localhost", "api_key_env": "KEY"},
            False,
            {"api_base": "http://localhost", "api_key_env": "KEY", "transport": "transport"},
        ),
        ("fake", ModelKind.CHAT, "fake", "FakeChatModel", {}, True, {"clock": "clock"}),
        ("fake", ModelKind.CHAT, "fake", "FakeChatModel", {}, False, {"clock": None}),
        ("fake", ModelKind.EMBEDDING, "fake", "FakeEmbeddingModel", {}, False, {"clock": None}),
        ("fake", ModelKind.RERANKER, "fake", "FakeReranker", {}, False, {"clock": None}),
    ],
)
def test_factories_pass_exact_adapter_kwargs(
    adapter, kind, module, class_name, endpoint_args, clock_given, expected, monkeypatch
) -> None:
    import importlib

    target = importlib.import_module(f"rag_foundry_providers.{('adapters.' if module != 'fake' else '')}{module}")
    calls = []

    class Recorder:
        def __init__(self, *args, **kwargs):
            self.profile = args[0]
            calls.append((args, kwargs))

    monkeypatch.setattr(target, class_name, Recorder)
    endpoint = Endpoint(id="ep", adapter=adapter, **{"local": adapter != "anthropic", **endpoint_args})
    clock = object()
    transport = object()
    # Use stable marker labels in the expected dict while checking identity below.
    expected = {
        key: clock if value == "clock" else transport if value == "transport" else value
        for key, value in expected.items()
    }
    registry = default_adapter_registry()
    if clock_given:
        registry.build(_profile(kind), endpoint, transport=transport, clock=clock)
    else:
        registry.build(_profile(kind), endpoint, transport=transport)
    args, kwargs = calls[0]
    assert args == (_profile(kind),)
    assert kwargs == expected


def test_unknown_adapter_error_names_adapter_not_profile_slot() -> None:
    registry = AdapterRegistry()
    with pytest.raises(UnknownAdapterError) as info:
        registry.get("not_registered")
    assert str(info.value) == "unknown adapter 'not_registered'"
    assert info.value.profile_id == ""


def test_pipeline_transport_and_clock_reach_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    from rag_foundry_providers.adapters import ollama

    calls = []

    class Recorder:
        def __init__(self, *args, **kwargs):
            self.profile = args[0]
            calls.append((args, kwargs))

    monkeypatch.setattr(ollama, "OllamaChatModel", Recorder)
    config = ProvidersConfig.model_validate(
        {
            "schema_version": 1,
            "endpoints": [{"id": "ollama", "adapter": "ollama", "base_url": "http://local", "local": True}],
            "profiles": [
                {"id": "chat", "provider": "ollama", "model": "m", "kind": "chat", "capabilities": {"local": True}}
            ],
            "stages": {"generator": "chat"},
        }
    )
    transport, clock = object(), object()
    build_registry(config, stages=["generator"], transports={"ollama": transport}, clock=clock)
    assert calls[0][1] == {"api_base": "http://local", "transport": transport, "clock": clock}


def test_index_cli_maps_invalid_endpoint_url_safely(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    from pydantic import ValidationError
    from rag_foundry_worker import index_cli

    with pytest.raises(ValidationError) as validation:
        Endpoint(id="ollama", adapter="ollama", base_url="ftp://localhost", local=True)
    monkeypatch.setattr(
        index_cli,
        "load_settings",
        lambda: index_cli.Settings(database_url="postgresql+psycopg://u:p@localhost/db"),
    )
    monkeypatch.setattr(index_cli, "_run", lambda *_: (_ for _ in ()).throw(validation.value))
    assert index_cli.main(["build", "--view-id", "view"]) == 2
    assert capsys.readouterr().err.strip() == "error: invalid RAG_FOUNDRY_* settings"


@pytest.mark.parametrize(
    ("adapter", "kind"),
    [("anthropic", ModelKind.EMBEDDING), ("ollama", ModelKind.RERANKER), ("openai_compatible", ModelKind.RERANKER)],
)
def test_unsupported_kind_error_keeps_provider_unavailable_message(adapter: str, kind: ModelKind) -> None:
    endpoint = Endpoint.model_construct(id="ep", adapter=adapter, local=True)
    pid = "unsupported"
    with pytest.raises(UnsupportedModelKindError) as info:
        default_adapter_registry().build(_profile(kind, profile_id=pid), endpoint)
    assert isinstance(info.value, ProviderUnavailableError)
    assert str(info.value) == str(ProviderUnavailableError(pid))


@pytest.mark.parametrize(
    ("adapter", "kind", "field"),
    [
        ("ollama", ModelKind.CHAT, "base_url"),
        ("openai_compatible", ModelKind.CHAT, "base_url"),
        ("anthropic", ModelKind.CHAT, "api_key_env"),
    ],
)
def test_missing_factory_requirement_is_provider_unavailable(adapter: str, kind: ModelKind, field: str) -> None:
    endpoint = Endpoint.model_construct(id="ep", adapter=adapter, local=True)
    with pytest.raises(ProviderUnavailableError):
        default_adapter_registry().build(_profile(kind), endpoint)


def test_unknown_adapter_and_duplicate_registration_errors() -> None:
    registry = default_adapter_registry()
    endpoint = Endpoint.model_construct(id="ep", adapter="unknown", local=True)
    with pytest.raises(UnknownAdapterError):
        registry.build(_profile(ModelKind.CHAT), endpoint)
    with pytest.raises(DuplicateAdapterError):
        registry.register(registry.get("fake"))


def test_test_factory_is_used_by_pipeline_and_wrapped() -> None:
    class EchoFactory:
        name = "fake"
        kinds = frozenset({ModelKind.CHAT})
        requires_base_url = False
        requires_api_key = False

        def build(self, profile, endpoint, *, transport, clock):
            from rag_foundry_providers.fake import FakeChatModel

            return FakeChatModel(profile, clock=clock)

    config = ProvidersConfig.model_validate(
        {
            "schema_version": 1,
            "endpoints": [{"id": "fake", "adapter": "fake", "local": True}],
            "profiles": [
                {"id": "chat", "provider": "fake", "model": "m", "kind": "chat", "capabilities": {"local": True}}
            ],
            "stages": {"generator": "chat"},
        }
    )
    adapters = AdapterRegistry()
    adapters.register(EchoFactory())
    result = build_registry(config, stages=["generator"], adapters=adapters)
    from rag_foundry_providers.resilience import ResilientChatModel

    assert isinstance(result.chat("chat"), ResilientChatModel)


def test_default_factory_names_match_adapter_literal() -> None:
    assert default_adapter_registry().names() == set(get_args(AdapterName))


@pytest.mark.parametrize(
    ("adapter", "requires_base_url", "requires_api_key"),
    [("fake", False, False), ("ollama", True, False), ("anthropic", False, True), ("openai_compatible", True, False)],
)
def test_endpoint_validation_matches_factory_requirement_flags(
    adapter: str, requires_base_url: bool, requires_api_key: bool
) -> None:
    factory = default_adapter_registry().get(adapter)
    assert factory.requires_base_url is requires_base_url
    assert factory.requires_api_key is requires_api_key
    base = {"id": "ep", "adapter": adapter, "local": adapter != "anthropic"}
    with_key = {**base, "api_key_env": "TEST_KEY"}
    without_base = with_key if requires_api_key else base
    with_base = {**base, "base_url": "http://localhost:11434"}
    without_key = with_base if requires_base_url else base
    if requires_base_url:
        with pytest.raises(ValueError):
            Endpoint(**without_base)
        Endpoint(**with_base)
    else:
        Endpoint(**without_base)
    if requires_api_key:
        with pytest.raises(ValueError):
            Endpoint(**without_key)
        Endpoint(**with_key)
    else:
        Endpoint(**without_key)


def test_importing_factories_does_not_import_optional_adapters() -> None:
    code = (
        "import sys; import rag_foundry_providers.factories; "
        "assert not any(name in sys.modules for name in ("
        "'rag_foundry_providers.adapters.ollama', 'rag_foundry_providers.adapters.anthropic', "
        "'rag_foundry_providers.adapters.openai_compat'))"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
