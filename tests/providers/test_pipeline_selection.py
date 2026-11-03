"""Per-stage selection from a providers config: pre-job rejection and the two-provider done test."""

import asyncio
import json
import uuid
from typing import Any

import httpx
import pytest
from rag_foundry_providers import (
    ChatMessage,
    ChatRequest,
    MissingCapabilityError,
    ProviderUnavailableError,
    UnboundStageError,
)
from rag_foundry_providers.adapters.anthropic import UnsupportedRequestError
from rag_foundry_providers.config import ProvidersConfig, ProvidersConfigError, parse_providers_config
from rag_foundry_providers.contract_checks import REGISTRY_CHECKS
from rag_foundry_providers.errors import MissingCredentialError
from rag_foundry_providers.pipeline import StageNeed, build_registry, resolve_pipeline

KEY_ENV = "TEST_MOONSHOT_KEY_P24"
SENTINEL = f"sk-TEST-SENTINEL-{uuid.uuid4()}"
DIM = 4

ENDPOINTS: list[dict[str, Any]] = [
    {"id": "fake", "adapter": "fake", "local": True},
    {"id": "ollama", "adapter": "ollama", "base_url": "http://127.0.0.1:11434", "local": True},
    {"id": "anthropic", "adapter": "anthropic", "api_key_env": "TEST_ANTHROPIC_KEY_P24", "local": False},
    {
        "id": "moon",
        "adapter": "openai_compatible",
        "base_url": "https://api.example.test/v1",
        "api_key_env": KEY_ENV,
        "local": False,
        "max_tokens_field": "max_completion_tokens",
        "supports_temperature": False,
        "default_max_tokens": 4096,
    },
    {
        "id": "lms",
        "adapter": "openai_compatible",
        "base_url": "http://localhost:1234/v1",
        "local": True,
        "max_tokens_field": "max_tokens",
    },
]
PROFILES: list[dict[str, Any]] = [
    {
        "id": "fake-chat",
        "provider": "fake",
        "model": "fake-chat",
        "kind": "chat",
        "capabilities": {"streaming": True, "local": True},
    },
    {
        "id": "local-chat",
        "provider": "ollama",
        "model": "gemma4:e4b",
        "kind": "chat",
        "capabilities": {"streaming": True, "tool_calling": False, "local": True},
    },
    {"id": "undeclared-chat", "provider": "ollama", "model": "m", "kind": "chat", "capabilities": {"local": True}},
    {
        "id": "local-embed",
        "provider": "ollama",
        "model": "bge-m3",
        "kind": "embedding",
        "capabilities": {"embedding_dim": DIM, "local": True},
    },
    {
        "id": "moon-chat",
        "provider": "moon",
        "model": "kimi-k3.1-pro",
        "kind": "chat",
        "capabilities": {"streaming": True},
    },
    {
        "id": "lms-chat",
        "provider": "lms",
        "model": "m",
        "kind": "chat",
        "capabilities": {"streaming": True, "local": True},
    },
    {
        "id": "claude-embed",
        "provider": "anthropic",
        "model": "x",
        "kind": "embedding",
        "capabilities": {"embedding_dim": DIM},
    },
]


def _config(stages: dict[str, str]) -> ProvidersConfig:
    return parse_providers_config({"schema_version": 1, "endpoints": ENDPOINTS, "profiles": PROFILES, "stages": stages})


def _fail_transport() -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no request expected")

    return httpx.MockTransport(handler)


def _embed_transport() -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        n = len(json.loads(request.content)["input"])
        return httpx.Response(200, json={"model": "bge-m3", "embeddings": [[0.1] * DIM] * n, "prompt_eval_count": 3})

    return httpx.MockTransport(handler)


@pytest.mark.parametrize("profile", ["local-chat", "undeclared-chat"])
def test_done1_tool_calling_rejected_before_any_adapter(profile: str, monkeypatch: pytest.MonkeyPatch) -> None:
    constructed: list[str] = []
    monkeypatch.setattr("rag_foundry_providers.pipeline._construct", lambda *a, **k: constructed.append("x"))
    cfg = _config({"generator": profile})
    with pytest.raises(MissingCapabilityError) as info:
        resolve_pipeline(cfg, {"generator": StageNeed("chat", frozenset({"tool_calling"}))})
    (problem,) = info.value.problems
    assert (problem.stage, problem.profile_id, problem.capability) == ("generator", profile, "tool_calling")
    assert problem.declared is (False if profile == "local-chat" else None)
    assert constructed == []


def test_done2_two_stages_on_two_providers() -> None:
    cfg = _config({"generator": "fake-chat", "embedder": "local-embed"})
    needs = {"generator": StageNeed("chat"), "embedder": StageNeed("embedding")}
    assert resolve_pipeline(cfg, needs)["embedder"].profile_id == "local-embed"
    reg = build_registry(cfg, stages=needs, transports={"ollama": _embed_transport()})
    chat = asyncio.run(reg.chat("fake-chat").complete(ChatRequest(messages=(ChatMessage(role="user", content="hi"),))))
    emb = asyncio.run(reg.embedding("local-embed").embed(["a", "b"]))
    assert chat.usage.provider == "fake" and emb.usage.provider == "ollama"
    assert len(emb.vectors) == 2


def test_kind_mismatch_unbound_local_and_all_gaps_together() -> None:
    cfg = _config({"generator": "local-embed", "judge": "moon-chat", "reranker": "local-chat"})
    with pytest.raises(UnboundStageError) as unbound:
        resolve_pipeline(cfg, {"scout": StageNeed("chat"), "generator": StageNeed("chat")})
    assert unbound.value.stages == ("scout",)
    needs = {
        "generator": StageNeed("chat", frozenset({"streaming"})),
        "judge": StageNeed("chat", frozenset({"local"})),
        "reranker": StageNeed("chat", frozenset({"tool_calling"})),
    }
    with pytest.raises(MissingCapabilityError) as info:
        resolve_pipeline(cfg, needs)
    got = {(p.stage, p.profile_id, p.capability) for p in info.value.problems}
    assert got == {
        ("generator", "local-embed", "chat"),
        ("generator", "local-embed", "streaming"),
        ("judge", "moon-chat", "local"),
        ("reranker", "local-chat", "tool_calling"),
    }


def test_unbound_paid_profile_needs_no_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(KEY_ENV, raising=False)
    cfg = _config({"generator": "fake-chat", "judge": "moon-chat"})
    reg = build_registry(cfg, stages=["generator"])
    assert "moon-chat" in reg.profile_ids()
    with pytest.raises(ProviderUnavailableError):
        reg.chat("moon-chat")


def test_missing_key_for_bound_profile_names_variable_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(KEY_ENV, raising=False)
    cfg = _config({"judge": "moon-chat"})
    with pytest.raises(MissingCredentialError) as info:
        build_registry(cfg, stages=["judge"])
    assert KEY_ENV in str(info.value) and SENTINEL not in str(info.value)
    monkeypatch.setenv(KEY_ENV, SENTINEL)
    reg = build_registry(cfg, stages=["judge"], transports={"moon": _fail_transport()})
    assert reg.chat("moon-chat").profile.id == "moon-chat"
    assert SENTINEL not in repr(reg.chat("moon-chat"))


def test_unsupported_adapter_kind_pair_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_ANTHROPIC_KEY_P24", SENTINEL)
    cfg = _config({"embedder": "claude-embed"})
    with pytest.raises(ProviderUnavailableError):
        build_registry(cfg, stages=["embedder"])


def test_registry_checks_pass_on_built_registry() -> None:
    cfg = _config({"generator": "fake-chat"})
    reg = build_registry(cfg, stages=["generator"])
    for check in REGISTRY_CHECKS:
        asyncio.run(check(reg, "fake-chat"))


def _ok_chat(seen: list[httpx.Request]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        body = {
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        return httpx.Response(200, json=body)

    return httpx.MockTransport(handler)


def _req(**kw: Any) -> ChatRequest:
    return ChatRequest(messages=(ChatMessage(role="user", content="hi"),), **kw)


def test_moonshot_style_profile_refuses_temperature_before_any_call(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(KEY_ENV, SENTINEL)
    reg = build_registry(_config({"judge": "moon-chat"}), stages=["judge"], transports={"moon": _fail_transport()})
    with pytest.raises(UnsupportedRequestError):
        asyncio.run(reg.chat("moon-chat").complete(_req(temperature=0.2)))


def test_paid_endpoint_sends_default_cap_4096(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(KEY_ENV, SENTINEL)
    seen: list[httpx.Request] = []
    reg = build_registry(_config({"judge": "moon-chat"}), stages=["judge"], transports={"moon": _ok_chat(seen)})
    asyncio.run(reg.chat("moon-chat").complete(_req()))
    payload = json.loads(seen[0].content)
    assert payload["max_completion_tokens"] == 4096 and "max_tokens" not in payload
    assert "temperature" not in payload


def test_lm_studio_payload_uses_max_tokens_and_allows_temperature() -> None:
    seen: list[httpx.Request] = []
    reg = build_registry(_config({"judge": "lms-chat"}), stages=["judge"], transports={"lms": _ok_chat(seen)})
    asyncio.run(reg.chat("lms-chat").complete(_req(max_output_tokens=50, temperature=0.2)))
    payload = json.loads(seen[0].content)
    assert payload["max_tokens"] == 50 and "max_completion_tokens" not in payload
    assert payload["temperature"] == 0.2


def test_non_local_openai_compatible_endpoint_requires_default_max_tokens() -> None:
    endpoints = [{k: v for k, v in e.items() if k != "default_max_tokens"} for e in ENDPOINTS]
    with pytest.raises(ProvidersConfigError) as info:
        parse_providers_config(
            {"schema_version": 1, "endpoints": endpoints, "profiles": PROFILES, "stages": {"judge": "moon-chat"}}
        )
    assert "default_max_tokens" in str(info.value)


def test_unset_supports_temperature_on_non_local_endpoint_refuses_temperature(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(KEY_ENV, SENTINEL)
    endpoints = [{k: v for k, v in e.items() if k != "supports_temperature"} for e in ENDPOINTS]
    cfg = parse_providers_config(
        {"schema_version": 1, "endpoints": endpoints, "profiles": PROFILES, "stages": {"judge": "moon-chat"}}
    )
    reg = build_registry(cfg, stages=["judge"], transports={"moon": _fail_transport()})
    with pytest.raises(UnsupportedRequestError):
        asyncio.run(reg.chat("moon-chat").complete(_req(temperature=0.2)))
