"""Run the reusable contract checks against the fake provider, with the network blocked."""

import asyncio
import socket

import pytest
from rag_foundry_providers import (
    FakeChatModel,
    FakeEmbeddingModel,
    FakeReranker,
    ProviderRegistry,
    fake_chat_profile,
    fake_embedding_profile,
    fake_reranker_profile,
)
from rag_foundry_providers.contract_checks import CHAT_CHECKS, EMBEDDING_CHECKS, REGISTRY_CHECKS, RERANK_CHECKS

_BLOCKED = "network blocked in provider tests"


def _ids(checks: tuple) -> list[str]:
    return [c.__name__ for c in checks]


@pytest.mark.parametrize("streaming", [True, False], ids=["streaming", "no-streaming"])
@pytest.mark.parametrize("check", CHAT_CHECKS, ids=_ids(CHAT_CHECKS))
def test_chat_contract(check, streaming: bool) -> None:
    asyncio.run(check(FakeChatModel(fake_chat_profile(streaming=streaming))))


@pytest.mark.parametrize("check", EMBEDDING_CHECKS, ids=_ids(EMBEDDING_CHECKS))
def test_embedding_contract(check) -> None:
    asyncio.run(check(FakeEmbeddingModel(fake_embedding_profile())))


@pytest.mark.parametrize("check", RERANK_CHECKS, ids=_ids(RERANK_CHECKS))
def test_rerank_contract(check) -> None:
    asyncio.run(check(FakeReranker(fake_reranker_profile())))


@pytest.mark.parametrize("check", REGISTRY_CHECKS, ids=_ids(REGISTRY_CHECKS))
@pytest.mark.parametrize("kind", ["chat", "embedding", "reranker"])
def test_registry_contract(check, kind: str) -> None:
    registry = ProviderRegistry()
    if kind == "chat":
        profile = fake_chat_profile()
        model = FakeChatModel(profile)
    elif kind == "embedding":
        profile = fake_embedding_profile()
        model = FakeEmbeddingModel(profile)
    else:
        profile = fake_reranker_profile()
        model = FakeReranker(profile)
    registry.add_profile(profile)
    registry.bind(profile.id, model)
    asyncio.run(check(registry, profile.id))


def test_network_guard_blocks_external_connection() -> None:
    with pytest.raises(RuntimeError, match=_BLOCKED):
        socket.create_connection(("example.com", 80))
    with pytest.raises(RuntimeError, match=_BLOCKED):
        socket.getaddrinfo("example.com", 80)
    s = socket.socket()
    try:
        with pytest.raises(RuntimeError, match=_BLOCKED):
            s.connect(("93.184.216.34", 80))
    finally:
        s.close()
