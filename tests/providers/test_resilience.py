"""Retry and deadline wrappers: scripted fakes, injected sleep and RNG, no real waiting."""

import asyncio
import random
from collections.abc import AsyncIterator

import pytest
from rag_foundry_providers import (
    ChatMessage,
    ChatRequest,
    FakeChatModel,
    FakeEmbeddingModel,
    MissingCapabilityError,
    ResilientChatModel,
    ResilientEmbeddingModel,
    RetryPolicy,
    fake_chat_profile,
    fake_embedding_profile,
)
from rag_foundry_providers.capabilities import Capabilities, ModelKind, ModelProfile
from rag_foundry_providers.contract_checks import CHAT_CHECKS, DETERMINISTIC_CHAT_CHECKS, EMBEDDING_CHECKS
from rag_foundry_providers.errors import ProviderCallError

REQ = ChatRequest(messages=(ChatMessage(role="user", content="hello world"),))


def _paid_profile() -> ModelProfile:
    return ModelProfile(
        id="paid-chat",
        provider="fake",
        model="paid-v1",
        kind=ModelKind.CHAT,
        capabilities=Capabilities(streaming=True, tool_calling=False, structured_output=False, local=False),
    )


class Sleeper:
    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.delays.append(seconds)


def _err(reason="server_error", retry_after_s=None, pid="fake-chat"):
    return ProviderCallError(pid, reason, retry_after_s=retry_after_s)


class ScriptedChat:
    """Raises queued errors (one per attempt), then delegates to the fake."""

    def __init__(self, errors, profile=None, *, hang=False):
        self.profile = profile or fake_chat_profile()
        self._inner = FakeChatModel(self.profile)
        self.errors = list(errors)
        self.attempts = 0
        self.hang = hang

    async def complete(self, request):
        self.attempts += 1
        if self.hang:
            await asyncio.Event().wait()
        if self.errors:
            raise self.errors.pop(0)
        return await self._inner.complete(request)

    async def stream(self, request) -> AsyncIterator:
        self.attempts += 1
        if self.hang:
            await asyncio.Event().wait()
        if self.errors:
            raise self.errors.pop(0)
        async for chunk in self._inner.stream(request):
            yield chunk


class ScriptedEmbed:
    def __init__(self, errors):
        self.profile = fake_embedding_profile()
        self._inner = FakeEmbeddingModel(self.profile)
        self.errors = list(errors)
        self.attempts = 0

    async def embed(self, texts, *, input_type="document"):
        self.attempts += 1
        if self.errors:
            raise self.errors.pop(0)
        return await self._inner.embed(texts, input_type=input_type)


def _chat(inner, sleeper, **policy):
    return ResilientChatModel(inner, RetryPolicy(**policy), sleep=sleeper, rng=random.Random(7))


def test_policy_defaults_match_owner_decisions():
    p = RetryPolicy()
    assert (p.max_attempts, p.backoff_min_s, p.backoff_max_s, p.max_retry_after_s, p.attempt_timeout_s) == (
        3,
        20.0,
        30.0,
        60.0,
        300.0,
    )


def test_retry_then_success_with_seeded_jitter():
    inner, sleeper = ScriptedChat([_err()]), Sleeper()
    result = asyncio.run(_chat(inner, sleeper).complete(REQ))
    assert result.text
    assert inner.attempts == 2
    expected = random.Random(7).uniform(20, 30)
    assert sleeper.delays == [expected]
    assert 20 <= sleeper.delays[0] <= 30


def test_retry_after_within_limit_uses_max_of_header_and_jitter():
    inner, sleeper = ScriptedChat([_err("rate_limited", retry_after_s=45.0)]), Sleeper()
    asyncio.run(_chat(inner, sleeper).complete(REQ))
    assert sleeper.delays == [45.0]
    inner, sleeper = ScriptedChat([_err("rate_limited", retry_after_s=7.0)]), Sleeper()
    asyncio.run(_chat(inner, sleeper).complete(REQ))
    assert sleeper.delays == [random.Random(7).uniform(20, 30)]


def test_retry_after_over_limit_fails_at_once():
    inner, sleeper = ScriptedChat([_err("rate_limited", retry_after_s=120.0)]), Sleeper()
    with pytest.raises(ProviderCallError) as info:
        asyncio.run(_chat(inner, sleeper).complete(REQ))
    assert info.value.reason == "rate_limited"
    assert inner.attempts == 1
    assert sleeper.delays == []


def test_last_real_error_raised_after_three_attempts():
    errors = [_err("overloaded"), _err("rate_limited"), _err("server_error")]
    inner, sleeper = ScriptedChat(errors), Sleeper()
    with pytest.raises(ProviderCallError) as info:
        asyncio.run(_chat(inner, sleeper).complete(REQ))
    assert info.value.reason == "server_error"
    assert inner.attempts == 3
    assert len(sleeper.delays) == 2


@pytest.mark.parametrize("reason", ["connection", "rate_limited", "overloaded", "server_error"])
def test_retryable_reasons_retry(reason):
    inner, sleeper = ScriptedChat([_err(reason)]), Sleeper()
    asyncio.run(_chat(inner, sleeper).complete(REQ))
    assert inner.attempts == 2


@pytest.mark.parametrize(
    "reason",
    ["auth", "billing", "invalid_request", "input_too_long", "model_not_found", "refused", "bad_response"],
)
def test_non_retryable_reasons_raise_at_once(reason):
    inner, sleeper = ScriptedChat([_err(reason)]), Sleeper()
    with pytest.raises(ProviderCallError) as info:
        asyncio.run(_chat(inner, sleeper).complete(REQ))
    assert info.value.reason == reason
    assert inner.attempts == 1
    assert sleeper.delays == []


def test_missing_capability_and_foreign_errors_propagate_at_once():
    for exc in (MissingCapabilityError.__new__(MissingCapabilityError), ValueError("x")):
        inner, sleeper = ScriptedChat([exc]), Sleeper()
        with pytest.raises(type(exc)):
            asyncio.run(_chat(inner, sleeper).complete(REQ))
        assert inner.attempts == 1


def test_timeout_retried_for_local_not_for_paid():
    inner, sleeper = ScriptedChat([_err("timeout")]), Sleeper()
    asyncio.run(_chat(inner, sleeper).complete(REQ))
    assert inner.attempts == 2
    paid = ScriptedChat([_err("timeout", pid="paid-chat")], profile=_paid_profile())
    with pytest.raises(ProviderCallError) as info:
        asyncio.run(_chat(paid, Sleeper()).complete(REQ))
    assert info.value.reason == "timeout"
    assert paid.attempts == 1


def test_deadline_gives_timeout_error_without_cause_or_context():
    inner = ScriptedChat([], hang=True)
    model = _chat(inner, Sleeper(), max_attempts=1, attempt_timeout_s=0.05)
    with pytest.raises(ProviderCallError) as info:
        asyncio.run(model.complete(REQ))
    assert info.value.reason == "timeout"
    assert info.value.__cause__ is None
    assert info.value.__context__ is None


async def _collect(model):
    return [c async for c in model.stream(REQ)]


def test_stream_error_before_first_chunk_is_retried():
    inner, sleeper = ScriptedChat([_err()]), Sleeper()
    chunks = asyncio.run(_collect(_chat(inner, sleeper)))
    assert chunks and chunks[-1].usage is not None
    assert inner.attempts == 2
    assert len(sleeper.delays) == 1


def test_stream_error_after_first_chunk_is_not_retried():
    class Broken(ScriptedChat):
        async def stream(self, request):
            self.attempts += 1
            yield (await FakeChatModel(self.profile).complete(request)) and _first_chunk()
            raise _err()

    def _first_chunk():
        from rag_foundry_providers import ChatChunk

        return ChatChunk(delta="a")

    inner, sleeper = Broken([]), Sleeper()
    with pytest.raises(ProviderCallError):
        asyncio.run(_collect(_chat(inner, sleeper)))
    assert inner.attempts == 1
    assert sleeper.delays == []


def test_stalled_stream_hits_deadline():
    inner = ScriptedChat([], hang=True)
    model = _chat(inner, Sleeper(), max_attempts=1, attempt_timeout_s=0.05)
    with pytest.raises(ProviderCallError) as info:
        asyncio.run(_collect(model))
    assert info.value.reason == "timeout"


def test_inner_stream_closed_on_early_consumer_close():
    closed = []

    class Tracked(ScriptedChat):
        async def stream(self, request):
            from rag_foundry_providers import ChatChunk

            try:
                yield ChatChunk(delta="a")
                yield ChatChunk(delta="b")
            finally:
                closed.append(True)

    async def run():
        gen = _chat(Tracked([]), Sleeper()).stream(REQ)
        await anext(gen)
        await gen.aclose()

    asyncio.run(run())
    assert closed == [True]


def test_embed_retry_and_non_retry():
    inner, sleeper = ScriptedEmbed([_err("overloaded")]), Sleeper()
    model = ResilientEmbeddingModel(inner, RetryPolicy(), sleep=sleeper, rng=random.Random(7))
    asyncio.run(model.embed(["a"]))
    assert inner.attempts == 2
    inner = ScriptedEmbed([_err("auth")])
    model = ResilientEmbeddingModel(inner, RetryPolicy(), sleep=Sleeper(), rng=random.Random(7))
    with pytest.raises(ProviderCallError):
        asyncio.run(model.embed(["a"]))
    assert inner.attempts == 1


@pytest.mark.parametrize("check", CHAT_CHECKS + DETERMINISTIC_CHAT_CHECKS, ids=lambda c: c.__name__)
def test_chat_contract_through_wrapper(check):
    asyncio.run(check(_chat(FakeChatModel(fake_chat_profile()), Sleeper())))


@pytest.mark.parametrize("check", EMBEDDING_CHECKS, ids=lambda c: c.__name__)
def test_embedding_contract_through_wrapper(check):
    model = ResilientEmbeddingModel(FakeEmbeddingModel(fake_embedding_profile()), RetryPolicy(), sleep=Sleeper())
    asyncio.run(check(model))
