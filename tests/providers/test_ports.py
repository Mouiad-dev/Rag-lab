"""Ports, message validators and check_chat_request."""

from collections.abc import AsyncIterator
from typing import Any

import pytest
from pydantic import ValidationError
from rag_foundry_providers import (
    Capabilities,
    ChatChunk,
    ChatMessage,
    ChatModel,
    ChatRequest,
    ChatResult,
    EmbeddingModel,
    EmbeddingResult,
    MissingCapabilityError,
    ModelKind,
    ModelProfile,
    Operation,
    Reranker,
    RerankItem,
    RerankResult,
    TokenSource,
    ToolCall,
    ToolSpec,
    Usage,
    check_chat_request,
)


def make_profile(**caps: bool | None) -> ModelProfile:
    return ModelProfile(id="p", provider="fake", model="m", kind=ModelKind.CHAT, capabilities=Capabilities(**caps))


def make_usage(op: Operation = Operation.CHAT) -> Usage:
    return Usage(
        profile_id="p",
        provider="fake",
        model="m",
        operation=op,
        input_tokens=1,
        output_tokens=1,
        token_source=TokenSource.ESTIMATED,
        latency_ms=0.0,
    )


MSG = (ChatMessage(role="user", content="hi"),)
TOOL = ToolSpec(name="t", description="d", parameters_schema={})
ALL_ON = make_profile(streaming=True, tool_calling=True, structured_output=True)


def caps_of(exc: pytest.ExceptionInfo[MissingCapabilityError]) -> set[str]:
    return {p.capability for p in exc.value.problems}


def test_pass_case() -> None:
    req = ChatRequest(messages=MSG, tools=(TOOL,), response_schema={"type": "object"})
    check_chat_request(ALL_ON, req, streaming=True)
    check_chat_request(make_profile(), ChatRequest(messages=MSG), streaming=False)


@pytest.mark.parametrize(
    ("request_kwargs", "streaming", "cap"),
    [
        ({"tools": (TOOL,)}, False, "tool_calling"),
        ({"response_schema": {}}, False, "structured_output"),
        ({}, True, "streaming"),
    ],
)
def test_single_missing_capability(request_kwargs: dict[str, Any], streaming: bool, cap: str) -> None:
    with pytest.raises(MissingCapabilityError) as exc:
        check_chat_request(make_profile(), ChatRequest(messages=MSG, **request_kwargs), streaming=streaming)
    assert caps_of(exc) == {cap}
    assert all(p.stage is None for p in exc.value.problems)


def test_all_three_missing_reported_together() -> None:
    req = ChatRequest(messages=MSG, tools=(TOOL,), response_schema={})
    with pytest.raises(MissingCapabilityError) as exc:
        check_chat_request(make_profile(), req, streaming=True)
    assert caps_of(exc) == {"tool_calling", "structured_output", "streaming"}


def test_declared_false_versus_none() -> None:
    req = ChatRequest(messages=MSG)
    with pytest.raises(MissingCapabilityError) as none_exc:
        check_chat_request(make_profile(), req, streaming=True)
    with pytest.raises(MissingCapabilityError) as false_exc:
        check_chat_request(make_profile(streaming=False), req, streaming=True)
    assert none_exc.value.problems[0].declared is None
    assert false_exc.value.problems[0].declared is False


def test_message_validators_reject_bad_input() -> None:
    with pytest.raises(ValidationError):
        ChatMessage(role="robot", content="x")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        ChatMessage(role="user", content="x", extra=1)  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        ChatRequest(messages=())
    with pytest.raises(ValidationError):
        ChatRequest(messages=MSG, max_output_tokens=0)
    with pytest.raises(ValidationError):
        ChatResult(text="x", finish_reason="weird", usage=make_usage())  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        ChatResult(text="x", finish_reason="stop")  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        EmbeddingResult(vectors=((1.0, 2.0), (1.0,)), usage=make_usage(Operation.EMBED))
    with pytest.raises(ValidationError):
        RerankItem(index=-1, score=0.5)
    with pytest.raises(ValidationError):
        RerankResult(items=())  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        ToolCall(id="1", name="n")  # type: ignore[call-arg]


def test_valid_results_and_frozen() -> None:
    emb = EmbeddingResult(vectors=(), usage=make_usage(Operation.EMBED))
    assert emb.vectors == ()
    item = RerankItem(index=0, score=1.0)
    with pytest.raises(ValidationError):
        item.index = 2  # type: ignore[misc]
    assert ChatChunk(delta="a").usage is None


class StubChat:
    profile = ALL_ON

    async def complete(self, request: ChatRequest) -> ChatResult:
        return ChatResult(text="ok", finish_reason="stop", usage=make_usage())

    async def stream(self, request: ChatRequest) -> AsyncIterator[ChatChunk]:
        yield ChatChunk(delta="ok", usage=make_usage(Operation.CHAT_STREAM))


class NotAPort:
    pass


def test_stub_satisfies_protocol() -> None:
    assert isinstance(StubChat(), ChatModel)
    assert not isinstance(NotAPort(), ChatModel)
    assert not isinstance(StubChat(), EmbeddingModel)
    assert not isinstance(StubChat(), Reranker)
