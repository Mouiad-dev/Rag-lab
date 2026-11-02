"""Negative tests: each contract check must fail on a deliberately broken adapter.

Broken adapters wrap a healthy fake and change exactly one behaviour. Pydantic's
`model_copy(update=...)` skips validation, which is how invalid values (negative tokens,
negative cost) reach the checks even though the types reject them at construction.
A control test per port proves the unmodified wrapper passes every check, so each
failure below is caused by the one change.
"""

import asyncio
from collections.abc import AsyncIterator, Callable, Sequence
from decimal import Decimal
from typing import Any, Literal

import pytest
from rag_foundry_providers import (
    Capabilities,
    CapabilityProblem,
    ChatChunk,
    ChatRequest,
    ChatResult,
    EmbeddingResult,
    FakeChatModel,
    FakeEmbeddingModel,
    FakeReranker,
    MissingCapabilityError,
    ModelProfile,
    Operation,
    ProviderRegistry,
    RerankResult,
    ToolCall,
    Usage,
    fake_chat_profile,
    fake_embedding_profile,
    fake_reranker_profile,
)
from rag_foundry_providers import contract_checks as cc

NAN = float("nan")


def run(coro: Any) -> Any:
    async def _wrap() -> Any:
        return await coro

    return asyncio.run(_wrap())


def _usage(u: Usage, **update: Any) -> Usage:
    return u.model_copy(update=update)


# --- broken adapters ------------------------------------------------------------------------


class BrokenChat:
    def __init__(
        self,
        *,
        declared: ModelProfile | None = None,
        result_fn: Callable[[ChatResult], ChatResult] = lambda r: r,
        chunks_fn: Callable[[list[ChatChunk]], list[ChatChunk]] = lambda c: c,
        strip_request: bool = False,
    ) -> None:
        self._base = FakeChatModel(fake_chat_profile(streaming=True))
        self.profile = declared if declared is not None else self._base.profile
        self._result_fn = result_fn
        self._chunks_fn = chunks_fn
        self._strip = strip_request

    async def complete(self, request: ChatRequest) -> ChatResult:
        if self._strip:  # skip the capability refusal the fake would do
            request = request.model_copy(update={"tools": (), "response_schema": None})
        return self._result_fn(await self._base.complete(request))

    async def stream(self, request: ChatRequest) -> AsyncIterator[ChatChunk]:
        chunks = [c async for c in self._base.stream(request)]
        for chunk in self._chunks_fn(chunks):
            yield chunk


def _chat_profile(**caps: Any) -> ModelProfile:
    base = {"streaming": True, "tool_calling": False, "structured_output": False, "local": True}
    base.update(caps)
    return fake_chat_profile().model_copy(update={"capabilities": Capabilities(**base)})


class BrokenEmbed:
    def __init__(
        self,
        *,
        declared: ModelProfile | None = None,
        fn: Callable[[EmbeddingResult, Sequence[str], str], EmbeddingResult] = lambda r, t, i: r,
    ) -> None:
        self._base = FakeEmbeddingModel(fake_embedding_profile())
        self.profile = declared if declared is not None else self._base.profile
        self._fn = fn

    async def embed(
        self, texts: Sequence[str], *, input_type: Literal["document", "query"] = "document"
    ) -> EmbeddingResult:
        return self._fn(await self._base.embed(texts, input_type=input_type), texts, input_type)


class BrokenRerank:
    def __init__(
        self,
        *,
        declared: ModelProfile | None = None,
        fn: Callable[[RerankResult, int | None], RerankResult] = lambda r, n: r,
        ignore_top_n: bool = False,
    ) -> None:
        self._base = FakeReranker(fake_reranker_profile())
        self.profile = declared if declared is not None else self._base.profile
        self._fn = fn
        self._ignore = ignore_top_n

    async def rerank(self, query: str, documents: Sequence[str], *, top_n: int | None = None) -> RerankResult:
        result = await self._base.rerank(query, documents, top_n=None if self._ignore else top_n)
        return self._fn(result, top_n)


# --- controls: unmodified wrappers pass ----------------------------------------------------


@pytest.mark.parametrize("check", cc.CHAT_CHECKS, ids=lambda c: c.__name__)
def test_control_chat(check: Any) -> None:
    run(check(BrokenChat()))


@pytest.mark.parametrize("check", cc.EMBEDDING_CHECKS, ids=lambda c: c.__name__)
def test_control_embed(check: Any) -> None:
    run(check(BrokenEmbed()))


@pytest.mark.parametrize("check", cc.RERANK_CHECKS, ids=lambda c: c.__name__)
def test_control_rerank(check: Any) -> None:
    run(check(BrokenRerank()))


# --- chat ----------------------------------------------------------------------------------


def _with_usage(**update: Any) -> Callable[[ChatResult], ChatResult]:
    return lambda r: r.model_copy(update={"usage": _usage(r.usage, **update)})


def _move_usage_to_first(chunks: list[ChatChunk]) -> list[ChatChunk]:
    out = list(chunks)
    out[0] = out[0].model_copy(update={"usage": out[-1].usage})
    out[-1] = out[-1].model_copy(update={"usage": None})
    return out


def _usage_on_two(chunks: list[ChatChunk]) -> list[ChatChunk]:
    out = list(chunks)
    out[0] = out[0].model_copy(update={"usage": out[-1].usage})
    return out


def _stream_op_chat(chunks: list[ChatChunk]) -> list[ChatChunk]:
    out = list(chunks)
    out[-1] = out[-1].model_copy(update={"usage": _usage(out[-1].usage, operation=Operation.CHAT)})
    return out


def _empty_deltas(chunks: list[ChatChunk]) -> list[ChatChunk]:
    return [c.model_copy(update={"delta": ""}) for c in chunks]


def _different_text(chunks: list[ChatChunk]) -> list[ChatChunk]:
    out = list(chunks)
    out[0] = out[0].model_copy(update={"delta": "X"})
    return out


_OTHER_TOOL = ToolCall(id="1", name="not_declared", arguments={})

CHAT_NEGATIVE: list[tuple[str, Callable[[Any], Any], Callable[[], BrokenChat]]] = [
    ("wrong_kind", cc.check_chat_kind, lambda: BrokenChat(declared=fake_reranker_profile())),
    ("usage_wrong_profile_id", cc.check_chat_complete, lambda: BrokenChat(result_fn=_with_usage(profile_id="other"))),
    (
        "usage_wrong_operation",
        cc.check_chat_complete,
        lambda: BrokenChat(result_fn=_with_usage(operation=Operation.CHAT_STREAM)),
    ),
    ("usage_negative_input_tokens", cc.check_chat_complete, lambda: BrokenChat(result_fn=_with_usage(input_tokens=-1))),
    (
        "usage_negative_output_tokens",
        cc.check_chat_complete,
        lambda: BrokenChat(result_fn=_with_usage(output_tokens=-1)),
    ),
    (
        "usage_negative_cost",
        cc.check_chat_complete,
        lambda: BrokenChat(result_fn=_with_usage(estimated_cost_usd=Decimal("-1"))),
    ),
    ("usage_negative_latency", cc.check_chat_complete, lambda: BrokenChat(result_fn=_with_usage(latency_ms=-1.0))),
    ("usage_token_source_unset", cc.check_chat_complete, lambda: BrokenChat(result_fn=_with_usage(token_source="x"))),
    (
        "arabic_usage_wrong_operation",
        cc.check_chat_arabic,
        lambda: BrokenChat(result_fn=_with_usage(operation=Operation.EMBED)),
    ),
    (
        "tool_calls_finish_without_calls",
        cc.check_chat_complete,
        lambda: BrokenChat(result_fn=lambda r: r.model_copy(update={"finish_reason": "tool_calls"})),
    ),
    (
        "empty_text_and_no_tool_calls",
        cc.check_chat_complete,
        lambda: BrokenChat(result_fn=lambda r: r.model_copy(update={"text": ""})),
    ),
    ("stream_usage_not_last", cc.check_chat_stream, lambda: BrokenChat(chunks_fn=_move_usage_to_first)),
    ("stream_usage_on_two_chunks", cc.check_chat_stream, lambda: BrokenChat(chunks_fn=_usage_on_two)),
    ("stream_operation_chat", cc.check_chat_stream, lambda: BrokenChat(chunks_fn=_stream_op_chat)),
    ("stream_empty_joined_text", cc.check_chat_stream, lambda: BrokenChat(chunks_fn=_empty_deltas)),
    ("stream_text_differs_from_complete", cc.check_chat_stream, lambda: BrokenChat(chunks_fn=_different_text)),
    (
        "stream_undeclared_but_yields",
        cc.check_chat_stream,
        lambda: BrokenChat(declared=_chat_profile(streaming=False)),
    ),
    ("tools_undeclared_but_succeeds", cc.check_chat_tools, lambda: BrokenChat(strip_request=True)),
    (
        "tools_declared_unknown_tool_called",
        cc.check_chat_tools,
        lambda: BrokenChat(
            declared=_chat_profile(tool_calling=True),
            strip_request=True,
            result_fn=lambda r: r.model_copy(update={"tool_calls": (_OTHER_TOOL,), "finish_reason": "tool_calls"}),
        ),
    ),
    ("structured_undeclared_but_succeeds", cc.check_chat_structured, lambda: BrokenChat(strip_request=True)),
    (
        "structured_declared_but_missing",
        cc.check_chat_structured,
        lambda: BrokenChat(declared=_chat_profile(structured_output=True), strip_request=True),
    ),
]


@pytest.mark.parametrize(("name", "check", "make"), CHAT_NEGATIVE, ids=[c[0] for c in CHAT_NEGATIVE])
def test_chat_check_fails_on_broken_adapter(name: str, check: Any, make: Callable[[], BrokenChat]) -> None:
    with pytest.raises(AssertionError):
        run(check(make()))


# --- embedding -----------------------------------------------------------------------------


def _vec(dim: int = 16) -> tuple[float, ...]:
    return (0.1,) * dim


def _emb(r: EmbeddingResult, vectors: tuple[tuple[float, ...], ...]) -> EmbeddingResult:
    return EmbeddingResult(vectors=vectors, usage=r.usage)


def _refuse_query(r: EmbeddingResult, texts: Sequence[str], input_type: str) -> EmbeddingResult:
    if input_type == "query":
        raise ValueError("query input_type refused")
    return r


EMBED_NEGATIVE: list[tuple[str, Callable[[Any], Any], Callable[[], BrokenEmbed], type[BaseException]]] = [
    ("wrong_kind", cc.check_embed_kind, lambda: BrokenEmbed(declared=fake_chat_profile()), AssertionError),
    (
        "wrong_count",
        cc.check_embed_vectors,
        lambda: BrokenEmbed(fn=lambda r, t, i: _emb(r, r.vectors[:-1])),
        AssertionError,
    ),
    (
        "wrong_dimension",
        cc.check_embed_vectors,
        lambda: BrokenEmbed(fn=lambda r, t, i: _emb(r, tuple(_vec(8) for _ in r.vectors))),
        AssertionError,
    ),
    (
        "nan_in_vector",
        cc.check_embed_vectors,
        lambda: BrokenEmbed(fn=lambda r, t, i: _emb(r, ((NAN, *r.vectors[0][1:]), *r.vectors[1:]))),
        AssertionError,
    ),
    (
        "inf_in_vector",
        cc.check_embed_vectors,
        lambda: BrokenEmbed(fn=lambda r, t, i: _emb(r, ((float("inf"), *r.vectors[0][1:]), *r.vectors[1:]))),
        AssertionError,
    ),
    (
        "output_tokens_nonzero",
        cc.check_embed_usage,
        lambda: BrokenEmbed(fn=lambda r, t, i: r.model_copy(update={"usage": _usage(r.usage, output_tokens=3)})),
        AssertionError,
    ),
    (
        "wrong_operation",
        cc.check_embed_usage,
        lambda: BrokenEmbed(
            fn=lambda r, t, i: r.model_copy(update={"usage": _usage(r.usage, operation=Operation.RERANK)})
        ),
        AssertionError,
    ),
    (
        "empty_input_returns_vectors",
        cc.check_embed_empty,
        lambda: BrokenEmbed(fn=lambda r, t, i: _emb(r, (_vec(),)) if not t else r),
        AssertionError,
    ),
    ("input_type_refused", cc.check_embed_input_types, lambda: BrokenEmbed(fn=_refuse_query), ValueError),
]


@pytest.mark.parametrize(("name", "check", "make", "exc"), EMBED_NEGATIVE, ids=[c[0] for c in EMBED_NEGATIVE])
def test_embedding_check_fails_on_broken_adapter(
    name: str, check: Any, make: Callable[[], BrokenEmbed], exc: type[BaseException]
) -> None:
    with pytest.raises(exc):
        run(check(make()))


# --- rerank --------------------------------------------------------------------------------


def _items(r: RerankResult, items: Any) -> RerankResult:
    return r.model_copy(update={"items": tuple(items)})


def _dup_index(r: RerankResult, n: int | None) -> RerankResult:
    items = list(r.items)
    items[1] = items[1].model_copy(update={"index": items[0].index})
    return _items(r, items)


def _out_of_range(r: RerankResult, n: int | None) -> RerankResult:
    items = list(r.items)
    items[0] = items[0].model_copy(update={"index": 99})
    return _items(r, items)


def _ascending(r: RerankResult, n: int | None) -> RerankResult:
    return _items(r, r.items[::-1]) if n is None else r


def _nonfinite(r: RerankResult, n: int | None) -> RerankResult:
    items = list(r.items)
    items[0] = items[0].model_copy(update={"score": float("inf")})
    return _items(r, items)


def _no_items(r: RerankResult, n: int | None) -> RerankResult:
    return _items(r, ())


def _wrong_top_n_items(r: RerankResult, n: int | None) -> RerankResult:
    return _items(r, r.items[::-1]) if n is not None else r


RERANK_NEGATIVE: list[tuple[str, Callable[[Any], Any], Callable[[], BrokenRerank]]] = [
    ("wrong_kind", cc.check_rerank_kind, lambda: BrokenRerank(declared=fake_chat_profile())),
    ("duplicate_index", cc.check_rerank_order, lambda: BrokenRerank(fn=_dup_index)),
    ("index_out_of_range", cc.check_rerank_order, lambda: BrokenRerank(fn=_out_of_range)),
    ("scores_increasing", cc.check_rerank_order, lambda: BrokenRerank(fn=_ascending)),
    ("score_not_finite", cc.check_rerank_order, lambda: BrokenRerank(fn=_nonfinite)),
    ("no_items_order", cc.check_rerank_order, lambda: BrokenRerank(fn=_no_items)),
    ("top_n_ignored", cc.check_rerank_top_n, lambda: BrokenRerank(ignore_top_n=True)),
    ("no_items_top_n", cc.check_rerank_top_n, lambda: BrokenRerank(fn=_no_items)),
    ("top_n_not_first_of_ranking", cc.check_rerank_top_n, lambda: BrokenRerank(fn=_wrong_top_n_items)),
    (
        "wrong_operation",
        cc.check_rerank_usage,
        lambda: BrokenRerank(
            fn=lambda r, n: r.model_copy(update={"usage": _usage(r.usage, operation=Operation.EMBED)})
        ),
    ),
]


@pytest.mark.parametrize(("name", "check", "make"), RERANK_NEGATIVE, ids=[c[0] for c in RERANK_NEGATIVE])
def test_rerank_check_fails_on_broken_adapter(name: str, check: Any, make: Callable[[], BrokenRerank]) -> None:
    with pytest.raises(AssertionError):
        run(check(make()))


# --- registry ------------------------------------------------------------------------------
# ProviderRegistry.bind() rejects an adapter whose profile differs, and validate_requirements
# is correct for a profile's own flags, so these two cases use registry subclasses that
# simulate a faulty registry implementation.


class MismatchedBindingRegistry(ProviderRegistry):
    def chat(self, pid: str) -> Any:
        return BrokenChat(declared=_chat_profile(streaming=False))


def _registry(cls: type[ProviderRegistry]) -> tuple[ProviderRegistry, str]:
    registry = cls()
    profile = fake_chat_profile()
    registry.add_profile(profile)
    registry.bind(profile.id, FakeChatModel(profile))
    return registry, profile.id


def test_registry_binding_check_fails_on_profile_mismatch() -> None:
    registry, pid = _registry(MismatchedBindingRegistry)
    with pytest.raises(AssertionError):
        run(cc.check_registry_binding(registry, pid))


def test_registry_validation_check_fails_when_own_capabilities_rejected() -> None:
    class Rejecting(ProviderRegistry):
        def validate_requirements(self, reqs: Any) -> None:
            raise MissingCapabilityError((CapabilityProblem(None, "fake-chat", "streaming", None),))

    registry, pid = _registry(Rejecting)
    with pytest.raises(MissingCapabilityError):
        run(cc.check_registry_validates_own_capabilities(registry, pid))


def test_registry_checks_pass_on_healthy_registry() -> None:
    registry, pid = _registry(ProviderRegistry)
    for check in cc.REGISTRY_CHECKS:
        run(check(registry, pid))
