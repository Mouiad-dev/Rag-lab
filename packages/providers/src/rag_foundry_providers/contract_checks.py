"""Reusable port-level contract checks (stdlib and pydantic only; no test framework).

Each check is `async def check_<name>(model) -> None` and raises AssertionError with a
clear message. Adapters (P2.2/P2.3) and the P2.5 suite run them against any port
implementation. The checks never weaken for the fake: a capability a profile does not
declare as True is checked on its refusal path.
"""

import math
from collections.abc import Awaitable, Callable
from decimal import Decimal
from typing import Any

from rag_foundry_providers.capabilities import Capability, ModelKind, ModelProfile
from rag_foundry_providers.errors import MissingCapabilityError
from rag_foundry_providers.messages import ChatMessage, ChatRequest, ChatResult, ToolSpec
from rag_foundry_providers.registry import ProviderRegistry, StageRequirement
from rag_foundry_providers.usage import Operation, TokenSource, Usage

_ARABIC = "مرحبا بالعالم"
_TOOL = ToolSpec(name="lookup", description="Look something up.", parameters_schema={"type": "object"})
_SCHEMA: dict[str, Any] = {"type": "object", "properties": {"answer": {"type": "string"}}}


def _req(text: str = "hello world", **kwargs: Any) -> ChatRequest:
    return ChatRequest(messages=(ChatMessage(role="user", content=text),), **kwargs)


def _fail(message: str) -> None:
    raise AssertionError(message)


def _check_usage(usage: Usage, profile: ModelProfile, operation: Operation) -> None:
    if usage.profile_id != profile.id:
        _fail(f"usage.profile_id '{usage.profile_id}' != profile id '{profile.id}'")
    if usage.provider != profile.provider or usage.model != profile.model:
        _fail("usage provider/model do not match the profile")
    if usage.operation is not operation:
        _fail(f"usage.operation is {usage.operation!r}, expected {operation!r}")
    if not isinstance(usage.token_source, TokenSource):
        _fail("usage.token_source must be set to a TokenSource")
    if usage.input_tokens < 0 or usage.output_tokens < 0:
        _fail("usage token counts must be >= 0")
    if not math.isfinite(usage.latency_ms) or usage.latency_ms < 0:
        _fail(f"usage.latency_ms must be finite and >= 0, got {usage.latency_ms}")
    cost = usage.estimated_cost_usd
    if cost is not None and (not isinstance(cost, Decimal) or not cost.is_finite() or cost < 0):
        _fail(f"usage.estimated_cost_usd must be None or a finite Decimal >= 0, got {cost!r}")


def _check_chat_result_shape(result: ChatResult) -> None:
    if result.finish_reason == "tool_calls" and not result.tool_calls:
        _fail("finish_reason 'tool_calls' requires non-empty tool_calls")
    if not result.text and not result.tool_calls and result.structured is None:
        _fail("result has empty text, no tool_calls and no structured output")


async def _expect_missing(factory: Callable[[], Awaitable[Any]], what: str) -> None:
    """Call `factory` inside the try block so a synchronous raise is also caught."""
    try:
        await factory()
    except MissingCapabilityError:
        return
    _fail(f"{what}: expected MissingCapabilityError for an undeclared capability")


async def _drain(model: Any, request: ChatRequest) -> list[Any]:
    return [chunk async for chunk in model.stream(request)]


# --- chat ---------------------------------------------------------------------------------


async def check_chat_kind(model: Any) -> None:
    if model.profile.kind is not ModelKind.CHAT:
        _fail(f"chat adapter profile.kind is {model.profile.kind!r}, expected CHAT")


async def check_chat_complete(model: Any) -> None:
    result = await model.complete(_req())
    if not isinstance(result, ChatResult):
        _fail(f"complete returned {type(result).__name__}, expected ChatResult")
    _check_usage(result.usage, model.profile, Operation.CHAT)
    _check_chat_result_shape(result)


async def check_chat_arabic(model: Any) -> None:
    result = await model.complete(_req(_ARABIC))
    _check_usage(result.usage, model.profile, Operation.CHAT)
    _check_chat_result_shape(result)


async def check_chat_stream(model: Any) -> None:
    request = _req()
    if model.profile.declared(Capability.STREAMING) is not True:
        await _expect_missing(lambda: _drain(model, request), "stream")
        return
    chunks = await _drain(model, request)
    if not chunks:
        _fail("stream yielded no chunks")
    with_usage = [i for i, c in enumerate(chunks) if c.usage is not None]
    if with_usage != [len(chunks) - 1]:
        _fail(f"exactly one chunk, the last, must carry usage; usage chunk indices: {with_usage}")
    _check_usage(chunks[-1].usage, model.profile, Operation.CHAT_STREAM)
    joined = "".join(c.delta for c in chunks)
    if not joined:
        _fail("concatenated stream text is empty")


async def check_chat_stream_matches_complete(model: Any) -> None:
    """Stream text equals complete() text. Only for deterministic adapters (e.g. the fake):
    real models sample, so two calls can differ even with the same request."""
    if model.profile.declared(Capability.STREAMING) is not True:
        return
    request = _req()
    joined = "".join(c.delta for c in await _drain(model, request))
    complete = await model.complete(request)
    if joined != complete.text:
        _fail("stream chunks joined do not equal complete() text for the same request")


async def check_chat_tools(model: Any) -> None:
    request = _req(tools=(_TOOL,))
    if model.profile.declared(Capability.TOOL_CALLING) is not True:
        await _expect_missing(lambda: model.complete(request), "tools")
        return
    result = await model.complete(request)
    _check_usage(result.usage, model.profile, Operation.CHAT)
    _check_chat_result_shape(result)
    for call in result.tool_calls:
        if call.name != _TOOL.name:
            _fail(f"tool call names an undeclared tool '{call.name}'")


async def check_chat_structured(model: Any) -> None:
    request = _req(response_schema=_SCHEMA)
    if model.profile.declared(Capability.STRUCTURED_OUTPUT) is not True:
        await _expect_missing(lambda: model.complete(request), "response_schema")
        return
    result = await model.complete(request)
    _check_usage(result.usage, model.profile, Operation.CHAT)
    if not isinstance(result.structured, dict):
        _fail("structured output declared but result.structured is not a dict")


CHAT_CHECKS: tuple[Callable[[Any], Awaitable[None]], ...] = (
    check_chat_kind,
    check_chat_complete,
    check_chat_arabic,
    check_chat_stream,
    check_chat_tools,
    check_chat_structured,
)

# Extra checks for deterministic adapters only (the fake provider); not for real models.
DETERMINISTIC_CHAT_CHECKS: tuple[Callable[[Any], Awaitable[None]], ...] = (check_chat_stream_matches_complete,)

# --- embeddings ---------------------------------------------------------------------------

_EMBED_TEXTS = ["hello world", _ARABIC, "third text"]


def _check_vectors(model: Any, vectors: tuple[tuple[float, ...], ...], count: int) -> None:
    dim = model.profile.capabilities.embedding_dim
    if len(vectors) != count:
        _fail(f"expected {count} vectors, got {len(vectors)}")
    for i, vec in enumerate(vectors):
        if len(vec) != dim:
            _fail(f"vector {i} has length {len(vec)}, declared embedding_dim is {dim}")
        if not all(math.isfinite(x) for x in vec):
            _fail(f"vector {i} contains NaN or inf")


async def check_embed_kind(model: Any) -> None:
    if model.profile.kind is not ModelKind.EMBEDDING:
        _fail(f"embedding adapter profile.kind is {model.profile.kind!r}, expected EMBEDDING")


async def check_embed_vectors(model: Any) -> None:
    result = await model.embed(_EMBED_TEXTS)
    _check_vectors(model, result.vectors, len(_EMBED_TEXTS))


async def check_embed_usage(model: Any) -> None:
    result = await model.embed(_EMBED_TEXTS)
    _check_usage(result.usage, model.profile, Operation.EMBED)
    if result.usage.output_tokens != 0:
        _fail(f"embed usage.output_tokens must be 0, got {result.usage.output_tokens}")


async def check_embed_empty(model: Any) -> None:
    result = await model.embed([])
    if len(result.vectors) != 0:
        _fail("empty input must return empty vectors")
    _check_usage(result.usage, model.profile, Operation.EMBED)


async def check_embed_input_types(model: Any) -> None:
    for input_type in ("document", "query"):
        result = await model.embed(_EMBED_TEXTS[:2], input_type=input_type)
        _check_vectors(model, result.vectors, 2)


EMBEDDING_CHECKS: tuple[Callable[[Any], Awaitable[None]], ...] = (
    check_embed_kind,
    check_embed_vectors,
    check_embed_usage,
    check_embed_empty,
    check_embed_input_types,
)

# --- rerank -------------------------------------------------------------------------------

_DOCS = ["the cat sat", "dogs bark loudly", "a cat and a dog", _ARABIC]


async def check_rerank_kind(model: Any) -> None:
    if model.profile.kind is not ModelKind.RERANKER:
        _fail(f"reranker adapter profile.kind is {model.profile.kind!r}, expected RERANKER")


async def check_rerank_order(model: Any) -> None:
    result = await model.rerank("cat", _DOCS)
    if len(result.items) != len(_DOCS):
        _fail(f"rerank without top_n must return {len(_DOCS)} items, got {len(result.items)}")
    indices = [item.index for item in result.items]
    if len(set(indices)) != len(indices):
        _fail(f"rerank indices are not unique: {indices}")
    if any(not 0 <= i < len(_DOCS) for i in indices):
        _fail(f"rerank index outside 0..{len(_DOCS) - 1}: {indices}")
    scores = [item.score for item in result.items]
    if not all(math.isfinite(s) for s in scores):
        _fail("rerank scores must be finite")
    if any(a < b for a, b in zip(scores, scores[1:], strict=False)):
        _fail(f"rerank scores must be sorted descending: {scores}")


async def check_rerank_top_n(model: Any) -> None:
    full = await model.rerank("cat", _DOCS)
    result = await model.rerank("cat", _DOCS, top_n=2)
    expected = min(2, len(_DOCS))
    if len(result.items) != expected:
        _fail(f"top_n=2 must return exactly {expected} items, got {len(result.items)}")
    got = [item.index for item in result.items]
    want = [item.index for item in full.items][:expected]
    if got != want:
        _fail(f"top_n=2 indices {got} are not the first items of the full ranking {want}")


async def check_rerank_usage(model: Any) -> None:
    result = await model.rerank("cat", _DOCS)
    _check_usage(result.usage, model.profile, Operation.RERANK)


RERANK_CHECKS: tuple[Callable[[Any], Awaitable[None]], ...] = (
    check_rerank_kind,
    check_rerank_order,
    check_rerank_top_n,
    check_rerank_usage,
)

# --- registry -----------------------------------------------------------------------------


async def check_registry_binding(registry: ProviderRegistry, pid: str) -> None:
    profile = registry.profile(pid)
    getter = {
        ModelKind.CHAT: registry.chat,
        ModelKind.EMBEDDING: registry.embedding,
        ModelKind.RERANKER: registry.reranker,
    }[profile.kind]
    model = getter(pid)
    if model.profile != profile:
        _fail(f"bound adapter profile for '{pid}' differs from the registered profile")


async def check_registry_validates_own_capabilities(registry: ProviderRegistry, pid: str) -> None:
    profile = registry.profile(pid)
    declared = frozenset(cap for cap in Capability if profile.declared(cap) is True)
    registry.validate_requirements({"self": StageRequirement(profile_id=pid, required=declared)})


REGISTRY_CHECKS: tuple[Callable[[ProviderRegistry, str], Awaitable[None]], ...] = (
    check_registry_binding,
    check_registry_validates_own_capabilities,
)
