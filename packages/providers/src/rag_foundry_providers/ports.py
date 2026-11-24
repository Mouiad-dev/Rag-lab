"""Async-only model ports (ADR-0001).

Adapters make one HTTP attempt; retries, deadlines and budgets are wrappers (P2.5, resilience.py and
budget.py). Rule for adapters: every failure after a request is sent must be a ProviderCallError; any
other ProviderError means "raised before I/O", and the budget releases its reservation for it.
"""

from collections.abc import AsyncIterator, Sequence
from typing import Literal, Protocol, runtime_checkable

from rag_foundry_providers.capabilities import Capability, ModelProfile
from rag_foundry_providers.errors import CapabilityProblem, MissingCapabilityError
from rag_foundry_providers.messages import (
    ChatChunk,
    ChatRequest,
    ChatResult,
    EmbeddingResult,
    MultivectorResult,
    RawCompletionRequest,
    RawCompletionResult,
    RerankResult,
    SparseEncodingResult,
    TokenEncodingResult,
)


@runtime_checkable
class ChatModel(Protocol):
    """Chat completion port.

    Streaming yields exactly one chunk carrying `usage`, and it is the last chunk.
    Retries and timeouts belong to adapters, not callers or this port.
    """

    profile: ModelProfile

    async def complete(self, request: ChatRequest) -> ChatResult: ...

    def stream(self, request: ChatRequest) -> AsyncIterator[ChatChunk]: ...


@runtime_checkable
class RawCompletionModel(Protocol):
    """Raw prompt completion with provider token log probabilities."""

    profile: ModelProfile

    async def complete_raw(self, request: RawCompletionRequest) -> RawCompletionResult: ...


@runtime_checkable
class EmbeddingModel(Protocol):
    """Embedding port. Empty `texts` returns empty vectors with zero tokens."""

    profile: ModelProfile

    async def embed(
        self, texts: Sequence[str], *, input_type: Literal["document", "query"] = "document"
    ) -> EmbeddingResult: ...


@runtime_checkable
class Reranker(Protocol):
    """Reranking port.

    Items are unique input indices, sorted by score descending with ties broken by
    index ascending, and capped by `top_n`.
    """

    profile: ModelProfile

    async def rerank(self, query: str, documents: Sequence[str], *, top_n: int | None = None) -> RerankResult: ...


@runtime_checkable
class SparseEncoder(Protocol):
    """Learned sparse encoder. Query encoding uses the model's static vocabulary weights."""

    profile: ModelProfile

    async def encode(
        self, texts: Sequence[str], *, input_type: Literal["document", "query"]
    ) -> SparseEncodingResult: ...


@runtime_checkable
class TokenEncoder(Protocol):
    """Per-token vector port (C20 hidden states, R45 ColBERT head).

    Special tokens are excluded from items. Offsets are code-point offsets into each input text.
    `max_tokens` is the longest input (special tokens included) encoded; longer text is truncated
    and flagged `truncated=True`.
    """

    profile: ModelProfile
    max_tokens: int

    async def encode_tokens(
        self,
        texts: Sequence[str],
        *,
        head: Literal["hidden", "colbert"],
        input_type: Literal["document", "query"],
    ) -> TokenEncodingResult: ...


@runtime_checkable
class PageImageEncoder(Protocol):
    """Page-image multivector port (R41, ColPali style).

    `encode_pages` takes PNG bytes and returns one patch multivector per page; `encode_queries` returns one
    token multivector per query. Both live in the same space and need MaxSim scoring. Invalid PNGs raise
    ExtractionError("page_image_invalid") before any decode.
    """

    profile: ModelProfile

    async def encode_pages(self, pngs: Sequence[bytes]) -> MultivectorResult: ...

    async def encode_queries(self, texts: Sequence[str]) -> MultivectorResult: ...


def check_chat_request(profile: ModelProfile, request: ChatRequest, *, streaming: bool) -> None:
    """Raise MissingCapabilityError (stage None) listing every capability the request needs
    but the profile does not declare as True."""
    needed: list[Capability] = []
    if request.tools:
        needed.append(Capability.TOOL_CALLING)
    if request.response_schema is not None:
        needed.append(Capability.STRUCTURED_OUTPUT)
    if streaming:
        needed.append(Capability.STREAMING)
    if request.cache_breakpoint == "system":
        needed.append(Capability.PROMPT_CACHE_CONTROL)
    problems = tuple(
        CapabilityProblem(stage=None, profile_id=profile.id, capability=cap.value, declared=profile.declared(cap))
        for cap in needed
        if profile.declared(cap) is not True
    )
    if problems:
        raise MissingCapabilityError(problems)
