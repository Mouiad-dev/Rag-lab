"""Async-only model ports (ADR-0001). Retries and timeouts belong to adapters (P2.5)."""

from collections.abc import AsyncIterator, Sequence
from typing import Literal, Protocol, runtime_checkable

from rag_foundry_providers.capabilities import Capability, ModelProfile
from rag_foundry_providers.errors import CapabilityProblem, MissingCapabilityError
from rag_foundry_providers.messages import (
    ChatChunk,
    ChatRequest,
    ChatResult,
    EmbeddingResult,
    RerankResult,
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
    problems = tuple(
        CapabilityProblem(stage=None, profile_id=profile.id, capability=cap.value, declared=profile.declared(cap))
        for cap in needed
        if profile.declared(cap) is not True
    )
    if problems:
        raise MissingCapabilityError(problems)
