"""Deterministic fake chat, embedding and reranker models for tests and offline use.

No network, no keys, no randomness, no wall-clock time. Token counts are
`len(text.split())`, a documented approximation and not a tokenizer, so every Usage
built here is marked `TokenSource.ESTIMATED`.
"""

import hashlib
import math
import re
import struct
from collections.abc import AsyncIterator, Sequence
from decimal import Decimal
from typing import Literal

from rag_foundry_providers.capabilities import Capabilities, ModelKind, ModelProfile
from rag_foundry_providers.messages import (
    ChatChunk,
    ChatRequest,
    ChatResult,
    EmbeddingResult,
    RerankItem,
    RerankResult,
)
from rag_foundry_providers.ports import check_chat_request
from rag_foundry_providers.usage import (
    Clock,
    Operation,
    TokenSource,
    Usage,
    estimate_cost,
    measure_latency_ms,
)

_ZERO = Decimal("0")
_PIECES = re.compile(r"\S+|\s+")


class StepClock:
    """Deterministic clock: each call advances by `step_s` seconds (integer ticks)."""

    def __init__(self, step_s: float = 0.001) -> None:
        self._step_s = step_s
        self._ticks = 0

    def __call__(self) -> float:
        self._ticks += 1
        return self._ticks * self._step_s


def fake_chat_profile(id: str = "fake-chat", *, streaming: bool = True) -> ModelProfile:  # noqa: A002
    return ModelProfile(
        id=id,
        provider="fake",
        model="fake-chat-v1",
        kind=ModelKind.CHAT,
        capabilities=Capabilities(streaming=streaming, tool_calling=False, structured_output=False, local=True),
        cost_per_mtok_in=_ZERO,
        cost_per_mtok_out=_ZERO,
    )


def fake_embedding_profile(id: str = "fake-embed", *, dim: int = 16) -> ModelProfile:  # noqa: A002
    return ModelProfile(
        id=id,
        provider="fake",
        model="fake-embedding-v1",
        kind=ModelKind.EMBEDDING,
        capabilities=Capabilities(local=True, embedding_dim=dim),
        cost_per_mtok_in=_ZERO,
        cost_per_mtok_out=_ZERO,
    )


def fake_reranker_profile(id: str = "fake-rerank") -> ModelProfile:  # noqa: A002
    return ModelProfile(
        id=id,
        provider="fake",
        model="fake-reranker-v1",
        kind=ModelKind.RERANKER,
        capabilities=Capabilities(local=True),
        cost_per_mtok_in=_ZERO,
        cost_per_mtok_out=_ZERO,
    )


def _tokens(text: str) -> int:
    return len(text.split())


def _usage(
    profile: ModelProfile, op: Operation, input_tokens: int, output_tokens: int, clock: Clock, start: float
) -> Usage:
    # Latency is rounded so float noise from the clock subtraction never leaks into outputs.
    return Usage(
        profile_id=profile.id,
        provider=profile.provider,
        model=profile.model,
        operation=op,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        token_source=TokenSource.ESTIMATED,
        latency_ms=round(measure_latency_ms(clock, start), 6),
        estimated_cost_usd=estimate_cost(profile, input_tokens, output_tokens),
    )


class FakeChatModel:
    def __init__(self, profile: ModelProfile, *, clock: Clock | None = None) -> None:
        self.profile = profile
        self._clock: Clock = clock if clock is not None else StepClock()

    @staticmethod
    def _text(request: ChatRequest) -> str:
        last_user = ""
        for message in request.messages:
            if message.role == "user":
                last_user = message.content
        return f"[fake] {last_user}"

    @staticmethod
    def _input_tokens(request: ChatRequest) -> int:
        return sum(_tokens(m.content) for m in request.messages)

    async def complete(self, request: ChatRequest) -> ChatResult:
        check_chat_request(self.profile, request, streaming=False)
        start = self._clock()
        text = self._text(request)
        usage = _usage(self.profile, Operation.CHAT, self._input_tokens(request), _tokens(text), self._clock, start)
        return ChatResult(text=text, finish_reason="stop", usage=usage)

    async def stream(self, request: ChatRequest) -> AsyncIterator[ChatChunk]:
        check_chat_request(self.profile, request, streaming=True)
        start = self._clock()
        text = self._text(request)
        pieces = _PIECES.findall(text)
        for piece in pieces[:-1]:
            yield ChatChunk(delta=piece)
        usage = _usage(
            self.profile, Operation.CHAT_STREAM, self._input_tokens(request), _tokens(text), self._clock, start
        )
        yield ChatChunk(delta=pieces[-1], usage=usage)


class FakeEmbeddingModel:
    def __init__(self, profile: ModelProfile, *, clock: Clock | None = None) -> None:
        self.profile = profile
        self._clock: Clock = clock if clock is not None else StepClock()

    def _vector(self, text: str, input_type: str) -> tuple[float, ...]:
        dim = self.profile.capabilities.embedding_dim
        assert dim is not None  # guaranteed by ModelProfile validation for embedding kind
        seed = f"{self.profile.model}\x00{input_type}\x00{text}".encode()
        raw = b""
        counter = 0
        while len(raw) < dim * 4:
            raw += hashlib.sha256(seed + struct.pack(">I", counter)).digest()
            counter += 1
        values = struct.unpack(f">{dim}I", raw[: dim * 4])
        floats = [v / 0xFFFFFFFF * 2.0 - 1.0 for v in values]
        norm = math.sqrt(sum(x * x for x in floats))
        return tuple(x / norm for x in floats)

    async def embed(
        self, texts: Sequence[str], *, input_type: Literal["document", "query"] = "document"
    ) -> EmbeddingResult:
        start = self._clock()
        vectors = tuple(self._vector(t, input_type) for t in texts)
        usage = _usage(self.profile, Operation.EMBED, sum(_tokens(t) for t in texts), 0, self._clock, start)
        return EmbeddingResult(vectors=vectors, usage=usage)


class FakeReranker:
    def __init__(self, profile: ModelProfile, *, clock: Clock | None = None) -> None:
        self.profile = profile
        self._clock: Clock = clock if clock is not None else StepClock()

    async def rerank(self, query: str, documents: Sequence[str], *, top_n: int | None = None) -> RerankResult:
        start = self._clock()
        q = set(query.casefold().split())
        scored: list[RerankItem] = []
        for index, doc in enumerate(documents):
            d = set(doc.casefold().split())
            union = q | d
            score = len(q & d) / len(union) if union else 0.0
            scored.append(RerankItem(index=index, score=score))
        scored.sort(key=lambda item: (-item.score, item.index))
        if top_n is not None:
            if top_n < 0:
                raise ValueError("top_n must be >= 0")
            scored = scored[:top_n]
        input_tokens = _tokens(query) + sum(_tokens(d) for d in documents)
        usage = _usage(self.profile, Operation.RERANK, input_tokens, 0, self._clock, start)
        return RerankResult(items=tuple(scored), usage=usage)
