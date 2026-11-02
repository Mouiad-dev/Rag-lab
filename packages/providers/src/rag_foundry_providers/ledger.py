"""In-memory append-only usage ledger and a recording wrapper for chat models."""

import threading
from collections.abc import AsyncIterator
from decimal import Decimal
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from rag_foundry_providers.capabilities import ModelProfile
from rag_foundry_providers.errors import ProviderCallError
from rag_foundry_providers.messages import ChatChunk, ChatRequest, ChatResult
from rag_foundry_providers.ports import ChatModel
from rag_foundry_providers.usage import TokenSource, Usage


class UsageSink(Protocol):
    """Receives one Usage per model call. A later step can persist through it."""

    def append(self, usage: Usage) -> None: ...


class UsageTotals(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    calls: int
    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int
    cache_write_input_tokens: int
    cost_usd: Decimal | None
    """None when any matching record has an unknown cost."""
    unknown_cost_calls: int
    estimated_token_calls: int


class UsageLedger(UsageSink):
    """Append-only, thread-safe. No remove or clear by design."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: list[Usage] = []

    def append(self, usage: Usage) -> None:
        with self._lock:
            self._records.append(usage)

    def records(self, *, profile_id: str | None = None, operation: str | None = None) -> tuple[Usage, ...]:
        with self._lock:
            snapshot = tuple(self._records)
        return tuple(
            u
            for u in snapshot
            if (profile_id is None or u.profile_id == profile_id) and (operation is None or u.operation == operation)
        )

    def totals(self, *, profile_id: str | None = None, operation: str | None = None) -> UsageTotals:
        recs = self.records(profile_id=profile_id, operation=operation)
        unknown = sum(1 for u in recs if u.estimated_cost_usd is None)
        cost: Decimal | None = None
        if unknown == 0:
            cost = sum((u.estimated_cost_usd for u in recs if u.estimated_cost_usd is not None), Decimal(0))
        return UsageTotals(
            calls=len(recs),
            input_tokens=sum(u.input_tokens for u in recs),
            output_tokens=sum(u.output_tokens for u in recs),
            cache_read_input_tokens=sum(u.cache_read_input_tokens for u in recs),
            cache_write_input_tokens=sum(u.cache_write_input_tokens for u in recs),
            cost_usd=cost,
            unknown_cost_calls=unknown,
            estimated_token_calls=sum(1 for u in recs if u.token_source is TokenSource.ESTIMATED),
        )


class RecordingChatModel:
    """Wraps a ChatModel and appends each successful call's usage to a sink.

    Errors propagate unchanged. An error records nothing unless it carries billed usage
    (`ProviderCallError.usage`, for example a refusal), which is recorded once. A stream records when its final
    usage chunk is reached; stopping early records nothing.
    """

    def __init__(self, model: ChatModel, sink: UsageSink) -> None:
        self._model = model
        self._sink = sink

    @property
    def profile(self) -> ModelProfile:
        return self._model.profile

    async def complete(self, request: ChatRequest) -> ChatResult:
        try:
            result = await self._model.complete(request)
        except ProviderCallError as exc:
            if exc.usage is not None:  # a billed reply that failed, for example a refusal
                self._sink.append(exc.usage)
            raise
        self._sink.append(result.usage)
        return result

    async def stream(self, request: ChatRequest) -> AsyncIterator[ChatChunk]:
        recorded = False  # never record twice, even if an adapter yields usage and then fails
        try:
            async for chunk in self._model.stream(request):
                if chunk.usage is not None and not recorded:
                    self._sink.append(chunk.usage)
                    recorded = True
                yield chunk
        except ProviderCallError as exc:
            if exc.usage is not None and not recorded:
                self._sink.append(exc.usage)
            raise
