"""Retry and per-attempt deadline wrappers for the chat and embedding ports (ADR-0001, P2.5).

Adapters make one attempt each; retries live only here. Defaults are the owner decisions
D1-D4 recorded in ADR-0001 "Update 2026-11-04 (P2.5)". Sleep and random source are injected so
tests never wait.
"""

import asyncio
import random
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rag_foundry_providers.capabilities import ModelProfile
from rag_foundry_providers.errors import RETRYABLE_REASONS, ProviderCallError
from rag_foundry_providers.messages import ChatChunk, ChatRequest, ChatResult, EmbeddingResult
from rag_foundry_providers.ports import ChatModel, EmbeddingModel

# ADR-0001 P2.5, owner decisions 2026-11-04.
DEFAULT_MAX_ATTEMPTS = 3  # D1: 1 call + 2 retries
DEFAULT_BACKOFF_MIN_S = 20.0  # D2: uniform random wait between min and max
DEFAULT_BACKOFF_MAX_S = 30.0
DEFAULT_MAX_RETRY_AFTER_S = 60.0  # D2: a larger Retry-After fails at once
DEFAULT_ATTEMPT_TIMEOUT_S = 300.0  # D4

Sleep = Callable[[float], Awaitable[None]]


class RetryPolicy(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    max_attempts: int = Field(DEFAULT_MAX_ATTEMPTS, ge=1)
    backoff_min_s: float = Field(DEFAULT_BACKOFF_MIN_S, ge=0)
    backoff_max_s: float = Field(DEFAULT_BACKOFF_MAX_S, ge=0)
    max_retry_after_s: float = Field(DEFAULT_MAX_RETRY_AFTER_S, ge=0)
    attempt_timeout_s: float = Field(DEFAULT_ATTEMPT_TIMEOUT_S, gt=0)
    retry_timeouts_for_local: bool = True

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.backoff_max_s < self.backoff_min_s:
            raise ValueError("backoff_max_s must be >= backoff_min_s")
        return self


class _Retrier:
    """Decides whether and how long to wait after a failed attempt."""

    def __init__(self, profile: ModelProfile, policy: RetryPolicy, sleep: Sleep, rng: random.Random | None) -> None:
        self._profile = profile
        self._policy = policy
        self._sleep = sleep
        self._rng = rng if rng is not None else random.Random()

    def _delay(self, error: ProviderCallError, attempt: int) -> float | None:
        """Seconds to wait before the next attempt, or None to raise the error now."""
        if attempt >= self._policy.max_attempts:
            return None
        retryable = error.reason in RETRYABLE_REASONS or (
            error.reason == "timeout"
            and self._policy.retry_timeouts_for_local
            and self._profile.capabilities.local is True
        )
        if not retryable:
            return None
        jitter = self._rng.uniform(self._policy.backoff_min_s, self._policy.backoff_max_s)
        if error.retry_after_s is None:
            return jitter
        if error.retry_after_s > self._policy.max_retry_after_s:
            return None
        return max(error.retry_after_s, jitter)

    async def wait_or_raise(self, error: ProviderCallError, attempt: int) -> None:
        delay = self._delay(error, attempt)
        if delay is None:
            raise error
        await self._sleep(delay)


class _End:
    """Marks the end of an inner stream."""


_END = _End()


async def _next(it: AsyncIterator[ChatChunk]) -> ChatChunk | _End:
    try:
        return await anext(it)
    except StopAsyncIteration:
        return _END


class ResilientChatModel:
    """Wraps a ChatModel with bounded retries and a hard per-attempt deadline."""

    def __init__(
        self, model: ChatModel, policy: RetryPolicy, *, sleep: Sleep = asyncio.sleep, rng: random.Random | None = None
    ) -> None:
        self._model = model
        self._policy = policy
        self._retrier = _Retrier(model.profile, policy, sleep, rng)

    @property
    def profile(self) -> ModelProfile:
        return self._model.profile

    async def complete(self, request: ChatRequest) -> ChatResult:
        for attempt in range(1, self._policy.max_attempts + 1):
            error: ProviderCallError
            try:
                async with asyncio.timeout(self._policy.attempt_timeout_s):
                    return await self._model.complete(request)
            except TimeoutError:
                error = ProviderCallError(self.profile.id, "timeout")
            except ProviderCallError as exc:
                error = exc
            # Outside the except block so a timeout carries no cause or context.
            await self._retrier.wait_or_raise(error, attempt)
        raise AssertionError("unreachable")  # pragma: no cover

    async def stream(self, request: ChatRequest) -> AsyncIterator[ChatChunk]:
        loop = asyncio.get_running_loop()
        for attempt in range(1, self._policy.max_attempts + 1):
            deadline = loop.time() + self._policy.attempt_timeout_s
            inner = self._model.stream(request)
            yielded = False
            error: ProviderCallError | None = None
            try:
                while True:
                    remaining = deadline - loop.time()
                    try:
                        if remaining <= 0:
                            raise TimeoutError
                        chunk = await asyncio.wait_for(_next(inner), remaining)
                    except TimeoutError:
                        err = ProviderCallError(self.profile.id, "timeout")
                        if yielded:
                            raise err from None
                        error = err
                        break
                    except ProviderCallError as exc:
                        if yielded:
                            raise
                        error = exc
                        break
                    if isinstance(chunk, _End):
                        return
                    yielded = True
                    yield chunk
            finally:
                await inner.aclose()  # type: ignore[attr-defined]
            if error is not None:
                await self._retrier.wait_or_raise(error, attempt)


class ResilientEmbeddingModel:
    """Wraps an EmbeddingModel with bounded retries and a hard per-attempt deadline."""

    def __init__(
        self,
        model: EmbeddingModel,
        policy: RetryPolicy,
        *,
        sleep: Sleep = asyncio.sleep,
        rng: random.Random | None = None,
    ) -> None:
        self._model = model
        self._policy = policy
        self._retrier = _Retrier(model.profile, policy, sleep, rng)

    @property
    def profile(self) -> ModelProfile:
        return self._model.profile

    async def embed(
        self, texts: Sequence[str], *, input_type: Literal["document", "query"] = "document"
    ) -> EmbeddingResult:
        for attempt in range(1, self._policy.max_attempts + 1):
            error: ProviderCallError
            try:
                async with asyncio.timeout(self._policy.attempt_timeout_s):
                    return await self._model.embed(texts, input_type=input_type)
            except TimeoutError:
                error = ProviderCallError(self.profile.id, "timeout")
            except ProviderCallError as exc:
                error = exc
            await self._retrier.wait_or_raise(error, attempt)
        raise AssertionError("unreachable")  # pragma: no cover
