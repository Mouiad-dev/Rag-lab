"""Per-run hard budget with worst-case reservation (ADR-0001 P2.5, D5-D9).

Before every attempt the budget reserves the worst-case cost and refuses if spent + reserved + worst
case would pass the cap. After the call it settles to the real cost. Money is Decimal only; the
budget is in memory for one run.
"""

import json
import threading
from collections.abc import AsyncIterator, Sequence
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict

from rag_foundry_providers.capabilities import ModelProfile
from rag_foundry_providers.errors import (
    BudgetError,
    BudgetExceededError,
    ProviderCallError,
    ProviderError,
    UnboundedCostError,
)
from rag_foundry_providers.messages import ChatChunk, ChatRequest, ChatResult, EmbeddingResult
from rag_foundry_providers.ports import ChatModel, EmbeddingModel
from rag_foundry_providers.usage import Usage

_MTOK = Decimal(1_000_000)
_FRAMING_TOKENS = 64
_ZERO = Decimal(0)


class BudgetSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    spent_usd: Decimal
    reserved_usd: Decimal
    calls: int
    unsettled_calls: int


class Reservation:
    """One reserved call. Settles at most once."""

    def __init__(self, profile_id: str, usd: Decimal) -> None:
        self.profile_id = profile_id
        self.usd = usd
        self._settled = False


class RunBudget:
    def __init__(self, max_usd: Decimal, max_calls: int | None = None) -> None:
        if not isinstance(max_usd, Decimal) or not max_usd.is_finite() or max_usd < 0:
            raise ValueError("max_usd must be a finite Decimal >= 0 (floats are not accepted)")
        if max_calls is not None and (isinstance(max_calls, bool) or not isinstance(max_calls, int) or max_calls < 0):
            raise ValueError("max_calls must be an int >= 0 or None")
        self.max_usd = max_usd
        self.max_calls = max_calls
        self._lock = threading.Lock()
        self._spent = _ZERO
        self._reserved = _ZERO
        self._calls = 0
        self._unsettled = 0

    def reserve(self, profile: ModelProfile, worst: Decimal) -> Reservation:
        """Reserve before any I/O. Local profiles reserve 0 USD (D6) but count a call."""
        if not isinstance(worst, Decimal) or not worst.is_finite() or worst < 0:
            raise ValueError("worst must be a finite Decimal >= 0")
        usd = _ZERO if profile.capabilities.local is True else worst
        with self._lock:
            if self.max_calls is not None and self._calls + 1 > self.max_calls:
                raise BudgetExceededError(profile.id, "calls", self.max_calls, self._calls, 0, 1)
            if usd > 0 and self._spent + self._reserved + usd > self.max_usd:
                raise BudgetExceededError(profile.id, "usd", self.max_usd, self._spent, self._reserved, usd)
            self._reserved += usd
            self._calls += 1
        return Reservation(profile.id, usd)

    def release(self, reservation: Reservation) -> None:
        """Give back the reservation and its call slot (the call failed validation before any I/O)."""
        with self._lock:
            if reservation._settled:
                return
            reservation._settled = True
            self._reserved -= reservation.usd
            self._calls -= 1

    def settle(self, reservation: Reservation, usage: Usage | None, *, unbilled: bool = False) -> None:
        """Replace the reservation with the real cost; unknown cost keeps the full reservation as spent.

        `unbilled=True` (a reply was received that carried an HTTP status and no usage, ADR-0001 D7/D8)
        settles to 0 USD. The call still counts toward max_calls.
        """
        with self._lock:
            if reservation._settled:
                return
            reservation._settled = True
            self._reserved -= reservation.usd
            if unbilled and usage is None:
                return
            cost = usage.estimated_cost_usd if usage is not None else None
            if cost is None:
                self._spent += reservation.usd
                self._unsettled += 1
            else:
                self._spent += cost

    def snapshot(self) -> BudgetSnapshot:
        with self._lock:
            return BudgetSnapshot(
                spent_usd=self._spent,
                reserved_usd=self._reserved,
                calls=self._calls,
                unsettled_calls=self._unsettled,
            )


def _need(profile: ModelProfile, rate: Decimal | None, name: str) -> Decimal:
    if rate is None:
        raise UnboundedCostError(profile.id, name)
    return rate


def _input_rate(profile: ModelProfile) -> Decimal:
    rate = _need(profile, profile.cost_per_mtok_in, "cost_per_mtok_in")
    write = profile.cost_per_mtok_cache_write
    return max(rate, write) if write is not None else rate


def _json_bytes(value: object) -> int:
    return len(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8"))


def worst_case_chat_cost(profile: ModelProfile, request: ChatRequest, default_max_output_tokens: int | None) -> Decimal:
    """Upper bound of one chat attempt (D9). Local profiles are 0. Paid profiles need both rates and an output cap."""
    if profile.capabilities.local is True:
        return _ZERO
    in_rate = _input_rate(profile)
    out_rate = _need(profile, profile.cost_per_mtok_out, "cost_per_mtok_out")
    out_tokens = request.max_output_tokens if request.max_output_tokens is not None else default_max_output_tokens
    if out_tokens is None:
        raise UnboundedCostError(profile.id, "output token cap")
    in_tokens = sum(len(m.content.encode("utf-8")) for m in request.messages) + _FRAMING_TOKENS
    for tool in request.tools:
        in_tokens += _json_bytes(tool.model_dump(mode="json"))
    if request.response_schema is not None:
        in_tokens += _json_bytes(request.response_schema)
    return (Decimal(in_tokens) * in_rate + Decimal(out_tokens) * out_rate) / _MTOK


def worst_case_embed_cost(profile: ModelProfile, texts: Sequence[str]) -> Decimal:
    """Upper bound of one embedding attempt: input only."""
    if profile.capabilities.local is True:
        return _ZERO
    in_rate = _input_rate(profile)
    in_tokens = sum(len(t.encode("utf-8")) for t in texts) + _FRAMING_TOKENS
    return Decimal(in_tokens) * in_rate / _MTOK


def _finish(
    budget: "RunBudget",
    reservation: Reservation,
    usage: Usage | None,
    error: BaseException | None,
    *,
    streamed: bool = False,
) -> None:
    """Settle exactly once from how the call ended.

    - Billed usage (result, final chunk, or error.usage): real cost.
    - ProviderCallError with a non-2xx HTTP status and no usage (an error reply, not billed): 0 USD. A 2xx
      reply that failed to parse may have been billed, so it keeps the worst case.
    - ProviderError that is not a ProviderCallError (request validation, raised before any I/O): the
      reservation and the call slot are released. Adapters must therefore raise every failure that happens
      after a request is sent as a ProviderCallError (see ports.py).
    - Anything else (connection error, timeout, cancellation, early stream stop, unknown exception): the
      full worst case stays spent. A stream that already yielded a chunk without usage also keeps it.
    """
    if usage is not None:
        budget.settle(reservation, usage)
    elif streamed:
        budget.settle(reservation, None)  # output may already have been generated: keep the worst case
    elif isinstance(error, ProviderCallError):
        budget.settle(reservation, None, unbilled=error.status is not None and not 200 <= error.status < 300)
    elif isinstance(error, ProviderError) and not isinstance(error, BudgetError):
        budget.release(reservation)
    else:
        budget.settle(reservation, None)


class BudgetedChatModel:
    """Reserves the worst case before each call and settles exactly once, however the call ends."""

    def __init__(self, model: ChatModel, budget: RunBudget, *, default_max_output_tokens: int | None) -> None:
        self._model = model
        self._budget = budget
        self._default_max = default_max_output_tokens

    @property
    def profile(self) -> ModelProfile:
        return self._model.profile

    async def complete(self, request: ChatRequest) -> ChatResult:
        worst = worst_case_chat_cost(self.profile, request, self._default_max)
        reservation = self._budget.reserve(self.profile, worst)
        usage: Usage | None = None
        error: BaseException | None = None
        try:
            result = await self._model.complete(request)
            usage = result.usage
            return result
        except BaseException as exc:
            error = exc
            if isinstance(exc, ProviderCallError):
                usage = exc.usage
            raise
        finally:
            _finish(self._budget, reservation, usage, error)

    async def stream(self, request: ChatRequest) -> AsyncIterator[ChatChunk]:
        worst = worst_case_chat_cost(self.profile, request, self._default_max)
        reservation = self._budget.reserve(self.profile, worst)
        usage: Usage | None = None
        error: BaseException | None = None
        streamed = False
        try:
            async for chunk in self._model.stream(request):
                streamed = True
                if chunk.usage is not None:
                    usage = chunk.usage
                yield chunk
        except BaseException as exc:
            error = exc
            if isinstance(exc, ProviderCallError) and usage is None:
                usage = exc.usage
            raise
        finally:
            _finish(self._budget, reservation, usage, error, streamed=streamed)


class BudgetedEmbeddingModel:
    def __init__(self, model: EmbeddingModel, budget: RunBudget) -> None:
        self._model = model
        self._budget = budget

    @property
    def profile(self) -> ModelProfile:
        return self._model.profile

    async def embed(
        self, texts: Sequence[str], *, input_type: Literal["document", "query"] = "document"
    ) -> EmbeddingResult:
        worst = worst_case_embed_cost(self.profile, texts)
        reservation = self._budget.reserve(self.profile, worst)
        usage: Usage | None = None
        error: BaseException | None = None
        try:
            result = await self._model.embed(texts, input_type=input_type)
            usage = result.usage
            return result
        except BaseException as exc:
            error = exc
            if isinstance(exc, ProviderCallError):
                usage = exc.usage
            raise
        finally:
            _finish(self._budget, reservation, usage, error)
