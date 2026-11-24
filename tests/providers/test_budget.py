"""RunBudget reservation, settlement and the budgeted wrappers (no network)."""

import asyncio
from decimal import Decimal

import pytest
from rag_foundry_providers import (
    BudgetedChatModel,
    BudgetedEmbeddingModel,
    BudgetError,
    BudgetExceededError,
    BudgetRequiredError,
    ChatChunk,
    ChatMessage,
    ChatRequest,
    ChatResult,
    EmbeddingResult,
    FakeChatModel,
    FakeEmbeddingModel,
    RecordingEmbeddingModel,
    RunBudget,
    UnboundedCostError,
    UsageLedger,
    fake_chat_profile,
    fake_embedding_profile,
    worst_case_chat_cost,
)
from rag_foundry_providers.adapters.anthropic import UnsupportedRequestError
from rag_foundry_providers.capabilities import Capabilities, ModelKind, ModelProfile
from rag_foundry_providers.contract_checks import CHAT_CHECKS, DETERMINISTIC_CHAT_CHECKS, EMBEDDING_CHECKS
from rag_foundry_providers.errors import CapabilityProblem, MissingCapabilityError, ProviderCallError
from rag_foundry_providers.usage import Operation, TokenSource, Usage

REQ = ChatRequest(messages=(ChatMessage(role="user", content="hello world"),), max_output_tokens=64)
_PROBLEM = (CapabilityProblem(stage=None, profile_id="paid", capability="tool_calling", declared=False),)
RATE = Decimal("10")  # USD per MTok, in and out


def _paid(kind=ModelKind.CHAT, rate_in=RATE, rate_out=RATE, pid="paid") -> ModelProfile:
    caps = Capabilities(streaming=True, tool_calling=False, structured_output=False, local=False)
    if kind is ModelKind.EMBEDDING:
        caps = Capabilities(local=False, embedding_dim=4)
    return ModelProfile(
        id=pid,
        provider="test",
        model="m",
        kind=kind,
        capabilities=caps,
        cost_per_mtok_in=rate_in,
        cost_per_mtok_out=rate_out,
    )


def _usage(profile, cost, op=Operation.CHAT) -> Usage:
    return Usage(
        profile_id=profile.id,
        provider=profile.provider,
        model=profile.model,
        operation=op,
        input_tokens=1,
        output_tokens=1,
        token_source=TokenSource.PROVIDER,
        latency_ms=0.0,
        estimated_cost_usd=cost,
    )


class Scripted:
    """A chat model returning a fixed cost and counting calls."""

    def __init__(self, profile, cost=Decimal("0.00001"), error=None, gate=None):
        self.profile = profile
        self.cost = cost
        self.error = error
        self.gate = gate  # an asyncio.Event the call waits on, so calls overlap
        self.calls = 0

    async def complete(self, request):
        self.calls += 1
        if self.gate is not None:
            await self.gate.wait()
        if self.error:
            raise self.error
        return ChatResult(text="ok", finish_reason="stop", usage=_usage(self.profile, self.cost))

    async def stream(self, request):
        self.calls += 1
        yield ChatChunk(delta="a")
        yield ChatChunk(delta="b")
        yield ChatChunk(delta="c", usage=_usage(self.profile, self.cost, Operation.CHAT_STREAM))


def _worst(profile=None):
    return worst_case_chat_cost(profile or _paid(), REQ, None)


def test_money_types():
    with pytest.raises(ValueError):
        RunBudget(0.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        RunBudget(Decimal("-1"))
    with pytest.raises(ValueError):
        RunBudget(Decimal("NaN"))
    assert RunBudget(Decimal("0")).snapshot().spent_usd == 0


def test_errors_are_budget_errors_not_call_errors():
    assert issubclass(BudgetExceededError, BudgetError)
    assert not issubclass(BudgetError, ProviderCallError)
    assert "paid" in str(BudgetRequiredError(("paid",)))


def test_cap_stops_calls_and_inner_not_called():
    profile = _paid()
    worst = _worst(profile)
    inner = Scripted(profile, cost=worst)  # each call really costs the worst case
    cap = worst * 2 + worst / 2
    model = BudgetedChatModel(inner, RunBudget(cap), default_max_output_tokens=None)
    asyncio.run(model.complete(REQ))
    asyncio.run(model.complete(REQ))
    with pytest.raises(BudgetExceededError) as info:
        asyncio.run(model.complete(REQ))
    assert info.value.limit == "usd"
    assert inner.calls == 2


def test_done_cap_of_one_tenth_of_a_cent():
    profile = _paid(rate_in=Decimal(1), rate_out=Decimal(1))
    inner = Scripted(profile)
    budget = RunBudget(Decimal("0.001"))
    model = BudgetedChatModel(inner, budget, default_max_output_tokens=None)
    n = 0
    with pytest.raises(BudgetExceededError):
        for _ in range(10_000):
            asyncio.run(model.complete(REQ))
            n += 1
    assert n >= 1
    assert budget.snapshot().spent_usd + _worst(profile) > Decimal("0.001")
    assert budget.snapshot().spent_usd <= Decimal("0.001")
    assert inner.calls == n


def test_max_calls_overrun():
    inner = Scripted(fake_chat_profile())
    model = BudgetedChatModel(inner, RunBudget(Decimal("0"), max_calls=2), default_max_output_tokens=None)
    asyncio.run(model.complete(REQ))
    asyncio.run(model.complete(REQ))
    with pytest.raises(BudgetExceededError) as info:
        asyncio.run(model.complete(REQ))
    assert info.value.limit == "calls"
    assert inner.calls == 2


def test_unknown_rate_refused_before_call():
    for profile in (_paid(rate_in=None), _paid(rate_out=None)):
        inner = Scripted(profile)
        model = BudgetedChatModel(inner, RunBudget(Decimal("1")), default_max_output_tokens=None)
        with pytest.raises(UnboundedCostError):
            asyncio.run(model.complete(REQ))
        assert inner.calls == 0
    no_cap = ChatRequest(messages=REQ.messages)
    with pytest.raises(UnboundedCostError):
        worst_case_chat_cost(_paid(), no_cap, None)
    assert worst_case_chat_cost(_paid(), no_cap, 100) > 0


def test_local_profile_never_blocked_by_usd():
    inner = FakeChatModel(fake_chat_profile())
    budget = RunBudget(Decimal("0"))
    model = BudgetedChatModel(inner, budget, default_max_output_tokens=None)
    for _ in range(3):
        asyncio.run(model.complete(REQ))
    assert budget.snapshot().calls == 3
    assert budget.snapshot().spent_usd == 0


def test_settle_frees_difference_and_unknown_cost_keeps_reservation():
    profile = _paid()
    worst = _worst(profile)
    budget = RunBudget(Decimal("1"))
    asyncio.run(
        BudgetedChatModel(Scripted(profile, cost=Decimal("0.000001")), budget, default_max_output_tokens=None).complete(
            REQ
        )
    )
    snap = budget.snapshot()
    assert snap.spent_usd == Decimal("0.000001")
    assert snap.reserved_usd == 0
    assert snap.unsettled_calls == 0
    asyncio.run(BudgetedChatModel(Scripted(profile, cost=None), budget, default_max_output_tokens=None).complete(REQ))
    snap = budget.snapshot()
    assert snap.spent_usd == Decimal("0.000001") + worst
    assert snap.unsettled_calls == 1


def test_error_without_usage_keeps_reservation_and_error_propagates():
    profile = _paid()
    budget = RunBudget(Decimal("1"))
    err = ProviderCallError("paid", "server_error")
    model = BudgetedChatModel(Scripted(profile, error=err), budget, default_max_output_tokens=None)
    with pytest.raises(ProviderCallError):
        asyncio.run(model.complete(REQ))
    assert budget.snapshot().unsettled_calls == 1
    assert budget.snapshot().reserved_usd == 0


def test_error_with_billed_usage_settles_to_that_cost():
    profile = _paid()
    budget = RunBudget(Decimal("1"))
    err = ProviderCallError("paid", "refused", usage=_usage(profile, Decimal("0.000002")))
    model = BudgetedChatModel(Scripted(profile, error=err), budget, default_max_output_tokens=None)
    with pytest.raises(ProviderCallError):
        asyncio.run(model.complete(REQ))
    assert budget.snapshot().spent_usd == Decimal("0.000002")


def test_stream_stopped_early_keeps_reservation_and_ledger_unchanged():
    profile = _paid()
    ledger = UsageLedger()
    from rag_foundry_providers import RecordingChatModel

    budget = RunBudget(Decimal("1"))
    model = BudgetedChatModel(RecordingChatModel(Scripted(profile), ledger), budget, default_max_output_tokens=None)

    async def run():
        gen = model.stream(REQ)
        await anext(gen)
        await gen.aclose()

    asyncio.run(run())
    snap = budget.snapshot()
    assert snap.unsettled_calls == 1
    assert snap.spent_usd == _worst(profile)
    assert snap.reserved_usd == 0
    assert ledger.totals().calls == 0


def test_stream_full_settles_to_real_cost():
    profile = _paid()
    budget = RunBudget(Decimal("1"))
    model = BudgetedChatModel(Scripted(profile), budget, default_max_output_tokens=None)

    async def run():
        return [c async for c in model.stream(REQ)]

    asyncio.run(run())
    assert budget.snapshot().spent_usd == Decimal("0.00001")
    assert budget.snapshot().unsettled_calls == 0


def test_arabic_bound_covers_utf8_bytes():
    text = "مرحبا بالعالم " * 20
    req = ChatRequest(messages=(ChatMessage(role="user", content=text),), max_output_tokens=1)
    profile = _paid(rate_out=Decimal("0"))
    worst = worst_case_chat_cost(profile, req, None)
    assert worst >= Decimal(len(text.encode("utf-8"))) * RATE / Decimal(1_000_000)


def test_ten_concurrent_overlapping_calls_never_pass_the_cap():
    profile = _paid()
    worst = _worst(profile)
    cap = worst * 3 + worst / 2  # room for exactly three reservations at once

    async def run():
        gate = asyncio.Event()
        inner = Scripted(profile, cost=worst, gate=gate)
        budget = RunBudget(cap)
        model = BudgetedChatModel(inner, budget, default_max_output_tokens=None)
        tasks = [asyncio.create_task(model.complete(REQ)) for _ in range(10)]
        for _ in range(5):
            await asyncio.sleep(0)  # let every task reach the reservation or the gate
        waiting = budget.snapshot()
        gate.set()
        outcomes = await asyncio.gather(*tasks, return_exceptions=True)
        return inner, budget, waiting, outcomes

    inner, budget, waiting, outcomes = asyncio.run(run())
    ok = [o for o in outcomes if isinstance(o, ChatResult)]
    refused = [o for o in outcomes if isinstance(o, BudgetExceededError)]
    assert len(ok) == 3
    assert len(refused) == 7
    assert inner.calls == 3
    assert waiting.reserved_usd == worst * 3
    assert budget.snapshot().spent_usd <= cap
    assert budget.snapshot().reserved_usd == 0


def test_cancellation_keeps_the_worst_case():
    profile = _paid()
    worst = _worst(profile)
    budget = RunBudget(Decimal("1"))

    async def run():
        inner = Scripted(profile, gate=asyncio.Event())  # never set: hangs
        model = BudgetedChatModel(inner, budget, default_max_output_tokens=None)
        async with asyncio.timeout(0.01):
            await model.complete(REQ)

    with pytest.raises(TimeoutError):
        asyncio.run(run())
    snap = budget.snapshot()
    assert snap.reserved_usd == 0
    assert snap.spent_usd == worst
    assert snap.unsettled_calls == 1


def test_task_cancel_keeps_the_worst_case():
    profile = _paid()
    worst = _worst(profile)
    budget = RunBudget(Decimal("1"))

    async def run():
        model = BudgetedChatModel(Scripted(profile, gate=asyncio.Event()), budget, default_max_output_tokens=None)
        task = asyncio.create_task(model.complete(REQ))
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())
    assert budget.snapshot().spent_usd == worst
    assert budget.snapshot().unsettled_calls == 1


@pytest.mark.parametrize("reason", ["server_error", "overloaded", "rate_limited"])
def test_reply_with_status_and_no_usage_settles_to_zero(reason):
    profile = _paid()
    budget = RunBudget(Decimal("1"))
    err = ProviderCallError("paid", reason, status=503)
    model = BudgetedChatModel(Scripted(profile, error=err), budget, default_max_output_tokens=None)
    with pytest.raises(ProviderCallError) as info:
        asyncio.run(model.complete(REQ))
    assert info.value is err
    snap = budget.snapshot()
    assert snap.spent_usd == 0
    assert snap.reserved_usd == 0
    assert snap.unsettled_calls == 0
    assert snap.calls == 1  # still counts toward max_calls


@pytest.mark.parametrize("status", [200, 204, 299])
def test_2xx_reply_without_usage_keeps_the_worst_case(status):
    profile = _paid()
    budget = RunBudget(Decimal("1"))
    err = ProviderCallError("paid", "bad_response", status=status)
    model = BudgetedChatModel(Scripted(profile, error=err), budget, default_max_output_tokens=None)
    with pytest.raises(ProviderCallError):
        asyncio.run(model.complete(REQ))
    assert budget.snapshot().spent_usd == _worst(profile)
    assert budget.snapshot().unsettled_calls == 1


@pytest.mark.parametrize("reason", ["connection", "timeout"])
def test_error_without_status_keeps_the_worst_case(reason):
    profile = _paid()
    budget = RunBudget(Decimal("1"))
    model = BudgetedChatModel(
        Scripted(profile, error=ProviderCallError("paid", reason)), budget, default_max_output_tokens=None
    )
    with pytest.raises(ProviderCallError):
        asyncio.run(model.complete(REQ))
    assert budget.snapshot().spent_usd == _worst(profile)
    assert budget.snapshot().unsettled_calls == 1


def test_stream_error_after_a_chunk_keeps_the_worst_case():
    profile = _paid()
    budget = RunBudget(Decimal("1"))

    class Broken(Scripted):
        async def stream(self, request):
            yield ChatChunk(delta="a")
            raise ProviderCallError("paid", "server_error", status=200)

    model = BudgetedChatModel(Broken(profile), budget, default_max_output_tokens=None)

    async def run():
        return [c async for c in model.stream(REQ)]

    with pytest.raises(ProviderCallError):
        asyncio.run(run())
    assert budget.snapshot().spent_usd == _worst(profile)


def test_stream_status_error_before_first_chunk_settles_to_zero():
    profile = _paid()
    budget = RunBudget(Decimal("1"))

    class Early(Scripted):
        async def stream(self, request):
            raise ProviderCallError("paid", "overloaded", status=529)
            yield  # pragma: no cover

    model = BudgetedChatModel(Early(profile), budget, default_max_output_tokens=None)

    async def run():
        return [c async for c in model.stream(REQ)]

    with pytest.raises(ProviderCallError):
        asyncio.run(run())
    assert budget.snapshot().spent_usd == 0
    assert budget.snapshot().unsettled_calls == 0


@pytest.mark.parametrize(
    "exc_factory", [lambda: UnsupportedRequestError("x"), lambda: MissingCapabilityError(_PROBLEM)]
)
def test_validation_error_before_io_burns_no_budget_and_no_call_slot(exc_factory):
    """Design chosen: the wrapper releases the reservation and the call slot for a ProviderError that is not a
    ProviderCallError (request validation runs inside the adapter, before any I/O), instead of duplicating
    validation before reserving."""
    profile = _paid()
    budget = RunBudget(Decimal("1"), max_calls=1)
    inner = Scripted(profile, error=exc_factory())
    model = BudgetedChatModel(inner, budget, default_max_output_tokens=None)
    for _ in range(3):  # would exhaust max_calls=1 if a slot were burned
        with pytest.raises(type(inner.error)):
            asyncio.run(model.complete(REQ))
    snap = budget.snapshot()
    assert (snap.spent_usd, snap.reserved_usd, snap.calls, snap.unsettled_calls) == (0, 0, 0, 0)


def test_validation_error_in_stream_releases_the_slot():
    profile = _paid()
    budget = RunBudget(Decimal("1"), max_calls=1)

    class Bad(Scripted):
        async def stream(self, request):
            raise UnsupportedRequestError("x")
            yield  # pragma: no cover

    model = BudgetedChatModel(Bad(profile), budget, default_max_output_tokens=None)

    async def run():
        return [c async for c in model.stream(REQ)]

    with pytest.raises(UnsupportedRequestError):
        asyncio.run(run())
    assert budget.snapshot().calls == 0


def test_recording_embedding_records_once_per_call():
    ledger = UsageLedger()
    model = RecordingEmbeddingModel(FakeEmbeddingModel(fake_embedding_profile()), ledger)
    asyncio.run(model.embed(["a", "b"]))
    assert ledger.totals().calls == 1
    asyncio.run(model.embed([]))
    assert ledger.totals().calls == 2


def test_budgeted_embedding_paid_cap():
    profile = _paid(ModelKind.EMBEDDING)

    class Emb:
        def __init__(self):
            self.profile = profile
            self.calls = 0

        async def embed(self, texts, *, input_type="document"):
            self.calls += 1
            return EmbeddingResult(
                vectors=tuple((0.0,) * 4 for _ in texts), usage=_usage(profile, Decimal("0.000001"), Operation.EMBED)
            )

    inner = Emb()
    model = BudgetedEmbeddingModel(inner, RunBudget(Decimal("0.0000001")))
    with pytest.raises(BudgetExceededError):
        asyncio.run(model.embed(["hello"]))
    assert inner.calls == 0


@pytest.mark.parametrize("check", CHAT_CHECKS + DETERMINISTIC_CHAT_CHECKS, ids=lambda c: c.__name__)
def test_chat_contract_through_budgeted(check):
    model = BudgetedChatModel(
        FakeChatModel(fake_chat_profile()), RunBudget(Decimal("0")), default_max_output_tokens=None
    )
    asyncio.run(check(model))


@pytest.mark.parametrize("check", EMBEDDING_CHECKS, ids=lambda c: c.__name__)
def test_embedding_contract_through_budgeted(check):
    model = BudgetedEmbeddingModel(FakeEmbeddingModel(fake_embedding_profile()), RunBudget(Decimal("0")))
    asyncio.run(check(model))
