"""Opt-in PAID live tests against Anthropic (P2.3-05). Skipped unless RAG_LIVE_ANTHROPIC=1 and ANTHROPIC_API_KEY is set.

Uses claude-haiku-5-5 with tiny prompts and a hard per-run spend cap (SPEND_CAP_USD). Before every call the
worst-case cost (prompt-size upper bound plus the full output allowance) is added to the ledger total; the run
stops if that would exceed the cap. About 5 to 8 calls in total. The key is never printed.

Run (PowerShell):
  $k = Read-Host -AsSecureString 'Anthropic key'  # hidden prompt, not saved in history
  $env:ANTHROPIC_API_KEY = [System.Net.NetworkCredential]::new('', $k).Password; $env:RAG_LIVE_ANTHROPIC = '1'
  uv run pytest tests/live/test_anthropic_live.py -m live -v -s
"""

import asyncio
import os
from collections.abc import AsyncIterator
from decimal import Decimal
from typing import Any

import pytest
from rag_foundry_providers.adapters.anthropic import AnthropicChatModel, anthropic_haiku_profile
from rag_foundry_providers.capabilities import ModelProfile
from rag_foundry_providers.contract_checks import CHAT_CHECKS
from rag_foundry_providers.ledger import RecordingChatModel, UsageLedger
from rag_foundry_providers.messages import ChatChunk, ChatMessage, ChatRequest, ChatResult
from rag_foundry_providers.usage import TokenSource, Usage

pytestmark = pytest.mark.live

SPEND_CAP_USD = Decimal("0.05")
MAX_OUTPUT_TOKENS = 64
SHORT_OUTPUT_TOKENS = 32
TIMEOUT_S = 60.0
_EN = "Reply with one short greeting."
_AR = "ردّ بتحية قصيرة واحدة."

_LEDGER = UsageLedger()


def _worst_case_cost(profile: ModelProfile, request: ChatRequest, default_max: int) -> Decimal:
    """Upper bound: UTF-8 bytes of all content (>= tokens) plus 64 for framing, and the full output allowance."""
    in_rate, out_rate = profile.cost_per_mtok_in, profile.cost_per_mtok_out
    if in_rate is None or out_rate is None:
        pytest.fail("profile has no verified rates; cannot enforce the spend cap")
    prompt_tokens = sum(len(m.content.encode("utf-8")) for m in request.messages) + 64
    out_tokens = request.max_output_tokens or default_max
    return (Decimal(prompt_tokens) * in_rate + Decimal(out_tokens) * out_rate) / Decimal(1_000_000)


def _guard(profile: ModelProfile, request: ChatRequest) -> None:
    totals = _LEDGER.totals()
    if totals.cost_usd is None:
        pytest.fail("a recorded call has unknown cost; stopping the paid run")
    worst = _worst_case_cost(profile, request, MAX_OUTPUT_TOKENS)
    if totals.cost_usd + worst > SPEND_CAP_USD:
        pytest.fail(f"spend cap: spent {totals.cost_usd} + worst case {worst} would exceed {SPEND_CAP_USD} USD")


def _check_usage(usage: Usage) -> None:
    assert usage.token_source is TokenSource.PROVIDER, "token counts must come from the provider"
    assert usage.estimated_cost_usd is not None, "cost must be known"
    assert usage.estimated_cost_usd > 0, "cost must be > 0 for a paid model"
    print(
        f"[live] {usage.operation.value}: in={usage.input_tokens} out={usage.output_tokens} "
        f"cache_read={usage.cache_read_input_tokens} cache_write={usage.cache_write_input_tokens} "
        f"cost_usd={usage.estimated_cost_usd} latency_ms={usage.latency_ms:.0f}"
    )


class _CappedModel:
    """Checks the spend cap before each call and the usage after it. Delegates recording to the inner model."""

    def __init__(self, inner: RecordingChatModel) -> None:
        self._inner = inner

    @property
    def profile(self) -> ModelProfile:
        return self._inner.profile

    async def complete(self, request: ChatRequest) -> ChatResult:
        _guard(self.profile, request)
        result = await self._inner.complete(request)
        _check_usage(result.usage)
        return result

    async def stream(self, request: ChatRequest) -> AsyncIterator[ChatChunk]:
        _guard(self.profile, request)
        async for chunk in self._inner.stream(request):
            if chunk.usage is not None:
                _check_usage(chunk.usage)
            yield chunk


@pytest.fixture(scope="module")
def capped_model() -> _CappedModel:
    if os.environ.get("RAG_LIVE_ANTHROPIC") != "1":
        pytest.skip("RAG_LIVE_ANTHROPIC=1 is not set; live paid Anthropic tests are opt-in")
    if not os.environ.get("ANTHROPIC_API_KEY", "").strip():
        pytest.skip("ANTHROPIC_API_KEY is not set (presence only is checked; the value is never shown)")
    model = AnthropicChatModel(anthropic_haiku_profile(), default_max_tokens=MAX_OUTPUT_TOKENS, timeout_s=TIMEOUT_S)
    return _CappedModel(RecordingChatModel(model, _LEDGER))


@pytest.mark.parametrize("check", CHAT_CHECKS, ids=lambda c: c.__name__)
def test_chat_contract(capped_model: _CappedModel, check: Any) -> None:
    asyncio.run(check(capped_model))


@pytest.mark.parametrize(("lang", "prompt"), [("english", _EN), ("arabic", _AR)], ids=["english", "arabic"])
def test_chat_text(capped_model: _CappedModel, lang: str, prompt: str) -> None:
    request = ChatRequest(messages=(ChatMessage(role="user", content=prompt),), max_output_tokens=SHORT_OUTPUT_TOKENS)
    result = asyncio.run(capped_model.complete(request))
    assert result.text.strip()
    assert result.finish_reason in ("stop", "length")
    print(f"[live] chat {lang}: finish_reason={result.finish_reason} chars={len(result.text)}")


def test_ledger_total_within_cap(capped_model: _CappedModel) -> None:
    """Keep last: prints the evidence summary and asserts the cap."""
    totals = _LEDGER.totals()
    print(
        f"[live] SUMMARY model={capped_model.profile.model} calls={totals.calls} "
        f"input_tokens={totals.input_tokens} output_tokens={totals.output_tokens} "
        f"cache_read={totals.cache_read_input_tokens} cache_write={totals.cache_write_input_tokens} "
        f"total_cost_usd={totals.cost_usd} cap_usd={SPEND_CAP_USD}"
    )
    assert totals.calls > 0
    assert totals.unknown_cost_calls == 0
    assert totals.estimated_token_calls == 0
    assert totals.cost_usd is not None
    assert totals.cost_usd <= SPEND_CAP_USD
