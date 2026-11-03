"""Opt-in PAID live tests against Anthropic (P2.3-05). Skipped unless RAG_LIVE_ANTHROPIC=1 and ANTHROPIC_API_KEY is set.

Uses claude-haiku-5-5 with tiny prompts and a hard per-run spend cap (SPEND_CAP_USD) enforced by RunBudget: before
every call the worst-case cost (prompt-size upper bound plus the full output allowance) is reserved and the call is
refused with BudgetExceededError if it would pass the cap. The model is the same Resilient(Budgeted(Recording(...)))
stack that build_registry binds (retries limited to 2 attempts here). About 5 to 8 calls in total. The key is never
printed.

Run (PowerShell):
  $k = Read-Host -AsSecureString 'Anthropic key'  # hidden prompt, not saved in history
  $env:ANTHROPIC_API_KEY = [System.Net.NetworkCredential]::new('', $k).Password; $env:RAG_LIVE_ANTHROPIC = '1'
  uv run pytest tests/live/test_anthropic_live.py -m live -v -s
"""

import asyncio
import os
from decimal import Decimal
from typing import Any

import pytest
from rag_foundry_providers.adapters.anthropic import AnthropicChatModel, anthropic_haiku_profile
from rag_foundry_providers.budget import BudgetedChatModel, RunBudget
from rag_foundry_providers.contract_checks import CHAT_CHECKS
from rag_foundry_providers.ledger import RecordingChatModel, UsageLedger
from rag_foundry_providers.messages import ChatMessage, ChatRequest
from rag_foundry_providers.resilience import ResilientChatModel, RetryPolicy
from rag_foundry_providers.usage import TokenSource, Usage

pytestmark = pytest.mark.live

SPEND_CAP_USD = Decimal("0.05")
MAX_OUTPUT_TOKENS = 64
SHORT_OUTPUT_TOKENS = 32
TIMEOUT_S = 60.0
_EN = "Reply with one short greeting."
_AR = "ردّ بتحية قصيرة واحدة."

_LEDGER = UsageLedger()


def _check_usage(usage: Usage) -> None:
    assert usage.token_source is TokenSource.PROVIDER, "token counts must come from the provider"
    assert usage.estimated_cost_usd is not None, "cost must be known"
    assert usage.estimated_cost_usd > 0, "cost must be > 0 for a paid model"
    print(
        f"[live] {usage.operation.value}: in={usage.input_tokens} out={usage.output_tokens} "
        f"cache_read={usage.cache_read_input_tokens} cache_write={usage.cache_write_input_tokens} "
        f"cost_usd={usage.estimated_cost_usd} latency_ms={usage.latency_ms:.0f}"
    )


@pytest.fixture(scope="module")
def capped_model() -> ResilientChatModel:
    if os.environ.get("RAG_LIVE_ANTHROPIC") != "1":
        pytest.skip("RAG_LIVE_ANTHROPIC=1 is not set; live paid Anthropic tests are opt-in")
    if not os.environ.get("ANTHROPIC_API_KEY", "").strip():
        pytest.skip("ANTHROPIC_API_KEY is not set (presence only is checked; the value is never shown)")
    model = AnthropicChatModel(anthropic_haiku_profile(), default_max_tokens=MAX_OUTPUT_TOKENS, timeout_s=TIMEOUT_S)
    budgeted = BudgetedChatModel(
        RecordingChatModel(model, _LEDGER), RunBudget(SPEND_CAP_USD), default_max_output_tokens=MAX_OUTPUT_TOKENS
    )
    return ResilientChatModel(budgeted, RetryPolicy(max_attempts=2))


@pytest.mark.parametrize("check", CHAT_CHECKS, ids=lambda c: c.__name__)
def test_chat_contract(capped_model: ResilientChatModel, check: Any) -> None:
    asyncio.run(check(capped_model))


@pytest.mark.parametrize(("lang", "prompt"), [("english", _EN), ("arabic", _AR)], ids=["english", "arabic"])
def test_chat_text(capped_model: ResilientChatModel, lang: str, prompt: str) -> None:
    request = ChatRequest(messages=(ChatMessage(role="user", content=prompt),), max_output_tokens=SHORT_OUTPUT_TOKENS)
    result = asyncio.run(capped_model.complete(request))
    assert result.text.strip()
    assert result.finish_reason in ("stop", "length")
    print(f"[live] chat {lang}: finish_reason={result.finish_reason} chars={len(result.text)}")


def test_ledger_total_within_cap(capped_model: ResilientChatModel) -> None:
    """Keep last: prints the evidence summary and asserts the cap."""
    totals = _LEDGER.totals()
    print(
        f"[live] SUMMARY model={capped_model.profile.model} calls={totals.calls} "
        f"input_tokens={totals.input_tokens} output_tokens={totals.output_tokens} "
        f"cache_read={totals.cache_read_input_tokens} cache_write={totals.cache_write_input_tokens} "
        f"total_cost_usd={totals.cost_usd} cap_usd={SPEND_CAP_USD}"
    )
    for usage in _LEDGER.records():
        _check_usage(usage)
    assert totals.calls > 0
    assert totals.unknown_cost_calls == 0
    assert totals.estimated_token_calls == 0
    assert totals.cost_usd is not None
    assert totals.cost_usd <= SPEND_CAP_USD
