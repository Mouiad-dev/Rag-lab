"""Opt-in PAID live R49 prefix probe against Anthropic Claude Haiku 5.5 (P7.4-03a).

One paid run, not an evaluation. Six streamed calls (arms warm, changed, eviction_attempt) with
`cache_breakpoint="system"` (5-minute ephemeral cache_control on the system block). Before any call the
worst-case total (every input byte billed at the dearer of input and 5-minute write rates, plus the full
output allowance) must not exceed USD 0.05, and the chat is wrapped in RunBudget(0.05, max_calls=6).
Skipped unless RAG_LIVE_ANTHROPIC_R49=1 and ANTHROPIC_API_KEY is present (presence only; never printed).

Run (PowerShell); the key is set for this one command and removed after:
  $k = Read-Host -AsSecureString 'Anthropic key'  # hidden prompt, not saved in history
  $env:ANTHROPIC_API_KEY = [System.Net.NetworkCredential]::new('', $k).Password; $env:RAG_LIVE_ANTHROPIC_R49 = '1'
  uv run pytest tests/live/test_prefix_anthropic_live.py -m live -v -s -p no:cacheprovider
  Remove-Item Env:ANTHROPIC_API_KEY, Env:RAG_LIVE_ANTHROPIC_R49 -ErrorAction SilentlyContinue; Remove-Variable k
"""

import asyncio
import os
import time
from decimal import Decimal

import pytest
from rag_foundry_cache.prefix import PrefixCacheProbe, PrefixProbeConfig
from rag_foundry_providers.adapters._ollama_http import estimate_tokens
from rag_foundry_providers.adapters.anthropic import AnthropicChatModel, anthropic_haiku_profile
from rag_foundry_providers.budget import BudgetedChatModel, RunBudget, worst_case_chat_cost
from rag_foundry_providers.ledger import RecordingChatModel, UsageLedger
from rag_foundry_providers.messages import ChatMessage, ChatRequest

pytestmark = pytest.mark.live

SPEND_CAP_USD = Decimal("0.05")
MAX_CALLS = 6
MAX_OUTPUT_TOKENS = 64
MIN_CACHEABLE_TOKENS = 512  # Claude Haiku 5.5 minimum cacheable prompt length
TIMEOUT_S = 60.0
QUESTIONS = ("What is the harbour tower used for?", "Which season is the market busiest?")

_PARAGRAPH = (
    "The harbour district keeps a ledger of every vessel that enters the bay. The tower at the end of the "
    "eastern pier guides ships at night, and the market on the old quay is busiest in late summer when the "
    "fishing fleets return. Clerks record cargo, crew and weather, and the archive stores one folder per year. "
)


def _prefix_text() -> str:
    parts = ["Reference notes about the harbour district."]
    parts += [f"Note {i}. {_PARAGRAPH}" for i in range(1, 31)]
    return "\n".join(parts)


def test_prefix_probe_anthropic_cache_breakpoint() -> None:
    if os.environ.get("RAG_LIVE_ANTHROPIC_R49") != "1":
        pytest.skip("RAG_LIVE_ANTHROPIC_R49=1 is not set; this paid live test is opt-in")
    if not os.environ.get("ANTHROPIC_API_KEY", "").strip():
        pytest.skip("ANTHROPIC_API_KEY is not set (presence only is checked; the value is never shown)")
    prefix = _prefix_text()
    if estimate_tokens(prefix) < MIN_CACHEABLE_TOKENS:
        pytest.skip("cache not applicable: prefix is below the minimum cacheable length")

    profile = anthropic_haiku_profile()
    probe_request = ChatRequest(
        messages=(
            ChatMessage(role="system", content=prefix),
            ChatMessage(role="user", content=max(QUESTIONS, key=len)),
        ),
        max_output_tokens=MAX_OUTPUT_TOKENS,
        cache_breakpoint="system",
    )
    worst_total = worst_case_chat_cost(profile, probe_request, MAX_OUTPUT_TOKENS) * MAX_CALLS
    print(f"[live] worst-case total USD {worst_total} (cap {SPEND_CAP_USD})")
    assert worst_total <= SPEND_CAP_USD, "worst-case cost is above the cap; refusing to make any call"

    ledger = UsageLedger()
    model = AnthropicChatModel(profile, default_max_tokens=MAX_OUTPUT_TOKENS, timeout_s=TIMEOUT_S)
    budget = RunBudget(SPEND_CAP_USD, max_calls=MAX_CALLS)
    chat = BudgetedChatModel(
        RecordingChatModel(model, ledger),
        budget,
        default_max_output_tokens=MAX_OUTPUT_TOKENS,
    )
    config = PrefixProbeConfig(
        prefix_text=prefix,
        questions=QUESTIONS,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        mode="stream",
        arms=("warm", "changed", "eviction_attempt"),
        cache_breakpoint="system",
        temperature=None,
    )
    report = asyncio.run(PrefixCacheProbe(config).run(chat, clock=time.perf_counter, salt_factory=lambda: "unused"))

    usages = ledger.records()
    for i, record in enumerate(report.records):
        cost = usages[i].estimated_cost_usd if i < len(usages) else None
        print(
            f"[live] arm={record.arm} label={record.label} cache_read={record.cache_read_input_tokens} "
            f"cache_write={record.cache_write_input_tokens} in={record.input_tokens} out={record.output_tokens} "
            f"ttft_ms={record.ttft_ms and round(record.ttft_ms)} cost_usd={'None' if cost is None else cost}"
        )
    snap = budget.snapshot()
    print(
        f"[live] budget spent_usd={snap.spent_usd} unsettled_calls={snap.unsettled_calls} "
        f"cache_metric={report.cache_metric} model={profile.model}"
    )
    assert len(report.records) == MAX_CALLS
    assert snap.spent_usd <= SPEND_CAP_USD
