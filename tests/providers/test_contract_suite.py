"""One contract suite over fake, Ollama, Anthropic and OpenAI-compatible adapters, bound through build_registry.

Every adapter sits behind Resilient(Budgeted(Recording(adapter))) with httpx.MockTransport replies (no network,
sentinel keys, no real sleeping). The same CHAT_CHECKS and EMBEDDING_CHECKS run for all of them.
"""

import asyncio
import json
import uuid
from decimal import Decimal
from typing import Any

import httpx
import pytest
from rag_foundry_providers import (
    BudgetExceededError,
    BudgetRequiredError,
    ChatMessage,
    ChatRequest,
    ProviderUnavailableError,
    RunBudget,
    UsageLedger,
    fake_chat_profile,
    fake_embedding_profile,
)
from rag_foundry_providers.adapters.anthropic import anthropic_haiku_profile
from rag_foundry_providers.adapters.ollama import ollama_chat_profile
from rag_foundry_providers.config import ProvidersConfig, parse_providers_config
from rag_foundry_providers.contract_checks import CHAT_CHECKS, EMBEDDING_CHECKS
from rag_foundry_providers.pipeline import build_registry

ANTHROPIC_ENV = "TEST_ANTHROPIC_KEY_P25"
OPENAI_ENV = "TEST_OPENAI_KEY_P25"
SENTINEL = f"sk-TEST-SENTINEL-{uuid.uuid4()}"
CAP = Decimal("1")
OLLAMA_DIM = 4
OAI_DIM = 8
OAI_USAGE = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}

CHAT_IDS = ["fake-chat", "ollama-chat", "anthropic-haiku", "oai-chat"]
EMBED_IDS = ["fake-embed", "ollama-embed", "oai-embed"]


def _json_profile(profile: Any) -> dict[str, Any]:
    return profile.model_dump(mode="json")


def _config() -> ProvidersConfig:
    endpoints = [
        {"id": "fake", "adapter": "fake", "local": True},
        {"id": "ollama", "adapter": "ollama", "base_url": "http://127.0.0.1:11434", "local": True},
        {"id": "anthropic", "adapter": "anthropic", "api_key_env": ANTHROPIC_ENV, "local": False},
        {
            "id": "oai",
            "adapter": "openai_compatible",
            "base_url": "https://api.example.test/v1",
            "api_key_env": OPENAI_ENV,
            "local": False,
            "max_tokens_field": "max_completion_tokens",
            "supports_temperature": False,
            "default_max_tokens": 256,
        },
    ]
    paid_caps = {"streaming": True, "tool_calling": False, "structured_output": False}
    profiles = [
        _json_profile(fake_chat_profile("fake-chat")),
        _json_profile(fake_embedding_profile("fake-embed")),
        _json_profile(ollama_chat_profile("ollama-chat")),
        {
            "id": "ollama-embed",
            "provider": "ollama",
            "model": "bge-m3",
            "kind": "embedding",
            "capabilities": {"embedding_dim": OLLAMA_DIM, "local": True},
            "cost_per_mtok_in": "0",
            "cost_per_mtok_out": "0",
        },
        _json_profile(anthropic_haiku_profile("anthropic-haiku")),
        {
            "id": "oai-chat",
            "provider": "oai",
            "model": "chat-m",
            "kind": "chat",
            "capabilities": paid_caps,
            "cost_per_mtok_in": "1",
            "cost_per_mtok_out": "2",
        },
        {
            "id": "oai-embed",
            "provider": "oai",
            "model": "embed-m",
            "kind": "embedding",
            "capabilities": {"embedding_dim": OAI_DIM},
            "cost_per_mtok_in": "1",
            "cost_per_mtok_out": "0",
        },
    ]
    stages = {"generator": "oai-chat", "embedder": "oai-embed"}
    return parse_providers_config({"schema_version": 1, "endpoints": endpoints, "profiles": profiles, "stages": stages})


ALL_STAGES = {
    "generator": "oai-chat",
    "embedder": "oai-embed",
}


def _config_with_stages(stages: dict[str, str]) -> ProvidersConfig:
    base = _config()
    data = base.model_dump(mode="json")
    data["stages"] = stages
    return parse_providers_config(data)


# --- MockTransport replies --------------------------------------------------------------------


def _ollama_handler(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    if request.url.path == "/api/embed":
        n = len(body["input"])
        vectors = [[0.1 * (i + 1)] * OLLAMA_DIM for i in range(n)]
        return httpx.Response(200, json={"model": "bge-m3", "embeddings": vectors, "prompt_eval_count": 7})
    final = {
        "model": "gemma4:e4b",
        "message": {"role": "assistant", "content": ""},
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": 26,
        "eval_count": 3,
    }
    if body.get("stream"):

        def part(text: str) -> dict:
            return {"model": "gemma4:e4b", "message": {"role": "assistant", "content": text}, "done": False}

        lines = [part("Hello"), part(" world"), final]
        return httpx.Response(200, content=("\n".join(json.dumps(o) for o in lines) + "\n").encode())
    final["message"]["content"] = "Hello world"
    return httpx.Response(200, json=final)


def _sse(*events: tuple[str, dict]) -> bytes:
    return "".join(f"event: {n}\ndata: {json.dumps(d, ensure_ascii=False)}\n\n" for n, d in events).encode()


def _anthropic_message(content: list, stop: str | None, usage: dict) -> dict:
    return {
        "id": "msg_01",
        "type": "message",
        "role": "assistant",
        "model": "claude-haiku-5-5",
        "content": content,
        "stop_reason": stop,
        "stop_sequence": None,
        "usage": usage,
    }


def _anthropic_handler(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    if not body.get("stream"):
        msg = _anthropic_message(
            [{"type": "text", "text": "Hello world"}], "end_turn", {"input_tokens": 25, "output_tokens": 6}
        )
        return httpx.Response(200, json=msg)
    start = _anthropic_message([], None, {"input_tokens": 25, "output_tokens": 1})
    events: list[tuple[str, dict]] = [
        ("message_start", {"type": "message_start", "message": start}),
        (
            "content_block_start",
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        ),
    ]
    for t in ("Hello", " world"):
        events.append(
            (
                "content_block_delta",
                {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": t}},
            )
        )
    events += [
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 6},
            },
        ),
        ("message_stop", {"type": "message_stop"}),
    ]
    return httpx.Response(200, content=_sse(*events))


def _oai_chunk(content: str | None = None, finish: str | None = None, delta: dict | None = None) -> dict:
    d = delta if delta is not None else ({} if content is None else {"content": content})
    return {
        "id": "c",
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": d, "finish_reason": finish}],
    }


def _oai_handler(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    if request.url.path.endswith("/embeddings"):
        n = len(body["input"])
        data = [{"object": "embedding", "index": i, "embedding": [0.1 * (i + 1)] * OAI_DIM} for i in range(n)]
        usage = {"prompt_tokens": 7, "total_tokens": 7}
        return httpx.Response(200, json={"object": "list", "model": "embed-m", "data": data, "usage": usage})
    if body.get("stream"):
        events = [
            _oai_chunk(delta={"role": "assistant", "content": ""}),
            _oai_chunk("Hello"),
            _oai_chunk(" world"),
            _oai_chunk(finish="stop"),
            {"id": "c", "object": "chat.completion.chunk", "choices": [], "usage": dict(OAI_USAGE)},
        ]
        text = "".join(f"data: {json.dumps(e)}\n\n" for e in events) + "data: [DONE]\n\n"
        return httpx.Response(200, content=text.encode())
    completion = {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello world"}, "finish_reason": "stop"}],
        "usage": dict(OAI_USAGE),
    }
    return httpx.Response(200, json=completion)


def _transports() -> dict[str, httpx.MockTransport]:
    return {
        "ollama": httpx.MockTransport(_ollama_handler),
        "anthropic": httpx.MockTransport(_anthropic_handler),
        "oai": httpx.MockTransport(_oai_handler),
    }


class NoSleep:
    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.delays.append(seconds)


@pytest.fixture(autouse=True)
def _keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ANTHROPIC_ENV, SENTINEL)
    monkeypatch.setenv(OPENAI_ENV, SENTINEL)


def _stack(budget: RunBudget | None = None, ledger: UsageLedger | None = None):
    # Stage names are a closed set of seven; build_registry binds by profile and does not check kind, so the
    # three embedding profiles borrow the names embedder, reranker and scout here.
    stages = {
        "generator": "fake-chat",
        "query_transform": "ollama-chat",
        "judge": "anthropic-haiku",
        "chunk_llm": "oai-chat",
        "embedder": "fake-embed",
        "reranker": "ollama-embed",
        "scout": "oai-embed",
    }
    cfg = _config_with_stages(stages)
    ledger = ledger if ledger is not None else UsageLedger()
    budget = budget if budget is not None else RunBudget(CAP)
    # The anthropic profile has a long default cap; its worst case at USD 1 is still far below the cap.
    reg = build_registry(
        cfg,
        stages=list(stages),
        transports=_transports(),
        ledger=ledger,
        budget=budget,
        retry_sleep=NoSleep(),
    )
    return reg, ledger, budget


@pytest.mark.parametrize("check", CHAT_CHECKS, ids=lambda c: c.__name__)
@pytest.mark.parametrize("pid", CHAT_IDS)
def test_chat_contract_through_registry(pid: str, check: Any) -> None:
    reg, _, _ = _stack()
    asyncio.run(check(reg.chat(pid)))


@pytest.mark.parametrize("check", EMBEDDING_CHECKS, ids=lambda c: c.__name__)
@pytest.mark.parametrize("pid", EMBED_IDS)
def test_embedding_contract_through_registry(pid: str, check: Any) -> None:
    reg, _, _ = _stack()
    asyncio.run(check(reg.embedding(pid)))


def test_ledger_and_budget_after_full_suite() -> None:
    reg, ledger, budget = _stack()

    async def run() -> None:
        for pid in CHAT_IDS:
            for check in CHAT_CHECKS:
                await check(reg.chat(pid))
        for pid in EMBED_IDS:
            for check in EMBEDDING_CHECKS:
                await check(reg.embedding(pid))

    asyncio.run(run())
    assert ledger.totals().calls > 0
    snap = budget.snapshot()
    assert snap.spent_usd <= CAP
    assert snap.reserved_usd == 0


# --- done tests -------------------------------------------------------------------------------


def _counting(seen: list[httpx.Request], status_first: int | None = None):
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if status_first is not None and len(seen) == 1:
            return httpx.Response(status_first, json={"error": {"type": "server_error", "message": "x"}})
        return _oai_handler(request)

    return httpx.MockTransport(handler)


def _req() -> ChatRequest:
    return ChatRequest(messages=(ChatMessage(role="user", content="hello"),), max_output_tokens=16)


def test_budget_overrun_stops_the_run_before_any_request() -> None:
    seen: list[httpx.Request] = []
    sleep = NoSleep()
    reg = build_registry(
        _config_with_stages({"generator": "oai-chat"}),
        stages=["generator"],
        transports={"oai": _counting(seen)},
        budget=RunBudget(Decimal("0.000001")),
        retry_sleep=sleep,
    )
    with pytest.raises(BudgetExceededError) as info:
        asyncio.run(reg.chat("oai-chat").complete(_req()))
    assert info.value.limit == "usd"
    assert seen == []
    assert sleep.delays == []


def test_503_then_200_is_one_success_with_one_ledger_record() -> None:
    seen: list[httpx.Request] = []
    sleep = NoSleep()
    ledger = UsageLedger()
    reg = build_registry(
        _config_with_stages({"generator": "oai-chat"}),
        stages=["generator"],
        transports={"oai": _counting(seen, status_first=503)},
        ledger=ledger,
        budget=RunBudget(CAP),
        retry_sleep=sleep,
    )
    result = asyncio.run(reg.chat("oai-chat").complete(_req()))
    assert result.text
    assert len(seen) == 2
    assert len(sleep.delays) == 1 and 20 <= sleep.delays[0] <= 30
    assert len(ledger.records()) == 1


def test_non_local_profile_without_budget_is_refused_before_anything_is_built(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(OPENAI_ENV, raising=False)  # a key read would raise MissingCredentialError instead
    seen: list[httpx.Request] = []
    with pytest.raises(BudgetRequiredError) as info:
        build_registry(
            _config_with_stages({"generator": "oai-chat"}),
            stages=["generator"],
            transports={"oai": _counting(seen)},
        )
    assert info.value.profile_ids == ("oai-chat",)
    assert seen == []


def test_local_only_config_builds_without_budget() -> None:
    reg = build_registry(_config_with_stages({"generator": "fake-chat"}), stages=["generator"])
    assert asyncio.run(reg.chat("fake-chat").complete(_req())).text


def test_budget_runs_out_on_attempt_two_raises_budget_error_not_the_503() -> None:
    seen: list[httpx.Request] = []
    sleep = NoSleep()
    # Attempt 1 gets a 503 (status, no usage: settles to 0 but still uses the one call slot).
    # Attempt 2 is refused by the budget before any request, and that error is never retried.
    reg = build_registry(
        _config_with_stages({"generator": "oai-chat"}),
        stages=["generator"],
        transports={"oai": _counting(seen, status_first=503)},
        budget=RunBudget(CAP, max_calls=1),
        retry_sleep=sleep,
    )
    with pytest.raises(BudgetExceededError) as info:
        asyncio.run(reg.chat("oai-chat").complete(_req()))
    assert info.value.limit == "calls"
    assert len(seen) == 1  # attempt 2 never reached the transport; no third attempt
    assert len(sleep.delays) == 1


def test_non_local_reranker_is_unavailable_until_p5_5() -> None:
    data = _config().model_dump(mode="json")
    data["profiles"].append(
        {
            "id": "paid-rerank",
            "provider": "oai",
            "model": "r",
            "kind": "reranker",
            "capabilities": {},
            "cost_per_mtok_in": "1",
            "cost_per_mtok_out": "0",
        }
    )
    data["stages"] = {"reranker": "paid-rerank"}
    cfg = parse_providers_config(data)
    with pytest.raises(ProviderUnavailableError):
        build_registry(cfg, stages=["reranker"], budget=RunBudget(CAP))
