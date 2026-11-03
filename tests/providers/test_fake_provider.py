"""Deterministic fake chat, embedding and reranker."""

import asyncio
import json
import math
import os
import subprocess
import sys
from collections.abc import Awaitable
from decimal import Decimal
from typing import Any

import pytest
from rag_foundry_providers import (
    ChatMessage,
    ChatModel,
    ChatRequest,
    EmbeddingModel,
    MissingCapabilityError,
    Operation,
    Reranker,
    TokenSource,
    ToolSpec,
    Usage,
)
from rag_foundry_providers.fake import (
    FakeChatModel,
    FakeEmbeddingModel,
    FakeReranker,
    StepClock,
    fake_chat_profile,
    fake_embedding_profile,
    fake_reranker_profile,
)

ENGLISH = "hello world"
ARABIC = "مرحبا بالعالم"

# First 3 floats of the default (dim=16, document) vectors; stable across processes and OSes.
GOLDEN_FIRST3: list[list[float]] = [
    [0.39173099179110815, 0.2016804496616919, 0.06233076791026497],
    [0.2180295413355717, 0.4025897956066819, -0.09392155977851394],
]


def run[T](coro: Awaitable[T]) -> T:
    async def _wrap() -> T:
        return await coro

    return asyncio.run(_wrap())


def req(text: str = "one two  three", **kw: Any) -> ChatRequest:
    return ChatRequest(messages=(ChatMessage(role="user", content=text),), **kw)


async def collect(model: FakeChatModel, request: ChatRequest) -> list[Any]:
    return [c async for c in model.stream(request)]


def test_ports_satisfied() -> None:
    assert isinstance(FakeChatModel(fake_chat_profile()), ChatModel)
    assert isinstance(FakeEmbeddingModel(fake_embedding_profile()), EmbeddingModel)
    assert isinstance(FakeReranker(fake_reranker_profile()), Reranker)


def test_two_instances_identical() -> None:
    a, b = FakeChatModel(fake_chat_profile()), FakeChatModel(fake_chat_profile())
    assert run(a.complete(req())) == run(b.complete(req()))
    ea, eb = FakeEmbeddingModel(fake_embedding_profile()), FakeEmbeddingModel(fake_embedding_profile())
    assert run(ea.embed(["x", "y"])) == run(eb.embed(["x", "y"]))
    ra, rb = FakeReranker(fake_reranker_profile()), FakeReranker(fake_reranker_profile())
    assert run(ra.rerank("a b", ["a", "b c"])) == run(rb.rerank("a b", ["a", "b c"]))


def test_chat_complete_fields() -> None:
    result = run(FakeChatModel(fake_chat_profile()).complete(req("one two  three")))
    assert result.text == "[fake] one two  three"
    assert result.finish_reason == "stop"
    assert result.usage.operation is Operation.CHAT
    assert result.usage.input_tokens == 3
    assert result.usage.output_tokens == 4


def test_usage_matches_profile_latency_and_cost() -> None:
    profile = fake_chat_profile()
    usage = run(FakeChatModel(profile).complete(req())).usage
    assert (usage.profile_id, usage.provider, usage.model) == (profile.id, "fake", "fake-chat-v1")
    assert usage.latency_ms == 1.0
    assert usage.estimated_cost_usd == Decimal("0")
    assert profile.capabilities.local is True
    assert profile.capabilities.tool_calling is False
    assert profile.capabilities.structured_output is False


def test_custom_clock_controls_latency() -> None:
    usage = run(FakeChatModel(fake_chat_profile(), clock=StepClock(step_s=0.005)).complete(req())).usage
    assert usage.latency_ms == 5.0


def test_stream_matches_complete_and_only_last_has_usage() -> None:
    model = FakeChatModel(fake_chat_profile())
    request = req("one two  three\tfour")
    chunks = run(collect(model, request))
    assert "".join(c.delta for c in chunks) == run(model.complete(request)).text
    assert all(c.usage is None for c in chunks[:-1])
    assert chunks[-1].usage is not None
    assert chunks[-1].usage.operation is Operation.CHAT_STREAM
    assert len(chunks) > 1


def test_token_source_is_estimated_everywhere() -> None:
    usages: list[Usage] = []
    usages.append(run(FakeChatModel(fake_chat_profile()).complete(req())).usage)
    last = run(collect(FakeChatModel(fake_chat_profile()), req()))[-1].usage
    assert last is not None
    usages.append(last)
    usages.append(run(FakeEmbeddingModel(fake_embedding_profile()).embed(["a b"])).usage)
    usages.append(run(FakeEmbeddingModel(fake_embedding_profile()).embed([])).usage)
    usages.append(run(FakeReranker(fake_reranker_profile()).rerank("a", ["a", "b"])).usage)
    assert {u.operation for u in usages} == {Operation.CHAT, Operation.CHAT_STREAM, Operation.EMBED, Operation.RERANK}
    assert all(u.token_source is TokenSource.ESTIMATED for u in usages)


def test_tools_and_schema_refused() -> None:
    model = FakeChatModel(fake_chat_profile())
    tool = ToolSpec(name="t", description="d", parameters_schema={})
    with pytest.raises(MissingCapabilityError):
        run(model.complete(req(tools=(tool,))))
    with pytest.raises(MissingCapabilityError):
        run(model.complete(req(response_schema={"type": "object"})))


def test_streaming_false_refuses_stream_but_completes() -> None:
    model = FakeChatModel(fake_chat_profile(streaming=False))
    with pytest.raises(MissingCapabilityError):
        run(collect(model, req()))
    assert run(model.complete(req())).text.startswith("[fake]")


def test_embedding_shape_norm_and_input_type() -> None:
    model = FakeEmbeddingModel(fake_embedding_profile(dim=40))  # needs two SHA-256 blocks
    result = run(model.embed(["alpha beta", ARABIC]))
    assert len(result.vectors) == 2
    for vec in result.vectors:
        assert len(vec) == 40
        assert abs(math.sqrt(sum(x * x for x in vec)) - 1.0) < 1e-9
    assert result.usage.input_tokens == 4
    assert result.usage.output_tokens == 0
    doc = run(model.embed(["alpha beta"], input_type="document")).vectors[0]
    query = run(model.embed(["alpha beta"], input_type="query")).vectors[0]
    assert doc != query


def test_embedding_empty_input() -> None:
    result = run(FakeEmbeddingModel(fake_embedding_profile()).embed([]))
    assert result.vectors == ()
    assert result.usage.input_tokens == 0
    assert result.usage.output_tokens == 0


_SNIPPET = (
    "import asyncio, json, sys\n"
    "from rag_foundry_providers.fake import FakeEmbeddingModel, fake_embedding_profile\n"
    "m = FakeEmbeddingModel(fake_embedding_profile())\n"
    "r = asyncio.run(m.embed(json.loads(sys.argv[1])))\n"
    "print(json.dumps([list(v) for v in r.vectors]))\n"
)


def test_embeddings_stable_across_hash_seed_and_golden() -> None:
    outputs = []
    for seed in ("0", "12345"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        proc = subprocess.run(
            [sys.executable, "-c", _SNIPPET, json.dumps([ENGLISH, ARABIC])],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        outputs.append(json.loads(proc.stdout))
    assert outputs[0] == outputs[1]
    local = run(FakeEmbeddingModel(fake_embedding_profile()).embed([ENGLISH, ARABIC]))
    assert [list(v) for v in local.vectors] == outputs[0]
    for got, want in zip((v[:3] for v in outputs[0]), GOLDEN_FIRST3, strict=True):
        assert got == pytest.approx(want, abs=1e-12)


def test_rerank_order_ties_and_top_n() -> None:
    reranker = FakeReranker(fake_reranker_profile())
    docs = ["red fox", "BLUE sky", "Red Fox jumps", "red fox", "nothing"]
    result = run(reranker.rerank("red fox", docs))
    assert [i.index for i in result.items] == [0, 3, 2, 1, 4]
    assert result.items[0].score == 1.0
    assert result.items[2].score == pytest.approx(2 / 3)
    assert result.items[3].score == result.items[4].score == 0.0
    top = run(reranker.rerank("red fox", docs, top_n=2))
    assert [i.index for i in top.items] == [0, 3]
    assert result.usage.operation is Operation.RERANK
    with pytest.raises(ValueError):
        run(reranker.rerank("red fox", docs, top_n=-1))
    assert result.usage.input_tokens == 2 + sum(len(d.split()) for d in docs)
    assert run(reranker.rerank("x", [])).items == ()
