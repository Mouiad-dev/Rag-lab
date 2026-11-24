"""Raw Ollama completion contract tests using an in-memory HTTP transport."""

import asyncio
import json

import httpx
import pytest
from pydantic import ValidationError
from rag_foundry_providers import contract_checks
from rag_foundry_providers.adapters.ollama import (
    OllamaRawCompletionModel,
    ollama_chat_profile,
    ollama_raw_completion_profile,
)
from rag_foundry_providers.errors import MissingCapabilityError, ProviderCallError
from rag_foundry_providers.fake import FakeChatModel, fake_chat_profile
from rag_foundry_providers.messages import RawCompletionRequest
from rag_foundry_providers.usage import TokenSource


def test_raw_completion_sends_raw_request_and_parses_token_logprobs() -> None:
    seen: dict[str, object] = {}

    async def respond(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen.update(json.loads(await request.aread()))
        return httpx.Response(
            200,
            json={
                "response": "[Retrieval] answer",
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 7,
                "eval_count": 2,
                "logprobs": [
                    {
                        "token": "[Retrieval]",
                        "logprob": -0.25,
                        "top_logprobs": [
                            {"token": "[Retrieval]", "logprob": -0.25},
                            {"token": "[No Retrieval]", "logprob": -1.5},
                        ],
                    },
                    {"token": " answer", "logprob": -0.5, "top_logprobs": []},
                ],
            },
        )

    transport = httpx.MockTransport(respond)
    model = OllamaRawCompletionModel(
        ollama_raw_completion_profile(), api_base="http://ollama.test", transport=transport
    )
    request = RawCompletionRequest(
        prompt="### Instruction:\nWhat is water?\n\n### Response:\n",
        max_output_tokens=12,
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        top_logprobs=20,
        stop=("\n",),
    )

    result = asyncio.run(model.complete_raw(request))

    assert seen["path"] == "/api/generate"
    assert seen["raw"] is True
    assert seen["logprobs"] is True
    assert seen["top_logprobs"] == 20
    assert seen["stream"] is False
    assert seen["prompt"] == request.prompt
    assert seen["options"] == {
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "num_predict": 12,
        "stop": ["\n"],
        "num_ctx": 2048,
    }
    assert not {"system", "template", "messages"}.intersection(seen)
    assert result.text == "[Retrieval] answer"
    assert result.tokens[0].token == "[Retrieval]"
    assert result.tokens[0].top == (("[Retrieval]", -0.25), ("[No Retrieval]", -1.5))
    assert result.tokens[1].top == ()
    assert result.usage.input_tokens == 7
    assert result.usage.output_tokens == 2


def test_non_opted_profile_is_rejected_before_transport() -> None:
    called = False

    async def respond(request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(500)

    model = OllamaRawCompletionModel(
        ollama_chat_profile(), api_base="http://ollama.test", transport=httpx.MockTransport(respond)
    )

    async def call() -> None:
        await model.complete_raw(RawCompletionRequest(prompt="hello", max_output_tokens=2, top_logprobs=1))

    with pytest.raises(MissingCapabilityError):
        asyncio.run(call())

    assert called is False


def test_raw_completion_request_validates_sampler_and_logprob_ranges() -> None:
    assert RawCompletionRequest(prompt="hello", max_output_tokens=2, top_logprobs=21).top_logprobs == 21
    with pytest.raises(ValidationError):
        RawCompletionRequest(prompt="hello", max_output_tokens=2, top_logprobs=1, top_p=0)
    with pytest.raises(ValidationError):
        RawCompletionRequest(prompt="hello", max_output_tokens=2, top_logprobs=1, top_k=-1)


def test_ollama_rejects_more_than_20_logprobs_before_transport() -> None:
    called = False

    async def respond(request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(500)

    model = OllamaRawCompletionModel(
        ollama_raw_completion_profile(), api_base="http://ollama.test", transport=httpx.MockTransport(respond)
    )
    request = RawCompletionRequest(prompt="hello", max_output_tokens=2, top_logprobs=21)

    with pytest.raises(ProviderCallError) as caught:
        asyncio.run(model.complete_raw(request))

    assert caught.value.reason == "invalid_request"
    assert not called


def test_token_logprob_ids_are_optional_and_validate_alignment() -> None:
    from rag_foundry_providers.messages import TokenLogprob

    old_shape = TokenLogprob(token="x", logprob=-0.1, top=(("x", -0.1),))
    assert old_shape.token_id is None
    assert old_shape.top_ids is None
    with pytest.raises(ValidationError):
        TokenLogprob(token="x", logprob=-0.1, top=(("x", -0.1),), top_ids=(1, 2))


def _good_body(**changes: object) -> dict[str, object]:
    body: dict[str, object] = {
        "response": "ok",
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": 3,
        "eval_count": 1,
        "logprobs": [
            {"token": "ok", "logprob": -0.5, "top_logprobs": [{"token": "ok", "logprob": -0.5}]},
        ],
    }
    body.update(changes)
    return body


def _entry(**changes: object) -> dict[str, object]:
    entry: dict[str, object] = {"token": "ok", "logprob": -0.5, "top_logprobs": []}
    entry.update(changes)
    return entry


def _model(handler: object) -> OllamaRawCompletionModel:
    return OllamaRawCompletionModel(
        ollama_raw_completion_profile(), api_base="http://ollama.test", transport=httpx.MockTransport(handler)
    )


def _call(model: OllamaRawCompletionModel):
    request = RawCompletionRequest(prompt="hello", max_output_tokens=4, top_logprobs=3)
    return asyncio.run(model.complete_raw(request))


def _raw_json(body: dict[str, object], logprob_literal: str) -> str:
    text = json.dumps(_good_body(logprobs=[_entry(logprob="@@")]))
    return text.replace('"@@"', logprob_literal)


_BAD_BODIES: list[tuple[str, str]] = [
    ("logprobs_missing", json.dumps({k: v for k, v in _good_body().items() if k != "logprobs"})),
    ("logprobs_not_list", json.dumps(_good_body(logprobs={"a": 1}))),
    ("entry_not_dict", json.dumps(_good_body(logprobs=["x"]))),
    ("token_not_str", json.dumps(_good_body(logprobs=[_entry(token=5)]))),
    ("token_missing", json.dumps(_good_body(logprobs=[{"logprob": -1.0, "top_logprobs": []}]))),
    ("logprob_bool", json.dumps(_good_body(logprobs=[_entry(logprob=True)]))),
    ("logprob_str", json.dumps(_good_body(logprobs=[_entry(logprob="-0.5")]))),
    ("logprob_nan", _raw_json({}, "NaN")),
    ("logprob_overflow_1e400", _raw_json({}, "1e400")),
    ("logprob_positive", json.dumps(_good_body(logprobs=[_entry(logprob=0.5)]))),
    ("top_logprobs_missing", json.dumps(_good_body(logprobs=[{"token": "ok", "logprob": -0.5}]))),
    ("alternative_not_dict", json.dumps(_good_body(logprobs=[_entry(top_logprobs=["x"])]))),
    (
        "alternative_token_not_str",
        json.dumps(_good_body(logprobs=[_entry(top_logprobs=[{"token": 1, "logprob": -1}])])),
    ),
    (
        "alternative_logprob_positive",
        json.dumps(_good_body(logprobs=[_entry(top_logprobs=[{"token": "a", "logprob": 0.3}])])),
    ),
    (
        "alternative_logprob_bool",
        json.dumps(_good_body(logprobs=[_entry(top_logprobs=[{"token": "a", "logprob": False}])])),
    ),
    ("error_present", json.dumps(_good_body(error="boom"))),
    ("done_false", json.dumps(_good_body(done=False))),
    ("done_missing", json.dumps({k: v for k, v in _good_body().items() if k != "done"})),
    ("done_reason_unknown", json.dumps(_good_body(done_reason="load"))),
    ("done_reason_missing", json.dumps({k: v for k, v in _good_body().items() if k != "done_reason"})),
    ("response_not_str", json.dumps(_good_body(response=7))),
]


@pytest.mark.parametrize("raw_body", [b for _, b in _BAD_BODIES], ids=[i for i, _ in _BAD_BODIES])
def test_untrusted_response_is_bad_response(raw_body: str) -> None:
    async def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=raw_body.encode(), headers={"content-type": "application/json"})

    with pytest.raises(ProviderCallError) as caught:
        _call(_model(respond))

    assert caught.value.reason == "bad_response"


def test_http_500_is_bad_response() -> None:
    async def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    with pytest.raises(ProviderCallError) as caught:
        _call(_model(respond))

    assert caught.value.reason == "bad_response"


def test_missing_eval_counts_fall_back_to_estimated_usage() -> None:
    body = _good_body()
    del body["prompt_eval_count"], body["eval_count"]

    async def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=body)

    result = _call(_model(respond))

    assert result.usage.token_source is TokenSource.ESTIMATED
    assert result.usage.input_tokens > 0


def test_contract_check_passes_for_ollama_raw_model() -> None:
    async def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_good_body())

    for check in contract_checks.RAW_COMPLETION_CHECKS:
        asyncio.run(check(_model(respond)))


def test_contract_check_passes_for_opted_fake() -> None:
    model = FakeChatModel(fake_chat_profile(raw_logprobs=True))
    for check in contract_checks.RAW_COMPLETION_CHECKS:
        asyncio.run(check(model))


def test_contract_check_covers_missing_capability_for_non_opted_fake() -> None:
    model = FakeChatModel(fake_chat_profile())
    for check in contract_checks.RAW_COMPLETION_CHECKS:
        asyncio.run(check(model))


def test_fake_complete_raw_opted_returns_scripted_tokens_with_fake_usage() -> None:
    model = FakeChatModel(fake_chat_profile(raw_logprobs=True))
    request = RawCompletionRequest(prompt="hello", max_output_tokens=4, top_logprobs=2)

    result = asyncio.run(model.complete_raw(request))

    assert result.tokens and result.text
    assert result.usage.provider == "fake"


def test_fake_complete_raw_non_opted_is_rejected() -> None:
    model = FakeChatModel(fake_chat_profile())
    request = RawCompletionRequest(prompt="hello", max_output_tokens=4, top_logprobs=2)

    with pytest.raises(MissingCapabilityError):
        asyncio.run(model.complete_raw(request))
