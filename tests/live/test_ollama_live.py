"""Opt-in live tests against a local Ollama (P2.2-05). Skipped unless RAG_LIVE_OLLAMA=1.

Talks only to `load_settings().ollama_base_url` (loopback by default). No keys, no paid calls.
Run: RAG_LIVE_OLLAMA=1 uv run pytest tests/live -m live -v -s
"""

import asyncio
import json
import math
import os
import urllib.request
from typing import Any

import pytest
from rag_foundry_core.settings import load_settings
from rag_foundry_providers.adapters.ollama import (
    OllamaChatModel,
    OllamaEmbeddingModel,
    ollama_chat_profile,
    ollama_embedding_profile,
)
from rag_foundry_providers.contract_checks import CHAT_CHECKS, EMBEDDING_CHECKS
from rag_foundry_providers.messages import ChatMessage, ChatRequest

pytestmark = pytest.mark.live

CHAT_TIMEOUT_S = 600.0  # the first call loads the model into the GPU
EMBED_TIMEOUT_S = 300.0
_EN = "hello world"
_AR = "مرحبا بالعالم"


def _tag_names(base_url: str) -> set[str]:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(f"{base_url.rstrip('/')}/api/tags", timeout=2) as response:
        data = json.loads(response.read())
    return {m.get("name", "") for m in data.get("models", []) if isinstance(m, dict)}


@pytest.fixture(scope="module")
def base_url() -> str:
    if os.environ.get("RAG_LIVE_OLLAMA") != "1":
        pytest.skip("RAG_LIVE_OLLAMA=1 is not set; live Ollama tests are opt-in")
    url = load_settings().ollama_base_url
    try:
        names = _tag_names(url)
    except Exception as exc:  # noqa: BLE001 - any failure means the server is unusable
        pytest.skip(f"Ollama not reachable at the configured base URL ({type(exc).__name__})")
    missing = []
    if "gemma4:e4b" not in names:
        missing.append("gemma4:e4b")
    if not names & {"bge-m3", "bge-m3:latest"}:
        missing.append("bge-m3")
    if missing:
        pytest.skip(f"Ollama /api/tags does not list: {', '.join(missing)}")
    return url


@pytest.fixture(scope="module")
def chat_model(base_url: str) -> OllamaChatModel:
    return OllamaChatModel(ollama_chat_profile(), api_base=base_url, timeout_s=CHAT_TIMEOUT_S)


@pytest.fixture(scope="module")
def embed_model(base_url: str) -> OllamaEmbeddingModel:
    return OllamaEmbeddingModel(ollama_embedding_profile(), api_base=base_url, timeout_s=EMBED_TIMEOUT_S)


def _show(label: str, usage: Any) -> None:
    print(
        f"[live] {label}: token_source={usage.token_source.value} in={usage.input_tokens} "
        f"out={usage.output_tokens} latency_ms={usage.latency_ms:.0f}"
    )


@pytest.mark.parametrize("check", CHAT_CHECKS, ids=lambda c: c.__name__)
def test_chat_contract(chat_model: OllamaChatModel, check: Any) -> None:
    asyncio.run(check(chat_model))


@pytest.mark.parametrize("check", EMBEDDING_CHECKS, ids=lambda c: c.__name__)
def test_embedding_contract(embed_model: OllamaEmbeddingModel, check: Any) -> None:
    asyncio.run(check(embed_model))


@pytest.mark.parametrize(("lang", "prompt"), [("english", _EN), ("arabic", _AR)], ids=["english", "arabic"])
def test_chat_text_and_finish_reason(chat_model: OllamaChatModel, lang: str, prompt: str) -> None:
    request = ChatRequest(messages=(ChatMessage(role="user", content=prompt),), max_output_tokens=64)
    result = asyncio.run(chat_model.complete(request))
    _show(f"chat {lang}", result.usage)
    assert result.text.strip()
    assert result.finish_reason in ("stop", "length")
    print(f"[live] chat {lang}: finish_reason={result.finish_reason} chars={len(result.text)}")


def test_embed_english_and_arabic(embed_model: OllamaEmbeddingModel) -> None:
    result = asyncio.run(embed_model.embed([_EN, _AR]))
    _show("embed en+ar", result.usage)
    english, arabic = result.vectors
    print(f"[live] embedding dims: en={len(english)} ar={len(arabic)}")
    assert len(english) == len(arabic) == 1024
    assert all(math.isfinite(x) for x in english + arabic)
    assert english != arabic
