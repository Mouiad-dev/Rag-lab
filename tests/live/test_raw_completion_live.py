"""Opt-in raw Self-RAG completion check against an already-created local Ollama model."""

import asyncio
import os

import pytest
from rag_foundry_providers.adapters.ollama import OllamaRawCompletionModel, ollama_raw_completion_profile
from rag_foundry_providers.messages import RawCompletionRequest

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        os.environ.get("RAG_LIVE_OLLAMA") != "1" or not os.environ.get("RAG_SELFRAG_MODEL"),
        reason="requires RAG_LIVE_OLLAMA=1 and RAG_SELFRAG_MODEL",
    ),
]

_REFLECTION_TOKENS = (
    "[No Retrieval]",
    "[Retrieval]",
    "[Continue to Use Evidence]",
    "[Irrelevant]",
    "[Relevant]",
    "[Utility:1]",
    "[Utility:2]",
    "[Utility:3]",
    "[Utility:4]",
    "[Utility:5]",
    "[Fully supported]",
    "[Partially supported]",
    "[No support / Contradictory]",
)
_PROMPT = "### Instruction:\nWhat is the capital of France?\n\n### Response:\n"


def test_raw_selfrag_logprobs_and_top_k_sampler_comparison() -> None:
    model_name = os.environ["RAG_SELFRAG_MODEL"]
    model = OllamaRawCompletionModel(ollama_raw_completion_profile(model=model_name), api_base="http://localhost:11434")
    base = {
        "prompt": _PROMPT,
        "max_output_tokens": 64,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_logprobs": 20,
    }

    async def check() -> None:
        result = await model.complete_raw(RawCompletionRequest(**base, top_k=0))
        for item in result.tokens[:10]:
            print(json_token(item.token, item.logprob, item.top))
        assert any(item.token in _REFLECTION_TOKENS for item in result.tokens)

        narrow = await model.complete_raw(RawCompletionRequest(**base, top_k=1))
        wide = await model.complete_raw(RawCompletionRequest(**base, top_k=0))
        same = bool(narrow.tokens and wide.tokens and narrow.tokens[0].top == wide.tokens[0].top)
        print(f"first_position_top20_equal_for_top_k_1_and_0={same}")

    asyncio.run(check())


def json_token(token: str, logprob: float, top: tuple[tuple[str, float], ...]) -> str:
    """Compact ASCII-safe line for pytest output."""
    import json

    return json.dumps({"token": token, "logprob": logprob, "top": top}, ensure_ascii=True)
