"""Opt-in local BAAI/bge-m3 per-token check (CPU, safetensors only, no network)."""

import asyncio
import math
import os

import pytest

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(os.environ.get("RAG_LIVE_TOKENS") != "1", reason="set RAG_LIVE_TOKENS=1"),
]

TEXTS = [
    "The basalt sample was collected near the northern glacier.",
    (
        "\u0648\u0635\u0644\u062a "
        "\u0627\u0644\u0642\u0627\u0641\u0644\u0629 "
        "\u0625\u0644\u0649 \u0645\u062f\u064a\u0646\u0629 "
        "\u0635\u0648\u0631 \u0628\u0639\u062f "
        "\u0627\u0644\u063a\u0631\u0648\u0628."
    ),
    "x " + chr(0x1F600) + " y",
]


def _normalised(vector):
    norm = math.sqrt(sum(x * x for x in vector))
    return [x / norm for x in vector]


@pytest.mark.parametrize("head", ["hidden", "colbert"])
def test_local_token_encoder_english_and_arabic(head):
    from rag_foundry_providers.sentence_transformers_tokens import SentenceTransformersTokenEncoder

    model_dir = os.environ.get("RAG_TOKEN_MODEL_DIR")
    if not model_dir:
        pytest.skip("set RAG_TOKEN_MODEL_DIR")
    encoder = SentenceTransformersTokenEncoder(model_dir=model_dir)
    result = asyncio.run(encoder.encode_tokens(TEXTS, head=head, input_type="document"))
    assert result.dim == 1024
    assert len(result.items) == len(TEXTS)
    for text, item in zip(TEXTS, result.items, strict=True):
        assert item.truncated is False
        assert len(item.vectors) == len(item.offsets) > 0
        assert all(len(v) == result.dim for v in item.vectors)
        assert all(0 <= s <= e <= len(text) for s, e in item.offsets)
        # Round trip: the slices, in order and without overlap, cover every non-space character.
        assert all(prev[1] <= cur[0] for prev, cur in zip(item.offsets, item.offsets[1:], strict=False))
        covered = "".join(text[s:e] for s, e in item.offsets)
        assert "".join(covered.split()) == "".join(text.split())
        assert all(any(x != 0.0 for x in v) for v in item.vectors)
        if head == "colbert":
            assert all(abs(math.sqrt(sum(x * x for x in v)) - 1.0) < 1e-3 for v in item.vectors)
    if head == "colbert":
        hidden = asyncio.run(encoder.encode_tokens(TEXTS, head="hidden", input_type="document"))
        for colbert_item, hidden_item in zip(result.items, hidden.items, strict=True):
            # A missing or wrong head would reproduce the normalised hidden states.
            assert colbert_item.offsets == hidden_item.offsets
            gap = max(
                abs(a - b)
                for c, h in zip(colbert_item.vectors, hidden_item.vectors, strict=True)
                for a, b in zip(c, _normalised(h), strict=True)
            )
            assert gap > 1e-2
    print(
        f"head={head} revision={encoder.revision} dim={result.dim} "
        f"tokens={[len(i.vectors) for i in result.items]} offsets_en={result.items[0].offsets[:4]}"
    )
