"""Opt-in live smoke test for the model-based chunkers (C15, C16, C17) against local Ollama.

Talks only to `load_settings().ollama_base_url` (loopback by default): bge-m3 and gemma4:e4b, no keys,
no paid calls. Run only with the owner's OK:
    RAG_LIVE_OLLAMA=1 uv run pytest tests/live/test_model_chunkers_live.py -m live -v -s
It asserts validation and caching behaviour, never a specific model boundary.
"""

import asyncio
import importlib.util
import os
from decimal import Decimal
from pathlib import Path

import pytest
from rag_foundry_answering.runner import ollama_model_factory, retry_policy
from rag_foundry_chunking import (
    InMemorySegmentationCache,
    LlmBoundaryChunker,
    LlmBoundaryConfig,
    ModelChunkContext,
    PropositionChunker,
    PropositionConfig,
    SemanticChunker,
    SemanticConfig,
)
from rag_foundry_core.settings import load_settings
from rag_foundry_domain.prompts import PromptTemplate, prompt_sha256
from rag_foundry_domain.runtime_settings import RunSettings
from rag_foundry_providers.budget import RunBudget
from rag_foundry_providers.ledger import UsageLedger
from rag_foundry_techniques import ChunkInput

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(os.environ.get("RAG_LIVE_OLLAMA") != "1", reason="RAG_LIVE_OLLAMA=1 is not set"),
]

_MIGRATION = (
    Path(__file__).parents[2]
    / "packages/storage/src/rag_foundry_storage/sql/migrations/versions/0006_chunking_prompts.py"
)
_KNOWN_REASONS = {
    "truncated",
    "invalid_json",
    "invalid_shape",
    "too_many_items",
    "invalid_item",
    "invalid_unit",
    "out_of_range",
    "not_ascending",
    "invalid_quote",
    "quote_too_short",
    "quote_mismatch",
    "quote_not_found",
    "unsupported_number",
    "unsupported_claim",
    "negation_dropped",
    "negation_added",
    "condition_dropped",
}
TEXT = (
    "Solar panels convert sunlight into electricity for homes. "
    "A battery stores extra energy for use at night. "
    "تزور النحلة الأزهار "
    "وتنقل حبوب اللقاح. "
    "لا ينتج النحل العسل "
    "في الشتاء."
)


def _template(name: str, purpose: str) -> PromptTemplate:
    spec = importlib.util.spec_from_file_location("seed_0006", _MIGRATION)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    text = getattr(module, name)
    return PromptTemplate(
        id=f"{purpose}-v1",
        family=purpose,
        version=1,
        purpose=purpose,
        system_text=text,
        content_sha256=prompt_sha256(text),
    )


def _assert_source_slices(result) -> None:
    for span in result.spans:
        piece = TEXT[span.start : span.end]
        assert piece and piece == piece.strip()


def test_model_chunkers_against_local_ollama():
    ledger = UsageLedger()
    budget = RunBudget(Decimal("0"), max_calls=40)
    chat, embedder = ollama_model_factory(load_settings().ollama_base_url)(
        ledger, budget, retry_policy(RunSettings()), 1024
    )
    doc = ChunkInput(text=TEXT)
    cache = InMemorySegmentationCache()

    async def exercise():
        semantic = SemanticChunker(SemanticConfig(threshold_type="absolute", threshold_amount=0.35))
        first = await semantic.chunk_with_models(doc, ModelChunkContext(embedder=embedder, cache=cache))
        again = await semantic.chunk_with_models(doc, ModelChunkContext(embedder=embedder, cache=cache))
        _assert_source_slices(first)
        assert first.report.model_calls >= 1 and again.report.cache_hit and again.report.model_calls == 0
        print("C15", first.report.details["distances"], [TEXT[s.start : s.end] for s in first.spans])

        for chunker, prompt in (
            (LlmBoundaryChunker(LlmBoundaryConfig()), _template("_CHUNK_BOUNDARIES_V1", "chunk-boundaries")),
            (PropositionChunker(PropositionConfig()), _template("_PROPOSITIONS_V1", "propositions")),
        ):
            ctx = ModelChunkContext(chat=chat, prompt=prompt, cache=cache)
            first = await chunker.chunk_with_models(doc, ctx)
            again = await chunker.chunk_with_models(doc, ctx)
            _assert_source_slices(first)
            assert {r.reason for r in first.report.rejections} <= _KNOWN_REASONS
            assert again.report.cache_hit and again.report.model_calls == 0
            assert again.spans == first.spans and again.report.rejections == first.report.rejections
            print(
                first.report.technique_id,
                {
                    "calls": first.report.model_calls,
                    "rejections": [(r.reason, r.detail) for r in first.report.rejections],
                    "spans": [TEXT[s.start : s.end] for s in first.spans],
                },
            )
            for record in getattr(first, "propositions", ()):
                print("  proposition (generated):", record.text, "->", TEXT[record.start : record.end])

    asyncio.run(exercise())
    print("usage records:", len(ledger.records()), "budget:", budget.snapshot())
