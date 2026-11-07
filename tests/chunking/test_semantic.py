import asyncio
import json
import statistics
import unicodedata

import pytest
from pydantic import ValidationError
from rag_foundry_chunking import SemanticChunker, SemanticConfig, register_builtin_plugins
from rag_foundry_chunking.model_based import InMemorySegmentationCache, ModelChunkContext, ModelChunkingError
from rag_foundry_providers.capabilities import ModelKind
from rag_foundry_providers.fake import fake_chat_profile, fake_embedding_profile
from rag_foundry_techniques import ChunkInput, TechniqueRegistry, TechniqueUnavailableError, load_catalogue

from tests.chunking._scripted_models import ScriptedEmbedder, topic_vector


def passage(left="cat", right="market"):
    return (
        " ".join(f"The {left} is here." for _ in range(4)) + " " + " ".join(f"The {right} is moving." for _ in range(4))
    )


def run(text, config=None, embedder=None, cache=None):
    c = SemanticChunker(config or SemanticConfig(threshold_type="absolute", threshold_amount=0.5))
    model = embedder or ScriptedEmbedder(topic_vector)
    return asyncio.run(
        c.chunk_with_models(
            ChunkInput(text=text), ModelChunkContext(embedder=model, cache=cache or InMemorySegmentationCache())
        )
    )


def test_topic_breaks_and_cache():
    text = passage()
    model, cache = ScriptedEmbedder(topic_vector), InMemorySegmentationCache()
    first = run(
        text,
        SemanticConfig(buffer_size=0, threshold_type="absolute", threshold_amount=0.5),
        embedder=model,
        cache=cache,
    )
    assert first.report.details["break_after"] == [3]
    second = run(
        text,
        SemanticConfig(buffer_size=0, threshold_type="absolute", threshold_amount=0.5),
        embedder=model,
        cache=cache,
    )
    assert second.report.cache_hit and second.report.model_calls == 0
    assert first.spans == second.spans
    assert all(
        text[s.start : s.end]
        and not text[s.start].isspace()
        and not text[s.end - 1].isspace()
        and s.end - s.start <= 800
        for s in first.spans
    )


def test_thresholds_and_windows():
    text = " ".join(f"The cat is resting {index}." for index in range(8))
    absolute = run(text)
    percentile = run(text, SemanticConfig(threshold_type="percentile", threshold_amount=90))
    std = run(text, SemanticConfig(threshold_type="std_dev", threshold_amount=3))
    assert absolute.report.details["break_after"] == []
    assert percentile.report.details["break_after"]
    assert std.report.details["break_after"] == []
    assert run("One. Two. Three.", SemanticConfig(buffer_size=0)).report.details["unit_count"] == 3


def test_threshold_values_and_buffer_one_windows():
    text = " ".join(f"The cat is resting {index}." for index in range(8))
    absolute = run(text, SemanticConfig(buffer_size=0, threshold_type="absolute", threshold_amount=0.5))
    assert absolute.report.details["threshold"] == 0.5
    std = run(text, SemanticConfig(buffer_size=0, threshold_type="std_dev", threshold_amount=3))
    assert std.report.details["threshold"] == pytest.approx(
        statistics.mean(std.report.details["distances"]) + 3 * statistics.pstdev(std.report.details["distances"])
    )
    percentile = run(text, SemanticConfig(buffer_size=0, threshold_type="percentile", threshold_amount=90))
    ordered = sorted(percentile.report.details["distances"])
    position = 0.9 * (len(ordered) - 1)
    low = int(position)
    fraction = position - low
    expected = ordered[low] + (ordered[min(low + 1, len(ordered) - 1)] - ordered[low]) * fraction
    assert percentile.report.details["threshold"] == pytest.approx(expected)
    model = ScriptedEmbedder(topic_vector)
    run("One. Two. Three.", SemanticConfig(buffer_size=1), model)
    assert model.calls == [("One. Two.", "One. Two. Three.", "Two. Three.")]


def test_cache_misses_and_corruption_recompute():
    text = passage()
    config = SemanticConfig(buffer_size=0, threshold_type="absolute", threshold_amount=0.5)
    cache = InMemorySegmentationCache()
    first = run(text, config, ScriptedEmbedder(topic_vector), cache)
    assert not first.report.cache_hit
    variants = [
        (text + " Extra.", config, fake_embedding_profile()),
        (text, config, fake_embedding_profile(id="other", dim=4)),
        (
            text,
            SemanticConfig(buffer_size=0, threshold_type="absolute", threshold_amount=0.4),
            fake_embedding_profile(),
        ),
    ]
    for variant_text, variant_config, profile in variants:
        model = ScriptedEmbedder(topic_vector, profile)
        result = run(variant_text, variant_config, model, cache)
        assert model.calls and not result.report.cache_hit
    clean_cache = InMemorySegmentationCache()
    clean = run(text, config, ScriptedEmbedder(topic_vector), clean_cache)
    for invalid in ('{"distances":[0]}', json.dumps({"distances": [3] * 7}), "not json"):
        damaged_cache = InMemorySegmentationCache()
        damaged_cache.put(clean.report.cache_key, invalid)
        model = ScriptedEmbedder(topic_vector)
        recovered = run(text, config, model, damaged_cache)
        assert model.calls and recovered.report.model_calls >= 1
        assert recovered.report.details["cache_invalid"] is True
        assert recovered.spans == clean.spans


def test_error_codes_and_batch_cost_accounting():
    text = "One. Two. Three."
    with pytest.raises(ModelChunkingError) as error:
        asyncio.run(
            SemanticChunker(SemanticConfig()).chunk_with_models(
                ChunkInput(text=text),
                ModelChunkContext(
                    embedder=ScriptedEmbedder(topic_vector, fake_chat_profile()), cache=InMemorySegmentationCache()
                ),
            )
        )
    assert error.value.code == "wrong_model_kind"
    model = ScriptedEmbedder(topic_vector)
    with pytest.raises(ModelChunkingError) as error:
        asyncio.run(
            SemanticChunker(SemanticConfig(buffer_size=0, max_units=2)).chunk_with_models(
                ChunkInput(text=text), ModelChunkContext(embedder=model, cache=InMemorySegmentationCache())
            )
        )
    assert error.value.code == "too_many_units" and model.calls == []

    class ShortEmbedder(ScriptedEmbedder):
        async def embed(self, texts, *, input_type="document"):
            from rag_foundry_providers.messages import EmbeddingResult

            self.calls.append(tuple(texts))
            return EmbeddingResult(vectors=(), usage=(await super().embed([], input_type=input_type)).usage)

    with pytest.raises(ModelChunkingError) as error:
        asyncio.run(
            SemanticChunker(SemanticConfig(buffer_size=0)).chunk_with_models(
                ChunkInput(text=text),
                ModelChunkContext(embedder=ShortEmbedder(topic_vector), cache=InMemorySegmentationCache()),
            )
        )
    assert error.value.code == "embedding_count_mismatch"
    five = ". ".join(f"Unit {index}" for index in range(5)) + "."
    batched = ScriptedEmbedder(topic_vector)
    result = run(five, SemanticConfig(buffer_size=0, embed_batch_size=2), batched)
    assert len(batched.calls) == result.report.model_calls == 3
    assert result.report.total_cost_usd == sum(
        (usage.estimated_cost_usd or 0 for usage in result.report.usage), start=type(result.report.total_cost_usd)(0)
    )


def test_validation_registry_sync_errors():
    with pytest.raises(ValidationError):
        SemanticConfig(threshold_amount=120)
    registry = TechniqueRegistry(load_catalogue("config/techniques.json"))
    register_builtin_plugins(registry)
    assert registry.validate_config("C15", {}) == SemanticConfig()
    with pytest.raises(TechniqueUnavailableError):
        SemanticChunker(SemanticConfig()).chunk(ChunkInput(text="text"))
    with pytest.raises(ModelChunkingError):
        asyncio.run(
            SemanticChunker(SemanticConfig()).chunk_with_models(
                ChunkInput(text="One. Two."), ModelChunkContext(cache=InMemorySegmentationCache())
            )
        )


def test_nfc_fixture_and_embed_batch_text():
    text = "\u0642\u0637\u0629. " * 4 + "\u0633\u0648\u0642. " * 4
    assert unicodedata.is_normalized("NFC", text)
    result = run(text, SemanticConfig(buffer_size=0, threshold_type="absolute", threshold_amount=0.5))
    assert result.report.details["break_after"] == [3]
    assert all(
        text[s.start : s.end]
        and not text[s.start : s.end][0].isspace()
        and not text[s.start : s.end][-1].isspace()
        and s.end - s.start <= 800
        for s in result.spans
    )
    model = ScriptedEmbedder(topic_vector)
    run("One. Two. Three.", SemanticConfig(buffer_size=0), model)
    assert model.calls[0] == ("One.", "Two.", "Three.")
    assert fake_chat_profile().kind is ModelKind.CHAT
