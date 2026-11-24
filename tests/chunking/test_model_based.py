import hashlib

import pytest
from pydantic import BaseModel
from rag_foundry_chunking.model_based import (
    InMemorySegmentationCache,
    build_units,
    make_nonce,
    parse_json_object,
    segmentation_cache_key,
    segments_to_spans,
)
from rag_foundry_providers.fake import fake_embedding_profile


class Config(BaseModel):
    size: int = 32


def test_cache_and_lru():
    cache = InMemorySegmentationCache(2)
    cache.put("a", "1")
    cache.put("b", "2")
    assert cache.get("a") == "1"
    cache.put("c", "3")
    assert cache.get("b") is None
    assert cache.hits == 1 and cache.misses == 1
    profile = fake_embedding_profile()
    args = dict(
        technique_id="C15", plugin_version="1", config=Config(), text="hello", profile=profile, prompt=None, sample=0
    )
    assert segmentation_cache_key(**args) == segmentation_cache_key(**args)
    base = segmentation_cache_key(**args)
    for field, value in (("text", "bye"), ("plugin_version", "2"), ("sample", 1)):
        changed = dict(args, **{field: value})
        assert segmentation_cache_key(**changed) != base


@pytest.mark.parametrize("value", ['{"a":1}', '```\n{"a":1}\n```', '```json\n{"a":1}\n```'])
def test_parse_object(value):
    assert parse_json_object(value) == {"a": 1}


@pytest.mark.parametrize("value", ['x {"a":1}', '{"a":1} {"b":2}', "[]", "{bad}", "x" * 100001])
def test_parse_rejects(value):
    assert parse_json_object(value) is None


def test_nonce_and_segments():
    text = "Alpha. Beta."
    first = hashlib.sha256(f"{hashlib.sha256(text.encode()).hexdigest()}:0".encode()).hexdigest()[:16]
    assert make_nonce(first + text) != first
    units = build_units(text, 32)
    spans = segments_to_spans(text, units, [0], 32)
    assert len(spans) == 2
    with pytest.raises(ValueError):
        segments_to_spans(text, units, [5], 32)
    long = " ".join(f"Alpha sentence number {index}." for index in range(20))
    packed = segments_to_spans(long, build_units(long, 32), [], 32)
    assert all(span.split_level == 1 for span in packed)
