import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from rag_foundry_domain.documents import Scope
from rag_foundry_domain.errors import CacheError
from rag_foundry_domain.response_cache import (
    RESPONSE_CACHE_STORE_CHECKS,
    CacheEntry,
    InMemoryResponseCacheStore,
    ResponseCacheStore,
)

T0 = datetime(2026, 1, 1, tzinfo=UTC)
SCOPE = Scope(tenant_id="7f3c2b1e-5d4a-4c8b-9e6f-1a2b3c4d5e6f", project_id="2d9e8f7a-6b5c-4a3d-8e2f-0c1b2a3d4e5f")
OTHER = Scope(tenant_id="11111111-2222-4333-8444-555555555555", project_id="66666666-7777-4888-9999-aaaaaaaaaaaa")


def make_entry(scope=SCOPE, **over):
    data = {
        "key": "a" * 64,
        "partition": "b" * 64,
        "tenant_id": scope.tenant_id,
        "project_id": scope.project_id,
        "response_json": "{}",
        "source_run_id": "run-1",
        "source_version_ids": ("dv-1",),
        "created_at": T0,
        "expires_at": T0 + timedelta(hours=1),
    }
    data.update(over)
    return CacheEntry(**data)


@pytest.mark.parametrize("check", RESPONSE_CACHE_STORE_CHECKS)
def test_in_memory_response_cache_store_checks(check):
    async def factory():
        store = InMemoryResponseCacheStore()
        assert isinstance(store, ResponseCacheStore)
        return store

    asyncio.run(check(factory))


def test_forced_key_collision_still_raises_scope_mismatch():
    """A store double that holds a foreign entry under the caller's slot must not leak it."""

    async def run():
        store = InMemoryResponseCacheStore()
        foreign = make_entry(OTHER, question_vector=(1.0, 0.0), vector_key="emb")
        store._entries[(SCOPE.tenant_id, SCOPE.project_id, foreign.key)] = foreign
        calls = (
            lambda: store.get(SCOPE, foreign.key),
            lambda: store.delete(SCOPE, foreign.key),
            lambda: store.put(SCOPE, foreign),
            lambda: store.similar(SCOPE, foreign.partition, "emb", (1.0, 0.0), 1),
        )
        for call in calls:
            with pytest.raises(CacheError) as info:
                await call()
            assert info.value.code == "cache_scope_mismatch"

    asyncio.run(run())


@pytest.mark.parametrize(
    "over",
    [
        {"key": "A" * 64},
        {"partition": "b" * 63},
        {"response_json": ""},
        {"response_json": "x" * 200_001},
        {"source_version_ids": ()},
        {"source_version_ids": tuple(f"v{i}" for i in range(201))},
        {"source_version_ids": ("v", "v")},
        {"expires_at": T0},
        {"created_at": datetime(2026, 1, 1)},
        {"question_vector": (1.0,)},
        {"vector_key": "emb"},
        {"question_vector": (float("nan"),), "vector_key": "emb"},
        {"question": "text"},
        {"features_json": "x" * 4001},
        {"source_run_id": "r" * 65},
        {"source_run_id": " "},
        {"question_vector": (1.0,), "vector_key": "k" * 201},
    ],
)
def test_entry_rejects_invalid(over):
    with pytest.raises(ValueError):
        make_entry(**over)


def test_entry_has_no_question_text_field_and_is_frozen():
    entry = make_entry()
    assert not [name for name in CacheEntry.model_fields if "question" in name and "vector" not in name]
    with pytest.raises(ValueError):
        entry.key = "c" * 64


def test_entry_limits_accept_boundary_values():
    entry = make_entry(features_json="x" * 4000, source_run_id="r" * 64, question_vector=(1.0,), vector_key="k" * 200)
    assert len(entry.features_json) == 4000
