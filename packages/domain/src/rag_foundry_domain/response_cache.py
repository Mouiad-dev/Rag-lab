"""Response-cache entry model, store port, in-memory store and shared store checks (R50/R51, ADR-0021).

Entries never hold the question text. The SQL store (P7.4-09a) implements the same port and must pass
`RESPONSE_CACHE_STORE_CHECKS` unchanged. Scope is checked before anything else on every operation.
"""

import hashlib
import math
import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from typing import Annotated, Protocol, Self, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from rag_foundry_domain.documents import Scope, Uuid4
from rag_foundry_domain.errors import CacheError

MAX_RESPONSE_JSON_CHARS = 200_000
MAX_SOURCE_VERSIONS = 200
MAX_FEATURES_JSON_CHARS = 4000
MAX_VECTOR_DIMS = 8192

Hex64 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
_NonBlank = Annotated[str, StringConstraints(min_length=1, max_length=256, pattern=r"\S")]
_RunId = Annotated[str, StringConstraints(min_length=1, max_length=64, pattern=r"\S")]
_VectorKey = Annotated[str, StringConstraints(min_length=1, max_length=200, pattern=r"\S")]


class CacheEntry(BaseModel):
    """One cached answer. Carries guarded question features, never the question text."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    key: Hex64
    partition: Hex64
    tenant_id: Uuid4
    project_id: Uuid4
    response_json: str = Field(min_length=1, max_length=MAX_RESPONSE_JSON_CHARS)
    source_run_id: _RunId
    source_version_ids: tuple[_NonBlank, ...] = Field(min_length=1, max_length=MAX_SOURCE_VERSIONS)
    created_at: datetime
    expires_at: datetime
    features_json: str | None = Field(default=None, min_length=1, max_length=MAX_FEATURES_JSON_CHARS)
    question_vector: tuple[float, ...] | None = Field(default=None, min_length=1, max_length=MAX_VECTOR_DIMS)
    vector_key: _VectorKey | None = None

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.created_at.tzinfo is None or self.expires_at.tzinfo is None:
            raise ValueError("timestamps must be timezone-aware")
        if self.expires_at <= self.created_at:
            raise ValueError("expires_at must be after created_at")
        if len(set(self.source_version_ids)) != len(self.source_version_ids):
            raise ValueError("source_version_ids must be unique")
        if (self.question_vector is None) != (self.vector_key is None):
            raise ValueError("question_vector and vector_key must be set together")
        if self.question_vector is not None and not all(math.isfinite(x) for x in self.question_vector):
            raise ValueError("question_vector must be finite")
        return self


@runtime_checkable
class ResponseCacheStore(Protocol):
    """Scoped store. Entries are never updated. Expiry is enforced by the caller, not by get/similar."""

    async def get(self, scope: Scope, key: str) -> CacheEntry | None: ...

    async def put(self, scope: Scope, entry: CacheEntry) -> None: ...

    async def delete(self, scope: Scope, key: str) -> bool: ...

    async def similar(
        self, scope: Scope, partition: str, vector_key: str, vector: tuple[float, ...], k: int
    ) -> list[tuple[CacheEntry, float]]: ...

    async def invalidate_sources(self, scope: Scope, version_ids: tuple[str, ...]) -> int: ...

    async def invalidate_partition(self, scope: Scope, partition: str) -> int: ...

    async def purge_expired(self, scope: Scope, now: datetime) -> int: ...


def _same_scope(scope: Scope, entry: CacheEntry) -> bool:
    return entry.tenant_id == scope.tenant_id and entry.project_id == scope.project_id


def cosine(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


class InMemoryResponseCacheStore:
    """Reference store. Keyed by (tenant, project, key) with a per-scope reverse index on source versions."""

    def __init__(self) -> None:
        self._entries: dict[tuple[str, str, str], CacheEntry] = {}
        self._by_version: dict[tuple[str, str], dict[str, set[str]]] = {}

    @staticmethod
    def _scope_id(scope: Scope) -> tuple[str, str]:
        return (scope.tenant_id, scope.project_id)

    @staticmethod
    def _slot(scope: Scope, key: str) -> tuple[str, str, str]:
        return (scope.tenant_id, scope.project_id, key)

    def _remove(self, scope: Scope, key: str) -> bool:
        entry = self._entries.pop(self._slot(scope, key), None)
        if entry is None:
            return False
        index = self._by_version.get(self._scope_id(scope), {})
        for version_id in entry.source_version_ids:
            keys = index.get(version_id)
            if keys is not None:
                keys.discard(key)
                if not keys:
                    del index[version_id]
        return True

    def _scoped(self, scope: Scope) -> list[CacheEntry]:
        tenant, project = self._scope_id(scope)
        return [e for (t, p, _), e in self._entries.items() if t == tenant and p == project]

    async def get(self, scope: Scope, key: str) -> CacheEntry | None:
        entry = self._entries.get(self._slot(scope, key))
        if entry is None:
            return None
        if not _same_scope(scope, entry):
            raise CacheError("cache_scope_mismatch")
        return entry

    async def put(self, scope: Scope, entry: CacheEntry) -> None:
        if not _same_scope(scope, entry):
            raise CacheError("cache_scope_mismatch")
        slot = self._slot(scope, entry.key)
        if slot in self._entries:
            raise CacheError("cache_exists")
        self._entries[slot] = entry
        index = self._by_version.setdefault(self._scope_id(scope), {})
        for version_id in entry.source_version_ids:
            index.setdefault(version_id, set()).add(entry.key)

    async def delete(self, scope: Scope, key: str) -> bool:
        entry = self._entries.get(self._slot(scope, key))
        if entry is not None and not _same_scope(scope, entry):
            raise CacheError("cache_scope_mismatch")
        return self._remove(scope, key)

    async def similar(
        self, scope: Scope, partition: str, vector_key: str, vector: tuple[float, ...], k: int
    ) -> list[tuple[CacheEntry, float]]:
        if k < 1 or not vector or not all(math.isfinite(x) for x in vector):
            raise CacheError("cache_policy_invalid")
        scored = []
        for entry in self._scoped(scope):
            if not _same_scope(scope, entry):
                raise CacheError("cache_scope_mismatch")
            if entry.partition != partition or entry.vector_key != vector_key or entry.question_vector is None:
                continue
            if len(entry.question_vector) != len(vector):
                raise CacheError("cache_entry_invalid")
            scored.append((entry, cosine(entry.question_vector, vector)))
        scored.sort(key=lambda pair: (-pair[1], pair[0].key))
        return scored[:k]

    async def invalidate_sources(self, scope: Scope, version_ids: tuple[str, ...]) -> int:
        index = self._by_version.get(self._scope_id(scope), {})
        keys: set[str] = set()
        for version_id in version_ids:
            keys |= index.get(version_id, set())
        return sum(self._remove(scope, key) for key in sorted(keys))

    async def invalidate_partition(self, scope: Scope, partition: str) -> int:
        keys = [e.key for e in self._scoped(scope) if e.partition == partition]
        return sum(self._remove(scope, key) for key in keys)

    async def purge_expired(self, scope: Scope, now: datetime) -> int:
        if now.tzinfo is None:
            raise CacheError("cache_policy_invalid")
        keys = [e.key for e in self._scoped(scope) if e.expires_at <= now]
        return sum(self._remove(scope, key) for key in keys)


StoreFactory = Callable[[], Awaitable[ResponseCacheStore]]

_T0 = datetime(2026, 1, 1, tzinfo=UTC)


def _new_scope() -> Scope:
    return Scope(tenant_id=str(uuid.uuid4()), project_id=str(uuid.uuid4()))


def _hex(seed: str) -> str:
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()


def _entry(
    scope: Scope,
    seed: str,
    *,
    partition: str = "p",
    versions: tuple[str, ...] = ("dv-1",),
    vector: tuple[float, ...] | None = None,
    vector_key: str | None = None,
    ttl: timedelta = timedelta(hours=1),
) -> CacheEntry:
    return CacheEntry(
        key=_hex("key-" + seed),
        partition=_hex("part-" + partition),
        tenant_id=scope.tenant_id,
        project_id=scope.project_id,
        response_json='{"answer": "stored"}',
        source_run_id="run-" + seed,
        source_version_ids=versions,
        created_at=_T0,
        expires_at=_T0 + ttl,
        features_json='{"numbers": []}',
        question_vector=vector,
        vector_key=vector_key if vector is not None else None,
    )


async def _expect_cache_error(awaitable, code: str) -> None:
    try:
        await awaitable
    except CacheError as exc:
        if exc.code == code:
            return
        raise AssertionError(f"expected {code}, got {exc.code}") from None
    raise AssertionError(f"expected {code}")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


async def check_round_trip_and_scope(factory: StoreFactory) -> None:
    store = await factory()
    scope = _new_scope()
    other_tenant = Scope(tenant_id=str(uuid.uuid4()), project_id=scope.project_id)
    other_project = Scope(tenant_id=scope.tenant_id, project_id=str(uuid.uuid4()))
    entry = _entry(scope, "a")
    _require(await store.get(scope, entry.key) is None, "absent key must give None")
    await store.put(scope, entry)
    _require(await store.get(scope, entry.key) == entry, "entry must round-trip")
    for stranger in (other_tenant, other_project):
        _require(await store.get(stranger, entry.key) is None, "another scope must get None")
        # The same key in another scope is an independent entry.
        mine = _entry(stranger, "a")
        _require(mine.key == entry.key, "check setup: same key expected")
        await store.put(stranger, mine)
        _require(await store.get(stranger, mine.key) == mine, "foreign scope entry must round-trip")
        _require(await store.get(scope, entry.key) == entry, "original entry must be unchanged")
    # A scope that differs from the entry's is rejected before anything else, even for a new key.
    await _expect_cache_error(store.put(other_tenant, _entry(scope, "b")), "cache_scope_mismatch")
    await _expect_cache_error(store.put(other_project, _entry(scope, "b")), "cache_scope_mismatch")
    _require(await store.get(scope, _hex("key-b")) is None, "rejected put must store nothing")
    # Scope mismatch wins over cache_exists.
    await _expect_cache_error(store.put(other_tenant, _entry(scope, "a")), "cache_scope_mismatch")


async def check_exists_and_delete(factory: StoreFactory) -> None:
    store = await factory()
    scope = _new_scope()
    stranger = Scope(tenant_id=str(uuid.uuid4()), project_id=scope.project_id)
    entry = _entry(scope, "a")
    await store.put(scope, entry)
    changed = entry.model_copy(update={"response_json": '{"answer": "other"}'})
    await _expect_cache_error(store.put(scope, changed), "cache_exists")
    _require(await store.get(scope, entry.key) == entry, "entries are never updated")
    _require(await store.delete(stranger, entry.key) is False, "another scope cannot delete")
    _require(await store.get(scope, entry.key) == entry, "foreign delete must not remove the entry")
    _require(await store.delete(scope, entry.key) is True, "delete must report removal")
    _require(await store.delete(scope, entry.key) is False, "second delete must report absence")
    _require(await store.get(scope, entry.key) is None, "deleted entry must be gone")
    await store.put(scope, entry)
    _require(await store.get(scope, entry.key) == entry, "a deleted key can be stored again")


async def check_invalidation(factory: StoreFactory) -> None:
    store = await factory()
    scope = _new_scope()
    other = Scope(tenant_id=str(uuid.uuid4()), project_id=scope.project_id)
    a = _entry(scope, "a", versions=("dv-1", "dv-2"))
    b = _entry(scope, "b", versions=("dv-2",))
    c = _entry(scope, "c", versions=("dv-3",), partition="q")
    foreign = _entry(other, "a", versions=("dv-1", "dv-2"))
    for item, owner in ((a, scope), (b, scope), (c, scope), (foreign, other)):
        await store.put(owner, item)
    _require(await store.invalidate_sources(scope, ()) == 0, "no versions invalidates nothing")
    _require(await store.invalidate_sources(scope, ("dv-unknown",)) == 0, "unknown version invalidates nothing")
    _require(await store.invalidate_sources(scope, ("dv-1", "dv-2")) == 2, "each entry counts once")
    _require(await store.get(scope, a.key) is None and await store.get(scope, b.key) is None, "dependents removed")
    _require(await store.get(scope, c.key) == c, "unrelated entry stays")
    _require(await store.get(other, foreign.key) == foreign, "invalidation must stay inside its scope")
    _require(await store.invalidate_sources(scope, ("dv-1", "dv-2")) == 0, "reverse index must be cleaned")
    # Deleting an entry also drops its reverse dependencies.
    d = _entry(scope, "d", versions=("dv-9",))
    await store.put(scope, d)
    await store.delete(scope, d.key)
    _require(await store.invalidate_sources(scope, ("dv-9",)) == 0, "delete must clean the reverse index")
    await store.put(scope, d)
    _require(await store.invalidate_sources(scope, ("dv-9",)) == 1, "re-stored entry is invalidated by its version")
    # Partition invalidation.
    e1 = _entry(scope, "e1", partition="x")
    e2 = _entry(scope, "e2", partition="x")
    e3 = _entry(scope, "e3", partition="y")
    f1 = _entry(other, "f1", partition="x")
    for item, owner in ((e1, scope), (e2, scope), (e3, scope), (f1, other)):
        await store.put(owner, item)
    _require(await store.invalidate_partition(scope, _hex("part-x")) == 2, "partition entries removed")
    _require(await store.get(scope, e3.key) == e3, "other partition stays")
    _require(await store.get(other, f1.key) == f1, "partition invalidation must stay inside its scope")
    _require(await store.invalidate_partition(scope, _hex("part-x")) == 0, "second partition invalidation is empty")
    # e1 and e2 are gone, so only e3 still depends on dv-1.
    _require(await store.invalidate_sources(scope, ("dv-1",)) == 1, "removed entries leave no dependencies")
    _require(await store.get(scope, e3.key) is None, "remaining dependent removed")


async def check_similar(factory: StoreFactory) -> None:
    store = await factory()
    scope = _new_scope()
    other = Scope(tenant_id=str(uuid.uuid4()), project_id=scope.project_id)
    partition = _hex("part-p")
    items = [
        _entry(scope, "best", vector=(1.0, 0.0), vector_key="emb-1"),
        _entry(scope, "mid", vector=(1.0, 1.0), vector_key="emb-1"),
        _entry(scope, "far", vector=(0.0, 1.0), vector_key="emb-1"),
        _entry(scope, "other_part", partition="q", vector=(1.0, 0.0), vector_key="emb-1"),
        _entry(scope, "other_vk", vector=(1.0, 0.0), vector_key="emb-2"),
        _entry(scope, "no_vector"),
    ]
    for item in items:
        await store.put(scope, item)
    await store.put(other, _entry(other, "foreign", vector=(1.0, 0.0), vector_key="emb-1"))
    result = await store.similar(scope, partition, "emb-1", (1.0, 0.0), 10)
    runs = [e.source_run_id for e, _ in result]
    _require(runs == ["run-best", "run-mid", "run-far"], "scope, partition, vector_key and order")
    scores = [s for _, s in result]
    _require(abs(scores[0] - 1.0) < 1e-9, "cosine of identical direction is 1")
    _require(abs(scores[1] - math.sqrt(0.5)) < 1e-9 and abs(scores[2]) < 1e-9, "cosine values")
    _require(scores == sorted(scores, reverse=True), "scores must be descending")
    _require(len(await store.similar(scope, partition, "emb-1", (1.0, 0.0), 2)) == 2, "k must limit results")
    _require(await store.similar(scope, partition, "emb-3", (1.0, 0.0), 5) == [], "unknown vector_key finds nothing")
    _require(await store.similar(scope, _hex("part-none"), "emb-1", (1.0, 0.0), 5) == [], "unknown partition")
    stranger = Scope(tenant_id=str(uuid.uuid4()), project_id=str(uuid.uuid4()))
    _require(await store.similar(stranger, partition, "emb-1", (1.0, 0.0), 5) == [], "scope isolation")
    only_other = await store.similar(other, partition, "emb-1", (1.0, 0.0), 5)
    _require([e.source_run_id for e, _ in only_other] == ["run-foreign"], "each scope sees only its own entries")
    # A different dimension under the same scope, partition and vector_key is a corrupt entry, not a miss.
    await store.put(scope, _entry(scope, "other_dim", vector=(1.0, 0.0, 0.0), vector_key="emb-1"))
    await _expect_cache_error(store.similar(scope, partition, "emb-1", (1.0, 0.0), 3), "cache_entry_invalid")
    await _expect_cache_error(store.similar(scope, partition, "emb-1", (1.0, 0.0), 0), "cache_policy_invalid")
    await _expect_cache_error(store.similar(scope, partition, "emb-1", (), 3), "cache_policy_invalid")


async def check_purge_expired(factory: StoreFactory) -> None:
    store = await factory()
    scope = _new_scope()
    other = Scope(tenant_id=str(uuid.uuid4()), project_id=scope.project_id)
    short = _entry(scope, "short", versions=("dv-1",), ttl=timedelta(minutes=1))
    edge = _entry(scope, "edge", versions=("dv-1",), ttl=timedelta(minutes=2))
    long = _entry(scope, "long", versions=("dv-1",), ttl=timedelta(hours=5))
    foreign = _entry(other, "short", ttl=timedelta(minutes=1))
    for item, owner in ((short, scope), (edge, scope), (long, scope), (foreign, other)):
        await store.put(owner, item)
    # get does not enforce expiry; the caller revalidates.
    _require(await store.get(scope, short.key) == short, "get must not enforce expiry")
    _require(await store.purge_expired(scope, _T0) == 0, "nothing expired yet")
    _require(await store.purge_expired(scope, _T0 + timedelta(minutes=2)) == 2, "expires_at <= now is purged")
    _require(await store.get(scope, short.key) is None and await store.get(scope, edge.key) is None, "purged")
    _require(await store.get(scope, long.key) == long, "unexpired entry stays")
    _require(await store.get(other, foreign.key) == foreign, "purge must stay inside its scope")
    _require(await store.invalidate_sources(scope, ("dv-1",)) == 1, "purge must clean the reverse index")
    await _expect_cache_error(store.purge_expired(scope, datetime(2026, 1, 1)), "cache_policy_invalid")


RESPONSE_CACHE_STORE_CHECKS = (
    check_round_trip_and_scope,
    check_exists_and_delete,
    check_invalidation,
    check_similar,
    check_purge_expired,
)
