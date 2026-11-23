"""PostgreSQL tests for the response-cache store (migration 0014) and its schema guards.

The shared checks use logical (tenant, project) pairs that share a project id and source ids such as "dv-1".
The schema forbids both (projects are owned by one tenant; sources reference real document versions), so the
checks run through `_LogicalStore`, which maps logical scopes and source ids to real, seeded rows and maps
returned entries back. The checks themselves are unchanged.
"""

import asyncio
import functools
import hashlib
import os
import sys
import uuid
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import SecretStr
from rag_foundry_domain.documents import Scope
from rag_foundry_domain.errors import CacheError
from rag_foundry_domain.response_cache import RESPONSE_CACHE_STORE_CHECKS, CacheEntry, ResponseCacheStore
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError

_URL = os.environ.get("RAGF_TEST_DATABASE_URL")
pytestmark = [pytest.mark.db, pytest.mark.skipif(not _URL, reason="RAGF_TEST_DATABASE_URL is not set")]

if _URL:
    from rag_foundry_storage.sql.database import create_database
    from rag_foundry_storage.sql.migrate import run_migrations
    from rag_foundry_storage.sql.response_cache_store import SqlResponseCacheStore

_T0 = datetime(2026, 1, 1, tzinfo=UTC)


def run(coro_fn):
    @functools.wraps(coro_fn)
    def wrapper(*args, **kwargs):
        factory = asyncio.SelectorEventLoop if sys.platform == "win32" else None
        with asyncio.Runner(loop_factory=factory) as runner:
            return runner.run(coro_fn(*args, **kwargs))

    return wrapper


def _url():
    assert _URL
    return SecretStr(_URL)


@pytest.fixture(scope="module", autouse=True)
def migrated():
    run_migrations(_url(), "upgrade", "head")


def _uuid4(seed: str) -> str:
    return str(uuid.UUID(bytes=hashlib.sha256(seed.encode("utf-8")).digest()[:16], version=4))


def _hex(seed: str) -> str:
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()


async def _seed(db, tenant_id: str, project_id: str, version_ids: tuple[str, ...] = ()) -> None:
    """Insert tenant, project and one single-version document per version id (idempotent)."""
    async with db.sessionmaker() as session, session.begin():
        await session.execute(text("INSERT INTO tenants (id) VALUES (:t) ON CONFLICT DO NOTHING"), {"t": tenant_id})
        await session.execute(
            text("INSERT INTO projects (id, tenant_id) VALUES (:p, :t) ON CONFLICT DO NOTHING"),
            {"p": project_id, "t": tenant_id},
        )
        for version_id in version_ids:
            await session.execute(
                text(
                    "INSERT INTO documents (id, tenant_id, project_id, display_name) "
                    "VALUES (:v, :t, :p, 'doc') ON CONFLICT DO NOTHING"
                ),
                {"v": version_id, "t": tenant_id, "p": project_id},
            )
            await session.execute(
                text(
                    "INSERT INTO document_versions (id, tenant_id, project_id, document_id, version_number, sha256, "
                    "size_bytes, media_type, text_encoding, original_filename, object_key) "
                    "VALUES (:v, :t, :p, :v, 1, :sha, 1, 'text/plain', 'utf-8', 'a.txt', :k) ON CONFLICT DO NOTHING"
                ),
                {"v": version_id, "t": tenant_id, "p": project_id, "sha": _hex(version_id), "k": "obj/" + version_id},
            )


class _LogicalStore:
    """Adapter that lets the shared checks (logical scopes, ids like "dv-1") run against real foreign keys."""

    def __init__(self, db) -> None:
        self._db = db
        self._store = SqlResponseCacheStore(db)
        self._project_back: dict[str, str] = {}
        self._version_back: dict[str, str] = {}

    def _scope(self, scope: Scope) -> Scope:
        project = _uuid4(f"project/{scope.tenant_id}/{scope.project_id}")
        self._project_back[project] = scope.project_id
        return Scope(tenant_id=scope.tenant_id, project_id=project)

    def _version(self, scope: Scope, name: str) -> str:
        version = _uuid4(f"version/{scope.tenant_id}/{scope.project_id}/{name}")
        self._version_back[version] = name
        return version

    def _entry_in(self, entry: CacheEntry) -> CacheEntry:
        logical = Scope(tenant_id=entry.tenant_id, project_id=entry.project_id)
        return entry.model_copy(
            update={
                "project_id": self._scope(logical).project_id,
                "source_version_ids": tuple(self._version(logical, name) for name in entry.source_version_ids),
            }
        )

    def _entry_out(self, entry: CacheEntry | None) -> CacheEntry | None:
        if entry is None:
            return None
        return entry.model_copy(
            update={
                "project_id": self._project_back[entry.project_id],
                "source_version_ids": tuple(self._version_back[v] for v in entry.source_version_ids),
            }
        )

    async def get(self, scope, key):
        return self._entry_out(await self._store.get(self._scope(scope), key))

    async def put(self, scope, entry):
        physical = self._entry_in(entry)
        await _seed(self._db, physical.tenant_id, physical.project_id, physical.source_version_ids)
        mine = self._scope(scope)
        await _seed(self._db, mine.tenant_id, mine.project_id)
        await self._store.put(mine, physical)

    async def delete(self, scope, key):
        return await self._store.delete(self._scope(scope), key)

    async def similar(self, scope, partition, vector_key, vector, k):
        found = await self._store.similar(self._scope(scope), partition, vector_key, vector, k)
        return [(self._entry_out(entry), score) for entry, score in found]

    async def invalidate_sources(self, scope, version_ids):
        return await self._store.invalidate_sources(
            self._scope(scope), tuple(self._version(scope, name) for name in version_ids)
        )

    async def invalidate_partition(self, scope, partition):
        return await self._store.invalidate_partition(self._scope(scope), partition)

    async def purge_expired(self, scope, now):
        return await self._store.purge_expired(self._scope(scope), now)


@pytest.mark.parametrize("check", RESPONSE_CACHE_STORE_CHECKS, ids=lambda check: check.__name__)
@run
async def test_sql_response_cache_store_shared_checks(check):
    db = create_database(_url())
    try:

        async def factory() -> ResponseCacheStore:
            return _LogicalStore(db)

        await check(factory)
    finally:
        await db.dispose()


def _real_entry(tenant_id: str, project_id: str, seed: str, versions: tuple[str, ...]) -> CacheEntry:
    return CacheEntry(
        key=_hex("key-" + seed),
        partition=_hex("part-" + seed),
        tenant_id=tenant_id,
        project_id=project_id,
        response_json='{"answer": "stored"}',
        source_run_id="run-" + seed,
        source_version_ids=versions,
        created_at=_T0,
        expires_at=_T0 + timedelta(hours=1),
    )


_INSERT_ENTRY = text(
    """INSERT INTO response_cache_entries
    (tenant_id, project_id, key, partition, response_json, source_run_id, created_at, expires_at,
     features_json, question_vector, vector_key)
    VALUES (:t, :p, :k, :part, '{}', 'run', :created, :expires, NULL, CAST(:vec AS vector), :vk)"""
)
_INSERT_SOURCE = text(
    "INSERT INTO response_cache_sources (tenant_id, project_id, key, document_version_id, ordinal) "
    "VALUES (:t, :p, :k, :v, 0)"
)


def _params(tenant_id: str, project_id: str, **overrides):
    base = {
        "t": tenant_id,
        "p": project_id,
        "k": _hex("raw-key"),
        "part": _hex("raw-part"),
        "created": _T0,
        "expires": _T0 + timedelta(hours=1),
        "vec": None,
        "vk": None,
    }
    return base | overrides


async def _rejected(db, statement, params) -> None:
    async with db.sessionmaker() as session:
        with pytest.raises(DBAPIError):
            async with session.begin():
                await session.execute(statement, params)


async def _count(db, table: str, tenant_id: str) -> int:
    async with db.sessionmaker() as session:
        result = await session.execute(text(f"SELECT count(*) FROM {table} WHERE tenant_id = :t"), {"t": tenant_id})
        return int(result.scalar_one())


@run
async def test_updates_are_rejected_on_both_tables():
    db = create_database(_url())
    try:
        tenant, project, version = _uuid4("u/t"), _uuid4("u/p"), _uuid4("u/v")
        await _seed(db, tenant, project, (version,))
        store = SqlResponseCacheStore(db)
        entry = _real_entry(tenant, project, "update", (version,))
        await store.put(Scope(tenant_id=tenant, project_id=project), entry)
        await _rejected(
            db, text("UPDATE response_cache_entries SET response_json = '{}' WHERE tenant_id = :t"), {"t": tenant}
        )
        await _rejected(db, text("UPDATE response_cache_sources SET ordinal = 1 WHERE tenant_id = :t"), {"t": tenant})
        assert await store.get(Scope(tenant_id=tenant, project_id=project), entry.key) == entry
    finally:
        await db.dispose()


@run
async def test_source_row_for_another_tenants_version_is_rejected():
    db = create_database(_url())
    try:
        t_a, p_a, v_a = _uuid4("x/ta"), _uuid4("x/pa"), _uuid4("x/va")
        t_b, p_b = _uuid4("x/tb"), _uuid4("x/pb")
        await _seed(db, t_a, p_a, (v_a,))
        await _seed(db, t_b, p_b)
        async with db.sessionmaker() as session, session.begin():
            await session.execute(_INSERT_ENTRY, _params(t_b, p_b))
        await _rejected(db, _INSERT_SOURCE, _params(t_b, p_b, v=v_a))
        assert await _count(db, "response_cache_sources", t_b) == 0
    finally:
        await db.dispose()


@run
async def test_source_row_with_cross_scope_entry_reference_is_rejected():
    db = create_database(_url())
    try:
        t_a, p_a, v_a = _uuid4("c/ta"), _uuid4("c/pa"), _uuid4("c/va")
        t_b, p_b, v_b = _uuid4("c/tb"), _uuid4("c/pb"), _uuid4("c/vb")
        await _seed(db, t_a, p_a, (v_a,))
        await _seed(db, t_b, p_b, (v_b,))
        async with db.sessionmaker() as session, session.begin():
            await session.execute(_INSERT_ENTRY, _params(t_a, p_a))
        # Scope B has a valid version but no entry with this key: the entry foreign key must refuse the row.
        await _rejected(db, _INSERT_SOURCE, _params(t_b, p_b, v=v_b))
        # An entry for a project that does not belong to the tenant is refused by the projects foreign key.
        await _rejected(db, _INSERT_ENTRY, _params(t_b, p_a, k=_hex("other")))
    finally:
        await db.dispose()


@run
async def test_bad_patterns_and_vector_pairing_are_rejected():
    db = create_database(_url())
    try:
        tenant, project = _uuid4("b/t"), _uuid4("b/p")
        await _seed(db, tenant, project)
        await _rejected(db, _INSERT_ENTRY, _params(tenant, project, k="Z" * 64))
        await _rejected(db, _INSERT_ENTRY, _params(tenant, project, k=_hex("short")[:63]))
        await _rejected(db, _INSERT_ENTRY, _params(tenant, project, part="not-hex"))
        await _rejected(db, _INSERT_ENTRY, _params(tenant, project, vec="[1,2]"))
        await _rejected(db, _INSERT_ENTRY, _params(tenant, project, vk="emb-1"))
        await _rejected(db, _INSERT_ENTRY, _params(tenant, project, expires=_T0))
        assert await _count(db, "response_cache_entries", tenant) == 0
        async with db.sessionmaker() as session, session.begin():
            await session.execute(_INSERT_ENTRY, _params(tenant, project, vec="[1,2]", vk="emb-1"))
        assert await _count(db, "response_cache_entries", tenant) == 1
    finally:
        await db.dispose()


@run
async def test_delete_cascades_to_sources_and_errors_are_coded():
    db = create_database(_url())
    try:
        tenant, project = _uuid4("d/t"), _uuid4("d/p")
        versions = (_uuid4("d/v1"), _uuid4("d/v2"))
        scope = Scope(tenant_id=tenant, project_id=project)
        store = SqlResponseCacheStore(db)
        entry = _real_entry(tenant, project, "cascade", versions)
        # No project row yet: the scope foreign key reports a scope problem, not a raw database error.
        with pytest.raises(CacheError) as unknown:
            await store.put(scope, entry)
        assert unknown.value.code == "cache_scope_mismatch"
        await _seed(db, tenant, project, versions)
        await store.put(scope, entry)
        assert await _count(db, "response_cache_sources", tenant) == 2
        assert (await store.get(scope, entry.key)) == entry  # source order is preserved
        # A version of another project is refused by the insert trigger and nothing is stored.
        other_version = _uuid4("d/other")
        await _seed(db, _uuid4("d/t2"), _uuid4("d/p2"), (other_version,))
        foreign = _real_entry(tenant, project, "foreign", (other_version,))
        with pytest.raises(CacheError) as outside:
            await store.put(scope, foreign)
        assert outside.value.code == "cache_scope_mismatch"
        assert await store.get(scope, foreign.key) is None
        # Source ids that are not canonical UUIDs cannot be stored.
        with pytest.raises(CacheError) as invalid:
            await store.put(scope, _real_entry(tenant, project, "bad-id", ("dv-1",)))
        assert invalid.value.code == "cache_entry_invalid"
        assert await store.delete(scope, entry.key) is True
        assert await _count(db, "response_cache_sources", tenant) == 0
        assert await store.invalidate_sources(scope, ("dv-1", versions[0])) == 0
    finally:
        await db.dispose()


@run
async def test_zero_vector_ranks_as_similarity_zero_like_the_in_memory_store():
    db = create_database(_url())
    try:
        tenant, project = _uuid4("z/t"), _uuid4("z/p")
        scope = Scope(tenant_id=tenant, project_id=project)
        version = _uuid4("z/v")
        await _seed(db, tenant, project, (version,))
        store = SqlResponseCacheStore(db)
        base = _real_entry(tenant, project, "zero", (version,)).model_dump()
        zero = CacheEntry(**{**base, "question_vector": (0.0, 0.0), "vector_key": "emb-1"})
        opposite = CacheEntry(
            **{**base, "key": _hex("key-opposite"), "question_vector": (-1.0, 0.0), "vector_key": "emb-1"}
        )
        await store.put(scope, zero)
        await store.put(scope, opposite)
        # NaN distance must not sort last: cosine 0.0 beats cosine -1.0, so k=1 returns the zero vector.
        found = await store.similar(scope, zero.partition, "emb-1", (1.0, 0.0), 1)
        assert [(entry.key, score) for entry, score in found] == [(zero.key, 0.0)]
        both = await store.similar(scope, zero.partition, "emb-1", (1.0, 0.0), 2)
        assert [entry.key for entry, _ in both] == [zero.key, opposite.key]
        assert both[1][1] == pytest.approx(-1.0)
    finally:
        await db.dispose()


def test_response_cache_migration_downgrade_and_upgrade():
    run_migrations(_url(), "downgrade", "0013_knowledge_graphs")
    try:

        @run
        async def tables_exist() -> list[str]:
            db = create_database(_url())
            try:
                async with db.sessionmaker() as session:
                    rows = await session.execute(
                        text("SELECT tablename FROM pg_tables WHERE tablename LIKE 'response_cache_%'")
                    )
                    return list(rows.scalars().all())
            finally:
                await db.dispose()

        assert tables_exist() == []
    finally:
        run_migrations(_url(), "upgrade", "head")

    @run
    async def usable_again() -> None:
        db = create_database(_url())
        try:
            tenant, project, version = _uuid4("m/t"), _uuid4("m/p"), _uuid4("m/v")
            await _seed(db, tenant, project, (version,))
            scope = Scope(tenant_id=tenant, project_id=project)
            entry = _real_entry(tenant, project, "again", (version,))
            store = SqlResponseCacheStore(db)
            await store.put(scope, entry)
            assert await store.get(scope, entry.key) == entry
        finally:
            await db.dispose()

    usable_again()
