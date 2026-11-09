"""SQL repository tests. They need a real PostgreSQL: set RAGF_TEST_DATABASE_URL (postgresql+psycopg://...).

The variable is deliberately not a RAG_FOUNDRY_* name, so load_settings() never sees it. Without it
every test here is skipped. The real-database run is part of task P3.1-06 (compose `testdb`, CI).
"""

import asyncio
import functools
import os
import sys

import pytest
from pydantic import SecretStr

_URL = os.environ.get("RAGF_TEST_DATABASE_URL")

pytestmark = [
    pytest.mark.db,
    pytest.mark.skipif(not _URL, reason="RAGF_TEST_DATABASE_URL is not set"),
]

if _URL:  # imports need the `sql` extra; keep collection safe when it is absent and no URL is given
    from rag_foundry_domain.testing import REPOSITORY_CHECKS, CheckSkippedError, make_new_version, make_scope
    from rag_foundry_domain.uow_adapters import per_call_documents
    from rag_foundry_storage.sql.database import create_database
    from rag_foundry_storage.sql.migrate import run_migrations
    from rag_foundry_storage.sql.uow import ensure_local_scope
    from sqlalchemy import text
    from sqlalchemy.exc import DBAPIError


def run(coro_fn):
    """Run an async test body on its own loop (psycopg async cannot use the Windows Proactor loop)."""

    @functools.wraps(coro_fn)
    def wrapper(*args, **kwargs):
        factory = asyncio.SelectorEventLoop if sys.platform == "win32" else None
        with asyncio.Runner(loop_factory=factory) as runner:
            return runner.run(coro_fn(*args, **kwargs))

    return wrapper


def _url() -> SecretStr:
    assert _URL
    return SecretStr(_URL)


@pytest.fixture(scope="module", autouse=True)
def migrated():
    run_migrations(_url(), "upgrade", "head")


class _Factory:
    """RepoFactory: truncate everything, then create the requested scopes. Disposes engines on close."""

    def __init__(self) -> None:
        self.databases = []

    async def __call__(self, scopes, *, clock=None):
        if clock is not None:
            raise CheckSkippedError("the SQL repository uses the database clock")
        db = create_database(_url())
        maker = db.sessionmaker
        self.databases.append(db)
        async with maker() as session, session.begin():
            await session.execute(
                text(
                    "TRUNCATE runtime_setting_overrides, chunk_sparse_vectors, chunks, index_version_views, "
                    "index_versions, extraction_views, "
                    "document_versions, documents, projects, tenants"
                )
            )
        repo = per_call_documents(db.uow)
        for scope in scopes:
            await ensure_local_scope(db.uow, scope)
        return repo

    async def close(self) -> None:
        for db in self.databases:
            await db.dispose()


@pytest.mark.parametrize("check", REPOSITORY_CHECKS if _URL else [None], ids=lambda c: getattr(c, "__name__", "none"))
@run
async def test_repository_check(check):
    factory = _Factory()
    try:
        await check(factory)
    except CheckSkippedError as exc:
        pytest.skip(str(exc))
    finally:
        await factory.close()


@run
async def test_ten_way_concurrent_same_bytes_creates_one_version():
    factory = _Factory()
    try:
        scope = make_scope()
        repo = await factory([scope])
        outcomes = await asyncio.gather(
            *(repo.create_document_or_get(scope, make_new_version("same"), display_name="d") for _ in range(10))
        )
        assert sum(o.created_document for o in outcomes) == 1
        assert len({o.version.id for o in outcomes}) == 1
        assert len(await repo.list_versions(scope, outcomes[0].document.id)) == 1
    finally:
        await factory.close()


@run
async def test_concurrent_new_versions_are_numbered_without_gaps():
    factory = _Factory()
    try:
        scope = make_scope()
        repo = await factory([scope])
        first = await repo.create_document_or_get(scope, make_new_version("v0"), display_name="d")
        await asyncio.gather(
            *(repo.add_version_or_get(scope, first.document.id, make_new_version(f"n{i}")) for i in range(8))
        )
        numbers = [v.version_number for v in await repo.list_versions(scope, first.document.id)]
        assert numbers == list(range(1, 10))
    finally:
        await factory.close()


@run
async def test_update_of_a_version_is_rejected_by_the_database():
    factory = _Factory()
    try:
        scope = make_scope()
        repo = await factory([scope])
        out = await repo.create_document_or_get(scope, make_new_version("x"), display_name="d")
        db = create_database(_url())
        maker = db.sessionmaker
        factory.databases.append(db)
        with pytest.raises(DBAPIError):
            async with maker() as session, session.begin():
                await session.execute(
                    text("UPDATE document_versions SET original_filename = 'changed' WHERE id = :i"),
                    {"i": out.version.id},
                )
    finally:
        await factory.close()


@run
async def test_document_scope_cannot_change_and_cross_scope_version_is_rejected():
    factory = _Factory()
    try:
        a, b = make_scope(), make_scope()
        repo = await factory([a, b])
        out = await repo.create_document_or_get(a, make_new_version("x"), display_name="d")
        db = create_database(_url())
        maker = db.sessionmaker
        factory.databases.append(db)
        with pytest.raises(DBAPIError):
            async with maker() as session, session.begin():
                await session.execute(
                    text("UPDATE documents SET project_id = :p, tenant_id = :t WHERE id = :i"),
                    {"p": b.project_id, "t": b.tenant_id, "i": out.document.id},
                )
        with pytest.raises(DBAPIError):  # composite FK: version points at scope b but document belongs to a
            async with maker() as session, session.begin():
                await session.execute(
                    text(
                        "INSERT INTO document_versions (id, tenant_id, project_id, document_id, version_number, sha256,"
                        " size_bytes, media_type, text_encoding, original_filename, object_key) VALUES"
                        " (gen_random_uuid(), :t, :p, :d, 2, :h, 1, 'text/plain', 'utf-8', 'a.txt', 'k')"
                    ),
                    {"t": b.tenant_id, "p": b.project_id, "d": out.document.id, "h": "a" * 64},
                )
    finally:
        await factory.close()


@run
async def test_check_constraints_reject_bad_rows():
    factory = _Factory()
    try:
        scope = make_scope()
        repo = await factory([scope])
        out = await repo.create_document_or_get(scope, make_new_version("x"), display_name="d")
        db = create_database(_url())
        maker = db.sessionmaker
        factory.databases.append(db)
        insert_sql = text(
            "INSERT INTO document_versions (id, tenant_id, project_id, document_id, version_number, sha256,"
            " size_bytes, media_type, text_encoding, original_filename, object_key) VALUES"
            " (gen_random_uuid(), :t, :p, :d, 2, :h, :s, :m, 'utf-8', 'a.txt', 'k')"
        )
        base = {
            "t": scope.tenant_id,
            "p": scope.project_id,
            "d": out.document.id,
            "h": "b" * 64,
            "s": 1,
            "m": "text/plain",
        }
        for override in ({"h": "B" * 64}, {"h": "b" * 63}, {"s": 0}, {"m": "image/png"}):
            with pytest.raises(DBAPIError):
                async with maker() as session, session.begin():
                    await session.execute(insert_sql, {**base, **override})
    finally:
        await factory.close()


def test_downgrade_to_base_then_upgrade_again():
    run_migrations(_url(), "downgrade", "base")
    run_migrations(_url(), "upgrade", "head")
