# ruff: noqa: E501
"""SQL IndexRepository checks and database-enforced immutability tests."""

import asyncio
import functools
import os
import sys

import pytest
from pydantic import SecretStr

_URL = os.environ.get("RAGF_TEST_DATABASE_URL")
pytestmark = [pytest.mark.db, pytest.mark.skipif(not _URL, reason="RAGF_TEST_DATABASE_URL is not set")]

if _URL:
    from rag_foundry_domain.errors import NotFoundError
    from rag_foundry_domain.index_testing import INDEX_REPOSITORY_CHECKS
    from rag_foundry_domain.testing import CheckSkippedError, make_new_extraction, make_new_version, make_scope
    from rag_foundry_domain.uow_adapters import per_call_documents, per_call_extractions, per_call_indexes
    from rag_foundry_storage.sql.database import create_database
    from rag_foundry_storage.sql.migrate import run_migrations
    from rag_foundry_storage.sql.uow import ensure_local_scope
    from sqlalchemy import text
    from sqlalchemy.exc import DBAPIError


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


class _Factory:
    def __init__(self):
        self.databases = []

    def database(self):
        db = create_database(_url())
        self.databases.append(db)
        return db

    async def __call__(self, scopes):
        db = self.database()
        maker = db.sessionmaker
        async with maker() as session, session.begin():
            await session.execute(
                text(
                    "TRUNCATE runtime_setting_overrides, chunks, index_version_views, index_versions, extraction_views, document_versions, documents, projects, tenants"
                )
            )
        docs = per_call_documents(db.uow)
        for scope in scopes:
            await ensure_local_scope(db.uow, scope)
        return docs, per_call_extractions(db.uow), per_call_indexes(db.uow)

    async def close(self):
        for db in self.databases:
            await db.dispose()


@pytest.mark.parametrize(
    "check", INDEX_REPOSITORY_CHECKS if _URL else [None], ids=lambda c: getattr(c, "__name__", "none")
)
@run
async def test_index_repository_check(check):
    factory = _Factory()
    try:
        await check(factory)
    except CheckSkippedError as exc:
        pytest.skip(str(exc))
    finally:
        await factory.close()


@run
async def test_database_guards_and_migration_roundtrip():
    factory = _Factory()
    try:
        scope, other = make_scope(), make_scope()
        docs, views, repo = await factory([scope, other])
        out = await docs.create_document_or_get(scope, make_new_version("index-guard-a"), display_name="d")
        view = (await views.create_view_or_get(scope, make_new_extraction(out.version, tag="guard-a"))).view
        foreign_out = await docs.create_document_or_get(other, make_new_version("index-guard-b"), display_name="d")
        foreign_view = (
            await views.create_view_or_get(other, make_new_extraction(foreign_out.version, tag="guard-b"))
        ).view
        from rag_foundry_domain.index_testing import make_chunk, make_new_index_version

        version = await repo.create_index_version(scope, make_new_index_version((view.id,)))
        async with factory.database().sessionmaker() as session:
            version_count = await session.scalar(text("SELECT count(*) FROM index_versions"))
        with pytest.raises(NotFoundError):
            await repo.create_index_version(scope, make_new_index_version((foreign_view.id,)))
        async with factory.database().sessionmaker() as session:
            assert await session.scalar(text("SELECT count(*) FROM index_versions")) == version_count

        async def rejected(sql, params, message):
            with pytest.raises(DBAPIError, match=message):
                async with factory.database().sessionmaker() as session, session.begin():
                    await session.execute(text(sql), params)

        base = {
            "id": "00000000-0000-4000-8000-000000000001",
            "t": scope.tenant_id,
            "p": scope.project_id,
            "i": version.id,
            "v": view.id,
            "h": "a" * 64,
        }
        common = "INSERT INTO chunks (id,tenant_id,project_id,index_version_id,extraction_view_id,ordinal,start_cp,end_cp,split_level,block_ids,text,text_sha256,embedding) VALUES (:id,:t,:p,:i,:v,0,0,1,0,ARRAY['b00000'],'x',:h,CAST(:e AS vector))"
        await rejected(
            "INSERT INTO chunks (id,tenant_id,project_id,index_version_id,extraction_view_id,ordinal,start_cp,end_cp,split_level,block_ids,text,text_sha256,embedding,parent_start_cp,parent_end_cp) VALUES (gen_random_uuid(),:t,:p,:i,:v,0,0,1,0,ARRAY['b00000'],'x',:h,CAST('[1,0,0,0]' AS vector),1,2)",
            {key: base[key] for key in ("t", "p", "i", "v", "h")},
            "ck_chunks_parent_span",
        )
        await rejected(common, {**base, "e": "[1,0,0]"}, "embedding dimension mismatch")
        await rejected(
            "UPDATE index_versions SET state='failed', corpus_sha256=:h WHERE id=:i",
            {"h": "b" * 64, "i": version.id},
            "invalid index version update",
        )
        second = await repo.create_index_version(scope, make_new_index_version((view.id,)))
        async with factory.database().sessionmaker() as session, session.begin():
            await session.execute(text("UPDATE index_versions SET state='failed' WHERE id=:i"), {"i": second.id})
        await rejected(
            "INSERT INTO chunks (id,tenant_id,project_id,index_version_id,extraction_view_id,ordinal,start_cp,end_cp,split_level,block_ids,text,text_sha256,embedding) VALUES (gen_random_uuid(),:t,:p,:i,:v,0,0,1,0,ARRAY['b00000'],'x',:h,CAST('[1,0,0,0]' AS vector))",
            {"t": scope.tenant_id, "p": scope.project_id, "i": version.id, "v": foreign_view.id, "h": "a" * 64},
            "fk_chunks_version_view",
        )

        await repo.add_chunks(scope, version.id, (make_chunk(view.id, 0, (1, 0, 0, 0)),))
        await repo.mark_ready(scope, version.id)
        await rejected(
            common,
            {**base, "id": "00000000-0000-4000-8000-000000000002", "e": "[1,0,0,0]"},
            "index version is not building",
        )
        await rejected(
            "UPDATE chunks SET text='z' WHERE index_version_id=:i", {"i": version.id}, "index rows are immutable"
        )
        await rejected(
            "UPDATE index_versions SET corpus_sha256=:h WHERE id=:i",
            {"h": "b" * 64, "i": version.id},
            "invalid index version update",
        )
        # Alembic runs its own event loop, so it must not run on this test's loop.
        await asyncio.to_thread(run_migrations, _url(), "downgrade", "0002_extraction_views")
        async with factory.databases[0].engine.connect() as connection:
            assert await connection.scalar(text("SELECT to_regclass('chunks') IS NULL"))
            assert await connection.scalar(text("SELECT to_regclass('index_versions') IS NULL"))
        await asyncio.to_thread(run_migrations, _url(), "upgrade", "head")
    finally:
        await factory.close()
