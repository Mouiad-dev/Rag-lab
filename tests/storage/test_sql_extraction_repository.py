"""SqlExtractionRepository tests. Need a real PostgreSQL: set RAGF_TEST_DATABASE_URL (postgresql+psycopg://...).

Without it every test here is skipped (same convention as test_sql_repository.py).
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

if _URL:
    from rag_foundry_domain.testing import (
        EXTRACTION_REPOSITORY_CHECKS,
        CheckSkippedError,
        make_new_extraction,
        make_new_version,
        make_scope,
    )
    from rag_foundry_storage.sql.engine import create_sessionmaker
    from rag_foundry_storage.sql.extraction_repository import SqlExtractionRepository
    from rag_foundry_storage.sql.migrate import run_migrations
    from rag_foundry_storage.sql.repository import SqlDocumentRepository
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
    """ExtractionRepoFactory: truncate everything, seed scopes, return the (documents, extractions) pair."""

    def __init__(self) -> None:
        self.engines = []

    async def __call__(self, scopes):
        maker = self.maker()
        async with maker() as session, session.begin():
            await session.execute(text("TRUNCATE extraction_views, document_versions, documents, projects, tenants"))
        docs = SqlDocumentRepository(maker)
        for scope in scopes:
            await docs.ensure_local_scope(scope)
        return docs, SqlExtractionRepository(maker)

    def maker(self):
        maker = create_sessionmaker(_url())
        self.engines.append(maker.kw["bind"])
        return maker

    async def close(self) -> None:
        for engine in self.engines:
            await engine.dispose()


@pytest.mark.parametrize(
    "check", EXTRACTION_REPOSITORY_CHECKS if _URL else [None], ids=lambda c: getattr(c, "__name__", "none")
)
@run
async def test_extraction_repository_check(check):
    factory = _Factory()
    try:
        await check(factory)
    except CheckSkippedError as exc:
        pytest.skip(str(exc))
    finally:
        await factory.close()


@run
async def test_update_of_a_view_is_rejected_by_the_database():
    factory = _Factory()
    try:
        scope = make_scope()
        docs, repo = await factory([scope])
        ver = (await docs.create_document_or_get(scope, make_new_version("x"), display_name="d")).version
        view = (await repo.create_view_or_get(scope, make_new_extraction(ver))).view
        with pytest.raises(DBAPIError):
            async with factory.maker()() as session, session.begin():
                await session.execute(
                    text("UPDATE extraction_views SET profile_id = 'changed' WHERE id = :i"), {"i": view.id}
                )
    finally:
        await factory.close()


@run
async def test_other_scopes_version_is_rejected_by_the_foreign_key():
    factory = _Factory()
    try:
        a, b = make_scope(), make_scope()
        docs, _ = await factory([a, b])
        ver = (await docs.create_document_or_get(a, make_new_version("x"), display_name="d")).version
        with pytest.raises(DBAPIError):  # scope b with a's document and version
            async with factory.maker()() as session, session.begin():
                await session.execute(
                    text(
                        "INSERT INTO extraction_views (id, tenant_id, project_id, document_id, document_version_id,"
                        " extractor, extractor_version, profile_id, unicode_version, canonical_sha256, char_count,"
                        " block_count, payload_sha256, payload_key, payload_schema_version) VALUES"
                        " (gen_random_uuid(), :t, :p, :d, :v, 'plain-text', '1', 'canonical-v1', '15.1.0', :h, 1, 1,"
                        " :h, 'k', 1)"
                    ),
                    {"t": b.tenant_id, "p": b.project_id, "d": ver.document_id, "v": ver.id, "h": "a" * 64},
                )
    finally:
        await factory.close()


@run
async def test_check_constraints_reject_bad_rows():
    factory = _Factory()
    try:
        scope = make_scope()
        docs, _ = await factory([scope])
        ver = (await docs.create_document_or_get(scope, make_new_version("x"), display_name="d")).version
        insert_sql = text(
            "INSERT INTO extraction_views (id, tenant_id, project_id, document_id, document_version_id,"
            " extractor, extractor_version, profile_id, unicode_version, canonical_sha256, char_count,"
            " block_count, page_count, payload_sha256, payload_key, payload_schema_version) VALUES"
            " (gen_random_uuid(), :t, :p, :d, :v, :e, '1', 'canonical-v1', '15.1.0', :h, :c, 1, :pc, :h, 'k', :s)"
        )
        base = {
            "t": scope.tenant_id,
            "p": scope.project_id,
            "d": ver.document_id,
            "v": ver.id,
            "e": "plain-text",
            "h": "a" * 64,
            "c": 1,
            "pc": None,
            "s": 1,
        }
        for override in ({"h": "A" * 64}, {"c": 0}, {"pc": 0}, {"s": 2}, {"e": "Bad Name"}):
            with pytest.raises(DBAPIError):
                async with factory.maker()() as session, session.begin():
                    await session.execute(insert_sql, {**base, **override})
    finally:
        await factory.close()


def test_downgrade_to_0001_then_upgrade_again():
    try:
        run_migrations(_url(), "downgrade", "0001_document_intake")
    finally:
        # Always restore head so later db tests in the session never see a half-migrated schema.
        run_migrations(_url(), "upgrade", "head")
