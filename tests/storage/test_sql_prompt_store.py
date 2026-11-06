"""PostgreSQL tests for prompt templates and activation history."""

import asyncio
import functools
import os
import sys

import pytest
from pydantic import SecretStr
from rag_foundry_domain.errors import PromptError
from rag_foundry_domain.prompt_testing import PROMPT_STORE_CHECKS
from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import DBAPIError

_URL = os.environ.get("RAGF_TEST_DATABASE_URL")
pytestmark = [pytest.mark.db, pytest.mark.skipif(not _URL, reason="RAGF_TEST_DATABASE_URL is not set")]

if _URL:
    from rag_foundry_storage.sql.engine import create_sessionmaker
    from rag_foundry_storage.sql.migrate import run_migrations
    from rag_foundry_storage.sql.prompt_store import SqlPromptTemplateStore


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


@pytest.mark.parametrize("check", PROMPT_STORE_CHECKS, ids=lambda check: check.__name__)
@run
async def test_sql_prompt_store_shared_checks(check):
    maker = create_sessionmaker(_url())
    try:
        await check(SqlPromptTemplateStore(maker))
    finally:
        await maker.kw["bind"].dispose()


@run
async def test_prompt_migration_seeds_and_active_templates():
    maker = create_sessionmaker(_url())
    try:
        async with maker() as session:
            templates = (await session.execute(text("SELECT id FROM prompt_templates ORDER BY id"))).scalars().all()
            activations = (
                await session.execute(text("SELECT purpose, template_id FROM prompt_activations ORDER BY seq"))
            ).all()
        assert set(templates) >= {"grounded-answer-v1", "grounded-answer-v2", "ungrounded-v1"}
        assert activations[:3] == [
            ("grounded", "grounded-answer-v1"),
            ("grounded", "grounded-answer-v2"),
            ("ungrounded", "ungrounded-v1"),
        ]
        assert await SqlPromptTemplateStore(maker).active_id("grounded") == "grounded-answer-v2"
        assert await SqlPromptTemplateStore(maker).active_id("ungrounded") == "ungrounded-v1"
    finally:
        await maker.kw["bind"].dispose()


@run
async def test_prompt_template_update_is_rejected():
    maker = create_sessionmaker(_url())
    try:
        async with maker() as session:
            with pytest.raises(DBAPIError):
                async with session.begin():
                    await session.execute(
                        text("UPDATE prompt_templates SET family = family WHERE id = 'grounded-answer-v2'")
                    )
    finally:
        await maker.kw["bind"].dispose()


@run
async def test_prompt_template_delete_is_rejected():
    maker = create_sessionmaker(_url())
    try:
        async with maker() as session:
            with pytest.raises(DBAPIError):
                async with session.begin():
                    await session.execute(text("DELETE FROM prompt_templates WHERE id = 'grounded-answer-v2'"))
    finally:
        await maker.kw["bind"].dispose()


@run
async def test_prompt_activation_update_is_rejected():
    maker = create_sessionmaker(_url())
    try:
        async with maker() as session:
            with pytest.raises(DBAPIError):
                async with session.begin():
                    await session.execute(
                        text("UPDATE prompt_activations SET decision_ref = decision_ref WHERE seq = 1")
                    )
    finally:
        await maker.kw["bind"].dispose()


@run
async def test_prompt_activation_delete_is_rejected():
    maker = create_sessionmaker(_url())
    try:
        async with maker() as session:
            with pytest.raises(DBAPIError):
                async with session.begin():
                    await session.execute(text("DELETE FROM prompt_activations WHERE seq = 1"))
    finally:
        await maker.kw["bind"].dispose()


@run
async def test_prompt_tables_reject_truncate():
    maker = create_sessionmaker(_url())
    try:
        for table in ("prompt_activations", "prompt_templates"):
            async with maker() as session:
                with pytest.raises(DBAPIError, match="prompt rows are immutable"):
                    async with session.begin():
                        await session.execute(text(f"TRUNCATE {table} CASCADE"))
    finally:
        await maker.kw["bind"].dispose()


@run
async def test_database_rejects_incorrect_prompt_hash():
    maker = create_sessionmaker(_url())
    try:
        async with maker() as session:
            with pytest.raises(DBAPIError):
                async with session.begin():
                    await session.execute(
                        text(
                            """INSERT INTO prompt_templates
                            (id, family, version, purpose, system_text, content_sha256, created_by)
                            VALUES ('bad-hash-v1', 'bad-hash', 1, 'grounded', 'invalid', :hash, 'test')"""
                        ),
                        {"hash": "0" * 64},
                    )
    finally:
        await maker.kw["bind"].dispose()


@run
async def test_unavailable_prompt_database_is_sanitized():
    broken_url = make_url(_url().get_secret_value()).set(port=1)
    maker = create_sessionmaker(SecretStr(str(broken_url)))
    try:
        with pytest.raises(PromptError) as caught:
            await SqlPromptTemplateStore(maker).active_id("grounded")
        assert caught.value.code == "store_unavailable"
        assert caught.value.__cause__ is None
    finally:
        await maker.kw["bind"].dispose()


def test_prompt_migration_downgrade_and_upgrade_restores_seeds():
    run_migrations(_url(), "downgrade", "0003_minimal_index")
    run_migrations(_url(), "upgrade", "head")

    async def check():
        maker = create_sessionmaker(_url())
        try:
            async with maker() as session:
                ids = (await session.execute(text("SELECT id FROM prompt_templates"))).scalars().all()
            assert {"grounded-answer-v1", "grounded-answer-v2", "ungrounded-v1"} <= set(ids)
        finally:
            await maker.kw["bind"].dispose()

    run(check)()
