"""PostgreSQL tests for prompt templates and activation history."""

import asyncio
import functools
import hashlib
import importlib.util
import os
import sys
from pathlib import Path

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
    from rag_foundry_domain.uow_adapters import per_call_prompts
    from rag_foundry_storage.sql.database import create_database
    from rag_foundry_storage.sql.migrate import run_migrations


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
    db = create_database(_url())
    try:
        await check(per_call_prompts(db.uow))
    finally:
        await db.dispose()


@run
async def test_prompt_migration_seeds_and_active_templates():
    db = create_database(_url())
    maker = db.sessionmaker
    try:
        async with maker() as session:
            templates = (
                await session.execute(
                    text("SELECT id, system_text FROM prompt_templates WHERE created_by = 'migration:0004' ORDER BY id")
                )
            ).all()
            activations = (
                await session.execute(
                    text(
                        "SELECT purpose, template_id FROM prompt_activations "
                        "WHERE activated_by = 'migration:0004' ORDER BY seq"
                    )
                )
            ).all()
            chunk_templates = (
                await session.execute(
                    text("SELECT id, system_text FROM prompt_templates WHERE created_by = 'migration:0006' ORDER BY id")
                )
            ).all()
        assert {row.id for row in templates} == {"grounded-answer-v1", "grounded-answer-v2", "ungrounded-v1"}
        migration = (
            Path(__file__).parents[2]
            / "packages/storage/src/rag_foundry_storage/sql/migrations/versions/0004_prompt_templates.py"
        )
        spec = importlib.util.spec_from_file_location("migration_0004", migration)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        seeded_texts = {
            "grounded-answer-v1": module._GROUNDED_V1,
            "grounded-answer-v2": module._GROUNDED_V2,
            "ungrounded-v1": module._UNGROUNDED_V1,
        }
        for row in templates:
            assert row.system_text == seeded_texts[row.id]
        assert len(activations) == 3
        assert activations == [
            ("grounded", "grounded-answer-v1"),
            ("grounded", "grounded-answer-v2"),
            ("ungrounded", "ungrounded-v1"),
        ]
        assert await per_call_prompts(db.uow).active_id("grounded") == "grounded-answer-v2"
        assert await per_call_prompts(db.uow).active_id("ungrounded") == "ungrounded-v1"
        assert {row.id for row in chunk_templates} == {"chunk-boundaries-v1", "propositions-v1"}
        migration_0006 = migration.with_name("0006_chunking_prompts.py")
        spec_0006 = importlib.util.spec_from_file_location("migration_0006", migration_0006)
        assert spec_0006 is not None and spec_0006.loader is not None
        module_0006 = importlib.util.module_from_spec(spec_0006)
        spec_0006.loader.exec_module(module_0006)
        assert {row.id: row.system_text for row in chunk_templates} == {
            "chunk-boundaries-v1": module_0006._CHUNK_BOUNDARIES_V1,
            "propositions-v1": module_0006._PROPOSITIONS_V1,
        }
        assert await per_call_prompts(db.uow).active_id("chunk-boundaries") == "chunk-boundaries-v1"
        assert await per_call_prompts(db.uow).active_id("propositions") == "propositions-v1"
    finally:
        await db.dispose()


@run
async def test_prompt_template_update_is_rejected():
    db = create_database(_url())
    maker = db.sessionmaker
    try:
        async with maker() as session:
            with pytest.raises(DBAPIError):
                async with session.begin():
                    await session.execute(
                        text("UPDATE prompt_templates SET family = family WHERE id = 'grounded-answer-v2'")
                    )
    finally:
        await db.dispose()


@run
async def test_prompt_template_delete_is_rejected():
    db = create_database(_url())
    maker = db.sessionmaker
    try:
        async with maker() as session:
            with pytest.raises(DBAPIError):
                async with session.begin():
                    await session.execute(text("DELETE FROM prompt_templates WHERE id = 'grounded-answer-v2'"))
    finally:
        await db.dispose()


@run
async def test_prompt_activation_update_is_rejected():
    db = create_database(_url())
    maker = db.sessionmaker
    try:
        async with maker() as session:
            with pytest.raises(DBAPIError):
                async with session.begin():
                    await session.execute(
                        text("UPDATE prompt_activations SET decision_ref = decision_ref WHERE seq = 1")
                    )
    finally:
        await db.dispose()


@run
async def test_prompt_activation_delete_is_rejected():
    db = create_database(_url())
    maker = db.sessionmaker
    try:
        async with maker() as session:
            with pytest.raises(DBAPIError):
                async with session.begin():
                    await session.execute(text("DELETE FROM prompt_activations WHERE seq = 1"))
    finally:
        await db.dispose()


@run
async def test_prompt_tables_reject_truncate():
    db = create_database(_url())
    maker = db.sessionmaker
    try:
        for table in ("prompt_activations", "prompt_templates"):
            async with maker() as session:
                with pytest.raises(DBAPIError, match="prompt rows are immutable"):
                    async with session.begin():
                        await session.execute(text(f"TRUNCATE {table} CASCADE"))
    finally:
        await db.dispose()


@run
async def test_database_rejects_incorrect_prompt_hash():
    db = create_database(_url())
    maker = db.sessionmaker
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
        await db.dispose()


@run
async def test_database_rejects_unknown_prompt_purpose():
    db = create_database(_url())
    maker = db.sessionmaker
    try:
        async with maker() as session:
            with pytest.raises(DBAPIError):
                async with session.begin():
                    await session.execute(
                        text(
                            """INSERT INTO prompt_templates
                            (id, family, version, purpose, system_text, content_sha256, created_by)
                            VALUES ('other-v1', 'other', 1, 'other', 'text', :hash, 'test')"""
                        ),
                        {"hash": hashlib.sha256(b"text").hexdigest()},
                    )
    finally:
        await db.dispose()


@run
async def test_unavailable_prompt_database_is_sanitized():
    broken_url = make_url(_url().get_secret_value()).set(port=1)
    db = create_database(SecretStr(str(broken_url)))
    try:
        with pytest.raises(PromptError) as caught:
            await per_call_prompts(db.uow).active_id("grounded")
        assert caught.value.code == "store_unavailable"
        assert caught.value.__cause__ is None
    finally:
        await db.dispose()


def test_prompt_migration_downgrade_and_upgrade_restores_seeds():
    run_migrations(_url(), "downgrade", "0003_minimal_index")
    run_migrations(_url(), "upgrade", "head")

    async def check():
        db = create_database(_url())
        maker = db.sessionmaker
        try:
            async with maker() as session:
                ids = (
                    (await session.execute(text("SELECT id FROM prompt_templates WHERE created_by = 'migration:0004'")))
                    .scalars()
                    .all()
                )
                chunking_ids = (
                    (await session.execute(text("SELECT id FROM prompt_templates WHERE created_by = 'migration:0006'")))
                    .scalars()
                    .all()
                )
            active_g = await per_call_prompts(db.uow).active_id("grounded")
            active_u = await per_call_prompts(db.uow).active_id("ungrounded")
            assert set(ids) == {"grounded-answer-v1", "grounded-answer-v2", "ungrounded-v1"}
            assert chunking_ids.count("chunk-boundaries-v1") == 1
            assert chunking_ids.count("propositions-v1") == 1
            assert len(chunking_ids) == 2
            assert active_g == "grounded-answer-v2" and active_u == "ungrounded-v1"
        finally:
            await db.dispose()

    run(check)()
