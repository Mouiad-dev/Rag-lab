"""SQL Unit of Work tests requiring PostgreSQL."""

import asyncio
import functools
import os

import pytest
from pydantic import SecretStr

_URL = os.environ.get("RAGF_TEST_DATABASE_URL")
pytestmark = [pytest.mark.db, pytest.mark.skipif(not _URL, reason="RAGF_TEST_DATABASE_URL is not set")]

if _URL:
    from rag_foundry_domain.errors import IndexingError, NotFoundError, PromptError, SettingsError, StoreError
    from rag_foundry_domain.index_testing import make_chunk, make_new_index_version
    from rag_foundry_domain.prompts import PromptTemplate, prompt_sha256
    from rag_foundry_domain.settings_store import NewSettingOverride
    from rag_foundry_domain.testing import make_new_extraction, make_new_version, make_scope
    from rag_foundry_domain.uow import WRITE_METHODS
    from rag_foundry_storage.sql.database import create_database, run_db_main
    from rag_foundry_storage.sql.migrate import run_migrations
    from rag_foundry_storage.sql.uow import ensure_local_scope
    from sqlalchemy import text
    from sqlalchemy.exc import DBAPIError


def run(coro_fn):
    """Run each async test on a selector loop compatible with psycopg on Windows."""
    if _URL:

        @functools.wraps(coro_fn)
        def wrapper(*args, **kwargs):
            return run_db_main(coro_fn(*args, **kwargs))

        return wrapper
    return coro_fn


def _url() -> SecretStr:
    assert _URL
    return SecretStr(_URL)


@pytest.fixture(scope="module", autouse=True)
def migrated():
    if _URL:
        run_migrations(_url(), "upgrade", "head")  # synchronous; never on a running loop


async def _document_and_extraction(uow, scope, tag="atomic"):
    outcome = await uow.documents.create_document_or_get(scope, make_new_version(tag), display_name="test.txt")
    extraction = await uow.extractions.create_view_or_get(scope, make_new_extraction(outcome.version, tag=tag))
    return outcome, extraction


@run
async def test_write_uow_rolls_back_document_and_extraction():
    db = create_database(_url())
    scope = make_scope()
    try:
        await ensure_local_scope(db.uow, scope)
        with pytest.raises(ValueError, match="force rollback"):
            async with db.uow() as uow:
                outcome, extraction = await _document_and_extraction(uow, scope, "rollback")
                ids = outcome.document.id, extraction.view.id
                raise ValueError("force rollback")
        async with db.uow(read_only=True) as uow:
            with pytest.raises(NotFoundError):
                await uow.documents.get_document(scope, ids[0])
            with pytest.raises(NotFoundError):
                await uow.extractions.get_view(scope, ids[1])
    finally:
        await db.dispose()


@run
async def test_write_uow_commits_document_and_extraction_together():
    db = create_database(_url())
    scope = make_scope()
    try:
        await ensure_local_scope(db.uow, scope)
        async with db.uow() as uow:
            outcome, extraction = await _document_and_extraction(uow, scope, "commit")
            await uow.commit()
        async with db.uow(read_only=True) as uow:
            assert (await uow.documents.get_document(scope, outcome.document.id)).id == outcome.document.id
            assert (await uow.extractions.get_view(scope, extraction.view.id)).id == extraction.view.id
    finally:
        await db.dispose()


@run
async def test_index_duplicate_ordinal_rolls_back_method_and_uow_remains_usable():
    db = create_database(_url())
    scope = make_scope()
    try:
        await ensure_local_scope(db.uow, scope)
        async with db.uow() as uow:
            document = await uow.documents.create_document_or_get(
                scope, make_new_version("index"), display_name="index.txt"
            )
            extraction = await uow.extractions.create_view_or_get(
                scope, make_new_extraction(document.version, tag="index")
            )
            view_id = extraction.view.id
            index = await uow.indexes.create_index_version(scope, make_new_index_version((view_id,)))
            valid = make_chunk(view_id, 0, (1, 0, 0, 0), "valid")
            duplicate = make_chunk(view_id, 0, (0, 1, 0, 0), "duplicate")
            with pytest.raises(IndexingError) as caught:
                await uow.indexes.add_chunks(scope, index.id, (valid, duplicate))
            assert caught.value.code == "duplicate_ordinal"
            assert await uow.indexes.list_chunks(scope, index.id) == []
            await uow.indexes.add_chunks(scope, index.id, (valid,))
            ready = await uow.indexes.mark_ready(scope, index.id)
            assert ready.state == "ready"
            await uow.commit()
    finally:
        await db.dispose()


@run
async def test_read_only_uow_rejects_every_repository_write():
    db = create_database(_url())
    scope = make_scope()
    try:
        await ensure_local_scope(db.uow, scope)
        async with db.uow() as setup:
            outcome, extraction = await _document_and_extraction(setup, scope, "readonly")
            await setup.commit()
        template = PromptTemplate(
            id="readonly-v1",
            family="readonly",
            version=1,
            purpose="grounded",
            system_text="INSUFFICIENT_EVIDENCE [S1]",
            content_sha256=prompt_sha256("INSUFFICIENT_EVIDENCE [S1]"),
        )
        setting = NewSettingOverride("project", scope, "answer.temperature", 0.2, "test", "test")
        async with db.uow(read_only=True) as uow:
            methods = {
                "documents": {
                    "create_document_or_get": uow.documents.create_document_or_get(
                        scope, make_new_version("ro-new"), display_name="new.txt"
                    ),
                    "add_version_or_get": uow.documents.add_version_or_get(
                        scope, outcome.document.id, make_new_version("ro-version")
                    ),
                },
                "extractions": {
                    "create_view_or_get": uow.extractions.create_view_or_get(
                        scope, make_new_extraction(outcome.version, tag="ro-extraction")
                    )
                },
                "indexes": {
                    "create_index_version": uow.indexes.create_index_version(
                        scope, make_new_index_version((extraction.view.id,))
                    ),
                    "add_chunks": uow.indexes.add_chunks(
                        scope,
                        "00000000-0000-4000-8000-000000000001",
                        (make_chunk(extraction.view.id, 0, (1, 0, 0, 0), "readonly"),),
                    ),
                    "mark_ready": uow.indexes.mark_ready(scope, "00000000-0000-4000-8000-000000000001"),
                    "mark_failed": uow.indexes.mark_failed(scope, "00000000-0000-4000-8000-000000000001"),
                },
                "prompts": {
                    "create": uow.prompts.create(template),
                    "activate": uow.prompts.activate("grounded", template.id, activated_by="test", decision_ref="test"),
                },
                "settings": {"append": uow.settings.append(setting)},
            }
            assert {repo: frozenset(calls) for repo, calls in methods.items()} == WRITE_METHODS
            for calls in methods.values():
                for call in calls.values():
                    with pytest.raises(RuntimeError, match="^read-only unit of work$"):
                        await call
        async with db.uow(read_only=True) as uow:
            assert await uow.documents.list_versions(scope, outcome.document.id) == [outcome.version]
            assert await uow.extractions.list_views(scope, outcome.version.id) == [extraction.view]
    finally:
        await db.dispose()


@run
async def test_uow_repository_access_is_task_owned():
    db = create_database(_url())
    try:
        async with db.uow(read_only=True) as uow:

            async def access():
                return uow.prompts

            with pytest.raises(RuntimeError, match="another task"):
                await asyncio.create_task(access())
    finally:
        await db.dispose()


@run
async def test_closed_port_connection_fails_inside_uow_read():
    from rag_foundry_storage.sql.database import create_database as create_db

    db = create_db(SecretStr("postgresql+psycopg://u:p@127.0.0.1:1/db?connect_timeout=1"))
    try:
        async with db.uow(read_only=True) as uow:
            with pytest.raises(PromptError) as caught:
                await uow.prompts.active_id("grounded")
            assert caught.value.code == "store_unavailable"
    finally:
        await db.dispose()


async def _kill_backend(db, uow):
    pid = await uow.session.scalar(text("SELECT pg_backend_pid()"))
    async with db.engine.connect() as connection:
        await connection.execute(text("SELECT pg_terminate_backend(:pid)"), {"pid": pid})
        await connection.commit()


@run
async def test_killed_backend_settings_write_maps_store_unavailable():
    db = create_database(_url())
    try:
        async with db.uow() as uow:
            await _kill_backend(db, uow)
            with pytest.raises(SettingsError) as caught:
                await uow.settings.append(NewSettingOverride("global", None, "answer.temperature", 0.2, "test", "test"))
            assert caught.value.code == "store_unavailable"
    finally:
        await db.dispose()


@run
async def test_killed_backend_document_write_propagates_dbapi_error():
    db = create_database(_url())
    scope = make_scope()
    try:
        await ensure_local_scope(db.uow, scope)
        async with db.uow() as uow:
            await _kill_backend(db, uow)
            with pytest.raises(DBAPIError) as caught:
                await uow.documents.create_document_or_get(
                    scope, make_new_version("killed-document"), display_name="killed.txt"
                )
            assert not isinstance(caught.value, RuntimeError)
    finally:
        await db.dispose()


@run
async def test_killed_backend_commit_maps_store_unavailable():
    db = create_database(_url())
    scope = make_scope()
    try:
        await ensure_local_scope(db.uow, scope)
        async with db.uow() as uow:
            await uow.documents.create_document_or_get(
                scope, make_new_version("killed-commit"), display_name="commit.txt"
            )
            await _kill_backend(db, uow)
            with pytest.raises(StoreError) as caught:
                await uow.commit()
            assert caught.value.code == "store_unavailable"
    finally:
        await db.dispose()
