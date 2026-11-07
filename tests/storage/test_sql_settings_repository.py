"""Database checks for SQL runtime setting overrides."""

import asyncio
import functools
import importlib.util
import os
import sys
import uuid
from pathlib import Path

import pytest

_URL = os.environ.get("RAGF_TEST_DATABASE_URL")


def run(coro_fn):
    @functools.wraps(coro_fn)
    def wrapper(*args, **kwargs):
        factory = asyncio.SelectorEventLoop if sys.platform == "win32" else None
        with asyncio.Runner(loop_factory=factory) as runner:
            return runner.run(coro_fn(*args, **kwargs))

    return wrapper


def test_migration_allowlist_matches_scoped_keys():
    path = (
        Path(__file__).parents[2]
        / "packages/storage/src/rag_foundry_storage/sql/migrations/versions/0005_runtime_setting_overrides.py"
    )
    spec = importlib.util.spec_from_file_location("settings_migration", path)
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)
    from rag_foundry_domain.runtime_settings import KEY_SCOPES

    assert set(migration.SETTING_KEYS) == {key for key, scopes in KEY_SCOPES.items() if scopes}


if _URL:
    from pydantic import SecretStr
    from rag_foundry_domain.documents import Scope
    from rag_foundry_domain.errors import SettingsError
    from rag_foundry_domain.settings_store import NewSettingOverride
    from rag_foundry_domain.settings_testing import SETTINGS_REPOSITORY_CHECKS
    from rag_foundry_domain.uow_adapters import per_call_settings
    from rag_foundry_storage.sql.database import create_database
    from rag_foundry_storage.sql.migrate import run_migrations
    from rag_foundry_storage.sql.uow import ensure_local_scope
    from sqlalchemy import text
    from sqlalchemy.exc import DBAPIError

    def _secret_url():
        return SecretStr(_URL)

    def _scope(seed: int) -> Scope:
        return Scope(tenant_id=str(uuid.UUID(int=seed, version=4)), project_id=str(uuid.UUID(int=seed + 1, version=4)))

    async def _factory(scopes):
        db = create_database(_secret_url())
        maker = db.sessionmaker
        async with maker() as session, session.begin():
            await session.execute(text("TRUNCATE runtime_setting_overrides, projects, tenants CASCADE"))
        # The third scope reuses the first project UUID under another tenant; it is read-only in the
        # checks and cannot exist as a row (a project belongs to one tenant), so only the first two are created.
        for scope in scopes[:2]:
            await ensure_local_scope(db.uow, scope)
        return per_call_settings(db.uow)

    @pytest.fixture(scope="module", autouse=True)
    def migrated():
        run_migrations(_secret_url(), "upgrade", "head")

    @pytest.mark.parametrize("check", SETTINGS_REPOSITORY_CHECKS, ids=lambda check: check.__name__)
    @pytest.mark.db
    @run
    async def test_settings_repository_checks(check):
        first = _scope(101)
        other_project = Scope(tenant_id=first.tenant_id, project_id=str(uuid.UUID(int=202, version=4)))
        other_tenant = Scope(tenant_id=str(uuid.UUID(int=301, version=4)), project_id=first.project_id)
        scopes = (first, other_project, other_tenant)
        repo = await _factory(scopes)
        await check(lambda _scopes: repo, scopes)

    @pytest.mark.db
    @run
    async def test_database_rejects_string_value():
        await _raw_rejected("answer.temperature", '"bad"')

    @pytest.mark.db
    @run
    async def test_database_rejects_object_string_value():
        await _raw_rejected("answer.min_scores", '{"ollama/bge-m3/1024":"bad"}')

    @pytest.mark.db
    @run
    async def test_database_rejects_object_key_outside_pattern():
        await _raw_rejected("answer.min_scores", '{"bad key":0.5}')

    @pytest.mark.db
    @run
    async def test_database_rejects_key_outside_allowlist():
        await _raw_rejected("run.max_concurrent", "2")

    @pytest.mark.db
    @run
    async def test_database_rejects_cross_tenant_project():
        one, two = _scope(401), _scope(501)
        await _factory((one, two))
        db = create_database(_secret_url())
        maker = db.sessionmaker
        with pytest.raises(DBAPIError):
            async with maker() as session, session.begin():
                await session.execute(
                    text("""INSERT INTO runtime_setting_overrides
                    (level, tenant_id, project_id, key, value_json, version, created_by, reason)
                    VALUES ('project', :tenant, :project, 'answer.temperature', '0.5', 1, 'test', 'test')"""),
                    {"tenant": one.tenant_id, "project": two.project_id},
                )

    async def _raw_rejected(key, value):
        scope = _scope(601)
        await _factory((scope,))
        db = create_database(_secret_url())
        maker = db.sessionmaker
        with pytest.raises(DBAPIError):
            async with maker() as session, session.begin():
                await session.execute(
                    text("""INSERT INTO runtime_setting_overrides
                    (level, key, value_json, version, created_by, reason)
                    VALUES ('global', :key, CAST(:value AS jsonb), 1, 'test', 'test')"""),
                    {"key": key, "value": value},
                )

    @pytest.mark.db
    @run
    async def test_database_rejects_update_and_delete():
        await _factory(())
        db = create_database(_secret_url())
        maker = db.sessionmaker
        repo = per_call_settings(db.uow)
        row = await repo.append(NewSettingOverride("global", None, "answer.temperature", 0.5, "test", "test"))
        for statement in (
            "UPDATE runtime_setting_overrides SET reason = 'changed' WHERE id = :id",
            "DELETE FROM runtime_setting_overrides WHERE id = :id",
        ):
            with pytest.raises(DBAPIError):
                async with maker() as session, session.begin():
                    await session.execute(text(statement), {"id": row.id})

    @run
    async def test_closed_port_maps_to_store_unavailable():
        db = create_database(SecretStr("postgresql+asyncpg://u:p@127.0.0.1:1/db"))
        with pytest.raises(SettingsError) as exc:
            await per_call_settings(db.uow).current(None)
        assert exc.value.code == "store_unavailable"

    @pytest.mark.db
    def test_migration_downgrade_upgrade_round_trip():
        run_migrations(_secret_url(), "downgrade", "0004_prompt_templates")
        run_migrations(_secret_url(), "upgrade", "head")
