"""SQL implementation of the append-only runtime settings repository."""

from __future__ import annotations

import json
import uuid
from typing import Any

from rag_foundry_domain.documents import Scope
from rag_foundry_domain.errors import SettingsError
from rag_foundry_domain.runtime_settings import ALLOWED_KEYS, KEY_SCOPES, Level
from rag_foundry_domain.settings_store import (
    NewSettingOverride,
    SettingOverride,
    validate_override,
)
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError, InterfaceError, OperationalError
from sqlalchemy.ext.asyncio import AsyncSession

from rag_foundry_storage.sql._tx import read, write


def _uuid(value: str) -> str:
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError, TypeError):
        raise SettingsError("invalid_override") from None
    if parsed.version != 4 or str(parsed) != value:
        raise SettingsError("invalid_override")
    return value


def _row(row: Any) -> SettingOverride:
    return SettingOverride(
        id=row["id"],
        level=row["level"],
        tenant_id=None if row["tenant_id"] is None else str(row["tenant_id"]),
        project_id=None if row["project_id"] is None else str(row["project_id"]),
        key=row["key"],
        value=row["value_json"],
        version=row["version"],
        created_by=row["created_by"],
        reason=row["reason"],
        created_at=row["created_at"],
    )


class SqlSettingsOverrideRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def append(self, new: NewSettingOverride) -> SettingOverride:
        validate_override(new)
        tenant_id = _uuid(new.scope.tenant_id) if new.scope is not None else None
        project_id = _uuid(new.scope.project_id) if new.level == "project" and new.scope is not None else None
        try:
            async with write(self._session) as session:
                version = (
                    await session.execute(
                        text("""SELECT COALESCE(MAX(version), 0) + 1
                    FROM runtime_setting_overrides WHERE level = :level
                    AND tenant_id IS NOT DISTINCT FROM CAST(:tenant_id AS uuid)
                    AND project_id IS NOT DISTINCT FROM CAST(:project_id AS uuid) AND key = :key"""),
                        {"level": new.level, "tenant_id": tenant_id, "project_id": project_id, "key": new.key},
                    )
                ).scalar_one()
                row = (
                    (
                        await session.execute(
                            text("""INSERT INTO runtime_setting_overrides
                    (level, tenant_id, project_id, key, value_json, version, created_by, reason)
                    VALUES (:level, CAST(:tenant_id AS uuid), CAST(:project_id AS uuid), :key,
                    CAST(:value_json AS jsonb), :version, :created_by, :reason) RETURNING *"""),
                            {
                                "level": new.level,
                                "tenant_id": tenant_id,
                                "project_id": project_id,
                                "key": new.key,
                                "value_json": None if new.value is None else json.dumps(new.value, allow_nan=False),
                                "version": version,
                                "created_by": new.created_by,
                                "reason": new.reason,
                            },
                        )
                    )
                    .mappings()
                    .one()
                )
                return _row(row)
        except IntegrityError as exc:
            if getattr(exc.orig, "sqlstate", None) == "23505" or getattr(exc.orig, "pgcode", None) == "23505":
                raise SettingsError("version_conflict") from None
            raise SettingsError("invalid_override") from None
        except (OperationalError, InterfaceError):
            raise SettingsError("store_unavailable") from None

    async def current(self, scope: Scope | None) -> tuple[SettingOverride, ...]:
        params = self._scope_params(scope)
        try:
            async with read(self._session) as session:
                rows = (
                    (
                        await session.execute(
                            text("""SELECT * FROM (
                    SELECT runtime_setting_overrides.*,
                        row_number() OVER (PARTITION BY level, key ORDER BY version DESC) AS row_num
                    FROM runtime_setting_overrides WHERE level = 'global'
                    OR (level = 'tenant' AND tenant_id = CAST(:tenant_id AS uuid))
                    OR (level = 'project' AND tenant_id = CAST(:tenant_id AS uuid)
                        AND project_id = CAST(:project_id AS uuid))
                    ) latest WHERE row_num = 1
                    ORDER BY CASE level WHEN 'global' THEN 0 WHEN 'tenant' THEN 1 ELSE 2 END, key"""),
                            params,
                        )
                    )
                    .mappings()
                    .all()
                )
            return tuple(_row(row) for row in rows)
        except (OperationalError, InterfaceError):
            raise SettingsError("store_unavailable") from None

    async def history(self, level: Level, scope: Scope | None, key: str) -> tuple[SettingOverride, ...]:
        if level not in ("global", "tenant", "project") or (level == "global") != (scope is None):
            raise SettingsError("invalid_override")
        if key not in ALLOWED_KEYS or level not in KEY_SCOPES[key]:
            raise SettingsError("invalid_override")
        params = self._scope_params(scope)
        if level == "tenant":
            params["project_id"] = None  # tenant rows are stored without a project
        try:
            async with read(self._session) as session:
                rows = (
                    (
                        await session.execute(
                            text("""SELECT * FROM runtime_setting_overrides
                    WHERE level = :level AND key = :key
                    AND tenant_id IS NOT DISTINCT FROM CAST(:tenant_id AS uuid)
                    AND project_id IS NOT DISTINCT FROM CAST(:project_id AS uuid) ORDER BY version"""),
                            {**params, "level": level, "key": key},
                        )
                    )
                    .mappings()
                    .all()
                )
            return tuple(_row(row) for row in rows)
        except (OperationalError, InterfaceError):
            raise SettingsError("store_unavailable") from None

    @staticmethod
    def _scope_params(scope: Scope | None) -> dict[str, str | None]:
        if scope is None:
            return {"tenant_id": None, "project_id": None}
        return {"tenant_id": _uuid(scope.tenant_id), "project_id": _uuid(scope.project_id)}
