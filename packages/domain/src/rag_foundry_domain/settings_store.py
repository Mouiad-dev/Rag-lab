"""Tenant-scoped append-only storage for runtime setting overrides."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol, runtime_checkable

from pydantic import ValidationError

from rag_foundry_domain.documents import Scope
from rag_foundry_domain.errors import SettingsError
from rag_foundry_domain.runtime_settings import (
    ALLOWED_KEYS,
    KEY_SCOPES,
    AnswerSettings,
    IndexingSettings,
    LayerOverride,
    Level,
    RunSettings,
    RunSnapshot,
    RuntimeSettings,
    resolve_indexing,
    resolve_run,
    resolve_run_settings,
)

_CREATED_BY = re.compile(r"^[A-Za-z0-9._:@-]{1,128}$")


@dataclass(frozen=True)
class NewSettingOverride:
    level: Level
    scope: Scope | None
    key: str
    value: int | float | bool | dict[str, float] | None
    created_by: str
    reason: str

    def __post_init__(self) -> None:
        if self.level not in ("global", "tenant", "project"):
            raise SettingsError("invalid_override")
        if self.scope is not None and not isinstance(self.scope, Scope):
            raise SettingsError("invalid_override")
        if (self.level == "global") != (self.scope is None):
            raise SettingsError("invalid_override")
        if (
            not isinstance(self.key, str)
            or not isinstance(self.created_by, str)
            or not _CREATED_BY.fullmatch(self.created_by)
        ):
            raise SettingsError("invalid_override")
        if not isinstance(self.reason, str) or not 1 <= len(self.reason) <= 500:
            raise SettingsError("invalid_override")


@dataclass(frozen=True)
class SettingOverride:
    id: int
    level: Level
    tenant_id: str | None
    project_id: str | None
    key: str
    value: int | float | bool | dict[str, float] | None
    version: int
    created_by: str
    reason: str
    created_at: datetime


def validate_override(new: NewSettingOverride) -> None:
    if not isinstance(new, NewSettingOverride):
        raise SettingsError("invalid_override")
    if new.key not in ALLOWED_KEYS:
        raise SettingsError("unknown_key")
    if new.level not in KEY_SCOPES[new.key]:
        raise SettingsError("scope_not_allowed")
    value = new.value
    if value is None:
        return
    if isinstance(value, str):
        raise SettingsError("invalid_value")
    if isinstance(value, Mapping):
        if any(type(item) not in (int, float) for item in value.values()):
            raise SettingsError("invalid_value")
        value = dict(value)
    elif type(value) not in (int, float, bool):
        raise SettingsError("invalid_value")
    section, _, name = new.key.partition(".")
    model = {"answer": AnswerSettings, "run": RunSettings, "indexing": IndexingSettings}[section]
    defaults = model().model_dump(mode="python")
    if section == "indexing" and new.key == "indexing.c05_chunk_size" and type(value) is int:
        defaults["c05_chunk_overlap"] = min(defaults["c05_chunk_overlap"], value - 1)
    elif section == "indexing" and new.key == "indexing.c05_chunk_overlap" and type(value) is int:
        defaults["c05_chunk_size"] = max(defaults["c05_chunk_size"], value + 1)
    try:
        model.model_validate({**defaults, name: value})
    except ValidationError:
        raise SettingsError("invalid_value") from None


@runtime_checkable
class SettingsOverrideRepository(Protocol):
    async def append(self, new: NewSettingOverride) -> SettingOverride: ...

    async def current(self, scope: Scope | None) -> tuple[SettingOverride, ...]: ...

    async def history(self, level: Level, scope: Scope | None, key: str) -> tuple[SettingOverride, ...]: ...


class InMemorySettingsOverrideRepository:
    def __init__(self) -> None:
        self._rows: list[SettingOverride] = []

    async def append(self, new: NewSettingOverride) -> SettingOverride:
        validate_override(new)
        tenant_id = new.scope.tenant_id if new.scope is not None else None
        project_id = new.scope.project_id if new.level == "project" and new.scope is not None else None
        same = [
            row
            for row in self._rows
            if (row.level, row.tenant_id, row.project_id, row.key) == (new.level, tenant_id, project_id, new.key)
        ]
        row = SettingOverride(
            id=len(self._rows) + 1,
            level=new.level,
            tenant_id=tenant_id,
            project_id=project_id,
            key=new.key,
            value=dict(new.value) if isinstance(new.value, Mapping) else new.value,
            version=max((item.version for item in same), default=0) + 1,
            created_by=new.created_by,
            reason=new.reason,
            created_at=datetime.now(UTC),
        )
        self._rows.append(row)
        return row

    async def current(self, scope: Scope | None) -> tuple[SettingOverride, ...]:
        selected = [row for row in self._rows if self._matches_scope(row, scope)]
        latest: dict[tuple[str, str], SettingOverride] = {}
        for row in selected:
            identity = (row.level, row.key)
            if identity not in latest or row.version > latest[identity].version:
                latest[identity] = row
        order = {"global": 0, "tenant": 1, "project": 2}
        return tuple(sorted(latest.values(), key=lambda row: (order[row.level], row.key)))

    async def history(self, level: Level, scope: Scope | None, key: str) -> tuple[SettingOverride, ...]:
        if level not in ("global", "tenant", "project") or (level == "global") != (scope is None):
            raise SettingsError("invalid_override")
        if key not in ALLOWED_KEYS or level not in KEY_SCOPES[key]:
            raise SettingsError("invalid_override")
        return tuple(
            sorted(
                (
                    row
                    for row in self._rows
                    if row.level == level and row.key == key and self._matches_scope(row, scope)
                ),
                key=lambda row: row.version,
            )
        )

    @staticmethod
    def _matches_scope(row: SettingOverride, scope: Scope | None) -> bool:
        if row.level == "global":
            return True
        if scope is None or row.tenant_id != scope.tenant_id:
            return False
        return row.level == "tenant" or row.project_id == scope.project_id


class RuntimeSettingsService:
    def __init__(
        self, base: RuntimeSettings, *, file_keys=frozenset(), repo: SettingsOverrideRepository | None = None
    ) -> None:
        self._base = base
        self._file_keys = frozenset(file_keys)
        self._repo = repo

    @property
    def base(self) -> RuntimeSettings:
        return self._base

    async def run_snapshot(self, scope: Scope | None) -> RunSnapshot:
        rows = await self._repo.current(scope) if self._repo is not None else ()
        return resolve_run(
            self._base,
            tuple(LayerOverride(row.level, row.key, row.value) for row in rows),
            file_keys=self._file_keys,
        )

    async def indexing(self, scope: Scope | None) -> IndexingSettings:
        rows = await self._repo.current(scope) if self._repo is not None else ()
        return resolve_indexing(
            self._base,
            tuple(LayerOverride(row.level, row.key, row.value) for row in rows),
            file_keys=self._file_keys,
        )

    async def run_settings(self, scope: Scope | None) -> RunSettings:
        rows = await self._repo.current(scope) if self._repo is not None else ()
        return resolve_run_settings(
            self._base,
            tuple(LayerOverride(row.level, row.key, row.value) for row in rows),
            file_keys=self._file_keys,
        )

    async def set_override(self, new: NewSettingOverride) -> SettingOverride:
        validate_override(new)
        if self._repo is None:
            raise SettingsError("store_unavailable")
        return await self._repo.append(new)
