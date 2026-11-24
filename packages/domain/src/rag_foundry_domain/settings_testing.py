"""Reusable conformance checks for settings override repositories."""

from collections.abc import Awaitable
from datetime import datetime
from typing import Protocol

from rag_foundry_domain.errors import SettingsError
from rag_foundry_domain.settings_store import NewSettingOverride, SettingsOverrideRepository


class SettingsRepositoryFactory(Protocol):
    def __call__(self, scopes: tuple[object, ...]) -> SettingsOverrideRepository: ...


async def _rejected(awaitable: Awaitable[object]) -> None:
    try:
        await awaitable
    except SettingsError:
        return
    raise AssertionError("invalid setting override was accepted")


async def check_settings_repository(factory: SettingsRepositoryFactory, scopes: tuple[object, ...]) -> None:
    if len(scopes) < 3:
        raise AssertionError("factory must provide at least three scopes")
    first, other_project, other_tenant = scopes[:3]
    repo = factory(scopes)

    def new(level: str, scope: object | None, key: str, value: object | None, reason: str = "check"):
        return NewSettingOverride(level, scope, key, value, "settings-check", reason)

    global_one = await repo.append(new("global", None, "answer.max_output_tokens", 600))
    global_two = await repo.append(new("global", None, "answer.max_output_tokens", 700))
    tenant_one = await repo.append(new("tenant", first, "answer.temperature", 0.2))
    tenant_two = await repo.append(new("tenant", first, "answer.temperature", 0.3))
    project_one = await repo.append(new("project", first, "answer.max_evidence_chars", 13000))
    project_two = await repo.append(new("project", first, "answer.max_evidence_chars", 14000))
    if (
        global_one.version,
        global_two.version,
        tenant_one.version,
        tenant_two.version,
        project_one.version,
        project_two.version,
    ) != (1, 2, 1, 2, 1, 2):
        raise AssertionError("versions must increase per scoped key")
    reset = await repo.append(new("global", None, "answer.max_output_tokens", None))
    if reset.version != 3 or reset.value is not None:
        raise AssertionError("null override must be stored as a reset row")
    current = await repo.current(first)
    if tuple((row.level, row.key, row.version) for row in current) != (
        ("global", "answer.max_output_tokens", 3),
        ("tenant", "answer.temperature", 2),
        ("project", "answer.max_evidence_chars", 2),
    ):
        raise AssertionError("current must return the latest row per level and key")
    other_project_rows = await repo.current(other_project)
    if tuple((row.level, row.key, row.version) for row in other_project_rows) != (
        ("global", "answer.max_output_tokens", 3),
        ("tenant", "answer.temperature", 2),
    ):
        raise AssertionError("other projects must see only global and tenant rows")
    other_tenant_rows = await repo.current(other_tenant)
    if tuple((row.level, row.key, row.version) for row in other_tenant_rows) != (
        ("global", "answer.max_output_tokens", 3),
    ):
        raise AssertionError("other tenants must see only global rows")
    for level, scope, key in (("tenant", first, "answer.temperature"), ("project", first, "answer.max_evidence_chars")):
        hist = await repo.history(level, scope, key)
        if tuple(row.version for row in hist) != (1, 2):
            raise AssertionError("history must be ordered and scope filtered")
        outside_scopes = (other_tenant,) if level == "tenant" else (other_project, other_tenant)
        for outside in outside_scopes:
            if await repo.history(level, outside, key):
                raise AssertionError("out-of-scope row leaked through history")
    before_first = await repo.current(first)
    before_global = await repo.current(None)
    for invalid in (
        new("global", None, "answer.nope", 1),
        new("global", None, "run.max_concurrent", 2),
        new("project", first, "run.max_attempts", 2),
        new("global", None, "answer.temperature", 3),
        new("global", None, "answer.temperature", "private-value"),
        new("global", None, "answer.min_scores", {"ollama/bge-m3/1024": "private-value"}),
    ):
        await _rejected(repo.append(invalid))
    if await repo.current(first) != before_first or await repo.current(None) != before_global:
        raise AssertionError("invalid rows must not be stored")
    if any(not isinstance(row.created_at, datetime) or row.created_at.tzinfo is None for row in current):
        raise AssertionError("created_at must be timezone-aware")


SETTINGS_REPOSITORY_CHECKS = (check_settings_repository,)
