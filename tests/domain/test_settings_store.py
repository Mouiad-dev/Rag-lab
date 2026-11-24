from __future__ import annotations

import asyncio

import pytest
from rag_foundry_domain.errors import SettingsError
from rag_foundry_domain.runtime_settings import RuntimeSettings
from rag_foundry_domain.settings_store import (
    InMemorySettingsOverrideRepository,
    NewSettingOverride,
    RuntimeSettingsService,
)
from rag_foundry_domain.settings_testing import SETTINGS_REPOSITORY_CHECKS
from rag_foundry_domain.testing import make_scope


def _scopes():
    first = make_scope()
    same_project_other_tenant = first.model_copy(update={"tenant_id": "t-other"})
    other_project = first.model_copy(update={"project_id": "p-other"})
    return first, other_project, same_project_other_tenant


def _new(level, scope, key, value, reason="reason"):
    return NewSettingOverride(level, scope, key, value, "test-user", reason)


async def _check_in_memory_repository_shared_checks():
    scopes = _scopes()
    for check in SETTINGS_REPOSITORY_CHECKS:
        await check(lambda _scopes: InMemorySettingsOverrideRepository(), scopes)


def test_in_memory_repository_shared_checks():
    asyncio.run(_check_in_memory_repository_shared_checks())


async def _check_service_layers_global_tenant_and_project_and_isolates_projects():
    first, other_project, _ = _scopes()
    repo = InMemorySettingsOverrideRepository()
    service = RuntimeSettingsService(RuntimeSettings(), repo=repo)
    await service.set_override(_new("global", None, "answer.max_output_tokens", 600))
    await service.set_override(_new("tenant", first, "answer.max_output_tokens", 700))
    await service.set_override(_new("project", first, "answer.max_output_tokens", 800))
    assert (await service.run_snapshot(first)).answer.max_output_tokens == 800
    assert (await service.run_snapshot(other_project)).answer.max_output_tokens == 700


def test_service_layers_global_tenant_and_project_and_isolates_projects():
    asyncio.run(_check_service_layers_global_tenant_and_project_and_isolates_projects())


async def _check_bad_indexing_pair_breaks_indexing_without_breaking_run_snapshot():
    scope = make_scope()
    repo = InMemorySettingsOverrideRepository()
    service = RuntimeSettingsService(RuntimeSettings(), repo=repo)
    await service.set_override(_new("global", None, "indexing.c05_chunk_size", 100))
    await service.set_override(_new("global", None, "indexing.c05_chunk_overlap", 100))
    assert (await service.run_snapshot(scope)).run.max_attempts == 2
    with pytest.raises(SettingsError) as exc:
        await service.indexing(scope)
    assert exc.value.code == "invalid_value"


def test_bad_indexing_pair_breaks_indexing_without_breaking_run_snapshot():
    asyncio.run(_check_bad_indexing_pair_breaks_indexing_without_breaking_run_snapshot())


async def _check_error_messages_do_not_expose_values_or_reasons():
    repo = InMemorySettingsOverrideRepository()
    secret_value = "super-secret-value"
    secret_reason = "private audit explanation"
    with pytest.raises(SettingsError) as exc:
        await repo.append(_new("global", None, "answer.temperature", secret_value, secret_reason))
    assert secret_value not in str(exc.value)
    assert secret_reason not in str(exc.value)


def test_error_messages_do_not_expose_values_or_reasons():
    asyncio.run(_check_error_messages_do_not_expose_values_or_reasons())


async def _check_mapping_value_is_copied_on_create():
    repo = InMemorySettingsOverrideRepository()
    value = {"ollama/bge-m3/1024": 0.4}
    await repo.append(_new("global", None, "answer.min_scores", value))
    value["ollama/bge-m3/1024"] = 0.9
    history = await repo.history("global", None, "answer.min_scores")
    assert history[0].value == {"ollama/bge-m3/1024": 0.4}


def test_mapping_value_is_copied_on_create():
    asyncio.run(_check_mapping_value_is_copied_on_create())


@pytest.mark.parametrize(
    "level,scope,created_by,reason",
    [
        ("tenant", None, "user", "why"),
        ("global", make_scope(), "user", "why"),
        ("global", None, "bad user", "why"),
        ("global", None, "user", ""),
    ],
)
def test_new_override_rejects_invalid_shape(level, scope, created_by, reason):
    with pytest.raises(SettingsError) as exc:
        NewSettingOverride(level, scope, "answer.temperature", 0.5, created_by, reason)
    assert exc.value.code == "invalid_override"
