import asyncio
from functools import wraps

import pytest
from rag_foundry_domain.documents import Scope
from rag_foundry_domain.errors import NotFoundError
from rag_foundry_domain.index_testing import (
    GRAPH_STORE_CHECKS,
    INDEX_REPOSITORY_CHECKS,
    InMemoryIndexRepository,
    make_new_index_version,
)
from rag_foundry_domain.prompt_testing import PROMPT_STORE_CHECKS
from rag_foundry_domain.prompts import InMemoryPromptTemplateStore, PromptTemplate, prompt_sha256
from rag_foundry_domain.settings_store import InMemorySettingsOverrideRepository, NewSettingOverride
from rag_foundry_domain.settings_testing import SETTINGS_REPOSITORY_CHECKS
from rag_foundry_domain.testing import (
    EXTRACTION_REPOSITORY_CHECKS,
    REPOSITORY_CHECKS,
    InMemoryDocumentRepository,
    InMemoryExtractionRepository,
    make_new_extraction,
    make_new_version,
    make_scope,
)
from rag_foundry_domain.uow import WRITE_METHODS
from rag_foundry_domain.uow_adapters import (
    per_call_documents,
    per_call_extractions,
    per_call_indexes,
    per_call_prompts,
    per_call_settings,
)
from rag_foundry_domain.uow_testing import InMemoryUnitOfWorkFactory


def run_async(test):
    @wraps(test)
    def wrapped(*args, **kwargs):
        return asyncio.run(test(*args, **kwargs))

    return wrapped


@run_async
async def test_rollback_restores_document_repository():
    scope = make_scope()
    factory = InMemoryUnitOfWorkFactory(documents=InMemoryDocumentRepository([scope]))
    async with factory() as uow:
        created = await uow.documents.create_document_or_get(
            scope, make_new_version("rollback-doc"), display_name="test"
        )
    assert await factory.documents.project_exists(scope)
    with pytest.raises(NotFoundError):
        await factory.documents.get_document(scope, created.document.id)


@run_async
async def test_rollback_restores_extraction_repository():
    scope = make_scope()
    docs = InMemoryDocumentRepository([scope])
    seeded = await docs.create_document_or_get(scope, make_new_version("rollback-view-doc"), display_name="test")
    factory = InMemoryUnitOfWorkFactory(documents=docs)
    async with factory() as uow:
        await uow.extractions.create_view_or_get(scope, make_new_extraction(seeded.version, tag="rollback-view"))
    assert await factory.extractions.list_views(scope, seeded.version.id) == []


@run_async
async def test_rollback_restores_index_repository():
    scope = make_scope()
    docs = InMemoryDocumentRepository([scope])
    extractions = InMemoryExtractionRepository(docs)
    seeded = await docs.create_document_or_get(scope, make_new_version("rollback-index-doc"), display_name="test")
    view = await extractions.create_view_or_get(scope, make_new_extraction(seeded.version, tag="rollback-index"))
    factory = InMemoryUnitOfWorkFactory(documents=docs, extractions=extractions)
    async with factory() as uow:
        created = await uow.indexes.create_index_version(scope, make_new_index_version((view.view.id,)))
    with pytest.raises(NotFoundError):
        await factory.indexes.get_index_version(scope, created.id)


@run_async
async def test_rollback_restores_prompt_repository():
    factory = InMemoryUnitOfWorkFactory()
    text = "test prompt rollback INSUFFICIENT_EVIDENCE [S1]"
    template = PromptTemplate(
        id="rollback-v1",
        family="rollback",
        version=1,
        purpose="grounded",
        system_text=text,
        content_sha256=prompt_sha256(text),
    )
    async with factory() as uow:
        await uow.prompts.create(template)
        await uow.commit()
    async with factory() as uow:
        await uow.prompts.activate("grounded", template.id, activated_by="test", decision_ref="rollback")
    assert await factory.prompts.activations("grounded") == []


@run_async
async def test_rollback_restores_settings_repository():
    factory = InMemoryUnitOfWorkFactory()
    new = NewSettingOverride("global", None, "answer.temperature", 0.4, "test", "rollback")
    async with factory() as uow:
        await uow.settings.append(new)
    assert await factory.settings.current(None) == ()


@run_async
async def test_commit_keeps_repository_state():
    factory = InMemoryUnitOfWorkFactory()
    new = NewSettingOverride("global", None, "answer.temperature", 0.4, "test", "commit")
    async with factory() as uow:
        await uow.settings.append(new)
        await uow.commit()
    assert len(await factory.settings.current(None)) == 1
    assert factory.committed == 1


@run_async
async def test_exception_rolls_back_and_reraises():
    factory = InMemoryUnitOfWorkFactory()
    new = NewSettingOverride("global", None, "answer.temperature", 0.4, "test", "exception")
    with pytest.raises(ValueError):
        async with factory() as uow:
            await uow.settings.append(new)
            raise ValueError("sentinel")
    assert await factory.settings.current(None) == ()


@run_async
async def test_reader_in_other_task_works_while_writer_open():
    factory = InMemoryUnitOfWorkFactory()
    opened = asyncio.Event()
    finish = asyncio.Event()

    async def writer():
        async with factory():
            opened.set()
            await finish.wait()

    task = asyncio.create_task(writer())
    await opened.wait()
    async with factory(read_only=True) as uow:
        assert await uow.settings.current(None) == ()
    finish.set()
    await task


@run_async
async def test_read_only_commit_and_every_write_are_rejected():
    factory = InMemoryUnitOfWorkFactory()
    async with factory(read_only=True) as uow:
        with pytest.raises(RuntimeError, match="read-only unit of work"):
            await uow.commit()
        for family, methods in WRITE_METHODS.items():
            for method in methods:
                with pytest.raises(RuntimeError, match="read-only unit of work"):
                    await getattr(getattr(uow, family), method)()
    assert factory.settings._rows == []


@run_async
async def test_nested_write_and_other_task_access_raise():
    factory = InMemoryUnitOfWorkFactory()
    async with factory() as uow:
        with pytest.raises(RuntimeError):
            async with factory():
                pass

        async def use_from_other_task():
            return await uow.settings.current(None)

        with pytest.raises(RuntimeError):
            await asyncio.create_task(use_from_other_task())


@run_async
async def test_per_call_adapters_pass_shared_suites():
    first = make_scope()
    other_project = Scope(tenant_id=first.tenant_id, project_id=make_scope().project_id)
    other_tenant = make_scope()
    scopes = [first, other_project, other_tenant]
    for check in REPOSITORY_CHECKS:

        async def document_factory(requested, *, clock=None):
            local = InMemoryUnitOfWorkFactory(documents=InMemoryDocumentRepository(requested, clock))
            return per_call_documents(local)

        await check(document_factory)

    for check in EXTRACTION_REPOSITORY_CHECKS:

        async def extraction_factory(requested):
            docs = InMemoryDocumentRepository(requested)
            extractions = InMemoryExtractionRepository(docs)
            local = InMemoryUnitOfWorkFactory(documents=docs, extractions=extractions)
            return per_call_documents(local), per_call_extractions(local)

        await check(extraction_factory)

    for check in INDEX_REPOSITORY_CHECKS + GRAPH_STORE_CHECKS:  # graph methods must pass through per-call too

        async def index_factory(requested):
            docs = InMemoryDocumentRepository(requested)
            extractions = InMemoryExtractionRepository(docs)
            indexes = InMemoryIndexRepository(extractions)
            local = InMemoryUnitOfWorkFactory(documents=docs, extractions=extractions, indexes=indexes)
            return per_call_documents(local), per_call_extractions(local), per_call_indexes(local)

        await check(index_factory)

    for check in PROMPT_STORE_CHECKS:
        local = InMemoryUnitOfWorkFactory(prompts=InMemoryPromptTemplateStore())
        await check(per_call_prompts(local))

    local_settings = InMemoryUnitOfWorkFactory(settings=InMemorySettingsOverrideRepository())

    def settings_factory(requested):
        return per_call_settings(local_settings)

    for check in SETTINGS_REPOSITORY_CHECKS:
        await check(settings_factory, tuple(scopes))


@run_async
async def test_per_call_commits_write_exactly_once():
    scope = make_scope()
    factory = InMemoryUnitOfWorkFactory(documents=InMemoryDocumentRepository([scope]))
    adapter = per_call_documents(factory)
    await adapter.create_document_or_get(scope, make_new_version("per-call"), display_name="test")
    assert factory.committed == 1
    assert factory.opened == 1


@run_async
async def test_per_call_unknown_attribute_raises_without_opening():
    calls = 0
    factory = InMemoryUnitOfWorkFactory()

    def counting_factory(**kwargs):
        nonlocal calls
        calls += 1
        return factory(**kwargs)

    adapter = per_call_settings(counting_factory)
    with pytest.raises(AttributeError):
        _ = adapter.missing_method
    assert calls == 0
    assert factory.opened == 0
