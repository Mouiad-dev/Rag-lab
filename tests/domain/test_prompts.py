import asyncio

import pytest
from rag_foundry_domain.errors import PromptError
from rag_foundry_domain.prompt_testing import PROMPT_STORE_CHECKS, check_prompt_template_store
from rag_foundry_domain.prompts import (
    CachedPromptTemplateStore,
    InMemoryPromptTemplateStore,
    PromptTemplate,
    prompt_sha256,
    template_from_row,
)


def make_template(family="test-prompt", purpose="grounded", version=1):
    text = "system instructions INSUFFICIENT_EVIDENCE [S1]"
    return PromptTemplate(
        id=f"{family}-v{version}",
        family=family,
        version=version,
        purpose=purpose,
        system_text=text,
        content_sha256=prompt_sha256(text),
    )


@pytest.mark.parametrize("check", PROMPT_STORE_CHECKS)
def test_prompt_store_shared_checks(check):
    asyncio.run(check(InMemoryPromptTemplateStore()))


def test_prompt_store_check_restores_active_pointer_after_failure():
    class Broken(InMemoryPromptTemplateStore):
        async def activations(self, purpose):
            raise RuntimeError("injected")

    async def run():
        original = make_template("prior-prompt")
        store = Broken((original,))
        await store.activate("grounded", original.id, activated_by="test", decision_ref="test")
        with pytest.raises(RuntimeError, match="injected"):
            await check_prompt_template_store(store)
        assert await store.active_id("grounded") == original.id
        rows = await super(Broken, store).activations("grounded")
        assert rows[-1].decision_ref == "conformance check restore"

    asyncio.run(run())


def test_cached_store_avoids_second_inner_get():
    class Counting(InMemoryPromptTemplateStore):
        gets = 0

        async def get(self, template_id):
            self.gets += 1
            return await super().get(template_id)

    async def run():
        template = make_template()
        inner = Counting((template,))
        cache = CachedPromptTemplateStore(inner)
        assert await cache.get(template.id) == template
        assert await cache.get(template.id) == template
        assert inner.gets == 1

    asyncio.run(run())


def test_cached_store_always_reads_active_id():
    class Counting(InMemoryPromptTemplateStore):
        active_reads = 0

        async def active_id(self, purpose):
            self.active_reads += 1
            return await super().active_id(purpose)

    async def run():
        template = make_template()
        inner = Counting((template,))
        await inner.activate("grounded", template.id, activated_by="test", decision_ref="test")
        cache = CachedPromptTemplateStore(inner)
        await cache.active_id("grounded")
        await cache.active_id("grounded")
        assert inner.active_reads == 2

    asyncio.run(run())


def test_active_template_reads_pointer_and_template_on_cold_and_warm_cache():
    class Counting(InMemoryPromptTemplateStore):
        gets = 0
        active_reads = 0

        async def get(self, template_id):
            self.gets += 1
            return await super().get(template_id)

        async def active_id(self, purpose):
            self.active_reads += 1
            return await super().active_id(purpose)

    async def run():
        from rag_foundry_domain.prompts import active_template

        template = make_template()
        inner = Counting((template,))
        await inner.activate("grounded", template.id, activated_by="test", decision_ref="test")
        gets_after_activation = inner.gets
        cache = CachedPromptTemplateStore(inner)
        assert await active_template(cache, "grounded") == template
        assert (inner.active_reads, inner.gets) == (1, gets_after_activation + 1)
        assert await active_template(cache, "grounded") == template
        assert (inner.active_reads, inner.gets) == (2, gets_after_activation + 1)

    asyncio.run(run())


def test_bad_hash_from_model_construct_is_rejected_and_not_cached():
    class ForgedStore:
        calls = 0

        async def get(self, template_id):
            self.calls += 1
            return PromptTemplate.model_construct(
                id="forged-v1",
                family="forged",
                version=1,
                purpose="grounded",
                system_text="private prompt content",
                content_sha256="0" * 64,
            )

        async def active_id(self, purpose):
            raise AssertionError

        async def create(self, template):
            raise AssertionError

        async def activate(self, *args, **kwargs):
            raise AssertionError

        async def activations(self, purpose):
            raise AssertionError

    async def run():
        inner = ForgedStore()
        cache = CachedPromptTemplateStore(inner)
        for _ in range(2):
            with pytest.raises(PromptError) as caught:
                await cache.get("forged-v1")
            assert caught.value.code == "hash_mismatch"
            assert "private prompt content" not in str(caught.value)
        assert inner.calls == 2

    asyncio.run(run())


def test_prompt_errors_never_include_prompt_text():
    async def run():
        store = InMemoryPromptTemplateStore()
        secret = "do not reveal this prompt"
        template = PromptTemplate.model_construct(
            id="safe-v1",
            family="safe",
            version=1,
            purpose="grounded",
            system_text=secret,
            content_sha256="0" * 64,
        )
        with pytest.raises(PromptError) as caught:
            await store.create(template)
        assert caught.value.code == "hash_mismatch"
        assert secret not in str(caught.value)

    asyncio.run(run())


@pytest.mark.parametrize(
    ("text", "purpose", "code"),
    [
        ("secret <<< injection INSUFFICIENT_EVIDENCE [S1]", "grounded", "invalid_template"),
        ("secret >>> injection INSUFFICIENT_EVIDENCE [S1]", "grounded", "invalid_template"),
        ("secret without markers", "grounded", "invalid_template"),
    ],
)
def test_load_validation_rejects_invalid_templates_without_prompt_text(text, purpose, code):
    async def run():
        template = PromptTemplate.model_construct(
            id="safe-v1",
            family="safe",
            version=1,
            purpose=purpose,
            system_text=text,
            content_sha256=prompt_sha256(text),
        )
        with pytest.raises(PromptError) as caught:
            await InMemoryPromptTemplateStore().create(template)
        assert caught.value.code == code
        assert text not in str(caught.value)

    asyncio.run(run())


def test_template_from_row_hides_pydantic_input_value():
    secret = "private prompt INSUFFICIENT_EVIDENCE [S1]"
    with pytest.raises(PromptError) as caught:
        template_from_row(
            id="bad-v1",
            family="bad",
            version=1,
            purpose="grounded",
            system_text=secret,
            content_sha256="0" * 64,
        )
    assert caught.value.code == "hash_mismatch"
    assert secret not in str(caught.value)


def test_prompt_store_no_active_template():
    async def run():
        with pytest.raises(PromptError) as caught:
            await InMemoryPromptTemplateStore().active_id("grounded")
        assert caught.value.code == "no_active_template"

    asyncio.run(run())


def test_cached_store_rejects_and_does_not_cache_invalid_inner_template():
    class InvalidStore:
        calls = 0

        async def get(self, template_id):
            self.calls += 1
            text = "unsafe <<< marker INSUFFICIENT_EVIDENCE [S1]"
            return PromptTemplate.model_construct(
                id="unsafe-v1",
                family="unsafe",
                version=1,
                purpose="grounded",
                system_text=text,
                content_sha256=prompt_sha256(text),
            )

        async def active_id(self, purpose):
            raise AssertionError

        async def create(self, template):
            raise AssertionError

        async def activate(self, *args, **kwargs):
            raise AssertionError

        async def activations(self, purpose):
            raise AssertionError

    async def run():
        inner = InvalidStore()
        cache = CachedPromptTemplateStore(inner)
        for _ in range(2):
            with pytest.raises(PromptError) as caught:
                await cache.get("unsafe-v1")
            assert caught.value.code == "invalid_template"
        assert inner.calls == 2

    asyncio.run(run())


def test_template_from_row_rejects_id_family_mismatch():
    text = "valid prompt INSUFFICIENT_EVIDENCE [S1]"
    with pytest.raises(PromptError) as caught:
        template_from_row(
            id="different-v1",
            family="actual",
            version=1,
            purpose="grounded",
            system_text=text,
            content_sha256=prompt_sha256(text),
        )
    assert caught.value.code == "invalid_template"


def test_load_validation_rejects_delimiter_on_ungrounded_template():
    async def run():
        text = "unsafe <<< marker"
        template = PromptTemplate.model_construct(
            id="unsafe-v1",
            family="unsafe",
            version=1,
            purpose="ungrounded",
            system_text=text,
            content_sha256=prompt_sha256(text),
        )
        with pytest.raises(PromptError) as caught:
            await InMemoryPromptTemplateStore().create(template)
        assert caught.value.code == "invalid_template"

    asyncio.run(run())
