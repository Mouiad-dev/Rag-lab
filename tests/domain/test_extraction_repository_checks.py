"""Extraction repository checks pass on the in-memory repository and fail on deliberately broken ones."""

import asyncio
from collections.abc import Sequence

import pytest
from pydantic import ConfigDict
from rag_foundry_domain import DocumentRepository, NotFoundError, Scope
from rag_foundry_domain.extraction import ExtractionOutcome, ExtractionView
from rag_foundry_domain.ports import ExtractionRepository
from rag_foundry_domain.testing import (
    EXTRACTION_REPOSITORY_CHECKS,
    ExtractionRepoFactory,
    InMemoryDocumentRepository,
    InMemoryExtractionRepository,
    make_new_extraction,
)


class _NoDedup(InMemoryExtractionRepository):
    def _match(self, scope, key):
        return None


class _IgnoresProfile(InMemoryExtractionRepository):
    def _match(self, scope, key):
        return next(
            (
                v
                for v in self._in_scope(scope)
                if (v.document_version_id, v.extractor, v.extractor_version)
                == (
                    key.document_version_id,
                    key.extractor,
                    key.extractor_version,
                )
            ),
            None,
        )


class _LeakyScope(InMemoryExtractionRepository):
    def _in_scope(self, scope):
        return list(self._views)

    def _require_version(self, scope, version_id):
        if not any(v.id == version_id for v in self._documents._versions):
            raise NotFoundError()


class _MalformedRaises(InMemoryExtractionRepository):
    async def get_view(self, scope, view_id):
        if view_id == "not-a-uuid":
            raise ValueError("invalid input syntax for type uuid")
        return await super().get_view(scope, view_id)


class _Racy(InMemoryExtractionRepository):
    async def create_view_or_get(self, scope, new):
        found = self._match(scope, new.key)
        await asyncio.sleep(0)
        if found is not None:
            return ExtractionOutcome(view=found, created=False)
        view = ExtractionView(
            **new.model_dump(),
            id=_new_id(),
            tenant_id=scope.tenant_id,
            project_id=scope.project_id,
            created_at=self._clock(),
        )
        self._views.append(view)
        return ExtractionOutcome(view=view, created=True)


def _new_id() -> str:
    from rag_foundry_domain import new_id

    return new_id()


class _ReverseOrder(InMemoryExtractionRepository):
    @staticmethod
    def _order(views):
        return sorted(views, key=lambda v: (v.created_at, v.id), reverse=True)


class _TrustsDocumentId(InMemoryExtractionRepository):
    """Checks only that the version exists in scope, not that the document owns it."""

    async def create_view_or_get(self, scope, new):
        async with self._lock:
            self._require_version(scope, new.document_version_id)
            found = self._match(scope, new.key)
            if found is not None:
                return ExtractionOutcome(view=found, created=False)
            view = ExtractionView(
                **new.model_dump(),
                id=_new_id(),
                tenant_id=scope.tenant_id,
                project_id=scope.project_id,
                created_at=self._clock(),
            )
            self._views.append(view)
            return ExtractionOutcome(view=view, created=True)


class _MutableView(ExtractionView):
    model_config = ConfigDict(frozen=False, extra="forbid")


class _MutableRecords(InMemoryExtractionRepository):
    async def create_view_or_get(self, scope, new):
        out = await super().create_view_or_get(scope, new)
        return ExtractionOutcome.model_construct(view=_MutableView(**out.view.model_dump()), created=out.created)


class _OverwritesOnDedup(InMemoryExtractionRepository):
    """Replaces the stored view instead of returning the existing one (a changed key is not needed)."""

    async def create_view_or_get(self, scope, new):
        async with self._lock:
            self._require_version(scope, new.document_version_id)
            view = ExtractionView(
                **new.model_dump(),
                id=_new_id(),
                tenant_id=scope.tenant_id,
                project_id=scope.project_id,
                created_at=self._clock(),
            )
            self._views = [v for v in self._views if v.key != new.key]
            self._views.append(view)
            return ExtractionOutcome(view=view, created=True)


BROKEN = {
    "check_extraction_same_key_dedup": _NoDedup,
    "check_extraction_changed_key_new_view": _IgnoresProfile,
    "check_extraction_cross_scope_not_found": _LeakyScope,
    "check_extraction_unknown_and_malformed_ids": _MalformedRaises,
    "check_extraction_concurrent_creates": _Racy,
    "check_extraction_list_order": _ReverseOrder,
    "check_extraction_records_frozen": _MutableRecords,
}


def _factory(cls: type[InMemoryExtractionRepository]) -> ExtractionRepoFactory:
    async def make(scopes: Sequence[Scope]) -> tuple[DocumentRepository, ExtractionRepository]:
        docs = InMemoryDocumentRepository(scopes)
        return docs, cls(docs)

    return make


def test_every_check_has_a_broken_repository() -> None:
    assert sorted(c.__name__ for c in EXTRACTION_REPOSITORY_CHECKS) == sorted(BROKEN)


def test_in_memory_satisfies_protocol() -> None:
    assert isinstance(InMemoryExtractionRepository(InMemoryDocumentRepository([])), ExtractionRepository)


@pytest.mark.parametrize("check", EXTRACTION_REPOSITORY_CHECKS, ids=lambda c: c.__name__)
def test_checks_pass_on_in_memory(check) -> None:
    asyncio.run(check(_factory(InMemoryExtractionRepository)))


@pytest.mark.parametrize("check", EXTRACTION_REPOSITORY_CHECKS, ids=lambda c: c.__name__)
def test_each_check_fails_on_broken_repository(check) -> None:
    with pytest.raises(AssertionError):
        asyncio.run(check(_factory(BROKEN[check.__name__])))


def test_overwriting_repository_fails_dedup_check() -> None:
    check = EXTRACTION_REPOSITORY_CHECKS[0]
    with pytest.raises(AssertionError):
        asyncio.run(check(_factory(_OverwritesOnDedup)))


def test_wrong_owner_document_check_fails_on_trusting_repository() -> None:
    check = next(c for c in EXTRACTION_REPOSITORY_CHECKS if c.__name__.endswith("unknown_and_malformed_ids"))
    with pytest.raises(AssertionError):
        asyncio.run(check(_factory(_TrustsDocumentId)))


def test_make_new_extraction_is_valid_and_keyed() -> None:
    async def run() -> None:
        scope = Scope(tenant_id=_new_id(), project_id=_new_id())
        docs, _ = await _factory(InMemoryExtractionRepository)([scope])
        from rag_foundry_domain.testing import make_new_version

        ver = (await docs.create_document_or_get(scope, make_new_version("x"), display_name="a")).version
        a, b = make_new_extraction(ver), make_new_extraction(ver, profile_id="p2")
        assert a.key != b.key and a.key.document_version_id == ver.id

    asyncio.run(run())
