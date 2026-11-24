"""Repository checks pass on the in-memory repository and fail on deliberately broken ones."""

import asyncio
from collections.abc import Sequence

import pytest
from pydantic import ConfigDict
from rag_foundry_domain import (
    Document,
    DocumentRepository,
    DocumentVersion,
    IntakeOutcome,
    NewVersion,
    NotFoundError,
    Scope,
    new_id,
)
from rag_foundry_domain.testing import (
    REPOSITORY_CHECKS,
    CheckSkippedError,
    InMemoryDocumentRepository,
    RepoFactory,
    make_new_version,
    make_scope,
)


async def _good(scopes: Sequence[Scope], *, clock=None) -> DocumentRepository:
    return InMemoryDocumentRepository(scopes, clock)


class _NoDedup(InMemoryDocumentRepository):
    def _dedup_candidates(self, scope):
        return []


class _LatestWins(InMemoryDocumentRepository):
    @staticmethod
    def _choose(matches):
        return max(matches, key=lambda v: (v.created_at, v.id))


class _NoVersionDedup(InMemoryDocumentRepository):
    @staticmethod
    def _existing_in_document(versions, sha256):
        return None


class _NeverNewVersion(InMemoryDocumentRepository):
    @staticmethod
    def _existing_in_document(versions, sha256):
        return versions[-1]


class _SkipNumbers(InMemoryDocumentRepository):
    @staticmethod
    def _next_number(versions):
        return max(v.version_number for v in versions) + 2


class _LeakyScope(InMemoryDocumentRepository):
    def _require_document(self, scope, document_id):
        doc = self._documents.get(document_id)
        if doc is None:
            raise NotFoundError()
        return doc


class _CrossProjectDedup(InMemoryDocumentRepository):
    def _dedup_candidates(self, scope):
        return list(self._versions)


class _AlwaysExists(InMemoryDocumentRepository):
    async def project_exists(self, scope):
        return True


class _TieBreakReversed(InMemoryDocumentRepository):
    @staticmethod
    def _choose(matches):
        earliest = min(v.created_at for v in matches)
        return max((v for v in matches if v.created_at == earliest), key=lambda v: v.id)


class _RacyDedup(InMemoryDocumentRepository):
    """Checks for a duplicate, yields, then creates without checking again."""

    async def create_document_or_get(self, scope, new: NewVersion, *, display_name):
        matches = [v for v in self._dedup_candidates(scope) if v.sha256 == new.sha256]
        await asyncio.sleep(0)
        if matches:
            return await super().create_document_or_get(scope, new, display_name=display_name)
        now = self._clock()
        doc = Document(
            id=new_id(),
            tenant_id=scope.tenant_id,
            project_id=scope.project_id,
            display_name=display_name,
            created_at=now,
        )
        version = self._make_version(doc, new, 1, now)
        self._documents[doc.id] = doc
        self._versions.append(version)
        return IntakeOutcome(document=doc, version=version, created_document=True, created_version=True)


class _VersionAnyDocument(InMemoryDocumentRepository):
    async def get_version(self, scope, document_id, version_id):
        self._require_document(scope, document_id)
        for v in self._versions:
            if v.id == version_id:
                return v
        raise NotFoundError()


class _MalformedIdRaises(InMemoryDocumentRepository):
    def _require_document(self, scope, document_id):
        if document_id == "not-a-uuid":
            raise ValueError("invalid input syntax for type uuid")
        return super()._require_document(scope, document_id)

    async def get_version(self, scope, document_id, version_id):
        if version_id == "not-a-uuid":
            raise ValueError("invalid input syntax for type uuid")
        return await super().get_version(scope, document_id, version_id)


class _MutableDocument(Document):
    model_config = ConfigDict(frozen=False, extra="forbid")


class _MutableVersion(DocumentVersion):
    model_config = ConfigDict(frozen=False, extra="forbid")


class _MutableRecords(InMemoryDocumentRepository):
    async def create_document_or_get(self, scope, new: NewVersion, *, display_name):
        out = await super().create_document_or_get(scope, new, display_name=display_name)
        return IntakeOutcome.model_construct(
            document=_MutableDocument(**out.document.model_dump()),
            version=_MutableVersion(**out.version.model_dump()),
            created_document=out.created_document,
            created_version=out.created_version,
        )


BROKEN = {
    "check_same_bytes_dedup_within_project": _NoDedup,
    "check_earliest_version_wins": _LatestWins,
    "check_reupload_v1_bytes_returns_v1": _NoVersionDedup,
    "check_changed_bytes_new_version": _NeverNewVersion,
    "check_version_numbers_sequential": _SkipNumbers,
    "check_cross_scope_reads_not_found": _LeakyScope,
    "check_same_bytes_other_project_separate_document": _CrossProjectDedup,
    "check_unknown_project": _AlwaysExists,
    "check_records_frozen": _MutableRecords,
    "check_dedup_tie_break_lowest_version_id": _TieBreakReversed,
    "check_concurrent_same_bytes_dedup": _RacyDedup,
    "check_version_id_under_other_document_not_found": _VersionAnyDocument,
    "check_malformed_ids_not_found": _MalformedIdRaises,
}


def _factory(cls: type[InMemoryDocumentRepository]) -> RepoFactory:
    async def make(scopes: Sequence[Scope], *, clock=None) -> DocumentRepository:
        return cls(scopes, clock)

    return make


def test_every_check_has_a_broken_repository() -> None:
    assert sorted(c.__name__ for c in REPOSITORY_CHECKS) == sorted(BROKEN)


def test_in_memory_satisfies_protocol() -> None:
    assert isinstance(InMemoryDocumentRepository([]), DocumentRepository)


@pytest.mark.parametrize("check", REPOSITORY_CHECKS, ids=lambda c: c.__name__)
def test_checks_pass_on_in_memory(check) -> None:
    try:
        asyncio.run(check(_good))
    except CheckSkippedError as exc:
        pytest.skip(str(exc))


@pytest.mark.parametrize("check", REPOSITORY_CHECKS, ids=lambda c: c.__name__)
def test_each_check_fails_on_broken_repository(check) -> None:
    with pytest.raises(AssertionError):
        asyncio.run(check(_factory(BROKEN[check.__name__])))


def test_tie_break_check_reports_unpinnable_clock() -> None:
    async def no_clock(scopes: Sequence[Scope], *, clock=None) -> DocumentRepository:
        if clock is not None:
            raise CheckSkippedError("clock cannot be pinned")
        return InMemoryDocumentRepository(scopes)

    check = next(c for c in REPOSITORY_CHECKS if c.__name__ == "check_dedup_tie_break_lowest_version_id")
    with pytest.raises(CheckSkippedError):
        asyncio.run(check(no_clock))


def test_public_helpers() -> None:
    assert make_new_version("x").sha256 == make_new_version("x").sha256 != make_new_version("y").sha256
    assert make_scope().tenant_id != make_scope().tenant_id
