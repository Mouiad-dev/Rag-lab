"""In-memory DocumentRepository and reusable repository checks (stdlib and pydantic only).

Each check is `async def check_<name>(factory) -> None` and raises AssertionError with a clear
message. The in-memory repository, and later the SQL repository (P3.1-04), run the same checks.
"""

import asyncio
import hashlib
from collections.abc import Awaitable, Callable, Iterable, Sequence
from datetime import UTC, datetime, timedelta
from typing import Protocol

from pydantic import ValidationError

from rag_foundry_domain.documents import (
    Document,
    DocumentVersion,
    IntakeOutcome,
    MediaType,
    NewVersion,
    Scope,
    new_id,
)
from rag_foundry_domain.errors import NotFoundError
from rag_foundry_domain.extraction import (
    ExtractionKey,
    ExtractionOutcome,
    ExtractionView,
    NewExtraction,
)
from rag_foundry_domain.ports import DocumentRepository, ExtractionRepository

Clock = Callable[[], datetime]


class CheckSkippedError(Exception):
    """Raised by a RepoFactory (or a check) when a check cannot run against that implementation."""


class RepoFactory(Protocol):
    """Returns an empty repository in which exactly `scopes` exist.

    `clock`, when given, must be used for every `created_at`. An implementation that cannot pin its
    clock (for example a SQL repository using the database clock) raises CheckSkippedError when
    `clock` is not None; callers then skip the tie-break check instead of failing it.
    """

    def __call__(self, scopes: Sequence[Scope], *, clock: Clock | None = None) -> Awaitable[DocumentRepository]: ...


class _MonotonicUtcClock:
    """UTC clock that never returns the same instant twice, so creation order is unambiguous."""

    def __init__(self) -> None:
        self._last: datetime | None = None

    def __call__(self) -> datetime:
        now = datetime.now(UTC)
        if self._last is not None and now <= self._last:
            now = self._last + timedelta(microseconds=1)
        self._last = now
        return now


class InMemoryDocumentRepository:
    """Reference implementation. A lock makes dedup and numbering atomic within one event loop."""

    def __init__(self, projects: Iterable[Scope], clock: Callable[[], datetime] | None = None) -> None:
        self._projects = {(p.tenant_id, p.project_id) for p in projects}
        self._clock = clock or _MonotonicUtcClock()
        self._documents: dict[str, Document] = {}
        self._versions: list[DocumentVersion] = []
        self._lock = asyncio.Lock()

    # Hooks: small seams so tests can build deliberately broken repositories.
    def _dedup_candidates(self, scope: Scope) -> list[DocumentVersion]:
        return [v for v in self._versions if (v.tenant_id, v.project_id) == (scope.tenant_id, scope.project_id)]

    @staticmethod
    def _choose(matches: list[DocumentVersion]) -> DocumentVersion:
        return min(matches, key=lambda v: (v.created_at, v.id))

    @staticmethod
    def _existing_in_document(versions: list[DocumentVersion], sha256: str) -> DocumentVersion | None:
        return next((v for v in versions if v.sha256 == sha256), None)

    @staticmethod
    def _next_number(versions: list[DocumentVersion]) -> int:
        return max(v.version_number for v in versions) + 1

    def _require_project(self, scope: Scope) -> None:
        if (scope.tenant_id, scope.project_id) not in self._projects:
            raise NotFoundError()

    def _require_document(self, scope: Scope, document_id: str) -> Document:
        doc = self._documents.get(document_id)
        if doc is None or (doc.tenant_id, doc.project_id) != (scope.tenant_id, scope.project_id):
            raise NotFoundError()
        return doc

    def _versions_of(self, document_id: str) -> list[DocumentVersion]:
        return sorted((v for v in self._versions if v.document_id == document_id), key=lambda v: v.version_number)

    def _make_version(self, doc: Document, new: NewVersion, number: int, now: datetime) -> DocumentVersion:
        return DocumentVersion(
            **new.model_dump(),
            id=new_id(),
            document_id=doc.id,
            tenant_id=doc.tenant_id,
            project_id=doc.project_id,
            version_number=number,
            created_at=now,
        )

    async def project_exists(self, scope: Scope) -> bool:
        return (scope.tenant_id, scope.project_id) in self._projects

    async def get_document(self, scope: Scope, document_id: str) -> Document:
        return self._require_document(scope, document_id)

    async def create_document_or_get(self, scope: Scope, new: NewVersion, *, display_name: str) -> IntakeOutcome:
        async with self._lock:
            self._require_project(scope)
            matches = [v for v in self._dedup_candidates(scope) if v.sha256 == new.sha256]
            if matches:
                found = self._choose(matches)
                return IntakeOutcome(
                    document=self._documents[found.document_id],
                    version=found,
                    created_document=False,
                    created_version=False,
                )
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

    async def add_version_or_get(self, scope: Scope, document_id: str, new: NewVersion) -> IntakeOutcome:
        async with self._lock:
            doc = self._require_document(scope, document_id)
            existing = self._versions_of(doc.id)
            found = self._existing_in_document(existing, new.sha256)
            if found is not None:
                return IntakeOutcome(document=doc, version=found, created_document=False, created_version=False)
            version = self._make_version(doc, new, self._next_number(existing), self._clock())
            self._versions.append(version)
            return IntakeOutcome(document=doc, version=version, created_document=False, created_version=True)

    async def get_version(self, scope: Scope, document_id: str, version_id: str) -> DocumentVersion:
        doc = self._require_document(scope, document_id)
        for v in self._versions_of(doc.id):
            if v.id == version_id:
                return v
        raise NotFoundError()

    async def list_versions(self, scope: Scope, document_id: str) -> list[DocumentVersion]:
        doc = self._require_document(scope, document_id)
        return self._versions_of(doc.id)


# ---------------------------------------------------------------- checks

_PAUSE = 0.02  # lets coarse clocks (real databases) order creations


def _fail(message: str) -> None:
    raise AssertionError(message)


def make_scope() -> Scope:
    return Scope(tenant_id=new_id(), project_id=new_id())


def make_new_version(tag: str, name: str = "a.txt") -> NewVersion:
    digest = hashlib.sha256(tag.encode()).hexdigest()
    return NewVersion(
        sha256=digest,
        size_bytes=len(tag.encode()),
        media_type=MediaType.TEXT_PLAIN,
        text_encoding="utf-8",
        original_filename=name,
        object_key=f"sha256/{digest[:2]}/{digest}",
    )


async def _expect_not_found(coro: Awaitable[object], what: str) -> str:
    try:
        await coro
    except NotFoundError as exc:
        return str(exc)
    except Exception as exc:  # noqa: BLE001 - report the wrong type
        _fail(f"{what}: expected NotFoundError, got {type(exc).__name__}")
    _fail(f"{what}: expected NotFoundError, nothing was raised")
    return ""


async def check_same_bytes_dedup_within_project(factory: RepoFactory) -> None:
    scope = make_scope()
    repo = await factory([scope])
    first = await repo.create_document_or_get(scope, make_new_version("x"), display_name="one")
    again = await repo.create_document_or_get(scope, make_new_version("x", "other.txt"), display_name="two")
    if not (first.created_document and first.created_version and first.version.version_number == 1):
        _fail("first upload must create a document with version 1")
    if again.created_document or again.created_version or not again.deduplicated:
        _fail("same bytes in the same project must deduplicate")
    if again.document.id != first.document.id or again.version.id != first.version.id:
        _fail("dedup must return the existing document and version")


async def check_earliest_version_wins(factory: RepoFactory) -> None:
    scope = make_scope()
    repo = await factory([scope])
    a = await repo.create_document_or_get(scope, make_new_version("x"), display_name="a")
    await asyncio.sleep(_PAUSE)
    b = await repo.create_document_or_get(scope, make_new_version("y"), display_name="b")
    await asyncio.sleep(_PAUSE)
    await repo.add_version_or_get(scope, b.document.id, make_new_version("x"))  # b v2 holds the same bytes as a v1
    hit = await repo.create_document_or_get(scope, make_new_version("x"), display_name="c")
    if hit.version.id != a.version.id or hit.document.id != a.document.id:
        _fail("with several matching documents, the earliest-created version must win")
    if not hit.deduplicated:
        _fail("a match must be reported as deduplicated")


async def check_reupload_v1_bytes_returns_v1(factory: RepoFactory) -> None:
    scope = make_scope()
    repo = await factory([scope])
    v1 = await repo.create_document_or_get(scope, make_new_version("x"), display_name="a")
    await repo.add_version_or_get(scope, v1.document.id, make_new_version("y"))
    again = await repo.add_version_or_get(scope, v1.document.id, make_new_version("x"))
    if again.created_version or again.version.id != v1.version.id:
        _fail("re-uploading v1 bytes as a version of the same document must return v1, even when v2 exists")
    if len(await repo.list_versions(scope, v1.document.id)) != 2:
        _fail("dedup must not add a version")


async def check_changed_bytes_new_version(factory: RepoFactory) -> None:
    scope = make_scope()
    repo = await factory([scope])
    v1 = await repo.create_document_or_get(scope, make_new_version("x"), display_name="a")
    v2 = await repo.add_version_or_get(scope, v1.document.id, make_new_version("y"))
    if not v2.created_version or v2.created_document or v2.deduplicated:
        _fail("changed bytes must create a new version of the same document")
    if v2.version.id == v1.version.id or v2.document.id != v1.document.id:
        _fail("new version must be distinct and stay on the same document")


async def check_version_numbers_sequential(factory: RepoFactory) -> None:
    scope = make_scope()
    repo = await factory([scope])
    first = await repo.create_document_or_get(scope, make_new_version("t0"), display_name="a")
    for i in range(1, 4):
        await repo.add_version_or_get(scope, first.document.id, make_new_version(f"t{i}"))
    numbers = [v.version_number for v in await repo.list_versions(scope, first.document.id)]
    if numbers != [1, 2, 3, 4]:
        _fail(f"version numbers must be 1..n ascending, got {numbers}")


async def check_cross_scope_reads_not_found(factory: RepoFactory) -> None:
    scope = make_scope()
    other_tenant = make_scope()
    other_project = Scope(tenant_id=scope.tenant_id, project_id=new_id())
    # A project ID belongs to one tenant, so this forged pair is never seeded; it must still read nothing.
    forged = Scope(tenant_id=other_tenant.tenant_id, project_id=scope.project_id)
    repo = await factory([scope, other_tenant, other_project])
    out = await repo.create_document_or_get(scope, make_new_version("x"), display_name="a")
    doc_id, ver_id = out.document.id, out.version.id
    unknown = await _expect_not_found(repo.get_document(scope, new_id()), "unknown document")
    for label, other in (("tenant", other_tenant), ("forged tenant", forged), ("project", other_project)):
        messages = [
            await _expect_not_found(repo.get_document(other, doc_id), f"get_document across {label}"),
            await _expect_not_found(repo.get_version(other, doc_id, ver_id), f"get_version across {label}"),
            await _expect_not_found(repo.list_versions(other, doc_id), f"list_versions across {label}"),
            await _expect_not_found(
                repo.add_version_or_get(other, doc_id, make_new_version("z")), f"add_version across {label}"
            ),
        ]
        if any(m != unknown for m in messages):
            _fail(f"out-of-scope ({label}) errors must be indistinguishable from unknown IDs")
    await _expect_not_found(repo.get_version(scope, doc_id, new_id()), "unknown version")
    if len(await repo.list_versions(scope, doc_id)) != 1:
        _fail("a refused cross-scope write must not change the document")


async def check_same_bytes_other_project_separate_document(factory: RepoFactory) -> None:
    a, b = make_scope(), make_scope()
    repo = await factory([a, b])
    first = await repo.create_document_or_get(a, make_new_version("x"), display_name="a")
    second = await repo.create_document_or_get(b, make_new_version("x"), display_name="a")
    if not second.created_document or second.document.id == first.document.id:
        _fail("same bytes in another project must create a separate document")
    if (second.document.tenant_id, second.document.project_id) != (b.tenant_id, b.project_id):
        _fail("new document must carry the caller's scope")


async def check_unknown_project(factory: RepoFactory) -> None:
    known, unknown = make_scope(), make_scope()
    repo = await factory([known])
    if not await repo.project_exists(known):
        _fail("project_exists must be True for a project that exists")
    if await repo.project_exists(unknown):
        _fail("project_exists must be False for an unknown project")
    await _expect_not_found(
        repo.create_document_or_get(unknown, make_new_version("x"), display_name="a"), "create in unknown project"
    )


async def check_records_frozen(factory: RepoFactory) -> None:
    scope = make_scope()
    repo = await factory([scope])
    out = await repo.create_document_or_get(scope, make_new_version("x"), display_name="a")
    for record, field, value in (
        (out, "created_version", False),
        (out.document, "display_name", "changed"),
        (out.version, "sha256", "0" * 64),
        (out.version, "version_number", 9),
    ):
        try:
            setattr(record, field, value)
        except (ValidationError, TypeError, AttributeError):
            continue
        _fail(f"{type(record).__name__}.{field} must be immutable")


async def check_dedup_tie_break_lowest_version_id(factory: RepoFactory) -> None:
    scope = make_scope()
    fixed = datetime(2026, 1, 1, tzinfo=UTC)
    repo = await factory([scope], clock=lambda: fixed)
    a = await repo.create_document_or_get(scope, make_new_version("x"), display_name="a")
    b = await repo.create_document_or_get(scope, make_new_version("y"), display_name="b")
    b2 = await repo.add_version_or_get(scope, b.document.id, make_new_version("x"))
    if a.version.created_at != b2.version.created_at:
        _fail("the factory did not pin created_at to the given clock")
    hit = await repo.create_document_or_get(scope, make_new_version("x"), display_name="c")
    expected = min(a.version.id, b2.version.id)
    if hit.version.id != expected:
        _fail("with equal created_at, dedup must return the version with the lowest ID")


async def check_concurrent_same_bytes_dedup(factory: RepoFactory) -> None:
    scope = make_scope()
    repo = await factory([scope])
    outcomes = await asyncio.gather(
        *(repo.create_document_or_get(scope, make_new_version("same"), display_name="d") for _ in range(10))
    )
    if sum(o.created_document for o in outcomes) != 1:
        _fail("exactly one concurrent upload of the same bytes may create the document")
    if len({o.version.id for o in outcomes}) != 1 or len({o.document.id for o in outcomes}) != 1:
        _fail("concurrent uploads of the same bytes must all resolve to one document and version")


async def check_version_id_under_other_document_not_found(factory: RepoFactory) -> None:
    scope = make_scope()
    repo = await factory([scope])
    a = await repo.create_document_or_get(scope, make_new_version("x"), display_name="a")
    b = await repo.create_document_or_get(scope, make_new_version("y"), display_name="b")
    unknown = await _expect_not_found(repo.get_version(scope, a.document.id, new_id()), "unknown version")
    message = await _expect_not_found(
        repo.get_version(scope, b.document.id, a.version.id), "get_version with another document's version ID"
    )
    if message != unknown:
        _fail("a version ID under the wrong document must look like an unknown ID")


async def check_malformed_ids_not_found(factory: RepoFactory) -> None:
    scope = make_scope()
    repo = await factory([scope])
    out = await repo.create_document_or_get(scope, make_new_version("x"), display_name="a")
    unknown = await _expect_not_found(repo.get_document(scope, new_id()), "unknown document")
    bad = "not-a-uuid"
    messages = [
        await _expect_not_found(repo.get_document(scope, bad), "get_document with a malformed ID"),
        await _expect_not_found(repo.list_versions(scope, bad), "list_versions with a malformed ID"),
        await _expect_not_found(
            repo.get_version(scope, bad, out.version.id), "get_version with a malformed document ID"
        ),
        await _expect_not_found(
            repo.get_version(scope, out.document.id, bad), "get_version with a malformed version ID"
        ),
        await _expect_not_found(repo.add_version_or_get(scope, bad, make_new_version("z")), "add_version malformed ID"),
    ]
    if any(m != unknown for m in messages):
        _fail("malformed IDs must look like unknown IDs")


REPOSITORY_CHECKS = (
    check_same_bytes_dedup_within_project,
    check_earliest_version_wins,
    check_reupload_v1_bytes_returns_v1,
    check_changed_bytes_new_version,
    check_version_numbers_sequential,
    check_cross_scope_reads_not_found,
    check_same_bytes_other_project_separate_document,
    check_unknown_project,
    check_records_frozen,
    check_dedup_tie_break_lowest_version_id,
    check_concurrent_same_bytes_dedup,
    check_version_id_under_other_document_not_found,
    check_malformed_ids_not_found,
)


# ---------------------------------------------------------------- extraction repository


class ExtractionRepoFactory(Protocol):
    """Returns an empty (document repository, extraction repository) pair in which exactly `scopes`
    exist. The extraction repository must see versions created through the document repository."""

    def __call__(self, scopes: Sequence[Scope]) -> Awaitable[tuple[DocumentRepository, ExtractionRepository]]: ...


class InMemoryExtractionRepository:
    """Reference implementation. A lock makes dedup atomic within one event loop."""

    def __init__(self, documents: InMemoryDocumentRepository, clock: Clock | None = None) -> None:
        self._documents = documents
        self._clock = clock or _MonotonicUtcClock()
        self._views: list[ExtractionView] = []
        self._lock = asyncio.Lock()

    # Hooks: small seams so tests can build deliberately broken repositories.
    def _in_scope(self, scope: Scope) -> list[ExtractionView]:
        return [v for v in self._views if (v.tenant_id, v.project_id) == (scope.tenant_id, scope.project_id)]

    def _match(self, scope: Scope, key: ExtractionKey) -> ExtractionView | None:
        return next((v for v in self._in_scope(scope) if v.key == key), None)

    @staticmethod
    def _order(views: list[ExtractionView]) -> list[ExtractionView]:
        return sorted(views, key=lambda v: (v.created_at, v.id))

    def _require_version(self, scope: Scope, version_id: str) -> None:
        for v in self._documents._versions:
            if v.id == version_id and (v.tenant_id, v.project_id) == (scope.tenant_id, scope.project_id):
                return
        raise NotFoundError()

    async def find_view(self, scope: Scope, key: ExtractionKey) -> ExtractionView | None:
        return self._match(scope, key)

    async def create_view_or_get(self, scope: Scope, new: NewExtraction) -> ExtractionOutcome:
        async with self._lock:
            await self._documents.get_version(scope, new.document_id, new.document_version_id)
            found = self._match(scope, new.key)
            if found is not None:
                return ExtractionOutcome(view=found, created=False)
            view = ExtractionView(
                **new.model_dump(),
                id=new_id(),
                tenant_id=scope.tenant_id,
                project_id=scope.project_id,
                created_at=self._clock(),
            )
            self._views.append(view)
            return ExtractionOutcome(view=view, created=True)

    async def get_view(self, scope: Scope, view_id: str) -> ExtractionView:
        for v in self._in_scope(scope):
            if v.id == view_id:
                return v
        raise NotFoundError()

    async def list_views(self, scope: Scope, document_version_id: str) -> list[ExtractionView]:
        self._require_version(scope, document_version_id)
        return self._order([v for v in self._in_scope(scope) if v.document_version_id == document_version_id])


def make_new_extraction(
    version: DocumentVersion,
    *,
    tag: str = "x",
    extractor: str = "plain-text",
    extractor_version: str = "1",
    profile_id: str = "canonical-v1",
    unicode_version: str = "15.1.0",
) -> NewExtraction:
    canonical = hashlib.sha256(f"canonical:{tag}".encode()).hexdigest()
    payload = hashlib.sha256(f"payload:{tag}".encode()).hexdigest()
    return NewExtraction(
        document_id=version.document_id,
        document_version_id=version.id,
        extractor=extractor,
        extractor_version=extractor_version,
        profile_id=profile_id,
        unicode_version=unicode_version,
        canonical_sha256=canonical,
        char_count=10,
        block_count=1,
        payload_sha256=payload,
        payload_key=f"extractions/{payload[:2]}/{payload}",
    )


async def _extraction_setup(
    factory: ExtractionRepoFactory, scopes: Sequence[Scope]
) -> tuple[DocumentRepository, ExtractionRepository, list[DocumentVersion]]:
    docs, repo = await factory(scopes)
    versions = []
    for i, scope in enumerate(scopes):
        out = await docs.create_document_or_get(scope, make_new_version(f"doc{i}"), display_name="a")
        versions.append(out.version)
    return docs, repo, versions


async def check_extraction_same_key_dedup(factory: ExtractionRepoFactory) -> None:
    scope = make_scope()
    _, repo, (ver,) = await _extraction_setup(factory, [scope])
    first = await repo.create_view_or_get(scope, make_new_extraction(ver))
    again = await repo.create_view_or_get(scope, make_new_extraction(ver, tag="other-payload"))
    if not first.created:
        _fail("first create must report created=True")
    if again.created or again.view.id != first.view.id:
        _fail("the same identity key must return the existing view with created=False")
    found = await repo.find_view(scope, first.view.key)
    if found is None or found.id != first.view.id:
        _fail("find_view must find a created view by its key")
    if len(await repo.list_views(scope, ver.id)) != 1:
        _fail("dedup must not add a view")


async def check_extraction_changed_key_new_view(factory: ExtractionRepoFactory) -> None:
    scope = make_scope()
    _, repo, (ver,) = await _extraction_setup(factory, [scope])
    base = await repo.create_view_or_get(scope, make_new_extraction(ver))
    other_version = await repo.create_view_or_get(scope, make_new_extraction(ver, extractor_version="2"))
    other_profile = await repo.create_view_or_get(scope, make_new_extraction(ver, profile_id="other-profile"))
    other_unicode = await repo.create_view_or_get(scope, make_new_extraction(ver, unicode_version="99.0.0"))
    ids = {base.view.id, other_version.view.id, other_profile.view.id, other_unicode.view.id}
    if len(ids) != 4 or not (other_version.created and other_profile.created and other_unicode.created):
        _fail("a changed extractor_version, profile_id or unicode_version must create a new view")
    if await repo.find_view(scope, base.view.key) is None:
        _fail("old views must stay valid after a new view is created")


async def check_extraction_cross_scope_not_found(factory: ExtractionRepoFactory) -> None:
    scope, other_tenant = make_scope(), make_scope()
    other_project = Scope(tenant_id=scope.tenant_id, project_id=new_id())
    _, repo, (ver, *_) = await _extraction_setup(factory, [scope, other_tenant, other_project])
    out = await repo.create_view_or_get(scope, make_new_extraction(ver))
    unknown = await _expect_not_found(repo.get_view(scope, new_id()), "unknown view")
    for label, other in (("tenant", other_tenant), ("project", other_project)):
        messages = [
            await _expect_not_found(repo.get_view(other, out.view.id), f"get_view across {label}"),
            await _expect_not_found(repo.list_views(other, ver.id), f"list_views across {label}"),
            await _expect_not_found(
                repo.create_view_or_get(other, make_new_extraction(ver)), f"create_view across {label}"
            ),
        ]
        if any(m != unknown for m in messages):
            _fail(f"out-of-scope ({label}) errors must be indistinguishable from unknown IDs")
        if await repo.find_view(other, out.view.key) is not None:
            _fail(f"find_view across {label} must return None")
    if len(await repo.list_views(scope, ver.id)) != 1:
        _fail("a refused cross-scope write must not change the version's views")


async def check_extraction_unknown_and_malformed_ids(factory: ExtractionRepoFactory) -> None:
    scope = make_scope()
    docs, repo, (ver,) = await _extraction_setup(factory, [scope])
    out = await repo.create_view_or_get(scope, make_new_extraction(ver))
    unknown = await _expect_not_found(repo.get_view(scope, new_id()), "unknown view")
    bad = "not-a-uuid"
    ghost = ver.model_copy(update={"id": new_id()})
    bad_new = make_new_extraction(ver).model_copy(update={"document_version_id": bad})
    bad_key = out.view.key.model_copy(update={"document_version_id": bad})
    other_doc = (await docs.create_document_or_get(scope, make_new_version("other-doc"), display_name="b")).document
    wrong_owner = make_new_extraction(ver).model_copy(update={"document_id": other_doc.id})
    messages = [
        await _expect_not_found(repo.create_view_or_get(scope, wrong_owner), "create_view with a foreign document ID"),
        await _expect_not_found(repo.get_view(scope, bad), "get_view with a malformed ID"),
        await _expect_not_found(repo.list_views(scope, bad), "list_views with a malformed ID"),
        await _expect_not_found(repo.list_views(scope, new_id()), "list_views with an unknown ID"),
        await _expect_not_found(
            repo.create_view_or_get(scope, make_new_extraction(ghost)), "create_view with an unknown version"
        ),
        await _expect_not_found(repo.create_view_or_get(scope, bad_new), "create_view with a malformed version ID"),
    ]
    if any(m != unknown for m in messages):
        _fail("unknown and malformed IDs must look the same")
    if await repo.find_view(scope, out.view.key.model_copy(update={"document_version_id": new_id()})) is not None:
        _fail("find_view with an unknown version must return None")
    if await repo.find_view(scope, bad_key) is not None:
        _fail("find_view with a malformed version ID must return None")
    if len(await repo.list_views(scope, ver.id)) != 1:
        _fail("a refused create must store nothing")


async def check_extraction_concurrent_creates(factory: ExtractionRepoFactory) -> None:
    scope = make_scope()
    _, repo, (ver,) = await _extraction_setup(factory, [scope])
    outcomes = await asyncio.gather(*(repo.create_view_or_get(scope, make_new_extraction(ver)) for _ in range(10)))
    if sum(o.created for o in outcomes) != 1 or len({o.view.id for o in outcomes}) != 1:
        _fail("concurrent creates of the same key must produce exactly one view")
    if len(await repo.list_views(scope, ver.id)) != 1:
        _fail("concurrent creates of the same key must leave one stored view")


async def check_extraction_list_order(factory: ExtractionRepoFactory) -> None:
    scope = make_scope()
    _, repo, (ver,) = await _extraction_setup(factory, [scope])
    created = []
    for i in range(4):
        created.append((await repo.create_view_or_get(scope, make_new_extraction(ver, extractor_version=str(i)))).view)
        await asyncio.sleep(_PAUSE)
    expected = [v.id for v in sorted(created, key=lambda v: (v.created_at, v.id))]
    got = [v.id for v in await repo.list_views(scope, ver.id)]
    if got != expected or got != [v.id for v in created]:
        _fail("list_views must be ordered by created_at, then id")


async def check_extraction_records_frozen(factory: ExtractionRepoFactory) -> None:
    scope = make_scope()
    _, repo, (ver,) = await _extraction_setup(factory, [scope])
    out = await repo.create_view_or_get(scope, make_new_extraction(ver))
    for record, field, value in (
        (out, "created", False),
        (out.view, "profile_id", "changed"),
        (out.view, "canonical_sha256", "0" * 64),
        (out.view, "char_count", 99),
    ):
        try:
            setattr(record, field, value)
        except (ValidationError, TypeError, AttributeError):
            continue
        _fail(f"{type(record).__name__}.{field} must be immutable")


EXTRACTION_REPOSITORY_CHECKS = (
    check_extraction_same_key_dedup,
    check_extraction_changed_key_new_view,
    check_extraction_cross_scope_not_found,
    check_extraction_unknown_and_malformed_ids,
    check_extraction_concurrent_creates,
    check_extraction_list_order,
    check_extraction_records_frozen,
)
