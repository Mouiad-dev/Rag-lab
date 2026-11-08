"""Portable source bundle v1: scoped export, validated import and a scope-bound local catalog.

A bundle carries canonical text, block structure and anchors for explicit extraction views, so a
consumer can resolve the same anchors without the original store, database or service. It never
carries tenant or project IDs, originals, signed URLs, chunk IDs or index IDs.

SHA-256 values detect accidental mismatch only; a bundle is untrusted input and is not signed.
"""

import hashlib
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError
from rag_foundry_contracts.bundle import (
    BUNDLE_ASSET_PATH,
    BundleAsset,
    BundleBlock,
    BundleDocument,
    BundleManifest,
    OmittedAsset,
)
from rag_foundry_contracts.resolve import AnchorResolution, ResolutionReason, ResolutionStatus
from rag_foundry_contracts.source import SourceAnchor, SourceRef
from rag_foundry_domain.errors import AnchorError
from rag_foundry_domain.extraction import SourceBlock
from rag_foundry_domain.ports import ObjectStore, Scope

from rag_foundry_ingestion.anchors import anchor_id_for, resolve_anchor
from rag_foundry_ingestion.extraction import read_payload

MANIFEST = "manifest.json"
DOCUMENTS = "documents.json"
STRUCTURE = "structure.jsonl"
ANCHORS = "anchors.jsonl"
_FIXED = {
    DOCUMENTS: ("documents", "application/json"),
    STRUCTURE: ("structure", "application/jsonl"),
    ANCHORS: ("anchors", "application/jsonl"),
}
_TEXT_MEDIA = "text/plain; charset=utf-8"


class BundleError(ValueError):
    """Bundle rejected; `code` is a stable reason such as `hash_mismatch`."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class BundleLimits:
    max_files: int = 1000
    max_file_bytes: int = 20_000_000
    max_total_bytes: int = 50_000_000
    max_anchors: int = 10_000


DEFAULT_LIMITS = BundleLimits()


@dataclass(frozen=True)
class _View:
    """What `resolve_anchor` needs from one extraction view."""

    canonical_text: str
    blocks: tuple[SourceBlock, ...]


@dataclass(frozen=True)
class ImportedBundle:
    manifest: BundleManifest
    documents: tuple[BundleDocument, ...]
    anchors: tuple[SourceAnchor, ...]
    manifest_digest: str
    views: Mapping[str, _View] = field(repr=False)


def _json(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _jsonl(values: Iterable) -> bytes:
    return b"".join(_json(v) + b"\n" for v in values)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _anchor_is_proven(anchor: SourceAnchor, ref: SourceRef, view) -> bool:
    """The anchor names `ref`, its id is the one derived from its own content, and it resolves exactly."""
    if anchor.source != ref or anchor.text_range is None:
        return False
    try:
        expected = anchor_id_for(ref, anchor.block_id, anchor.text_range.start, anchor.text_range.end)
    except AnchorError:
        return False
    return anchor.anchor_id == expected and resolve_anchor(view, ref, anchor).status is ResolutionStatus.EXACT


def _utc_stamp(moment: datetime) -> str:
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _check_limits(files: Mapping[str, bytes], limits: BundleLimits) -> None:
    if not all(isinstance(name, str) for name in files):
        raise BundleError("bad_path")
    if not all(isinstance(body, bytes) for body in files.values()):
        raise BundleError("invalid_structure")
    if (
        len(files) > limits.max_files
        or any(len(body) > limits.max_file_bytes for body in files.values())
        or sum(len(body) for body in files.values()) > limits.max_total_bytes
    ):
        raise BundleError("too_large")


def _check_name(name: str) -> None:
    if not isinstance(name, str) or (name != MANIFEST and not BUNDLE_ASSET_PATH.fullmatch(name)):
        raise BundleError("bad_path")


class BundleExportService:
    """Exports explicit views and anchors of one scope. Every view is authorized before any payload read."""

    def __init__(
        self,
        store: ObjectStore,
        uow,
        *,
        provider_namespace: str,
        clock: Callable[[], datetime],
        limits: BundleLimits = DEFAULT_LIMITS,
    ):
        self._store = store
        self._uow = uow
        self._namespace = provider_namespace
        self._clock = clock
        self._limits = limits

    async def export(self, scope: Scope, view_ids: Sequence[str], anchors: Sequence[SourceAnchor]) -> dict[str, bytes]:
        if not view_ids or len(set(view_ids)) != len(view_ids):
            raise BundleError("invalid_structure")
        if len(anchors) > self._limits.max_anchors:
            raise BundleError("too_large")
        async with self._uow(read_only=True) as uow:
            views = [await uow.extractions.get_view(scope, view_id) for view_id in view_ids]
        payloads = [await read_payload(self._store, scope, view) for view in views]
        refs = [
            SourceRef(
                provider_namespace=self._namespace,
                document_id=v.document_id,
                document_version_id=v.document_version_id,
                extraction_view_id=v.id,
            )
            for v in views
        ]
        by_view = {ref.extraction_view_id: (payload, ref) for payload, ref in zip(payloads, refs, strict=True)}
        if len({a.anchor_id for a in anchors}) != len(anchors):
            raise BundleError("anchor_not_exportable")
        for anchor in anchors:
            item = by_view.get(anchor.source.extraction_view_id)
            if item is None or not _anchor_is_proven(anchor, item[1], item[0]):
                raise BundleError("anchor_not_exportable")

        files: dict[str, bytes] = {}
        documents, blocks, omitted = [], [], []
        for payload, ref in zip(payloads, refs, strict=True):
            text_path = f"text/{ref.extraction_view_id}.txt"
            text = payload.canonical_text.encode("utf-8")
            files[text_path] = text
            documents.append(
                BundleDocument(
                    source=ref,
                    extractor=payload.extractor,
                    extractor_version=payload.extractor_version,
                    profile_id=payload.profile_id,
                    unicode_version=payload.unicode_version,
                    canonical_sha256=_sha(text),
                    char_count=len(payload.canonical_text),
                    block_count=len(payload.blocks),
                    page_count=None if payload.page_starts is None else len(payload.page_starts),
                    text_path=text_path,
                )
            )
            blocks.extend(
                BundleBlock(
                    extraction_view_id=ref.extraction_view_id,
                    block_id=b.block_id,
                    ordinal=b.ordinal,
                    kind=b.kind.value,
                    start=b.start,
                    end=b.end,
                    section_path=b.section_path,
                    heading_level=b.heading_level,
                    page_index=b.page_index,
                )
                for b in payload.blocks
            )
            omitted.append(
                OmittedAsset(document_version_id=ref.document_version_id, kind="original", reason="not_exported")
            )
            if payload.page_starts is not None:
                omitted.append(
                    OmittedAsset(document_version_id=ref.document_version_id, kind="page_render", reason="not_exported")
                )
        files[DOCUMENTS] = _json([d.model_dump(mode="json") for d in documents])
        files[STRUCTURE] = _jsonl(b.model_dump(mode="json") for b in blocks)
        files[ANCHORS] = _jsonl(a.model_dump(mode="json") for a in anchors)

        assets = tuple(
            BundleAsset(
                path=path,
                role=_FIXED.get(path, ("text", _TEXT_MEDIA))[0],
                media_type=_FIXED.get(path, ("text", _TEXT_MEDIA))[1],
                size_bytes=len(body),
                sha256=_sha(body),
            )
            for path, body in sorted(files.items())
        )
        identity = _json([self._namespace, sorted(view_ids), sorted(a.anchor_id for a in anchors)])
        manifest = BundleManifest(
            bundle_id="bun_" + _sha(identity)[:32],
            provider_namespace=self._namespace,
            created_at=_utc_stamp(self._clock()),
            inclusion_policy="text_and_structure",
            views=tuple(refs),
            assets=assets,
            omitted=tuple(omitted),
        )
        files[MANIFEST] = _json(manifest.model_dump(mode="json"))
        _check_limits(files, self._limits)
        return files


def _unique_keys(pairs):
    keys = [key for key, _ in pairs]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate JSON key")
    return dict(pairs)


def _loads(raw: bytes):
    """Strict JSON: UTF-8 and no repeated keys. Deep nesting is rejected like any other malformed input."""
    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_keys)
    except (UnicodeDecodeError, ValueError, RecursionError):
        raise BundleError("invalid_json") from None


def _parse_manifest(raw: bytes) -> BundleManifest:
    data = _loads(raw)
    version = data.get("contract_version") if isinstance(data, dict) else None
    if isinstance(version, str) and not version.startswith("1."):
        raise BundleError("unsupported_version")
    try:
        return BundleManifest.model_validate(data)
    except ValidationError:
        raise BundleError("invalid_json") from None


def _parse_lines(raw: bytes, model):
    # Split on LF only: str.splitlines() would also split inside values holding U+2028, U+0085 and similar.
    lines = raw.split(b"\n")
    if lines and lines[-1] == b"":
        lines.pop()
    try:
        return tuple(model.model_validate(_loads(line.removesuffix(b"\r"))) for line in lines)
    except ValidationError:
        raise BundleError("invalid_json") from None


def import_bundle(files: Mapping[str, bytes], limits: BundleLimits = DEFAULT_LIMITS) -> ImportedBundle:
    """Validates a whole bundle; returns it or raises `BundleError`. Nothing is partly imported."""
    _check_limits(files, limits)
    for name in files:
        _check_name(name)
    if MANIFEST not in files:
        raise BundleError("missing_file")
    manifest = _parse_manifest(files[MANIFEST])

    listed = {asset.path: asset for asset in manifest.assets}
    if set(files) - set(listed) - {MANIFEST}:
        raise BundleError("unlisted_file")
    if set(listed) - set(files) or not set(_FIXED) <= set(listed):
        raise BundleError("missing_file")
    for path, asset in listed.items():
        if len(files[path]) != asset.size_bytes or _sha(files[path]) != asset.sha256:
            raise BundleError("hash_mismatch")

    raw_docs = _loads(files[DOCUMENTS])
    if not isinstance(raw_docs, list):
        raise BundleError("invalid_json")
    try:
        documents = tuple(BundleDocument.model_validate(d) for d in raw_docs)
    except ValidationError:
        raise BundleError("invalid_json") from None
    blocks = _parse_lines(files[STRUCTURE], BundleBlock)
    anchors = _parse_lines(files[ANCHORS], SourceAnchor)
    if len(anchors) > limits.max_anchors:
        raise BundleError("too_large")

    refs = {ref.extraction_view_id: ref for ref in manifest.views}
    text_paths = {p for p, a in listed.items() if a.role == "text"}
    if (
        len(refs) != len(manifest.views)
        or [d.source for d in documents] != list(manifest.views)
        or {d.text_path for d in documents} != text_paths
        or len(text_paths) != len(documents)
        or any(d.text_path != f"text/{d.source.extraction_view_id}.txt" for d in documents)
        or any(d.source.provider_namespace != manifest.provider_namespace for d in documents)
    ):
        raise BundleError("invalid_structure")

    texts: dict[str, str] = {}
    for doc in documents:
        try:
            text = files[doc.text_path].decode("utf-8")
        except UnicodeDecodeError:
            raise BundleError("invalid_structure") from None
        if len(text) != doc.char_count or _sha(files[doc.text_path]) != doc.canonical_sha256:
            raise BundleError("hash_mismatch")
        texts[doc.source.extraction_view_id] = text

    grouped: dict[str, list[SourceBlock]] = {view_id: [] for view_id in texts}
    for block in blocks:
        view_blocks = grouped.get(block.extraction_view_id)
        if view_blocks is None:
            raise BundleError("invalid_structure")
        previous_end = view_blocks[-1].end if view_blocks else 0
        if (
            block.ordinal != len(view_blocks)
            or block.start < previous_end
            or block.end > len(texts[block.extraction_view_id])
        ):
            raise BundleError("invalid_structure")
        try:
            view_blocks.append(SourceBlock.model_validate(block.model_dump(exclude={"extraction_view_id"})))
        except ValidationError:
            raise BundleError("invalid_structure") from None
    for doc in documents:
        if len(grouped[doc.source.extraction_view_id]) != doc.block_count:
            raise BundleError("invalid_structure")
    views = {view_id: _View(texts[view_id], tuple(grouped[view_id])) for view_id in texts}

    if len({a.anchor_id for a in anchors}) != len(anchors):
        raise BundleError("dangling_anchor")
    for anchor in anchors:
        ref = refs.get(anchor.source.extraction_view_id)
        if ref is None or not _anchor_is_proven(anchor, ref, views[ref.extraction_view_id]):
            raise BundleError("dangling_anchor")

    # `created_at` is left out so a later re-export of identical content is the same bundle.
    digest = _sha(_json(manifest.model_dump(mode="json", exclude={"created_at"})))
    return ImportedBundle(manifest=manifest, documents=documents, anchors=anchors, manifest_digest=digest, views=views)


class LocalBundleCatalog:
    """Imported bundles keyed by (scope, bundle_id). A scope only ever sees bundles added under it."""

    def __init__(self):
        self._bundles: dict[tuple[Scope, str], ImportedBundle] = {}

    def add(self, scope: Scope, bundle: ImportedBundle) -> None:
        key = (scope, bundle.manifest.bundle_id)
        current = self._bundles.get(key)
        if current is not None and current.manifest_digest != bundle.manifest_digest:
            raise BundleError("bundle_conflict")
        # One view ref has one text per scope; another bundle may repeat it only with the same content.
        incoming = {d.source: d.canonical_sha256 for d in bundle.documents}
        for (owner, other_id), other in self._bundles.items():
            if owner != scope or other_id == bundle.manifest.bundle_id:
                continue
            for doc in other.documents:
                if incoming.get(doc.source, doc.canonical_sha256) != doc.canonical_sha256:
                    raise BundleError("bundle_conflict")
        self._bundles[key] = bundle

    def resolve(self, scope: Scope, anchors: Sequence[SourceAnchor]) -> tuple[AnchorResolution, ...]:
        in_scope = [b for (owner, _), b in self._bundles.items() if owner == scope]
        namespaces = {b.manifest.provider_namespace for b in in_scope}
        results = []
        for anchor in anchors:
            source = anchor.source
            match = next(((b, r) for b in in_scope for r in b.manifest.views if r == source), None)
            if match is not None:
                bundle, ref = match
                results.append(resolve_anchor(bundle.views[ref.extraction_view_id], ref, anchor))
                continue
            # A scope with no bundles, or a known namespace without that view, answers "does not exist";
            # only a namespace this scope has never imported is reported as foreign.
            reason = (
                ResolutionReason.FOREIGN_NAMESPACE
                if namespaces and source.provider_namespace not in namespaces
                else ResolutionReason.NOT_FOUND
            )
            results.append(
                AnchorResolution(anchor_id=anchor.anchor_id, status=ResolutionStatus.UNAVAILABLE, reason=reason)
            )
        return tuple(results)


def write_bundle_dir(files: Mapping[str, bytes], root: Path) -> None:
    """Writes a bundle into `root`, which must be absent or an empty directory."""
    for name in files:
        _check_name(name)
    if root.is_symlink() or root.is_junction() or (root.exists() and (not root.is_dir() or any(root.iterdir()))):
        raise BundleError("bad_path")
    for name, body in files.items():
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.parent.is_symlink() or target.parent.is_junction() or target.is_symlink():
            raise BundleError("bad_path")
        target.write_bytes(body)


def read_bundle_dir(root: Path, limits: BundleLimits = DEFAULT_LIMITS) -> dict[str, bytes]:
    """Reads a bundle directory. Rejects symlinks, any directory but `text/`, and names off the allowlist."""
    if root.is_symlink() or root.is_junction() or not root.is_dir():
        raise BundleError("bad_path")
    files: dict[str, bytes] = {}
    total = 0
    pending = [root]
    while pending:
        directory = pending.pop()
        for entry in directory.iterdir():
            name = entry.relative_to(root).as_posix()
            if entry.is_symlink() or entry.is_junction():
                raise BundleError("bad_path")
            if entry.is_dir():
                if name != "text":
                    raise BundleError("bad_path")
                pending.append(entry)
                continue
            _check_name(name)
            if not entry.is_file():
                raise BundleError("bad_path")
            size = entry.stat().st_size
            total += size
            if len(files) >= limits.max_files or size > limits.max_file_bytes or total > limits.max_total_bytes:
                raise BundleError("too_large")
            files[name] = entry.read_bytes()
    return files
