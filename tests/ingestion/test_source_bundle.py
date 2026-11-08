import dataclasses
import json
import os
from datetime import UTC, datetime

import pytest
from rag_foundry_contracts.resolve import ResolutionReason, ResolutionStatus
from rag_foundry_contracts.source import SourceRef
from rag_foundry_domain import NotFoundError
from rag_foundry_domain.testing import make_scope
from rag_foundry_domain.uow_testing import InMemoryUnitOfWorkFactory
from rag_foundry_ingestion.anchors import make_anchor
from rag_foundry_ingestion.bundle import (
    ANCHORS,
    DOCUMENTS,
    MANIFEST,
    STRUCTURE,
    BundleError,
    BundleExportService,
    BundleLimits,
    LocalBundleCatalog,
    _parse_lines,
    import_bundle,
    read_bundle_dir,
    write_bundle_dir,
)
from rag_foundry_ingestion.extract import EXTRACTORS, ExtractionLimits
from rag_foundry_ingestion.extraction import ExtractionService
from test_anchor_rechunk import ARABIC, MARKDOWN, MIXED, REPEATED, sync
from test_extraction_service import ingest, setup

NS = "lab.local"
CLOCK = lambda: datetime(2026, 11, 8, 10, 0, tzinfo=UTC)  # noqa: E731


async def _world(tmp_path):
    owner, stranger = make_scope(), make_scope()
    _svc, intake, spy, _store, docs, views, _ = setup(tmp_path / "store", owner, stranger)
    uow = InMemoryUnitOfWorkFactory(documents=docs, extractions=views)
    extraction = ExtractionService(spy, uow, extractors=EXTRACTORS, limits=ExtractionLimits())
    found = {}
    for scope, text in ((owner, MARKDOWN), (stranger, "# Other tenant\n\nSecret paragraph.\n")):
        doc_id, version_id = await ingest(intake, scope, text.encode("utf-8"), name="a.md", media="text/markdown")
        view = (await extraction.extract(scope, doc_id, version_id)).view
        found[scope] = (view, await extraction.load_payload(scope, view))
    view, payload = found[owner]
    ref = SourceRef(
        provider_namespace=NS,
        document_id=view.document_id,
        document_version_id=view.document_version_id,
        extraction_view_id=view.id,
    )
    anchors = []
    for quote in ("evidence stays pinned", ARABIC[:9], "\U0001f680 keeps", "repeated paragraph"):
        block = [b for b in payload.blocks if quote in payload.canonical_text[b.start : b.end]][-1]
        start = payload.canonical_text.index(quote, block.start) - block.start
        anchors.append(make_anchor(payload, ref, block.block_id, start, start + len(quote)))
    service = BundleExportService(spy, uow, provider_namespace=NS, clock=CLOCK)
    return owner, stranger, spy, service, found, anchors


async def _bundle(tmp_path):
    owner, _stranger, _spy, service, found, anchors = await _world(tmp_path)
    return owner, await service.export(owner, [found[owner][0].id], anchors), anchors


def _relist(files, changed):
    """Rewrites the manifest so `changed` files are listed with matching hashes (a consistent tampered bundle)."""
    import hashlib

    files = dict(files) | changed
    manifest = json.loads(files[MANIFEST])
    for asset in manifest["assets"]:
        body = files[asset["path"]]
        asset["size_bytes"], asset["sha256"] = len(body), hashlib.sha256(body).hexdigest()
    files[MANIFEST] = json.dumps(manifest).encode()
    return files


def _jsonl_body(lines):
    newline = chr(10)
    return "".join(line + newline for line in lines).encode()


def _code(files, **kwargs):
    with pytest.raises(BundleError) as caught:
        import_bundle(files, **kwargs)
    return caught.value.code


@sync
async def test_round_trip_resolves_every_anchor_in_another_scope_without_the_original_store(tmp_path):
    owner, files, anchors = await _bundle(tmp_path)
    write_bundle_dir(files, tmp_path / "bundle")
    imported = import_bundle(read_bundle_dir(tmp_path / "bundle"))
    consumer = make_scope()
    catalog = LocalBundleCatalog()
    catalog.add(consumer, imported)
    (tmp_path / "store").rename(tmp_path / "gone")  # the original object store is gone; only the bundle remains
    results = catalog.resolve(consumer, anchors)
    assert [r.status for r in results] == [ResolutionStatus.EXACT] * len(anchors)
    for anchor, result in zip(anchors, results, strict=True):
        assert (result.exact, result.block_id, result.section_path) == (
            anchor.exact,
            anchor.block_id,
            anchor.section_path,
        )
    # the repeated paragraph anchor points at the second copy, not the first
    assert (
        anchors[3].block_id
        == [
            b
            for b in imported.views[anchors[3].source.extraction_view_id].blocks
            if REPEATED in imported.views[anchors[3].source.extraction_view_id].canonical_text[b.start : b.end]
        ][-1].block_id
    )
    assert MIXED[:5] in imported.views[anchors[2].source.extraction_view_id].canonical_text
    for body in files.values():
        assert str(owner.tenant_id).encode() not in body and str(owner.project_id).encode() not in body
        assert str(consumer.tenant_id).encode() not in body
    manifest = json.loads(files[MANIFEST])
    assert [o["kind"] for o in manifest["omitted"]] == ["original"]


@sync
async def test_export_is_deterministic_and_lists_every_file(tmp_path):
    owner, _stranger, _spy, service, found, anchors = await _world(tmp_path)
    first = await service.export(owner, [found[owner][0].id], anchors)
    second = await service.export(owner, [found[owner][0].id], list(reversed(anchors)))
    assert json.loads(first[MANIFEST])["bundle_id"] == json.loads(second[MANIFEST])["bundle_id"]
    listed = {a["path"] for a in json.loads(first[MANIFEST])["assets"]}
    assert listed | {MANIFEST} == set(first)


@sync
async def test_export_of_another_tenants_view_fails_before_any_payload_read(tmp_path):
    owner, stranger, spy, service, found, anchors = await _world(tmp_path)
    spy.reads.clear()
    with pytest.raises(NotFoundError):
        await service.export(owner, [found[owner][0].id, found[stranger][0].id], anchors)
    assert spy.reads == []


@sync
async def test_export_rejects_anchors_it_cannot_prove(tmp_path):
    owner, _stranger, _spy, service, found, anchors = await _world(tmp_path)
    view_id = found[owner][0].id
    foreign = anchors[0].model_copy(update={"source": anchors[0].source.model_copy(update={"provider_namespace": "x"})})
    edited = anchors[0].model_copy(update={"exact": anchors[0].exact + "!"})
    for bad in (foreign, edited):
        with pytest.raises(BundleError) as caught:
            await service.export(owner, [view_id], [bad])
        assert caught.value.code == "anchor_not_exportable"


@sync
async def test_scope_isolation_and_idempotent_add(tmp_path):
    _owner, files, anchors = await _bundle(tmp_path)
    imported = import_bundle(files)
    consumer, third = make_scope(), make_scope()
    catalog = LocalBundleCatalog()
    catalog.add(consumer, imported)
    catalog.add(consumer, import_bundle(files))  # same bytes: no-op
    assert {r.reason for r in catalog.resolve(third, anchors)} == {ResolutionReason.NOT_FOUND}
    foreign = anchors[0].model_copy(update={"source": anchors[0].source.model_copy(update={"provider_namespace": "x"})})
    assert catalog.resolve(consumer, [foreign])[0].reason is ResolutionReason.FOREIGN_NAMESPACE
    missing = anchors[0].model_copy(
        update={
            "source": anchors[0].source.model_copy(
                update={"extraction_view_id": "00000000-0000-4000-8000-000000000000"}
            )
        }
    )
    result = catalog.resolve(consumer, [missing])[0]
    assert (result.status, result.reason) == (ResolutionStatus.UNAVAILABLE, ResolutionReason.NOT_FOUND)
    later = json.loads(files[MANIFEST]) | {"created_at": "2026-11-09T10:00:00.000000Z"}
    catalog.add(consumer, import_bundle(dict(files) | {MANIFEST: json.dumps(later).encode()}))  # re-export: no-op
    changed = json.loads(files[MANIFEST])
    changed["omitted"].append(dict(changed["omitted"][0], kind="page_render"))
    with pytest.raises(BundleError) as caught:
        catalog.add(consumer, import_bundle(dict(files) | {MANIFEST: json.dumps(changed).encode()}))
    assert caught.value.code == "bundle_conflict"
    # another bundle that carries the same view ref with different text may not shadow the first one
    doc = imported.documents[0].model_copy(update={"canonical_sha256": "f" * 64})
    shadow = dataclasses.replace(
        imported, manifest=imported.manifest.model_copy(update={"bundle_id": "bun_shadow"}), documents=(doc,)
    )
    with pytest.raises(BundleError) as caught:
        catalog.add(consumer, shadow)
    assert caught.value.code == "bundle_conflict"


@sync
async def test_import_rejects_tampered_bundles(tmp_path):
    _owner, files, _anchors = await _bundle(tmp_path)
    text_path = next(p for p in files if p.startswith("text/"))
    assert _code(dict(files) | {text_path: files[text_path] + b"x"}) == "hash_mismatch"
    assert _code(dict(files) | {f"text/{'1' * 8}-1111-4111-8111-{'1' * 12}.txt": b"x"}) == "unlisted_file"
    assert _code({k: v for k, v in files.items() if k != STRUCTURE}) == "missing_file"
    assert _code({k: v for k, v in files.items() if k != MANIFEST}) == "missing_file"
    assert _code(dict(files) | {"../evil": b""}) == "bad_path"
    assert _code(dict(files) | {MANIFEST: b"{not json"}) == "invalid_json"
    version = json.loads(files[MANIFEST]) | {"contract_version": "2.0"}
    assert _code(dict(files) | {MANIFEST: json.dumps(version).encode()}) == "unsupported_version"
    # consistent hashes but the text was changed: the canonical hash in documents.json no longer matches
    assert _code(_relist(files, {text_path: files[text_path].replace(b"evidence", b"EVIDENCE")})) == "hash_mismatch"
    # an anchor whose quote was edited no longer resolves
    lines = files[ANCHORS].decode().splitlines()
    first = json.loads(lines[0])
    first["exact"] = first["exact"][::-1]
    edited = "\n".join([json.dumps(first, ensure_ascii=False), *lines[1:]]).encode() + b"\n"
    assert _code(_relist(files, {ANCHORS: edited})) == "dangling_anchor"
    # blocks out of order
    blocks = files[STRUCTURE].decode().splitlines()
    swapped = "\n".join([blocks[1], blocks[0], *blocks[2:]]).encode() + b"\n"
    assert _code(_relist(files, {STRUCTURE: swapped})) == "invalid_structure"
    docs = json.loads(files[DOCUMENTS])
    docs[0]["source"]["provider_namespace"] = "other"
    assert _code(_relist(files, {DOCUMENTS: json.dumps(docs).encode()})) == "invalid_structure"


@sync
async def test_import_limits(tmp_path):
    _owner, files, _anchors = await _bundle(tmp_path)
    assert _code(files, limits=BundleLimits(max_files=2)) == "too_large"
    assert _code(files, limits=BundleLimits(max_file_bytes=10)) == "too_large"
    assert _code(files, limits=BundleLimits(max_anchors=1)) == "too_large"
    write_bundle_dir(files, tmp_path / "b")
    with pytest.raises(BundleError) as caught:
        read_bundle_dir(tmp_path / "b", BundleLimits(max_total_bytes=10))
    assert caught.value.code == "too_large"


@sync
async def test_bundle_directory_rules(tmp_path):
    _owner, files, _anchors = await _bundle(tmp_path)
    root = tmp_path / "bundle"
    write_bundle_dir(files, root)
    with pytest.raises(BundleError):
        write_bundle_dir(files, root)  # not empty
    with pytest.raises(BundleError):
        write_bundle_dir({"../escape.txt": b""}, tmp_path / "other")
    (root / "extra").mkdir()
    with pytest.raises(BundleError):
        read_bundle_dir(root)
    (root / "extra").rmdir()
    (root / "notes.txt").write_bytes(b"x")
    with pytest.raises(BundleError):
        read_bundle_dir(root)
    (root / "notes.txt").unlink()
    link = root / "anchors.jsonl"
    target = tmp_path / "outside.jsonl"
    target.write_bytes(link.read_bytes())
    link.unlink()
    try:
        os.symlink(target, link)
    except (OSError, NotImplementedError):
        pytest.skip("this OS refuses symlinks")
    with pytest.raises(BundleError):
        read_bundle_dir(root)


@sync
async def test_import_rejects_crafted_json_and_forged_records(tmp_path):
    _owner, files, _anchors = await _bundle(tmp_path)
    assert _code(dict(files) | {MANIFEST: b"[" * 200_000}) == "invalid_json"
    assert _code(_relist(files, {DOCUMENTS: b"[" * 200_000})) == "invalid_json"
    assert _code(dict(files) | {1: b""}) == "bad_path"
    assert _code(dict(files) | {DOCUMENTS: "text"}) == "invalid_structure"
    raw = files[MANIFEST].decode()
    assert _code(dict(files) | {MANIFEST: ('{"bundle_id":"bun_other",' + raw[1:]).encode()}) == "invalid_json"
    blocks = [json.loads(line) for line in files[STRUCTURE].decode().splitlines()]
    blocks[1]["ordinal"] = 0
    duplicate = _jsonl_body(json.dumps(b, ensure_ascii=False) for b in blocks)
    assert _code(_relist(files, {STRUCTURE: duplicate})) == "invalid_structure"
    lines = files[ANCHORS].decode().splitlines()
    forged = json.loads(lines[0]) | {"anchor_id": "anc_" + "0" * 32}
    forged_body = _jsonl_body([json.dumps(forged, ensure_ascii=False), *lines[1:]])
    assert _code(_relist(files, {ANCHORS: forged_body})) == "dangling_anchor"
    assert _code(_relist(files, {ANCHORS: _jsonl_body([lines[0], *lines])})) == "dangling_anchor"


@sync
async def test_text_path_must_belong_to_its_view(tmp_path):
    _owner, files, _anchors = await _bundle(tmp_path)
    old = next(p for p in files if p.startswith("text/"))
    new = "text/00000000-0000-4000-8000-000000000000.txt"
    docs = json.loads(files[DOCUMENTS])
    docs[0]["text_path"] = new
    manifest = json.loads(files[MANIFEST])
    for asset in manifest["assets"]:
        if asset["path"] == old:
            asset["path"] = new
    moved = {k: v for k, v in files.items() if k != old} | {new: files[old], MANIFEST: json.dumps(manifest).encode()}
    assert _code(_relist(moved, {DOCUMENTS: json.dumps(docs).encode()})) == "invalid_structure"


def test_jsonl_splits_on_line_feed_only():
    from rag_foundry_contracts.bundle import BundleBlock

    title = "a" + chr(0x2028) + "b" + chr(0x85) + "c"
    block = {
        "extraction_view_id": "v",
        "block_id": "b00000",
        "ordinal": 0,
        "kind": "paragraph",
        "start": 0,
        "end": 1,
        "section_path": [title],
    }
    body = json.dumps(block, ensure_ascii=False).encode() + bytes([13, 10])  # CRLF line end
    assert _parse_lines(body, BundleBlock)[0].section_path == (title,)


@sync
async def test_junction_root_is_rejected(tmp_path):
    _owner, files, _anchors = await _bundle(tmp_path)
    try:
        import _winapi
    except ImportError:
        pytest.skip("junctions are Windows-only")
    target = tmp_path / "target"
    write_bundle_dir(files, target)
    junction = tmp_path / "junction"
    _winapi.CreateJunction(str(target), str(junction))
    with pytest.raises(BundleError):
        read_bundle_dir(junction)
    empty = tmp_path / "empty"
    empty.mkdir()
    _winapi.CreateJunction(str(empty), str(tmp_path / "into_empty"))
    with pytest.raises(BundleError):
        write_bundle_dir(files, tmp_path / "into_empty")
    assert not any(empty.iterdir())
