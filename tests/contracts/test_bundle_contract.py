import pytest
from pydantic import ValidationError
from rag_foundry_contracts.bundle import BundleAsset, BundleManifest
from rag_foundry_contracts.source import SourceRef

SHA = "0" * 64
VIEW = "3f6c2a8e-1b4d-4c5e-9a7f-0d2e4b6c8a1f"


def _asset(path="documents.json", role="documents"):
    return BundleAsset(path=path, role=role, media_type="application/json", size_bytes=0, sha256=SHA)


def _ref(view=VIEW):
    return SourceRef(provider_namespace="lab", document_id="d", document_version_id="v", extraction_view_id=view)


def _manifest(**changes):
    data = {
        "bundle_id": "bun_x",
        "provider_namespace": "lab",
        "created_at": "2026-11-08T10:00:00.000000Z",
        "inclusion_policy": "text_and_structure",
        "views": (_ref(),),
        "assets": (_asset(),),
        "omitted": (),
    }
    return BundleManifest(**(data | changes))


@pytest.mark.parametrize(
    "path",
    [
        "../x",
        "/abs",
        "C:\\x",
        "text/x.txt",
        "text/../documents.json",
        "Documents.json",
        "manifest.json",
        f"text/{VIEW}.txt/..",
        f"text\\{VIEW}.txt",
        f"./text/{VIEW}.txt",
        "documents.json\n",
    ],
)
def test_asset_path_rejects_paths_off_the_allowlist(path):
    with pytest.raises(ValidationError):
        _asset(path=path, role="text")


@pytest.mark.parametrize("path", ["documents.json", "structure.jsonl", "anchors.jsonl", f"text/{VIEW}.txt"])
def test_asset_path_accepts_the_allowlist(path):
    assert _asset(path=path).path == path


def test_manifest_accepts_1x_and_rejects_other_majors():
    assert _manifest(contract_version="1.7").contract_version == "1.7"
    for version in ("2.0", "0.9", "1", "v1.0"):
        with pytest.raises((ValidationError, ValueError)):
            _manifest(contract_version=version)


def test_manifest_rejects_duplicate_assets_and_views():
    with pytest.raises(ValidationError):
        _manifest(assets=(_asset(), _asset()))
    with pytest.raises(ValidationError):
        _manifest(views=(_ref(), _ref()))


def test_manifest_requires_utc_timestamp_and_has_no_scope_fields():
    with pytest.raises(ValidationError):
        _manifest(created_at="2026-11-08T10:00:00+02:00")
    with pytest.raises(ValidationError):
        _manifest(tenant_id=VIEW)
    fields = set(BundleManifest.model_fields)
    assert not fields & {"tenant_id", "project_id", "chunk_id", "index_id"}
