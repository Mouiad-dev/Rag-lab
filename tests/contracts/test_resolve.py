"""Anchor resolve v1 contract tests."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from rag_foundry_contracts import (
    AnchorResolution,
    AnchorResolveRequest,
    AnchorResolveResponse,
    ResolutionReason,
    ResolutionStatus,
)
from rag_foundry_contracts.export import render, schemas

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_DIR = ROOT / "packages" / "contracts" / "schemas"
FIXTURES = ROOT / "packages" / "contracts" / "fixtures" / "resolve"


def _load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_bytes().decode("utf-8"))


def _anchor(anchor_id: str = "a1") -> dict:
    return {
        "anchor_id": anchor_id,
        "source": {
            "provider_namespace": "lab",
            "document_id": "d1",
            "document_version_id": "dv1",
            "extraction_view_id": "ev1",
        },
        "block_id": "b1",
        "precision": "exact_text",
        "text_range": {"start": 0, "end": 1},
    }


def _exact(**over) -> dict:
    data = {"anchor_id": "a1", "status": "exact", "exact": "x", "block_id": "b1"}
    data.update(over)
    return data


def _failed(**over) -> dict:
    data = {"anchor_id": "a1", "status": "mismatch", "reason": "quote_mismatch"}
    data.update(over)
    return data


def test_fixtures_validate_and_correspond():
    request = AnchorResolveRequest.model_validate(_load("request.json"))
    response = AnchorResolveResponse.model_validate(_load("response.json"))
    assert [a.anchor_id for a in request.anchors] == [r.anchor_id for r in response.results]
    assert [r.status for r in response.results] == [ResolutionStatus.EXACT, ResolutionStatus.MISMATCH]
    assert response.results[1].reason is ResolutionReason.QUOTE_MISMATCH
    assert {a.source.document_id for a in request.anchors} == {"doc-en", "doc-ar"}


@pytest.mark.parametrize("name", ["anchor-resolve-request.v1.json", "anchor-resolve-response.v1.json"])
def test_committed_schema_matches_models(name):
    committed = (SCHEMA_DIR / name).read_bytes().decode("utf-8")
    assert committed == render(schemas()[name]), f"{name} drifted; run the export module"


def test_request_bounds():
    AnchorResolveRequest(anchors=[_anchor("a1")])
    AnchorResolveRequest(anchors=[_anchor(f"a{i}") for i in range(100)])
    for count in (0, 101):
        with pytest.raises(ValidationError):
            AnchorResolveRequest(anchors=[_anchor(f"a{i}") for i in range(count)])


def test_request_rejects_overlong_anchor_id():
    AnchorResolveRequest(anchors=[_anchor("a" * 256)])
    with pytest.raises(ValidationError, match="at most 256"):
        AnchorResolveRequest(anchors=[_anchor("a" * 257)])


@pytest.mark.parametrize("model", [AnchorResolveRequest, AnchorResolveResponse])
def test_rejects_major_version_2(model):
    payload = {"anchors": [_anchor()]} if model is AnchorResolveRequest else {"results": []}
    with pytest.raises(ValidationError, match="unsupported contract major version 2"):
        model(contract_version="2.0", **payload)
    model(contract_version="1.3", **payload)


def test_resolution_rejects_major_version_2():
    with pytest.raises(ValidationError):
        AnchorResolution(contract_version="2.0", **_exact())


def test_valid_resolutions():
    ok = AnchorResolution(**_exact(section_path=["A"], page_index=0))
    assert ok.status is ResolutionStatus.EXACT and ok.reason is None
    for status in ("mismatch", "unavailable"):
        for reason in ResolutionReason:
            AnchorResolution(**_failed(status=status, reason=reason.value))


@pytest.mark.parametrize(
    "data",
    [
        _exact(reason="not_found"),  # exact with a reason
        _failed(reason=None),  # failure without a reason
        _exact(exact=None),
        _exact(block_id=None),
        _failed(exact="x"),
        _failed(block_id="b1"),
        _failed(section_path=["A"]),
        _failed(page_index=0),
        _failed(status="unavailable", page_index=0),
        _exact(page_index=-1),
    ],
)
def test_rejects_inconsistent_resolution(data):
    with pytest.raises(ValidationError):
        AnchorResolution(**data)


def test_resolution_anchor_id_bounds():
    AnchorResolution(**_exact(anchor_id="a" * 256))
    for bad in ("", "a" * 257):
        with pytest.raises(ValidationError):
            AnchorResolution(**_exact(anchor_id=bad))


def test_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        AnchorResolution(**_exact(extra_field=1))
    with pytest.raises(ValidationError):
        AnchorResolveRequest(anchors=[_anchor()], tenant_id="t1")
    with pytest.raises(ValidationError):
        AnchorResolveResponse(results=[], chunk_id="c1")
