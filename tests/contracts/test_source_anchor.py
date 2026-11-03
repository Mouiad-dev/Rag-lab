"""Source anchor v1 contract tests."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from rag_foundry_contracts import (
    PageRegion,
    Precision,
    SourceAnchor,
    SourceRef,
    TextRange,
    UnsupportedContractVersionError,
    check_contract_version,
)
from rag_foundry_contracts.export import render, schemas

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_DIR = ROOT / "packages" / "contracts" / "schemas"

SOURCE = {
    "provider_namespace": "lab",
    "document_id": "d1",
    "document_version_id": "dv1",
    "extraction_view_id": "ev1",
}


def _anchor(**over):
    data = {
        "anchor_id": "a1",
        "source": SOURCE,
        "block_id": "b1",
        "precision": "exact_text",
        "text_range": {"start": 0, "end": 1},
    }
    data.update(over)
    return data


@pytest.mark.parametrize(
    "over",
    [
        {"text_range": None},
        {"precision": "exact_region"},
        {"precision": "approximate_region"},
        {"precision": "page_only"},
        {"page_index": 1, "regions": [{"page_index": 0, "x0": 0.1, "y0": 0.1, "x1": 0.2, "y1": 0.2}]},
        {"regions": [{"page_index": 0, "x0": 0.1, "y0": 0.1, "x1": 0.2, "y1": 0.2}]},
    ],
)
def test_rejects_precision_not_backed_by_data(over):
    with pytest.raises(ValidationError):
        SourceAnchor(**_anchor(**over))


def test_accepts_precision_backed_by_data():
    region = {"page_index": 2, "x0": 0.1, "y0": 0.1, "x1": 0.2, "y1": 0.2}
    SourceAnchor(**_anchor(precision="exact_region", page_index=2, regions=[region]))
    SourceAnchor(**_anchor(precision="page_only", text_range=None, page_index=2))
    SourceAnchor(**_anchor(precision="unavailable", text_range=None))


def test_valid_anchor_round_trips():
    anchor = SourceAnchor(
        **_anchor(
            section_path=["Intro", "Arabic"],
            text_range={"start": 2, "end": 7},
            exact="مرحبا",
            prefix="ab",
            page_index=0,
            page_label="iv",
            regions=[{"page_index": 0, "x0": 0.1, "y0": 0.2, "x1": 0.5, "y1": 0.3}],
            warnings=["ocr"],
        )
    )
    again = SourceAnchor.model_validate_json(anchor.model_dump_json())
    assert again == anchor
    assert again.precision is Precision.EXACT_TEXT
    assert again.contract_version == "1.0"


def test_rejects_end_before_start():
    with pytest.raises(ValidationError):
        TextRange(start=5, end=4)


def test_rejects_negative_start():
    with pytest.raises(ValidationError):
        TextRange(start=-1, end=3)


@pytest.mark.parametrize("field", ["x0", "y0", "x1", "y1"])
@pytest.mark.parametrize("value", [-0.01, 1.01])
def test_rejects_region_coords_outside_unit_range(field, value):
    coords = {"x0": 0.1, "y0": 0.1, "x1": 0.9, "y1": 0.9}
    coords[field] = value
    with pytest.raises(ValidationError):
        PageRegion(page_index=0, **coords)


def test_rejects_unordered_region():
    with pytest.raises(ValidationError):
        PageRegion(page_index=0, x0=0.6, y0=0.1, x1=0.5, y1=0.9)


def test_rejects_exact_length_mismatch():
    with pytest.raises(ValidationError):
        SourceAnchor(**_anchor(text_range={"start": 0, "end": 3}, exact="abcd"))


def test_exact_requires_text_range():
    with pytest.raises(ValidationError):
        SourceAnchor(**_anchor(exact="abc"))


def test_rejects_unknown_field():
    with pytest.raises(ValidationError):
        SourceAnchor(**_anchor(chunk_id="c1"))
    with pytest.raises(ValidationError):
        SourceRef(**SOURCE, extra_field=1)


def test_rejects_empty_ids():
    with pytest.raises(ValidationError):
        SourceAnchor(**_anchor(anchor_id=""))


def test_rejects_unsupported_major_version():
    with pytest.raises(UnsupportedContractVersionError):
        check_contract_version("2.0")
    # pydantic wraps ValueError subclasses raised in validators into ValidationError.
    with pytest.raises(ValidationError, match="unsupported contract major version 2"):
        SourceAnchor(**_anchor(contract_version="2.0"))


def test_version_check_accepts_1x_and_rejects_malformed():
    assert check_contract_version("1.7") == "1.7"
    for bad in ["", "1", "v1.0", "1.0.0", "0.9", "abc"]:
        with pytest.raises(UnsupportedContractVersionError):
            check_contract_version(bad)


def test_committed_schema_matches_models():
    for name, schema in schemas().items():
        committed = (SCHEMA_DIR / name).read_bytes().decode("utf-8")
        assert committed == render(schema), f"{name} drifted; run the export module"


FORBIDDEN_KEYS = {"chunk_id", "index_id", "course", "lesson", "video", "tts", "start_ms"}


def _keys(node):
    if isinstance(node, dict):
        for k, v in node.items():
            yield k
            yield from _keys(v)
    elif isinstance(node, list):
        for item in node:
            yield from _keys(item)


def test_schema_has_no_retrieval_or_consumer_keys():
    for name in schemas():
        schema = json.loads((SCHEMA_DIR / name).read_text(encoding="utf-8"))
        assert FORBIDDEN_KEYS.isdisjoint(_keys(schema))
