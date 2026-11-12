import hashlib
import json
from pathlib import Path

import pytest
from rag_foundry_techniques import CatalogueError, UnknownTechniqueError, load_catalogue

ROOT = Path(__file__).resolve().parents[2]
CATALOGUE = ROOT / "config" / "techniques.json"


def _copy(tmp_path, mutate):
    data = json.loads(CATALOGUE.read_text(encoding="utf-8"))
    mutate(data)
    path = tmp_path / "techniques.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_real_catalogue_loads_and_is_not_modified():
    before = hashlib.sha256(CATALOGUE.read_bytes()).hexdigest()
    cat = load_catalogue(CATALOGUE)
    assert len(cat) == 72
    expected = {f"C{i:02}" for i in range(1, 22)} | {f"R{i:02}" for i in range(1, 52)}
    assert set(cat.ids()) == expected
    assert "C05" in cat and "X99" not in cat
    assert hashlib.sha256(CATALOGUE.read_bytes()).hexdigest() == before


def test_get_known_id():
    assert load_catalogue(CATALOGUE).get("C05").id == "C05"


@pytest.mark.parametrize("tid", ["X99", "c05"])
def test_get_unknown_id_raises(tid):
    with pytest.raises(UnknownTechniqueError, match=tid) as info:
        load_catalogue(CATALOGUE).get(tid)
    assert not isinstance(info.value, KeyError | LookupError)
    assert info.value.technique_id == tid


def test_schema_version_rejected(tmp_path):
    path = _copy(tmp_path, lambda d: d.update(schema_version=3))
    with pytest.raises(CatalogueError):
        load_catalogue(path)


@pytest.mark.parametrize("bad", [2.0, "2", True])
def test_schema_version_must_be_int(tmp_path, bad):
    path = _copy(tmp_path, lambda d: d.update(schema_version=bad))
    with pytest.raises(CatalogueError, match="schema_version"):
        load_catalogue(path)


def test_proposed_entry_rejected(tmp_path):
    def mutate(d):
        d["allowed_catalog_statuses"] = sorted({*d["allowed_catalog_statuses"], "proposed"})
        d["techniques"][0]["catalog_status"] = "proposed"

    path = _copy(tmp_path, mutate)
    with pytest.raises(CatalogueError, match="C01"):
        load_catalogue(path)


def test_unknown_entry_field_rejected(tmp_path):
    path = _copy(tmp_path, lambda d: d["techniques"][0].update(surprise=1))
    with pytest.raises(CatalogueError, match="C01"):
        load_catalogue(path)


def test_duplicate_id_rejected(tmp_path):
    path = _copy(tmp_path, lambda d: d["techniques"].append(dict(d["techniques"][0])))
    with pytest.raises(CatalogueError, match="duplicate"):
        load_catalogue(path)


def test_bad_status_rejected(tmp_path):
    path = _copy(tmp_path, lambda d: d["techniques"][0].update(status="bogus"))
    with pytest.raises(CatalogueError, match="C01"):
        load_catalogue(path)


def test_invalid_json_rejected(tmp_path):
    path = tmp_path / "techniques.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(CatalogueError):
        load_catalogue(path)


def test_round_trip_matches_committed_file():
    raw = json.loads(CATALOGUE.read_text(encoding="utf-8"))
    cat = load_catalogue(CATALOGUE)
    assert [e.model_dump(exclude_unset=True) for e in cat.entries] == raw["techniques"]


def test_kinds_for():
    cat = load_catalogue(CATALOGUE)
    assert cat.kinds_for("C01") == ("chunker",)
    with pytest.raises(UnknownTechniqueError):
        cat.kinds_for("X99")


def _drop_map(d):
    del d["allowed_plugin_kinds"]


def _not_object(d):
    d["allowed_plugin_kinds"] = []


def _missing_category(d):
    del d["allowed_plugin_kinds"]["cache"]


def _extra_category(d):
    d["allowed_plugin_kinds"]["bogus"] = ["cache"]


def _empty_list(d):
    d["allowed_plugin_kinds"]["cache"] = []


def _unknown_kind(d):
    d["allowed_plugin_kinds"]["cache"] = ["nonsense"]


@pytest.mark.parametrize(
    "mutate", [_drop_map, _not_object, _missing_category, _extra_category, _empty_list, _unknown_kind]
)
def test_bad_kind_map_rejected(tmp_path, mutate):
    with pytest.raises(CatalogueError, match="allowed_plugin_kinds"):
        load_catalogue(_copy(tmp_path, mutate))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("id", "C1"),
        ("id", "C01\n"),
        ("roadmap_step", "P1"),
        ("roadmap_step", "P1.2\n"),
        ("phase", 9),
        ("date_added", "2026-13-40"),
        ("date_added", "2026-1-01"),
        ("date_added", "2026-01-01\n"),
        ("primary_source", "http://example.org"),
        ("status", "validated"),
    ],
)
def test_bad_entry_field_rejected(tmp_path, field, value):
    def mutate(d):
        entry = d["techniques"][0]
        entry[field] = value
        if field == "status":
            entry["evidence_path"] = None

    with pytest.raises(CatalogueError, match=r"C01|C1"):
        load_catalogue(_copy(tmp_path, mutate))
