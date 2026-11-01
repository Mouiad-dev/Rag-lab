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
