import hashlib
import importlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from rag_foundry_techniques import TechniqueUnavailableError, check_plugin_contract, load_catalogue

ROOT = Path(__file__).resolve().parents[2]
CATALOGUE = ROOT / "config/techniques.json"
SCRIPT = ROOT / "scripts/new_technique.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("new_technique_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


script = _load_script()


def _sha() -> str:
    return hashlib.sha256(CATALOGUE.read_bytes()).hexdigest()


_SHA_AT_IMPORT = _sha()


def _free_id(kind, preferred=None):
    """First catalogue ID that allows this kind, is not built-in and is not implemented/validated."""
    catalogue = load_catalogue(CATALOGUE)
    builtin = script._builtin_ids()
    entries = {e["id"]: e for e in json.loads(CATALOGUE.read_text(encoding="utf-8"))["techniques"]}
    ids = [preferred] if preferred else sorted(entries)
    for tid in ids:
        if (
            kind in catalogue.kinds_for(tid)
            and tid not in builtin
            and entries[tid]["status"]
            not in (
                "implemented",
                "validated",
            )
        ):
            return tid
    return None


def _case(kind, preferred=None):
    tid = _free_id(kind, preferred)
    if tid is None:
        reason = f"no unbuilt catalogue ID allows kind '{kind}'; generator case skipped"
        return pytest.param("", kind, "", marks=pytest.mark.skip(reason=reason)), None
    return (tid, kind, f"rag_foundry_plugin_{tid.lower()}"), (tid, kind)


_PARAMS_AND_REAL = [_case("chunker", "C11"), _case("retriever")]
PARAMS = [p for p, _ in _PARAMS_AND_REAL]
CASES = [r for _, r in _PARAMS_AND_REAL if r is not None]


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    before = _sha()
    result = {}
    for tid, kind in CASES:
        out = tmp_path_factory.mktemp(tid) / "plugin"
        assert script.main([tid, "--kind", kind, "--fidelity", "simulation", "--out", str(out)]) == 0
        result[tid] = out
    assert _sha() == before
    return result


@pytest.mark.parametrize(("tid", "kind", "pkg"), PARAMS)
def test_files_exist_and_are_ascii(generated, tid, kind, pkg):
    out = generated[tid]
    files = ["pyproject.toml", f"src/{pkg}/__init__.py", "tests/test_contract.py", "README.md"]
    for rel in files:
        path = out / rel
        assert path.is_file(), rel
        path.read_bytes().decode("ascii")
    assert f'{tid} = "{pkg}:Plugin"' in (out / "pyproject.toml").read_text()
    assert "docs/templates/TECHNIQUE.md" in (out / "README.md").read_text()
    compile((out / "tests/test_contract.py").read_text(), "test_contract.py", "exec")


def test_catalogue_unchanged_after_generation(generated):
    assert _sha() == _SHA_AT_IMPORT


@pytest.mark.parametrize(("tid", "kind", "pkg"), PARAMS)
def test_generated_module_passes_contract(generated, monkeypatch, tid, kind, pkg):
    monkeypatch.syspath_prepend(str(generated[tid] / "src"))
    monkeypatch.delitem(sys.modules, pkg, raising=False)
    module = importlib.import_module(pkg)
    manifest = check_plugin_contract(module.Plugin, load_catalogue(CATALOGUE), sample_config={})
    assert manifest.technique_id == tid
    assert manifest.kind.value == kind
    if kind == "chunker":
        with pytest.raises(TechniqueUnavailableError):
            module.Plugin(module.Config()).chunk(None)
    else:
        assert not hasattr(module.Plugin, "chunk")
    monkeypatch.delitem(sys.modules, pkg, raising=False)


@pytest.mark.parametrize(("tid", "kind", "pkg"), PARAMS)
def test_generated_test_passes_in_subprocess(generated, tid, kind, pkg):
    out = generated[tid]
    env = {**os.environ, "PYTHONPATH": str(out / "src"), "RAG_FOUNDRY_CATALOGUE": str(CATALOGUE)}
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "tests/test_contract.py"],
        cwd=out,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def _refuse(tmp_path, capsys, tid, *extra, out=None, kind="chunker"):
    args = [tid, "--fidelity", "simulation", "--out", str(out or tmp_path / "o")]
    if kind:
        args += ["--kind", kind]
    code = script.main(args + list(extra))
    err = capsys.readouterr().err
    return code, err


@pytest.mark.parametrize("tid", ["X99", "c05", "R02", "C05"])
def test_refusals_by_id(tmp_path, capsys, tid):
    before = _sha()
    code, err = _refuse(tmp_path, capsys, tid)
    assert code == 2
    assert len(err.strip().splitlines()) == 1
    assert not (tmp_path / "o").exists()
    assert _sha() == before


def test_builtin_id_refused_with_planned_status(tmp_path, capsys):
    # Every built-in is `implemented` in the real catalogue, so the status check fires first.
    # A copy with C05 set to `planned` reaches the built-in check (built-ins still load from the repo).
    data = json.loads(CATALOGUE.read_text(encoding="utf-8"))
    (entry,) = [e for e in data["techniques"] if e["id"] == "C05"]
    entry["status"] = "planned"
    copy = tmp_path / "techniques.json"
    copy.write_text(json.dumps(data), encoding="utf-8")
    code, err = _refuse(tmp_path, capsys, "C05", "--catalogue", str(copy))
    assert code == 2
    assert len(err.strip().splitlines()) == 1
    assert "DuplicatePluginError" in err
    assert not (tmp_path / "o").exists()


def test_builtin_ids_include_implemented_chunkers():
    assert "C05" in script._builtin_ids()


def test_wrong_kind_refused(tmp_path, capsys):
    code, err = _refuse(tmp_path, capsys, "C11", kind="graph_builder")
    assert code == 2
    assert len(err.strip().splitlines()) == 1
    assert "not allowed" in err


def test_non_empty_out_refused(tmp_path, capsys):
    out = tmp_path / "busy"
    out.mkdir()
    (out / "keep.txt").write_text("x")
    code, err = _refuse(tmp_path, capsys, "C11", out=out)
    assert code == 2
    assert len(err.strip().splitlines()) == 1
    assert [p.name for p in out.iterdir()] == ["keep.txt"]


@pytest.mark.parametrize("pkg", ["Bad-Name", "ab", "1abc"])
def test_bad_package_refused(tmp_path, capsys, pkg):
    code, err = _refuse(tmp_path, capsys, "C11", "--package", pkg)
    assert code == 2
    assert len(err.strip().splitlines()) == 1
    assert not (tmp_path / "o").exists()
