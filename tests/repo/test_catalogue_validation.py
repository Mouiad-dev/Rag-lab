"""verify_package.py must reject broken catalogue entries.

Runs the validator against a temporary copy of only the files it reads, so the real
config/ and docs/ are never written. The local `.env` and other non-source files are
never copied, and no file contents are printed.
"""

import copy
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
# verify_package.py reads these (plus link targets such as LICENSE and flow HTML); skip everything else.
# Keep in step with the globs in verify_package.py; a missing type fails test_unmodified_copy_passes loudly.
COPY_SUFFIXES = {".py", ".json", ".md", ".yml", ".html"}
SKIP_DIRS = {"node_modules", "private", "__pycache__", "dist", "build", "data", "volumes"}
KEEP_HIDDEN_DIRS = {".claude", ".github"}


def _copy_validator_inputs(dst: Path) -> None:
    for current, dirs, names in os.walk(ROOT):
        dirs[:] = [
            d
            for d in dirs
            if d not in SKIP_DIRS and not d.endswith(".egg-info") and (not d.startswith(".") or d in KEEP_HIDDEN_DIRS)
        ]
        for name in names:
            if name.startswith(".env") or name.endswith(".local.json"):
                continue
            if Path(name).suffix in COPY_SUFFIXES or name in {".gitignore", "LICENSE"}:
                src = Path(current) / name
                target = dst / src.relative_to(ROOT)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, target)


@pytest.fixture(scope="module")
def repo_copy(tmp_path_factory: pytest.TempPathFactory) -> Path:
    dst = tmp_path_factory.mktemp("repo_copy")
    _copy_validator_inputs(dst)
    assert not (dst / ".env").exists()
    return dst


@pytest.fixture
def work(repo_copy: Path):
    """Yield the copy; restore the two mutable files afterwards."""
    files = [repo_copy / "config" / "techniques.json", repo_copy / "docs" / "techniques" / "TECHNIQUE_CATALOG.md"]
    saved = {f: f.read_bytes() for f in files}
    yield repo_copy
    for f, data in saved.items():
        f.write_bytes(data)


def _run(tmp: Path) -> subprocess.CompletedProcess:
    script = tmp / "scripts" / "verify_package.py"
    return subprocess.run(
        [sys.executable, str(script)], cwd=tmp, capture_output=True, text=True, timeout=120, check=False
    )


def _registry(tmp: Path) -> dict:
    return json.loads((tmp / "config" / "techniques.json").read_text(encoding="utf-8"))


def _write_registry(tmp: Path, registry: dict) -> None:
    (tmp / "config" / "techniques.json").write_text(json.dumps(registry, indent=2), encoding="utf-8")


def _entry(registry: dict, tid: str) -> dict:
    return next(e for e in registry["techniques"] if e["id"] == tid)


def _set(tid: str, **fields):
    def mutate(tmp: Path) -> None:
        registry = _registry(tmp)
        _entry(registry, tid).update(fields)
        _write_registry(tmp, registry)

    return mutate


def _delete(tid: str):
    def mutate(tmp: Path) -> None:
        registry = _registry(tmp)
        registry["techniques"] = [e for e in registry["techniques"] if e["id"] != tid]
        _write_registry(tmp, registry)

    return mutate


def _duplicate(tid: str):
    def mutate(tmp: Path) -> None:
        registry = _registry(tmp)
        registry["techniques"].append(copy.deepcopy(_entry(registry, tid)))
        _write_registry(tmp, registry)

    return mutate


def _phase_mismatch(tid: str):
    def mutate(tmp: Path) -> None:
        registry = _registry(tmp)
        entry = _entry(registry, tid)
        entry["phase"] = entry["phase"] + 1
        _write_registry(tmp, registry)

    return mutate


def _drop_catalogue_row(tid: str):
    def mutate(tmp: Path) -> None:
        path = tmp / "docs" / "techniques" / "TECHNIQUE_CATALOG.md"
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        kept = [line for line in lines if not line.startswith(f"| {tid} |")]
        assert len(kept) == len(lines) - 1
        path.write_text("".join(kept), encoding="utf-8")

    return mutate


def _invalid_json(tmp: Path) -> None:
    (tmp / "config" / "techniques.json").write_text("{not json", encoding="utf-8")


def test_unmodified_copy_passes(work: Path) -> None:
    result = _run(work)
    assert result.returncode == 0, result.stderr
    assert "72 catalogue entries" in result.stdout


UNIQUE_IDS = "Technique IDs must be unique and include the committed 72-entry curriculum"

BROKEN_CASES = [
    pytest.param(_set("C05", status="done-ish"), "Invalid status", id="bad-status"),
    pytest.param(
        _set("C05", catalog_status="proposed"),
        "Proposed entries belong in the app database",
        id="proposed-entry",
    ),
    pytest.param(_delete("C05"), UNIQUE_IDS, id="deleted-entry"),
    pytest.param(_duplicate("C05"), UNIQUE_IDS, id="duplicate-id"),
    pytest.param(_set("R10", roadmap_step="P99.9"), "Unknown roadmap_step", id="unknown-step"),
    pytest.param(_phase_mismatch("R10"), "phase does not match roadmap_step", id="phase-mismatch"),
    pytest.param(
        _set("R10", primary_source="http://example.com"),
        "primary_source must be null or an https URL",
        id="non-https-source",
    ),
    pytest.param(_set("R10", date_added="2026/01/01"), "date_added must be YYYY-MM-DD", id="bad-date"),
    pytest.param(_drop_catalogue_row("R10"), "Missing or out-of-sync catalogue row", id="missing-row"),
    pytest.param(_invalid_json, "Invalid JSON: config", id="invalid-json"),
]


@pytest.mark.parametrize(("mutate", "expected"), BROKEN_CASES)
def test_broken_catalogue_fails(work: Path, mutate, expected: str) -> None:
    mutate(work)
    result = _run(work)
    assert result.returncode == 1, result.stderr
    assert expected in result.stderr
