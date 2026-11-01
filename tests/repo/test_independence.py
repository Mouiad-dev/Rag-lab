"""Static guards for ADR-0008: RAG Foundry must not depend on a consumer repository.

Stdlib only. Checks workspace manifests, the lockfile and Python sources for references
to book2video and for path sources that escape the repository root.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FORBIDDEN_NAME = "book2video"

# A TOML `path = "..."` key, not e.g. `file_path = "..."`.
PATH_SOURCE = re.compile(r'(?<![\w-])path\s*=\s*"([^"]*)"')


def manifest_files() -> list[Path]:
    files = [REPO_ROOT / "pyproject.toml"]
    files += sorted(REPO_ROOT.glob("packages/*/pyproject.toml"))
    files += sorted(REPO_ROOT.glob("apps/*/pyproject.toml"))
    return files


def python_sources() -> list[Path]:
    found: list[Path] = []
    for top in ("packages", "apps"):
        for path in sorted((REPO_ROOT / top).rglob("*.py")):
            if {".venv", "__pycache__"} & set(path.parts):
                continue
            found.append(path)
    return found


def escaping_path_sources(text: str, base_dir: Path, root: Path = REPO_ROOT) -> list[str]:
    """Return `path = "..."` values in `text` that resolve outside `root`.

    Relative paths are resolved against `base_dir` (the directory of the manifest).
    """
    resolved_root = root.resolve()
    escapes: list[str] = []
    for match in PATH_SOURCE.finditer(text):
        target = (base_dir / match.group(1)).resolve()
        if not target.is_relative_to(resolved_root):
            escapes.append(match.group(1))
    return escapes


def imports_forbidden_module(source: str) -> bool:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names = [node.module]
        for name in names:
            if any(part.lower() == FORBIDDEN_NAME for part in name.split(".")):
                return True
    return False


def test_manifests_and_lock_do_not_name_book2video() -> None:
    files = [*manifest_files(), REPO_ROOT / "uv.lock"]
    assert (REPO_ROOT / "uv.lock").is_file()
    offenders = [
        str(path.relative_to(REPO_ROOT)) for path in files if FORBIDDEN_NAME in path.read_text(encoding="utf-8").lower()
    ]
    assert offenders == [], f"consumer name found in: {offenders}"


def test_path_sources_stay_inside_repo() -> None:
    offenders: list[str] = []
    for manifest in manifest_files():
        text = manifest.read_text(encoding="utf-8")
        for value in escaping_path_sources(text, manifest.parent):
            offenders.append(f"{manifest.relative_to(REPO_ROOT)}: {value}")
    assert offenders == [], f"path sources escape the repo root: {offenders}"


def test_python_sources_do_not_import_book2video() -> None:
    offenders: list[str] = []
    for path in python_sources():
        if imports_forbidden_module(path.read_text(encoding="utf-8")):
            offenders.append(str(path.relative_to(REPO_ROOT)))
    assert offenders == [], f"consumer import found in: {offenders}"


def test_path_escape_helper_flags_synthetic_escape() -> None:
    # Negative self-test: the helper must catch a real escape and accept an in-repo path.
    synthetic = 'rag-foundry-consumer = { path = "../book2video" }\n'
    assert escaping_path_sources(synthetic, REPO_ROOT) == ["../book2video"]
    in_repo = 'rag-foundry-core = { path = "packages/core" }\n'
    assert escaping_path_sources(in_repo, REPO_ROOT) == []


def test_import_detector_flags_consumer_imports() -> None:
    # Negative self-test: a broken detector must not pass silently.
    assert imports_forbidden_module("import book2video.x")
    assert imports_forbidden_module("from book2video import y")
    assert not imports_forbidden_module("import os\nfrom pathlib import Path")
