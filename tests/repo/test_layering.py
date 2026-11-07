"""Enforce package import boundaries using AST only (never import project modules)."""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    result = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            result.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            result.add(node.module)
    return result


def test_api_storage_import_is_confined_to_container():
    root = ROOT / "apps/api/src/rag_foundry_api"
    for path in root.rglob("*.py"):
        imports = _imports(path)
        if any(name == "rag_foundry_storage" or name.startswith("rag_foundry_storage.") for name in imports):
            assert path.name == "container.py", path


def test_api_does_not_import_sqlalchemy():
    root = ROOT / "apps/api/src/rag_foundry_api"
    for path in root.rglob("*.py"):
        assert not any(name == "sqlalchemy" or name.startswith("sqlalchemy.") for name in _imports(path)), path


def test_only_storage_package_imports_sqlalchemy_or_storage():
    for path in (ROOT / "packages").glob("*/src/**/*.py"):
        package = path.parts[path.parts.index("packages") + 1]
        if package == "storage":
            continue
        for name in _imports(path):
            assert not (name == "sqlalchemy" or name.startswith("sqlalchemy.")), path
            assert not (name == "rag_foundry_storage" or name.startswith("rag_foundry_storage.")), path
