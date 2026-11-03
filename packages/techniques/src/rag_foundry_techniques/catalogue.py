"""Read-only loader for config/techniques.json (schema v2)."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, ValidationError

from rag_foundry_techniques.errors import CatalogueError, UnknownTechniqueError

SUPPORTED_SCHEMA_VERSION = 2


class TechniqueEntry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    id: str
    name: str
    category: str
    family: str
    phase: int
    roadmap_step: str
    legacy_milestone: str
    status: str
    catalog_status: str
    mechanism: str
    acceptance: str
    requirements_note: str
    primary_source: str | None
    date_added: str
    implementation_path: str | None
    evidence_path: str | None
    production_eligible: bool
    design_reference: str | None = None


class Catalogue(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: int
    entries: tuple[TechniqueEntry, ...]

    def ids(self) -> tuple[str, ...]:
        return tuple(entry.id for entry in self.entries)

    def get(self, tid: str) -> TechniqueEntry:
        for entry in self.entries:
            if entry.id == tid:
                return entry
        raise UnknownTechniqueError(tid)

    def __contains__(self, tid: object) -> bool:
        return any(entry.id == tid for entry in self.entries)

    def __len__(self) -> int:
        return len(self.entries)


def load_catalogue(path: Path) -> Catalogue:
    """Load the catalogue. Opens the file for reading only."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CatalogueError(f"cannot read catalogue {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise CatalogueError("catalogue root must be a JSON object")
    version = data.get("schema_version")
    if type(version) is not int or version != SUPPORTED_SCHEMA_VERSION:
        raise CatalogueError(f"unsupported schema_version {version!r}; expected {SUPPORTED_SCHEMA_VERSION}")
    raw_entries = data.get("techniques")
    if not isinstance(raw_entries, list):
        raise CatalogueError("catalogue 'techniques' must be a list")
    allowed = {
        "status": data.get("allowed_statuses"),
        "catalog_status": data.get("allowed_catalog_statuses"),
        "category": data.get("allowed_categories"),
    }
    for name, values in allowed.items():
        if not isinstance(values, list):
            raise CatalogueError(f"catalogue is missing the allowed list for '{name}'")

    entries: list[TechniqueEntry] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_entries):
        label = raw.get("id") if isinstance(raw, dict) else None
        label = label if isinstance(label, str) else f"#{index}"
        try:
            entry = TechniqueEntry.model_validate(raw)
        except ValidationError as exc:
            raise CatalogueError(f"invalid catalogue entry {label}: {exc}") from exc
        for field, values in allowed.items():
            if getattr(entry, field) not in values:
                raise CatalogueError(f"entry {entry.id}: {field} '{getattr(entry, field)}' is not allowed")
        if entry.catalog_status == "proposed":
            # Proposed entries live in the app database until a human approves them.
            raise CatalogueError(f"entry {entry.id}: proposed entries do not belong in the repo catalogue")
        if entry.id in seen:
            raise CatalogueError(f"duplicate technique ID {entry.id}")
        seen.add(entry.id)
        entries.append(entry)
    return Catalogue(schema_version=version, entries=tuple(entries))
