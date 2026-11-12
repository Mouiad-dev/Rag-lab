"""Read-only loader for config/techniques.json (schema v2)."""

from __future__ import annotations

import datetime
import json
import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict, ValidationError

from rag_foundry_techniques.errors import CatalogueError, UnknownTechniqueError
from rag_foundry_techniques.protocol import PluginKind

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
    allowed_plugin_kinds: tuple[tuple[str, tuple[str, ...]], ...]

    def kinds_for(self, tid: str) -> tuple[str, ...]:
        """Plugin kinds allowed for the entry's category. Raises UnknownTechniqueError."""
        category = self.get(tid).category
        for name, kinds in self.allowed_plugin_kinds:
            if name == category:
                return kinds
        raise UnknownTechniqueError(tid)  # unreachable: load_catalogue checks every category

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

    allowed_kinds = _parse_kind_map(data.get("allowed_plugin_kinds"), allowed["category"])

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
        _check_entry_rules(entry)
        if entry.id in seen:
            raise CatalogueError(f"duplicate technique ID {entry.id}")
        seen.add(entry.id)
        entries.append(entry)
    return Catalogue(schema_version=version, entries=tuple(entries), allowed_plugin_kinds=allowed_kinds)


def _parse_kind_map(raw: object, categories: list) -> tuple[tuple[str, tuple[str, ...]], ...]:
    if not isinstance(raw, dict):
        raise CatalogueError("catalogue 'allowed_plugin_kinds' must be an object")
    missing = sorted(str(c) for c in categories if c not in raw)
    extra = sorted(str(k) for k in raw if k not in categories)
    if missing or extra:
        raise CatalogueError(
            f"allowed_plugin_kinds must list exactly the allowed categories; missing {missing}, extra {extra}"
        )
    valid = {kind.value for kind in PluginKind}
    pairs: list[tuple[str, tuple[str, ...]]] = []
    for category in sorted(raw):
        kinds = raw[category]
        if not isinstance(kinds, list) or not kinds:
            raise CatalogueError(f"allowed_plugin_kinds['{category}'] must be a non-empty list")
        bad = [k for k in kinds if not isinstance(k, str) or k not in valid]
        if bad:
            raise CatalogueError(f"allowed_plugin_kinds['{category}'] names unknown plugin kinds {bad!r}")
        pairs.append((category, tuple(sorted(set(kinds)))))
    return tuple(pairs)


def _check_entry_rules(entry: TechniqueEntry) -> None:
    tid = entry.id
    if not re.fullmatch(r"[A-Z][0-9]{2}", tid):
        raise CatalogueError(f"entry {tid}: id must look like C01")
    if not re.fullmatch(r"P[0-9]+\.[0-9]+", entry.roadmap_step):
        raise CatalogueError(f"entry {tid}: roadmap_step '{entry.roadmap_step}' must look like P3.4")
    if entry.phase != int(entry.roadmap_step[1:].split(".")[0]):
        raise CatalogueError(f"entry {tid}: phase {entry.phase} does not match roadmap_step {entry.roadmap_step}")
    if not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", entry.date_added):
        raise CatalogueError(f"entry {tid}: date_added must be YYYY-MM-DD")
    try:
        datetime.date.fromisoformat(entry.date_added)
    except ValueError as exc:
        raise CatalogueError(f"entry {tid}: date_added is not a real date: {exc}") from exc
    if entry.primary_source is not None and not entry.primary_source.startswith("https://"):
        raise CatalogueError(f"entry {tid}: primary_source must be null or an https URL")
    if entry.status == "validated" and not entry.evidence_path:
        raise CatalogueError(f"entry {tid}: validated status requires evidence_path")
