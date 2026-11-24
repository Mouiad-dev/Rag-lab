"""Typed errors for the techniques package. None subclass KeyError or LookupError."""

from collections.abc import Iterable
from dataclasses import dataclass


class TechniqueError(Exception):
    """Base class for technique errors."""


class CatalogueError(TechniqueError):
    """The catalogue file is missing, malformed or violates its own rules."""


class UnknownTechniqueError(TechniqueError):
    """A technique ID is not in the catalogue."""

    def __init__(self, technique_id: str) -> None:
        self.technique_id = technique_id
        super().__init__(f"unknown technique ID '{technique_id}': not in the catalogue")


class TechniqueUnavailableError(TechniqueError):
    """A catalogued technique has no available implementation (used in P1.5-05)."""


class PluginKindError(TechniqueError):
    """A plugin's declared kind does not fit the category of its catalogue entry."""


class DuplicatePluginError(TechniqueError):
    """Two plugins claim the same catalogue ID (used in P1.5-05)."""


@dataclass(frozen=True)
class PluginDiscoveryFailure:
    """One entry point that failed discovery."""

    entry_point: str
    value: str
    distribution: str | None
    reason: str
    error_type: str


class PluginDiscoveryError(TechniqueError):
    """One or more entry points failed discovery; nothing was registered."""

    def __init__(self, failures: Iterable[PluginDiscoveryFailure]) -> None:
        self.failures: tuple[PluginDiscoveryFailure, ...] = tuple(
            sorted(failures, key=lambda f: (f.entry_point, f.value, f.distribution or ""))
        )
        lines = [
            f"- {f.entry_point} ({f.value}, dist={f.distribution or 'unknown'}): {f.error_type}: {f.reason}"
            for f in self.failures
        ]
        super().__init__(f"plugin discovery failed for {len(self.failures)} entry point(s):\n" + "\n".join(lines))
