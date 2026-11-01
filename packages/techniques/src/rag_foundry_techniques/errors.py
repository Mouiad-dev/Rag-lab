"""Typed errors for the techniques package. None subclass KeyError or LookupError."""


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


class DuplicatePluginError(TechniqueError):
    """Two plugins claim the same catalogue ID (used in P1.5-05)."""
