"""Typed errors for the providers package. None subclass KeyError or LookupError."""

from dataclasses import dataclass


class ProviderError(Exception):
    """Base class for provider errors."""


class UnknownProfileError(ProviderError):
    """A profile ID is not registered."""

    def __init__(self, profile_id: str) -> None:
        self.profile_id = profile_id
        super().__init__(f"unknown model profile '{profile_id}'")


class DuplicateProfileError(ProviderError):
    """Two profiles use the same ID."""

    def __init__(self, profile_id: str) -> None:
        self.profile_id = profile_id
        super().__init__(f"duplicate model profile '{profile_id}'")


class ProfileKindError(ProviderError):
    """A profile has the wrong kind for the requested port."""

    def __init__(self, profile_id: str, expected: str, actual: str) -> None:
        self.profile_id = profile_id
        self.expected = expected
        self.actual = actual
        super().__init__(f"profile '{profile_id}' has kind '{actual}', expected '{expected}'")


class ProviderUnavailableError(ProviderError):
    """A known profile has no bound adapter."""

    def __init__(self, profile_id: str) -> None:
        self.profile_id = profile_id
        super().__init__(f"profile '{profile_id}' is known but has no bound provider adapter")


@dataclass(frozen=True)
class CapabilityProblem:
    """One capability a stage needs that its profile does not declare as supported."""

    stage: str | None
    profile_id: str
    capability: str
    declared: bool | None

    def describe(self) -> str:
        state = "not declared" if self.declared is None else f"declared {str(self.declared).lower()}"
        prefix = f"stage '{self.stage}': " if self.stage is not None else ""
        return f"{prefix}profile '{self.profile_id}' lacks '{self.capability}' ({state})"


class MissingCapabilityError(ProviderError):
    """One or more profiles lack a required capability."""

    def __init__(self, problems: tuple[CapabilityProblem, ...]) -> None:
        if not problems:
            raise ValueError("MissingCapabilityError needs at least one problem")
        self.problems = problems
        super().__init__("; ".join(p.describe() for p in problems))
