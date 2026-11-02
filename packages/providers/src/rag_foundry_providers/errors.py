"""Typed errors for the providers package. None subclass KeyError or LookupError."""

from dataclasses import dataclass
from typing import Literal


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


class AdapterDependencyError(ProviderError):
    """An adapter needs an optional dependency that is not installed."""

    def __init__(self) -> None:
        super().__init__("install rag-foundry-providers[ollama]")


ProviderCallReason = Literal["connection", "timeout", "model_not_found", "input_too_long", "bad_response"]


class ProviderCallError(ProviderError):
    """A provider call failed. The message holds only the reason and HTTP status.

    It never includes request text, response bodies, headers or environment values.
    """

    def __init__(self, profile_id: str, reason: ProviderCallReason, *, status: int | None = None) -> None:
        self.profile_id = profile_id
        self.reason = reason
        self.status = status
        suffix = f" (HTTP {status})" if status is not None else ""
        super().__init__(f"profile '{profile_id}' call failed: {reason}{suffix}")
