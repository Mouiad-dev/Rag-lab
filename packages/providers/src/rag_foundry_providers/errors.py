"""Typed errors for the providers package. None subclass KeyError or LookupError."""

from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from rag_foundry_providers.usage import Usage


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


class UnboundStageError(ProviderError):
    """A pipeline needs stages that the providers config does not bind. Stage names only."""

    def __init__(self, stages: tuple[str, ...]) -> None:
        if not stages:
            raise ValueError("UnboundStageError needs at least one stage")
        self.stages = tuple(sorted(stages))
        super().__init__("providers config does not bind stage(s): " + ", ".join(f"'{s}'" for s in self.stages))


class AdapterDependencyError(ProviderError):
    """An adapter needs an optional dependency that is not installed."""

    def __init__(self, extra: str = "ollama") -> None:
        super().__init__(f"install rag-foundry-providers[{extra}]")


class MissingCredentialError(ProviderError):
    """A required credential environment variable is not set. The message names the variable only."""

    def __init__(self, env_var: str) -> None:
        self.env_var = env_var
        super().__init__(f"environment variable {env_var} is not set or empty")


# "billing" is used for HTTP 402 (Anthropic billing_error); "auth" covers 401 and 403.
ProviderCallReason = Literal[
    "connection",
    "timeout",
    "model_not_found",
    "input_too_long",
    "bad_response",
    "auth",
    "billing",
    "rate_limited",
    "overloaded",
    "server_error",
    "invalid_request",
    "refused",
]

# Reasons the retry wrapper may retry (ADR-0001 P2.5 D3). `timeout` is decided by the wrapper (local only).
RETRYABLE_REASONS: frozenset[ProviderCallReason] = frozenset(
    {"connection", "rate_limited", "overloaded", "server_error"}
)


def parse_retry_after(value: object) -> float | None:
    """Delta-seconds only: a str of 1 to 6 ASCII digits. Anything else (HTTP-date, sign, decimals, spaces) is None."""
    if not isinstance(value, str) or not (1 <= len(value) <= 6):
        return None
    if not all(c in "0123456789" for c in value):
        return None
    return float(int(value))


class ProviderCallError(ProviderError):
    """A provider call failed. The message holds only the reason and HTTP status.

    It never includes request text, response bodies, headers or environment values.
    """

    def __init__(
        self,
        profile_id: str,
        reason: ProviderCallReason,
        *,
        status: int | None = None,
        request_id: str | None = None,
        usage: "Usage | None" = None,
        retry_after_s: float | None = None,
    ) -> None:
        self.profile_id = profile_id
        self.reason = reason
        self.status = status
        self.request_id = request_id  # non-secret provider request id; kept out of the message
        # Billed usage of a 200 reply that still failed (for example a refusal), so callers can
        # record the cost. Never part of the message, str or repr.
        self.usage = usage
        # Parsed Retry-After in seconds (bounded number); never part of the message, str or repr.
        self.retry_after_s = retry_after_s
        suffix = f" (HTTP {status})" if status is not None else ""
        super().__init__(f"profile '{profile_id}' call failed: {reason}{suffix}")


class BudgetError(ProviderError):
    """A run budget refused a call. Never a ProviderCallError, so retry wrappers never retry it."""


class BudgetExceededError(BudgetError):
    """Spent + reserved + the worst case of this call would pass a cap. Numbers and profile id only."""

    def __init__(
        self,
        profile_id: str,
        limit: Literal["usd", "calls"],
        cap: Decimal | int,
        spent: Decimal | int,
        reserved: Decimal | int,
        requested: Decimal | int,
    ) -> None:
        self.profile_id = profile_id
        self.limit = limit
        self.cap = cap
        self.spent = spent
        self.reserved = reserved
        self.requested = requested
        super().__init__(
            f"budget exceeded for profile '{profile_id}': {limit} cap {cap}, spent {spent}, "
            f"reserved {reserved}, requested {requested}"
        )


class UnboundedCostError(BudgetError):
    """A paid profile lacks a rate or output cap needed to bound the cost of a call."""

    def __init__(self, profile_id: str, missing: str) -> None:
        self.profile_id = profile_id
        self.missing = missing
        super().__init__(f"profile '{profile_id}' cannot be budgeted: missing {missing}")


class BudgetRequiredError(BudgetError):
    """A run binds non-local profiles but gave no budget. Profile ids only."""

    def __init__(self, profile_ids: tuple[str, ...]) -> None:
        if not profile_ids:
            raise ValueError("BudgetRequiredError needs at least one profile id")
        self.profile_ids = tuple(sorted(profile_ids))
        super().__init__(
            "a RunBudget is required for non-local profile(s): " + ", ".join(f"'{p}'" for p in self.profile_ids)
        )
