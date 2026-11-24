"""Strict, versioned providers config (ADR-0001 items 2 to 4): endpoints, model profiles, stage bindings.

The file holds only the NAME of an API key's environment variable, never a key value. Errors name the
field location and a fixed problem text; they never echo input values. No environment variable is read here.
"""

import json
import re
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal, Self, get_args
from urllib.parse import urlsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    ValidationError,
    field_validator,
    model_validator,
)

from rag_foundry_providers._urls import is_loopback, validate_base_url
from rag_foundry_providers.capabilities import _ID_PATTERN, ModelProfile
from rag_foundry_providers.errors import ProviderError

MAX_CONFIG_BYTES = 1024 * 1024

AdapterName = Literal["fake", "ollama", "anthropic", "openai_compatible"]
# Closed set: the stage roles of ADR-0001 Decision item 3, verbatim.
StageName = Literal["generator", "embedder", "reranker", "query_transform", "judge", "chunk_llm", "scout"]

# Capabilities each adapter actually implements. A profile may not declare a flag True that its adapter lacks.
# In P2.4 no adapter implements tool calling, structured output or vision.
_IMPLEMENTED: dict[str, frozenset[str]] = {
    "fake": frozenset(),
    "ollama": frozenset(),
    "anthropic": frozenset(),
    "openai_compatible": frozenset(),
}
_GATED_FLAGS = ("tool_calling", "structured_output", "vision")
_RATE_FIELDS = ("cost_per_mtok_in", "cost_per_mtok_out", "cost_per_mtok_cache_read", "cost_per_mtok_cache_write")
_QUOTED = re.compile(r"'[^']*'")
_RATE_FORMAT = re.compile(r"^[0-9]+(\.[0-9]+)?$")


class ProvidersConfigError(ProviderError):
    """The providers config is invalid. ``errors`` holds (location, problem) pairs; no input values."""

    def __init__(self, errors: list[tuple[str, str]] | tuple[tuple[str, str], ...]) -> None:
        if not errors:
            raise ValueError("ProvidersConfigError needs at least one problem")
        self.errors: tuple[tuple[str, str], ...] = tuple(errors)
        super().__init__("invalid providers config: " + "; ".join(f"{loc}: {problem}" for loc, problem in self.errors))


class Endpoint(BaseModel):
    """Where a provider is reached. ``api_key_env`` is an env-var NAME, never a key."""

    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)

    id: str = Field(pattern=_ID_PATTERN)
    adapter: AdapterName
    base_url: str | None = None
    api_key_env: str | None = Field(default=None, pattern=r"^[A-Z][A-Z0-9_]{0,63}$")
    local: StrictBool
    # Settings of the openai_compatible adapter live on the endpoint, not on ModelProfile: they describe how
    # the server's API behaves, and ModelProfile is shared by every adapter. One endpoint = one server and key.
    max_tokens_field: Literal["max_tokens", "max_completion_tokens"] | None = None
    supports_temperature: StrictBool | None = None  # unset: True on a local endpoint, False on a non-local one
    default_max_tokens: StrictInt | None = Field(default=None, gt=0)  # output cap when a request gives none

    @field_validator("base_url")
    @classmethod
    def _check_url(cls, value: str | None) -> str | None:
        return None if value is None else validate_base_url(value, "base_url")

    @model_validator(mode="after")
    def _check_adapter_rules(self) -> Self:
        if self.adapter == "fake" and (self.base_url is not None or self.api_key_env is not None):
            raise ValueError("adapter 'fake' takes no base_url and no api_key_env")
        if self.adapter != "openai_compatible" and (
            self.max_tokens_field is not None
            or self.supports_temperature is not None
            or self.default_max_tokens is not None
        ):
            raise ValueError("max_tokens_field, supports_temperature and default_max_tokens are openai_compatible only")
        if self.adapter == "openai_compatible" and not self.local and self.default_max_tokens is None:
            raise ValueError("a non-local openai_compatible endpoint requires default_max_tokens")
        if self.adapter in ("ollama", "openai_compatible") and self.base_url is None:
            raise ValueError("this adapter requires base_url")
        if self.adapter == "anthropic" and self.api_key_env is None:
            raise ValueError("adapter 'anthropic' requires api_key_env")
        # Whenever a key is set, require https unless the host is loopback (scheme compared case-insensitively).
        if (
            self.api_key_env is not None
            and self.base_url is not None
            and urlsplit(self.base_url).scheme.lower() != "https"
            and not is_loopback(self.base_url)
        ):
            raise ValueError("a key requires an https base_url unless the host is loopback")
        return self


def _rate_problems(data: object) -> list[tuple[str, str]]:
    problems: list[tuple[str, str]] = []
    profiles = data.get("profiles") if isinstance(data, dict) else None
    if not isinstance(profiles, (list, tuple)):
        return problems
    for i, profile in enumerate(profiles):
        if not isinstance(profile, dict):
            continue
        for name in _RATE_FIELDS:
            value = profile.get(name)
            if value is None:
                continue
            if not isinstance(value, (str, Decimal)):
                problems.append((f"profiles.{i}.{name}", 'rate must be a JSON string such as "0.10", not a number'))
            elif not _RATE_FORMAT.fullmatch(str(value)):
                problems.append((f"profiles.{i}.{name}", 'rate must be a plain decimal string such as "0.10"'))
    return problems


class ProvidersConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)

    schema_version: Literal[1]
    note: str | None = None
    endpoints: tuple[Endpoint, ...]
    profiles: tuple[ModelProfile, ...]
    stages: dict[StageName, str]

    @model_validator(mode="before")
    @classmethod
    def _rates_are_strings(cls, data: Any) -> Any:
        problems = _rate_problems(data)
        if problems:
            raise ProvidersConfigError(problems)
        return data

    @model_validator(mode="after")
    def _cross_checks(self) -> Self:
        problems: list[tuple[str, str]] = []
        endpoints: dict[str, Endpoint] = {}
        for i, endpoint in enumerate(self.endpoints):
            if endpoint.id in endpoints:
                problems.append((f"endpoints.{i}.id", "duplicate endpoint id"))
            endpoints.setdefault(endpoint.id, endpoint)
        profile_ids: set[str] = set()
        for i, profile in enumerate(self.profiles):
            if profile.id in profile_ids:
                problems.append((f"profiles.{i}.id", "duplicate profile id"))
            profile_ids.add(profile.id)
            endpoint = endpoints.get(profile.provider)
            if endpoint is None:
                problems.append((f"profiles.{i}.provider", "does not name a defined endpoint"))
                continue
            if profile.capabilities.local != endpoint.local:
                problems.append((f"profiles.{i}.capabilities.local", "must equal the endpoint's local flag"))
            implemented = _IMPLEMENTED[endpoint.adapter]
            for flag in _GATED_FLAGS:
                if getattr(profile.capabilities, flag) is True and flag not in implemented:
                    problems.append((f"profiles.{i}.capabilities.{flag}", "not implemented by this profile's adapter"))
        for stage, target in self.stages.items():
            if target not in profile_ids:
                problems.append((f"stages.{stage}", "does not name a defined profile"))
        if problems:
            raise ProvidersConfigError(problems)
        return self


def _location(loc: tuple[int | str, ...]) -> str:
    return ".".join(str(part) for part in loc) or "<root>"


def _wrap(exc: ValidationError) -> ProvidersConfigError:
    problems: list[tuple[str, str]] = []
    for err in exc.errors(include_input=False, include_url=False, include_context=False):
        msg = err["msg"]
        if err["type"] == "value_error":
            # Our own messages carry no values; strip any quoted text defensively (e.g. profile ids).
            msg = _QUOTED.sub("'...'", msg.removeprefix("Value error, "))
        loc = list(err["loc"])
        # An unknown key or dict key is caller input (it could be a pasted secret): never echo it.
        if err["type"] == "extra_forbidden" and loc:
            loc[-1] = "<unknown key>"
        elif loc and loc[-1] == "[key]" and len(loc) >= 2:
            loc[-2] = "<unknown key>"
        # Value errors under "stages" carry the dict key in the location too.
        if len(loc) > 1 and loc[0] == "stages" and loc[1] not in get_args(StageName):
            loc[1] = "<unknown key>"
        problems.append((_location(tuple(loc)), msg))
    return ProvidersConfigError(problems)


def parse_providers_config(data: object) -> ProvidersConfig:
    """Validate already-decoded JSON data; raises ProvidersConfigError only."""
    error: ProvidersConfigError | None = None
    try:
        return ProvidersConfig.model_validate(data)
    except ValidationError as exc:
        error = _wrap(exc)
    except ProvidersConfigError as exc:
        error = ProvidersConfigError(exc.errors)
    # Raised outside the except block so the error has no __cause__ or __context__ (no input values).
    raise error


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    seen: set[str] = set()
    for key, _ in pairs:
        if key in seen:
            raise ValueError("duplicate key")
        seen.add(key)
    return dict(pairs)


def load_providers_config(path: Path) -> ProvidersConfig:
    """Read a JSON providers config. UTF-8 only, at most 1 MiB, no duplicate keys, object root."""
    problem: str | None = None
    data: object = None
    try:
        with open(path, "rb") as fh:
            raw = fh.read(MAX_CONFIG_BYTES + 1)
        if len(raw) > MAX_CONFIG_BYTES:
            problem = "file is larger than 1 MiB"
        else:
            data = json.loads(raw.decode("utf-8"), object_pairs_hook=_no_duplicate_keys)
            if not isinstance(data, dict):
                problem = "root must be a JSON object"
    except OSError:
        problem = "file cannot be read"
    except UnicodeDecodeError:
        problem = "file is not valid UTF-8"
    except json.JSONDecodeError:
        problem = "file is not valid JSON"
    except ValueError:
        problem = "file has a duplicate key in a JSON object"
    if problem is not None:
        raise ProvidersConfigError([("<file>", problem)])
    return parse_providers_config(data)
