"""Validated runtime tunables and section-scoped resolution."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from rag_foundry_domain.errors import SettingsError

MIN_SCORE_KEY = r"^[a-z0-9._-]{1,64}/[A-Za-z0-9._:-]{1,128}/[1-9][0-9]{0,4}$"
RUNTIME_CONFIG_MAX_BYTES = 64 * 1024
ANSWER_K_MIN, ANSWER_K_MAX, ANSWER_K_DEFAULT = 1, 20, 5
ANSWER_MAX_OUTPUT_TOKENS_MIN, ANSWER_MAX_OUTPUT_TOKENS_MAX = 1, 4096
ANSWER_MAX_EVIDENCE_CHARS_MIN, ANSWER_MAX_EVIDENCE_CHARS_MAX = 500, 100000
ANSWER_TEMPERATURE_MIN, ANSWER_TEMPERATURE_MAX = 0.0, 2.0
Level = Literal["global", "tenant", "project"]


class _SettingsModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True, allow_inf_nan=False)


class AnswerSettings(_SettingsModel):
    max_output_tokens: int = Field(
        default=512, strict=True, ge=ANSWER_MAX_OUTPUT_TOKENS_MIN, le=ANSWER_MAX_OUTPUT_TOKENS_MAX
    )
    temperature: float = Field(default=0.0, strict=True, ge=ANSWER_TEMPERATURE_MIN, le=ANSWER_TEMPERATURE_MAX)
    max_evidence_chars: int = Field(
        default=12000, strict=True, ge=ANSWER_MAX_EVIDENCE_CHARS_MIN, le=ANSWER_MAX_EVIDENCE_CHARS_MAX
    )
    min_scores: dict[str, float] = Field(default_factory=lambda: {"ollama/bge-m3/1024": 0.45})

    @field_validator("min_scores")
    @classmethod
    def _validate_min_scores(cls, value: dict[str, float]) -> dict[str, float]:
        if len(value) > 32 or any(re.fullmatch(MIN_SCORE_KEY, key) is None for key in value):
            raise ValueError("min_scores keys or count are invalid")
        if any(not -1 <= score <= 1 for score in value.values()):
            raise ValueError("min_scores values are out of bounds")
        return value


class RunSettings(_SettingsModel):
    max_attempts: int = Field(default=2, strict=True, ge=1, le=5)
    attempt_timeout_s: float = Field(default=120.0, strict=True, gt=0, le=600)
    retry_timeouts_for_local: bool = Field(default=False, strict=True)
    deadline_s: float = Field(default=300.0, strict=True, gt=0, le=1800)
    max_concurrent: int = Field(default=1, strict=True, ge=1, le=8)


class IndexingSettings(_SettingsModel):
    embed_batch_size: int = Field(default=32, strict=True, ge=1, le=256)
    c05_chunk_size: int = Field(default=800, strict=True, ge=32, le=8000)
    c05_chunk_overlap: int = Field(default=100, strict=True, ge=0)

    @model_validator(mode="after")
    def _overlap_below_size(self) -> IndexingSettings:
        if self.c05_chunk_overlap >= self.c05_chunk_size:
            raise ValueError("chunk overlap must be smaller than chunk size")
        return self


class RuntimeSettings(_SettingsModel):
    schema_version: Literal[1] = 1
    answer: AnswerSettings = Field(default_factory=AnswerSettings)
    run: RunSettings = Field(default_factory=RunSettings)
    indexing: IndexingSettings = Field(default_factory=IndexingSettings)

    @field_validator("schema_version", mode="before")
    @classmethod
    def _schema_version_must_be_int(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("schema_version must be an integer")
        return value


KEY_SCOPES: Mapping[str, frozenset[Level]] = MappingProxyType(
    {
        "answer.max_output_tokens": frozenset(("global", "tenant", "project")),
        "answer.temperature": frozenset(("global", "tenant", "project")),
        "answer.max_evidence_chars": frozenset(("global", "tenant", "project")),
        "answer.min_scores": frozenset(("global", "tenant", "project")),
        "run.max_attempts": frozenset(("global",)),
        "run.attempt_timeout_s": frozenset(("global",)),
        "run.retry_timeouts_for_local": frozenset(("global",)),
        "run.deadline_s": frozenset(("global",)),
        "run.max_concurrent": frozenset(),
        "indexing.embed_batch_size": frozenset(("global",)),
        "indexing.c05_chunk_size": frozenset(("global", "tenant", "project")),
        "indexing.c05_chunk_overlap": frozenset(("global", "tenant", "project")),
    }
)
ALLOWED_KEYS = frozenset(KEY_SCOPES)


@dataclass(frozen=True)
class LayerOverride:
    level: Level
    key: str
    value: int | float | bool | dict[str, float] | None


@dataclass(frozen=True)
class RunSnapshot:
    answer: AnswerSettings
    run: RunSettings
    sources: Mapping[str, str]

    def effective_json(self) -> str:
        return json.dumps(
            {"answer": self.answer.model_dump(mode="json"), "run": self.run.model_dump(mode="json")},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )

    def sha256(self) -> str:
        return hashlib.sha256(self.effective_json().encode("ascii")).hexdigest()


def _resolve(
    base: RuntimeSettings, overrides: tuple[LayerOverride, ...], *, sections: frozenset[str], file_keys: frozenset[str]
):
    values = base.model_dump(mode="python")
    sources = {key: ("file" if key in file_keys else "default") for key in ALLOWED_KEYS}
    order = {"global": 0, "tenant": 1, "project": 2}
    for override in sorted(overrides, key=lambda item: order[item.level]):
        section = override.key.partition(".")[0]
        if section not in {"answer", "run", "indexing"} or override.key not in ALLOWED_KEYS:
            raise SettingsError("unknown_key")
        if section not in sections:
            continue
        if override.level not in KEY_SCOPES[override.key]:
            raise SettingsError("scope_not_allowed")
        if override.value is not None:
            values[section][override.key.partition(".")[2]] = override.value
            sources[override.key] = override.level
    validated = {}
    try:
        for section in sections:
            model = {"answer": AnswerSettings, "run": RunSettings, "indexing": IndexingSettings}[section]
            validated[section] = model.model_validate(values[section])
    except ValidationError:
        error = SettingsError("invalid_value")
    else:
        error = None
    if error is not None:
        raise error
    return validated, MappingProxyType(
        {key: value for key, value in sources.items() if key.startswith(tuple(s + "." for s in sections))}
    )


def resolve_run(
    base: RuntimeSettings,
    overrides: tuple[LayerOverride, ...] | list[LayerOverride],
    *,
    file_keys: frozenset[str] = frozenset(),
) -> RunSnapshot:
    validated, sources = _resolve(base, tuple(overrides), sections=frozenset(("answer", "run")), file_keys=file_keys)
    return RunSnapshot(validated["answer"], validated["run"], sources)


def resolve_run_settings(
    base: RuntimeSettings,
    overrides: tuple[LayerOverride, ...] | list[LayerOverride],
    *,
    file_keys: frozenset[str] = frozenset(),
) -> RunSettings:
    validated, _ = _resolve(base, tuple(overrides), sections=frozenset(("run",)), file_keys=file_keys)
    return validated["run"]


def resolve_indexing(
    base: RuntimeSettings,
    overrides: tuple[LayerOverride, ...] | list[LayerOverride],
    *,
    file_keys: frozenset[str] = frozenset(),
) -> IndexingSettings:
    validated, _ = _resolve(base, tuple(overrides), sections=frozenset(("indexing",)), file_keys=file_keys)
    return validated["indexing"]


def _no_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate key")
        result[key] = value
    return result


def load_runtime_settings_file(path: Path) -> tuple[RuntimeSettings, frozenset[str]]:
    try:
        with open(path, "rb") as stream:
            raw = stream.read(RUNTIME_CONFIG_MAX_BYTES + 1)
        if len(raw) > RUNTIME_CONFIG_MAX_BYTES:
            raise ValueError
        data = json.loads(raw.decode("utf-8"), object_pairs_hook=_no_duplicate_keys)
        if not isinstance(data, dict):
            raise ValueError
        keys: set[str] = set()
        for section, content in data.items():
            if section == "schema_version":
                continue
            if section not in ("answer", "run", "indexing") or not isinstance(content, dict):
                raise ValueError
            for key in content:
                dotted = f"{section}.{key}"
                if dotted not in ALLOWED_KEYS:
                    raise ValueError
                keys.add(dotted)
        settings = RuntimeSettings.model_validate(data)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError, ValidationError):
        error = SettingsError("invalid_file")
    else:
        error = None
    if error is not None:
        raise error
    return settings, frozenset(keys)
