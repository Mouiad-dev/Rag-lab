"""Immutable, content-verified prompt templates and their store port."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, ValidationError, model_validator

from rag_foundry_domain.errors import PromptError

PromptPurpose = Literal["grounded", "ungrounded", "chunk-boundaries", "propositions"]


def prompt_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class _PromptModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


class PromptTemplate(_PromptModel):
    id: StrictStr = Field(pattern=r"^[a-z0-9-]+-v[0-9]+$", max_length=64)
    family: StrictStr = Field(min_length=1, max_length=59)
    version: StrictInt = Field(ge=1)
    purpose: PromptPurpose
    system_text: StrictStr = Field(min_length=1, max_length=20000)
    content_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _identity_and_hash(self) -> PromptTemplate:
        if self.id != f"{self.family}-v{self.version}":
            raise ValueError("template identity is invalid")
        if self.content_sha256 != prompt_sha256(self.system_text):
            raise ValueError("template hash is invalid")
        return self


def template_from_row(**fields: object) -> PromptTemplate:
    try:
        return PromptTemplate.model_validate(fields)
    except ValidationError as exc:
        if any("template hash is invalid" in error.get("msg", "") for error in exc.errors()):
            raise PromptError("hash_mismatch") from None
        raise PromptError("invalid_template") from None


def validate_loaded(template: PromptTemplate) -> None:
    if template.content_sha256 != prompt_sha256(template.system_text):
        raise PromptError("hash_mismatch")
    if "<<<" in template.system_text or ">>>" in template.system_text:
        raise PromptError("invalid_template")
    if template.purpose == "grounded" and not all(
        marker in template.system_text for marker in ("INSUFFICIENT_EVIDENCE", "[S1]")
    ):
        raise PromptError("invalid_template")
    if template.purpose == "chunk-boundaries" and not all(
        marker in template.system_text for marker in ("breaks", "starts_with")
    ):
        raise PromptError("invalid_template")
    if template.purpose == "propositions" and not all(
        marker in template.system_text for marker in ("propositions", "quotes")
    ):
        raise PromptError("invalid_template")


class PromptActivation(_PromptModel):
    seq: StrictInt
    purpose: PromptPurpose
    template_id: StrictStr
    activated_by: StrictStr
    activated_at: datetime
    decision_ref: StrictStr = Field(min_length=1, max_length=200)


class PromptTemplateStore(Protocol):
    async def get(self, template_id: str) -> PromptTemplate: ...

    async def active_id(self, purpose: PromptPurpose) -> str: ...

    async def create(self, template: PromptTemplate) -> None: ...

    async def activate(
        self, purpose: PromptPurpose, template_id: str, *, activated_by: str, decision_ref: str
    ) -> PromptActivation: ...

    async def activations(self, purpose: PromptPurpose) -> list[PromptActivation]: ...


class InMemoryPromptTemplateStore:
    def __init__(
        self,
        templates: Iterable[PromptTemplate] = (),
        activations: Iterable[PromptActivation] = (),
    ) -> None:
        self._templates: dict[str, PromptTemplate] = {}
        for template in templates:
            if template.id in self._templates:
                raise PromptError("template_exists")
            validate_loaded(template)
            self._templates[template.id] = template
        self._activations = list(activations)

    async def get(self, template_id: str) -> PromptTemplate:
        try:
            template = self._templates[template_id]
        except KeyError:
            raise PromptError("template_not_found") from None
        validate_loaded(template)
        return template

    async def active_id(self, purpose: PromptPurpose) -> str:
        for activation in reversed(self._activations):
            if activation.purpose == purpose:
                return activation.template_id
        raise PromptError("no_active_template")

    async def create(self, template: PromptTemplate) -> None:
        if template.id in self._templates:
            raise PromptError("template_exists")
        validate_loaded(template)
        self._templates[template.id] = template

    async def activate(
        self, purpose: PromptPurpose, template_id: str, *, activated_by: str, decision_ref: str
    ) -> PromptActivation:
        template = await self.get(template_id)
        if template.purpose != purpose:
            raise PromptError("purpose_mismatch")
        activation = PromptActivation(
            seq=max((row.seq for row in self._activations), default=0) + 1,
            purpose=purpose,
            template_id=template_id,
            activated_by=activated_by,
            activated_at=datetime.now(UTC),
            decision_ref=decision_ref,
        )
        self._activations.append(activation)
        return activation

    async def activations(self, purpose: PromptPurpose) -> list[PromptActivation]:
        return sorted((a for a in self._activations if a.purpose == purpose), key=lambda a: a.seq)


class PromptTemplateCache:
    """Share validated prompt rows across per-run prompt stores."""

    def __init__(self) -> None:
        self.templates: dict[str, PromptTemplate] = {}


class CachedPromptTemplateStore:
    def __init__(self, inner: PromptTemplateStore, cache: PromptTemplateCache | None = None) -> None:
        self._inner = inner
        self._cache = cache or PromptTemplateCache()

    async def get(self, template_id: str) -> PromptTemplate:
        template = self._cache.templates.get(template_id)
        if template is None:
            template = await self._inner.get(template_id)
        try:
            validate_loaded(template)
        except PromptError:
            self._cache.templates.pop(template_id, None)
            raise
        self._cache.templates[template_id] = template
        return template

    async def active_id(self, purpose: PromptPurpose) -> str:
        return await self._inner.active_id(purpose)

    async def create(self, template: PromptTemplate) -> None:
        await self._inner.create(template)

    async def activate(
        self, purpose: PromptPurpose, template_id: str, *, activated_by: str, decision_ref: str
    ) -> PromptActivation:
        return await self._inner.activate(purpose, template_id, activated_by=activated_by, decision_ref=decision_ref)

    async def activations(self, purpose: PromptPurpose) -> list[PromptActivation]:
        return await self._inner.activations(purpose)


async def active_template(store: PromptTemplateStore, purpose: PromptPurpose) -> PromptTemplate:
    return await store.get(await store.active_id(purpose))
