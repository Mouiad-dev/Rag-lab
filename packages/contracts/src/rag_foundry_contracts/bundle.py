"""Portable source bundle v1 contract models (REUSE_AND_INTEGRATION section 5).

No tenant or project fields, no chunk or index IDs. Asset paths come from a fixed allowlist.
"""

import re
from typing import Annotated, Literal

from pydantic import Field, StringConstraints, field_validator, model_validator

from rag_foundry_contracts.source import SourceRef, _Model, check_contract_version

BUNDLE_CONTRACT_VERSION = "1.0"
_UUID = r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
BUNDLE_ASSET_PATH = re.compile(rf"(?:documents\.json|structure\.jsonl|anchors\.jsonl|text/{_UUID}\.txt)")
NonEmpty = Annotated[str, StringConstraints(min_length=1)]


class BundleAsset(_Model):
    path: str
    role: Literal["documents", "structure", "anchors", "text"]
    media_type: NonEmpty
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("path")
    @classmethod
    def _path(cls, value: str) -> str:
        if not BUNDLE_ASSET_PATH.fullmatch(value):
            raise ValueError("invalid bundle asset path")
        return value


class OmittedAsset(_Model):
    document_version_id: NonEmpty
    kind: Literal["original", "page_render"]
    reason: Literal["not_exported"]


class BundleDocument(_Model):
    source: SourceRef
    extractor: NonEmpty
    extractor_version: NonEmpty
    profile_id: NonEmpty
    unicode_version: NonEmpty
    canonical_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    char_count: int = Field(ge=0)
    block_count: int = Field(ge=0)
    page_count: int | None = Field(default=None, ge=0)
    text_path: str

    @field_validator("text_path")
    @classmethod
    def _text_path(cls, value: str) -> str:
        if not BUNDLE_ASSET_PATH.fullmatch(value) or not value.startswith("text/"):
            raise ValueError("invalid text path")
        return value


class BundleBlock(_Model):
    extraction_view_id: NonEmpty
    block_id: NonEmpty
    ordinal: int = Field(ge=0)
    kind: NonEmpty
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    section_path: tuple[str, ...] = ()
    heading_level: int | None = Field(default=None, ge=1)
    page_index: int | None = Field(default=None, ge=0)


class BundleManifest(_Model):
    contract_version: str = BUNDLE_CONTRACT_VERSION
    bundle_id: NonEmpty
    provider_namespace: NonEmpty
    created_at: str
    inclusion_policy: Literal["text_and_structure"]
    views: tuple[SourceRef, ...]
    assets: tuple[BundleAsset, ...]
    omitted: tuple[OmittedAsset, ...]

    @field_validator("contract_version")
    @classmethod
    def _version(cls, value: str) -> str:
        return check_contract_version(value)

    @model_validator(mode="after")
    def _unique(self) -> "BundleManifest":
        if len({a.path for a in self.assets}) != len(self.assets):
            raise ValueError("duplicate asset path")
        if len({v.extraction_view_id for v in self.views}) != len(self.views):
            raise ValueError("duplicate view")
        return self

    @field_validator("created_at")
    @classmethod
    def _utc(cls, value: str) -> str:
        if not re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d+)?Z", value):
            raise ValueError("created_at must be UTC ISO 8601")
        return value
