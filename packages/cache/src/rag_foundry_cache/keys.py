"""Response-cache identity: typed key inputs hashed server-side (R50/R51, ADR-0021).

Keys are SHA-256 hex of canonical JSON. No plaintext reaches a key, an error or a stored entry.
"""

import hashlib
import json
import unicodedata
from typing import Annotated, Any, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field
from rag_foundry_contracts.runs import RunResponse
from rag_foundry_domain.documents import Scope
from rag_foundry_domain.errors import CacheError

_KEY_VERSION = 1
_STRICT = ConfigDict(frozen=True, extra="forbid", strict=True)


def _not_blank(value: str) -> str:
    if not value.strip():
        raise ValueError("must not be blank")
    return value


NonBlank = Annotated[str, Field(min_length=1), AfterValidator(_not_blank)]


class GenerationParams(BaseModel):
    """Generation settings that change the answer. All fields are required."""

    model_config = _STRICT
    max_output_tokens: int = Field(ge=1)
    temperature: float | None = Field(allow_inf_nan=False)
    k: int = Field(ge=1)
    min_score: float | None = Field(allow_inf_nan=False)
    max_evidence_chars: int = Field(ge=1)
    packer: NonBlank
    """`"none"` or a config SHA-256."""
    citation_check: Literal["off", "warn"]


class ResponseKeyInputs(BaseModel):
    """Every input that defines the effective request except scope and question. No defaults."""

    model_config = _STRICT
    authz_revision: NonBlank
    history: tuple[tuple[Literal["user", "assistant"], NonBlank], ...]
    technique_id: NonBlank
    technique_version: NonBlank
    retriever_config_sha256: NonBlank
    index_version_id: NonBlank | None
    embedding_key: NonBlank | None
    prompt_template: NonBlank
    prompt_sha256: NonBlank
    chat_profile_id: NonBlank
    chat_provider: NonBlank
    chat_model: NonBlank
    generation: GenerationParams
    settings_sha256: NonBlank | None
    mode: Literal["source_only", "enriched"]
    freshness: Literal["static_snapshot", "current"]
    tools: tuple[NonBlank, ...]
    is_action_run: bool
    output_language: Literal["en", "ar", "auto"]
    contract_version: NonBlank


def _nfc(value: Any) -> Any:
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    if isinstance(value, dict):
        return {key: _nfc(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_nfc(item) for item in value]
    return value


def _digest(payload: dict[str, Any]) -> str:
    try:
        text = json.dumps(_nfc(payload), sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError):
        raise CacheError("cache_key_invalid") from None
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def _base(kind: str, scope: Scope, inputs: ResponseKeyInputs) -> dict[str, Any]:
    if not isinstance(scope, Scope) or not isinstance(inputs, ResponseKeyInputs):
        raise CacheError("cache_key_invalid")
    return {
        "v": _KEY_VERSION,
        "kind": kind,
        "scope": {"tenant_id": scope.tenant_id, "project_id": scope.project_id},
        "inputs": inputs.model_dump(mode="json"),
    }


def partition_key(scope: Scope, inputs: ResponseKeyInputs) -> str:
    """Hash of everything except the question. R51 searches inside one partition."""
    return _digest(_base("partition", scope, inputs))


def exact_key(scope: Scope, question: str, inputs: ResponseKeyInputs) -> str:
    """Hash of scope, inputs and the question. Only NFC is applied to strings."""
    if not isinstance(question, str) or not question.strip():
        raise CacheError("cache_key_invalid")
    payload = _base("exact", scope, inputs)
    payload["question"] = question
    return _digest(payload)


def source_version_ids(response: RunResponse) -> tuple[str, ...]:
    """Sorted unique document version IDs over all citation and evidence anchors."""
    anchors = [a for c in response.citations for a in c.anchors]
    anchors += [a for e in response.evidence for a in e.anchors]
    return tuple(sorted({a.source.document_version_id for a in anchors}))
