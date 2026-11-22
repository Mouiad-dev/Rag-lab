"""R47 CAG knowledge snapshot: every chunk of one ready index version, in a fixed order, with a stable nonce.

ADR-0021: the snapshot is built in memory and never persisted. Capacity is checked explicitly; nothing
is silently selected or truncated. The nonce is secret: only its SHA-256 enters `snapshot_id`, and it
never appears in repr, dumps, logs or error messages.
"""

import hashlib
import json
import math
import re
import secrets
from collections.abc import Callable
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field
from rag_foundry_chunking import LabRegexTokenizer
from rag_foundry_domain.documents import Scope
from rag_foundry_domain.errors import RunError
from rag_foundry_domain.indexing import IndexState
from rag_foundry_domain.ports import IndexRepository
from rag_foundry_domain.prompts import PromptTemplate
from rag_foundry_domain.runtime_settings import ANSWER_MAX_EVIDENCE_CHARS_MAX, ANSWER_MAX_EVIDENCE_CHARS_MIN
from rag_foundry_providers.capabilities import ModelProfile

from rag_foundry_answering.prompt import build_grounded_request

CAG_SCHEMA = "rag_foundry.cag_snapshot/1"
MAX_CAG_CHUNKS = 99
_NONCE_DRAWS = 8
_PAGE = 100
_NONCE = re.compile(r"^[0-9a-f]{16}$")
_FROZEN = ConfigDict(frozen=True, extra="forbid")
_Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class CagItem(BaseModel):
    model_config = _FROZEN
    marker: str = Field(pattern=r"^S[1-9][0-9]?$")
    chunk_id: str
    extraction_view_id: str
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    block_ids: tuple[str, ...]
    text: str


class CagSnapshot(BaseModel):
    model_config = _FROZEN
    schema_id: Literal["rag_foundry.cag_snapshot/1"] = Field(default=CAG_SCHEMA, alias="schema")
    snapshot_id: _Sha256
    tenant_id: str
    project_id: str
    index_version_id: str
    embedding_key: _Sha256
    items: tuple[CagItem, ...] = Field(min_length=1, max_length=MAX_CAG_CHUNKS)
    nonce: str = Field(pattern=r"^[0-9a-f]{16}$", repr=False, exclude=True)
    prompt_sha256: _Sha256
    chat_profile_id: str
    chat_model: str
    context_window: int = Field(gt=0)
    reserve_output_tokens: int = Field(ge=0)
    max_question_tokens: int = Field(ge=0)
    max_evidence_chars: int = Field(ge=ANSWER_MAX_EVIDENCE_CHARS_MIN, le=ANSWER_MAX_EVIDENCE_CHARS_MAX)
    estimate_margin: float = Field(ge=1.0)
    estimated_prompt_tokens: int = Field(ge=0)
    tokenizer: Literal["lab-regex-v1"] = "lab-regex-v1"

    def check_scope(self, scope: Scope, index_version_id: str) -> None:
        if (
            scope.tenant_id != self.tenant_id
            or scope.project_id != self.project_id
            or index_version_id != self.index_version_id
        ):
            raise RunError("cag_snapshot_mismatch")


def _overflow(detail: str) -> RunError:
    """RunError takes only a code; attach a numbers-only detail without registering new codes."""
    err = RunError("cag_overflow")
    err.message = detail
    err.args = (detail,)
    return err


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


async def build_cag_snapshot(
    scope: Scope,
    index: IndexRepository,
    index_version_id: str,
    *,
    template: PromptTemplate,
    chat_profile: ModelProfile,
    reserve_output_tokens: int,
    max_question_tokens: int,
    max_evidence_chars: int,
    estimate_margin: float = 1.25,
    nonce_factory: Callable[[], str] = lambda: secrets.token_hex(8),
) -> CagSnapshot:
    if not ANSWER_MAX_EVIDENCE_CHARS_MIN <= max_evidence_chars <= ANSWER_MAX_EVIDENCE_CHARS_MAX:
        raise ValueError("max_evidence_chars out of range")
    if estimate_margin < 1 or reserve_output_tokens < 0 or max_question_tokens < 0:
        raise ValueError("estimate_margin must be >= 1 and token reserves must be >= 0")
    version = await index.get_index_version(scope, index_version_id)  # stranger -> NotFoundError
    if version.state is not IndexState.READY:
        raise RunError("cag_snapshot_mismatch")
    records = []
    while True:
        page = await index.list_chunks(scope, index_version_id, offset=len(records), limit=_PAGE)
        records.extend(page)
        if len(records) > MAX_CAG_CHUNKS:
            raise RunError("cag_too_many_chunks")
        if len(page) < _PAGE:
            break
    if not records:
        raise RunError("cag_empty")
    if len(records) > MAX_CAG_CHUNKS:
        raise RunError("cag_too_many_chunks")
    records.sort(key=lambda c: (c.extraction_view_id, c.start, c.end, c.id))
    total = sum(len(c.text) for c in records)
    if total > max_evidence_chars:
        raise _overflow(f"evidence_chars={total}/{max_evidence_chars}")
    items = tuple(
        CagItem(
            marker=f"S{i}",
            chunk_id=c.id,
            extraction_view_id=c.extraction_view_id,
            start=c.start,
            end=c.end,
            block_ids=c.block_ids,
            text=c.text,
        )
        for i, c in enumerate(records, 1)
    )
    nonce = _draw_nonce(nonce_factory, [i.text for i in items])
    window = chat_profile.capabilities.context_window
    request = build_grounded_request(
        template,
        "?",
        [(i.marker, i.text) for i in items],
        nonce=nonce,
        max_output_tokens=max(1, reserve_output_tokens),
        temperature=None,
    )
    # lab-regex-v1 is an estimate, not the model tokenizer; the live test checks provider prompt tokens.
    tokenizer = LabRegexTokenizer()
    estimate = sum(len(tokenizer.token_spans(m.content)) for m in request.messages)
    needed = math.ceil(estimate_margin * (estimate + max_question_tokens)) + reserve_output_tokens
    if window is None or needed > window:
        raise _overflow(f"prompt_tokens={needed}/{window}")
    identity = {
        "schema": CAG_SCHEMA,
        "tenant_id": scope.tenant_id,
        "project_id": scope.project_id,
        "index_version_id": index_version_id,
        "embedding_key": version.embedding_key,
        "chunks": [[i.chunk_id, _sha(i.text)] for i in items],
        "prompt_sha256": template.content_sha256,
        "chat_profile_id": chat_profile.id,
        "chat_model": chat_profile.model,
        "context_window": window,
        "reserve_output_tokens": reserve_output_tokens,
        "max_question_tokens": max_question_tokens,
        "max_evidence_chars": max_evidence_chars,
        "estimate_margin": estimate_margin,
        "nonce_sha256": _sha(nonce),
    }
    snapshot_id = _sha(json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=True))
    return CagSnapshot(
        snapshot_id=snapshot_id,
        tenant_id=scope.tenant_id,
        project_id=scope.project_id,
        index_version_id=index_version_id,
        embedding_key=version.embedding_key,
        items=items,
        nonce=nonce,
        prompt_sha256=template.content_sha256,
        chat_profile_id=chat_profile.id,
        chat_model=chat_profile.model,
        context_window=window,
        reserve_output_tokens=reserve_output_tokens,
        max_question_tokens=max_question_tokens,
        max_evidence_chars=max_evidence_chars,
        estimate_margin=estimate_margin,
        estimated_prompt_tokens=estimate,
    )


def _draw_nonce(factory: Callable[[], str], texts: list[str]) -> str:
    for _ in range(_NONCE_DRAWS):
        draw = factory()
        if _NONCE.fullmatch(draw) and not any(draw in t for t in texts):
            return draw
    raise RuntimeError("could not draw a nonce that avoids the chunk texts")
