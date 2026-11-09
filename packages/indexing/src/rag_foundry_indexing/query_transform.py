"""Shared models and chat helpers for query transformations."""

from __future__ import annotations

import json
import re
import unicodedata
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from rag_foundry_domain.errors import QueryTransformError
from rag_foundry_providers.messages import ChatMessage, ChatRequest, ChatResult
from rag_foundry_providers.usage import Usage
from rag_foundry_techniques import RetrievalResult


class GeneratedQuery(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    role: Literal["original", "rewrite", "variant", "subquestion", "step_back", "hypothetical_document"]
    text: str
    synthetic: bool
    id: str | None = None
    depends_on: tuple[str, ...] = ()
    lost_constraints: tuple[str, ...] = ()


class QueryTransformTrace(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    technique_id: str
    status: Literal["applied", "rejected", "constraint_loss"]
    reason: str | None
    original_question: str
    generated: tuple[GeneratedQuery, ...]
    retrieval_queries: tuple[str, ...]
    usage: tuple[Usage, ...]
    prompt_sha256: str
    raw_text: str = Field(max_length=500)
    lost_constraints: tuple[str, ...]
    duplicates_dropped: int = 0


class TransformedRetrieval(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    trace: QueryTransformTrace
    result: RetrievalResult
    per_query: tuple[RetrievalResult, ...] = ()


def check_chat(chat) -> None:
    if chat is None:
        raise QueryTransformError("missing_model")
    if getattr(chat.profile.kind, "value", chat.profile.kind) != "chat":
        raise QueryTransformError("wrong_model_kind")


def check_question(q: str) -> None:
    if not q.strip():
        raise QueryTransformError("empty_query")
    if len(q) > 2000:
        raise QueryTransformError("query_too_long")


async def call_json(chat, *, system, payload, schema, max_output_tokens, temperature) -> tuple[dict, ChatResult]:
    check_chat(chat)
    request = ChatRequest(
        messages=(
            ChatMessage(role="system", content=system),
            ChatMessage(role="user", content=json.dumps(payload, ensure_ascii=False)),
        ),
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        response_schema=schema if chat.profile.capabilities.structured_output is True else None,
    )
    result = await chat.complete(request)
    from rag_foundry_chunking.model_based import parse_json_object

    data = result.structured if result.structured is not None else parse_json_object(result.text)
    if result.finish_reason == "length" or not isinstance(data, dict):
        error = QueryTransformError("invalid_model_output")
        error.chat_result = result
        raise error
    return data, result


_TOKEN_RE = re.compile(r"[^\W_]+(?:[-./][^\W_]+)*", re.UNICODE)
_WORD_RE = re.compile(r"[^\W_]+(?:['\u2019][^\W_]+)?", re.UNICODE)
_QUOTED_RE = re.compile(r'"([^"]*)"|\u00ab([^\u00bb]*)\u00bb')
_EN_NEG = {"not", "no", "never", "without", "except", "excluding", "cannot"}
_AR_NEG = {
    "\u0644\u0627",
    "\u0644\u064a\u0633",
    "\u063a\u064a\u0631",
    "\u0628\u062f\u0648\u0646",
    "\u0644\u0645",
    "\u0644\u0646",
}
_AR_PREFIX = ("\u0648", "\u0641", "\u0628")


def _normal(value: str) -> str:
    return unicodedata.normalize("NFC", value).casefold()


def _digit_ascii(value: str) -> str:
    return "".join(str(unicodedata.decimal(ch)) if ch.isdecimal() else ch for ch in value)


def _constraints(text: str) -> tuple[str, ...]:
    value = _normal(text)
    out: list[str] = []
    for match in _TOKEN_RE.finditer(value):
        token = match.group()
        if any(ch.isdecimal() for ch in token):
            out.append(_digit_ascii(token))
    out.extend(
        _normal(match.group(1) if match.group(1) is not None else match.group(2))
        for match in _QUOTED_RE.finditer(value)
    )
    for match in _WORD_RE.finditer(value):
        word = match.group()
        if word in _EN_NEG or word.endswith(("n't", "n\u2019t")):
            out.append("not")
        elif word in _AR_NEG:
            out.append(word)
        else:
            for prefix in _AR_PREFIX:
                if word.startswith(prefix) and word[len(prefix) :] in _AR_NEG:
                    out.append(word[len(prefix) :])
                    break
    return tuple(dict.fromkeys(out))


def hard_constraints(text: str) -> tuple[str, ...]:
    return _constraints(text)


def missing_constraints(original: str, texts) -> tuple[str, ...]:
    present = set().union(*(_constraints(text) for text in texts)) if texts else set()
    return tuple(item for item in _constraints(original) if item not in present)
