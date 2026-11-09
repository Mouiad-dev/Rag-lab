import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from .errors import EvaluationError

KEY = r"^[a-z0-9][a-z0-9._-]{0,63}$"


class _FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class EvalDocument(_FrozenModel):
    doc_key: str = Field(pattern=KEY)
    text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    length: int = Field(ge=1)


class RelevantSpan(_FrozenModel):
    doc_key: str = Field(pattern=KEY)
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    grade: int = Field(ge=1, le=3)
    quote: str

    @model_validator(mode="after")
    def quote_matches_offsets(self) -> "RelevantSpan":
        if self.end <= self.start:
            raise ValueError("end must be greater than start")
        if len(self.quote) != self.end - self.start:
            raise ValueError("quote length must equal end - start")
        return self


class EvalCase(_FrozenModel):
    case_id: str = Field(pattern=KEY)
    question: str = Field(min_length=1, max_length=2000)
    language: Literal["en", "ar", "mixed"]
    tags: tuple[Literal["identifier", "paraphrase", "diacritics", "table", "ocr"], ...] = ()
    split: Literal["dev", "validation", "held_out"]
    answerable: bool
    relevant: tuple[RelevantSpan, ...]
    notes: str = Field(default="", max_length=500)

    @field_validator("question")
    @classmethod
    def question_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be blank")
        return value

    @field_validator("tags")
    @classmethod
    def tags_sorted_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if value != tuple(sorted(set(value))):
            raise ValueError("tags must be sorted and unique")
        return value

    @model_validator(mode="after")
    def spans_consistent(self) -> "EvalCase":
        if self.answerable != bool(self.relevant):
            raise ValueError("answerable cases need spans; unanswerable cases need none")
        by_doc: dict[str, list[RelevantSpan]] = {}
        for span in self.relevant:
            by_doc.setdefault(span.doc_key, []).append(span)
        for spans in by_doc.values():
            spans.sort(key=lambda item: (item.start, item.end))
            if any(right.start < left.end for left, right in zip(spans, spans[1:], strict=False)):
                raise ValueError("spans on one document must not overlap")
        return self


class EvalDataset(_FrozenModel):
    format: Literal["rag-foundry.eval-dataset"]
    format_version: Literal["1"]
    dataset_id: str = Field(pattern=KEY)
    dataset_version: str = Field(pattern=r"^[0-9]+(\.[0-9]+)*$")
    description: str = Field(max_length=500)
    documents: tuple[EvalDocument, ...] = Field(min_length=1)
    cases: tuple[EvalCase, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def references_valid(self) -> "EvalDataset":
        docs = {doc.doc_key: doc for doc in self.documents}
        if len(docs) != len(self.documents):
            raise ValueError("document doc_key values must be unique")
        ids = [case.case_id for case in self.cases]
        if len(set(ids)) != len(ids):
            raise ValueError("case_id values must be unique")
        for case in self.cases:
            for span in case.relevant:
                doc = docs.get(span.doc_key)
                if doc is None:
                    raise ValueError("span references unknown doc_key")
                if span.end > doc.length:
                    raise ValueError("span end exceeds document length")
        return self


def _reject_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant: {value}")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_dataset(path: str | Path) -> EvalDataset:
    try:
        text = Path(path).read_text(encoding="utf-8", errors="strict")
        json.loads(text, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
        return EvalDataset.model_validate_json(text, strict=True)
    except ValidationError as exc:
        error = exc.errors()[0]
        reason = f"{'.'.join(map(str, error['loc']))}: {error['msg']} [{error['type']}]"
        raise EvaluationError("invalid_dataset", reason) from exc
    except Exception as exc:
        reason = str(exc).splitlines()[0][:200] or type(exc).__name__
        raise EvaluationError("invalid_dataset", reason) from exc


def canonical_json(obj: object) -> str:
    return json.dumps(obj, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def dataset_sha256(ds: EvalDataset) -> str:
    payload = canonical_json(ds.model_dump(mode="json"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def dump_dataset(ds: EvalDataset) -> str:
    return json.dumps(ds.model_dump(mode="json"), ensure_ascii=True, sort_keys=True, indent=2) + "\n"
