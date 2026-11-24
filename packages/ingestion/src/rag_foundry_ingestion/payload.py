"""Extraction payload v1: canonical JSON stored in the object store (ADR-0010 D5). No I/O here."""

import json
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError
from rag_foundry_domain.errors import ExtractionError
from rag_foundry_domain.extraction import SourceBlock
from rag_foundry_domain.offsets import OffsetMap

SCHEMA_VERSION = 1
_FIELDS = frozenset(
    {
        "schema_version",
        "extractor",
        "extractor_version",
        "profile_id",
        "unicode_version",
        "source_space",
        "page_starts",
        "canonical_text",
        "blocks",
        "source_map",
    }
)


def _bad() -> ExtractionError:
    return ExtractionError("integrity_mismatch", "extraction payload is invalid")


def _reject_constant(_name: str) -> Any:
    raise ValueError("non-finite number")


@dataclass(frozen=True)
class ExtractionPayload:
    extractor: str
    extractor_version: str
    profile_id: str
    unicode_version: str
    source_space: str
    page_starts: tuple[int, ...] | None
    canonical_text: str
    blocks: tuple[SourceBlock, ...]
    source_map: OffsetMap
    schema_version: int = SCHEMA_VERSION

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "extractor": self.extractor,
            "extractor_version": self.extractor_version,
            "profile_id": self.profile_id,
            "unicode_version": self.unicode_version,
            "source_space": self.source_space,
            "page_starts": None if self.page_starts is None else list(self.page_starts),
            "canonical_text": self.canonical_text,
            "blocks": [b.model_dump(mode="json") for b in self.blocks],
            "source_map": self.source_map.to_json(),
        }

    def dumps(self) -> bytes:
        return json.dumps(
            self.to_json(), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode("utf-8")

    @classmethod
    def loads(cls, data: bytes) -> "ExtractionPayload":
        try:
            raw = json.loads(data.decode("utf-8"), parse_constant=_reject_constant)
            return cls._from_json(raw)
        except ExtractionError:
            raise
        except (ValueError, TypeError, KeyError, ValidationError):
            raise _bad() from None

    @classmethod
    def _from_json(cls, raw: Any) -> "ExtractionPayload":
        if not isinstance(raw, dict) or set(raw) != _FIELDS or raw["schema_version"] != SCHEMA_VERSION:
            raise _bad()
        for name in ("extractor", "extractor_version", "profile_id", "unicode_version", "canonical_text"):
            if not isinstance(raw[name], str):
                raise _bad()
        if raw["source_space"] not in ("decoded_text", "pdf_page_text"):
            raise _bad()
        text = raw["canonical_text"]
        if not text or not isinstance(raw["blocks"], list) or not raw["blocks"]:
            raise _bad()
        blocks = tuple(SourceBlock.model_validate(b) for b in raw["blocks"])
        prev_end = 0
        for i, b in enumerate(blocks):
            if b.ordinal != i or b.start < prev_end or b.end <= b.start or b.end > len(text):
                raise _bad()
            prev_end = b.end
        starts: tuple[int, ...] | None = None
        if raw["page_starts"] is not None:
            ps = raw["page_starts"]
            if not isinstance(ps, list) or any(type(p) is not int for p in ps):
                raise _bad()
            starts = tuple(ps)
            if any(not 0 <= p <= len(text) for p in starts) or any(
                a >= b for a, b in zip(starts, starts[1:], strict=False)
            ):
                raise _bad()
        if not isinstance(raw["source_map"], list):
            raise _bad()
        source_map = OffsetMap.from_json(raw["source_map"])
        if source_map.n_len != len(text):
            raise _bad()
        return cls(
            extractor=raw["extractor"],
            extractor_version=raw["extractor_version"],
            profile_id=raw["profile_id"],
            unicode_version=raw["unicode_version"],
            source_space=raw["source_space"],
            page_starts=starts,
            canonical_text=text,
            blocks=blocks,
            source_map=source_map,
        )
