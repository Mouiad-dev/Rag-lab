from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import Literal

from pydantic import BaseModel, ConfigDict
from rag_foundry_chunking.boundaries import sentence_ranges
from rag_foundry_indexing.evidence_constraints import content_terms, evidence_constraints

from rag_foundry_answering.citations import parse_markers, strip_markers


class ClaimFlag(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    code: Literal["unknown_marker", "not_provided", "constraint_missing", "low_overlap"]
    value: str


class ClaimCheck(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    index: int
    start: int
    end: int
    text: str
    markers: tuple[str, ...]
    status: Literal["supported_lexical", "unsupported_lexical", "uncited", "invalid_citation"]
    flags: tuple[ClaimFlag, ...]
    overlap: float | None


class CitationReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    rule: Literal["lexical-v1"] = "lexical-v1"
    claims: tuple[ClaimCheck, ...]
    supported: int
    unsupported: int
    uncited: int
    invalid: int
    ok: bool


def validate_citations(
    answer: str,
    *,
    provided: Mapping[str, str],
    known: Collection[str],
    min_overlap: float = 0.5,
) -> CitationReport:
    """Check each claim's citations with the lexical-v1 rule.

    This is a lexical rule and does not assess meaning.

    1. Claims are ``sentence_ranges(answer, 0, len(answer))``. A segment that is empty after
       ``strip_markers`` adds its markers to the previous claim and extends its end; a leading
       marker-only segment goes to the next claim. Claim text is ``strip_markers(segment).strip()``.
    2. A marker not in ``known`` is flagged ``unknown_marker``; a known marker not in ``provided``
       is flagged ``not_provided``.
    3. Status order: any such flag gives ``invalid_citation``; else no markers gives ``uncited``;
       else the lexical checks run against the cited provided texts (overlap is None in the
       first two cases).
    4. Each ``evidence_constraints`` value of the claim missing from the union over the cited
       texts is flagged ``constraint_missing``. ``overlap = |claim terms & union of cited terms|
       / |claim terms|`` using ``content_terms``; it is None when the claim has no terms, and
       ``overlap < min_overlap`` is flagged ``low_overlap`` with value ``f"{overlap:.2f}"``. Any
       flag gives ``unsupported_lexical``; otherwise ``supported_lexical``.
    5. ``ok`` is True only when there is at least one claim and every claim is
       ``supported_lexical``.
    """
    if not 0 < min_overlap <= 1:
        raise ValueError("min_overlap must be in (0, 1]")

    segments: list[tuple[int, int, list[tuple[int, int]]]] = []
    for start, end in sentence_ranges(answer, 0, len(answer)):
        segment = answer[start:end]
        if not strip_markers(segment).strip():
            if segments:
                old_start, _, parts = segments[-1]
                parts.append((start, end))
                segments[-1] = (old_start, end, parts)
            else:
                segments.append((start, end, [(start, end)]))
            continue
        if segments and not strip_markers(answer[segments[-1][0] : segments[-1][1]]).strip():
            old_start, _, parts = segments[-1]
            parts.append((start, end))
            segments[-1] = (old_start, end, parts)
        else:
            segments.append((start, end, [(start, end)]))

    checks: list[ClaimCheck] = []
    for start, end, parts in segments:
        raw = " ".join(answer[a:b] for a, b in parts)
        markers = tuple(dict.fromkeys(marker for a, b in parts for marker in parse_markers(answer[a:b])))
        claim_text = strip_markers(raw).strip()
        flags: list[ClaimFlag] = []
        for marker in markers:
            if marker not in known:
                flags.append(ClaimFlag(code="unknown_marker", value=marker))
            elif marker not in provided:
                flags.append(ClaimFlag(code="not_provided", value=marker))
        overlap: float | None = None
        if any(flag.code in {"unknown_marker", "not_provided"} for flag in flags):
            status = "invalid_citation"
        elif not markers:
            status = "uncited"
        else:
            claim_constraints = evidence_constraints(claim_text)
            cited_constraints = set().union(*(evidence_constraints(provided[marker]) for marker in markers))
            cited_terms = frozenset().union(*(content_terms(provided[marker]) for marker in markers))
            for constraint in claim_constraints:
                if constraint not in cited_constraints:
                    flags.append(ClaimFlag(code="constraint_missing", value=constraint))
            claim_terms = content_terms(claim_text)
            if claim_terms:
                overlap = len(claim_terms & cited_terms) / len(claim_terms)
                if overlap < min_overlap:
                    flags.append(ClaimFlag(code="low_overlap", value=f"{overlap:.2f}"))
            status = "unsupported_lexical" if flags else "supported_lexical"
        checks.append(
            ClaimCheck(
                index=len(checks),
                start=start,
                end=end,
                text=claim_text,
                markers=markers,
                status=status,
                flags=tuple(flags),
                overlap=overlap,
            )
        )

    statuses = [claim.status for claim in checks]
    return CitationReport(
        claims=tuple(checks),
        supported=statuses.count("supported_lexical"),
        unsupported=statuses.count("unsupported_lexical"),
        uncited=statuses.count("uncited"),
        invalid=statuses.count("invalid_citation"),
        ok=bool(checks) and all(status == "supported_lexical" for status in statuses),
    )
