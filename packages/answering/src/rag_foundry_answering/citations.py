"""Deterministic checks for citation marker form."""

import re
from collections.abc import Collection
from typing import Literal

from pydantic import BaseModel, ConfigDict

from rag_foundry_answering.prompt import DECLINE_TOKEN

MARKER_GROUP = re.compile(r"\[(S[1-9][0-9]?(?:\s*,\s*S[1-9][0-9]?)*)\]")


class AnswerCheck(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    outcome: Literal["ok", "declined", "truncated", "uncited", "invalid"]
    cited: tuple[str, ...]


def parse_markers(text: str) -> tuple[str, ...]:
    """Return unique recognized markers in their first appearance order."""
    found: list[str] = []
    for group in MARKER_GROUP.finditer(text):
        for marker in re.findall(r"S[1-9][0-9]?", group.group(1)):
            if marker not in found:
                found.append(marker)
    return tuple(found)


def strip_markers(text: str) -> str:
    """Remove every recognized citation marker group."""
    return MARKER_GROUP.sub("", text)


def check_answer(text: str, finish_reason: str, known: Collection[str]) -> AnswerCheck:
    """Check answer form, not support ("The passages do not state X [S1]" is ok)."""
    markers = parse_markers(text)
    if finish_reason == "length":
        return AnswerCheck(outcome="truncated", cited=markers)
    if DECLINE_TOKEN in text:
        return AnswerCheck(outcome="declined", cited=markers)
    if not text.strip() or not markers or not strip_markers(text).strip():
        return AnswerCheck(outcome="uncited", cited=markers)
    if any(marker not in known for marker in markers):
        return AnswerCheck(outcome="invalid", cited=markers)
    return AnswerCheck(outcome="ok", cited=markers)
