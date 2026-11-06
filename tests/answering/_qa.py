"""Shared bilingual answer fixture helpers."""

import json
import unicodedata
from pathlib import Path

from _fixture import DISTRACTORS, FACT_AR, FACT_EN, MUSEUM
from rag_foundry_answering.citations import strip_markers


def load_fixture():
    return json.loads((Path(__file__).with_name("qa_fixture.v1.json")).read_text(encoding="utf-8"))


def corpus_markdown():
    fixture = load_fixture()
    return "# Foundry field notes\n\n" + "\n\n".join(
        (FACT_EN, FACT_AR, *fixture["extra_paragraphs"], *DISTRACTORS, MUSEUM)
    )


def answer_matches(answer, case):
    value = unicodedata.normalize("NFC", strip_markers(answer)).casefold()
    any_match = not case["expected_any"] or any(
        unicodedata.normalize("NFC", option).casefold() in value for option in case["expected_any"]
    )
    all_match = all(unicodedata.normalize("NFC", option).casefold() in value for option in case["expected_all"])
    return any_match and all_match
