import json
import unicodedata
from collections import Counter
from pathlib import Path

from _fixture import FACT_AR, FACT_EN, QUERY_AR, QUERY_EN
from _qa import answer_matches, corpus_markdown, load_fixture


def test_fixture_structure_and_content():
    path = Path(__file__).with_name("qa_fixture.v1.json")
    raw = path.read_text(encoding="utf-8")
    raw.encode("ascii")
    fixture = load_fixture()
    cases = fixture["cases"]
    assert len(cases) == len({case["id"] for case in cases}) == 10
    assert Counter(case["cohort"] for case in cases) == {"en": 5, "ar": 3, "mixed": 2}
    assert sum(not case["answerable"] for case in cases) == 3
    assert sum(case["off_topic"] for case in cases) == 1
    assert next(case for case in cases if case["off_topic"])["answerable"] is False
    assert {case["id"] for case in cases if case["reused_from_p34"]} == {
        "en_fact_observatory",
        "ar_fact_manuscripts",
        "en_unanswerable_population",
    }
    assert sum(case["security"] for case in cases) == 1
    assert fixture["canary"] in fixture["extra_paragraphs"][3]
    corpus = corpus_markdown()
    for fact in (case["fact"] for case in cases if case["fact"]):
        assert corpus.count(fact) == 1
    assert cases[0]["question"] == QUERY_EN and cases[0]["fact"] == FACT_EN
    assert cases[1]["question"] == QUERY_AR and cases[1]["fact"] == FACT_AR
    assert all(case["question"].endswith("\u061f") for case in cases if case["cohort"] == "ar")
    assert all(fixture["canary"] not in case["question"] for case in cases)
    assert not answer_matches("Open at 10 [S9].", cases[4])


def test_fixture_text_is_nfc():
    fixture = json.loads(Path(__file__).with_name("qa_fixture.v1.json").read_text(encoding="utf-8"))
    assert all(unicodedata.normalize("NFC", value) == value for value in fixture["extra_paragraphs"])
    for case in fixture["cases"]:
        for value in (case["question"], case["fact"], *case["expected_any"], *case["expected_all"]):
            if value is not None:
                assert unicodedata.normalize("NFC", value) == value
