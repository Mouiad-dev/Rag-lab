import json
import subprocess
import sys
import unicodedata
from pathlib import Path

import pytest
from rag_foundry_chunking import register_builtin_plugins
from rag_foundry_chunking.comparison import (
    LexicalOverlapRanker,
    calculate_ranked_metrics,
    compare_chunkers,
    load_span_fixture,
    span_metrics,
)
from rag_foundry_techniques.catalogue import load_catalogue
from rag_foundry_techniques.registry import TechniqueRegistry


def test_hand_computed_span_metrics_and_ranking():
    assert span_metrics(2, 6, [(0, 8), (2, 6)]) == {
        "best_coverage": 1.0,
        "best_precision": 1.0,
        "contained": True,
        "pieces": 2,
    }
    assert span_metrics(2, 6, [(0, 8), (0, 6)]) == {
        "best_coverage": 1.0,
        "best_precision": 0.6667,
        "contained": True,
        "pieces": 2,
    }


def test_public_row_toy_recall_mrr_and_unanswerable():
    class Case:
        case_id, cohort = "toy", "en"
        relevant = ({"start": 0, "end": 4}, {"start": 8, "end": 12})

    ranked = [(1.0, 0, 4), (0.5, 4, 8), (0.2, 8, 12)]
    metrics = calculate_ranked_metrics(Case.relevant, ranked, [(0, 4), (8, 12)])
    assert [metrics[f"span_recall@{k}"] for k in (1, 3)] == [0.5, 1.0]
    assert metrics["mrr"] == 1.0
    late = calculate_ranked_metrics(Case.relevant, [(1.0, 4, 8), (0.9, 12, 16), (0.5, 0, 4)], [(0, 4)])
    assert (late["mrr"], late["first_hit_k"], late["hit@1"], late["hit@3"]) == (0.3333, 3, 0, 1)
    Case.relevant = ()
    metrics = calculate_ranked_metrics(Case.relevant, [(0.4, 0, 3)], [])
    assert set(metrics) == {"context_chars@1", "context_chars@3", "context_chars@5"}


def test_fixture_nfc_and_arabic_nfd_query_matches_nfc_chunk():
    root = Path("tests/fixtures/span_relevance/v1")
    docs, cases, _ = load_span_fixture(root)
    assert len(cases) >= 12
    assert all(unicodedata.is_normalized("NFC", text) for text in docs.values())
    nfc_word = "\u0623\u064e"
    nfd_query = unicodedata.normalize("NFD", nfc_word)
    ranker = LexicalOverlapRanker()
    assert ranker.rank(nfd_query, nfc_word, [(0, len(nfc_word))])[0][0] > 0
    assert not (ranker.tokens(nfd_query) & ranker.tokens(nfc_word))


def test_ranker_ties_by_start():
    ranked = LexicalOverlapRanker().rank("alpha", "alpha xx alpha", [(9, 14), (0, 5)])
    assert [row[1] for row in ranked] == [0, 9]


def test_missing_quote_bad_occurrence_and_unknown_doc_raise(tmp_path):
    (tmp_path / "guide.md").write_text("one", encoding="utf-8")
    cases = [{"case_id": "x", "cohort": "en", "query": "q", "relevant": [{"doc": "guide.md", "quote": "missing"}]}]
    (tmp_path / "cases.json").write_text(json.dumps({"version": 1, "cases": cases}), encoding="utf-8")
    (tmp_path / "chunkers.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="x"):
        load_span_fixture(tmp_path)
    cases[0]["relevant"] = [{"doc": "guide.md", "quote": "one", "occurrence": 1}]
    (tmp_path / "cases.json").write_text(json.dumps({"version": 1, "cases": cases}), encoding="utf-8")
    with pytest.raises(ValueError, match="x"):
        load_span_fixture(tmp_path)
    cases[0]["relevant"] = [{"doc": "missing.md", "quote": "one"}]
    (tmp_path / "cases.json").write_text(json.dumps({"version": 1, "cases": cases}), encoding="utf-8")
    with pytest.raises(ValueError, match="x.*missing.md"):
        load_span_fixture(tmp_path)


def test_chunker_statuses_and_parent_expansion():
    root = Path("tests/fixtures/span_relevance/v1")
    docs, cases, chunkers = load_span_fixture(root)
    selected = tuple(c for c in chunkers if c["technique_id"] in {"C15", "C18"})
    registry = TechniqueRegistry(load_catalogue(Path("config/techniques.json")))
    register_builtin_plugins(registry)
    report = compare_chunkers(docs, cases[:1], selected, registry)
    assert any(r.label == "C15 model" and r.status == "requires_models" for r in report.rows)
    assert any("C18 parent" in r.label and r.status == "ok" for r in report.rows)
    assert any("parent expansion" in r.label and r.status == "ok" for r in report.rows)


def test_evidence_reproducible_without_writing_docs(tmp_path):
    command = [
        sys.executable,
        "-m",
        "rag_foundry_chunking.comparison",
        "tests/fixtures/span_relevance/v1",
        "--catalogue",
        "config/techniques.json",
    ]
    out1, out2 = tmp_path / "first", tmp_path / "second"
    subprocess.run([*command, "--out", str(out1)], check=True)
    subprocess.run([*command, "--out", str(out2)], check=True)
    names = ("span_comparison.json", "span_comparison.md")
    assert {n: (out1 / n).read_bytes() for n in names} == {n: (out2 / n).read_bytes() for n in names}
    committed = Path("docs/evidence/P4.5")
    assert {n: (out1 / n).read_bytes() for n in names} == {n: (committed / n).read_bytes() for n in names}, (
        "evidence drift; rerun the CLI"
    )
