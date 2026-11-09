import json

import pytest
from rag_foundry_evaluation import EvaluationError, dataset_sha256, dump_dataset, load_dataset


def dataset() -> dict[str, object]:
    return {
        "format": "rag-foundry.eval-dataset",
        "format_version": "1",
        "dataset_id": "sample",
        "dataset_version": "1.0",
        "description": "Small evaluation fixture",
        "documents": [
            {"doc_key": "doc-a", "text_sha256": "a" * 64, "length": 20},
            {"doc_key": "doc-b", "text_sha256": "b" * 64, "length": 20},
        ],
        "cases": [
            {
                "case_id": "english",
                "question": "What is recorded?",
                "language": "en",
                "tags": [],
                "split": "dev",
                "answerable": True,
                "relevant": [{"doc_key": "doc-a", "start": 0, "end": 4, "grade": 1, "quote": "Tide"}],
            },
            {
                "case_id": "arabic",
                "question": "\u0643\u0645 \u0645\u062e\u0637\u0648\u0637\u0629\u061f",
                "language": "ar",
                "tags": ["paraphrase"],
                "split": "validation",
                "answerable": True,
                "relevant": [{"doc_key": "doc-b", "start": 0, "end": 4, "grade": 2, "quote": "Text"}],
            },
            {
                "case_id": "unanswerable",
                "question": "Who recorded it?",
                "language": "en",
                "split": "held_out",
                "answerable": False,
                "relevant": [],
            },
        ],
    }


def write_json(tmp_path, value: object, name: str = "dataset.json"):
    path = tmp_path / name
    path.write_text(json.dumps(value, ensure_ascii=True), encoding="utf-8")
    return path


def test_round_trip_and_stable_hash(tmp_path):
    path = write_json(tmp_path, dataset())
    loaded = load_dataset(path)
    dumped = tmp_path / "dump.json"
    dumped.write_text(dump_dataset(loaded), encoding="utf-8")
    assert load_dataset(dumped) == loaded
    assert dataset_sha256(load_dataset(path)) == dataset_sha256(load_dataset(dumped))


def test_grade_changes_hash(tmp_path):
    first = load_dataset(write_json(tmp_path, dataset(), "first.json"))
    changed = dataset()
    changed["cases"][0]["relevant"][0]["grade"] = 3
    second = load_dataset(write_json(tmp_path, changed, "second.json"))
    assert dataset_sha256(first) != dataset_sha256(second)


@pytest.mark.parametrize(
    "mutation, expected_substring",
    [
        ("unknown_field", "extra_forbidden"),
        ("duplicate_json_key", "duplicate JSON key"),
        ("nan", "invalid JSON constant"),
        ("answerable_empty", "need spans"),
        ("unanswerable_span", "need none"),
        ("end_too_large", "exceeds document length"),
        ("unknown_doc", "unknown doc_key"),
        ("quote_mismatch", "quote length"),
        ("duplicate_case_id", "case_id values must be unique"),
        ("bad_split", "literal_error"),
        ("overlap", "must not overlap"),
        ("unsorted_tags", "sorted and unique"),
        ("bad_version", "literal_error"),
        ("string_grade", "int_type"),
    ],
)
def test_invalid_datasets_raise_evaluation_error(tmp_path, mutation, expected_substring):
    value = dataset()
    raw = None
    case = value["cases"][0]
    span = case["relevant"][0]
    if mutation == "unknown_field":
        value["surprise"] = True
    elif mutation == "duplicate_json_key":
        raw = json.dumps(value, ensure_ascii=True)[:-1] + ',"format":"rag-foundry.eval-dataset"}'
    elif mutation == "nan":
        raw = json.dumps(value, ensure_ascii=True).replace('"length": 20', '"length": NaN', 1)
    elif mutation == "answerable_empty":
        case["relevant"] = []
    elif mutation == "unanswerable_span":
        value["cases"][2]["relevant"] = [span.copy()]
    elif mutation == "end_too_large":
        span["end"] = 21
        span["quote"] = "x" * (21 - span["start"])
    elif mutation == "unknown_doc":
        span["doc_key"] = "missing"
    elif mutation == "quote_mismatch":
        span["quote"] = "x"
    elif mutation == "duplicate_case_id":
        value["cases"][1]["case_id"] = "english"
    elif mutation == "bad_split":
        case["split"] = "test"
    elif mutation == "overlap":
        case["relevant"].append({"doc_key": "doc-a", "start": 5, "end": 12, "grade": 1, "quote": "1234567"})
        span["start"] = 0
        span["end"] = 10
        span["quote"] = "0123456789"
    elif mutation == "unsorted_tags":
        case["tags"] = ["paraphrase", "identifier"]
    elif mutation == "bad_version":
        value["format_version"] = "2"
    elif mutation == "string_grade":
        span["grade"] = "1"
    path = tmp_path / "invalid.json"
    if raw is None:
        path.write_text(json.dumps(value, ensure_ascii=True), encoding="utf-8")
    else:
        path.write_text(raw, encoding="utf-8")
    with pytest.raises(EvaluationError) as caught:
        load_dataset(path)
    assert caught.value.code == "invalid_dataset"
    assert expected_substring in caught.value.detail
    assert str(caught.value) == f"invalid_dataset: {caught.value.detail}"
