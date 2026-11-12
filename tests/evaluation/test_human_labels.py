import os
from pathlib import Path

import pytest
from rag_foundry_evaluation.errors import EvaluationError
from rag_foundry_evaluation.human_labels import (
    HumanLabel,
    agreement,
    dump_human_labels,
    load_human_labels,
    placeholder_labels,
    render_agreement,
)
from rag_foundry_evaluation.judge import RUBRIC_ANSWER_SUPPORT_V1, JudgeVerdict, dump_judge_items, judge_items_sha256

from tests.evaluation._judge_items import judge_items_v1

ROOT = Path(__file__).resolve().parents[2]
ITEMS_PATH = ROOT / "tests/evaluation/data/judge_items_v1.json"
LABELS_PATH = ROOT / "tests/evaluation/data/human_labels_answer_support_v1.json"
RUBRIC = RUBRIC_ANSWER_SUPPORT_V1
S, P, U = "supported", "partially_supported", "unsupported"


def _verdict(item_id: str, label: str | None, status: str = "ok", **override) -> JudgeVerdict:
    fields = {
        "item_id": item_id,
        "rubric_id": RUBRIC.rubric_id,
        "rubric_version": RUBRIC.rubric_version,
        "prompt_sha256": RUBRIC.prompt_sha256(),
        "judge_profile_id": "test-judge",
        "judge_model": "test-model",
        "mode": "complete",
        "status": status,
        "label": label,
    }
    fields.update(override)
    return JudgeVerdict(**fields)


def _human(item_id: str, label: str, annotator: str = "human-a") -> HumanLabel:
    return HumanLabel(item_id=item_id, annotator_id=annotator, label=label, labeled_on="2026-01-01", source="human")


def _pairs(pairs, annotator: str = "human-a"):
    # synthetic labels for arithmetic only; not human data
    verdicts, labels = [], []
    for index, (judge, human) in enumerate(pairs):
        item_id = f"x-{index:02d}"
        verdicts.append(_verdict(item_id, judge))
        labels.append(_human(item_id, human, annotator))
    return verdicts, labels


def _one(pairs):
    verdicts, labels = _pairs(pairs)
    (report,) = agreement(verdicts, labels, rubric=RUBRIC)
    return report


def _expand(spec):
    return [pair for pair, count in spec for _ in range(count)]


def _write(tmp_path, **changes):
    items = judge_items_v1()
    data = placeholder_labels(items, RUBRIC).model_dump(mode="json")
    data.update(changes)
    import json

    path = tmp_path / "labels.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_committed_files_equal_dumps():
    items = judge_items_v1()
    placeholder = placeholder_labels(items, RUBRIC)
    if os.getenv("RAG_WRITE_EVIDENCE") == "1":
        ITEMS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with ITEMS_PATH.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(dump_judge_items(items))
        with LABELS_PATH.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(dump_human_labels(placeholder))
        return
    assert ITEMS_PATH.read_bytes().decode("ascii") == dump_judge_items(items)
    assert LABELS_PATH.read_bytes().decode("ascii") == dump_human_labels(placeholder)
    loaded = load_human_labels(LABELS_PATH, items=items, rubric=RUBRIC)
    assert loaded.status == "placeholder_not_labeled"
    assert loaded.labels == ()
    assert loaded.items_sha256 == judge_items_sha256(items)


def test_placeholder_agreement_is_not_yet_measured():
    verdicts = [_verdict(item.item_id, S) for item in judge_items_v1().items]
    reports = agreement(verdicts, (), rubric=RUBRIC)
    assert len(reports) == 1
    report = reports[0]
    assert report.status == "not_yet_measured"
    assert report.annotator_id is None
    assert report.n_unlabeled == 6
    assert report.percent_agreement is None
    assert report.cohens_kappa is None
    assert report.kappa_status == "not_computed"
    text = render_agreement(reports)
    assert "Judge agreement: not yet measured (0 human labels).\n" in text
    assert "Judge labels are evaluator output, not ground truth." in text


def test_binary_case():
    report = _one(_expand((((S, S), 4), ((S, U), 1), ((U, S), 1), ((U, U), 4))))
    assert report.status == "measured"
    assert report.percent_agreement == 0.8
    assert report.cohens_kappa == 0.6
    assert report.kappa_status == "defined"


def test_three_classes():
    spec = (((S, S), 3), ((S, P), 1), ((P, P), 2), ((U, U), 2), ((U, S), 1), ((P, U), 1))
    report = _one(_expand(spec))
    assert report.percent_agreement == 0.7
    assert report.cohens_kappa == 0.545455
    assert report.confusion == (
        ("partially_supported", "partially_supported", 2),
        ("partially_supported", "unsupported", 1),
        ("supported", "partially_supported", 1),
        ("supported", "supported", 3),
        ("unsupported", "supported", 1),
        ("unsupported", "unsupported", 2),
    )


def test_chance_agreement_one():
    report = _one(_expand((((S, S), 3),)))
    assert report.percent_agreement == 1.0
    assert report.cohens_kappa is None
    assert report.kappa_status == "undefined_chance_agreement_1"
    assert "undefined_chance_agreement_1" in render_agreement([report])


def test_exclusions():
    verdicts = [
        _verdict("a-1", S),
        _verdict("a-2", U),
        _verdict("a-3", None, status="invalid_output"),
        _verdict("a-4", S),
    ]
    labels = [_human("a-1", S), _human("a-2", S), _human("a-3", S)]
    (report,) = agreement(verdicts, labels, rubric=RUBRIC)
    assert (report.n_compared, report.n_judge_invalid, report.n_unlabeled) == (2, 1, 1)
    invalid = [_verdict("a-1", None, status="invalid_output"), _verdict("a-2", S)]
    (report,) = agreement(invalid, [_human("a-1", S)], rubric=RUBRIC)
    assert report.status == "no_comparable_items"
    assert report.n_compared == 0
    assert report.percent_agreement is None
    assert report.kappa_status == "not_computed"


def test_two_annotators_sorted():
    verdicts = [_verdict("a-1", S), _verdict("a-2", U)]
    labels = [_human("a-1", S, "human-zed"), _human("a-2", S, "human-zed"), _human("a-1", S, "human-amy")]
    reports = agreement(verdicts, labels, rubric=RUBRIC)
    assert [r.annotator_id for r in reports] == ["human-amy", "human-zed"]
    assert reports[0].n_unlabeled == 1
    assert reports[1].percent_agreement == 0.5


def test_verdict_rubric_mismatch():
    for override in ({"rubric_version": "2"}, {"rubric_id": "other"}, {"prompt_sha256": "0" * 64}):
        with pytest.raises(EvaluationError) as info:
            agreement([_verdict("a-1", S, **override)], [_human("a-1", S)], rubric=RUBRIC)
        assert info.value.code == "verdict_rubric_mismatch"


def test_in_memory_duplicate_and_orphan_labels_rejected():
    verdicts = [_verdict("a-1", S)]
    with pytest.raises(EvaluationError) as info:
        agreement(verdicts, [_human("a-1", S), _human("a-1", S)], rubric=RUBRIC)
    assert info.value.code == "invalid_human_labels"
    with pytest.raises(EvaluationError) as info:
        agreement(verdicts, [_human("a-1", S), _human("a-9", S)], rubric=RUBRIC)
    assert info.value.code == "label_unknown_item"


def _label_dict(**changes):
    data = {
        "item_id": "j-01",
        "annotator_id": "human-a",
        "label": S,
        "labeled_on": "2026-01-01",
        "source": "human",
    }
    data.update(changes)
    return data


@pytest.mark.parametrize(
    ("changes", "code"),
    [
        ({"labels": [_label_dict(source="llm_judge")], "status": "labeled"}, "invalid_human_labels"),
        ({"labels": [_label_dict(annotator_id="gemma")], "status": "labeled"}, "invalid_human_labels"),
        ({"labels": [_label_dict()]}, "invalid_human_labels"),
        ({"status": "labeled"}, "invalid_human_labels"),
        ({"labels": [_label_dict(item_id="j-99")], "status": "labeled"}, "label_unknown_item"),
        ({"labels": [_label_dict(label="maybe")], "status": "labeled"}, "label_unknown_value"),
        ({"rubric_version": "2"}, "label_rubric_mismatch"),
        ({"items_sha256": "0" * 64}, "label_items_mismatch"),
        ({"labels": [_label_dict(), _label_dict(label=U)], "status": "labeled"}, "invalid_human_labels"),
    ],
)
def test_loader_rejections(tmp_path, changes, code):
    path = _write(tmp_path, **changes)
    with pytest.raises(EvaluationError) as info:
        load_human_labels(path, items=judge_items_v1(), rubric=RUBRIC)
    assert info.value.code == code


def test_loader_accepts_valid_labeled_set(tmp_path):
    path = _write(tmp_path, labels=[_label_dict()], status="labeled")
    loaded = load_human_labels(path, items=judge_items_v1(), rubric=RUBRIC)
    assert loaded.labels[0].item_id == "j-01"


def test_loader_rejects_duplicate_key_and_nan(tmp_path):
    for text in ('{"format": "a", "format": "b"}', '{"labels": NaN}'):
        path = tmp_path / "bad.json"
        path.write_text(text, encoding="utf-8")
        with pytest.raises(EvaluationError) as info:
            load_human_labels(path, items=judge_items_v1(), rubric=RUBRIC)
        assert info.value.code == "invalid_human_labels"


def test_item_ids_are_neutral():
    banned = ("faithful", "wrong", "flip", "partial", "unrelated", "supported", "unsupported")
    for item in judge_items_v1().items:
        assert not any(word in item.item_id for word in banned)
