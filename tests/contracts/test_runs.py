"""Run v1 contract tests."""

import copy
import json
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError
from rag_foundry_contracts import RunRequest, RunResponse, RunStatus
from rag_foundry_contracts.export import render, schemas

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_DIR = ROOT / "packages" / "contracts" / "schemas"
FIXTURES = ROOT / "packages" / "contracts" / "fixtures" / "runs"


def _load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_bytes().decode("utf-8"))


def _answered() -> dict:
    return _load("response-answered.json")


def test_fixtures_validate():
    RunRequest.model_validate(_load("request.json"))
    assert RunResponse.model_validate(_answered()).status is RunStatus.ANSWERED
    assert RunResponse.model_validate(_load("response-abstained.json")).status is RunStatus.ABSTAINED
    assert RunResponse.model_validate(_load("response-ungrounded.json")).status is RunStatus.UNGROUNDED


def test_run_pins_hashes_round_trip():
    data = _answered()
    data["pins"]["settings_sha256"] = "a" * 64
    data["pins"]["prompt_sha256"] = "b" * 64
    model = RunResponse.model_validate(data)
    restored = RunResponse.model_validate_json(model.model_dump_json())
    assert restored.pins.settings_sha256 == "a" * 64
    assert restored.pins.prompt_sha256 == "b" * 64


def test_run_pins_reject_invalid_hashes():
    for field, value in (("settings_sha256", "a" * 63), ("prompt_sha256", "A" * 64)):
        data = _answered()
        data["pins"][field] = value
        with pytest.raises(ValidationError):
            RunResponse.model_validate(data)


@pytest.mark.parametrize("name", ["run-request.v1.json", "run-response.v1.json"])
def test_committed_schema_matches_models(name):
    committed = (SCHEMA_DIR / name).read_bytes().decode("utf-8")
    assert committed == render(schemas()[name]), f"{name} drifted; run the export module"


@pytest.mark.parametrize("question", [" \t\n", "x" * 2001])
def test_rejects_invalid_question(question):
    data = _load("request.json")
    data["question"] = question
    with pytest.raises(ValidationError):
        RunRequest.model_validate(data)


@pytest.mark.parametrize("k", [0, 21])
def test_rejects_k_out_of_bounds(k):
    with pytest.raises(ValidationError):
        RunRequest.model_validate({**_load("request.json"), "k": k})


@pytest.mark.parametrize("score", [1.5, float("nan")])
def test_rejects_invalid_min_score(score):
    with pytest.raises(ValidationError):
        RunRequest.model_validate({**_load("request.json"), "min_score": score})


@pytest.mark.parametrize("model", [RunRequest, RunResponse])
def test_accepts_minor_version_13(model):
    data = _load("request.json") if model is RunRequest else _answered()
    data["contract_version"] = "1.3"
    assert model.model_validate(data).contract_version == "1.3"


@pytest.mark.parametrize("field", ["score", "pins.min_score"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_rejects_non_finite_evidence_and_pin_scores(field, value):
    data = _answered()
    if field == "score":
        data["evidence"][0]["score"] = value
    else:
        data["pins"]["min_score"] = value
    with pytest.raises(ValidationError, match="Input should be a finite number"):
        RunResponse.model_validate(data)


@pytest.mark.parametrize("model", [RunRequest, RunResponse])
def test_rejects_major_version_2(model):
    data = _load("request.json") if model is RunRequest else _answered()
    data["contract_version"] = "2.0"
    with pytest.raises(ValidationError, match="unsupported contract major version 2"):
        model.model_validate(data)


def test_rejects_unknown_field():
    with pytest.raises(ValidationError):
        RunRequest.model_validate({**_load("request.json"), "extra": True})


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda d: d["evidence"].append(copy.deepcopy(d["evidence"][0])), "evidence markers must be unique"),
        (lambda d: d["evidence"][0].update(rank=2), "evidence marker must match its rank"),
        (lambda d: d.update(abstention_reason="low_score"), "answered response cannot have an abstention reason"),
        (lambda d: d.update(answer=" \t"), "answered response requires a non-blank answer"),
        (lambda d: d.update(citations=[]), "answered response requires citations"),
        (lambda d: d["citations"].append(copy.deepcopy(d["citations"][0])), "citation markers must be unique"),
    ],
)
def test_answered_invariants_reject_one_changed_field(change, message):
    data = _answered()
    change(data)
    with pytest.raises(ValidationError, match=message):
        RunResponse.model_validate(data)


def test_answered_citation_cannot_reference_unused_evidence():
    data = _answered()
    data["evidence"][0]["used"] = False
    with pytest.raises(ValidationError, match="citations must reference used evidence"):
        RunResponse.model_validate(data)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda d: d.update(abstention_reason=None), "abstained response requires a reason"),
        (lambda d: d.update(answer="leaked"), "abstained response cannot contain answer or citations"),
        (
            lambda d: d.update(citations=[_answered()["citations"][0]]),
            "abstained response cannot contain answer or citations",
        ),
    ],
)
def test_abstained_invariants_reject_one_changed_field(change, message):
    data = _load("response-abstained.json")
    change(data)
    with pytest.raises(ValidationError, match=message):
        RunResponse.model_validate(data)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda d: d.update(abstention_reason="low_score"), "ungrounded response cannot have an abstention reason"),
        (lambda d: d.update(answer=" \t"), "ungrounded response requires a non-blank answer"),
        (
            lambda d: d.update(citations=[_answered()["citations"][0]]),
            "ungrounded response cannot contain citations or evidence",
        ),
        (
            lambda d: d.update(evidence=[_answered()["evidence"][0]]),
            "ungrounded response cannot contain citations or evidence",
        ),
        (lambda d: d["pins"].update(index_version_id="idxv-1"), "ungrounded response cannot pin an index version"),
    ],
)
def test_ungrounded_invariants_reject_one_changed_field(change, message):
    data = _load("response-ungrounded.json")
    change(data)
    with pytest.raises(ValidationError, match=message):
        RunResponse.model_validate(data)


def test_warning_pattern_and_decimal_cost_round_trip():
    data = _answered()
    data["warnings"] = ["retry_used"]
    data["usage"]["estimated_cost_usd"] = "0.123456"
    model = RunResponse.model_validate(data)
    assert model.usage.estimated_cost_usd == Decimal("0.123456")
    restored = RunResponse.model_validate_json(model.model_dump_json())
    assert restored.usage.estimated_cost_usd == Decimal("0.123456")
    data["warnings"] = ["Bad-Code"]
    with pytest.raises(ValidationError):
        RunResponse.model_validate(data)
