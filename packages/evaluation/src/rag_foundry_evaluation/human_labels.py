"""Human labels for a judge item set, and judge-human agreement (percent, Cohen kappa)."""

import json
from collections import Counter
from collections.abc import Sequence
from fractions import Fraction
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .dataset import KEY, _reject_constant, _unique_object
from .errors import EvaluationError
from .judge import JudgeItemSet, JudgeRubric, JudgeVerdict, judge_items_sha256

_PLACEHOLDER_NOTE = "No human has labeled these items yet. Do not fill this file with model output or author guesses."


class _FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class HumanLabel(_FrozenModel):
    item_id: str = Field(pattern=KEY)
    annotator_id: str = Field(pattern=r"^human-[a-z0-9-]{1,32}$")
    label: str
    labeled_on: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    source: Literal["human"]


class HumanLabelSet(_FrozenModel):
    format: Literal["rag-foundry.human-labels"]
    format_version: Literal["1"]
    rubric_id: str = Field(pattern=KEY)
    rubric_version: str = Field(pattern=r"^[0-9]+$")
    items_id: str = Field(pattern=KEY)
    items_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: Literal["placeholder_not_labeled", "labeled"]
    note: str = Field(max_length=500)
    labels: tuple[HumanLabel, ...]

    @model_validator(mode="after")
    def consistent(self) -> "HumanLabelSet":
        if self.status == "placeholder_not_labeled" and self.labels:
            raise ValueError("a placeholder must have no labels")
        if self.status == "labeled" and not self.labels:
            raise ValueError("status labeled requires at least one label")
        pairs = [(label.item_id, label.annotator_id) for label in self.labels]
        if len(set(pairs)) != len(pairs):
            raise ValueError("duplicate (item_id, annotator_id) pair")
        return self


def load_human_labels(path: str | Path, *, items: JudgeItemSet, rubric: JudgeRubric) -> HumanLabelSet:
    try:
        text = Path(path).read_text(encoding="utf-8", errors="strict")
        json.loads(text, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
        data = HumanLabelSet.model_validate_json(text, strict=True)
    except ValidationError as exc:
        error = exc.errors()[0]
        reason = f"{'.'.join(map(str, error['loc']))}: {error['msg']} [{error['type']}]"
        raise EvaluationError("invalid_human_labels", reason) from exc
    except Exception as exc:
        reason = str(exc).splitlines()[0][:200] or type(exc).__name__
        raise EvaluationError("invalid_human_labels", reason) from exc
    if data.rubric_id != rubric.rubric_id or data.rubric_version != rubric.rubric_version:
        raise EvaluationError("label_rubric_mismatch")
    if data.items_id != items.items_id or data.items_sha256 != judge_items_sha256(items):
        raise EvaluationError("label_items_mismatch")
    known = {item.item_id for item in items.items}
    for label in data.labels:
        if label.item_id not in known:
            raise EvaluationError("label_unknown_item", label.item_id)
        if label.label not in rubric.labels:
            raise EvaluationError("label_unknown_value", label.item_id)
    return data


def placeholder_labels(items: JudgeItemSet, rubric: JudgeRubric) -> HumanLabelSet:
    return HumanLabelSet(
        format="rag-foundry.human-labels",
        format_version="1",
        rubric_id=rubric.rubric_id,
        rubric_version=rubric.rubric_version,
        items_id=items.items_id,
        items_sha256=judge_items_sha256(items),
        status="placeholder_not_labeled",
        note=_PLACEHOLDER_NOTE,
        labels=(),
    )


def dump_human_labels(labels: HumanLabelSet) -> str:
    return json.dumps(labels.model_dump(mode="json"), ensure_ascii=True, sort_keys=True, indent=2) + "\n"


class AgreementReport(_FrozenModel):
    rubric_id: str
    rubric_version: str
    prompt_sha256: str
    annotator_id: str | None
    status: Literal["not_yet_measured", "no_comparable_items", "measured"]
    n_items: int
    n_compared: int
    n_judge_invalid: int
    n_unlabeled: int
    percent_agreement: float | None
    cohens_kappa: float | None
    kappa_status: Literal["defined", "undefined_chance_agreement_1", "not_computed"]
    confusion: tuple[tuple[str, str, int], ...]


def agreement(
    verdicts: Sequence[JudgeVerdict], labels: Sequence[HumanLabel], *, rubric: JudgeRubric
) -> tuple[AgreementReport, ...]:
    verdicts = tuple(verdicts)
    sha = rubric.prompt_sha256()
    for verdict in verdicts:
        if (verdict.rubric_id, verdict.rubric_version, verdict.prompt_sha256) != (
            rubric.rubric_id,
            rubric.rubric_version,
            sha,
        ):
            raise EvaluationError("verdict_rubric_mismatch", verdict.item_id)
    base = {"rubric_id": rubric.rubric_id, "rubric_version": rubric.rubric_version, "prompt_sha256": sha}
    labels = tuple(labels)
    if not labels:
        return (
            AgreementReport(
                **base,
                annotator_id=None,
                status="not_yet_measured",
                n_items=len(verdicts),
                n_compared=0,
                n_judge_invalid=0,
                n_unlabeled=len(verdicts),
                percent_agreement=None,
                cohens_kappa=None,
                kappa_status="not_computed",
                confusion=(),
            ),
        )
    verdict_ids = {verdict.item_id for verdict in verdicts}
    by_annotator: dict[str, dict[str, str]] = {}
    for label in labels:
        if label.label not in rubric.labels:
            raise EvaluationError("label_unknown_value", label.item_id)
        if label.item_id not in verdict_ids:
            raise EvaluationError("label_unknown_item", label.item_id)
        given = by_annotator.setdefault(label.annotator_id, {})
        if label.item_id in given:
            # The loader rejects duplicates too; in-memory labels must not overwrite silently.
            raise EvaluationError("invalid_human_labels", f"duplicate {label.item_id}/{label.annotator_id}")
        given[label.item_id] = label.label
    reports = []
    for annotator in sorted(by_annotator):
        given = by_annotator[annotator]
        pairs: list[tuple[str, str]] = []
        n_invalid = 0
        n_unlabeled = 0
        for verdict in verdicts:
            human = given.get(verdict.item_id)
            if human is None:
                n_unlabeled += 1
            elif verdict.status == "ok" and verdict.label is not None:
                pairs.append((verdict.label, human))
            else:
                # invalid_output, or an ok verdict without a label: not comparable.
                n_invalid += 1
        n = len(pairs)
        counts = {
            "n_items": len(verdicts),
            "n_compared": n,
            "n_judge_invalid": n_invalid,
            "n_unlabeled": n_unlabeled,
        }
        if n == 0:
            reports.append(
                AgreementReport(
                    **base,
                    annotator_id=annotator,
                    status="no_comparable_items",
                    **counts,
                    percent_agreement=None,
                    cohens_kappa=None,
                    kappa_status="not_computed",
                    confusion=(),
                )
            )
            continue
        p_o = Fraction(sum(1 for j, h in pairs if j == h), n)
        judge_counts = Counter(j for j, _ in pairs)
        human_counts = Counter(h for _, h in pairs)
        p_e = sum((Fraction(judge_counts[c], n) * Fraction(human_counts[c], n) for c in rubric.labels), Fraction(0))
        kappa: float | None = None
        kappa_status: Literal["defined", "undefined_chance_agreement_1"] = "undefined_chance_agreement_1"
        if p_e != 1:
            kappa = round(float((p_o - p_e) / (1 - p_e)), 6)
            kappa_status = "defined"
        confusion = tuple((j, h, c) for (j, h), c in sorted(Counter(pairs).items()))
        reports.append(
            AgreementReport(
                **base,
                annotator_id=annotator,
                status="measured",
                **counts,
                percent_agreement=round(float(p_o), 6),
                cohens_kappa=kappa,
                kappa_status=kappa_status,
                confusion=confusion,
            )
        )
    return tuple(reports)


def render_agreement(reports: Sequence[AgreementReport]) -> str:
    reports = tuple(reports)
    lines: list[str] = []
    if any(report.status == "not_yet_measured" for report in reports):
        lines.append("Judge agreement: not yet measured (0 human labels).")
    else:
        lines.append("| annotator | n_compared | n_judge_invalid | n_unlabeled | percent | kappa | kappa_status |")
        lines.append("|---|---|---|---|---|---|---|")
        for r in reports:
            percent = "-" if r.percent_agreement is None else f"{r.percent_agreement:.4f}"
            kappa = "-" if r.cohens_kappa is None else f"{r.cohens_kappa:.4f}"
            lines.append(
                f"| {r.annotator_id} | {r.n_compared} | {r.n_judge_invalid} | {r.n_unlabeled} | "
                f"{percent} | {kappa} | {r.kappa_status} |"
            )
    lines.append("")
    lines.append("Judge labels are evaluator output, not ground truth.")
    return "\n".join(lines) + "\n"
