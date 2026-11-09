"""Frozen, deterministic evaluation runs."""

import hashlib
import json
import re
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field
from rag_foundry_domain.indexing import IndexConfig
from rag_foundry_providers.ledger import UsageLedger, UsageTotals

from .dataset import EvalDataset, canonical_json, dataset_sha256
from .errors import EvaluationError
from .leakage import check_leakage, raise_on_errors
from .metrics import EvalHit, aggregate, case_metrics


@dataclass(frozen=True)
class BoundDocument:
    extraction_view_id: str
    text: str


def check_binding(dataset: EvalDataset, binding: dict[str, BoundDocument]) -> None:
    documents = {document.doc_key: document for document in dataset.documents}
    for key, document in documents.items():
        bound = binding.get(key)
        if bound is None:
            raise EvaluationError("binding_missing_document", key)
        digest = hashlib.sha256(bound.text.encode("utf-8")).hexdigest()
        if digest != document.text_sha256 or len(bound.text) != document.length:
            raise EvaluationError("binding_text_mismatch", key)
    views = [item.extraction_view_id for item in binding.values()]
    if len(views) != len(set(views)):
        raise EvaluationError("binding_duplicate_view")
    for case in dataset.cases:
        for span in case.relevant:
            bound = binding.get(span.doc_key)
            if bound is not None and bound.text[span.start : span.end] != span.quote:
                raise EvaluationError("binding_quote_mismatch", case.case_id)


async def index_fingerprint(scope, index, version, binding: dict[str, BoundDocument]) -> str:
    rows = []
    offset = 0
    while True:
        chunks = await index.list_chunks(scope, version.id, offset=offset, limit=100)
        for chunk in chunks:
            match = next(
                (doc_key for doc_key, bound in binding.items() if bound.extraction_view_id == chunk.extraction_view_id),
                None,
            )
            if match is None:
                raise EvaluationError("binding_unbound_view", chunk.extraction_view_id)
            rows.append([match, chunk.start, chunk.end, hashlib.sha256(chunk.text.encode("utf-8")).hexdigest()])
        offset += len(chunks)
        if len(chunks) < 100:
            break
    payload = {
        "index_config": version.model_dump(mode="json", include=set(IndexConfig.model_fields)),
        "chunks": sorted(rows),
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


class TechniqueSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    label: str = Field(pattern=r"^[A-Za-z0-9+._-]{1,40}$")
    technique_id: str
    config: dict


class ResolvedTechnique(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    label: str
    technique_id: str
    plugin_version: str
    config: dict


class RunConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    format: str = "rag-foundry.eval-run-config"
    format_version: str = "1"
    dataset_id: str
    dataset_version: str
    dataset_sha256: str
    corpus: tuple[tuple[str, str], ...]
    index_fingerprint: str
    code_version: str
    splits: tuple[str, ...]
    allow_held_out: bool
    ks: tuple[int, ...]
    coverage_threshold: float
    leakage_jaccard_threshold: float
    techniques: tuple[ResolvedTechnique, ...]


def build_run_config(
    dataset: EvalDataset,
    *,
    registry,
    techniques,
    index_fingerprint: str,
    code_version: str,
    splits=("dev", "validation"),
    ks=(1, 3, 5),
    coverage_threshold=0.8,
    leakage_jaccard_threshold=0.8,
    allow_held_out=False,
) -> RunConfig:
    if "held_out" in splits and not allow_held_out:
        raise EvaluationError("held_out_locked")
    resolved = []
    labels = set()
    for spec in techniques:
        if spec.label in labels:
            raise EvaluationError("duplicate_label", spec.label)
        labels.add(spec.label)
        model = registry.validate_config(spec.technique_id, spec.config)
        plugin = registry.plugin(spec.technique_id)
        resolved.append(
            ResolvedTechnique(
                label=spec.label,
                technique_id=spec.technique_id,
                plugin_version=plugin.manifest.version,
                config=model.model_dump(mode="json"),
            )
        )
    corpus = tuple(sorted((doc.doc_key, doc.text_sha256) for doc in dataset.documents))
    return RunConfig(
        dataset_id=dataset.dataset_id,
        dataset_version=dataset.dataset_version,
        dataset_sha256=dataset_sha256(dataset),
        corpus=corpus,
        index_fingerprint=index_fingerprint,
        code_version=code_version,
        splits=tuple(splits),
        allow_held_out=allow_held_out,
        ks=tuple(ks),
        coverage_threshold=coverage_threshold,
        leakage_jaccard_threshold=leakage_jaccard_threshold,
        techniques=tuple(resolved),
    )


def run_id(config: RunConfig) -> str:
    return "run-" + hashlib.sha256(canonical_json(config.model_dump(mode="json")).encode("utf-8")).hexdigest()[:16]


class StageSinks:
    def __init__(self):
        self._ledgers = {}

    def sink(self, stage: str) -> UsageLedger:
        if not re.fullmatch(r"[a-z_]{1,32}", stage):
            raise EvaluationError("invalid_stage", stage)
        return self._ledgers.setdefault(stage, UsageLedger())

    def totals(self) -> dict[str, UsageTotals]:
        return {stage: ledger.totals() for stage, ledger in self._ledgers.items()}


class RunOutput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    run_id: str
    config: RunConfig
    results_json: str
    timings: dict


async def run_experiment(
    dataset: EvalDataset,
    config: RunConfig,
    *,
    binding,
    registry,
    factory,
    scope,
    index,
    version,
    index_build_usage: UsageTotals | None = None,
    clock: Callable = time.perf_counter,
) -> RunOutput:
    if dataset_sha256(dataset) != config.dataset_sha256:
        raise EvaluationError("config_dataset_mismatch")
    check_binding(dataset, binding)
    if await index_fingerprint(scope, index, version, binding) != config.index_fingerprint:
        raise EvaluationError("config_index_mismatch")
    if "held_out" in config.splits and not config.allow_held_out:
        raise EvaluationError("held_out_locked")
    leakage = check_leakage(dataset, jaccard_threshold=config.leakage_jaccard_threshold)
    raise_on_errors(leakage)
    validated = {}
    for resolved in config.techniques:
        if resolved.plugin_version != registry.plugin(resolved.technique_id).manifest.version:
            raise EvaluationError("plugin_version_mismatch", resolved.label)
        validated[resolved.label] = registry.validate_config(resolved.technique_id, resolved.config)
    rid = run_id(config)
    result_rows, timing_rows, usage = [], {}, {}
    view_to_doc = {item.extraction_view_id: key for key, item in binding.items()}
    for resolved in config.techniques:
        sinks = StageSinks()
        retriever = factory(resolved, validated[resolved.label], sinks)
        rows = []
        timing_rows[resolved.label] = {}
        for case in dataset.cases:
            if case.split not in config.splits:
                continue
            status, error, hits = "ok", None, []
            before = clock()
            try:
                response = await retriever.retrieve(case.question)
                for hit in response.hits:
                    hits.append(
                        EvalHit(
                            doc_key=view_to_doc.get(hit.extraction_view_id),
                            start=hit.start,
                            end=hit.end,
                            rank=hit.rank,
                            score=hit.score,
                            text_sha256=hashlib.sha256(hit.text.encode("utf-8")).hexdigest(),
                        )
                    )
            except Exception as exc:
                status = "error"
                error = f"{type(exc).__name__}:{getattr(exc, 'code', '')}"
                hits = []
            after = clock()
            timing_rows[resolved.label][case.case_id] = {
                "total_ms": round((after - before) * 1000, 6),
                "ttft": "not_applicable",
            }
            metrics = case_metrics(case, hits, ks=config.ks, threshold=config.coverage_threshold)
            rows.append((case, metrics, status == "error"))
            result_rows.append(
                {
                    "case_id": case.case_id,
                    "label": resolved.label,
                    "hits": [hit.model_dump(mode="json") | {"score": round(hit.score, 6)} for hit in hits],
                    "metrics": _rounded(metrics.model_dump(mode="json")),
                    "status": status,
                    "error": error,
                }
            )
        usage[resolved.label] = {stage: totals.model_dump(mode="json") for stage, totals in sinks.totals().items()}
    payload = {
        "format": "rag-foundry.eval-run-results",
        "format_version": "1",
        "run_id": rid,
        "config": config.model_dump(mode="json"),
        "leakage": leakage.model_dump(mode="json"),
        "usage": usage,
        "index_build_usage": index_build_usage.model_dump(mode="json") if index_build_usage else None,
        "cases": result_rows,
        "summaries": {
            tech.label: [
                item.model_dump(mode="json")
                for item in aggregate(_rows_for_label(result_rows, dataset, tech.label, config), config.ks)
            ]
            for tech in config.techniques
        },
    }
    return RunOutput(
        run_id=rid,
        config=config,
        results_json=json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2) + "\n",
        timings=timing_rows,
    )


def _rounded(value):
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, dict):
        return {key: _rounded(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_rounded(item) for item in value]
    return value


def _rows_for_label(rows, dataset, label, config):
    cases = {case.case_id: case for case in dataset.cases}
    output = []
    for row in rows:
        if row["label"] != label:
            continue
        case = cases[row["case_id"]]
        hits = tuple(EvalHit.model_validate(hit) for hit in row["hits"])
        output.append(
            (
                case,
                case_metrics(case, hits, ks=config.ks, threshold=config.coverage_threshold),
                row["status"] == "error",
            )
        )
    return output


def git_code_version(repo_root) -> str:
    """Return the HEAD commit, with a "-dirty" suffix when `git status --porcelain` prints anything.

    Run ids are reproducible only on a clean tree. A "-dirty" suffix marks a run that is not reproducible:
    different uncommitted trees share the same suffix, and untracked files also set it.
    """
    root = str(repo_root)
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=root, check=True, capture_output=True, text=True
    ).stdout
    return head + ("-dirty" if dirty else "")
