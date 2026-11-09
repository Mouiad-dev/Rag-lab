import asyncio
import hashlib
import json
import re
import shutil
from pathlib import Path

import pytest
from pydantic import ValidationError
from rag_foundry_domain.errors import IndexingError
from rag_foundry_evaluation import (
    BoundDocument,
    EvaluationError,
    RunConfig,
    StageSinks,
    TechniqueSpec,
    build_run_config,
    check_binding,
    git_code_version,
    index_fingerprint,
    run_experiment,
    run_id,
)
from rag_foundry_indexing import register_builtin_plugins
from rag_foundry_providers.ledger import RecordingEmbeddingModel, UsageLedger
from rag_foundry_techniques import TechniqueRegistry, load_catalogue

from tests.evaluation._corpus import DOC_KEY, bound_index, build_dataset, make_factory
from tests.indexing._lexical_embedder import LexicalTestEmbedder
from tests.indexing._paired_fixture import SPANS

ROOT = Path(__file__).resolve().parents[2]
ROWS = (
    ("en-fact", "en", (), "dev"),
    ("id-exact", "en", ("identifier",), "dev"),
    ("ar-fact", "ar", (), "validation"),
    ("unanswerable", "en", (), "validation"),
)
LABELS = ("R02", "R03", "R06")


class FakeClock:
    def __init__(self, step):
        self._value = 0.0
        self._step = step

    def __call__(self):
        self._value += self._step
        return self._value


def make_registry():
    registry = TechniqueRegistry(load_catalogue(ROOT / "config/techniques.json"))
    register_builtin_plugins(registry)
    return registry


def specs(r03_k=5):
    return (
        TechniqueSpec(label="R02", technique_id="R02", config={"k": 5}),
        TechniqueSpec(label="R03", technique_id="R03", config={"k": r03_k}),
        TechniqueSpec(label="R06", technique_id="R06", config={"k": 5, "branch_k": 10}),
    )


async def build(spans=SPANS, with_usage=False):
    embedder = LexicalTestEmbedder()
    ledger = UsageLedger()
    used = RecordingEmbeddingModel(embedder, ledger) if with_usage else embedder
    (owner, index, version), binding = await bound_index(used, spans)
    return embedder, ledger, owner, index, version, binding


async def execute(
    *,
    dataset=None,
    r03_k=5,
    code_version="code-1",
    step=1.0,
    spans=SPANS,
    with_usage=False,
    config_override=None,
    fingerprint_spans=None,
    calls=None,
    rows=ROWS,
):
    dataset = dataset or build_dataset(rows)
    registry = make_registry()
    embedder, ledger, owner, index, version, binding = await build(spans, with_usage)
    if fingerprint_spans is None:
        fingerprint = await index_fingerprint(owner, index, version, binding)
    else:
        _, _, f_owner, f_index, f_version, f_binding = await build(fingerprint_spans)
        fingerprint = await index_fingerprint(f_owner, f_index, f_version, f_binding)
    config = config_override or build_run_config(
        dataset,
        registry=registry,
        techniques=specs(r03_k),
        index_fingerprint=fingerprint,
        code_version=code_version,
    )
    output = await run_experiment(
        dataset,
        config,
        binding=binding,
        registry=registry,
        factory=make_factory(owner, index, version, embedder, calls),
        scope=owner,
        index=index,
        version=version,
        index_build_usage=ledger.totals() if with_usage else None,
        clock=FakeClock(step),
    )
    chunks = await index.list_chunks(owner, version.id, limit=100)
    ids = {str(version.id), str(binding[DOC_KEY].extraction_view_id)} | {str(chunk.id) for chunk in chunks}
    return output, ids, config


def run(coro):
    return asyncio.run(coro)


def test_rerun_on_two_builds_is_byte_identical():
    first, ids1, _ = run(execute(step=1.0))
    second, ids2, _ = run(execute(step=2.5))
    assert first.run_id == second.run_id
    assert re.fullmatch(r"run-[0-9a-f]{16}", first.run_id)
    assert first.results_json == second.results_json
    assert first.timings != second.timings
    assert first.timings["R02"]["en-fact"]["ttft"] == "not_applicable"
    for ids in (ids1, ids2):
        assert len(ids) > 2
        assert not [item for item in ids if item in first.results_json]
    for forbidden in ("total_ms", "ttft", "timings", "elapsed"):
        assert forbidden not in first.results_json
    data = json.loads(first.results_json)
    assert data["format"] == "rag-foundry.eval-run-results"
    assert data["run_id"] == first.run_id
    assert len(data["cases"]) == 12
    assert {(row["case_id"], row["label"]) for row in data["cases"]} == {
        (case_id, label) for case_id, *_ in ROWS for label in LABELS
    }
    for row in data["cases"]:
        assert row["status"] == "ok"
        if row["case_id"] != "unanswerable":
            assert row["hits"], (row["case_id"], row["label"])
    assert set(data["summaries"]) == set(LABELS)


def test_fingerprint_equal_across_fresh_builds_and_excludes_random_fields():
    async def go():
        a = await build()
        b = await build()
        fa = await index_fingerprint(a[2], a[3], a[4], a[5])
        fb = await index_fingerprint(b[2], b[3], b[4], b[5])
        shorter = await build(SPANS[:-1])
        fc = await index_fingerprint(shorter[2], shorter[3], shorter[4], shorter[5])
        return a, b, fa, fb, fc

    a, b, fa, fb, fc = run(go())
    assert fa == fb
    assert re.fullmatch(r"[0-9a-f]{64}", fa)
    assert fa != fc
    # The random per-build fields really differ, so equality above proves they are excluded.
    assert a[4].id != b[4].id
    assert a[4].extraction_view_ids != b[4].extraction_view_ids


def test_run_id_and_results_change_with_frozen_inputs():
    base, _, _ = run(execute())
    k3, _, _ = run(execute(r03_k=3))
    assert base.run_id != k3.run_id
    assert base.results_json != k3.results_json
    assert run(execute(code_version="code-2"))[0].run_id != base.run_id
    dataset = build_dataset(ROWS)
    cases = list(dataset.cases)
    span = cases[0].relevant[0]
    cases[0] = cases[0].model_copy(update={"relevant": (span.model_copy(update={"grade": 2}),)})
    graded = dataset.model_copy(update={"cases": tuple(cases)})
    assert run(execute(dataset=graded))[0].run_id != base.run_id


def test_allow_held_out_flag_is_part_of_run_id():
    dataset = build_dataset(ROWS)
    registry = make_registry()

    def config(allow):
        return build_run_config(
            dataset,
            registry=registry,
            techniques=specs(),
            index_fingerprint="f" * 64,
            code_version="c",
            splits=("dev",),
            allow_held_out=allow,
        )

    assert run_id(config(True)) != run_id(config(False))


def test_r03_exact_identifier_has_rr_one():
    output, _, _ = run(execute())
    rows = {(row["case_id"], row["label"]): row for row in json.loads(output.results_json)["cases"]}
    assert rows[("id-exact", "R03")]["metrics"]["rr"] == 1.0


def test_usage_is_per_technique_and_stage_and_index_build_is_separate():
    output, _, _ = run(execute(with_usage=True))
    data = json.loads(output.results_json)
    assert data["usage"]["R02"]["query_embedding"]["calls"] == 4
    assert data["usage"]["R03"] == {}
    assert data["usage"]["R06"]["query_embedding"]["calls"] == 4
    assert isinstance(data["usage"]["R02"]["query_embedding"]["cost_usd"], str)
    build_usage = data["index_build_usage"]
    assert build_usage["calls"] == len(SPANS) == 19
    assert isinstance(build_usage["cost_usd"], str)
    assert "index_build_usage" not in data["usage"]["R02"]
    plain, _, _ = run(execute())
    assert json.loads(plain.results_json)["index_build_usage"] is None


def test_stage_sinks_validate_names_and_share_ledger_per_stage():
    sinks = StageSinks()
    assert sinks.sink("query_embedding") is sinks.sink("query_embedding")
    assert sinks.sink("other") is not sinks.sink("query_embedding")
    with pytest.raises(EvaluationError) as caught:
        sinks.sink("Bad Stage")
    assert caught.value.code == "invalid_stage"
    assert set(sinks.totals()) == {"query_embedding", "other"}


def test_run_id_format():
    output, _, config = run(execute())
    assert re.fullmatch(r"run-[0-9a-f]{16}", run_id(config))
    assert output.run_id == run_id(config)


def test_label_must_match_pattern():
    with pytest.raises(ValueError):
        TechniqueSpec(label="bad label", technique_id="R03", config={})
    with pytest.raises(ValueError):
        TechniqueSpec(label="x" * 41, technique_id="R03", config={})


def test_build_run_config_errors():
    dataset = build_dataset(ROWS)
    registry = make_registry()
    common = {"registry": registry, "index_fingerprint": "f" * 64, "code_version": "c"}
    with pytest.raises(EvaluationError) as caught:
        build_run_config(
            dataset,
            techniques=(
                TechniqueSpec(label="A", technique_id="R03", config={}),
                TechniqueSpec(label="A", technique_id="R02", config={}),
            ),
            **common,
        )
    assert caught.value.code == "duplicate_label"
    with pytest.raises(EvaluationError) as caught:
        build_run_config(dataset, techniques=specs(), splits=("dev", "held_out"), **common)
    assert caught.value.code == "held_out_locked"
    with pytest.raises(ValidationError):
        build_run_config(dataset, techniques=(TechniqueSpec(label="A", technique_id="R03", config={"k": 0}),), **common)
    config = build_run_config(dataset, techniques=specs(), splits=("dev", "held_out"), allow_held_out=True, **common)
    assert config.allow_held_out is True


@pytest.mark.parametrize("bad", [{"k": 0}, {"unknown_field": 1}])
def test_invalid_technique_config_raises_before_any_factory_call(bad):
    calls = []
    dataset = build_dataset(ROWS)
    registry = make_registry()
    technique = TechniqueSpec(label="R03", technique_id="R03", config=bad)

    async def go():
        embedder, _, owner, index, version, binding = await build()
        fingerprint = await index_fingerprint(owner, index, version, binding)
        common = {"registry": registry, "index_fingerprint": fingerprint, "code_version": "c"}
        with pytest.raises(ValidationError):
            build_run_config(dataset, techniques=(technique,), **common)
        # A RunConfig built directly bypasses build_run_config; run_experiment still validates before the factory.
        good = build_run_config(dataset, techniques=specs(), **common)
        tech = list(good.model_dump()["techniques"])
        tech[1]["config"] = bad  # R03 is the second technique, so R02 would reach the factory first
        direct = RunConfig(**(good.model_dump() | {"techniques": tech}))
        with pytest.raises(ValidationError):
            await run_experiment(
                dataset,
                direct,
                binding=binding,
                registry=registry,
                factory=make_factory(owner, index, version, embedder, calls),
                scope=owner,
                index=index,
                version=version,
            )

    run(go())
    assert calls == []


def test_direct_run_config_with_held_out_is_locked_in_run_experiment():
    dataset = build_dataset(ROWS)
    calls = []

    async def go():
        registry = make_registry()
        embedder, _, owner, index, version, binding = await build()
        fingerprint = await index_fingerprint(owner, index, version, binding)
        good = build_run_config(
            dataset, registry=registry, techniques=specs(), index_fingerprint=fingerprint, code_version="c"
        )
        direct = RunConfig(**(good.model_dump() | {"splits": ("dev", "held_out"), "allow_held_out": False}))
        with pytest.raises(EvaluationError) as caught:
            await run_experiment(
                dataset,
                direct,
                binding=binding,
                registry=registry,
                factory=make_factory(owner, index, version, embedder, calls),
                scope=owner,
                index=index,
                version=version,
            )
        return caught.value.code

    assert run(go()) == "held_out_locked"
    assert calls == []


def test_index_mismatch_is_detected_before_any_factory_call():
    calls = []
    with pytest.raises(EvaluationError) as caught:
        run(execute(spans=SPANS[:-1], fingerprint_spans=SPANS, calls=calls))
    assert caught.value.code == "config_index_mismatch"
    assert calls == []


def test_dataset_changed_after_freezing_is_rejected():
    dataset = build_dataset(ROWS)
    changed = dataset.model_copy(update={"description": "edited"})
    calls = []

    async def go():
        registry = make_registry()
        embedder, _, owner, index, version, binding = await build()
        fingerprint = await index_fingerprint(owner, index, version, binding)
        config = build_run_config(
            dataset, registry=registry, techniques=specs(), index_fingerprint=fingerprint, code_version="c"
        )
        await run_experiment(
            changed,
            config,
            binding=binding,
            registry=registry,
            factory=make_factory(owner, index, version, embedder, calls),
            scope=owner,
            index=index,
            version=version,
        )

    with pytest.raises(EvaluationError) as caught:
        run(go())
    assert caught.value.code == "config_dataset_mismatch"
    assert calls == []


def test_leaking_dataset_raises_before_any_factory_call():
    dataset = build_dataset(ROWS)
    cases = list(dataset.cases)
    dev_question = cases[0].question
    cases[2] = cases[2].model_copy(update={"question": dev_question})
    leaking = dataset.model_copy(update={"cases": tuple(cases)})
    calls = []
    with pytest.raises(EvaluationError) as caught:
        run(execute(dataset=leaking, calls=calls))
    assert caught.value.code == "leakage"
    assert calls == []


def test_plugin_version_mismatch_raises_before_any_factory_call():
    dataset = build_dataset(ROWS)
    calls = []

    async def go():
        registry = make_registry()
        embedder, _, owner, index, version, binding = await build()
        fingerprint = await index_fingerprint(owner, index, version, binding)
        good = build_run_config(
            dataset, registry=registry, techniques=specs(), index_fingerprint=fingerprint, code_version="c"
        )
        tech = list(good.model_dump()["techniques"])
        tech[2]["plugin_version"] = "0.0.0-other"
        stale = RunConfig(**(good.model_dump() | {"techniques": tech}))
        await run_experiment(
            dataset,
            stale,
            binding=binding,
            registry=registry,
            factory=make_factory(owner, index, version, embedder, calls),
            scope=owner,
            index=index,
            version=version,
        )

    with pytest.raises(EvaluationError) as caught:
        run(go())
    assert caught.value.code == "plugin_version_mismatch"
    assert calls == []


def test_binding_errors_and_no_factory_call():
    dataset = build_dataset(ROWS)
    good = BoundDocument(extraction_view_id="v1", text=_doc())
    check_binding(dataset, {DOC_KEY: good})

    def code(binding):
        with pytest.raises(EvaluationError) as caught:
            check_binding(dataset, binding)
        return caught.value.code

    assert code({}) == "binding_missing_document"
    changed = _doc()[:-1] + ("x" if _doc()[-1] != "x" else "y")
    assert code({DOC_KEY: BoundDocument("v1", changed)}) == "binding_text_mismatch"
    assert code({DOC_KEY: BoundDocument("v1", _doc() + " ")}) == "binding_text_mismatch"
    other = dataset.model_copy(
        update={"documents": dataset.documents + (dataset.documents[0].model_copy(update={"doc_key": "second"}),)}
    )
    with pytest.raises(EvaluationError) as caught:
        check_binding(other, {DOC_KEY: good, "second": BoundDocument("v1", _doc())})
    assert caught.value.code == "binding_duplicate_view"
    case = dataset.cases[0]
    bad_span = case.relevant[0].model_copy(update={"quote": case.relevant[0].quote + "!"})
    quote_dataset = dataset.model_copy(
        update={"cases": (case.model_copy(update={"relevant": (bad_span,)}),) + dataset.cases[1:]}
    )
    with pytest.raises(EvaluationError) as caught:
        check_binding(quote_dataset, {DOC_KEY: good})
    assert caught.value.code == "binding_quote_mismatch"

    calls = []

    async def go():
        registry = make_registry()
        embedder, _, owner, index, version, binding = await build()
        fingerprint = await index_fingerprint(owner, index, version, binding)
        config = build_run_config(
            dataset, registry=registry, techniques=specs(), index_fingerprint=fingerprint, code_version="c"
        )
        tampered = {DOC_KEY: BoundDocument(binding[DOC_KEY].extraction_view_id, changed)}
        await run_experiment(
            dataset,
            config,
            binding=tampered,
            registry=registry,
            factory=make_factory(owner, index, version, embedder, calls),
            scope=owner,
            index=index,
            version=version,
        )

    with pytest.raises(EvaluationError) as caught:
        run(go())
    assert caught.value.code == "binding_text_mismatch"
    assert calls == []


def test_unbound_view_chunk_is_rejected_by_fingerprint():
    async def go():
        _, _, owner, index, version, _ = await build()
        await index_fingerprint(owner, index, version, {DOC_KEY: BoundDocument("other-view", _doc())})

    with pytest.raises(EvaluationError) as caught:
        run(go())
    assert caught.value.code == "binding_unbound_view"


def test_retriever_error_is_recorded_and_run_continues():
    dataset = build_dataset(ROWS)

    class Failing:
        async def retrieve(self, question):
            if question == dataset.cases[1].question:
                raise IndexingError("empty_query")
            return await self.inner.retrieve(question)

    async def go():
        registry = make_registry()
        embedder, _, owner, index, version, binding = await build()
        fingerprint = await index_fingerprint(owner, index, version, binding)
        config = build_run_config(
            dataset, registry=registry, techniques=specs(), index_fingerprint=fingerprint, code_version="c"
        )
        real = make_factory(owner, index, version, embedder)

        def factory(resolved, validated, sinks):
            wrapper = Failing()
            wrapper.inner = real(resolved, validated, sinks)
            return wrapper

        return await run_experiment(
            dataset,
            config,
            binding=binding,
            registry=registry,
            factory=factory,
            scope=owner,
            index=index,
            version=version,
            clock=FakeClock(1.0),
        )

    data = json.loads(run(go()).results_json)
    failed = [row for row in data["cases"] if row["case_id"] == "id-exact"]
    assert len(failed) == 3
    for row in failed:
        assert row["status"] == "error"
        assert row["error"] == "IndexingError:empty_query"
        assert row["hits"] == []
    others = [row for row in data["cases"] if row["case_id"] != "id-exact"]
    assert len(others) == 9 and all(row["status"] == "ok" for row in others)
    for label in LABELS:
        assert sum(item["n_errors"] for item in data["summaries"][label] if item["group"] == "all") == 1


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
def test_git_code_version_format():
    assert re.fullmatch(r"[0-9a-f]{40}(-dirty)?", git_code_version(ROOT))


def _doc():
    from tests.indexing._paired_fixture import DOC

    return DOC


def test_doc_hash_matches_dataset_document():
    dataset = build_dataset(ROWS)
    assert dataset.documents[0].text_sha256 == hashlib.sha256(_doc().encode("utf-8")).hexdigest()
