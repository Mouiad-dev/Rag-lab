import asyncio
import json
import os
import re
from pathlib import Path

from rag_foundry_evaluation import (
    TechniqueSpec,
    build_run_config,
    check_leakage,
    dump_dataset,
    git_code_version,
    index_fingerprint,
    load_dataset,
    render_markdown,
    run_experiment,
)
from rag_foundry_evaluation.languages import language_pair, render_language_pairs
from rag_foundry_providers.ledger import RecordingEmbeddingModel, UsageLedger

from tests.evaluation._corpus import bound_index, make_factory
from tests.evaluation._cross_language import cross_language_dataset
from tests.evaluation.test_runner import make_registry
from tests.indexing._lexical_embedder import LexicalTestEmbedder

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "tests/evaluation/data/cross_language_fixture_v1.json"
EVIDENCE = ROOT / "docs/evidence/P6.4/cross_language_pairs.md"
TITLE = "P6.4 cross-language fixture run: R02, R03, R06"
NOTES = (
    "R02 uses the test-only hashed bag-of-words embedder (`LexicalTestEmbedder`), so it can only match shared "
    "tokens; no semantic cross-language match is possible offline.",
    "Expect cross-script cases to fail, except where a token is shared (AB-1234, or the Arabic word in the mixed "
    "question).",
    "A shared token only makes a hit possible, not certain: in the mixed question the shared English words can pull "
    "the English-heavy MIXED paragraph above the Arabic one. Read the per-case table for what each technique found.",
    "The document language is script-detected, and Latin script is labelled en.",
    "n is tiny and all 9 cases are dev-only; no tuning is done on this fixture. It is not a benchmark.",
    "The code version is the working tree at generation. A -dirty suffix means the tree had uncommitted changes when "
    "this file was generated, so the run id is not reproducible from that commit alone.",
    "The real cross-language check needs a multilingual embedder: the live bge-m3 run in P6.4-04.",
)
SPECS = (
    TechniqueSpec(label="R02", technique_id="R02", config={"k": 5}),
    TechniqueSpec(label="R03", technique_id="R03", config={"k": 5}),
    TechniqueSpec(label="R06", technique_id="R06", config={"k": 5, "branch_k": 10}),
)
EXPECTED_PAIRS = {"ar->ar", "ar->en", "ar->mixed", "ar->none", "en->ar", "en->en", "en->none", "mixed->ar"}


async def execute(dataset, code_version):
    registry = make_registry()
    embedder = LexicalTestEmbedder()
    ledger = UsageLedger()
    (owner, index, version), binding = await bound_index(RecordingEmbeddingModel(embedder, ledger))
    fingerprint = await index_fingerprint(owner, index, version, binding)
    config = build_run_config(
        dataset, registry=registry, techniques=SPECS, index_fingerprint=fingerprint, code_version=code_version
    )
    return await run_experiment(
        dataset,
        config,
        binding=binding,
        registry=registry,
        factory=make_factory(owner, index, version, embedder),
        scope=owner,
        index=index,
        version=version,
        index_build_usage=ledger.totals(),
    )


def render(code_version):
    dataset = cross_language_dataset()
    first = asyncio.run(execute(dataset, code_version))
    second = asyncio.run(execute(dataset, code_version))
    assert first.results_json == second.results_json
    results = json.loads(first.results_json)
    return (
        render_markdown(results, title=TITLE, notes=NOTES, dataset=dataset)
        + "\n"
        + render_language_pairs(results, dataset)
    )


def test_committed_json_equals_dataset():
    if os.getenv("RAG_WRITE_EVIDENCE") == "1":
        with DATA.open("w", encoding="ascii", newline="\n") as stream:
            stream.write(dump_dataset(cross_language_dataset()))
    assert DATA.read_bytes().decode("ascii") == dump_dataset(cross_language_dataset())
    assert load_dataset(DATA) == cross_language_dataset()


def test_no_leakage_findings():
    report = check_leakage(cross_language_dataset())
    assert report.errors == ()
    assert report.findings == ()


def test_pairs():
    cases = {case.case_id: case for case in cross_language_dataset().cases}
    assert {language_pair(case) for case in cases.values()} == EXPECTED_PAIRS
    assert language_pair(cases["xl-en-ar"]) == "en->ar"
    assert language_pair(cases["xl-en-ar-id"]) == "en->ar"


def test_cross_language_report_is_current():
    if os.getenv("RAG_WRITE_EVIDENCE") == "1":
        text = render(git_code_version(ROOT))
        EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
        with EVIDENCE.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
        return
    committed = EVIDENCE.read_text(encoding="utf-8").replace("\r\n", "\n")
    version = re.search(r"^- Code version: (\S+)$", committed, re.MULTILINE).group(1)
    assert render(version) == committed
