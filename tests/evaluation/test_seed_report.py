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
from rag_foundry_providers.ledger import RecordingEmbeddingModel, UsageLedger

from tests.evaluation._corpus import bound_index, make_factory
from tests.evaluation._seed import seed_dataset
from tests.evaluation.test_runner import make_registry
from tests.indexing._lexical_embedder import LexicalTestEmbedder

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "tests/evaluation/data/seed_fixture_v1.json"
EVIDENCE = ROOT / "docs/evidence/P6.1/seed_fixture_run.md"
TITLE = "P6.1 seed fixture run: R02, R03, R06"
NOTES = (
    "R02 here uses the test-only hashed bag-of-words embedder (`LexicalTestEmbedder`), not a semantic model.",
    "n is tiny: dev 5 answerable cases, validation 2 answerable and 1 unanswerable case.",
    "The table and OCR cohorts are absent (n=0).",
    "The 2 held_out cases were not run; held_out stays locked.",
    "R04 is absent (it needs PostgreSQL).",
    "The code version is the working tree at generation. A -dirty suffix means the tree had uncommitted changes when "
    "this file was generated, so the run id is not reproducible from that commit alone.",
)
SPECS = (
    TechniqueSpec(label="R02", technique_id="R02", config={"k": 5}),
    TechniqueSpec(label="R03", technique_id="R03", config={"k": 5}),
    TechniqueSpec(label="R06", technique_id="R06", config={"k": 5, "branch_k": 10}),
)


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
    dataset = seed_dataset()
    first = asyncio.run(execute(dataset, code_version))
    second = asyncio.run(execute(dataset, code_version))
    assert first.results_json == second.results_json
    return render_markdown(json.loads(first.results_json), title=TITLE, notes=NOTES, dataset=dataset)


def test_committed_json_equals_seed_dataset():
    assert DATA.read_bytes().decode("ascii") == dump_dataset(seed_dataset())
    assert load_dataset(DATA) == seed_dataset()


def test_seed_has_no_leakage_findings():
    report = check_leakage(seed_dataset())
    assert report.errors == ()
    assert report.findings == ()


def test_seed_report_is_current():
    if os.getenv("RAG_WRITE_EVIDENCE") == "1":
        text = render(git_code_version(ROOT))
        EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
        with EVIDENCE.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
        return
    committed = EVIDENCE.read_text(encoding="utf-8").replace("\r\n", "\n")
    version = re.search(r"^- Code version: (\S+)$", committed, re.MULTILINE).group(1)
    assert render(version) == committed
