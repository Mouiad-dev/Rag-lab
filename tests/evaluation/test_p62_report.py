import asyncio
import datetime
import json
import os
import re
from pathlib import Path

from rag_foundry_evaluation import (
    TechniqueSpec,
    build_run_config,
    git_code_version,
    index_fingerprint,
    render_markdown,
    run_experiment,
)
from rag_foundry_indexing.context_expansion import SourceBlockRef, ViewSource
from rag_foundry_providers.ledger import RecordingEmbeddingModel, UsageLedger

from tests.evaluation._corpus import DOC, DOC_KEY, bound_index, c18_spans, make_factory, sentence_spans
from tests.evaluation._seed import seed_dataset
from tests.evaluation.test_runner import make_registry
from tests.indexing._lexical_embedder import LexicalTestEmbedder
from tests.indexing._source_reader import StaticSourceReader

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "docs/evidence/P6.2/seed_context_runs.md"
HEADER_PREFIX = "<!-- generated "
R02 = TechniqueSpec(label="R02", technique_id="R02", config={"k": 5})
RUN_A = (
    R02,
    TechniqueSpec(label="R21", technique_id="R21", config={"window": 1, "stay_in_paragraph": True}),
    TechniqueSpec(label="R21-cross", technique_id="R21", config={"window": 1, "stay_in_paragraph": False}),
)
RUN_B = (
    R02,
    TechniqueSpec(label="R22", technique_id="R22", config={}),
    TechniqueSpec(label="R25", technique_id="R25", config={}),
    TechniqueSpec(label="R25-min1", technique_id="R25", config={"min_children": 1}),
)
COMMON_NOTES = (
    "The R02 base retriever uses the test-only hashed bag-of-words embedder (`LexicalTestEmbedder`), not a semantic "
    "model.",
    "n is tiny: dev 5 answerable cases, validation 2 answerable and 1 unanswerable case. This is a tiny fixture, "
    "not a benchmark; no difference here is a measured win.",
)
TAIL_NOTES = (
    "The base retriever config (R02, k=5) is fixed in the test factory and is NOT part of RunConfig; the run id "
    "does not cover it.",
    "Each run uses one index; the compared labels change one factor at a time. The held_out split was not run.",
    "The code version is the working tree at generation. A -dirty suffix means the tree had uncommitted changes "
    "when this file was generated.",
)
NOTES_A = (
    *COMMON_NOTES,
    "Seed paragraphs are single sentences, so R21 with `stay_in_paragraph` can equal R02 (window 1 stays inside "
    "one paragraph).",
    "`R21-cross` only adds neighbour paragraphs (it lets the window cross paragraph boundaries).",
    *TAIL_NOTES,
)
NOTES_B = (
    *COMMON_NOTES,
    "The index holds C18 children (parent_size 200, child_size 50); R22 and R25 read the stored parent spans.",
    "R25-min1 equals R25 on this fixture: the two labels return the same hits and the same context sizes.",
    *TAIL_NOTES,
)


async def execute(dataset, specs, spans, code_version):
    registry = make_registry()
    embedder = LexicalTestEmbedder()
    ledger = UsageLedger()
    (owner, index, version), binding = await bound_index(RecordingEmbeddingModel(embedder, ledger), spans)
    view_id = binding[DOC_KEY].extraction_view_id
    source = StaticSourceReader(
        {
            (owner.tenant_id, owner.project_id, view_id): ViewSource(
                extraction_view_id=view_id,
                text=DOC,
                blocks=(SourceBlockRef(block_id="b00000", start=0, end=len(DOC)),),
            )
        }
    )
    fingerprint = await index_fingerprint(owner, index, version, binding)
    config = build_run_config(
        dataset, registry=registry, techniques=specs, index_fingerprint=fingerprint, code_version=code_version
    )
    return await run_experiment(
        dataset,
        config,
        binding=binding,
        registry=registry,
        factory=make_factory(owner, index, version, embedder, source=source),
        scope=owner,
        index=index,
        version=version,
        index_build_usage=ledger.totals(),
    )


def context_table(results, dataset):
    splits = {case.case_id: case.split for case in dataset.cases}
    labels = [tech["label"] for tech in results["config"]["techniques"]]
    lines = [
        "### Context size (returned code points, `end - start` summed over hits)",
        "",
        "| label | split | cases | hits | sum | mean per hit |",
        "|---|---|---|---|---|---|",
    ]
    for label in labels:
        for split in ("dev", "validation"):
            rows = [c for c in results["cases"] if c["label"] == label and splits[c["case_id"]] == split]
            sizes = [hit["end"] - hit["start"] for row in rows for hit in row["hits"]]
            mean = "-" if not sizes else f"{sum(sizes) / len(sizes):.2f}"
            lines.append(f"| {label} | {split} | {len(rows)} | {len(sizes)} | {sum(sizes)} | {mean} |")
    return "\n".join(lines) + "\n"


def render_run(dataset, specs, spans, title, notes, code_version):
    first = asyncio.run(execute(dataset, specs, spans, code_version))
    second = asyncio.run(execute(dataset, specs, spans, code_version))
    assert first.results_json == second.results_json
    results = json.loads(first.results_json)
    by_label = {}
    for case in results["cases"]:
        by_label.setdefault(case["label"], {})[case["case_id"]] = case["hits"]
    if "R25-min1" in by_label:  # the run B note claims this; fail if it stops being true
        assert by_label["R25-min1"] == by_label["R25"]
    body = render_markdown(results, title=title, notes=notes, dataset=dataset)
    return body.rstrip("\n") + "\n\n" + context_table(results, dataset)


def render(code_version):
    dataset = seed_dataset()
    run_a = render_run(
        dataset, RUN_A, sentence_spans(), "P6.2 run A (sentence index): R02, R21, R21-cross", NOTES_A, code_version
    )
    run_b = render_run(
        dataset, RUN_B, c18_spans(), "P6.2 run B (C18 child index): R02, R22, R25, R25-min1", NOTES_B, code_version
    )
    return run_a + "\n" + run_b


def test_seed_context_runs_are_current():
    if os.getenv("RAG_WRITE_EVIDENCE") == "1":
        text = render(git_code_version(ROOT))
        header = (
            f"{HEADER_PREFIX}{datetime.date.today().isoformat()} by "
            "`RAG_WRITE_EVIDENCE=1 uv run pytest tests/evaluation/test_p62_report.py`; "
            "offline, test-only lexical embedder, tiny fixture, not a benchmark -->\n"
        )
        EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
        with EVIDENCE.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(header + text)
        return
    committed = EVIDENCE.read_text(encoding="utf-8").replace("\r\n", "\n")
    assert committed.startswith(HEADER_PREFIX)
    body = committed.split("\n", 1)[1]
    version = re.search(r"^- Code version: (\S+)$", body, re.MULTILINE).group(1)
    assert render(version) == body
