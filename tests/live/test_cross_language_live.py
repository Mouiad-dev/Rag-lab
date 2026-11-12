"""Opt-in live R42 cross-language run: frozen runner, R02/R03/R06, bge-m3 dense retrieval on local Ollama.

Ollama embeddings are not guaranteed bit-identical across runs; this live run is not a frozen reproducible result.
One run, not an evaluation.
"""

import asyncio
import json
import os
from pathlib import Path

import pytest
from rag_foundry_core.settings import load_settings
from rag_foundry_evaluation import (
    TechniqueSpec,
    build_run_config,
    git_code_version,
    index_fingerprint,
    run_experiment,
)
from rag_foundry_evaluation.languages import render_language_pairs
from rag_foundry_providers.adapters.ollama import OllamaEmbeddingModel, ollama_embedding_profile
from rag_foundry_providers.ledger import RecordingEmbeddingModel, UsageLedger

from tests.evaluation._corpus import bound_index, make_factory
from tests.evaluation._cross_language import cross_language_dataset
from tests.evaluation.test_runner import make_registry

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(os.environ.get("RAG_LIVE_OLLAMA") != "1", reason="RAG_LIVE_OLLAMA=1 is not set"),
]

ROOT = Path(__file__).resolve().parents[2]
SPECS = (
    TechniqueSpec(label="R02", technique_id="R02", config={"k": 5}),
    TechniqueSpec(label="R03", technique_id="R03", config={"k": 5}),
    TechniqueSpec(label="R06", technique_id="R06", config={"k": 5, "branch_k": 10}),
)
EXPECTED_PAIRS = {"ar->ar", "ar->en", "ar->mixed", "ar->none", "en->ar", "en->en", "en->none", "mixed->ar"}


def test_cross_language_live_bge_m3():
    async def run():
        settings = load_settings()
        dataset = cross_language_dataset()
        registry = make_registry()
        embedder = OllamaEmbeddingModel(ollama_embedding_profile(model="bge-m3"), api_base=settings.ollama_base_url)
        build_ledger = UsageLedger()
        (owner, index, version), binding = await bound_index(RecordingEmbeddingModel(embedder, build_ledger))
        fingerprint = await index_fingerprint(owner, index, version, binding)
        config = build_run_config(
            dataset,
            registry=registry,
            techniques=SPECS,
            index_fingerprint=fingerprint,
            code_version=git_code_version(ROOT),
        )
        outcome = await run_experiment(
            dataset,
            config,
            binding=binding,
            registry=registry,
            factory=make_factory(owner, index, version, embedder),
            scope=owner,
            index=index,
            version=version,
            index_build_usage=build_ledger.totals(),
        )
        results = json.loads(outcome.results_json)
        rendered = render_language_pairs(results, dataset)
        for label in ("R02", "R03", "R06"):
            for case in dataset.cases:
                if case.split not in results["config"]["splits"]:
                    continue
                rows = [row for row in results["cases"] if row["case_id"] == case.case_id and row["label"] == label]
                assert len(rows) == 1, (label, case.case_id)
        for pair in EXPECTED_PAIRS:
            assert pair in rendered
        print(rendered)
        print(f"run_id={outcome.run_id}")
        print(f"index_build_usage={build_ledger.totals()}")

    asyncio.run(run())
