import asyncio
import json
from pathlib import Path

import pytest
from rag_foundry_domain.errors import IndexingError
from rag_foundry_indexing import MultiQueryExpansion, RagFusion, RagFusionConfig, register_builtin_plugins
from rag_foundry_indexing.fusion import check_inputs
from rag_foundry_techniques import TechniqueRegistry, load_catalogue

from tests.chunking._scripted_models import ScriptedChat
from tests.indexing._recording_retriever import RecordingRetriever


def _chat(queries):
    return ScriptedChat(lambda _request: (json.dumps({"queries": queries}), "stop"))


def test_rrf_score_and_r18_ranks_shared_hits_over_one_best_hit():
    async def check():
        q = "q"
        variants = ["v1", "v2"]
        retriever = RecordingRetriever({q: ("single", "shared"), "v1": ("shared",), "v2": ("shared",)})
        out = await RagFusion().run(q, chat=_chat(variants), retriever=retriever)
        assert retriever.queries == [q, *variants]
        shared = next(hit for hit in out.result.hits if hit.chunk_id == "shared")
        single = next(hit for hit in out.result.hits if hit.chunk_id == "single")
        assert shared.score == 1 / 62 + 1 / 61 + 1 / 61
        assert shared.rank == 1 and single.rank == 2
        assert out.result.scoring == "rag-fusion-rrf-k60"
        assert [(c.name, c.value) for c in shared.score_components] == [
            ("q0_rrf", 1 / 62),
            ("q1_rrf", 1 / 61),
            ("q2_rrf", 1 / 61),
        ]
        registry = TechniqueRegistry(load_catalogue(Path(__file__).resolve().parents[2] / "config/techniques.json"))
        register_builtin_plugins(registry)
        assert registry.validate_config("R18", {}) == RagFusionConfig()

    asyncio.run(check())


def test_rrf_exact_hand_computation():
    async def check():
        q = "q"
        retriever = RecordingRetriever({q: ("x", "y", "target"), "v1": ("target",), "v2": ()})
        out = await RagFusion().run(q, chat=_chat(["v1", "v2"]), retriever=retriever)
        target = next(hit for hit in out.result.hits if hit.chunk_id == "target")
        assert target.score == 1 / 61 + 1 / 63

    asyncio.run(check())


def test_r14_r18_ranking_difference_and_input_validation():
    async def check():
        q = "q"
        variants = ["v1", "v2"]
        inputs = {q: ("single", "shared"), "v1": ("filler1", "shared"), "v2": ("filler2", "shared")}
        r14 = await MultiQueryExpansion().run(q, chat=_chat(variants), retriever=RecordingRetriever(inputs))
        r18 = await RagFusion().run(q, chat=_chat(variants), retriever=RecordingRetriever(inputs))
        r14_ids = [h.chunk_id for h in r14.result.hits]
        r18_ids = [h.chunk_id for h in r18.result.hits]
        assert r14_ids.index("single") < r14_ids.index("shared")
        assert r18_ids.index("shared") < r18_ids.index("single")

        base = RecordingRetriever({"q": ("c",)})
        one = await base.retrieve("q")
        mixed = one.model_copy(update={"index_version_id": "v2"})
        with pytest.raises(IndexingError, match="fusion_version_mismatch"):
            check_inputs((one, mixed), distinct_branches=False, min_branches=1)
        gaps = one.model_copy(
            update={
                "hits": (
                    one.hits[0].model_copy(update={"rank": 1}),
                    one.hits[0].model_copy(update={"chunk_id": "d", "rank": 3}),
                )
            }
        )
        with pytest.raises(IndexingError, match="fusion_invalid_ranks"):
            check_inputs((gaps,), distinct_branches=False, min_branches=1)
        duplicates = one.model_copy(update={"hits": (one.hits[0], one.hits[0].model_copy(update={"rank": 2}))})
        with pytest.raises(IndexingError, match="fusion_duplicate_candidate"):
            check_inputs((duplicates,), distinct_branches=False, min_branches=1)

    asyncio.run(check())


@pytest.mark.parametrize(
    ("mode", "code"),
    [
        ("version", "fusion_version_mismatch"),
        ("ranks", "fusion_invalid_ranks"),
        ("duplicate", "fusion_duplicate_candidate"),
    ],
)
def test_retrieve_all_checks_through_public_run(mode, code):
    class BrokenRetriever:
        def __init__(self):
            self.base = RecordingRetriever({"q": ("a", "b"), "v": ("c",)})

        async def retrieve(self, query):
            result = await self.base.retrieve(query)
            if mode == "version" and query == "v":
                return result.model_copy(update={"index_version_id": "v2"})
            if mode == "ranks" and query == "q":
                return result.model_copy(
                    update={"hits": (result.hits[0], result.hits[1].model_copy(update={"rank": 3}))}
                )
            if mode == "duplicate" and query == "q":
                return result.model_copy(
                    update={"hits": (result.hits[0], result.hits[0].model_copy(update={"rank": 2}))}
                )
            return result

    async def check():
        for plugin in (MultiQueryExpansion(), RagFusion()):
            with pytest.raises(IndexingError, match=code):
                await plugin.run("q", chat=_chat(["v"]), retriever=BrokenRetriever())

    asyncio.run(check())
