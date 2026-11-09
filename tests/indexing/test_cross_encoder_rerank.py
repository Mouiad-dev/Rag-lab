import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from rag_foundry_domain.errors import IndexingError
from rag_foundry_indexing import register_builtin_plugins
from rag_foundry_indexing.cross_encoder_rerank import CrossEncoderRerankConfig, CrossEncoderReranker
from rag_foundry_providers.capabilities import Capabilities, ModelKind, ModelProfile
from rag_foundry_providers.errors import AdapterDependencyError
from rag_foundry_providers.messages import RerankItem, RerankResult
from rag_foundry_providers.sentence_transformers_rerank import SentenceTransformersCrossEncoder
from rag_foundry_providers.usage import Operation, TokenSource, Usage
from rag_foundry_techniques import (
    RetrievalResult,
    RetrievedChunk,
    ScoreComponent,
    TechniqueRegistry,
    TechniqueUnavailableError,
    load_catalogue,
)


class StubCrossEncoder:
    num_labels = 1
    activation_fn = SimpleNamespace()

    def __init__(self, model_dir, **kwargs):
        self.model_dir = model_dir
        self.kwargs = kwargs

    def predict(self, pairs, *, convert_to_numpy):
        return [0.5] * len(pairs)

    def tokenizer(self, queries, documents, *, truncation):
        return {"input_ids": [[0] for _ in documents]}


class FakeRerankerNotR19:
    def __init__(self, kind=ModelKind.RERANKER, *, mode="jaccard"):
        self.profile = ModelProfile(
            id="fake-rerank-test",
            provider="fake",
            model="not-r19-test-double",
            kind=kind,
            capabilities=Capabilities(local=True),
        )
        self.mode = mode
        self.calls = 0

    async def rerank(self, query, documents, *, top_n=None):
        self.calls += 1
        if self.mode == "invalid":
            items = (RerankItem(index=7, score=1.0),)
        else:
            q = set(query.split())
            scores = [(i, len(q & set(text.split())) / len(q | set(text.split()))) for i, text in enumerate(documents)]
            scores.sort(key=lambda pair: (-pair[1], pair[0]))
            items = tuple(RerankItem(index=i, score=score) for i, score in scores[:top_n])
        return RerankResult(
            items=items,
            usage=Usage(
                profile_id=self.profile.id,
                provider="fake",
                model=self.profile.model,
                operation=Operation.RERANK,
                input_tokens=1,
                output_tokens=0,
                token_source=TokenSource.ESTIMATED,
                latency_ms=0,
                estimated_cost_usd=0,
            ),
        )


def _result(texts):
    hits = tuple(
        RetrievedChunk(
            chunk_id=f"chunk-{i}",
            extraction_view_id="view",
            start=0,
            end=len(text),
            block_ids=(),
            text=text,
            distance=0.0,
            score=1.0,
            rank=i + 1,
            score_components=(ScoreComponent(name="input", value=1.0),),
            parent_start=None,
            parent_end=None,
        )
        for i, text in enumerate(texts)
    )
    return RetrievalResult(technique_id="R01", grounding="corpus", index_version_id="version", hits=hits)


def test_jaccard_double_order_and_changes():
    fake = FakeRerankerNotR19()
    output = asyncio.run(
        CrossEncoderReranker(CrossEncoderRerankConfig(k=2), reranker=fake).rerank(
            "cat", _result(["cat sat", "dog", "cat"])
        )
    )
    assert [hit.text for hit in output.result.hits] == ["cat", "cat sat"]
    assert [hit.score_components[-1].value for hit in output.result.hits] == [1.0, 0.5]
    assert output.trace.dropped == ("chunk-1",)
    assert [row.moved for row in output.changes[:2]] == [2, -1]
    assert len(output.trace.usage) == 1


def test_unavailable_wrong_kind_invalid_order_and_empty():
    with pytest.raises(TechniqueUnavailableError):
        asyncio.run(CrossEncoderReranker(CrossEncoderRerankConfig(), reranker=None).rerank("cat", _result(["cat"])))
    wrong = FakeRerankerNotR19(ModelKind.CHAT)
    with pytest.raises(IndexingError) as exc:
        asyncio.run(CrossEncoderReranker(CrossEncoderRerankConfig(), reranker=wrong).rerank("cat", _result(["cat"])))
    assert exc.value.code == "rerank_wrong_model_kind"
    invalid = FakeRerankerNotR19(mode="invalid")
    with pytest.raises(IndexingError) as exc:
        asyncio.run(CrossEncoderReranker(CrossEncoderRerankConfig(), reranker=invalid).rerank("cat", _result(["cat"])))
    assert exc.value.code == "rerank_invalid_order"
    empty = FakeRerankerNotR19()
    output = asyncio.run(CrossEncoderReranker(CrossEncoderRerankConfig(), reranker=empty).rerank("cat", _result([])))
    assert output.result.hits == () and output.trace.usage == () and empty.calls == 0


@pytest.mark.parametrize("mode", ["missing_dir", "missing_package", "wrong_labels"])
def test_adapter_dependency_becomes_unavailable_through_r19(tmp_path, monkeypatch, mode):
    model_dir = tmp_path / "model"
    if mode != "missing_dir":
        model_dir.mkdir()
    if mode == "missing_package":
        monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    elif mode == "wrong_labels":

        class StubMultilabelCrossEncoder(StubCrossEncoder):
            num_labels = 2

        monkeypatch.setitem(
            sys.modules, "sentence_transformers", SimpleNamespace(CrossEncoder=StubMultilabelCrossEncoder)
        )
    else:
        monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(CrossEncoder=StubCrossEncoder))
    reranker = SentenceTransformersCrossEncoder(model_dir)
    with pytest.raises(TechniqueUnavailableError):
        asyncio.run(CrossEncoderReranker(CrossEncoderRerankConfig(), reranker=reranker).rerank("q", _result(["a"])))


def test_duplicate_index_is_invalid_order():
    class DuplicateReranker(FakeRerankerNotR19):
        async def rerank(self, query, documents, *, top_n=None):
            result = await super().rerank(query, documents, top_n=top_n)
            return RerankResult(
                items=(RerankItem(index=0, score=0.9), RerankItem(index=0, score=0.8)), usage=result.usage
            )

    with pytest.raises(IndexingError) as exc:
        asyncio.run(
            CrossEncoderReranker(CrossEncoderRerankConfig(), reranker=DuplicateReranker()).rerank(
                "q", _result(["a", "b"])
            )
        )
    assert exc.value.code == "rerank_invalid_order"


def test_registration_and_empty_hits():
    catalogue_path = Path(__file__).parents[2] / "config" / "techniques.json"
    registry = TechniqueRegistry(load_catalogue(catalogue_path))
    register_builtin_plugins(registry)
    assert registry.validate_config("R19", {}) == CrossEncoderRerankConfig()
    empty = FakeRerankerNotR19()
    source = _result([])
    output = asyncio.run(CrossEncoderReranker(CrossEncoderRerankConfig(), reranker=empty).rerank("q", source))
    assert output.trace.status == "applied"
    assert output.result.index_version_id == source.index_version_id
    assert empty.calls == 0


def test_adapter_dependency_becomes_unavailable():
    class NotR19Unavailable(FakeRerankerNotR19):
        async def rerank(self, query, documents, *, top_n=None):
            raise AdapterDependencyError("sparse")

    with pytest.raises(TechniqueUnavailableError):
        asyncio.run(
            CrossEncoderReranker(CrossEncoderRerankConfig(), reranker=NotR19Unavailable()).rerank(
                "cat", _result(["cat"])
            )
        )
