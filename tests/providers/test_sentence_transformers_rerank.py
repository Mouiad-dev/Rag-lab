import asyncio
import sys
from types import SimpleNamespace

import pytest
from rag_foundry_providers.errors import AdapterDependencyError
from rag_foundry_providers.sentence_transformers_rerank import SentenceTransformersCrossEncoder
from rag_foundry_providers.usage import Operation


class NotR19StubCrossEncoder:
    num_labels = 1
    activation_fn = SimpleNamespace()
    init_calls = []

    def __init__(self, model_dir, **kwargs):
        type(self).init_calls.append((model_dir, kwargs))

    def predict(self, pairs, *, convert_to_numpy):
        assert convert_to_numpy is True
        return [0.2, 0.9, 0.2][: len(pairs)]

    def tokenizer(self, queries, documents, *, truncation):
        assert truncation is True
        return {"input_ids": [[0] * 4, [0] * 5, [0] * 6][: len(documents)]}


def test_order_tokens_usage_and_top_n(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(CrossEncoder=NotR19StubCrossEncoder))
    adapter = SentenceTransformersCrossEncoder(tmp_path)
    result = asyncio.run(adapter.rerank("q", ["a", "b", "c"]))
    assert [(item.index, item.score) for item in result.items] == [(1, 0.9), (0, 0.2), (2, 0.2)]
    assert result.usage.input_tokens == 15
    assert result.usage.operation is Operation.RERANK
    assert len(asyncio.run(adapter.rerank("q", ["a", "b", "c"], top_n=1)).items) == 1
    _, kwargs = NotR19StubCrossEncoder.init_calls[-1]
    assert kwargs["local_files_only"] is True
    assert kwargs["trust_remote_code"] is False


def test_whitespace_token_fallback(tmp_path, monkeypatch):
    class NotR19WithoutTokenizer(NotR19StubCrossEncoder):
        tokenizer = None

    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(CrossEncoder=NotR19WithoutTokenizer))
    result = asyncio.run(SentenceTransformersCrossEncoder(tmp_path).rerank("a b", ["c", "d e"]))
    assert result.usage.input_tokens == 7


@pytest.mark.parametrize("mode", ["missing_dir", "missing_package", "wrong_labels"])
def test_dependency_failures(tmp_path, monkeypatch, mode):
    model_dir = tmp_path / "model"
    if mode != "missing_dir":
        model_dir.mkdir()
    if mode == "missing_package":
        monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    if mode == "wrong_labels":

        class NotR19Multilabel(NotR19StubCrossEncoder):
            num_labels = 2

        monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(CrossEncoder=NotR19Multilabel))
    with pytest.raises(AdapterDependencyError):
        asyncio.run(SentenceTransformersCrossEncoder(model_dir).rerank("q", ["a"]))


def test_empty_documents_do_not_load_model(tmp_path, monkeypatch):
    class CountingStubCrossEncoder(NotR19StubCrossEncoder):
        predict_calls = 0

        def predict(self, pairs, *, convert_to_numpy):
            type(self).predict_calls += 1
            return super().predict(pairs, convert_to_numpy=convert_to_numpy)

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(CrossEncoder=CountingStubCrossEncoder))
    result = asyncio.run(SentenceTransformersCrossEncoder(model_dir).rerank("q", []))
    assert result.items == ()
    assert result.usage.input_tokens == 0
    assert CountingStubCrossEncoder.predict_calls == 0
