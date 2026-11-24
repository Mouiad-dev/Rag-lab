import asyncio
import math
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
from _vocab_sparse_encoder import VocabSparseEncoder
from pydantic import ValidationError
from rag_foundry_domain.errors import IndexingError
from rag_foundry_domain.indexing import IndexState
from rag_foundry_domain.testing import make_scope
from rag_foundry_indexing import register_builtin_plugins
from rag_foundry_indexing.sparse import LearnedSparseRetriever, SparseConfig, SparseIndexService, sparse_encoder_key
from rag_foundry_providers.messages import SparseVector
from rag_foundry_techniques import TechniqueRegistry, load_catalogue
from rag_foundry_techniques.errors import TechniqueUnavailableError


@pytest.mark.parametrize(
    "terms,weights", [(("a", "a"), (1.0, 2.0)), (("a",), (0.0,)), (("a",), (math.inf,)), (("a", "b"), (1.0,))]
)
def test_sparse_vector_validation(terms, weights):
    with pytest.raises(ValidationError):
        SparseVector(terms=terms, weights=weights)


def test_every_sparse_indexing_error_code_is_registered():
    for code in (
        "incompatible_sparse_encoder",
        "sparse_encoding_count_mismatch",
        "duplicate_sparse_vector",
        "sparse_unavailable",
    ):
        assert IndexingError(code).code == code


def test_unavailable_and_real_registry_registration():
    registry = TechniqueRegistry(load_catalogue(Path(__file__).resolve().parents[2] / "config/techniques.json"))
    register_builtin_plugins(registry)
    assert registry.plugin("R05").manifest.technique_id == "R05"
    with pytest.raises(TechniqueUnavailableError):
        asyncio.run(
            LearnedSparseRetriever(
                SparseConfig(),
                scope=make_scope(),
                index_version=SimpleNamespace(state=IndexState.READY, id="version"),
                index=object(),
            ).retrieve("query")
        )


def test_vocab_double_key_is_stable():
    assert sparse_encoder_key(VocabSparseEncoder()) == sparse_encoder_key(VocabSparseEncoder())


def test_missing_local_model_and_missing_package_are_unavailable(tmp_path, monkeypatch):
    from rag_foundry_providers.sentence_transformers_sparse import SentenceTransformersSparseEncoder

    index = SimpleNamespace()
    retriever = LearnedSparseRetriever(
        SparseConfig(),
        scope=make_scope(),
        index_version=SimpleNamespace(state=IndexState.READY, id="version"),
        index=index,
        encoder=SentenceTransformersSparseEncoder(model_dir=tmp_path / "missing"),
    )
    with pytest.raises(TechniqueUnavailableError):
        asyncio.run(retriever.retrieve("query"))
    encoder = SentenceTransformersSparseEncoder(model_dir=tmp_path)
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    retriever = LearnedSparseRetriever(
        SparseConfig(),
        scope=make_scope(),
        index_version=SimpleNamespace(state=IndexState.READY, id="version"),
        index=index,
        encoder=encoder,
    )
    with pytest.raises(TechniqueUnavailableError):
        asyncio.run(retriever.retrieve("query"))


def test_build_paginates_and_retriever_scores_terms():
    scope = make_scope()
    chunks = [SimpleNamespace(id=str(uuid.uuid4()), text=f"common word{i}") for i in range(501)]

    class MemoryIndex:
        def __init__(self):
            self.rows = {}

        async def list_chunks(self, _scope, _version, *, offset=0, limit=100):
            return chunks[offset : offset + limit]

        async def add_sparse(self, _scope, _version, key, rows):
            self.rows.update({cid: vector for cid, vector in rows})

        async def sparse_search(self, _scope, _version, _key, query, _k):
            terms, weights = query
            found = []
            for chunk in chunks[:2]:
                vec = self.rows[chunk.id]
                matching = tuple((t, w) for t, w in zip(terms, weights, strict=True) if t in vec[0])
                score = sum(w * dict(zip(vec[0], vec[1], strict=True))[t] for t, w in matching)
                if score:
                    found.append(
                        SimpleNamespace(
                            chunk=SimpleNamespace(
                                id=chunk.id,
                                extraction_view_id=str(uuid.uuid4()),
                                start=0,
                                end=10,
                                block_ids=("b00001",),
                                text=chunk.text,
                            ),
                            score=score,
                            rank=0,
                            terms=matching,
                        )
                    )
            found.sort(key=lambda hit: hit.score, reverse=True)
            return [
                hit.model_copy(update={"rank": rank})
                if hasattr(hit, "model_copy")
                else SimpleNamespace(**{**vars(hit), "rank": rank})
                for rank, hit in enumerate(found, 1)
            ]

    index = MemoryIndex()
    encoder = VocabSparseEncoder()
    stats = asyncio.run(SparseIndexService(index).build(scope, SimpleNamespace(id="version"), encoder))
    assert stats == {"chunks": 501, "nonzero": 501, "terms": 1002}

    # A query sharing a distinctive term ranks that chunk first and retains per-term evidence.
    chunks[0].text = "rare alpha"
    chunks[1].text = "rare beta"
    index.rows[chunks[0].id] = (("rare", "alpha"), (1.0, 1.0))
    index.rows[chunks[1].id] = (("rare", "beta"), (1.0, 1.0))
    result = asyncio.run(
        LearnedSparseRetriever(
            SparseConfig(),
            scope=scope,
            index_version=SimpleNamespace(state=IndexState.READY, id="version"),
            index=index,
            encoder=encoder,
        ).retrieve("alpha")
    )
    assert result.scoring == "learned-sparse-dot"
    assert result.hits[0].chunk_id == chunks[0].id
    assert result.hits[0].score_components[0].name == "sparse_term"


def test_build_rejects_encoder_count_mismatch():
    class BadEncoder(VocabSparseEncoder):
        async def encode(self, texts, *, input_type):
            result = await super().encode(texts, input_type=input_type)
            return result.model_copy(update={"vectors": result.vectors[:-1]})

    class MemoryIndex:
        async def list_chunks(self, *_args, **_kwargs):
            return [SimpleNamespace(id="id", text="word")]

    with pytest.raises(IndexingError) as error:
        asyncio.run(SparseIndexService(MemoryIndex()).build(make_scope(), SimpleNamespace(id="version"), BadEncoder()))
    assert error.value.code == "sparse_encoding_count_mismatch"


def test_add_sparse_runs_in_a_write_unit_of_work():
    from rag_foundry_domain.errors import NotFoundError
    from rag_foundry_domain.uow_adapters import per_call_indexes
    from rag_foundry_domain.uow_testing import InMemoryUnitOfWorkFactory

    inner = InMemoryUnitOfWorkFactory()
    modes = []

    def factory(*, read_only):
        modes.append(read_only)
        return inner(read_only=read_only)

    # The SQL unit of work rejects writes in read-only mode, so add_sparse must open a write one.
    with pytest.raises(NotFoundError):
        asyncio.run(per_call_indexes(factory).add_sparse(make_scope(), "missing-version", "test-encoder", ()))
    assert modes == [False]
