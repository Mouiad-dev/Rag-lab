import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from rag_foundry_domain.errors import IndexingError, NotFoundError
from rag_foundry_domain.index_testing import InMemoryIndexRepository, make_chunk, make_new_index_version
from rag_foundry_domain.indexing import IndexState, embedding_key_for
from rag_foundry_domain.testing import (
    InMemoryDocumentRepository,
    InMemoryExtractionRepository,
    make_new_extraction,
    make_new_version,
    make_scope,
)
from rag_foundry_indexing import BM25Config, BM25Retriever, DenseConfig, DenseRetriever, lexical_tokens
from rag_foundry_techniques import TechniqueRegistry, load_catalogue


class FakeIndex:
    def __init__(self, texts, state=IndexState.READY):
        self.state = state
        self.chunks = [
            SimpleNamespace(
                id=f"c{i}",
                extraction_view_id="v",
                start=0,
                end=len(text),
                block_ids=("b",),
                text=text,
                ordinal=i,
            )
            for i, text in enumerate(texts)
        ]

    async def get_index_version(self, scope, version_id):
        return SimpleNamespace(state=self.state)

    async def list_chunks(self, scope, version_id, *, offset=0, limit=100):
        return self.chunks[offset : offset + limit]


def run(awaitable):
    return asyncio.run(awaitable)


def retriever(texts, query_scope=None, state=IndexState.READY):
    scope = make_scope()
    version = SimpleNamespace(id="version")
    return BM25Retriever(
        BM25Config(),
        scope=query_scope or scope,
        index_version=version,
        index=FakeIndex(texts, state=state),
    ), scope


def test_bm25_formula_and_components():
    instance, _ = retriever(["alpha beta", "alpha", "gamma"])
    result = run(instance.retrieve("beta"))
    assert len(result.hits) == 1
    assert result.hits[0].score == pytest.approx(0.814273, abs=1e-4)
    assert sum(c.value for c in result.hits[0].score_components) == pytest.approx(result.hits[0].score)
    assert result.scoring == "bm25-okapi-lucene-idf"


def test_bm25_preserves_parent_offsets():
    instance, _ = retriever(["alpha beta"])
    instance._stats = None
    instance._index.chunks[0].parent_start_cp = 0
    instance._index.chunks[0].parent_end_cp = len("alpha beta")
    hit = run(instance.retrieve("alpha")).hits[0]
    assert (hit.parent_start, hit.parent_end) == (0, len("alpha beta"))


@pytest.mark.parametrize("hidden", [False, True])
def test_identifier_is_a_whole_token(hidden):
    bidi = "\u200f" if hidden else ""
    prefix = "\u0631\u0642\u0645 \u0627\u0644\u0642\u0637\u0639\u0629 "  # "part number"
    first = f"{prefix}AB-{bidi}1234"
    instance, _ = retriever([first, f"{prefix}AB-1243", f"{prefix}AB 1234"])
    hit = run(instance.retrieve("AB-1234")).hits[0]
    assert hit.chunk_id == "c0"
    assert any(c.name == "bm25_term" and c.term == "ab-1234" for c in hit.score_components)


@pytest.mark.parametrize(
    ("query", "text"),
    [
        ("\u0643\u0650\u062a\u064e\u0640\u0640\u0627\u0628", "\u0643\u062a\u0627\u0628"),
        ("\u0625\u062f\u0627\u0631\u0629", "\u0627\u062f\u0627\u0631\u0629"),
        ("\u0642-\u0667\u0667\u0662", "\u0642-772"),
    ],
)
def test_arabic_normalization(query, text):
    instance, _ = retriever([text])
    assert run(instance.retrieve(query)).hits


def test_does_not_stem_or_split_clitics():
    instance, _ = retriever(["\u0627\u0644\u0645\u0643\u062a\u0628\u0629"])
    assert not run(instance.retrieve("\u0645\u0643\u062a\u0628\u0629")).hits


def test_error_paths_and_registry():
    instance, _ = retriever(["alpha"], state=IndexState.BUILDING)
    with pytest.raises(IndexingError, match="not_ready"):
        run(instance.retrieve("alpha"))
    instance, scope = retriever(["alpha"])
    # Punctuation only: not empty, but no lexical tokens survive the tokenizer.
    with pytest.raises(IndexingError) as caught:
        run(instance.retrieve("---"))
    assert caught.value.code == "no_query_terms"

    class MissingIndex(FakeIndex):
        async def get_index_version(self, scope, version_id):
            raise NotFoundError()

    with pytest.raises(NotFoundError):
        instance = BM25Retriever(
            BM25Config(),
            scope=make_scope(),
            index_version=SimpleNamespace(id="version"),
            index=MissingIndex(["a"]),
        )
        run(instance.retrieve("a"))
    registry = TechniqueRegistry(load_catalogue(Path(__file__).resolve().parents[2] / "config/techniques.json"))
    from rag_foundry_indexing import register_builtin_plugins

    register_builtin_plugins(registry)
    assert registry.validate_config("R03", {}) == BM25Config()
    with pytest.raises(ValidationError):
        registry.validate_config("R03", {"unknown": True})


def test_over_max_chunks_fails_explicitly():
    instance = BM25Retriever(
        BM25Config(max_chunks=2),
        scope=make_scope(),
        index_version=SimpleNamespace(id="version"),
        index=FakeIndex(["alpha", "beta", "gamma"]),
    )
    with pytest.raises(IndexingError) as caught:
        run(instance.retrieve("alpha"))
    assert caught.value.code == "lexical_corpus_too_large"


async def _ready_version(texts):
    owner, stranger = make_scope(), make_scope()
    docs = InMemoryDocumentRepository([owner, stranger])
    views = InMemoryExtractionRepository(docs)
    index = InMemoryIndexRepository(views)
    out = await docs.create_document_or_get(owner, make_new_version("bm25"), display_name="fixture")
    view = (await views.create_view_or_get(owner, make_new_extraction(out.version, tag="bm25"))).view
    version = await index.create_index_version(owner, make_new_index_version((view.id,)))
    chunks = tuple(make_chunk(view.id, i, (1.0, 0.0, 0.0, 0.0), text) for i, text in enumerate(texts))
    await index.add_chunks(owner, version.id, chunks)
    version = await index.mark_ready(owner, version.id)
    return owner, stranger, index, version


def test_scope_is_rechecked_and_other_scopes_see_nothing():
    async def scenario():
        owner, stranger, index, version = await _ready_version(["alpha beta", "gamma"])
        own = BM25Retriever(BM25Config(), scope=owner, index_version=version, index=index)
        assert [h.text for h in (await own.retrieve("alpha")).hits] == ["alpha beta"]
        foreign = BM25Retriever(BM25Config(), scope=stranger, index_version=version, index=index)
        with pytest.raises(NotFoundError):
            await foreign.retrieve("alpha")

    run(scenario())


def test_cached_stats_do_not_skip_the_ready_check():
    index = FakeIndex(["alpha"])
    instance = BM25Retriever(BM25Config(), scope=make_scope(), index_version=SimpleNamespace(id="v"), index=index)
    assert run(instance.retrieve("alpha")).hits
    index.state = IndexState.FAILED
    with pytest.raises(IndexingError) as caught:
        run(instance.retrieve("alpha"))
    assert caught.value.code == "not_ready"


def test_tokenizer_id_and_ascii_surface():
    assert lexical_tokens("AB-1234") == ("ab", "1234", "ab-1234")
    assert DenseConfig().k == 5


def test_dense_retrieval_exposes_cosine_component():
    scope = make_scope()
    profile = SimpleNamespace(
        provider="test",
        model="test-model",
        capabilities=SimpleNamespace(embedding_dim=2),
    )

    class Embedder:
        def __init__(self):
            self.profile = profile

        async def embed(self, texts, *, input_type):
            return SimpleNamespace(vectors=((1.0, 0.0),))

    class DenseIndex:
        async def search(self, scope_arg, version_id, vector, k):
            chunk = SimpleNamespace(
                id="dense-chunk",
                extraction_view_id="v",
                start=0,
                end=5,
                block_ids=("b",),
                text="alpha",
            )
            return [SimpleNamespace(chunk=chunk, distance=0.2, rank=1)]

    version = SimpleNamespace(
        id="version",
        embedding_key=embedding_key_for("test", "test-model", 2),
    )
    retriever = DenseRetriever(
        DenseConfig(),
        scope=scope,
        index_version=version,
        index=DenseIndex(),
        embedder=Embedder(),
    )
    hit = run(retriever.retrieve("alpha")).hits[0]
    assert hit.score_components[0].name == "cosine_similarity"
    assert hit.score_components[0].value == pytest.approx(hit.score)
