import asyncio
import hashlib
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError
from rag_foundry_domain.errors import IndexingError, NotFoundError, QueryTransformError
from rag_foundry_domain.index_testing import make_representation
from rag_foundry_domain.representations import RepresentationSet, representation_key_for
from rag_foundry_indexing import (
    ContextualRetriever,
    ContextualSearchConfig,
    MultiRepresentationRetriever,
    RepresentationSearchConfig,
)
from rag_foundry_techniques import PluginKind, TechniqueRegistry, load_catalogue

sys.path.insert(0, str(Path(__file__).parents[1] / "chunking"))
sys.path.insert(0, str(Path(__file__).parent))
from _inmemory_index import ready_index  # noqa: E402
from _scripted_models import ScriptedEmbedder  # noqa: E402

TEXT_A = "Port Sallow tide log."
TEXT_B = "Mill Lane bakery opens early."
ARABIC_PREFIX = "\u0633\u064a\u0627\u0642 A"
VECTORS = {
    TEXT_A: (0.0, 0.0, 1.0, 0.0),
    TEXT_B: (0.0, 0.0, 0.0, 1.0),
    "q1": (1.0, 0.0, 0.0, 0.0),
    "q2": (0.0, 0.0, 1.0, 0.0),
}
SYNTHETIC = ("S-A", "Q-B0", "Q-B1", ARABIC_PREFIX)
R23_KEY = representation_key_for({"technique": "R23", "t": 1})
R24_KEY = representation_key_for({"technique": "R24", "t": 1})


def _embedder():
    return ScriptedEmbedder(lambda text: VECTORS[text])


async def _setup(ready=True):
    embedder = _embedder()
    spans = [(0, len(TEXT_A), TEXT_A), (len(TEXT_A), len(TEXT_A) + len(TEXT_B), TEXT_B)]
    owner, stranger, index, version = await ready_index(spans, embedder, ready=ready)
    if ready:
        chunks = await index.list_chunks(owner, version.id)
        a, b = chunks
        await index.add_representations(
            owner,
            version.id,
            RepresentationSet(representation_key=R23_KEY, technique_id="R23", generator_json="{}"),
            (
                make_representation(a.id, "summary", 0, "S-A", (1, 0, 0, 0)),
                make_representation(b.id, "question", 0, "Q-B0", (0, 1, 0, 0)),
                make_representation(b.id, "question", 1, "Q-B1", (0.6, 0.8, 0, 0)),
            ),
        )
        await index.add_representations(
            owner,
            version.id,
            RepresentationSet(representation_key=R24_KEY, technique_id="R24", generator_json="{}"),
            (make_representation(a.id, "context_prefix", 0, ARABIC_PREFIX, (1, 0, 0, 0)),),
        )
    return owner, stranger, index, version, embedder


def _r23(owner, version, index, embedder, **config):
    cfg = RepresentationSearchConfig(representation_key=config.pop("key", R23_KEY), **config)
    return MultiRepresentationRetriever(cfg, scope=owner, index_version=version, index=index, embedder=embedder)


def _no_synthetic_in_hits(retrieval):
    for hit in retrieval.result.hits:
        assert hit.text in (TEXT_A, TEXT_B)
        assert not any(s in hit.text for s in SYNTHETIC)


def test_summary_and_question_matches_return_original_chunks():
    async def run():
        owner, _, index, version, embedder = await _setup()
        retrieval = await _r23(owner, version, index, embedder, k=2).retrieve_traced("q1")
        r1, r2 = retrieval.result.hits
        assert (r1.text, r1.rank, r1.score_components[0].term) == (TEXT_A, 1, "summary:0")
        assert r1.score == pytest.approx(1.0)
        assert (r2.text, r2.rank, r2.score_components[0].term) == (TEXT_B, 2, "question:1")
        assert r2.score == pytest.approx(0.6)
        assert r2.score_components[0].name == "representation_cosine"
        assert retrieval.matches[0].text == "S-A"
        assert retrieval.matches[0].synthetic is True
        assert retrieval.matches[1].text == "Q-B1"
        assert retrieval.result.technique_id == "R23"
        assert retrieval.result.scoring == "representation-max-cosine"
        assert retrieval.result.grounding == "corpus"
        assert retrieval.result.index_version_id == version.id
        _no_synthetic_in_hits(retrieval)
        assert embedder.calls.count(("q1",)) == 1

    asyncio.run(run())


def test_include_original_wins_on_exact_match_and_ties():
    async def run():
        owner, _, index, version, embedder = await _setup()
        with_original = await _r23(owner, version, index, embedder, k=2, include_original=True).retrieve_traced("q2")
        r1, r2 = with_original.result.hits
        assert (r1.text, r1.score_components[0].term) == (TEXT_A, "original:0")
        assert r1.score == pytest.approx(1.0)
        assert (r2.text, r2.score_components[0].term) == (TEXT_B, "original:0")
        assert r2.score == pytest.approx(0.0)
        first = with_original.matches[0]
        assert (first.kind, first.ordinal, first.synthetic, first.text) == ("original", 0, False, None)
        _no_synthetic_in_hits(with_original)

        without = await _r23(owner, version, index, embedder, k=2, include_original=False).retrieve_traced("q2")
        r1, r2 = without.result.hits
        assert (r1.text, r1.score_components[0].term) == (TEXT_A, "summary:0")
        assert r1.score == pytest.approx(0.0)
        assert (r2.text, r2.score_components[0].term) == (TEXT_B, "question:0")
        _no_synthetic_in_hits(without)

    asyncio.run(run())


def test_r24_returns_original_chunk_and_prefix_only_in_matches():
    async def run():
        owner, _, index, version, embedder = await _setup()
        config = ContextualSearchConfig(representation_key=R24_KEY, k=1)
        assert config.include_original is False
        retriever = ContextualRetriever(config, scope=owner, index_version=version, index=index, embedder=embedder)
        retrieval = await retriever.retrieve_traced("q1")
        hit = retrieval.result.hits[0]
        assert (hit.text, hit.score_components[0].term) == (TEXT_A, "context_prefix:0")
        assert retrieval.matches[0].text == ARABIC_PREFIX
        assert ARABIC_PREFIX not in hit.text
        assert retrieval.result.technique_id == "R24"
        result = await retriever.retrieve("q1")
        assert result.hits[0].text == TEXT_A

    asyncio.run(run())


def test_config_validation_and_registration():
    with pytest.raises(ValidationError):
        RepresentationSearchConfig(representation_key=R23_KEY, k=5, candidate_k=4)
    with pytest.raises(ValidationError):
        RepresentationSearchConfig(representation_key="nope")
    assert RepresentationSearchConfig(representation_key=R23_KEY).include_original is True
    registry = TechniqueRegistry(load_catalogue(Path(__file__).resolve().parents[2] / "config/techniques.json"))
    from rag_foundry_indexing import register_builtin_plugins

    register_builtin_plugins(registry)
    assert registry.validate_config("R23", {"representation_key": R23_KEY}).include_original is True
    assert registry.validate_config("R24", {"representation_key": R24_KEY}).include_original is False
    for plugin in (MultiRepresentationRetriever, ContextualRetriever):
        assert plugin.manifest.kind is PluginKind.REPRESENTATION
        assert plugin.manifest.version == "1.0.0"


def test_errors():
    async def run():
        owner, stranger, index, version, embedder = await _setup()
        r24_on_r23 = ContextualRetriever(
            ContextualSearchConfig(representation_key=R23_KEY),
            scope=owner,
            index_version=version,
            index=index,
            embedder=embedder,
        )
        with pytest.raises(IndexingError) as info:
            await r24_on_r23.retrieve("q1")
        assert info.value.code == "representation_technique_mismatch"
        assert embedder.calls == [(TEXT_A,), (TEXT_B,)]

        with pytest.raises(NotFoundError):
            await _r23(stranger, version, index, embedder).retrieve("q1")
        unknown = hashlib.sha256(b"unknown").hexdigest()
        with pytest.raises(NotFoundError):
            await _r23(owner, version, index, embedder, key=unknown).retrieve("q1")

        other = embedder.profile.model_copy(update={"model": "another-model"})
        other_embedder = ScriptedEmbedder(lambda t: VECTORS[t], other)
        with pytest.raises(IndexingError) as info:
            await _r23(owner, version, index, other_embedder).retrieve("q1")
        assert info.value.code == "incompatible_embedding"
        assert other_embedder.calls == []

        with pytest.raises(QueryTransformError) as info:
            await _r23(owner, version, index, embedder).retrieve("   ")
        assert info.value.code == "empty_query"
        with pytest.raises(QueryTransformError) as info:
            await _r23(owner, version, index, embedder).retrieve("x" * 2001)
        assert info.value.code == "query_too_long"

    asyncio.run(run())


def test_not_ready_version_never_embeds():
    async def run():
        owner, _, index, version, embedder = await _setup(ready=False)
        calls_before = len(embedder.calls)
        with pytest.raises(IndexingError) as info:
            await _r23(owner, version, index, embedder).retrieve("q1")
        assert info.value.code == "not_ready"
        assert len(embedder.calls) == calls_before

    asyncio.run(run())


class _TamperedChunkIndex:
    """Delegates to the real index but returns chunks whose text no longer matches text_sha256."""

    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, name):
        return getattr(self._inner, name)

    async def representation_search(self, *args, **kwargs):
        found = await self._inner.representation_search(*args, **kwargs)
        return [item.model_copy(update={"chunk": item.chunk.model_copy(update={"text": "tampered"})}) for item in found]


def test_kind_outside_plugin_kinds_is_rejected():
    async def run():
        owner, _, index, version, embedder = await _setup()
        key = representation_key_for({"technique": "R24", "t": "wrong-kind"})
        chunk_a = (await index.list_chunks(owner, version.id))[0]
        await index.add_representations(
            owner,
            version.id,
            RepresentationSet(representation_key=key, technique_id="R24", generator_json="{}"),
            (make_representation(chunk_a.id, "summary", 0, "S-wrong", (1, 0, 0, 0)),),
        )
        retriever = ContextualRetriever(
            ContextualSearchConfig(representation_key=key),
            scope=owner,
            index_version=version,
            index=index,
            embedder=embedder,
        )
        with pytest.raises(IndexingError) as info:
            await retriever.retrieve("q1")
        assert info.value.code == "representation_kind_mismatch"

    asyncio.run(run())


def test_chunk_text_hash_mismatch_is_rejected():
    async def run():
        owner, _, index, version, embedder = await _setup()
        retriever = _r23(owner, version, _TamperedChunkIndex(index), embedder, include_original=False)
        with pytest.raises(IndexingError) as info:
            await retriever.retrieve("q1")
        assert info.value.code == "representation_source_mismatch"

    asyncio.run(run())
