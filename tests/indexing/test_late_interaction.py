import asyncio
import math
import re
from pathlib import Path

import pytest
from _lexical_embedder import LexicalTestEmbedder
from rag_foundry_domain.errors import IndexingError, NotFoundError
from rag_foundry_domain.indexing import IndexState
from rag_foundry_indexing.late_interaction import (
    LateInteractionConfig,
    LateInteractionRetriever,
    build_multivector_index,
)
from rag_foundry_providers.errors import AdapterDependencyError
from rag_foundry_techniques import Fidelity, PluginKind
from rag_foundry_techniques.errors import TechniqueUnavailableError

from tests.indexing._inmemory_index import ready_index
from tests.providers._fake_token_encoder import FakeTokenEncoder

R = 0.5**0.5
ARABIC_A = "".join(chr(c) for c in (0x645, 0x631, 0x62D, 0x628, 0x627))
ARABIC_B = "".join(chr(c) for c in (0x643, 0x62A, 0x627, 0x628))
VECS = {
    "red": (1.0, 0.0),
    "blue": (0.0, 1.0),
    "green": (0.6, 0.8),
    "orange": (R, R),
    ARABIC_A: (0.0, -1.0),
    ARABIC_B: (-1.0, 0.0),
}
LINES = ["red blue green", "orange", "green", "red", f"{ARABIC_A} {ARABIC_B}"]
TEXT = "\n".join(LINES)


class DictEncoder(FakeTokenEncoder):
    """Fake whose per-word vectors are fixed by `VECS` (2-dim), so MaxSim can be computed by hand."""

    def __init__(self, dim: int = 2) -> None:
        super().__init__(max_tokens=16, dim=dim)
        self.out_dim = dim

    async def encode_tokens(self, texts, *, head, input_type):
        assert head == "colbert"
        result = await super().encode_tokens(texts, head=head, input_type=input_type)
        items = []
        for text, item in zip(texts, result.items, strict=True):
            vectors = tuple(VECS.get(text[s:e], (0.0, 0.0)) + (0.0,) * (self.out_dim - 2) for s, e in item.offsets)
            items.append(item.model_copy(update={"vectors": vectors}))
        return result.model_copy(update={"items": tuple(items), "dim": self.out_dim})


def _spans(text):
    return [(m.start(), m.end(), m.group()) for m in re.finditer(r"[^\n]+", text)]


async def _fixture(ready=True):
    scope, stranger, index, version = await ready_index(_spans(TEXT), LexicalTestEmbedder(), ready=ready)
    return scope, stranger, index, version


async def _build(encoder=None):
    scope, stranger, index, version = await _fixture()
    encoder = encoder or DictEncoder()
    built = await build_multivector_index(scope, version, index, encoder)
    return scope, stranger, version, built, encoder


def _retriever(scope, version, built, encoder, **config):
    return LateInteractionRetriever(
        LateInteractionConfig(**config), scope=scope, index_version=version, index=built, encoder=encoder
    )


def _by_text(result):
    return {hit.text: hit for hit in result.hits}


def test_manifest_and_registration():
    manifest = LateInteractionRetriever.manifest
    assert manifest.technique_id == "R45"
    assert manifest.kind is PluginKind.RETRIEVER
    assert manifest.fidelity is Fidelity.ADAPTATION
    from rag_foundry_indexing import register_builtin_plugins
    from rag_foundry_techniques import TechniqueRegistry, load_catalogue

    registry = TechniqueRegistry(load_catalogue(Path(__file__).resolve().parents[2] / "config/techniques.json"))
    register_builtin_plugins(registry)
    assert registry.validate_config("R45", {}) == LateInteractionConfig()


async def _hand_computed():
    scope, _, version, built, encoder = await _build()
    result = await _retriever(scope, version, built, encoder).retrieve("red blue")
    assert result.technique_id == "R45" and result.scoring == "r45-maxsim"
    hits = _by_text(result)
    assert hits["red blue green"].score == pytest.approx(2.0)  # red->red 1 + blue->blue 1
    assert hits["orange"].score == pytest.approx(2 * R)  # 0.7071 + 0.7071
    assert hits["green"].score == pytest.approx(0.6 + 0.8)
    assert hits["red"].score == pytest.approx(1.0)  # blue matches nothing (0)
    assert [h.text for h in result.hits][:4] == ["red blue green", "orange", "green", "red"]
    assert [h.rank for h in result.hits] == [1, 2, 3, 4, 5][: len(result.hits)]
    assert all(h.distance is None for h in result.hits)


def test_hand_computed_maxsim():
    asyncio.run(_hand_computed())


async def _maxsim_beats_mean():
    scope, _, version, built, encoder = await _build()
    query = ("red", "blue")
    qmean = tuple(sum(VECS[w][i] for w in query) / 2 for i in range(2))

    def mean_cos(chunk):
        words = chunk.text.split()
        mean = tuple(sum(VECS[w][i] for w in words) / len(words) for i in range(2))
        dot = sum(a * b for a, b in zip(qmean, mean, strict=True))
        return dot / (math.hypot(*qmean) * math.hypot(*mean))

    by_mean = sorted((c for c in built.chunks if c.text in {"red blue green", "orange"}), key=mean_cos, reverse=True)
    assert by_mean[0].text == "orange"  # the pooled single-vector cosine prefers the wrong chunk
    result = await _retriever(scope, version, built, encoder, k=2).retrieve("red blue")
    assert result.hits[0].text == "red blue green"  # MaxSim ranks the chunk matching each token first
    assert result.hits[1].text == "orange"


def test_maxsim_beats_mean_of_same_vectors():
    asyncio.run(_maxsim_beats_mean())


async def _components():
    scope, _, version, built, encoder = await _build()
    result = await _retriever(scope, version, built, encoder).retrieve("red blue")
    for hit in result.hits:
        assert all(c.name == "maxsim_token" for c in hit.score_components)
        assert len(hit.score_components) <= 2
        assert sum(c.value for c in hit.score_components) == pytest.approx(hit.score)
    hit = _by_text(result)["red blue green"]
    assert {c.term for c in hit.score_components} == {"red->red", "blue->blue"}
    green = _by_text(result)["green"]
    assert {c.term for c in green.score_components} == {"red->green", "blue->green"}  # argmax pairs


def test_contributions_sum_and_match_pairs():
    asyncio.run(_components())


async def _top5_components():
    scope, _, version, built, encoder = await _build()
    result = await _retriever(scope, version, built, encoder).retrieve("red blue green orange red blue")
    hit = result.hits[0]
    assert len(hit.score_components) == 5
    values = [c.value for c in hit.score_components]
    assert values == sorted(values, reverse=True)


def test_components_capped_at_five():
    asyncio.run(_top5_components())


async def _max_query_tokens():
    scope, _, version, built, encoder = await _build()
    result = await _retriever(scope, version, built, encoder, max_query_tokens=1).retrieve("red blue")
    assert _by_text(result)["red"].score == pytest.approx(1.0)
    assert _by_text(result)["green"].score == pytest.approx(0.6)  # only the first query token is used


def test_max_query_tokens_truncates():
    asyncio.run(_max_query_tokens())


async def _stats():
    _, _, _, built, _ = await _build()
    stats = built.stats()
    assert stats.chunks == 5 and stats.tokens == 3 + 1 + 1 + 1 + 2 and stats.dim == 2
    assert stats.estimated_bytes == 8 * 2 * 4
    for vectors in built.vectors.values():
        for v in vectors:
            assert math.hypot(*v) == pytest.approx(1.0)


def test_stats_and_normalisation():
    asyncio.run(_stats())


async def _query_errors():
    scope, _, version, built, encoder = await _build()
    retriever = _retriever(scope, version, built, encoder)
    for query, code in (("   ", "empty_query"), ("x" * 2001, "query_too_long")):
        with pytest.raises(IndexingError) as exc:
            await retriever.retrieve(query)
        assert exc.value.code == code
    building = version.model_copy(update={"state": IndexState.BUILDING})
    with pytest.raises(IndexingError) as exc:
        await _retriever(scope, building, built, encoder).retrieve("red")
    assert exc.value.code == "not_ready"
    with pytest.raises(IndexingError) as exc:
        await _retriever(scope, version, None, encoder).retrieve("red")
    assert exc.value.code == "multivector_index_missing"
    with pytest.raises(TechniqueUnavailableError):
        await _retriever(scope, version, built, None).retrieve("red")

    class Missing(DictEncoder):
        async def encode_tokens(self, texts, *, head, input_type):
            raise AdapterDependencyError("missing extra")

    with pytest.raises(TechniqueUnavailableError):
        await _retriever(scope, version, built, Missing()).retrieve("red")
    with pytest.raises(IndexingError) as exc:
        await _retriever(scope, version, built, DictEncoder(dim=3)).retrieve("red")
    assert exc.value.code == "multivector_dim_mismatch"

    class Empty(DictEncoder):
        async def encode_tokens(self, texts, *, head, input_type):
            result = await super().encode_tokens(texts, head=head, input_type=input_type)
            empty = tuple(i.model_copy(update={"vectors": (), "offsets": ()}) for i in result.items)
            return result.model_copy(update={"items": empty})

    with pytest.raises(IndexingError) as exc:
        await _retriever(scope, version, built, Empty()).retrieve("red")
    assert exc.value.code == "late_chunk_span_empty"


def test_query_errors():
    asyncio.run(_query_errors())


async def _other_version_rejected():
    scope, _, version, built, encoder = await _build()
    other = version.model_copy(update={"id": "99999999-9999-4999-8999-999999999999"})
    with pytest.raises(IndexingError) as exc:
        await _retriever(scope, other, built, encoder).retrieve("red")
    assert exc.value.code == "multivector_index_missing"


def test_index_for_another_version_rejected():
    asyncio.run(_other_version_rejected())


async def _cross_tenant():
    scope, stranger, version, built, encoder = await _build()
    assert built.tenant_id == scope.tenant_id and built.project_id == scope.project_id
    calls = encoder.calls
    with pytest.raises(NotFoundError):
        await _retriever(stranger, version, built, encoder).retrieve("red blue")
    assert encoder.calls == calls  # refused before any encoder call
    with pytest.raises(NotFoundError):  # even an invalid query hits the scope check first
        await _retriever(stranger, version, built, encoder).retrieve("")
    assert encoder.calls == calls


def test_cross_tenant_query_refused_before_encoder():
    asyncio.run(_cross_tenant())


async def _build_errors():
    scope, stranger, index, version = await _fixture()
    with pytest.raises(NotFoundError):  # chunks of another scope are not readable
        await build_multivector_index(stranger, version, index, DictEncoder())

    class Drift(DictEncoder):
        async def encode_tokens(self, texts, *, head, input_type):
            result = await super().encode_tokens(texts, head=head, input_type=input_type)
            items = tuple(i.model_copy(update={"vectors": tuple(v + (0.0,) for v in i.vectors)}) for i in result.items)
            return result.model_copy(update={"items": items})

    with pytest.raises(IndexingError) as exc:
        await build_multivector_index(scope, version, index, Drift())
    assert exc.value.code == "multivector_dim_mismatch"
    with pytest.raises(TechniqueUnavailableError):

        class Missing(DictEncoder):
            async def encode_tokens(self, texts, *, head, input_type):
                raise AdapterDependencyError("missing extra")

        await build_multivector_index(scope, version, index, Missing())


def test_build_errors():
    asyncio.run(_build_errors())


async def _arabic():
    scope, _, version, built, encoder = await _build()
    query = f"{ARABIC_A} {ARABIC_B}"
    result = await _retriever(scope, version, built, encoder).retrieve(query)
    top = result.hits[0]
    assert top.text == query and top.score == pytest.approx(2.0)
    assert {c.term for c in top.score_components} == {f"{ARABIC_A}->{ARABIC_A}", f"{ARABIC_B}->{ARABIC_B}"}


def test_arabic_query():
    asyncio.run(_arabic())
