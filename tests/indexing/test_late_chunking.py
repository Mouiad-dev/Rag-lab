import asyncio
import re

import pytest
from _lexical_embedder import LexicalTestEmbedder
from rag_foundry_domain.errors import IndexingError, NotFoundError
from rag_foundry_indexing.context_expansion import ViewSource
from rag_foundry_indexing.late_chunking import (
    LateChunkingConfig,
    LateChunkingRepresentation,
    LateChunkingSearch,
)
from rag_foundry_providers.errors import AdapterDependencyError
from rag_foundry_providers.messages import TokenEncoding
from rag_foundry_techniques import Fidelity, PluginKind
from rag_foundry_techniques.errors import TechniqueUnavailableError

from tests.indexing._inmemory_index import ready_index
from tests.indexing._source_reader import StaticSourceReader
from tests.providers._fake_token_encoder import FakeTokenEncoder

WORDS = ["aa", "bb", "cc", "dd", "ee", "ff", "gg", "hh", "ii", "jj", "kk", "ll"]
TEXT = " ".join(WORDS)


class ContextEncoder(FakeTokenEncoder):
    """Fake whose token vectors depend on the first token of the encoded input (prior context)."""

    def __init__(self, max_tokens: int = 16, dim: int = 4) -> None:
        super().__init__(max_tokens, dim)
        self.texts: list[str] = []

    async def encode_tokens(self, texts, *, head, input_type):
        self.texts.extend(texts)
        result = await super().encode_tokens(texts, head=head, input_type=input_type)
        items = []
        for item in result.items:
            bias = item.vectors[0][0] + 1.0 if item.vectors else 0.0
            vectors = tuple(tuple(x + bias for x in v) for v in item.vectors)
            items.append(TokenEncoding(vectors=vectors, offsets=item.offsets, truncated=item.truncated))
        return result.model_copy(update={"items": tuple(items)})


def _spans(text, per_chunk):
    words = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
    out = []
    for i in range(0, len(words), per_chunk):
        group = words[i : i + per_chunk]
        start, end = group[0][0], group[-1][1]
        out.append((start, end, text[start:end]))
    return out


async def _fixture(text=TEXT, per_chunk=3):
    scope, stranger, index, version = await ready_index(_spans(text, per_chunk), LexicalTestEmbedder())
    view = version.extraction_view_ids[0]
    source = StaticSourceReader(
        {(scope.tenant_id, scope.project_id, view): ViewSource(extraction_view_id=view, text=text, blocks=())}
    )
    return scope, stranger, index, version, source


def _plugin(index, source, encoder, **config):
    return LateChunkingRepresentation(LateChunkingConfig(**config), index=index, source=source, encoder=encoder)


def _pool(vectors):
    return tuple(sum(v[i] for v in vectors) / len(vectors) for i in range(len(vectors[0])))


def test_manifest():
    manifest = LateChunkingRepresentation.manifest
    assert manifest.technique_id == "C20"
    assert manifest.kind is PluginKind.REPRESENTATION
    assert manifest.fidelity is Fidelity.ADAPTATION


async def _late_differs_early_equals_alone():
    scope, _, index, version, source = await _fixture()
    encoder = ContextEncoder(max_tokens=16)
    late = await _plugin(index, source, encoder, mode="late", window_tokens=16, window_overlap_tokens=0).build(
        scope, version
    )
    early = await _plugin(index, source, encoder, mode="early", window_tokens=16, window_overlap_tokens=0).build(
        scope, version
    )
    assert late.mode == "late" and early.mode == "early"
    assert late.tenant_id == scope.tenant_id and late.project_id == scope.project_id
    chunks = sorted(late.chunks, key=lambda c: c.ordinal)
    assert late.vectors[chunks[0].id] == early.vectors[chunks[0].id]  # no prior context for the first chunk
    for chunk in chunks[1:]:
        assert late.vectors[chunk.id] != early.vectors[chunk.id]
    for chunk in chunks:
        alone = (await encoder.encode_tokens([chunk.text], head="hidden", input_type="document")).items[0]
        assert early.vectors[chunk.id] == _pool(alone.vectors)
        assert early.token_count[chunk.id] == 3 and late.token_count[chunk.id] == 3
    assert not late.truncated and not early.truncated


def test_late_differs_early_equals_alone():
    asyncio.run(_late_differs_early_equals_alone())


async def _one_encode_per_window_never_per_chunk():
    scope, _, index, version, source = await _fixture(per_chunk=3)  # 4 chunks
    encoder = ContextEncoder(max_tokens=8)
    plugin = _plugin(index, source, encoder, mode="late", window_tokens=6, window_overlap_tokens=2)
    vectors = await plugin.build(scope, version)
    assert len(vectors.chunks) == 4
    assert encoder.calls == vectors.windows_used == 3
    assert len(encoder.texts) == 3  # one text per window, never one per chunk
    chunk_texts = {c.text for c in vectors.chunks}
    assert not chunk_texts & set(encoder.texts)
    assert encoder.texts[0].startswith("aa bb") and len(encoder.texts[0]) > 11
    # Short view: a single window covers everything.
    encoder2 = ContextEncoder(max_tokens=16)
    short = await _plugin(index, source, encoder2, mode="late", window_tokens=16, window_overlap_tokens=0).build(
        scope, version
    )
    assert encoder2.calls == short.windows_used == 1


def test_one_encode_per_window_never_per_chunk():
    asyncio.run(_one_encode_per_window_never_per_chunk())


async def _overlap_assignment_is_deterministic():
    scope, _, index, version, source = await _fixture(per_chunk=1)
    plugin_args = dict(mode="late", window_tokens=6, window_overlap_tokens=2)
    first = await _plugin(index, source, ContextEncoder(max_tokens=8), **plugin_args).build(scope, version)
    second = await _plugin(index, source, ContextEncoder(max_tokens=8), **plugin_args).build(scope, version)
    assert first.vectors == second.vectors
    plain = FakeTokenEncoder(max_tokens=16)
    base = {}
    for word in WORDS:
        base[word] = (await plain.encode_tokens([word], head="hidden", input_type="document")).items[0].vectors[0]
    # Windows start at words 0, 4, 8. A token comes from the window where it is farthest from an edge.
    source_window = [0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2]
    for chunk in first.chunks:
        word = WORDS[chunk.ordinal]
        bias = base[WORDS[4 * source_window[chunk.ordinal]]][0] + 1.0
        assert first.vectors[chunk.id] == tuple(x + bias for x in base[word]), word


def test_overlap_assignment_is_deterministic():
    asyncio.run(_overlap_assignment_is_deterministic())


class _Bad(FakeTokenEncoder):
    def __init__(self, kind):
        super().__init__(max_tokens=16)
        self.kind = kind

    async def encode_tokens(self, texts, *, head, input_type):
        if self.kind == "dependency":
            raise AdapterDependencyError("token")
        result = await super().encode_tokens(texts, head=head, input_type=input_type)
        if self.kind == "count":
            return result.model_copy(update={"items": result.items + result.items})
        if self.kind == "offsets":
            item = result.items[0]
            bad = TokenEncoding(
                vectors=item.vectors, offsets=tuple((s, e + 1000) for s, e in item.offsets), truncated=False
            )
            return result.model_copy(update={"items": (bad,) + result.items[1:]})
        return result


async def _error_paths():
    scope, _, index, version, source = await _fixture()

    async def code(encoder, **config):
        with pytest.raises(IndexingError) as exc:
            await _plugin(index, source, encoder, **config).build(scope, version)
        return exc.value.code

    assert (
        await code(FakeTokenEncoder(max_tokens=8), window_tokens=9, window_overlap_tokens=0)
        == "late_chunk_window_invalid"
    )
    assert (
        await code(FakeTokenEncoder(max_tokens=8), window_tokens=4, window_overlap_tokens=4)
        == "late_chunk_window_invalid"
    )
    assert await code(_Bad("count"), window_tokens=16, window_overlap_tokens=0) == "token_encoding_count_mismatch"
    assert await code(_Bad("offsets"), window_tokens=16, window_overlap_tokens=0) == "token_offsets_invalid"
    assert await code(_Bad("count"), mode="early") == "token_encoding_count_mismatch"
    with pytest.raises(TechniqueUnavailableError):
        await _plugin(index, source, None).build(scope, version)
    with pytest.raises(TechniqueUnavailableError):
        await _plugin(index, source, _Bad("dependency"), window_tokens=16, window_overlap_tokens=0).build(
            scope, version
        )

    # A chunk span that holds no token start (a lone space) has no vector.
    scope2, _, index2, version2 = await ready_index([(2, 3, " ")], LexicalTestEmbedder())
    view2 = version2.extraction_view_ids[0]
    source2 = StaticSourceReader(
        {(scope2.tenant_id, scope2.project_id, view2): ViewSource(extraction_view_id=view2, text="aa bb cc", blocks=())}
    )
    with pytest.raises(IndexingError) as exc:
        await _plugin(
            index2, source2, FakeTokenEncoder(max_tokens=16), window_tokens=16, window_overlap_tokens=0
        ).build(scope2, version2)
    assert exc.value.code == "late_chunk_span_empty"
    assert exc.value.ordinal == 0 and exc.value.chunk_id  # ids only, no content


def test_error_paths():
    asyncio.run(_error_paths())


class _TenantSwap:
    """Index double whose chunk records carry another tenant than the caller's scope."""

    def __init__(self, inner, tenant_id):
        self.inner, self.tenant_id = inner, tenant_id

    async def list_chunks(self, *args, **kwargs):
        rows = await self.inner.list_chunks(*args, **kwargs)
        return [r.model_copy(update={"tenant_id": self.tenant_id}) for r in rows]


async def _tenant_scope():
    scope, stranger, index, version, source = await _fixture()
    encoder = ContextEncoder(max_tokens=16)
    plugin = _plugin(index, source, encoder, window_tokens=16, window_overlap_tokens=0)
    with pytest.raises(NotFoundError):
        await plugin.build(stranger, version)  # another tenant's view and index
    assert encoder.calls == 0
    swapped = _plugin(
        _TenantSwap(index, stranger.tenant_id), source, encoder, window_tokens=16, window_overlap_tokens=0
    )
    with pytest.raises(NotFoundError):
        await swapped.build(scope, version)
    assert encoder.calls == 0

    vectors = await plugin.build(scope, version)  # built for tenant A
    calls = encoder.calls
    search = LateChunkingSearch(vectors, encoder)
    with pytest.raises(NotFoundError):
        await search.search(stranger, "aa bb")  # tenant B queries tenant A's vectors
    assert encoder.calls == calls  # refused before any encoder call
    same_tenant_other_project = stranger.model_copy(update={"tenant_id": scope.tenant_id})
    with pytest.raises(NotFoundError):
        await search.search(same_tenant_other_project, "aa bb")
    assert encoder.calls == calls


def test_tenant_scope():
    asyncio.run(_tenant_scope())


async def _search_late_and_early():
    scope, _, index, version, source = await _fixture()
    encoder = ContextEncoder(max_tokens=16)
    for mode in ("early", "late"):
        plugin = _plugin(index, source, encoder, mode=mode, window_tokens=16, window_overlap_tokens=0, k=2)
        vectors = await plugin.build(scope, version)
        result = await plugin.searcher(vectors).search(scope, "gg hh ii")
        assert result.technique_id == "C20" and result.scoring == f"c20-{mode}-mean-pool"
        assert result.index_version_id == version.id and len(result.hits) == 2
        assert [h.rank for h in result.hits] == [1, 2]
        assert result.hits[0].score >= result.hits[1].score
        assert all(h.score_components[0].name == f"{mode}_cosine" for h in result.hits)
        assert all(h.text == TEXT[h.start : h.end] for h in result.hits)
        if mode == "early":
            assert result.hits[0].text == "gg hh ii" and result.hits[0].score == pytest.approx(1.0)
    with pytest.raises(IndexingError) as exc:
        await LateChunkingSearch(vectors, encoder).search(scope, "  ")
    assert exc.value.code == "empty_query"
    with pytest.raises(TechniqueUnavailableError):
        await LateChunkingSearch(vectors, None).search(scope, "aa")
    other = FakeTokenEncoder(max_tokens=16)
    other.profile = other.profile.model_copy(update={"model": "other-model"})
    with pytest.raises(IndexingError) as exc:
        await LateChunkingSearch(vectors, other).search(scope, "aa")
    assert exc.value.code == "incompatible_embedding"


def test_search_late_and_early():
    asyncio.run(_search_late_and_early())


def _ar(*points):
    return "".join(chr(p) for p in points)


async def _arabic_view():
    words = [
        _ar(0x645, 0x631, 0x62D, 0x628, 0x627),
        _ar(0x628, 0x643, 0x645),
        _ar(0x641, 0x64A),
        _ar(0x645, 0x62E, 0x62A, 0x628, 0x631),
        _ar(0x627, 0x644, 0x628, 0x62D, 0x62B),
        _ar(0x627, 0x644, 0x630, 0x643, 0x64A),
    ]
    text = " ".join(words)
    assert any(ord(ch) > 0x600 for ch in text)
    scope, _, index, version, source = await _fixture(text=text, per_chunk=2)
    encoder = ContextEncoder(max_tokens=16)
    plugin = _plugin(index, source, encoder, window_tokens=16, window_overlap_tokens=0, k=1)
    vectors = await plugin.build(scope, version)
    assert len(vectors.chunks) == 3 and all(n == 2 for n in vectors.token_count.values())
    assert all(c.text == text[c.start : c.end] for c in vectors.chunks)
    late_hit = (await plugin.searcher(vectors).search(scope, words[4] + " " + words[5])).hits[0]
    assert late_hit.text == text[late_hit.start : late_hit.end]
    early_plugin = _plugin(index, source, encoder, mode="early", k=3)
    early_vectors = await early_plugin.build(scope, version)
    hits = (await early_plugin.searcher(early_vectors).search(scope, words[4] + " " + words[5])).hits
    exact = next(h for h in hits if h.start == text.index(words[4]))  # code-point offsets of an Arabic span
    assert exact.text == words[4] + " " + words[5] and exact.score == pytest.approx(1.0)


def test_arabic_view():
    asyncio.run(_arabic_view())


class SplitFirstWordEncoder(FakeTokenEncoder):
    """Tokenization depends on where the input starts: the first word is split into [0:2] and the rest.

    A window that starts inside a word therefore tokenizes the overlap differently from the window before."""

    async def encode_tokens(self, texts, *, head, input_type):
        result = await super().encode_tokens(texts, head=head, input_type=input_type)
        items = []
        for text in texts:
            spans = []
            for n, m in enumerate(re.finditer(r"\S+", text)):
                if n == 0 and m.end() - m.start() > 2:
                    spans += [(m.start(), m.start() + 2), (m.start() + 2, m.end())]
                else:
                    spans.append(m.span())
            truncated = len(spans) > self.max_tokens
            spans = spans[: self.max_tokens]
            vectors = tuple(
                tuple(float((sum(ord(c) for c in text[s:e]) + k) % 7) for k in range(self.dim)) for s, e in spans
            )
            items.append(TokenEncoding(vectors=vectors, offsets=tuple(spans), truncated=truncated))
        return result.model_copy(update={"items": tuple(items)})


async def _midword_retokenization_pools_each_position_once():
    text = "abcdef gh ij kl mn op qr st"
    scope, _, index, version, source = await _fixture(text=text, per_chunk=1)
    encoder = SplitFirstWordEncoder(max_tokens=10)
    # step is one token, so later windows start inside "abcdef" and re-tokenize it as "cd", "ef".
    plugin = _plugin(index, source, encoder, window_tokens=5, window_overlap_tokens=4)
    vectors = await plugin.build(scope, version)
    assert vectors.windows_used > 2
    by_ordinal = {c.ordinal: vectors.token_count[c.id] for c in vectors.chunks}
    # One whole-view tokenization has 9 tokens: "ab", "cdef" (both start in chunk 0) and 7 words.
    assert by_ordinal == {0: 2, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1}
    assert sum(by_ordinal.values()) == 9
    # The pooled vector of each later chunk is the vector the encoder gave that single token.
    for chunk in vectors.chunks:
        if chunk.ordinal:
            assert len(vectors.vectors[chunk.id]) == 4


def test_midword_retokenization_pools_each_position_once():
    asyncio.run(_midword_retokenization_pools_each_position_once())


async def _char_cap_limits_windows_without_gaps_or_duplicates():
    words = ["".join(chr(97 + (i + j) % 26) for j in range(20)) for i in range(5)]  # 20 code points each
    text = " ".join(words)
    scope, _, index, version, source = await _fixture(text=text, per_chunk=1)
    encoder = FakeTokenEncoder(max_tokens=8)
    plugin = _plugin(index, source, encoder, window_tokens=3, window_overlap_tokens=1)
    vectors = await plugin.build(scope, version)
    assert vectors.windows_used >= 4  # the 36 code point cap (3 tokens x 12) limits each window
    assert [vectors.token_count[c.id] for c in sorted(vectors.chunks, key=lambda c: c.ordinal)] == [1] * 5
    plain = FakeTokenEncoder(max_tokens=8)
    for chunk in vectors.chunks:
        whole = (await plain.encode_tokens([chunk.text], head="hidden", input_type="document")).items[0]
        assert vectors.vectors[chunk.id] == whole.vectors[0]


def test_char_cap_limits_windows_without_gaps_or_duplicates():
    asyncio.run(_char_cap_limits_windows_without_gaps_or_duplicates())


class _NoChunks:
    async def list_chunks(self, *args, **kwargs):
        return []


async def _zero_chunks_is_empty_index():
    scope, _, _, version, source = await _fixture()
    with pytest.raises(IndexingError) as exc:
        await _plugin(
            _NoChunks(), source, ContextEncoder(max_tokens=16), window_tokens=8, window_overlap_tokens=0
        ).build(scope, version)
    assert exc.value.code == "empty_index"


def test_zero_chunks_is_empty_index():
    asyncio.run(_zero_chunks_is_empty_index())
