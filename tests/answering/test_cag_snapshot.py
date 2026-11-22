import asyncio
import hashlib
import uuid
from datetime import UTC, datetime

import pytest
from _templates import prompt_templates
from rag_foundry_answering.cag import CagSnapshot, build_cag_snapshot
from rag_foundry_domain.documents import Scope
from rag_foundry_domain.errors import NotFoundError, RunError
from rag_foundry_domain.indexing import ChunkRecord, IndexState, IndexVersion
from rag_foundry_providers.capabilities import Capabilities, ModelKind, ModelProfile

NONCE = "0123456789abcdef"
ARABIC = "\u0645\u0631\u062d\u0628\u0627 \u0628\u0627\u0644\u0639\u0627\u0644\u0645"
GROUNDED = prompt_templates()[0]


def _sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


class FakeIndex:
    def __init__(self, scope, texts, *, state=IndexState.READY):
        self.scope = scope
        self.version_id = str(uuid.uuid4())
        view_ids = sorted(str(uuid.uuid4()) for _ in range(2))
        self.view_ids = view_ids
        config = "{}"
        self.version = IndexVersion(
            chunker_id="C01",
            chunker_version="1.0.0",
            chunker_config_json=config,
            chunker_config_sha256=_sha(config),
            embedding_profile_id="emb",
            embedding_provider="test",
            embedding_model="m",
            embedding_dim=4,
            distance="cosine",
            embedding_key=_sha("key"),
            extraction_view_ids=tuple(view_ids),
            corpus_sha256=_sha("corpus"),
            unanchored_chunk_count=0,
            id=self.version_id,
            tenant_id=scope.tenant_id,
            project_id=scope.project_id,
            state=state,
            chunk_count=len(texts),
            created_at=datetime.now(UTC),
            ready_at=datetime.now(UTC),
        )
        pos = 0
        self.chunks = []
        for ordinal, text in enumerate(texts):
            self.chunks.append(
                ChunkRecord(
                    extraction_view_id=view_ids[ordinal % 2],
                    ordinal=ordinal,
                    start=pos,
                    end=pos + len(text),
                    split_level=0,
                    block_ids=("b000001",),
                    text=text,
                    text_sha256=_sha(text),
                    id=str(uuid.uuid4()),
                    index_version_id=self.version_id,
                    tenant_id=scope.tenant_id,
                    project_id=scope.project_id,
                )
            )
            if ordinal % 2:
                pos += len(text)

    async def get_index_version(self, scope, index_version_id):
        if scope != self.scope or index_version_id != self.version_id:
            raise NotFoundError
        return self.version

    async def list_chunks(self, scope, index_version_id, *, offset=0, limit=100):
        await self.get_index_version(scope, index_version_id)
        return list(reversed(self.chunks))[offset : offset + limit]  # shuffled on purpose


def _scope():
    return Scope(tenant_id=str(uuid.uuid4()), project_id=str(uuid.uuid4()))


def _profile(window=8192):
    return ModelProfile(
        id="chat",
        provider="test",
        model="chat-m",
        kind=ModelKind.CHAT,
        capabilities=Capabilities(context_window=window),
    )


def _build(index, scope, *, window=8192, cap=5000, factory=lambda: NONCE, reserve=256, question=64):
    return asyncio.run(
        build_cag_snapshot(
            scope,
            index,
            index.version_id,
            template=GROUNDED,
            chat_profile=_profile(window),
            reserve_output_tokens=reserve,
            max_question_tokens=question,
            max_evidence_chars=cap,
            nonce_factory=factory,
        )
    )


def _texts(n=3):
    return [f"chunk number {i} about topic {i}." for i in range(n)]


def test_fixed_order_markers_and_stable_id():
    scope = _scope()
    index = FakeIndex(scope, _texts())
    a = _build(index, scope)
    b = _build(index, scope)
    assert [i.marker for i in a.items] == ["S1", "S2", "S3"]
    # chunks alternate between two views; view-first order puts view 0 (even ordinals) before view 1
    assert [i.text for i in a.items] == [_texts()[0], _texts()[2], _texts()[1]]
    assert [i.extraction_view_id for i in a.items] == [index.view_ids[0]] * 2 + [index.view_ids[1]]
    assert a.snapshot_id == b.snapshot_id
    assert len(a.snapshot_id) == 64
    a.check_scope(scope, index.version_id)


def test_cap_change_and_rebuilt_index_change_id():
    scope = _scope()
    index = FakeIndex(scope, _texts())
    base = _build(index, scope)
    assert _build(index, scope, cap=6000).snapshot_id != base.snapshot_id
    assert _build(FakeIndex(scope, _texts()), scope).snapshot_id != base.snapshot_id
    assert _build(index, scope, factory=lambda: "fedcba9876543210").snapshot_id != base.snapshot_id


def test_default_factory_gives_different_nonces():
    scope = _scope()
    index = FakeIndex(scope, _texts())
    a = asyncio.run(_default(index, scope))
    b = asyncio.run(_default(index, scope))
    assert a.nonce != b.nonce
    assert a.snapshot_id != b.snapshot_id


async def _default(index, scope):
    return await build_cag_snapshot(
        scope,
        index,
        index.version_id,
        template=GROUNDED,
        chat_profile=_profile(),
        reserve_output_tokens=256,
        max_question_tokens=64,
        max_evidence_chars=5000,
    )


def test_nonce_redrawn_when_in_chunk_text():
    scope = _scope()
    index = FakeIndex(scope, [f"has {NONCE} inside", "other"])
    draws = iter([NONCE, "aaaaaaaaaaaaaaaa"])
    snap = _build(index, scope, factory=lambda: next(draws))
    assert snap.nonce == "aaaaaaaaaaaaaaaa"
    with pytest.raises(RuntimeError) as err:
        _build(index, scope, factory=lambda: NONCE)
    assert NONCE not in str(err.value)


def test_nonce_never_leaks():
    scope = _scope()
    snap = _build(FakeIndex(scope, _texts()), scope)
    assert NONCE not in repr(snap)
    assert NONCE not in str(snap)
    assert "nonce" not in snap.model_dump(by_alias=True)
    assert NONCE not in snap.model_dump_json()
    assert NONCE not in snap.model_dump_json(by_alias=True)
    assert "nonce" not in snap.model_dump()
    assert '"schema":"rag_foundry.cag_snapshot/1"' in snap.model_dump_json(by_alias=True)
    assert NONCE not in snap.snapshot_id
    assert not hasattr(snap, "to_json")
    assert isinstance(snap, CagSnapshot)


def test_errors_are_explicit_and_content_free():
    scope = _scope()
    secret_texts = ["SECRETCHUNKTEXT " * 50]
    with pytest.raises(RunError) as err:
        _build(FakeIndex(scope, secret_texts), scope, cap=500)
    assert err.value.code == "cag_overflow"
    assert "evidence_chars=800/500" in str(err.value)
    assert "SECRET" not in str(err.value) and NONCE not in str(err.value)

    with pytest.raises(RunError) as err:
        _build(FakeIndex(scope, ["x" * 10] * 100), scope)
    assert err.value.code == "cag_too_many_chunks"
    assert _build(FakeIndex(scope, ["x" * 10] * 99), scope).items[-1].marker == "S99"

    with pytest.raises(RunError) as err:
        _build(FakeIndex(scope, ["word " * 20] * 3), scope, window=300)
    assert err.value.code == "cag_overflow" and "prompt_tokens=" in str(err.value)
    assert NONCE not in str(err.value)

    with pytest.raises(RunError) as err:
        _build(FakeIndex(scope, _texts()), scope, window=None)
    assert err.value.code == "cag_overflow"


def test_empty_index_fails_cag_empty():
    scope = _scope()
    with pytest.raises(RunError) as err:
        _build(FakeIndex(scope, []), scope)
    assert err.value.code == "cag_empty"


def test_not_ready_and_stranger_scope():
    scope = _scope()
    with pytest.raises(RunError) as err:
        _build(FakeIndex(scope, _texts(), state=IndexState.BUILDING), scope)
    assert err.value.code == "cag_snapshot_mismatch"
    index = FakeIndex(scope, _texts())
    with pytest.raises(NotFoundError):
        _build(index, _scope())
    snap = _build(index, scope)
    with pytest.raises(RunError) as err:
        snap.check_scope(_scope(), index.version_id)
    assert err.value.code == "cag_snapshot_mismatch"
    with pytest.raises(RunError):
        snap.check_scope(scope, str(uuid.uuid4()))


@pytest.mark.parametrize("cap", [499, 100001])
def test_cap_bounds_rejected(cap):
    scope = _scope()
    with pytest.raises(ValueError):
        _build(FakeIndex(scope, _texts()), scope, cap=cap)


def test_arabic_chunk_survives():
    scope = _scope()
    snap = _build(FakeIndex(scope, [ARABIC, "english"]), scope)
    assert ARABIC in [i.text for i in snap.items]


@pytest.mark.parametrize(
    "kwargs",
    [{"reserve": -1}, {"question": -1}],
)
def test_negative_token_bounds_rejected(kwargs):
    scope = _scope()
    with pytest.raises(ValueError):
        _build(FakeIndex(scope, _texts()), scope, **kwargs)


def test_margin_below_one_rejected():
    scope = _scope()
    index = FakeIndex(scope, _texts())
    with pytest.raises(ValueError):
        asyncio.run(
            build_cag_snapshot(
                scope,
                index,
                index.version_id,
                template=GROUNDED,
                chat_profile=_profile(),
                reserve_output_tokens=1,
                max_question_tokens=1,
                max_evidence_chars=5000,
                estimate_margin=0.5,
            )
        )
