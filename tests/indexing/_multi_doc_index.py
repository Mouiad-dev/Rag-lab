"""Small two-scope corpus used by metadata filtering tests."""

import hashlib
from datetime import UTC, datetime
from types import SimpleNamespace

from rag_foundry_domain.documents import MediaType
from rag_foundry_domain.index_testing import InMemoryIndexRepository, make_new_index_version
from rag_foundry_domain.indexing import NewChunk, embedding_key_for
from rag_foundry_domain.testing import (
    InMemoryDocumentRepository,
    InMemoryExtractionRepository,
    make_new_extraction,
    make_new_version,
    make_scope,
)


async def multi_doc_index(embedder):
    owner, stranger = make_scope(), make_scope()
    tick = 0

    def clock():
        nonlocal tick
        tick += 1
        return datetime(2026, tick, 1, tzinfo=UTC)

    documents = InMemoryDocumentRepository([owner, stranger], clock=clock)
    extractions = InMemoryExtractionRepository(documents)
    index = InMemoryIndexRepository(extractions)
    tags = {}
    views = []
    for tag, media in (
        ("report-2024", MediaType.TEXT_MARKDOWN),
        ("report-2025", MediaType.TEXT_PLAIN),
        ("notes", MediaType.TEXT_HTML),
    ):
        source = make_new_version(tag, f"{tag}.txt").model_copy(update={"media_type": media})
        intake = await documents.create_document_or_get(owner, source, display_name=tag)
        view = (await extractions.create_view_or_get(owner, make_new_extraction(intake.version, tag=tag))).view
        tags[tag] = intake.document.id
        views.append(view)
    stranger_source = make_new_version("stranger", "stranger.txt")
    stranger_doc = await documents.create_document_or_get(stranger, stranger_source, display_name="stranger")
    stranger_view = (
        await extractions.create_view_or_get(stranger, make_new_extraction(stranger_doc.version, tag="stranger"))
    ).view
    tags["stranger"] = stranger_doc.document.id

    async def build(scope, selected):
        profile = embedder.profile
        dim = profile.capabilities.embedding_dim
        config = make_new_index_version(tuple(v.id for v in selected), dim=dim).model_copy(
            update={
                "embedding_profile_id": profile.id,
                "embedding_provider": profile.provider,
                "embedding_model": profile.model,
                "embedding_key": embedding_key_for(profile.provider, profile.model, dim),
            }
        )
        version = await index.create_index_version(scope, config)
        chunks = []
        for view in selected:
            for text in ("quarterly report shared words", "financial summary shared words"):
                embedded = await embedder.embed([text], input_type="document")
                chunks.append(
                    NewChunk(
                        extraction_view_id=view.id,
                        ordinal=len(chunks),
                        start=0,
                        end=len(text),
                        split_level=0,
                        block_ids=("b00000",),
                        text=text,
                        text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                        embedding=embedded.vectors[0],
                    )
                )
        await index.add_chunks(scope, version.id, tuple(chunks))
        return await index.mark_ready(scope, version.id)

    owner_version = await build(owner, views)
    stranger_version = await build(stranger, [stranger_view])
    return SimpleNamespace(
        owner=owner,
        stranger=stranger,
        documents=documents,
        extractions=extractions,
        index=index,
        owner_version=owner_version,
        stranger_version=stranger_version,
        document_ids=tags,
    )
