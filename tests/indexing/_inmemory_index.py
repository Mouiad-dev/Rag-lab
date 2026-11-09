"""Reusable ready in-memory index fixture for retrieval tests."""

import hashlib

from rag_foundry_domain.index_testing import InMemoryIndexRepository, make_new_index_version
from rag_foundry_domain.indexing import NewChunk, embedding_key_for
from rag_foundry_domain.testing import (
    InMemoryDocumentRepository,
    InMemoryExtractionRepository,
    make_new_extraction,
    make_new_version,
    make_scope,
)


async def ready_index(paragraph_spans, embedder, *, ready=True):
    owner, stranger = make_scope(), make_scope()
    documents = InMemoryDocumentRepository([owner, stranger])
    views = InMemoryExtractionRepository(documents)
    index = InMemoryIndexRepository(views)
    document = await documents.create_document_or_get(owner, make_new_version("hybrid"), display_name="fixture")
    view = (await views.create_view_or_get(owner, make_new_extraction(document.version, tag="hybrid"))).view
    profile = embedder.profile
    dim = profile.capabilities.embedding_dim
    version_config = make_new_index_version((view.id,), dim=dim).model_copy(
        update={
            "embedding_profile_id": profile.id,
            "embedding_provider": profile.provider,
            "embedding_model": profile.model,
            "embedding_key": embedding_key_for(profile.provider, profile.model, dim),
        }
    )
    version = await index.create_index_version(owner, version_config)
    chunks = []
    for ordinal, (start, end, text, *parent) in enumerate(paragraph_spans):
        embedded = await embedder.embed([text], input_type="document")
        chunks.append(
            NewChunk(
                extraction_view_id=view.id,
                ordinal=ordinal,
                start=start,
                end=end,
                split_level=0,
                block_ids=("b00000",),
                text=text,
                text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                embedding=embedded.vectors[0],
                parent_start_cp=parent[0] if parent else None,
                parent_end_cp=parent[1] if parent else None,
            )
        )
    await index.add_chunks(owner, version.id, tuple(chunks))
    if ready:
        version = await index.mark_ready(owner, version.id)
    return owner, stranger, index, version
