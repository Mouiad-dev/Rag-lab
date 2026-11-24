"""Offline corpus helpers for frozen runner tests."""

import hashlib

from rag_foundry_evaluation import BoundDocument, EvalCase, EvalDataset, EvalDocument, RelevantSpan

from tests.indexing._inmemory_index import ready_index
from tests.indexing._paired_fixture import CASES, DOC, SPANS

DOC_KEY = "paired-fixture"


def build_dataset(rows):
    examples = {row[0]: row for row in CASES}
    cases = []
    for case_id, language, tags, split in rows:
        _, _, _, question, paragraph = examples[case_id]
        relevant = ()
        if paragraph is not None:
            start = DOC.index(paragraph)
            relevant = (
                RelevantSpan(doc_key=DOC_KEY, start=start, end=start + len(paragraph), grade=1, quote=paragraph),
            )
        cases.append(
            EvalCase(
                case_id=case_id,
                question=question,
                language=language,
                tags=tuple(sorted(tags)),
                split=split,
                answerable=paragraph is not None,
                relevant=relevant,
            )
        )
    doc = EvalDocument(doc_key=DOC_KEY, text_sha256=hashlib.sha256(DOC.encode("utf-8")).hexdigest(), length=len(DOC))
    return EvalDataset(
        format="rag-foundry.eval-dataset",
        format_version="1",
        dataset_id="paired",
        dataset_version="1.0",
        description="offline paired fixture",
        documents=(doc,),
        cases=tuple(cases),
    )


def make_factory(owner, index, version, embedder, calls=None, source=None):
    from rag_foundry_indexing import (
        AutoMergingExpander,
        BM25Retriever,
        DenseConfig,
        DenseRetriever,
        ExpandingRetriever,
        HybridRetriever,
        ParentDocumentExpander,
        SentenceWindowExpander,
    )
    from rag_foundry_providers.ledger import RecordingEmbeddingModel

    def factory(resolved, config, sinks):
        if calls is not None:
            calls.append(resolved.label)
        if resolved.technique_id == "R03":
            return BM25Retriever(config, scope=owner, index_version=version, index=index)
        wrapped = RecordingEmbeddingModel(embedder, sinks.sink("query_embedding"))
        if resolved.technique_id == "R02":
            return DenseRetriever(config, scope=owner, index_version=version, index=index, embedder=wrapped)
        if resolved.technique_id == "R06":
            return HybridRetriever(config, scope=owner, index_version=version, index=index, embedder=wrapped)
        expanders = {"R21": SentenceWindowExpander, "R22": ParentDocumentExpander, "R25": AutoMergingExpander}
        if resolved.technique_id in expanders:
            # The base retriever is fixed to R02 k=5 here; it is not part of RunConfig.
            base = DenseRetriever(DenseConfig(k=5), scope=owner, index_version=version, index=index, embedder=wrapped)
            expander = expanders[resolved.technique_id](
                config, scope=owner, index_version=version, index=index, source=source
            )
            return ExpandingRetriever(base, expander)
        raise AssertionError(resolved.technique_id)

    return factory


async def bound_index(embedder, spans=SPANS):
    owner, _, index, version = await ready_index(spans, embedder)
    # The fixture has one extraction view; obtain it from the first stored chunk.
    chunks = await index.list_chunks(owner, version.id, limit=1)
    binding = {DOC_KEY: BoundDocument(extraction_view_id=chunks[0].extraction_view_id, text=DOC)}
    return (owner, index, version), binding


def sentence_spans():
    from rag_foundry_chunking.boundaries import sentence_ranges

    return tuple((start, end, DOC[start:end]) for start, end in sentence_ranges(DOC, 0, len(DOC)))


def c18_spans():
    from rag_foundry_chunking.hierarchy import ParentChildChunker, ParentChildConfig
    from rag_foundry_techniques import ChunkInput

    chunker = ParentChildChunker(ParentChildConfig(parent_size=200, child_size=50))
    return tuple(
        (span.start, span.end, DOC[span.start : span.end], span.parent_start, span.parent_end)
        for span in chunker.chunk_with_parents(ChunkInput(text=DOC))
    )
