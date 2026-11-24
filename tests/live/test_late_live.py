"""Opt-in local bge-m3 comparison: C20 early vs late, R45 single-vector vs MaxSim (CPU, no network, no download).

Prints per-question relevant rank and score for every arm plus build and query times, token counts and sizes.
Asserts structural invariants only: no quality or speed claim. One run is not an evaluation.
"""

import asyncio
import os

import pytest

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(os.environ.get("RAG_LIVE_TOKENS") != "1", reason="set RAG_LIVE_TOKENS=1"),
]


def test_late_chunking_and_late_interaction_live():
    from _late_fixture import ARMS, DOCS, QUESTIONS, format_cell, run_doc
    from rag_foundry_providers.sentence_transformers_tokens import SentenceTransformersTokenEncoder

    model_dir = os.environ.get("RAG_TOKEN_MODEL_DIR")
    if not model_dir:
        pytest.skip("set RAG_TOKEN_MODEL_DIR")
    encoder = SentenceTransformersTokenEncoder(model_dir=model_dir)

    async def run():
        runs = []
        for doc in DOCS:
            runs.append(await run_doc(encoder, doc, window_tokens=512, overlap=64))
        return runs

    runs = asyncio.run(run())
    print(f"model={encoder.profile.model} revision={encoder.revision}")
    by_doc = {r.doc.name: r for r in runs}
    for run in runs:
        stats = run.multivector.stats()
        secs = run.build_seconds
        print(
            f"DOC {run.doc.name} chunks={stats.chunks} mv_tokens={stats.tokens} dim={stats.dim} "
            f"estimated_bytes={stats.estimated_bytes} early_tokens={sum(run.early.token_count.values())} "
            f"late_tokens={sum(run.late.token_count.values())} late_windows={run.late.windows_used} "
            f"early_truncated={run.early.truncated} late_truncated={run.late.truncated} "
            f"build_s early={secs['early']:.2f} late={secs['late']:.2f} multivector={secs['multivector']:.2f}"
        )
        for chunk in run.early.chunks:
            print(
                f"  chunk {chunk.ordinal} tokens early={run.early.token_count[chunk.id]} "
                f"late={run.late.token_count[chunk.id]} mv={len(run.multivector.vectors[chunk.id])}"
            )
    for question in QUESTIONS:
        run = by_doc[question.doc]
        row = next(r for r in run.rows if r["question"].id == question.id)
        for arm in ARMS:
            print(
                f"RANK {question.id} {arm} {format_cell(question, run.doc, row['rankings'][arm])} "
                f"query_s={row['query_seconds'][arm]:.3f}"
            )

    for run in runs:
        chunks = run.early.chunks
        assert run.early.mode == "early" and run.late.mode == "late"
        assert run.early.dim == run.late.dim == run.multivector.dim == 1024
        assert run.early.windows_used == 0 and run.late.windows_used == 1
        assert not run.early.truncated and not run.late.truncated
        assert len(run.multivector.vectors) == len(chunks) == len(run.early.vectors) == len(run.late.vectors)
        for chunk in chunks:
            assert run.early.token_count[chunk.id] > 0 and run.late.token_count[chunk.id] > 0
            vectors, offsets = run.multivector.vectors[chunk.id], run.multivector.offsets[chunk.id]
            assert len(vectors) == len(offsets) == run.early.token_count[chunk.id]  # same tokenizer, same text
            assert all(0 <= s < e <= len(chunk.text) for s, e in offsets)
            assert all(len(v) == 1024 for v in vectors)
    for run in runs:
        for row in run.rows:
            for arm in ARMS:
                ranking = row["rankings"][arm]
                assert len(ranking) == len(run.early.chunks)
                assert sorted(start for _score, start in ranking) == sorted(c.start for c in run.early.chunks)

    # Guard against a question being scored with another question's vector: within a document, two
    # questions must not produce the same score for every chunk in any arm, and their query encodings differ.
    for run in runs:
        for i, first in enumerate(run.rows):
            for second in run.rows[i + 1 :]:
                for arm in ARMS:
                    a, b = (
                        dict((st, sc) for sc, st in first["rankings"][arm]),
                        dict((st, sc) for sc, st in second["rankings"][arm]),
                    )
                    assert a != b, (arm, first["question"].id, second["question"].id)
    texts = [q.text for q in QUESTIONS]
    encoded = asyncio.run(encoder.encode_tokens(texts, head="hidden", input_type="query")).items
    means = [tuple(round(sum(v[i] for v in e.vectors) / len(e.vectors), 6) for i in range(8)) for e in encoded]
    assert len(set(means)) == len(texts)
