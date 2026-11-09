"""Opt-in local-model retrieval quality check using sparse dot product."""

import asyncio
import os

import pytest

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(os.environ.get("RAG_LIVE_SPARSE") != "1", reason="set RAG_LIVE_SPARSE=1"),
]


def test_local_sparse_model_english_and_arabic():
    from rag_foundry_providers.sentence_transformers_sparse import SentenceTransformersSparseEncoder

    model_dir = os.environ.get("RAG_SPARSE_MODEL_DIR")
    if not model_dir:
        pytest.skip("set RAG_SPARSE_MODEL_DIR")
    encoder = SentenceTransformersSparseEncoder(model_dir=model_dir)
    docs = [
        "The basalt sample was collected near the northern glacier.",
        (
            "\u0648\u0635\u0644\u062a \u0627\u0644\u0642\u0627\u0641\u0644\u0629 \u0625\u0644\u0649 "
            "\u0645\u062f\u064a\u0646\u0629 \u0635\u0648\u0631 \u0628\u0639\u062f "
            "\u0627\u0644\u063a\u0631\u0648\u0628."
        ),
        "A copper kettle was polished before the ceremony.",
        "The observatory records distant radio signals each night.",
        "Wild orchids bloom beside the limestone footpath.",
        "A ceramic bowl cracked during the winter shipment.",
        "The violin maker stores maple wood in a dry room.",
        "Fresh apricots were packed into blue market baskets.",
    ]
    queries = (
        "Where was the basalt sample collected?",
        (
            "\u0645\u062a\u0649 \u0648\u0635\u0644\u062a \u0627\u0644\u0642\u0627\u0641\u0644\u0629 "
            "\u0625\u0644\u0649 \u0645\u062f\u064a\u0646\u0629 \u0635\u0648\u0631\u061f"
        ),
    )

    async def run():
        encoded_docs = await encoder.encode(docs, input_type="document")
        for query, expected in zip(queries, (0, 1), strict=True):
            encoded_query = await encoder.encode([query], input_type="query")
            q = dict(zip(encoded_query.vectors[0].terms, encoded_query.vectors[0].weights, strict=True))
            scores = [
                sum(
                    weight * dict(zip(vector.terms, vector.weights, strict=True)).get(term, 0.0)
                    for term, weight in q.items()
                )
                for vector in encoded_docs.vectors
            ]
            ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            print(f"expected={expected} ranked={ranked} scores={[round(s, 3) for s in scores]}")
            # Top-1 among 8 docs; top-5 would pass a random ranking about 62% of the time.
            assert ranked[0] == expected
            assert scores[expected] > 0.0

    asyncio.run(run())
