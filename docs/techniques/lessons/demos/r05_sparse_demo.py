"""Print a small, opt-in inspection of the local R05 sparse model."""

import asyncio
import json
import os
import re
import sys
from pathlib import Path

SKIP = "SKIP: set RAG_SPARSE_MODEL_DIR to the approved local model dir (R05 needs the [sparse] extra)"

DOCS = [
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

QUERIES = [
    ("EN", "Where was the basalt sample collected?", 0),
    (
        "AR",
        "\u0645\u062a\u0649 \u0648\u0635\u0644\u062a \u0627\u0644\u0642\u0627\u0641\u0644\u0629 "
        "\u0625\u0644\u0649 \u0645\u062f\u064a\u0646\u0629 \u0635\u0648\u0631\u061f",
        1,
    ),
    ("EN paraphrase", "Which volcanic rock was gathered close to the icy north?", 0),
    ("Shared words, wrong fact", "Where was the copper sample collected?", None),
    ("Negation", "Which sample was not collected near the glacier?", None),
    (
        "AR plural",
        "\u0645\u062a\u0649 \u0648\u0635\u0644\u062a \u0627\u0644\u0642\u0648\u0627\u0641\u0644 "
        "\u0625\u0644\u0649 \u0645\u062f\u064a\u0646\u0629 \u0635\u0648\u0631\u061f",
        1,
    ),
    (
        "AR with diacritics",
        "\u0645\u062a\u0649 \u0648\u0635\u0644\u062a \u0627\u0644\u0642\u064e\u0627\u0641\u0650\u0644\u064e\u0629 "
        "\u0625\u0644\u0649 \u0645\u064e\u062f\u0650\u064a\u0646\u064e\u0629 \u0635\u0648\u0631\u061f",
        1,
    ),
]


def _revision(model_dir: Path) -> str:
    config_path = model_dir / "config.json"
    if config_path.is_file():
        try:
            revision = json.loads(config_path.read_text(encoding="utf-8")).get("_commit_hash")
        except (OSError, json.JSONDecodeError):
            revision = None
        if revision:
            return str(revision)
    revision_path = model_dir / "revision"
    if revision_path.is_file():
        try:
            return revision_path.read_text(encoding="utf-8").strip() or "unknown"
        except OSError:
            pass
    return "not recorded"


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.casefold(), flags=re.UNICODE))


def _own_tokens(tokenizer, text: str) -> set[str]:
    return _tokens(text) | set(tokenizer.tokenize(text))


def _dot(query: dict[str, float], document: dict[str, float]) -> float:
    return sum(weight * document.get(term, 0.0) for term, weight in query.items())


async def _run(model_dir: Path) -> None:
    from rag_foundry_providers.sentence_transformers_sparse import SentenceTransformersSparseEncoder

    encoder = SentenceTransformersSparseEncoder(model_dir=model_dir)
    try:
        encoded_docs = await encoder.encode(DOCS, input_type="document")
    except Exception as exc:
        from rag_foundry_providers.errors import AdapterDependencyError

        if isinstance(exc, (ImportError, AdapterDependencyError)):
            print(f"{SKIP}: {type(exc).__name__}")
            return
        raise

    vectors = [dict(zip(vector.terms, vector.weights, strict=True)) for vector in encoded_docs.vectors]
    avg_term_bytes = sum(len(term.encode("utf-8")) for vector in encoded_docs.vectors for term in vector.terms) / max(
        1, sum(len(vector.terms) for vector in encoded_docs.vectors)
    )
    tokenizer = encoder._encoder.tokenizer  # loaded by the first encode() call above
    own_tokens = [_own_tokens(tokenizer, text) for text in DOCS]
    counts = [len(vector) for vector in vectors]
    estimated = [round(count * (avg_term_bytes + 4)) for count in counts]
    print(f"# R05 learned sparse demo: {model_dir.name} ({_revision(model_dir)})")
    print("One tiny local fixture (8 docs); not a benchmark.")
    print("\n## Memory estimate")
    print("Estimate for term-string plus float32 storage; not measured DB size.")
    print("Compare with a 1024-dim float32 dense vector: 4096 bytes.")
    print("\ndoc | nonzero terms | estimated bytes")
    print("---:|---:|---:")
    for i, (count, size) in enumerate(zip(counts, estimated, strict=True)):
        print(f"{i} | {count} | {size}")

    for label, query_text, expected in QUERIES:
        encoded_query = await encoder.encode([query_text], input_type="query")
        query = dict(zip(encoded_query.vectors[0].terms, encoded_query.vectors[0].weights, strict=True))
        scores = [_dot(query, vector) for vector in vectors]
        ranking = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        rank_by_doc = {doc: rank for rank, doc in enumerate(ranking, start=1)}
        print(f"\n## {label}: {query_text}")
        print(
            "rank | doc | score | top 5 matched terms (term:q_weight*d_weight) | "
            "terms the doc vector adds beyond its text"
        )
        print("---:|---:|---:|---|---:")
        for rank, doc_id in enumerate(ranking, start=1):
            doc = vectors[doc_id]
            matches = sorted(
                ((term, weight * doc[term], weight, doc[term]) for term, weight in query.items() if term in doc),
                key=lambda item: item[1],
                reverse=True,
            )
            terms = ", ".join(f"{term}:{q_weight:.3f}*{d_weight:.3f}" for term, _, q_weight, d_weight in matches[:5])
            additions = sum(term not in own_tokens[doc_id] for term, _, _, _ in matches)
            print(f"{rank} | {doc_id} | {scores[doc_id]:.3f} | {terms or '(none)'} | {additions}")
        expected_label = "none (top doc shown above)" if expected is None else str(expected)
        expected_rank = "n/a" if expected is None else str(rank_by_doc[expected])
        print(f"expected doc: {expected_label}; rank of expected doc: {expected_rank}")


def main() -> int:
    configured = os.environ.get("RAG_SPARSE_MODEL_DIR")
    model_dir = Path(configured) if configured else None
    if model_dir is None or not model_dir.is_dir():
        print(SKIP)
        return 0
    asyncio.run(_run(model_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
