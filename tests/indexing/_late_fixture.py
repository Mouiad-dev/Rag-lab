"""Shared fixture and runner for the C20 / R45 demo and the opt-in live comparison (P7.5-05).

Three short documents (English, Arabic, mixed) whose later chunks refer back to an entity named in an earlier
chunk with a pronoun ("She opened it ..."), which is the case late chunking is meant for. The English document
also holds a keyword-heavy distractor chunk (the losing case for any method that follows keywords). Six
questions, one of them unanswerable. Ranks are computed among the chunks of the question's own document: each
document is its own small index, so a rank here is not a corpus-wide rank. Sources are ASCII; Arabic text is
written as escapes.
"""

import math
import time
from dataclasses import dataclass

from _lexical_embedder import LexicalTestEmbedder
from rag_foundry_indexing.context_expansion import ViewSource
from rag_foundry_indexing.late_chunking import (
    LateChunkingConfig,
    LateChunkingRepresentation,
    LateChunkingSearch,
)
from rag_foundry_indexing.late_interaction import (
    LateInteractionConfig,
    LateInteractionRetriever,
    build_multivector_index,
)

from tests.indexing._inmemory_index import ready_index
from tests.indexing._source_reader import StaticSourceReader

ARMS = ("c20_early", "c20_late", "r45_single", "r45_maxsim")
K = 10  # larger than every document, so every chunk appears in every ranking


@dataclass(frozen=True)
class Doc:
    name: str
    chunks: tuple[str, ...]

    @property
    def text(self) -> str:
        return " ".join(self.chunks)

    def spans(self) -> list[tuple[int, int, str]]:
        out, pos = [], 0
        for chunk in self.chunks:
            out.append((pos, pos + len(chunk), chunk))
            pos += len(chunk) + 1
        return out


@dataclass(frozen=True)
class Question:
    id: str
    doc: str
    text: str
    relevant: int | None  # chunk ordinal inside the document; None means unanswerable


EN = Doc(
    "EN",
    (
        "Marta Okafor founded the Harbor Seed Library in Lagos.",
        "She opened it in 2009 with forty donated jars.",
        "It now lends heirloom beans to farmers across the delta.",
        "The Harbor Seed Library gift shop sells library seed tote bags and seed library posters.",
    ),
)
AR = Doc(
    "AR",
    (
        "\u0623\u0633\u0633\u062a \u0644\u064a\u0644\u0649 \u0645\u0643\u062a\u0628\u0629 "
        "\u0627\u0644\u0646\u062e\u064a\u0644 \u0641\u064a \u0635\u0648\u0631.",
        "\u0641\u062a\u062d\u062a\u0647\u0627 \u0639\u0627\u0645 2015 \u0628\u0643\u062a\u0628 "
        "\u0645\u062a\u0628\u0631\u0639 \u0628\u0647\u0627.",
        "\u0648\u062a\u0639\u064a\u0631 \u0627\u0644\u0622\u0646 \u0627\u0644\u0643\u062a\u0628 "
        "\u0644\u0644\u0637\u0644\u0627\u0628.",
    ),
)
MIXED = Doc(
    "MIXED",
    (
        "Samir Haddad curated the Cedar Archive in Beirut.",
        "He keeps old letters in a cool room \u0641\u064a \u0627\u0644\u0623\u0631\u0634\u064a\u0641.",
        "Visitors may read them on Tuesdays.",
    ),
)
DOCS = (EN, AR, MIXED)

QUESTIONS = (
    Question("EN1", "EN", "When did Marta Okafor open the library?", 1),
    Question("EN2", "EN", "What does the Harbor Seed Library lend to farmers?", 2),
    Question(
        "AR1",
        "AR",
        "\u0645\u062a\u0649 \u0641\u062a\u062d\u062a \u0644\u064a\u0644\u0649 "
        "\u0627\u0644\u0645\u0643\u062a\u0628\u0629\u061f",
        1,
    ),
    Question("MIX1", "MIXED", "Where does Samir Haddad keep the old letters?", 1),
    Question("MIX2", "MIXED", "When may visitors read the letters of the Cedar Archive?", 2),
    Question("UNANS", "EN", "What is the capital city of Mongolia?", None),
)


def _mean(vectors):
    n = len(vectors)
    return tuple(math.fsum(v[i] for v in vectors) / n for i in range(len(vectors[0])))


def _cosine(a, b):
    na, nb = math.sqrt(math.fsum(x * x for x in a)), math.sqrt(math.fsum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return math.fsum(x * y for x, y in zip(a, b, strict=True)) / (na * nb)


async def _single_vector(index, encoder, query):
    """R45 single-vector arm: mean of the same ColBERT token vectors, cosine against the query mean."""
    item = (await encoder.encode_tokens([query], head="colbert", input_type="query")).items[0]
    qvec = _mean(item.vectors)
    scored = [(_cosine(qvec, _mean(index.vectors[c.id])), c.start) for c in index.chunks]
    scored.sort(key=lambda row: (-row[0], row[1]))
    return scored


@dataclass
class DocRun:
    doc: Doc
    early: object  # ChunkVectors
    late: object  # ChunkVectors
    multivector: object  # MultivectorIndex
    build_seconds: dict
    rows: list  # one dict per question of this document


async def run_doc(encoder, doc: Doc, *, window_tokens: int, overlap: int) -> DocRun:
    """Build early and late vectors and the multivector index for one document, then answer its questions."""
    embedder = LexicalTestEmbedder()  # stores the unused dense column of the in-memory index only
    scope, _stranger, index, version = await ready_index(doc.spans(), embedder)
    view = version.extraction_view_ids[0]
    source = StaticSourceReader(
        {(scope.tenant_id, scope.project_id, view): ViewSource(extraction_view_id=view, text=doc.text, blocks=())}
    )
    seconds, built = {}, {}
    for mode in ("early", "late"):
        config = LateChunkingConfig(mode=mode, window_tokens=window_tokens, window_overlap_tokens=overlap, k=K)
        plugin = LateChunkingRepresentation(config, index=index, source=source, encoder=encoder)
        t0 = time.perf_counter()
        built[mode] = await plugin.build(scope, version)
        seconds[mode] = time.perf_counter() - t0
    t0 = time.perf_counter()
    multivector = await build_multivector_index(scope, version, index, encoder)
    seconds["multivector"] = time.perf_counter() - t0
    retriever = LateInteractionRetriever(
        LateInteractionConfig(k=K), scope=scope, index_version=version, index=multivector, encoder=encoder
    )
    rows = []
    for question in (q for q in QUESTIONS if q.doc == doc.name):
        rankings, query_seconds = {}, {}
        for mode, arm in (("early", "c20_early"), ("late", "c20_late")):
            t0 = time.perf_counter()
            result = await LateChunkingSearch(built[mode], encoder, k=K).search(scope, question.text)
            query_seconds[arm] = time.perf_counter() - t0
            rankings[arm] = [(h.score, h.start) for h in result.hits]
        t0 = time.perf_counter()
        rankings["r45_single"] = await _single_vector(multivector, encoder, question.text)
        query_seconds["r45_single"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        result = await retriever.retrieve(question.text)
        query_seconds["r45_maxsim"] = time.perf_counter() - t0
        rankings["r45_maxsim"] = [(h.score, h.start) for h in result.hits]
        # The packages break exact score ties by random chunk IDs; make the printed rank reproducible instead.
        rankings = {arm: sorted(r, key=lambda row: (-round(row[0], 9), row[1])) for arm, r in rankings.items()}
        rows.append({"question": question, "rankings": rankings, "query_seconds": query_seconds})
    return DocRun(doc, built["early"], built["late"], multivector, seconds, rows)


def rank_and_score(question: Question, doc: Doc, ranking):
    """(rank of the relevant chunk, its score), or (None, top score) for the unanswerable question."""
    if question.relevant is None:
        return None, ranking[0][0]
    start = doc.spans()[question.relevant][0]
    for rank, (score, chunk_start) in enumerate(ranking, start=1):
        if chunk_start == start:
            return rank, score
    raise AssertionError("relevant chunk missing from ranking")


def ordinal_of(doc: Doc, start: int) -> int:
    return next(i for i, span in enumerate(doc.spans()) if span[0] == start)


def format_cell(question: Question, doc: Doc, ranking) -> str:
    """Relevant rank and score (or top-1 score when unanswerable), the top-1 chunk ordinal, and the full ranking."""
    rank, score = rank_and_score(question, doc, ranking)
    head = f"top1={score:.3f}" if rank is None else f"rank={rank} score={score:.3f}"
    full = " ".join(f"{ordinal_of(doc, start)}:{value:.3f}" for value, start in ranking)
    return f"{head} top1_chunk={ordinal_of(doc, ranking[0][1])} ranking(chunk:score)={full}"
