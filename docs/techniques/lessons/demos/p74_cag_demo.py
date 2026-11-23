# ruff: noqa: UP031
"""P7.4 lesson demo: R47 CAG snapshot, prefix stability and citations, against full-context-uncached and R02.

Offline. Answers are scripted chat replies (not a model); the embedder is the test-only lexical vectorizer; the store
is in memory. No run, chunk, snapshot or nonce value and no timing is printed, so the output is byte-identical between
runs. R49 (prefix reuse) has no offline demo: a scripted chat cannot show it.
"""

import asyncio
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
for rel in ("tests/answering", "tests/cache", "tests/indexing", "tests/ingestion"):
    sys.path.insert(0, str(ROOT / rel))

from _fixture import FACT_AR, FACT_EN, TEST_CHUNK_SIZE, TEST_OVERLAP  # noqa: E402
from _lexical_embedder import LexicalTestEmbedder  # noqa: E402
from _qa import corpus_markdown  # noqa: E402
from _scripted_chat import ScriptedChatModel, cite_where  # noqa: E402
from _templates import prompt_templates  # noqa: E402
from rag_foundry_answering.cag import CagConfig, CagWorkflow, build_cag_snapshot  # noqa: E402
from rag_foundry_answering.service import AnswerConfig, AnswerService, RunPrompts  # noqa: E402
from rag_foundry_chunking import RecursiveChunker, RecursiveConfig  # noqa: E402
from rag_foundry_domain.errors import RunError  # noqa: E402
from rag_foundry_domain.index_testing import InMemoryIndexRepository  # noqa: E402
from rag_foundry_domain.testing import make_scope  # noqa: E402
from rag_foundry_domain.uow_testing import InMemoryUnitOfWorkFactory  # noqa: E402
from rag_foundry_indexing import DenseConfig, DenseRetriever, IndexService  # noqa: E402
from rag_foundry_ingestion.anchor_service import AnchorService  # noqa: E402
from rag_foundry_providers.capabilities import Capabilities, ModelKind, ModelProfile  # noqa: E402
from rag_foundry_providers.ledger import RecordingChatModel, RecordingEmbeddingModel, UsageLedger  # noqa: E402
from rag_foundry_providers.usage import Operation  # noqa: E402
from test_cag_workflow import AR_Q, EN_Q  # noqa: E402  (question texts shared with the tests)
from test_extraction_service import ingest, setup  # noqa: E402

OUT = []
PROMPTS = RunPrompts(*prompt_templates())
QUESTIONS = (
    ("EN", EN_Q, cite_where(FACT_EN, FACT_EN)),
    ("AR", AR_Q, cite_where(FACT_AR, FACT_AR)),
    ("out-of-corpus", "What colour is the moon base?", cite_where("no such fact", "x")),
)


def emit(line=""):
    OUT.append(line)


def table(header, rows):
    emit("| " + " | ".join(header) + " |")
    emit("|" + "---|" * len(header))
    for row in rows:
        emit("| " + " | ".join(str(cell) for cell in row) + " |")
    emit()


def profile(window):
    caps = Capabilities(context_window=window, local=True)
    return ModelProfile(id="cag-chat", provider="fake", model="fake-chat-v1", kind=ModelKind.CHAT, capabilities=caps)


async def build(markdown, folder):
    """Ingest, chunk and index one corpus in memory; returns (scope, index, anchors, embedder, version)."""
    scope = make_scope()
    extraction, intake, _spy, store, _docs, views, _ = setup(folder, scope)
    doc_id, ver_id = await ingest(intake, scope, markdown.encode("utf-8"), name="notes.md", media="text/markdown")
    view = (await extraction.extract(scope, doc_id, ver_id)).view
    index = InMemoryIndexRepository(views)
    anchors = AnchorService(store, InMemoryUnitOfWorkFactory(extractions=views), provider_namespace="test")
    embedder = LexicalTestEmbedder()
    chunker = RecursiveChunker(RecursiveConfig(chunk_size=TEST_CHUNK_SIZE, chunk_overlap=TEST_OVERLAP))
    uow = InMemoryUnitOfWorkFactory(extractions=views, indexes=index)
    return (
        scope,
        index,
        anchors,
        embedder,
        await IndexService(store, uow, anchors).build(scope, (view.id,), chunker, embedder),
    )


def snapshot(env, window=200_000, cap=100_000):
    scope, index, _anchors, _emb, version = env
    return build_cag_snapshot(
        scope,
        index,
        version.id,
        template=PROMPTS.grounded,
        chat_profile=profile(window),
        reserve_output_tokens=512,
        max_question_tokens=64,
        max_evidence_chars=cap,
        nonce_factory=lambda: "0123456789abcdef",
    )


def prefix(request):
    user = request.messages[1].content
    return request.messages[0].content + "\n" + user[: user.rindex("Question:\n<<<QUESTION ")]


async def ask(env, arm, snap, question, responder):
    """One call through R47 (arm cag or full_context_uncached) or R02 (arm r02); returns a row and the prefix text."""
    scope, index, anchors, embedder, version = env
    ledger, chat = UsageLedger(), ScriptedChatModel(responder, profile(200_000))
    recorded = RecordingChatModel(chat, ledger)
    if arm == "r02":
        retriever = DenseRetriever(
            DenseConfig(k=3),
            scope=scope,
            index_version=version,
            index=index,
            embedder=RecordingEmbeddingModel(embedder, ledger),
        )
        service = AnswerService(anchors, nonce_factory=lambda: "0123456789abcdef")
        result = await service.answer(
            scope,
            question,
            prompts=PROMPTS,
            retriever=retriever,
            technique_version="1.0.0",
            index_version=version,
            chat=recorded,
            ledger=ledger,
            config=AnswerConfig(min_score=0.0, k=3),
        )
        key = prefix(chat.requests[0])
    else:
        flow = CagWorkflow(CagConfig(arm=arm), snapshot=snap, anchors=anchors, index=index)
        result = await flow.answer(scope, question, prompts=PROMPTS, chat=recorded, ledger=ledger)
        key = flow.last_trace["prefix_sha256"]
    cited = ", ".join(c.marker for c in result.citations) or "-"
    status = result.status.value if hasattr(result.status, "value") else result.status
    return (
        status,
        result.abstention_reason or "-",
        cited,
        ledger.totals().calls,
        ledger.totals(operation=Operation.CHAT).input_tokens,
    ), key


async def main():
    sys.stdout.reconfigure(encoding="utf-8", newline="\n")
    emit("# P7.4 R47 CAG: demo output")
    emit()
    emit(
        "Command: `uv run python docs/techniques/lessons/demos/p74_cag_demo.py`. Date: 2026-11-23. Offline; scripted "
        "answers; shows snapshot bounds, prefix stability and citations, not speed or answer quality; prefix reuse "
        "needs a real runtime (see live runs). R49 has no offline demo: a scripted chat cannot show prefix reuse "
        "(R49 evidence is `prefix_anthropic_live.md` from P7.4-03a and the live section of P7.4-11)."
    )
    emit()
    emit(
        "Scripted chat (fixed replies, not a model), test-only lexical embedder, in-memory store. Chunk, snapshot and "
        "run IDs, keys and the nonce are never printed. Token figures are the scripted model's estimates, not provider "
        "counts. Cold versus warm is shown only as calls avoided in the response-cache demo. "
        "None of this is a benchmark."
    )
    emit()
    with tempfile.TemporaryDirectory() as tmp:
        env = await build(corpus_markdown(), Path(tmp) / "a")
        snap = await snapshot(env)
        emit("## 1. Snapshot")
        emit()
        table(
            ("item", "value"),
            [
                ("chunks in the snapshot (S1..Sn, fixed order)", len(snap.items)),
                ("estimated prompt tokens (lab-regex-v1, without question)", snap.estimated_prompt_tokens),
                ("context window (tokens)", snap.context_window),
                ("estimate margin", snap.estimate_margin),
                ("reserved output tokens", snap.reserve_output_tokens),
                ("reserved question tokens", snap.max_question_tokens),
                ("max_evidence_chars (cap)", snap.max_evidence_chars),
                ("evidence chars in use", sum(len(i.text) for i in snap.items)),
            ],
        )
        rows = []
        for title, kwargs in (("window 300 tokens", {"window": 300}), ("evidence cap 500 chars", {"cap": 500})):
            try:
                await snapshot(env, **kwargs)
                rows.append((title, "built", "-"))
            except RunError as err:
                rows.append((title, err.code, str(err)))
        table(("overflow case", "refusal code", "numbers (needed/limit)"), rows)
        emit(
            "Nothing is selected or truncated: a corpus that does not fit is refused at build time, "
            "before any chat call. The refusal figure is the full prompt estimate including margin and reserve, "
            "so it is larger than the snapshot estimate above; the per-call input tokens in section 2 come from "
            "the scripted chat model, a different estimator."
        )
        emit()

        emit("## 2. Arms on three questions")
        emit()
        emit(
            "Each arm runs as a block: all `cag`, then all `full_context_uncached`, then R02 with k=3. Calls come from "
            "the answer ledger (R02 includes one query-embedding call); input tokens are the scripted chat estimate. "
            "The out-of-corpus reply is the scripted `INSUFFICIENT_EVIDENCE`, so abstaining there is scripted. "
            "Markers are numbered within each prompt: `S1` in R02 is not `S1` in the CAG snapshot."
        )
        emit()
        rows, same = [], {}
        for arm in ("cag", "full_context_uncached", "r02"):
            keys = []
            for label, question, responder in QUESTIONS:
                row, key = await ask(env, arm, snap, question, responder)
                rows.append((arm, label) + row)
                keys.append(key)
            same[arm] = "yes" if len(set(keys)) == 1 else "no"
        table(
            (
                "arm",
                "question",
                "status",
                "abstention reason",
                "cited markers",
                "ledger calls",
                "input tokens (scripted)",
            ),
            rows,
        )
        table(("arm", "prefix identical across questions"), list(same.items()))
        emit(
            "`cag` keeps one snapshot nonce, so everything before the question is byte-identical and a runtime that "
            "supports prefix caching could reuse it (not shown here). `full_context_uncached` draws a fresh nonce per "
            "call, so its prefix never repeats. R02 sends different evidence per question."
        )
        emit()

        emit("## 3. Losing cases")
        emit()
        small = {}
        try:
            await snapshot(env, window=300)
        except RunError as err:
            small = {"code": err.code}
        row, _ = await ask(env, "r02", None, EN_Q, QUESTIONS[0][2])
        table(
            ("case", "R47 result", "R02 k=3 result"),
            [
                (
                    "corpus overflows a 300-token window",
                    "refused: " + small.get("code", "built"),
                    "%s, cited %s" % (row[0], row[2]),
                ),
                (
                    "out-of-corpus question",
                    "; ".join(r[2] for r in rows if r[0] == "cag" and r[1] == "out-of-corpus"),
                    "; ".join(r[2] for r in rows if r[0] == "r02" and r[1] == "out-of-corpus"),
                ),
            ],
        )
        emit(
            "Out-of-corpus questions abstain in all three arms (cited markers `-`); that comes from the scripted "
            "reply, not from the arms."
        )
        emit()
        extra = corpus_markdown() + "\n\nThe harbour pilots rotate their shifts every 9 hours."
        env2 = await build(extra, Path(tmp) / "b")
        snap2 = await snapshot(env2)
        _, key1 = await ask(env, "cag", snap, EN_Q, QUESTIONS[0][2])
        _, key2 = await ask(env2, "cag", snap2, EN_Q, QUESTIONS[0][2])
        table(
            ("after a corpus update (one paragraph added)", "value"),
            [
                ("chunks before / after", "%d / %d" % (len(snap.items), len(snap2.items))),
                (
                    "evidence chars before / after",
                    "%d / %d" % (sum(len(i.text) for i in snap.items), sum(len(i.text) for i in snap2.items)),
                ),
                ("snapshot identity changed", "yes" if snap.snapshot_id != snap2.snapshot_id else "no"),
                ("prefix identical to the old snapshot", "yes" if key1 == key2 else "no"),
            ],
        )
        emit(
            "A corpus update means a new snapshot: rebuild plus a cold prefill of the whole prefix. "
            "Not measured offline. "
            "Snapshot identity includes chunk IDs, so a new index version changes it even before the text differs."
        )
        emit()
    emit("## Live run (orchestrator)")
    emit()
    emit("<orchestrator>")
    emit("Live run result goes here (P7.4-11). Live output is not deterministic and is recorded separately.")
    emit("</orchestrator>")
    emit()
    print("\n".join(OUT))


if __name__ == "__main__":
    asyncio.run(main())
