"""P7.4 lesson demo: R50 exact and R51 semantic response caches, with calibration and usage per stage.

Offline. Answers are scripted chat replies (not a model); the embedder is the test-only lexical vectorizer unless a
row says "fixed vectors"; the store is in memory. Question text, run IDs, keys and timings are never printed, so the
output is byte-identical between runs.
"""

import asyncio
import sys
import tempfile
from datetime import timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
for rel in ("tests/answering", "tests/cache", "tests/indexing", "tests/ingestion"):
    sys.path.insert(0, str(ROOT / rel))

from _fixture import FACT_AR, FACT_EN, QUERY_EN, TEST_CHUNK_SIZE, TEST_OVERLAP  # noqa: E402
from _lexical_embedder import LexicalTestEmbedder  # noqa: E402
from _qa import corpus_markdown  # noqa: E402
from _scripted_chat import cite_where  # noqa: E402
from rag_foundry_cache import calibrate  # noqa: E402
from rag_foundry_cache.features import eligible, features  # noqa: E402
from rag_foundry_chunking import RecursiveChunker, RecursiveConfig  # noqa: E402
from rag_foundry_domain.index_testing import InMemoryIndexRepository  # noqa: E402
from rag_foundry_domain.response_cache import cosine  # noqa: E402
from rag_foundry_domain.testing import make_scope  # noqa: E402
from rag_foundry_domain.uow_testing import InMemoryUnitOfWorkFactory  # noqa: E402
from rag_foundry_indexing import IndexService  # noqa: E402
from rag_foundry_ingestion.anchor_service import AnchorService  # noqa: E402
from rag_foundry_providers.ledger import RecordingEmbeddingModel, UsageLedger  # noqa: E402
from test_cag_workflow import AR_Q  # noqa: E402  (Arabic question text shared with the tests)
from test_exact_cache import T0, CollidingStore, Env  # noqa: E402
from test_extraction_service import ingest, setup  # noqa: E402
from test_semantic_cache import GRID, PARAPHRASE, StubEmbedder, load_pairs, sem, sgo  # noqa: E402

OUT = []
NOT = chr(0x644) + chr(0x627)  # the Arabic word for "not"
WORDS = AR_Q.split(" ")
AR_NEG = " ".join(WORDS[:2] + [NOT] + WORDS[2:])


def emit(line=""):
    OUT.append(line)


def table(header, rows):
    emit("| " + " | ".join(header) + " |")
    emit("|" + "---|" * len(header))
    for row in rows:
        emit("| " + " | ".join(str(cell) for cell in row) + " |")
    emit()


def frac(hits, total):
    return f"{hits}/{total}"


class Deny:
    checks_revocation = True

    async def denied(self, scope, version_ids):
        return frozenset(version_ids)


async def build(folder):
    scope = make_scope()
    extraction, intake, _spy, store, _docs, views, _ = setup(folder, scope)
    doc_id, ver_id = await ingest(intake, scope, corpus_markdown().encode("utf-8"), name="n.md", media="text/markdown")
    view = (await extraction.extract(scope, doc_id, ver_id)).view
    index = InMemoryIndexRepository(views)
    anchors = AnchorService(store, InMemoryUnitOfWorkFactory(extractions=views), provider_namespace="test")
    embedder = LexicalTestEmbedder()
    chunker = RecursiveChunker(RecursiveConfig(chunk_size=TEST_CHUNK_SIZE, chunk_overlap=TEST_OVERLAP))
    uow = InMemoryUnitOfWorkFactory(extractions=views, indexes=index)
    version = await IndexService(store, uow, anchors).build(scope, (view.id,), chunker, embedder)
    return scope, index, anchors, embedder, version


def made(env, before):
    return sum(ledger.totals().calls for ledger in env.ledgers[before:])


async def r50(built):
    rows, other, env0 = [], make_scope(), Env(built)

    async def case(label, seeded=True, ttl=86400, later=0, k=None, **kwargs):
        env = Env(built)
        if seeded:
            await env.go(cache=env.cache(ttl=ttl))
        env.clock.now = T0 + timedelta(seconds=later)
        if ttl != 86400:
            kwargs["cache"] = env.cache(ttl=ttl)
        if k:
            kwargs["fn"] = env.answer_fn(k=k)
        before = len(env.ledgers)
        t = (await env.go(**kwargs)).trace
        rows.append((label, t.outcome, t.reason or "-", t.store_reason or "-", made(env, before), t.calls_avoided))

    await case("first run", seeded=False)
    await case("repeat of the same request")
    await case("changed model", inputs=env0.inputs(chat_model="other-model"))
    await case("history turn added", inputs=env0.inputs(history=(("user", "earlier turn"),)))
    await case("authz revision changed", inputs=env0.inputs(authz_revision="rev-2"))
    await case("index version in the key", inputs=env0.inputs(index_version_id="other"), expected="other")
    await case("index rebuilt since the entry was stored", expected="other")
    await case("other tenant", scope=other)
    env = Env(built)
    await env.go()
    foreign = (await env.store.get(env.scope, env.key())).model_copy(update={"key": env.key(other)})
    before = len(env.ledgers)
    t = (await env.go(scope=other, cache=env.cache(store=CollidingStore(foreign)))).trace
    rows.append(
        ("forced collision (store returns a foreign entry)", t.outcome, t.reason, t.store_reason, made(env, before), 0)
    )
    await case("source denied by the gate", gate=Deny())
    await case("entry expired (ttl 60 s, test clock)", ttl=60, later=60)
    await case("pins mismatch (answer made with k=3, key says k=5)", seeded=False, k=3)
    await case("bypass: freshness current", seeded=False, inputs=env0.inputs(freshness="current"))
    await case("bypass: enriched mode", seeded=False, inputs=env0.inputs(mode="enriched"))
    return rows


async def r51(built):
    rows, ledger = [], UsageLedger()
    lexical = RecordingEmbeddingModel(LexicalTestEmbedder(), ledger)
    q1, q2, q3 = QUERY_EN + " one", QUERY_EN + " two", QUERY_EN + " three"
    stub = StubEmbedder(table={q1: (1.0, 0.0, 0.0), q2: (0.0, 1.0, 0.0), q3: (1.0, 1.0, 0.0)})
    base = QUERY_EN[:-1]
    cases = (
        ("paraphrase", QUERY_EN, PARAPHRASE, 0.8, lexical, None, {}),
        ("2025 vs 2026", base + " in 2025?", base + " in 2026?", 0.5, lexical, None, {}),
        ("with vs without", base + " with the gauge?", base + " without the gauge?", 0.5, lexical, None, {}),
        ("AR negation", AR_Q, AR_NEG, 0.5, lexical, FACT_AR, {}),
        ("identifier", "Ticket TCK-4410 at Port Sallow?", "Ticket TCK-4401 at Port Sallow?", 0.5, lexical, None, {}),
        ("other tenant", QUERY_EN, PARAPHRASE, 0.8, lexical, None, {"scope": make_scope()}),
    )
    for label, seed, ask, threshold, embedder, fact, extra in cases:
        env = Env(built, responder=cite_where(fact, fact) if fact else None)
        cache = sem(env, threshold=threshold)
        await sgo(env, seed, embedder=embedder, cache=cache)
        before = len(env.ledgers)
        t = (await sgo(env, ask, embedder=embedder, cache=cache, **extra)).trace
        va, vb = (await embedder.embed([seed, ask])).vectors
        rows.append(
            (label, threshold, t.outcome, t.reason or "-", f"{cosine(va, vb):.2f}", made(env, before), t.calls_avoided)
        )
    env = Env(built)
    cache = sem(env, threshold=0.6)
    for question, text in ((q1, "Answer one"), (q2, "Answer two")):
        env.responder = cite_where(FACT_EN, text)
        await sgo(env, question, embedder=stub, cache=cache)
    before = len(env.ledgers)
    t = (await sgo(env, q3, embedder=stub, cache=cache)).trace
    v1, v2, v3 = (await stub.embed([q1, q2, q3])).vectors
    s1, s2 = cosine(v3, v1), cosine(v3, v2)
    rows.append(
        (
            f"ambiguous (fixed vectors, {s1:.2f} and {s2:.2f})",
            0.6,
            t.outcome,
            t.reason,
            f"{max(s1, s2):.2f}",
            made(env, before),
            t.calls_avoided,
        )
    )
    return rows


async def main():
    sys.stdout.reconfigure(encoding="utf-8", newline="\n")
    emit("# P7.4 R50 and R51 response caches: demo output")
    emit()
    emit(
        "Command: `uv run python docs/techniques/lessons/demos/p74_response_cache_demo.py`. Date: 2026-11-23. "
        "Offline; scripted answers; test-only lexical embedder; small illustrative pair sets; not a benchmark. "
        "Calls are ledger calls (one fresh answer = 1 query embedding + 1 chat). Cold versus warm is shown as calls "
        "avoided, never latency."
    )
    emit()
    with tempfile.TemporaryDirectory() as tmp:
        built = await build(Path(tmp))
        emit("## 1. R50 exact cache")
        emit()
        emit(
            "The R50 key covers scope, question, history, authz revision, model, prompt, index version, retriever "
            "and generation settings. Every hit is revalidated (scope, expiry, index version, anchors, source gate). "
            "Bypass requests never touch the store."
        )
        emit()
        table(("request", "outcome", "reason", "not stored because", "calls made", "calls avoided"), await r50(built))
        emit("A miss always runs a fresh answer; a foreign entry is never returned, deleted or overwritten.")
        emit()
        emit(
            "The `changed model` and `index version in the key` rows show `pins_mismatch` because this demo's "
            "scripted answer is still produced with the original model and index while the key names another one. "
            "The cache refuses to store an answer whose pins differ from its key (see `pins mismatch`); with a "
            "matching answer these requests would simply be stored under their own key."
        )
        emit()

        emit("## 2. R51 semantic cache")
        emit()
        emit(
            "A hit needs the same R50 partition, similarity at or above the threshold, a margin over the runner-up, "
            "equal guarded features and the R50 revalidation. Guard rows run at threshold 0.50 so the guard, not "
            "similarity, decides. Similarity is the cosine of the two question vectors computed here (the package "
            "trace reports it only on hits). The ambiguous row uses fixed vectors and two cached answers that differ."
        )
        emit()
        head = ("request", "threshold", "outcome", "reason", "similarity", "calls made", "calls avoided")
        table(head, await r51(built))

        emit("## 3. Calibration on `semantic_pairs_v1.json`")
        emit()
        pairs, vectors, embedder = load_pairs()["pairs"], {}, LexicalTestEmbedder()
        for p in pairs:
            vectors[p["id"]] = tuple((await embedder.embed([p["a"], p["b"]])).vectors)
        report = calibrate(pairs, vectors, GRID)
        emit(
            f"{len(pairs)} hand-written pairs (dev and held-out split), lexical test embedder. The threshold is "
            "chosen from dev only; the package reports held-out counts only at the chosen threshold. "
            "Format: hits/total."
        )
        emit()
        rows = []
        for r in report.rows:
            if r.threshold in (0.5, 0.8, 0.9, 0.92, 0.93, 0.95, 0.97, 1.0):
                d, n = r.dev, r.dev.paraphrase_total
                rows.append(
                    (r.threshold, frac(d.paraphrase_hits_unguarded, n), frac(d.paraphrase_hits_guarded, n))
                    + (
                        frac(d.false_hits_unguarded, d.adversarial_total),
                        frac(d.false_hits_guarded, d.adversarial_total),
                    )
                )
        table(
            (
                "threshold",
                "dev paraphrase unguarded",
                "dev paraphrase guarded",
                "dev false hits unguarded",
                "dev false hits guarded",
            ),
            rows,
        )
        if report.chosen_threshold is None:
            emit(f"Chosen threshold: none. Reason: `{report.reason}`.")
        else:
            emit(f"Chosen threshold: `{report.chosen_threshold}`.")
        emit()
        h = report.held_out
        table(
            ("held-out at the chosen threshold", "unguarded", "guarded"),
            [
                ("paraphrase hits", frac(h.paraphrase_hits_unguarded, h.paraphrase_total), frac(h.paraphrase_hits_guarded, h.paraphrase_total)),  # noqa: E501
                ("false hits on adversarial pairs", frac(h.false_hits_unguarded, h.adversarial_total), frac(h.false_hits_guarded, h.adversarial_total)),  # noqa: E501
            ],
        )  # fmt: skip
        rows = []
        for p in (p for p in pairs if p["kind"] == "adversarial"):
            sim = cosine(*vectors[p["id"]])
            same, names = eligible(features(p["a"]), features(p["b"]))
            close = sim >= report.chosen_threshold
            verdict = "stopped by similarity" if not close else ("stopped by guard" if not same else "FALSE HIT")
            rows.append((p["id"], p["split"], p["note"], f"{sim:.2f}", ", ".join(names) or "-", verdict))
        table(("pair", "split", "kind", "similarity", "guard differs on", "at the chosen threshold"), rows)
        emit(
            "Losing cases: a month name (held-10) and a unit (held-11) pass both similarity and the guard, so they are "
            "false hits. Entity swaps without digits (dev-11, held-12) also pass the guard (it sees no difference) and "
            "are stopped only because their similarity sits just under the threshold "
            "(held-12 rounds up to the threshold in the table but is below it); with another embedder or "
            "threshold they would be false hits too. The guard sees numbers, identifiers, quoted text, negation, "
            "comparatives and script only. Not a benchmark: a few pairs, one embedder."
        )
        emit()

        emit("## 4. Usage per stage")
        emit()
        emit("Answer ledger = chat and query-embedding calls of the answered run; cache-embed ledger = the R51 lookup.")
        emit()
        env = Env(built)
        cold, warm = await env.go(), await env.go()
        first = env.ledgers[0].totals()
        rows = [("R50 cold", first.calls, first.input_tokens, 0, cold.trace.calls_avoided)]
        rest = [ledger.totals() for ledger in env.ledgers[1:]]
        warm_calls, warm_tokens = sum(x.calls for x in rest), sum(x.input_tokens for x in rest)
        rows.append(("R50 warm", warm_calls, warm_tokens, 0, warm.trace.calls_avoided))
        env, ledger = Env(built), UsageLedger()
        rec = RecordingEmbeddingModel(LexicalTestEmbedder(), ledger)
        cold = await sgo(env, QUERY_EN, embedder=rec)
        cold_embed, first = ledger.totals().calls, env.ledgers[0].totals()
        warm = await sgo(env, PARAPHRASE, embedder=rec)
        rows.append(("R51 cold", first.calls, first.input_tokens, cold_embed, cold.trace.calls_avoided))
        rest = [x.totals() for x in env.ledgers[1:]]
        warm_calls, warm_tokens = sum(x.calls for x in rest), sum(x.input_tokens for x in rest)
        rows.append(("R51 warm", warm_calls, warm_tokens, ledger.totals().calls - cold_embed, warm.trace.calls_avoided))
        table(
            ("run", "answer ledger calls", "answer input tokens (scripted)", "cache-embed calls", "calls avoided"), rows
        )
        emit(
            "A warm R51 hit still pays one cache-embedding call; it avoids the retrieval embedding and the chat call. "
            "R50 needs no embedding at all. Token figures are the scripted model's estimates."
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
