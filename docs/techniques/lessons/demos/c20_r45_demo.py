"""P7.5 demo: C20 early vs late chunking and R45 single-vector vs MaxSim, with the test-only fake encoder.

Simulation only: the fake encoder gives one token per word with a vector derived from the word's code points. It
has no language understanding, so the ranks below show the mechanics (what is built, scored and ranked), never a
model result. Output has no timings, IDs or paths and is byte-identical between runs.
"""

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
for rel in ("", "tests/indexing"):
    sys.path.insert(0, str(ROOT / rel))

from _late_fixture import ARMS, DOCS, QUESTIONS, format_cell, run_doc  # noqa: E402

from tests.providers._fake_token_encoder import FakeTokenEncoder  # noqa: E402

WINDOW = 64
OVERLAP = 8


class ContextFakeEncoder(FakeTokenEncoder):
    """Fake whose token vector also carries half the running mean of the earlier words of the same input.

    This imitates context flowing from earlier text into later tokens (the thing late chunking relies on), so early
    and late vectors differ. It is a toy rule, not a model.
    """

    async def encode_tokens(self, texts, *, head, input_type):
        result = await super().encode_tokens(texts, head=head, input_type=input_type)
        items = []
        for item in result.items:
            vectors, total = [], [0.0] * self.dim
            for i, vec in enumerate(item.vectors):
                ctx = [t / i for t in total] if i else [0.0] * self.dim
                vectors.append(tuple(x + 0.5 * c for x, c in zip(vec, ctx, strict=True)))
                total = [t + x for t, x in zip(total, vec, strict=True)]
            items.append(item.model_copy(update={"vectors": tuple(vectors)}))
        return result.model_copy(update={"items": tuple(items)})


async def main() -> None:
    print("SIMULATION: fake token encoder, not a model result")
    print(
        "Ties between equal scores are broken by chunk start. Fixture: 3 documents (EN, AR, mixed); "
        "ranks are among the chunks of the question's own document. "
        "Late chunking = encode the whole document once, then mean-pool the token vectors inside each chunk span. "
        "Early = encode each chunk alone (same encoder, same pooling)."
    )
    runs = []
    for doc in DOCS:
        runs.append(await run_doc(ContextFakeEncoder(max_tokens=WINDOW), doc, window_tokens=WINDOW, overlap=OVERLAP))
    print()
    print("## Index sizes")
    for run in runs:
        stats = run.multivector.stats()
        print(
            f"{run.doc.name}: chunks={stats.chunks} tokens={stats.tokens} dim={stats.dim} "
            f"estimated_bytes={stats.estimated_bytes} late_windows={run.late.windows_used} "
            f"early_truncated={run.early.truncated}"
        )
    print()
    print("## Relevant-chunk rank and score per question")
    print("| question | arm | result |")
    print("|---|---|---|")
    by_doc = {run.doc.name: run for run in runs}
    for question in QUESTIONS:
        run = by_doc[question.doc]
        row = next(r for r in run.rows if r["question"].id == question.id)
        for arm in ARMS:
            print(f"| {question.id} | {arm} | {format_cell(question, run.doc, row['rankings'][arm])} |")
    print()
    print("UNANS has no relevant chunk: only the top-1 score is shown. Scores of different arms are not comparable.")
    print(
        "Losing case: EN2 has a keyword-heavy distractor chunk (chunk 3); an arm loses on EN2 when top1_chunk is not 2."
    )


if __name__ == "__main__":
    asyncio.run(main())
