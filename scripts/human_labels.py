"""Helper for human labelling of the answer-support judge items (P6.5).

Usage (from the repository root):
    uv run python scripts/human_labels.py items       # print the 6 items readably (no judge output)
    uv run python scripts/human_labels.py check       # strictly load the label file; print status and count
    uv run python scripts/human_labels.py agreement   # run the judge on local Ollama, then compare with the labels

`agreement` makes 6 local chat calls (default model gemma4:e4b, cost 0; needs Ollama running).
Judge labels are evaluator output, not ground truth.
"""

import argparse
import asyncio
import sys
from pathlib import Path

from rag_foundry_evaluation.human_labels import agreement, load_human_labels, render_agreement
from rag_foundry_evaluation.judge import RUBRIC_ANSWER_SUPPORT_V1, LlmJudge, load_judge_items

ROOT = Path(__file__).resolve().parents[1]
ITEMS = ROOT / "tests/evaluation/data/judge_items_v1.json"
LABELS = ROOT / "tests/evaluation/data/human_labels_answer_support_v1.json"


def _items():
    return load_judge_items(ITEMS)


def _labels(items):
    return load_human_labels(LABELS, items=items, rubric=RUBRIC_ANSWER_SUPPORT_V1)


def cmd_items(_args) -> int:
    rubric = RUBRIC_ANSWER_SUPPORT_V1
    print(f"Rubric {rubric.rubric_id} v{rubric.rubric_version}; allowed labels: {', '.join(rubric.labels)}\n")
    for item in _items().items:
        print(f"== {item.item_id}")
        print(f"Question: {item.question}")
        print(f"Answer:   {item.answer}")
        for evidence in item.evidence:
            print(f"[{evidence.marker}] {evidence.text}")
        print()
    return 0


def cmd_check(_args) -> int:
    labels = _labels(_items())
    annotators = sorted({label.annotator_id for label in labels.labels})
    print(f"status={labels.status} labels={len(labels.labels)} annotators={','.join(annotators) or '-'}")
    return 0


def cmd_agreement(args) -> int:
    from rag_foundry_core.settings import load_settings
    from rag_foundry_providers.adapters.ollama import OllamaChatModel, ollama_chat_profile

    items = _items()
    labels = _labels(items)
    chat = OllamaChatModel(
        ollama_chat_profile(model=args.model), api_base=load_settings().ollama_base_url, timeout_s=300
    )
    verdicts = asyncio.run(LlmJudge(RUBRIC_ANSWER_SUPPORT_V1).judge_all(items.items, chat=chat))
    print(render_agreement(agreement(verdicts, labels.labels, rubric=RUBRIC_ANSWER_SUPPORT_V1)))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("items").set_defaults(func=cmd_items)
    sub.add_parser("check").set_defaults(func=cmd_check)
    agree = sub.add_parser("agreement")
    agree.add_argument("--model", default="gemma4:e4b")
    agree.set_defaults(func=cmd_agreement)
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
