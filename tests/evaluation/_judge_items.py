"""Judge item set v1: neutral ids, no expected labels; non-ASCII only as escapes."""

from rag_foundry_evaluation.judge import JudgeEvidence, JudgeItem, JudgeItemSet

from tests.indexing._fixture import DISTRACTORS, FACT_AR, FACT_EN, QUERY_AR, QUERY_EN
from tests.indexing._paired_fixture import MIXED

ARABIC_ANSWER = (
    "\u0644\u0627 \u062a\u062d\u0641\u0638 \u0645\u0643\u062a\u0628\u0629 "
    "\u0627\u0644\u0648\u0627\u062d\u0629 \u0623\u064a \u0645\u062e\u0637\u0648\u0637\u0627\u062a"
)


def _item(item_id: str, question: str, answer: str, evidence: str) -> JudgeItem:
    return JudgeItem(
        item_id=item_id,
        question=question,
        answer=answer,
        evidence=(JudgeEvidence(marker="S1", text=evidence),),
    )


def judge_items_v1() -> JudgeItemSet:
    items = (
        # design intent, not a label: answer repeats the evidence figure
        _item("j-01", QUERY_EN, "The observatory records sea levels every 17 minutes [S1].", FACT_EN),
        # design intent, not a label: answer changes one number
        _item("j-02", QUERY_EN, "The observatory records sea levels every 15 minutes [S1].", FACT_EN),
        # design intent, not a label: Arabic answer copies the Arabic evidence
        _item("j-03", QUERY_AR, FACT_AR + " [S1]", FACT_AR),
        # design intent, not a label: Arabic answer negates the evidence
        _item("j-04", QUERY_AR, ARABIC_ANSWER + " [S1].", FACT_AR),
        # design intent, not a label: answer adds a closing time absent from the evidence
        _item(
            "j-05",
            "When does the Oasis Library open its reading room?",
            "It opens at 08:30 and closes at 20:00 [S1].",
            MIXED,
        ),
        # design intent, not a label: evidence is about a different club
        _item(
            "j-06",
            "Which club meets on Tuesday evenings?",
            "The choir meets on Tuesday evenings [S1].",
            DISTRACTORS[2],
        ),
    )
    return JudgeItemSet(
        format="rag-foundry.judge-items",
        format_version="1",
        items_id="answer-support-items",
        items=items,
    )
