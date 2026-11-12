"""Cross-language fixture dataset over the same paired-fixture document as the seed dataset."""

import hashlib

from rag_foundry_evaluation import EvalCase, EvalDataset, EvalDocument, RelevantSpan

from tests.evaluation._corpus import DOC_KEY
from tests.indexing._fixture import FACT_AR, FACT_EN
from tests.indexing._paired_fixture import DOC, ID1, MIXED

QUESTION_AR_AR = (
    "\u0623\u064a\u0646\u0020\u062a\u062d\u0641\u0638\u0020"
    "\u0645\u0643\u062a\u0628\u0629\u0020"
    "\u0627\u0644\u0648\u0627\u062d\u0629\u0020"
    "\u0645\u062e\u0637\u0648\u0637\u0627\u062a\u0647\u0627\u061f"
)
QUESTION_AR_EN = (
    "\u0643\u0644\u0020\u0643\u0645\u0020"
    "\u062f\u0642\u064a\u0642\u0629\u0020\u064a\u0633\u062c\u0644\u0020"
    "\u0627\u0644\u0645\u0631\u0635\u062f\u0020"
    "\u0645\u0633\u062a\u0648\u0649\u0020\u0633\u0637\u062d\u0020"
    "\u0627\u0644\u0628\u062d\u0631\u061f"
)
QUESTION_MIXED_AR = "How many " + "\u0645\u062e\u0637\u0648\u0637\u0629" + " does the Oasis Library keep?"
QUESTION_AR_MIXED = (
    "\u0645\u062a\u0649\u0020\u062a\u0641\u062a\u062d\u0020"
    "\u0645\u0643\u062a\u0628\u0629\u0020"
    "\u0627\u0644\u0648\u0627\u062d\u0629\u0020"
    "\u0642\u0627\u0639\u0629\u0020"
    "\u0627\u0644\u0642\u0631\u0627\u0621\u0629\u061f"
)
QUESTION_AR_NEG = (
    "\u0623\u064a\u0020\u0645\u0643\u062a\u0628\u0629\u0020"
    "\u0644\u0627\u0020\u062a\u062d\u0641\u0638\u0020"
    "\u0645\u062e\u0637\u0648\u0637\u0627\u062a\u061f"
)

# (case_id, language, tags, question, relevant paragraph or None)
ROWS = (
    ("xl-en-en", "en", ("name",), "How often are sea levels recorded at Port Sallow?", FACT_EN),
    ("xl-ar-ar", "ar", ("name",), QUESTION_AR_AR, FACT_AR),
    ("xl-en-ar", "en", ("name", "number"), "How many manuscripts does the Oasis Library keep?", FACT_AR),
    ("xl-ar-en", "ar", ("number",), QUESTION_AR_EN, FACT_EN),
    ("xl-en-ar-id", "en", ("identifier",), "Which warehouse holds part AB-1234?", ID1),
    ("xl-mixed-ar", "mixed", ("number",), QUESTION_MIXED_AR, FACT_AR),
    ("xl-ar-mixed", "ar", (), QUESTION_AR_MIXED, MIXED),
    ("xl-en-neg", "en", ("negation",), "Which reading room does not open at 08:30?", None),
    ("xl-ar-neg", "ar", ("negation",), QUESTION_AR_NEG, None),
)


def cross_language_dataset() -> EvalDataset:
    cases = []
    for case_id, language, tags, question, paragraph in ROWS:
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
                split="dev",
                answerable=paragraph is not None,
                relevant=relevant,
            )
        )
    doc = EvalDocument(doc_key=DOC_KEY, text_sha256=hashlib.sha256(DOC.encode("utf-8")).hexdigest(), length=len(DOC))
    return EvalDataset(
        format="rag-foundry.eval-dataset",
        format_version="1",
        dataset_id="cross-language-fixture",
        dataset_version="1",
        description="Tiny offline cross-language fixture, not a benchmark",
        documents=(doc,),
        cases=tuple(cases),
    )
