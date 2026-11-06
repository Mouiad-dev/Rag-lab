"""Golden files are frozen (ADR-0013 D14, ADR-0014 D15). Change the builder call, never the JSON."""

import json
from pathlib import Path

from rag_foundry_answering.prompt import build_grounded_request, build_ungrounded_request
from rag_foundry_answering.service import AnswerConfig

QUESTION = "Where is the observatory?"
EVIDENCE = (
    ("S1", "The observatory stands on Cedar Hill."),
    (
        "S2",
        "\u0627\u0644\u0645\u0631\u0635\u062f \u0641\u064a \u0627\u0644\u062a\u0644.",
    ),
)
NONCE = "0123456789abcdef"
GOLDEN_DIR = Path(__file__).parent / "golden"


def test_answer_defaults():
    config = AnswerConfig()
    assert config.k == 5
    assert config.max_output_tokens == 512
    assert config.temperature == 0.0
    assert config.max_evidence_chars == 12000
    assert config.min_score is None


def test_grounded_request_matches_golden():
    config = AnswerConfig()
    request = build_grounded_request(
        QUESTION,
        EVIDENCE,
        nonce=NONCE,
        max_output_tokens=config.max_output_tokens,
        temperature=config.temperature,
    )
    rendered = json.dumps(request.model_dump(mode="json"), ensure_ascii=True, sort_keys=True, indent=2) + "\n"
    assert rendered.encode("utf-8") == (GOLDEN_DIR / "request_v2.json").read_bytes()


def test_ungrounded_request_matches_golden():
    config = AnswerConfig()
    request = build_ungrounded_request(
        QUESTION,
        nonce=NONCE,
        max_output_tokens=config.max_output_tokens,
        temperature=config.temperature,
    )
    rendered = json.dumps(request.model_dump(mode="json"), ensure_ascii=True, sort_keys=True, indent=2) + "\n"
    assert rendered.encode("utf-8") == (GOLDEN_DIR / "request_ungrounded_v1.json").read_bytes()
