import asyncio
import json

import pytest
from rag_foundry_evaluation import EvaluationError
from rag_foundry_evaluation.judge import (
    RUBRIC_ANSWER_SUPPORT_V1,
    JudgeConfig,
    JudgeItem,
    JudgeItemSet,
    LlmJudge,
    dump_judge_items,
    load_judge_items,
)
from rag_foundry_providers.fake import fake_chat_profile
from rag_foundry_providers.ledger import RecordingChatModel, UsageLedger
from rag_foundry_providers.messages import ChatResult

from tests.chunking._scripted_models import ScriptedChat
from tests.evaluation._stream_chat import StreamScriptedChat

PINNED_ANSWER_SUPPORT_V1_SHA256 = "b6d6c59076e1a02bf275b8e8f3662824c3feb7f0d8e91addf3107917785fbe78"


def make_item(item_id="i1"):
    return JudgeItem(
        item_id=item_id,
        question="How many?",
        answer="There are 15.",
        evidence=[{"marker": "S1", "text": "There are 17."}],
    )


def reply(text, finish="stop"):
    return ScriptedChat(lambda request: (text, finish))


def test_ok():
    chat = reply('{"label":"unsupported","reason":"15 is not 17"}')
    item = make_item()
    verdict = asyncio.run(LlmJudge().judge(item, chat=chat))
    assert verdict.status == "ok"
    assert verdict.label == "unsupported"
    assert len(verdict.usage) == 1
    assert verdict.source == "llm_judge"
    assert verdict.prompt_sha256 == RUBRIC_ANSWER_SUPPORT_V1.prompt_sha256()
    system = chat.requests[0].messages[0].content
    assert "untrusted" in system
    assert '{"label":' in system
    user = json.loads(chat.requests[0].messages[1].content)
    assert user == {
        "question": item.question,
        "answer": item.answer,
        "evidence": [{"marker": "S1", "text": "There are 17."}],
    }


@pytest.mark.parametrize(
    ("text", "finish"),
    [
        ("not json", "stop"),
        ('{"label":"maybe","reason":"x"}', "stop"),
        ('{"label":"supported"}', "stop"),
        ('{"label":"supported","reason":"x","score":1}', "stop"),
        ('{"label":"supported","label":"unsupported","reason":"x"}', "stop"),
        ('{"label":"supported","reason":"x"}', "length"),
        (json.dumps({"label": "supported", "reason": "r" * 501}), "stop"),
    ],
)
def test_invalid_output(text, finish):
    verdict = asyncio.run(LlmJudge().judge(make_item(), chat=reply(text, finish)))
    assert verdict.status == "invalid_output"
    assert verdict.label is None
    assert len(verdict.usage) == 1


def test_fenced():
    chat = reply('```json\n{"label":"supported","reason":"ok"}\n```')
    verdict = asyncio.run(LlmJudge().judge(make_item(), chat=chat))
    assert verdict.status == "ok"
    assert verdict.label == "supported"


def test_structured_preferred():
    usage = asyncio.run(_fake_usage())

    def responder(request):
        return ChatResult(
            text="garbage", structured={"label": "supported", "reason": "ok"}, finish_reason="stop", usage=usage
        )

    verdict = asyncio.run(LlmJudge().judge(make_item(), chat=ScriptedChat(responder)))
    assert verdict.status == "ok"


async def _fake_usage():
    from rag_foundry_providers.fake import FakeChatModel
    from rag_foundry_providers.messages import ChatMessage, ChatRequest

    result = await FakeChatModel(fake_chat_profile()).complete(
        ChatRequest(messages=(ChatMessage(role="user", content="x"),))
    )
    return result.usage


def test_stream():
    chat = StreamScriptedChat(['{"label":', '"supported",', '"reason":"ok"}'])
    verdict = asyncio.run(LlmJudge(config=JudgeConfig(mode="stream")).judge(make_item(), chat=chat))
    assert verdict.status == "ok"
    assert chat.requests[0].response_schema is None
    assert len(verdict.usage) == 1
    no_stream = StreamScriptedChat([], profile=fake_chat_profile(streaming=False))
    with pytest.raises(EvaluationError) as info:
        asyncio.run(LlmJudge(config=JudgeConfig(mode="stream")).judge(make_item(), chat=no_stream))
    assert info.value.code == "judge_stream_unsupported"
    assert no_stream.requests == []


def test_metering():
    ledger = UsageLedger()
    inner = reply('{"label":"supported","reason":"ok"}')
    chat = RecordingChatModel(inner, ledger)
    items = [make_item("a"), make_item("b"), make_item("c")]
    verdicts = asyncio.run(LlmJudge().judge_all(items, chat=chat))
    assert [v.item_id for v in verdicts] == ["a", "b", "c"]
    assert ledger.totals().calls == 3
    with pytest.raises(EvaluationError) as info:
        asyncio.run(LlmJudge().judge_all([make_item("a"), make_item("a")], chat=inner))
    assert info.value.code == "duplicate_item_id"
    assert len(inner.requests) == 3


def test_version_pin():
    assert RUBRIC_ANSWER_SUPPORT_V1.prompt_sha256() == PINNED_ANSWER_SUPPORT_V1_SHA256
    changed = RUBRIC_ANSWER_SUPPORT_V1.model_copy(
        update={"criteria": ("Changed criterion.",) + RUBRIC_ANSWER_SUPPORT_V1.criteria[1:]}
    )
    assert changed.prompt_sha256() != PINNED_ANSWER_SUPPORT_V1_SHA256


def test_item_validation(tmp_path):
    with pytest.raises(ValueError):
        JudgeItem(item_id="i", question="q", answer="a", evidence=[{"marker": "X1", "text": "t"}])
    with pytest.raises(ValueError):
        JudgeItem(
            item_id="i",
            question="q",
            answer="a",
            evidence=[{"marker": "S1", "text": "t"}, {"marker": "S1", "text": "u"}],
        )
    with pytest.raises(ValueError):
        JudgeItem(
            item_id="i",
            question="q",
            answer="a",
            evidence=[{"marker": f"S{n}", "text": "t"} for n in range(11)],
        )
    item_set = JudgeItemSet(format="rag-foundry.judge-items", format_version="1", items_id="set1", items=[make_item()])
    path = tmp_path / "items.json"
    path.write_text(dump_judge_items(item_set), encoding="utf-8")
    assert load_judge_items(path) == item_set
    dup = tmp_path / "dup.json"
    dup.write_text('{"format":"rag-foundry.judge-items","format":"x"}', encoding="utf-8")
    with pytest.raises(EvaluationError) as info:
        load_judge_items(dup)
    assert info.value.code == "invalid_judge_items"
