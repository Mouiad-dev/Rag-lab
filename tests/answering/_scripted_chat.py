"""Scripted ChatModel double with provider-shaped usage."""

import re

from rag_foundry_providers.fake import FakeChatModel, fake_chat_profile
from rag_foundry_providers.messages import ChatResult


class ScriptedChatModel:
    def __init__(self, responder, profile=None):
        self.profile = profile or fake_chat_profile()
        self.responder = responder
        self.requests = []

    async def complete(self, request):
        self.requests.append(request)
        text, finish_reason = self.responder(request)
        usage = (await FakeChatModel(self.profile).complete(request)).usage
        return ChatResult(text=text, finish_reason=finish_reason, usage=usage)


def cite_where(substring, text):
    def responder(request):
        user = next(message.content for message in request.messages if message.role == "user")
        pattern = re.compile(r"<<<EVIDENCE (S\d+) ([0-9a-f]{16})>>>\n(.*?)\n<<<END \1 \2>>>", re.DOTALL)
        for match in pattern.finditer(user):
            if substring in match.group(3):
                return f"{text} [{match.group(1)}]", "stop"
        return "INSUFFICIENT_EVIDENCE", "stop"

    return responder
