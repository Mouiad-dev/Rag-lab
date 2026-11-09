import asyncio
import os
from datetime import UTC, datetime

import pytest
from rag_foundry_core.settings import load_settings
from rag_foundry_indexing import SelfQuery
from rag_foundry_indexing.metadata_filter import ViewMetadata
from rag_foundry_providers.adapters.ollama import OllamaChatModel, ollama_chat_profile

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(os.environ.get("RAG_LIVE_OLLAMA") != "1", reason="opt-in Ollama test"),
]


def test_self_query_live():
    async def check():
        model = OllamaChatModel(ollama_chat_profile("gemma4:e4b"), base_url=load_settings().ollama_base_url)
        choices = (
            ViewMetadata(
                view_id="10000000-0000-4000-8000-000000000001",
                document_id="20000000-0000-4000-8000-000000000001",
                document_version_id="30000000-0000-4000-8000-000000000001",
                display_name="sample",
                media_type="text/plain",
                version_created_at=datetime.now(UTC),
            ),
        )
        result = await SelfQuery().parse("What is in the document?", choices=choices, chat=model)
        assert result.status in {"applied", "no_filter", "rejected"}

    asyncio.run(check())
