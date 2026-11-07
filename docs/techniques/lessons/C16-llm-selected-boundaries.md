# C16: LLM-selected boundaries

## Purpose

C16 asks a chat model to identify numbered sentence units where a new topic begins. The model selects boundaries only; chunk text and offsets always come from the original document.

## Worked example

The scripted model receives delimited units and returns JSON such as `{"breaks":[{"unit":2,"starts_with":"The market shifts rapidly"}]}`. Unit 2 is accepted only when its quote matches the beginning of that source unit. The resulting spans are packed from source offsets around that boundary.

## Demo

```python
import asyncio, hashlib, json
from rag_foundry_chunking import LlmBoundaryChunker, LlmBoundaryConfig
from rag_foundry_chunking.model_based import InMemorySegmentationCache, ModelChunkContext
from rag_foundry_domain.prompts import PromptTemplate
from rag_foundry_providers.fake import fake_chat_profile
from rag_foundry_providers.messages import ChatResult
from rag_foundry_providers.usage import Usage
from rag_foundry_techniques import ChunkInput

TEXT = "The cat rests quietly. The cat sleeps peacefully. The market shifts rapidly. The market rises steadily."
class ScriptedChat:
    profile = fake_chat_profile()
    async def complete(self, request):
        reply = {"breaks": [{"unit": 2, "starts_with": "invented quote absent"}]}
        usage = Usage(profile_id=self.profile.id, provider=self.profile.provider,
            model=self.profile.model, operation="chat", input_tokens=1, output_tokens=1,
            token_source="estimated", latency_ms=0)
        return ChatResult(text=json.dumps(reply), finish_reason="stop", usage=usage)

prompt_text = "Select breaks and starts_with quotes."
prompt = PromptTemplate(id="chunk-boundaries-v1", family="chunk-boundaries", version=1,
    purpose="chunk-boundaries", system_text=prompt_text,
    content_sha256=hashlib.sha256(prompt_text.encode()).hexdigest())
result = asyncio.run(LlmBoundaryChunker(LlmBoundaryConfig()).chunk_with_models(
    ChunkInput(text=TEXT), ModelChunkContext(chat=ScriptedChat(), prompt=prompt,
    cache=InMemorySegmentationCache())))
print("spans:", [(s.start, s.end, TEXT[s.start:s.end]) for s in result.spans])
print("rejections:", [r.model_dump() for r in result.report.rejections])
```

Output from running the demo:

```text
spans: [(0, 103, 'The cat rests quietly. The cat sleeps peacefully. The market shifts rapidly. The market rises steadily.')]
rejections: [{'call_index': 0, 'reason': 'quote_mismatch', 'detail': "{'unit': 2, 'starts_with': 'invented quote absent'}"}]
```

## Failure case

For `{"breaks":[{"unit":4,"starts_with":"invented words absent"}]}`, the report includes a `quote_mismatch` rejection, no boundary is accepted, and the output span still slices the source document.

## Limitations

- Each window costs a model call, within the configured maximum call count.
- Repeated samples can choose different boundaries; compare offsets with `boundary_agreement`.
- Some models ignore JSON-only instructions and produce rejected output.
- gemma4 on Ollama has `structured_output=False` in this repository, so C16 parses JSON from response text.
