"""C19 lesson demo: rule router decisions and a comparison with fixed C05. No model calls."""

from rag_foundry_chunking import AdaptiveChunker, AdaptiveConfig, RecursiveChunker, RecursiveConfig, compare_with_fixed
from rag_foundry_techniques import ChunkInput

TEXT = (
    "English prose paragraph. Another sentence.\n\n"
    "- first item\n- second item\n\n"
    "Name | Value | Extra\n--- | --- | ---\na | b | c\n\n"
    "    def run():\n    return 1\n\n"
    "English \u0643\u0644\u0645\u0627\u062a \u0639\u0631\u0628\u064a."
)
config = AdaptiveConfig(chunk_size=64)
spans, decisions = AdaptiveChunker(config).segment(ChunkInput(text=TEXT))
print("kind | policy | start | end | chunks | reason")
for d in decisions:
    print(f"{d.kind} | {d.policy} | {d.start} | {d.end} | {d.chunk_count} | {d.reason}")
print(compare_with_fixed(ChunkInput(text=TEXT), config, "C05").model_dump(exclude={"regions"}))
fixed = RecursiveChunker(RecursiveConfig(chunk_size=64, chunk_overlap=0)).chunk(ChunkInput(text=TEXT))
print("fixed C05 chunks:", [TEXT[s.start : s.end] for s in fixed])
