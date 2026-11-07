from rag_foundry_chunking.recursive import RecursiveChunker, RecursiveConfig
from rag_foundry_chunking.sentence import SentenceChunker, SentenceConfig


def register_builtin_plugins(registry) -> None:
    registry.register(RecursiveChunker)
    registry.register(SentenceChunker)


__all__ = ["RecursiveChunker", "RecursiveConfig", "SentenceChunker", "SentenceConfig", "register_builtin_plugins"]
