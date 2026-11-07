from rag_foundry_chunking._units import LabRegexTokenizer, Tokenizer
from rag_foundry_chunking.fixed import (
    FixedCharChunker,
    FixedCharConfig,
    FixedTokenChunker,
    FixedTokenConfig,
    FixedWordChunker,
    FixedWordConfig,
)
from rag_foundry_chunking.recursive import RecursiveChunker, RecursiveConfig


def register_builtin_plugins(registry) -> None:
    registry.register(RecursiveChunker)
    registry.register(FixedCharChunker)
    registry.register(FixedWordChunker)
    registry.register(FixedTokenChunker)


__all__ = [
    "RecursiveChunker",
    "RecursiveConfig",
    "register_builtin_plugins",
    "FixedCharChunker",
    "FixedCharConfig",
    "FixedWordChunker",
    "FixedWordConfig",
    "FixedTokenChunker",
    "FixedTokenConfig",
    "Tokenizer",
    "LabRegexTokenizer",
]
