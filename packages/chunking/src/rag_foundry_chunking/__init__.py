from rag_foundry_chunking._units import LabRegexTokenizer, Tokenizer
from rag_foundry_chunking.fixed import (
    FixedCharChunker,
    FixedCharConfig,
    FixedTokenChunker,
    FixedTokenConfig,
    FixedWordChunker,
    FixedWordConfig,
)
from rag_foundry_chunking.multilingual import MultilingualChunker, MultilingualConfig, normalize_with_map
from rag_foundry_chunking.paragraph import ParagraphChunker, ParagraphConfig
from rag_foundry_chunking.recursive import RecursiveChunker, RecursiveConfig
from rag_foundry_chunking.sentence import SentenceChunker, SentenceConfig
from rag_foundry_chunking.sliding import SlidingWindowChunker, SlidingWindowConfig, WindowStats, window_stats


def register_builtin_plugins(registry) -> None:
    registry.register(RecursiveChunker)
    registry.register(FixedCharChunker)
    registry.register(FixedWordChunker)
    registry.register(FixedTokenChunker)
    registry.register(SlidingWindowChunker)
    registry.register(SentenceChunker)
    registry.register(ParagraphChunker)
    registry.register(MultilingualChunker)


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
    "SlidingWindowChunker",
    "SlidingWindowConfig",
    "WindowStats",
    "window_stats",
    "SentenceChunker",
    "SentenceConfig",
    "ParagraphChunker",
    "ParagraphConfig",
    "MultilingualChunker",
    "MultilingualConfig",
    "normalize_with_map",
]
