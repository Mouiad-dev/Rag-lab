from rag_foundry_chunking._units import LabRegexTokenizer, Tokenizer
from rag_foundry_chunking.code import AstCodeChunker, AstCodeConfig, CodeSeparatorChunker, CodeSeparatorConfig
from rag_foundry_chunking.fixed import (
    FixedCharChunker,
    FixedCharConfig,
    FixedTokenChunker,
    FixedTokenConfig,
    FixedWordChunker,
    FixedWordConfig,
)
from rag_foundry_chunking.json_structure import JsonStructureChunker, JsonStructureConfig, json_path_at
from rag_foundry_chunking.markdown import MarkdownSectionChunker, MarkdownSectionConfig
from rag_foundry_chunking.multilingual import MultilingualChunker, MultilingualConfig, normalize_with_map
from rag_foundry_chunking.paragraph import ParagraphChunker, ParagraphConfig
from rag_foundry_chunking.recursive import RecursiveChunker, RecursiveConfig
from rag_foundry_chunking.sentence import SentenceChunker, SentenceConfig
from rag_foundry_chunking.sliding import SlidingWindowChunker, SlidingWindowConfig, WindowStats, window_stats
from rag_foundry_chunking.structure import (
    ContextSpan,
    SkippedRange,
    StructuredChunk,
    StructureParseError,
    StructureResult,
    render,
)
from rag_foundry_chunking.tables import TableCell, TableChunker, TableConfig, table_cells


def register_builtin_plugins(registry) -> None:
    registry.register(RecursiveChunker)
    registry.register(FixedCharChunker)
    registry.register(FixedWordChunker)
    registry.register(FixedTokenChunker)
    registry.register(SlidingWindowChunker)
    registry.register(SentenceChunker)
    registry.register(ParagraphChunker)
    registry.register(MultilingualChunker)
    registry.register(MarkdownSectionChunker)
    registry.register(TableChunker)
    registry.register(JsonStructureChunker)
    registry.register(CodeSeparatorChunker)
    registry.register(AstCodeChunker)


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
    "MarkdownSectionChunker",
    "MarkdownSectionConfig",
    "TableChunker",
    "TableConfig",
    "StructuredChunk",
    "ContextSpan",
    "SkippedRange",
    "StructureResult",
    "render",
    "TableCell",
    "table_cells",
    "JsonStructureChunker",
    "JsonStructureConfig",
    "StructureParseError",
    "json_path_at",
    "CodeSeparatorChunker",
    "CodeSeparatorConfig",
    "AstCodeChunker",
    "AstCodeConfig",
]
