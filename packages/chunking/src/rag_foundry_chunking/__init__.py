from rag_foundry_chunking._units import LabRegexTokenizer, Tokenizer
from rag_foundry_chunking.adaptive import (
    AdaptiveChunker,
    AdaptiveConfig,
    PolicyComparison,
    RegionDecision,
    compare_with_fixed,
)
from rag_foundry_chunking.code import AstCodeChunker, AstCodeConfig, CodeSeparatorChunker, CodeSeparatorConfig
from rag_foundry_chunking.fixed import (
    FixedCharChunker,
    FixedCharConfig,
    FixedTokenChunker,
    FixedTokenConfig,
    FixedWordChunker,
    FixedWordConfig,
)
from rag_foundry_chunking.hierarchy import (
    ChildSpan,
    Hierarchy,
    ParentChildChunker,
    ParentChildConfig,
    ParentSpan,
    parent_id_for,
)
from rag_foundry_chunking.html_structure import HtmlStructureChunker, HtmlStructureConfig
from rag_foundry_chunking.json_structure import JsonStructureChunker, JsonStructureConfig, json_path_at
from rag_foundry_chunking.llm_boundaries import LlmBoundaryChunker, LlmBoundaryConfig, boundary_agreement
from rag_foundry_chunking.markdown import MarkdownSectionChunker, MarkdownSectionConfig
from rag_foundry_chunking.model_based import (
    InMemorySegmentationCache,
    ModelChunkContext,
    ModelChunkingError,
    ModelChunkReport,
    ModelChunkResult,
    Rejection,
    SegmentationCache,
    segmentation_cache_key,
)
from rag_foundry_chunking.multilingual import MultilingualChunker, MultilingualConfig, normalize_with_map
from rag_foundry_chunking.paragraph import ParagraphChunker, ParagraphConfig
from rag_foundry_chunking.propositions import (
    PropositionChunker,
    PropositionConfig,
    PropositionRecord,
    PropositionResult,
)
from rag_foundry_chunking.recursive import RecursiveChunker, RecursiveConfig
from rag_foundry_chunking.semantic import SemanticChunker, SemanticConfig
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
    registry.register(ParentChildChunker)
    registry.register(AdaptiveChunker)
    registry.register(RecursiveChunker)
    registry.register(FixedCharChunker)
    registry.register(FixedWordChunker)
    registry.register(FixedTokenChunker)
    registry.register(SlidingWindowChunker)
    registry.register(SentenceChunker)
    registry.register(ParagraphChunker)
    registry.register(MultilingualChunker)
    registry.register(SemanticChunker)
    registry.register(LlmBoundaryChunker)
    registry.register(PropositionChunker)
    registry.register(MarkdownSectionChunker)
    registry.register(TableChunker)
    registry.register(JsonStructureChunker)
    registry.register(HtmlStructureChunker)
    registry.register(CodeSeparatorChunker)
    registry.register(AstCodeChunker)


__all__ = [
    "ParentChildChunker",
    "ParentChildConfig",
    "ParentSpan",
    "ChildSpan",
    "Hierarchy",
    "parent_id_for",
    "AdaptiveChunker",
    "AdaptiveConfig",
    "RegionDecision",
    "PolicyComparison",
    "compare_with_fixed",
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
    "SemanticChunker",
    "SemanticConfig",
    "LlmBoundaryChunker",
    "LlmBoundaryConfig",
    "boundary_agreement",
    "ModelChunkContext",
    "ModelChunkResult",
    "ModelChunkReport",
    "Rejection",
    "ModelChunkingError",
    "SegmentationCache",
    "InMemorySegmentationCache",
    "segmentation_cache_key",
    "PropositionChunker",
    "PropositionConfig",
    "PropositionRecord",
    "PropositionResult",
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
    "HtmlStructureChunker",
    "HtmlStructureConfig",
    "JsonStructureConfig",
    "StructureParseError",
    "json_path_at",
    "CodeSeparatorChunker",
    "CodeSeparatorConfig",
    "AstCodeChunker",
    "AstCodeConfig",
]
