from rag_foundry_chunking._units import LabRegexTokenizer, Tokenizer
from rag_foundry_chunking.adaptive import (
    AdaptiveChunker,
    AdaptiveConfig,
    PolicyComparison,
    RegionDecision,
    compare_with_fixed,
)
from rag_foundry_chunking.fixed import (
    FixedCharChunker,
    FixedCharConfig,
    FixedTokenChunker,
    FixedTokenConfig,
    FixedWordChunker,
    FixedWordConfig,
)
from rag_foundry_chunking.llm_boundaries import LlmBoundaryChunker, LlmBoundaryConfig, boundary_agreement
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
from rag_foundry_chunking.recursive import RecursiveChunker, RecursiveConfig
from rag_foundry_chunking.semantic import SemanticChunker, SemanticConfig
from rag_foundry_chunking.sentence import SentenceChunker, SentenceConfig
from rag_foundry_chunking.sliding import SlidingWindowChunker, SlidingWindowConfig, WindowStats, window_stats


def register_builtin_plugins(registry) -> None:
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


__all__ = [
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
]
