"""Extractors by media type. Importing this package registers the built-in text extractors."""

from rag_foundry_ingestion.extract import markdown, plain  # noqa: F401 (registration side effect)
from rag_foundry_ingestion.extract.base import (
    EXTRACTORS,
    ExtractionLimits,
    ExtractionResult,
    Extractor,
    get_extractor,
)

__all__ = ["EXTRACTORS", "ExtractionLimits", "ExtractionResult", "Extractor", "get_extractor"]
