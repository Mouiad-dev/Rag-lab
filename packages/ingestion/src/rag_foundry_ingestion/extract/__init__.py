"""Extractors by media type. Importing this package registers the built-in text extractors."""

from rag_foundry_ingestion.extract import (  # noqa: F401 (registration side effect)
    html,  # noqa: F401 (registers the HTML extractor, or a typed missing-extra stub)
    markdown,
    pdf,  # noqa: F401 (registers the PDF extractor, or a typed missing-extra stub)
    plain,
)
from rag_foundry_ingestion.extract.base import (
    EXTRACTORS,
    ExtractionLimits,
    ExtractionResult,
    Extractor,
    get_extractor,
)

__all__ = ["EXTRACTORS", "ExtractionLimits", "ExtractionResult", "Extractor", "get_extractor"]
