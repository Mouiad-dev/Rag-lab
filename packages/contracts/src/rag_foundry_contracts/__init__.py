"""Public, framework-neutral RAG Foundry contracts (v1)."""

from rag_foundry_contracts.resolve import (
    MAX_RESOLVE_ANCHORS,
    AnchorResolution,
    AnchorResolveRequest,
    AnchorResolveResponse,
    ResolutionReason,
    ResolutionStatus,
)
from rag_foundry_contracts.source import (
    CONTRACT_VERSION,
    PageRegion,
    Precision,
    SourceAnchor,
    SourceRef,
    TextRange,
    UnsupportedContractVersionError,
    check_contract_version,
)

__all__ = [
    "CONTRACT_VERSION",
    "MAX_RESOLVE_ANCHORS",
    "AnchorResolution",
    "AnchorResolveRequest",
    "AnchorResolveResponse",
    "ResolutionReason",
    "ResolutionStatus",
    "PageRegion",
    "Precision",
    "SourceAnchor",
    "SourceRef",
    "TextRange",
    "UnsupportedContractVersionError",
    "check_contract_version",
]
