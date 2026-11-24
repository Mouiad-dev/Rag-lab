"""Pure prompt, citation, and retrieval gate helpers."""

from rag_foundry_answering.claims import CitationReport as CitationReport
from rag_foundry_answering.claims import validate_citations as validate_citations


def register_builtin_plugins(registry) -> None:
    from rag_foundry_answering.cag import CagWorkflow
    from rag_foundry_answering.self_rag import SelfRagWorkflow

    registry.register(CagWorkflow)
    registry.register(SelfRagWorkflow)
