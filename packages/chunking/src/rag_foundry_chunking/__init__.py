from rag_foundry_chunking.recursive import RecursiveChunker, RecursiveConfig


def register_builtin_plugins(registry) -> None:
    registry.register(RecursiveChunker)


__all__ = ["RecursiveChunker", "RecursiveConfig", "register_builtin_plugins"]
