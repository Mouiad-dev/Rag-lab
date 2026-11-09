"""Local sentence-transformers cross-encoder adapter; no files are fetched at runtime."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Sequence
from pathlib import Path
from time import monotonic

from rag_foundry_providers.capabilities import Capabilities, ModelKind, ModelProfile
from rag_foundry_providers.errors import AdapterDependencyError
from rag_foundry_providers.messages import RerankItem, RerankResult
from rag_foundry_providers.ports import Reranker
from rag_foundry_providers.usage import Operation, TokenSource, Usage


class SentenceTransformersCrossEncoder(Reranker):
    """Score query/document pairs locally. If tokenizer is absent, estimate by whitespace word count."""

    def __init__(self, model_dir: str | Path | None = None) -> None:
        configured = model_dir or os.environ.get("RAG_RERANK_MODEL_DIR")
        self._model_dir = Path(configured) if configured else None
        self._model = None
        self.activation: str | None = None
        model_name = self._model_dir.name if self._model_dir else "bge-reranker-v2-m3"
        self.profile = ModelProfile(
            id="r19-cross-encoder",
            provider="sentence-transformers",
            model=model_name,
            kind=ModelKind.RERANKER,
            capabilities=Capabilities(local=True),
        )

    def _load(self) -> None:
        if self._model is not None:
            return
        if self._model_dir is None or not self._model_dir.is_dir():
            raise AdapterDependencyError("sparse")
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise AdapterDependencyError("sparse") from exc
        try:
            model = CrossEncoder(str(self._model_dir), local_files_only=True, device="cpu", trust_remote_code=False)
        except (OSError, ValueError) as exc:
            raise AdapterDependencyError("sparse") from exc
        if model.num_labels != 1:
            raise AdapterDependencyError("sparse")
        self._model = model
        self.activation = type(model.activation_fn).__name__

    async def rerank(self, query: str, documents: Sequence[str], *, top_n: int | None = None) -> RerankResult:
        start = monotonic()
        items, tokens = await asyncio.to_thread(self._rerank, query, list(documents), top_n)
        return RerankResult(
            items=items,
            usage=Usage(
                profile_id=self.profile.id,
                provider=self.profile.provider,
                model=self.profile.model,
                operation=Operation.RERANK,
                input_tokens=tokens,
                output_tokens=0,
                token_source=TokenSource.ESTIMATED,
                latency_ms=(monotonic() - start) * 1000,
                estimated_cost_usd=0,
            ),
        )

    def _rerank(self, query: str, documents: list[str], top_n: int | None):
        if not documents:
            return (), 0
        self._load()
        scores = self._model.predict([(query, doc) for doc in documents], convert_to_numpy=True)
        items = [RerankItem(index=i, score=float(score)) for i, score in enumerate(scores)]
        items.sort(key=lambda item: (-item.score, item.index))
        if top_n is not None:
            items = items[:top_n]
        tokenizer = getattr(self._model, "tokenizer", None)
        if tokenizer is not None:
            tokens = sum(
                len(ids) for ids in tokenizer([query] * len(documents), documents, truncation=True)["input_ids"]
            )
        else:
            tokens = sum(len((query + " " + doc).split()) for doc in documents)
        return tuple(items), tokens
