"""Optional local learned sparse adapter. Query encoding is tokenizer/table lookup only."""

import asyncio
import os
from collections.abc import Sequence
from pathlib import Path
from time import monotonic
from typing import Literal

from rag_foundry_providers.capabilities import Capabilities, ModelKind, ModelProfile
from rag_foundry_providers.errors import AdapterDependencyError
from rag_foundry_providers.messages import SparseEncodingResult, SparseVector
from rag_foundry_providers.ports import SparseEncoder
from rag_foundry_providers.usage import Operation, TokenSource, Usage


class SentenceTransformersSparseEncoder(SparseEncoder):
    """Uses only approved local files; ``encode_query`` is tokenizer/table lookup with no model compute."""

    def __init__(self, model_dir: str | Path | None = None) -> None:
        configured = model_dir or os.environ.get("RAG_SPARSE_MODEL_DIR")
        self._model_dir = Path(configured) if configured else None
        self._encoder = None
        self.profile = ModelProfile(
            id="opensearch-sparse",
            provider="sentence-transformers",
            model="opensearch-neural-sparse-encoding-multilingual-v1",
            kind=ModelKind.SPARSE,
            capabilities=Capabilities(local=True),
        )

    def _load(self) -> None:
        if self._encoder is not None:
            return
        if self._model_dir is None or not self._model_dir.is_dir():
            raise AdapterDependencyError("sparse")
        try:
            from sentence_transformers import SparseEncoder as _SparseEncoder
        except ImportError as exc:
            raise AdapterDependencyError("sparse") from exc
        try:
            self._encoder = _SparseEncoder(str(self._model_dir), local_files_only=True, device="cpu")
        except (OSError, ValueError) as exc:
            raise AdapterDependencyError("sparse") from exc

    async def encode(self, texts: Sequence[str], *, input_type: Literal["document", "query"]) -> SparseEncodingResult:
        start = monotonic()
        result = await asyncio.to_thread(self._encode, list(texts), input_type)
        elapsed = (monotonic() - start) * 1000
        return SparseEncodingResult(
            vectors=result[0],
            usage=Usage(
                profile_id=self.profile.id,
                provider=self.profile.provider,
                model=self.profile.model,
                operation=Operation.EMBED,
                input_tokens=result[1],
                output_tokens=0,
                token_source=TokenSource.ESTIMATED,
                latency_ms=elapsed,
                estimated_cost_usd=0,
            ),
        )

    def _encode(self, texts: list[str], input_type: str) -> tuple[tuple[SparseVector, ...], int]:
        self._load()
        method = self._encoder.encode_document if input_type == "document" else self._encoder.encode_query
        encoded = method(texts)
        decoded = self._encoder.decode(encoded)
        vectors = tuple(
            SparseVector(
                terms=tuple(str(term) for term, weight in row if float(weight) > 0),
                weights=tuple(float(weight) for _, weight in row if float(weight) > 0),
            )
            for row in decoded
        )
        tokens = sum(len(ids) for ids in self._encoder.tokenizer(texts, truncation=True)["input_ids"])
        return vectors, tokens
