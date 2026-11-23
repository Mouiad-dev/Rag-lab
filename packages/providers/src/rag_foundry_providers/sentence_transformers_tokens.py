"""Optional local per-token adapter for BAAI/bge-m3 (C20 hidden states, R45 ColBERT head).

Loads only safetensors files from an approved local directory: ``model.safetensors`` and, for the
ColBERT head, ``colbert_linear.safetensors``. It never falls back to the original ``.bin`` or ``.pt``
files and never uses object-deserialising loaders. Nothing is downloaded.

ColBERT head reference (BAAI/bge-m3 official code, MIT licence): FlagEmbedding commit
897baa67c0ffddbfccf4e6776ff55310d67809d4, ``FlagEmbedding/finetune/embedder/encoder_only/m3/modeling.py``
``_colbert_embedding``: ``colbert_linear(last_hidden_state[:, 1:])`` masked by the attention mask, then
L2 normalisation in the inference path. Source: https://github.com/FlagOpen/FlagEmbedding

Deviation: official ColBERT vectors drop only the first token and keep the final ``</s>``. This port
drops all special tokens, so item vectors align one to one with text offsets.

``input_type`` is ignored: BGE-M3 has no query instruction, so documents and queries encode the same way.
"""

import asyncio
import os
from collections.abc import Sequence
from pathlib import Path
from time import monotonic
from typing import Any, Literal

from rag_foundry_providers.capabilities import Capabilities, ModelKind, ModelProfile
from rag_foundry_providers.errors import AdapterDependencyError
from rag_foundry_providers.messages import TokenEncoding, TokenEncodingResult
from rag_foundry_providers.ports import TokenEncoder
from rag_foundry_providers.usage import Operation, TokenSource, Usage


class SentenceTransformersTokenEncoder(TokenEncoder):
    def __init__(self, model_dir: str | Path | None = None, *, max_tokens: int = 8192) -> None:
        configured = model_dir or os.environ.get("RAG_TOKEN_MODEL_DIR")
        self._model_dir = Path(configured) if configured else None
        self.max_tokens = max_tokens
        self._model: Any = None
        self._linear: Any = None
        self.revision: str | None = None
        self.profile = self._make_profile(None)

    @staticmethod
    def _make_profile(revision: str | None) -> ModelProfile:
        model = "BAAI/bge-m3" if revision is None else f"BAAI/bge-m3@{revision}"
        return ModelProfile(
            id="bge-m3-tokens",
            provider="sentence-transformers",
            model=model,
            kind=ModelKind.EMBEDDING,
            capabilities=Capabilities(local=True, embedding_dim=1024),
        )

    def _load(self) -> None:
        if self._model is not None:
            return
        directory = self._model_dir
        if directory is None or not directory.is_dir() or not (directory / "model.safetensors").is_file():
            raise AdapterDependencyError("tokens")  # model files missing or RAG_TOKEN_MODEL_DIR unset
        try:
            from safetensors import SafetensorError
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise AdapterDependencyError("tokens") from exc
        try:
            model = SentenceTransformer(
                str(directory),
                device="cpu",
                local_files_only=True,
                trust_remote_code=False,
                model_kwargs={"use_safetensors": True},
            )
        except (OSError, ValueError, RuntimeError, SafetensorError) as exc:
            raise AdapterDependencyError("tokens") from exc
        model.eval()
        self._model = model
        revision_file = directory / "REVISION"
        if revision_file.is_file():
            self.revision = revision_file.read_text(encoding="ascii", errors="replace").strip() or None
            self.profile = self._make_profile(self.revision)

    def _load_head(self) -> Any:
        if self._linear is not None:
            return self._linear
        if self._model_dir is None:
            raise AdapterDependencyError("tokens")
        path = self._model_dir / "colbert_linear.safetensors"
        if not path.is_file():
            raise AdapterDependencyError("tokens")
        try:
            import torch
            from safetensors import SafetensorError
            from safetensors.torch import load_file
        except ImportError as exc:
            raise AdapterDependencyError("tokens") from exc
        try:
            tensors = load_file(str(path), device="cpu")
            weight, bias = tensors["weight"], tensors["bias"]
            linear = torch.nn.Linear(weight.shape[1], weight.shape[0])
            with torch.no_grad():
                linear.weight.copy_(weight)
                linear.bias.copy_(bias)
        except (OSError, ValueError, KeyError, RuntimeError, SafetensorError) as exc:
            raise AdapterDependencyError("tokens") from exc
        linear.eval()
        self._linear = linear
        return linear

    async def encode_tokens(
        self,
        texts: Sequence[str],
        *,
        head: Literal["hidden", "colbert"],
        input_type: Literal["document", "query"],
    ) -> TokenEncodingResult:
        start = monotonic()
        items, dim, tokens = await asyncio.to_thread(self._encode, list(texts), head)
        elapsed = (monotonic() - start) * 1000
        return TokenEncodingResult(
            items=items,
            dim=dim,
            usage=Usage(
                profile_id=self.profile.id,
                provider=self.profile.provider,
                model=self.profile.model,
                operation=Operation.EMBED,
                input_tokens=tokens,
                output_tokens=0,
                token_source=TokenSource.ESTIMATED,
                latency_ms=elapsed,
                estimated_cost_usd=0,
            ),
        )

    def _encode(self, texts: list[str], head: str) -> tuple[tuple[TokenEncoding, ...], int, int]:
        self._load()
        import torch

        linear = self._load_head() if head == "colbert" else None
        tokenizer = self._model.tokenizer
        if not getattr(tokenizer, "is_fast", False):
            raise AdapterDependencyError("tokens")  # offsets need a fast tokenizer
        transformer = self._model[0].auto_model
        items: list[TokenEncoding] = []
        dim = int(transformer.config.hidden_size) if linear is None else int(linear.out_features)
        total = 0
        for text in texts:
            full_length = len(tokenizer(text, add_special_tokens=True)["input_ids"])
            enc = tokenizer(
                text,
                truncation=True,
                max_length=self.max_tokens,
                return_offsets_mapping=True,
                return_special_tokens_mask=True,
                return_tensors="pt",
            )
            total += int(enc["input_ids"].shape[1])
            with torch.no_grad():
                hidden = transformer(
                    input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]
                ).last_hidden_state[0]
                if linear is not None:
                    hidden = torch.nn.functional.normalize(linear(hidden), dim=-1)
            keep = [i for i, special in enumerate(enc["special_tokens_mask"][0].tolist()) if not special]
            offsets = enc["offset_mapping"][0].tolist()
            items.append(
                TokenEncoding(
                    vectors=tuple(tuple(float(x) for x in hidden[i].tolist()) for i in keep),
                    offsets=tuple((int(offsets[i][0]), int(offsets[i][1])) for i in keep),
                    truncated=full_length > self.max_tokens,
                )
            )
        return tuple(items), dim, total
