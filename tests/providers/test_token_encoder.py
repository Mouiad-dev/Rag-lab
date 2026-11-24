"""TokenEncoder port, message validation and adapter guard rails. Offline; no model is loaded."""

import asyncio
import inspect
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from _fake_token_encoder import FakeTokenEncoder
from pydantic import ValidationError
from rag_foundry_providers import sentence_transformers_tokens
from rag_foundry_providers.errors import AdapterDependencyError
from rag_foundry_providers.messages import TokenEncoding
from rag_foundry_providers.ports import TokenEncoder
from rag_foundry_providers.sentence_transformers_tokens import SentenceTransformersTokenEncoder

ARABIC = (
    "\u0627\u0644\u0643\u062a\u0627\u0628 "
    "\u0627\u0644\u062c\u062f\u064a\u062f \u0639\u0644\u0649 "
    "\u0627\u0644\u0637\u0627\u0648\u0644\u0629"
)


def run(coro):
    return asyncio.run(coro)


@pytest.mark.parametrize("text", ["alpha beta  gamma", ARABIC, f"mix {ARABIC} end"])
def test_offsets_round_trip(text):
    result = run(FakeTokenEncoder(max_tokens=50).encode_tokens([text], head="hidden", input_type="document"))
    item = result.items[0]
    assert item.truncated is False
    assert [text[s:e] for s, e in item.offsets] == text.split()
    assert len(item.vectors) == len(item.offsets)
    assert result.dim == 4 and all(len(v) == 4 for v in item.vectors)


def test_truncation_flagged_and_calls_counted():
    encoder = FakeTokenEncoder(max_tokens=3)
    result = run(encoder.encode_tokens(["a b c d e", "a b"], head="colbert", input_type="query"))
    assert [i.truncated for i in result.items] == [True, False]
    assert len(result.items[0].offsets) == 3
    assert encoder.calls == 1


def test_protocol_checks():
    assert isinstance(FakeTokenEncoder(), TokenEncoder)
    assert isinstance(SentenceTransformersTokenEncoder(model_dir="missing"), TokenEncoder)
    assert not isinstance(object(), TokenEncoder)


def test_token_encoding_validation():
    with pytest.raises(ValidationError):
        TokenEncoding(vectors=((1.0,),), offsets=(), truncated=False)
    with pytest.raises(ValidationError):
        TokenEncoding(vectors=((1.0,), (1.0,)), offsets=((3, 4), (1, 2)), truncated=False)
    with pytest.raises(ValidationError):
        TokenEncoding(vectors=((1.0,),), offsets=((3, 2),), truncated=False)
    with pytest.raises(ValidationError):
        TokenEncoding(vectors=((1.0,),), offsets=((-1, 2),), truncated=False)
    with pytest.raises(ValidationError):
        TokenEncoding(vectors=((1.0,), (1.0, 2.0)), offsets=((0, 1), (1, 2)), truncated=False)
    assert TokenEncoding(vectors=((1.0,), (1.0,)), offsets=((0, 0), (0, 2)), truncated=False)


def _encode(encoder, head="hidden"):
    return run(encoder.encode_tokens(["text"], head=head, input_type="document"))


def test_missing_dir_raises(tmp_path):
    with pytest.raises(AdapterDependencyError):
        _encode(SentenceTransformersTokenEncoder(model_dir=tmp_path / "nope"))


def test_unset_dir_raises(monkeypatch):
    monkeypatch.delenv("RAG_TOKEN_MODEL_DIR", raising=False)
    with pytest.raises(AdapterDependencyError):
        _encode(SentenceTransformersTokenEncoder())


def test_bin_without_safetensors_never_falls_back(tmp_path: Path):
    (tmp_path / "pytorch_model.bin").write_bytes(b"not a real model")
    (tmp_path / "colbert_linear.pt").write_bytes(b"not a real head")
    with pytest.raises(AdapterDependencyError):
        _encode(SentenceTransformersTokenEncoder(model_dir=tmp_path))


def test_missing_colbert_head_raises(tmp_path: Path):
    encoder = SentenceTransformersTokenEncoder(model_dir=tmp_path)
    encoder._model = object()  # skip the model load; only the head file is checked
    with pytest.raises(AdapterDependencyError):
        encoder._load_head()


def test_module_source_has_no_pickle_loading():
    source = inspect.getsource(sentence_transformers_tokens)
    assert "torch." + "load" not in source
    assert "pickle" not in source
    assert "weights_only" not in source


class _StubTokenizer:
    is_fast = True

    def __init__(self, fast: bool = True) -> None:
        self.is_fast = fast

    def __call__(self, text, **kwargs):
        import torch

        if "return_tensors" not in kwargs:
            return {"input_ids": [0, 5, 6, 7, 2]}
        return {
            "input_ids": torch.tensor([[0, 5, 6, 7, 2]]),
            "attention_mask": torch.ones(1, 5, dtype=torch.long),
            "special_tokens_mask": torch.tensor([[1, 0, 0, 0, 1]]),
            "offset_mapping": torch.tensor([[[0, 0], [0, 2], [3, 5], [6, 8], [0, 0]]]),
        }


class _StubTransformer:
    def __init__(self, hidden) -> None:
        self._hidden = hidden
        self.config = type("Config", (), {"hidden_size": hidden.shape[-1]})()

    def __call__(self, input_ids, attention_mask):
        return type("Output", (), {"last_hidden_state": self._hidden.unsqueeze(0)})()


class _StubModel:
    def __init__(self, hidden, fast: bool = True) -> None:
        self.tokenizer = _StubTokenizer(fast)
        self._module = type("Module", (), {"auto_model": _StubTransformer(hidden)})()

    def __getitem__(self, index):
        return self._module


def _stub_encoder(max_tokens=8, fast=True):
    torch = pytest.importorskip("torch")
    hidden = torch.tensor([[9.0, 9.0, 9.0], [1.0, 2.0, 2.0], [0.0, 3.0, 4.0], [5.0, 0.0, 0.0], [8.0, 8.0, 8.0]])
    linear = torch.nn.Linear(3, 2)
    with torch.no_grad():
        linear.weight.copy_(torch.tensor([[1.0, 0.0, 1.0], [0.0, 2.0, 0.0]]))
        linear.bias.copy_(torch.tensor([0.5, -0.5]))
    encoder = SentenceTransformersTokenEncoder(model_dir="unused", max_tokens=max_tokens)
    encoder._model = _StubModel(hidden, fast)
    encoder._linear = linear
    return encoder, hidden, linear


def test_encode_hidden_drops_special_tokens_and_keeps_raw_vectors():
    torch = pytest.importorskip("torch")
    encoder, hidden, _ = _stub_encoder()
    result = run(encoder.encode_tokens(["ab cd ef"], head="hidden", input_type="document"))
    item = result.items[0]
    assert item.offsets == ((0, 2), (3, 5), (6, 8))
    assert item.truncated is False
    assert result.dim == 3
    assert torch.equal(torch.tensor(item.vectors), hidden[1:4])
    assert result.usage.input_tokens == 5


def test_encode_colbert_applies_linear_then_l2_normalise():
    torch = pytest.importorskip("torch")
    encoder, hidden, linear = _stub_encoder()
    result = run(encoder.encode_tokens(["ab cd ef"], head="colbert", input_type="query"))
    item = result.items[0]
    expected = torch.nn.functional.normalize(linear(hidden[1:4]), dim=-1).detach()
    assert result.dim == 2
    assert item.offsets == ((0, 2), (3, 5), (6, 8))
    assert torch.allclose(torch.tensor(item.vectors), expected, atol=1e-6)
    assert all(abs(sum(x * x for x in v) - 1.0) < 1e-5 for v in item.vectors)


def test_encode_flags_truncation_when_untruncated_length_exceeds_max_tokens():
    encoder, _, _ = _stub_encoder(max_tokens=4)
    result = run(encoder.encode_tokens(["ab cd ef"], head="hidden", input_type="document"))
    assert result.items[0].truncated is True


def test_encode_requires_fast_tokenizer():
    encoder, _, _ = _stub_encoder(fast=False)
    with pytest.raises(AdapterDependencyError):
        run(encoder.encode_tokens(["ab cd ef"], head="hidden", input_type="document"))
