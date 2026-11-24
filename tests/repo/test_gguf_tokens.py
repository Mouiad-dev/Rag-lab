"""scripts/gguf_tokens.py: synthetic GGUF v3 header parsing, bounds and truncation."""

import importlib.util
import struct
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "gguf_tokens.py"
_spec = importlib.util.spec_from_file_location("gguf_tokens", SCRIPT)
gt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gt)

TOKENS = ["<unk>", "<s>", "a", "[Retrieval]", "[No Retrieval]", "[Utility:5]"]
TYPES = [2, 3, 1, 4, 4, 4]


def s(text: str) -> bytes:
    raw = text.encode()
    return struct.pack("<Q", len(raw)) + raw


def kv_str(key: str, val: str) -> bytes:
    return s(key) + struct.pack("<I", 8) + s(val)


def kv_u32(key: str, val: int) -> bytes:
    return s(key) + struct.pack("<II", 4, val)


def header(kvs: list[bytes], version: int = 3) -> bytes:
    return b"GGUF" + struct.pack("<IQQ", version, 7, len(kvs)) + b"".join(kvs)


def build() -> bytes:
    toks = s("tokenizer.ggml.tokens") + struct.pack("<IIQ", 9, 8, len(TOKENS))
    toks += b"".join(s(t) for t in TOKENS)
    types = s("tokenizer.ggml.token_type") + struct.pack("<IIQ", 9, 5, len(TYPES))
    types += struct.pack(f"<{len(TYPES)}i", *TYPES)
    other = s("some.other.array") + struct.pack("<IIQ", 9, 8, 2) + s("x") + s("y")
    return header(
        [
            kv_str("general.architecture", "llama"),
            kv_str("general.name", "synthetic"),
            kv_u32("llama.context_length", 4096),
            other,
            kv_str("tokenizer.ggml.model", "llama"),
            toks,
            types,
        ]
    )


def test_parses_metadata_tokens_ids_types():
    hdr = gt.parse_header(build())
    meta = hdr["metadata"]
    assert hdr["version"] == 3 and hdr["tensor_count"] == 7
    assert meta["general.architecture"] == "llama"
    assert meta["tokenizer.ggml.model"] == "llama"
    assert "llama.context_length" not in meta and "some.other.array" not in meta
    rows = gt.lookup_tokens(meta, ["[Retrieval]", "[Utility:5]", "<s>", "[Nope]"])
    assert [r["id"] for r in rows] == [3, 5, 1, None]
    assert [r["type"] for r in rows] == [4, 4, 3, None]
    assert rows[0]["type_name"] == "user_defined" and rows[3]["found"] is False
    assert rows[0]["occurrences"] == 1


def test_truncated_prefix_fails_cleanly():
    data = build()
    for cut in (0, 3, 10, 30, len(data) - 5):
        with pytest.raises(gt.GGUFError, match="too short|bad magic"):
            gt.parse_header(data[:cut], truncated_prefix=True)


def test_bad_magic_and_version():
    with pytest.raises(gt.GGUFError, match="bad magic"):
        gt.parse_header(b"XXXX" + build()[4:])
    with pytest.raises(gt.GGUFError, match="unsupported GGUF version"):
        gt.parse_header(header([], version=2))


def test_oversized_lengths_rejected():
    big_str = s("general.name") + struct.pack("<I", 8) + struct.pack("<Q", 1 << 40)
    with pytest.raises(gt.GGUFError, match="string length"):
        gt.parse_header(header([big_str]))
    big_arr = s("tokenizer.ggml.tokens") + struct.pack("<IIQ", 9, 8, 1 << 40)
    with pytest.raises(gt.GGUFError, match="array length"):
        gt.parse_header(header([big_arr]))
    big_key = struct.pack("<Q", 1 << 40)
    with pytest.raises(gt.GGUFError, match="string length"):
        gt.parse_header(header([big_key]))
    huge_count = b"GGUF" + struct.pack("<IQQ", 3, 0, 1 << 40)
    with pytest.raises(gt.GGUFError, match="metadata count"):
        gt.parse_header(huge_count)


def test_length_within_cap_but_beyond_data_is_truncation_not_allocation():
    arr = s("some.arr") + struct.pack("<IIQ", 9, 10, 1_000_000)  # 8 MB claimed, none present
    with pytest.raises(gt.GGUFError, match="too short"):
        gt.parse_header(header([arr]))


def test_nesting_depth_capped():
    nested = b"".join(struct.pack("<IQ", 9, 1) for _ in range(10))
    arr = s("some.arr") + struct.pack("<I", 9) + nested
    with pytest.raises(gt.GGUFError, match="nesting"):
        gt.parse_header(header([arr]))


def test_missing_tokens_key():
    hdr = gt.parse_header(header([kv_str("general.name", "x")]))
    with pytest.raises(gt.GGUFError, match="not found"):
        gt.lookup_tokens(hdr["metadata"], ["a"])


def test_wrong_token_array_shapes_rejected():
    with pytest.raises(gt.GGUFError, match="array of strings"):
        gt.lookup_tokens({"tokenizer.ggml.tokens": "abc"}, ["a"])
    with pytest.raises(gt.GGUFError, match="array of strings"):
        gt.lookup_tokens({"tokenizer.ggml.tokens": 7}, ["a"])
    with pytest.raises(gt.GGUFError, match="array of integers"):
        gt.lookup_tokens({"tokenizer.ggml.tokens": ["a"], "tokenizer.ggml.token_type": 4}, ["a"])
    with pytest.raises(gt.GGUFError, match="array of integers"):
        gt.lookup_tokens({"tokenizer.ggml.tokens": ["a"], "tokenizer.ggml.token_type": ["x"]}, ["a"])


def test_cli_escapes_metadata(tmp_path):
    f = tmp_path / "e.gguf"
    f.write_bytes(header([kv_str("general.name", "evil\x1b[2Jname")]))
    run = subprocess.run([sys.executable, "-I", str(SCRIPT), str(f)], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    assert "\x1b" not in run.stdout and "\\x1b" in run.stdout


def test_cli_isolated(tmp_path):
    f = tmp_path / "h.gguf"
    f.write_bytes(build())
    run = subprocess.run(
        [sys.executable, "-I", str(SCRIPT), str(f), "--tokens", "[Retrieval]"],
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, run.stderr
    assert "'general.name': 'synthetic'" in run.stdout
    assert "id=3" in run.stdout and "user_defined" in run.stdout
    short = tmp_path / "s.gguf"
    short.write_bytes(build()[:40])
    run = subprocess.run([sys.executable, "-I", str(SCRIPT), str(short)], capture_output=True, text=True)
    assert run.returncode == 1 and "too short" in run.stderr
    run = subprocess.run(
        [sys.executable, "-I", str(SCRIPT), str(tmp_path / "missing")],
        capture_output=True,
        text=True,
    )
    assert run.returncode == 2
