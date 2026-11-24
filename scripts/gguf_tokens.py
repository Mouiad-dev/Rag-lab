"""Read-only GGUF header inspector (stdlib only).

Implements the GGUF v3 header layout (little-endian): magic "GGUF", uint32 version,
uint64 tensor_count, uint64 metadata_kv_count, then key/value pairs. Value types:
0 u8, 1 i8, 2 u16, 3 i16, 4 u32, 5 i32, 6 f32, 7 bool, 8 string, 9 array, 10 u64,
11 i64, 12 f64. Tokenizer data comes from `tokenizer.ggml.tokens` (array of string)
and `tokenizer.ggml.token_type` (array of int32; 1 normal, 2 unknown, 3 control,
4 user defined, 5 unused, 6 byte).

The file is untrusted data. This tool only reads bytes and never executes, imports or
evaluates anything from it. Reads are bounded: at most --max-bytes of the file are read,
strings and arrays have hard size caps, and nesting depth is capped.

Usage (run isolated):
    python -I scripts/gguf_tokens.py FILE [--tokens TOKEN ...] [--max-bytes N]

FILE may be a whole GGUF file or a truncated prefix that still contains the metadata.
Exit codes: 0 ok, 1 invalid or too short or too large header, 2 usage or I/O error.
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

MAGIC = b"GGUF"
SUPPORTED_VERSIONS = (3,)
DEFAULT_MAX_BYTES = 64 * 1024 * 1024
MAX_STRING_BYTES = 1 << 20  # 1 MiB per string
MAX_ARRAY_LEN = 2_000_000
MAX_KV_COUNT = 100_000
MAX_DEPTH = 4
TOKENS_KEY = "tokenizer.ggml.tokens"
TYPES_KEY = "tokenizer.ggml.token_type"
MODEL_KEY = "tokenizer.ggml.model"

TOKEN_TYPE_NAMES = {
    1: "normal",
    2: "unknown",
    3: "control",
    4: "user_defined",
    5: "unused",
    6: "byte",
}

_SCALARS = {
    0: ("<B", 1),
    1: ("<b", 1),
    2: ("<H", 2),
    3: ("<h", 2),
    4: ("<I", 4),
    5: ("<i", 4),
    6: ("<f", 4),
    7: ("<?", 1),
    10: ("<Q", 8),
    11: ("<q", 8),
    12: ("<d", 8),
}
_STRING = 8
_ARRAY = 9


class GGUFError(Exception):
    """Header is invalid, truncated, or exceeds a safety limit."""


class _Reader:
    def __init__(self, data: bytes, truncated_prefix: bool) -> None:
        self.data = data
        self.pos = 0
        self.truncated_prefix = truncated_prefix

    def take(self, n: int) -> bytes:
        if n < 0 or self.pos + n > len(self.data):
            hint = " (input is a prefix; fetch more bytes)" if self.truncated_prefix else ""
            raise GGUFError(
                f"header too short: need {n} bytes at offset {self.pos}, have {len(self.data) - self.pos}{hint}"
            )
        out = self.data[self.pos : self.pos + n]
        self.pos += n
        return out

    def unpack(self, fmt: str, size: int):
        return struct.unpack(fmt, self.take(size))[0]

    def string(self, keep: bool = True) -> str:
        n = self.unpack("<Q", 8)
        if n > MAX_STRING_BYTES:
            raise GGUFError(f"string length {n} exceeds cap {MAX_STRING_BYTES}")
        raw = self.take(n)
        return raw.decode("utf-8", errors="replace") if keep else ""


def _read_value(r: _Reader, vtype: int, keep: bool, depth: int):
    """Read one value. When keep is False the value is consumed but not stored."""
    if vtype in _SCALARS:
        fmt, size = _SCALARS[vtype]
        return r.unpack(fmt, size)
    if vtype == _STRING:
        return r.string(keep)
    if vtype == _ARRAY:
        if depth >= MAX_DEPTH:
            raise GGUFError("array nesting too deep")
        etype = r.unpack("<I", 4)
        n = r.unpack("<Q", 8)
        if n > MAX_ARRAY_LEN:
            raise GGUFError(f"array length {n} exceeds cap {MAX_ARRAY_LEN}")
        if etype not in _SCALARS and etype not in (_STRING, _ARRAY):
            raise GGUFError(f"unknown array element type {etype}")
        if etype in _SCALARS:
            fmt, size = _SCALARS[etype]
            blob = r.take(n * size)  # bounded by the cap and by the bytes actually present
            if not keep:
                return None
            return list(struct.unpack("<" + str(n) + fmt[1], blob))
        items = []
        for _ in range(n):
            v = _read_value(r, etype, keep, depth + 1)
            if keep:
                items.append(v)
        return items if keep else None
    raise GGUFError(f"unknown value type {vtype}")


def parse_header(data: bytes, truncated_prefix: bool = False) -> dict:
    """Parse a GGUF v3 header from bytes. Returns version, counts and kept metadata.

    Kept metadata: every `general.*` key, `tokenizer.ggml.model`, and the two tokenizer
    arrays. Everything else is read past and discarded.
    """
    r = _Reader(data, truncated_prefix)
    if r.take(4) != MAGIC:
        raise GGUFError("not a GGUF file (bad magic)")
    version = r.unpack("<I", 4)
    if version not in SUPPORTED_VERSIONS:
        raise GGUFError(f"unsupported GGUF version {version}; implemented: 3")
    tensor_count = r.unpack("<Q", 8)
    kv_count = r.unpack("<Q", 8)
    if kv_count > MAX_KV_COUNT:
        raise GGUFError(f"metadata count {kv_count} exceeds cap {MAX_KV_COUNT}")
    meta: dict = {}
    for _ in range(kv_count):
        key = r.string()
        vtype = r.unpack("<I", 4)
        keep = key.startswith("general.") or key in (TOKENS_KEY, TYPES_KEY, MODEL_KEY)
        value = _read_value(r, vtype, keep, 0)
        if keep:
            meta[key] = value
    return {
        "version": version,
        "tensor_count": tensor_count,
        "kv_count": kv_count,
        "metadata": meta,
    }


def lookup_tokens(meta: dict, wanted: list[str]) -> list[dict]:
    """Return id/type/occurrences for each wanted token string."""
    tokens = meta.get(TOKENS_KEY)
    if tokens is None:
        raise GGUFError(f"{TOKENS_KEY} not found in header")
    if not isinstance(tokens, list) or not all(isinstance(t, str) for t in tokens):
        raise GGUFError(f"{TOKENS_KEY} must be an array of strings")
    types = meta.get(TYPES_KEY, [])
    if not isinstance(types, list) or not all(isinstance(t, int) and not isinstance(t, bool) for t in types):
        raise GGUFError(f"{TYPES_KEY} must be an array of integers")
    index: dict[str, list[int]] = {}
    for i, t in enumerate(tokens):
        index.setdefault(t, []).append(i)
    out = []
    for w in wanted:
        ids = index.get(w, [])
        tid = ids[0] if ids else None
        ttype = types[tid] if tid is not None and tid < len(types) else None
        out.append(
            {
                "token": w,
                "found": bool(ids),
                "id": tid,
                "occurrences": len(ids),
                "type": ttype,
                "type_name": TOKEN_TYPE_NAMES.get(ttype, "?") if ttype is not None else None,
            }
        )
    return out


def read_prefix(path: Path, max_bytes: int) -> tuple[bytes, bool]:
    """Read at most max_bytes; the bool says the file is longer than what was read."""
    with path.open("rb") as fh:
        data = fh.read(max_bytes + 1)
    if len(data) > max_bytes:
        return data[:max_bytes], True
    return data, False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Read-only GGUF header inspector.")
    ap.add_argument("file", type=Path)
    ap.add_argument("--tokens", nargs="*", default=[], help="token strings to look up")
    ap.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)
    args = ap.parse_args(argv)
    if args.max_bytes <= 0:
        print("error: --max-bytes must be positive", file=sys.stderr)
        return 2
    try:
        data, more = read_prefix(args.file, args.max_bytes)
    except OSError as exc:
        print(f"error: cannot read file: {exc.strerror or type(exc).__name__}", file=sys.stderr)
        return 2
    try:
        hdr = parse_header(data, truncated_prefix=True)
        meta = hdr["metadata"]
        print(f"file_bytes_read: {len(data)}{' (more bytes not read)' if more else ''}")
        print(f"gguf_version: {hdr['version']}")
        print(f"tensor_count: {hdr['tensor_count']}")
        print(f"kv_count: {hdr['kv_count']}")
        for k in sorted(k for k in meta if k.startswith("general.")):
            v = meta[k]
            shown = f"<array of {len(v)}>" if isinstance(v, list) else repr(v)
            print(f"{k!r}: {shown}")
        print(f"{MODEL_KEY}: {meta.get(MODEL_KEY)!r}")
        tokens = meta.get(TOKENS_KEY)
        print(f"vocab_size: {len(tokens) if tokens is not None else 'n/a'}")
        for row in lookup_tokens(meta, args.tokens) if args.tokens else []:
            print(
                f"token {row['token']!r}: found={row['found']} id={row['id']} "
                f"occurrences={row['occurrences']} type={row['type']} ({row['type_name']})"
            )
    except GGUFError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
