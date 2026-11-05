"""Offset maps between a normalized text and its original text (ADR-0010 D6/D7/D11).

A map goes from "normalized" (n) to "original" (o), both measured in Unicode code points. Segments are stored flat in
`array('q')`, five integers each: `n0, n1, o0, o1, kind`. Pure Python, no dependencies.

Kinds:
- `CP`: equal lengths, exact per code point. Adjacent `CP` runs merge while building.
- `ATOMIC`: any lengths, never split. Zero normalized length is a deletion, zero original length an insertion.
- `ATTACH`: deletion of a mark that belongs to the preceding character (zero normalized length).
"""

from array import array
from collections.abc import Iterator
from typing import NamedTuple

CP = 0
ATOMIC = 1
ATTACH = 2
_KINDS = (CP, ATOMIC, ATTACH)
_STRIDE = 5


class MappedRange(NamedTuple):
    start: int
    end: int
    exact: bool


class OffsetMapBuilder:
    """Appends segments in order and merges adjacent `CP` runs (and adjacent `ATTACH` runs)."""

    __slots__ = ("_s", "n", "o")

    def __init__(self) -> None:
        self._s = array("q")
        self.n = 0
        self.o = 0

    def add(self, n_len: int, o_len: int, kind: int) -> None:
        if n_len == 0 and o_len == 0:
            return
        s = self._s
        if s and kind == s[-1] and kind in (CP, ATTACH):
            s[-4] += n_len
            s[-2] += o_len
        else:
            s.extend((self.n, self.n + n_len, self.o, self.o + o_len, kind))
        self.n += n_len
        self.o += o_len

    def cp(self, length: int) -> None:
        self.add(length, length, CP)

    def atomic(self, n_len: int, o_len: int) -> None:
        self.add(n_len, o_len, ATOMIC)

    def attach(self, o_len: int) -> None:
        self.add(0, o_len, ATTACH)

    def build(self) -> "OffsetMap":
        return OffsetMap._trusted(self._s, self.n, self.o)


class OffsetMap:
    """Immutable validated map normalized -> original. Use `OffsetMapBuilder` to create one."""

    __slots__ = ("_s", "n_len", "o_len")

    def __init__(self, segments: array, n_len: int, o_len: int) -> None:
        if segments.typecode != "q":
            raise ValueError("segments must be array('q')")
        self._validate(segments, n_len, o_len)
        self._s = segments
        self.n_len = n_len
        self.o_len = o_len

    @classmethod
    def _trusted(cls, segments: array, n_len: int, o_len: int) -> "OffsetMap":
        self = object.__new__(cls)
        self._s = segments
        self.n_len = n_len
        self.o_len = o_len
        return self

    @staticmethod
    def _validate(s: array, n_len: int, o_len: int) -> None:
        if len(s) % _STRIDE:
            raise ValueError("segment array length must be a multiple of 5")
        n = o = 0
        for i in range(0, len(s), _STRIDE):
            n0, n1, o0, o1, kind = s[i : i + _STRIDE]
            if kind not in _KINDS:
                raise ValueError(f"unknown segment kind {kind}")
            if n0 != n or o0 != o:
                raise ValueError("segments must be contiguous")
            if n1 < n0 or o1 < o0:
                raise ValueError("segment ends must not precede starts")
            if kind == CP and (n1 - n0 != o1 - o0 or n1 == n0):
                raise ValueError("CP segments need equal, non-zero lengths")
            if kind == ATOMIC and n1 == n0 and o1 == o0:
                raise ValueError("ATOMIC segments must not be empty")
            if kind == ATTACH and (n1 != n0 or o1 == o0):
                raise ValueError("ATTACH segments delete original text only")
            n, o = n1, o1
        if n != n_len or o != o_len:
            raise ValueError("segments must cover the full normalized and original ranges")

    @classmethod
    def identity(cls, length: int) -> "OffsetMap":
        if length < 0:
            raise ValueError("length must be >= 0")
        b = OffsetMapBuilder()
        b.cp(length)
        return b.build()

    # -- inspection ---------------------------------------------------------------------------------------------

    @property
    def segment_count(self) -> int:
        return len(self._s) // _STRIDE

    @property
    def nbytes(self) -> int:
        return len(self._s) * self._s.itemsize

    @property
    def is_identity(self) -> bool:
        return self.n_len == self.o_len and len(self._s) <= _STRIDE and all(k == CP for k in self._s[4::5])

    def segments(self) -> Iterator[tuple[int, int, int, int, int]]:
        s = self._s
        for i in range(0, len(s), _STRIDE):
            yield tuple(s[i : i + _STRIDE])  # type: ignore[misc]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, OffsetMap):
            return NotImplemented
        return self.n_len == other.n_len and self.o_len == other.o_len and self._s == other._s

    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        return f"OffsetMap(n_len={self.n_len}, o_len={self.o_len}, segments={self.segment_count})"

    # -- JSON ---------------------------------------------------------------------------------------------------

    def to_json(self) -> list[list[int]]:
        s = self._s
        return [list(s[i : i + _STRIDE]) for i in range(0, len(s), _STRIDE)]

    @classmethod
    def from_json(cls, data: list[list[int]]) -> "OffsetMap":
        flat = array("q")
        for row in data:
            if len(row) != _STRIDE or not all(type(v) is int for v in row):
                raise ValueError("each segment must be [n0, n1, o0, o1, kind] with integers")
            flat.extend(row)
        n_len = flat[-4] if flat else 0
        o_len = flat[-2] if flat else 0
        return cls(flat, n_len, o_len)

    # -- lookups ------------------------------------------------------------------------------------------------

    def _find(self, col: int, pos: int) -> int:
        """Index of the last segment whose start in column `col` is <= pos."""
        s = self._s
        lo, hi = 0, len(s) // _STRIDE
        while lo < hi:
            mid = (lo + hi) // 2
            if s[mid * _STRIDE + col] <= pos:
                lo = mid + 1
            else:
                hi = mid
        return lo - 1

    def to_original(self, start: int, end: int) -> MappedRange:
        """Map a half-open range of normalized code points to original code points."""
        return self._map(start, end, True)

    def to_normalized(self, start: int, end: int) -> MappedRange:
        """Map a half-open range of original code points to normalized code points."""
        return self._map(start, end, False)

    def _map(self, start: int, end: int, forward: bool) -> MappedRange:
        f0, f1, t0, t1 = (0, 1, 2, 3) if forward else (2, 3, 0, 1)
        from_len, to_len = (self.n_len, self.o_len) if forward else (self.o_len, self.n_len)
        if not 0 <= start <= end <= from_len:
            raise ValueError("range out of bounds")
        s = self._s
        if start == end:
            if start == from_len:
                return MappedRange(to_len, to_len, True)
            b = self._find(f0, start) * _STRIDE
            if s[b + 4] == CP:
                pos = s[b + t0] + start - s[b + f0]
                return MappedRange(pos, pos, True)
            if start == s[b + f0]:
                return MappedRange(s[b + t0], s[b + t0], True)
            return MappedRange(s[b + t0], s[b + t1], False)
        i = self._find(f0, start)
        j = self._find(f0, end - 1)
        bi, bj = i * _STRIDE, j * _STRIDE
        exact = True
        if s[bi + 4] == CP:
            t_start = s[bi + t0] + start - s[bi + f0]
        else:
            t_start = s[bi + t0]
            exact = start == s[bi + f0]
        if s[bj + 4] == CP:
            t_end = s[bj + t0] + end - s[bj + f0]
        else:
            t_end = s[bj + t1]
            if end < s[bj + f1]:
                exact = False
        if forward and end == s[bj + f1]:
            k = (j + 1) * _STRIDE
            while k < len(s) and s[k + 4] == ATTACH:
                t_end = s[k + 3]
                k += _STRIDE
        if exact:
            for idx in range(i * _STRIDE, (j + 1) * _STRIDE, _STRIDE):
                if s[idx + 4] != CP and s[idx + f0] < s[idx + f1] and s[idx + t0] == s[idx + t1]:
                    exact = False
                    break
        return MappedRange(t_start, t_end, exact)

    # -- composition --------------------------------------------------------------------------------------------

    def compose(self, other: "OffsetMap") -> "OffsetMap":
        """`self` maps X->Y and `other` maps Y->Z; the result maps X->Z.

        Zero-width segments at a cut are emitted with `other`'s deletions first, then `self`'s insertions, so an ATTACH
        deletion stays directly behind the character it belongs to. Maps with insertions (no normalization step
        makes any) compose into valid maps, but an insertion that sits exactly at a group boundary may land inside
        or outside the merged atomic segment depending on association, so associativity is only guaranteed without
        insertions.
        """
        if self.o_len != other.n_len:
            raise ValueError("cannot compose: original length of self must equal normalized length of other")
        a, b = self._s, other._s
        na, nb = len(a) // _STRIDE * _STRIDE, len(b) // _STRIDE * _STRIDE
        out = OffsetMapBuilder()
        y_len = self.o_len
        y = i = j = 0  # i, j are element offsets (index * 5)
        while True:
            while j < nb and b[j] == b[j + 1]:
                out.add(0, b[j + 3] - b[j + 2], b[j + 4])
                j += _STRIDE
            while i < na and a[i + 2] == a[i + 3]:
                out.add(a[i + 1] - a[i], 0, ATOMIC)
                i += _STRIDE
            if y >= y_len:
                break
            ai, bj = i, j
            a_kind, b_kind = a[i + 4], b[j + 4]
            if a_kind == CP and b_kind == CP:
                g = min(a[i + 3], b[j + 1])
            else:
                g = y
                if a_kind != CP:
                    g = max(g, a[i + 3])
                if b_kind != CP:
                    g = max(g, b[j + 1])
                while True:
                    while a[ai + 3] < g:
                        ai += _STRIDE
                    while b[bj + 1] < g:
                        bj += _STRIDE
                    changed = False
                    if a[ai + 3] > g and a[ai + 4] != CP:
                        g = a[ai + 3]
                        changed = True
                    if b[bj + 1] > g and b[bj + 4] != CP:
                        g = b[bj + 1]
                        changed = True
                    if not changed:
                        break
            x_end = a[ai + 1] if a[ai + 3] == g else a[ai] + (g - a[ai + 2])
            z_end = b[bj + 3] if b[bj + 1] == g else b[bj + 2] + (g - b[bj])
            x_len, z_len = x_end - out.n, z_end - out.o
            if a_kind == CP and b_kind == CP:
                out.add(x_len, z_len, CP)
            elif a_kind == ATTACH and ai == i and bj == j and b_kind == CP and a[ai + 3] == g:
                out.add(0, z_len, ATTACH)
            else:
                out.add(x_len, z_len, ATOMIC)
            i = ai + _STRIDE if a[ai + 3] == g else ai
            j = bj + _STRIDE if b[bj + 1] == g else bj
            y = g
        return out.build()
