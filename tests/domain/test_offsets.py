import random
import time
import tracemalloc
import unicodedata
from array import array
from pathlib import Path

import pytest
from rag_foundry_domain.offsets import ATOMIC, ATTACH, CP, MappedRange, OffsetMap, OffsetMapBuilder
from rag_foundry_ingestion.normalize import CANONICAL_V1, SEARCH_AR_EN_V1, apply

FIXTURES = Path(__file__).resolve().parents[2] / "packages" / "contracts" / "fixtures" / "sources"


def build(*ops: tuple) -> OffsetMap:
    b = OffsetMapBuilder()
    for op in ops:
        getattr(b, op[0])(*op[1:])
    return b.build()


def arr(*rows: tuple[int, int, int, int, int]) -> array:
    a = array("q")
    for r in rows:
        a.extend(r)
    return a


# -- validation -------------------------------------------------------------------------------------------------


def test_valid_map_and_empty_map():
    m = OffsetMap(arr((0, 2, 0, 2, CP), (2, 3, 2, 5, ATOMIC)), 3, 5)
    assert (m.n_len, m.o_len, m.segment_count) == (3, 5, 2)
    empty = OffsetMap(array("q"), 0, 0)
    assert empty.to_original(0, 0) == MappedRange(0, 0, True)


@pytest.mark.parametrize(
    "rows",
    [
        [(0, 2, 0, 2, CP), (3, 4, 2, 3, CP)],  # gap in n
        [(0, 2, 0, 2, CP), (2, 4, 3, 5, CP)],  # gap in o
        [(0, 2, 0, 3, CP)],  # CP lengths differ
        [(0, 0, 0, 0, CP)],  # empty CP
        [(0, 0, 0, 0, ATOMIC)],  # empty ATOMIC
        [(0, 1, 0, 1, ATTACH)],  # ATTACH must delete original text only
        [(0, 0, 0, 0, ATTACH)],
        [(0, 1, 0, 1, 9)],  # unknown kind
        [(1, 0, 0, 1, ATOMIC)],  # end before start
    ],
)
def test_invalid_segments_are_rejected(rows):
    with pytest.raises(ValueError):
        OffsetMap(arr(*rows), 10, 10)


def test_coverage_and_array_shape_are_validated():
    with pytest.raises(ValueError, match="cover"):
        OffsetMap(arr((0, 2, 0, 2, CP)), 3, 2)
    with pytest.raises(ValueError, match="cover"):
        OffsetMap(arr((0, 2, 0, 2, CP)), 2, 3)
    with pytest.raises(ValueError, match="cover"):
        OffsetMap(array("q"), 1, 0)
    with pytest.raises(ValueError, match="multiple of 5"):
        OffsetMap(array("q", [0, 1, 0, 1]), 1, 1)
    with pytest.raises(ValueError, match="array"):
        OffsetMap(array("i", [0, 1, 0, 1, 0]), 1, 1)


def test_builder_merges_cp_and_attach_runs_and_skips_empty():
    m = build(("cp", 2), ("cp", 3), ("atomic", 0, 0), ("attach", 1), ("attach", 2), ("cp", 1))
    assert m.to_json() == [[0, 5, 0, 5, CP], [5, 5, 5, 8, ATTACH], [5, 6, 8, 9, CP]]


def test_identity_and_bounds():
    m = OffsetMap.identity(10)
    assert m.is_identity
    assert m.to_original(2, 5) == MappedRange(2, 5, True)
    assert m.to_normalized(0, 10) == MappedRange(0, 10, True)
    assert m.to_original(10, 10) == MappedRange(10, 10, True)
    assert OffsetMap.identity(0).segment_count == 0
    for bad in [(-1, 2), (3, 11), (5, 4)]:
        with pytest.raises(ValueError):
            m.to_original(*bad)
    assert not build(("cp", 2), ("atomic", 1, 2)).is_identity


# -- kinds and edge rules ---------------------------------------------------------------------------------------


def test_cp_positions_map_one_to_one():
    m = build(("cp", 3), ("atomic", 1, 1), ("cp", 3))
    assert m.to_original(1, 3) == MappedRange(1, 3, True)
    assert m.to_original(5, 7) == MappedRange(5, 7, True)
    assert m.to_original(2, 2) == MappedRange(2, 2, True)


def test_atomic_widens_to_whole_segment():
    m = build(("cp", 1), ("atomic", 2, 2), ("cp", 1))  # reordering of two code points
    assert m.to_original(1, 3) == MappedRange(1, 3, True)  # whole segment
    assert m.to_original(2, 3) == MappedRange(1, 3, False)  # start inside
    assert m.to_original(1, 2) == MappedRange(1, 3, False)  # end inside
    assert m.to_original(0, 2) == MappedRange(0, 3, False)
    assert m.to_original(2, 2) == MappedRange(1, 3, False)  # empty range inside
    assert m.to_original(1, 1) == MappedRange(1, 1, True)  # empty range at a boundary
    assert m.to_normalized(2, 3) == MappedRange(1, 3, False)
    wide = build(("cp", 2), ("atomic", 1, 3), ("cp", 2))
    assert wide.to_original(2, 3) == MappedRange(2, 5, True)
    assert wide.to_normalized(2, 5) == MappedRange(2, 3, True)
    assert wide.to_normalized(3, 4) == MappedRange(2, 3, False)


def test_attach_extends_end_only_over_directly_following_attach():
    # n: a b c ; o: a b [m1 m2] c [ws] [m3]
    m = build(("cp", 2), ("attach", 2), ("cp", 1), ("atomic", 0, 1), ("attach", 1))
    assert m.n_len == 3 and m.o_len == 7
    assert m.to_original(0, 2) == MappedRange(0, 4, True)  # end extends over both marks
    assert m.to_original(1, 2) == MappedRange(1, 4, True)  # start never extends backwards
    assert m.to_original(0, 1) == MappedRange(0, 1, True)  # end inside a CP run: no extension
    assert m.to_original(2, 3) == MappedRange(4, 5, True)
    assert m.to_original(0, 3) == MappedRange(0, 5, True)  # detached deletion blocks the later mark
    assert m.to_original(3, 3) == MappedRange(7, 7, True)


def test_attach_after_atomic_extends_only_at_segment_end():
    m = build(("atomic", 2, 2), ("attach", 1), ("cp", 1))
    assert m.to_original(0, 2) == MappedRange(0, 3, True)
    assert m.to_original(0, 1) == MappedRange(0, 2, False)  # end inside the atomic segment: no extension


def test_insertions_make_ranges_inexact():
    m = build(("cp", 1), ("atomic", 2, 0), ("cp", 1))  # n: a X X b ; o: a b
    assert m.to_original(1, 3) == MappedRange(1, 1, False)  # fully inside the insertion: empty range at o_start
    assert m.to_original(1, 2) == MappedRange(1, 1, False)
    assert m.to_original(0, 2) == MappedRange(0, 1, False)
    assert m.to_original(0, 4) == MappedRange(0, 2, False)  # insertion strictly inside the range
    assert m.to_original(3, 4) == MappedRange(1, 2, True)
    assert m.to_original(0, 1) == MappedRange(0, 1, True)  # insertion outside the range
    assert m.to_normalized(0, 2) == MappedRange(0, 4, True)  # original text spans the inserted text
    assert m.to_normalized(1, 2) == MappedRange(3, 4, True)


def test_deletions_mirror_for_to_normalized():
    m = build(("cp", 1), ("atomic", 0, 2), ("cp", 1))  # o: a D D b ; n: a b
    assert m.to_normalized(1, 3) == MappedRange(1, 1, False)  # exactly the deleted run: empty, inexact
    assert m.to_normalized(1, 2) == MappedRange(1, 1, False)
    assert m.to_normalized(0, 4) == MappedRange(0, 2, False)  # deletion inside the range
    assert m.to_original(0, 2) == MappedRange(0, 4, True)
    assert m.to_original(0, 1) == MappedRange(0, 1, True)
    assert m.to_original(1, 2) == MappedRange(3, 4, True)


def test_attach_mirror_is_inexact_when_mark_is_covered():
    m = build(("cp", 1), ("attach", 1), ("cp", 1))  # o: a m b ; n: a b
    assert m.to_normalized(0, 3) == MappedRange(0, 2, False)
    assert m.to_normalized(2, 3) == MappedRange(1, 2, True)
    assert m.to_normalized(1, 2) == MappedRange(1, 1, False)


# -- composition ------------------------------------------------------------------------------------------------


def test_compose_requires_matching_lengths():
    with pytest.raises(ValueError, match="compose"):
        OffsetMap.identity(3).compose(OffsetMap.identity(4))


def test_compose_cp_runs_merge_and_split():
    a = build(("cp", 3), ("cp", 2))
    b = build(("cp", 2), ("atomic", 1, 1), ("cp", 2))
    c = a.compose(b)
    assert c.to_json() == [[0, 2, 0, 2, CP], [2, 3, 2, 3, ATOMIC], [3, 5, 3, 5, CP]]
    assert c.to_original(0, 5) == MappedRange(0, 5, True)
    assert c.to_original(2, 3) == MappedRange(2, 3, True)


def test_compose_merges_straddling_atomic_segments():
    a = build(("cp", 1), ("atomic", 1, 2))  # X2 -> Y3
    b = build(("atomic", 2, 3), ("cp", 1))  # Y3 -> Z4
    assert a.compose(b).to_json() == [[0, 2, 0, 4, ATOMIC]]


def test_compose_keeps_cp_around_atomic():
    a = build(("cp", 2), ("atomic", 1, 2), ("cp", 2))  # X5 -> Y6
    b = build(("cp", 6))
    assert a.compose(b) == a
    assert b.compose(OffsetMap.identity(6)) == b


def test_compose_attach_survives():
    ident3 = OffsetMap.identity(3)
    a = build(("cp", 1), ("attach", 2))  # X1 -> Y3
    assert a.compose(ident3).to_json() == [[0, 1, 0, 1, CP], [1, 1, 1, 3, ATTACH]]
    b = build(("cp", 1), ("attach", 2))  # Y1 -> Z3
    assert OffsetMap.identity(1).compose(b).to_json() == [[0, 1, 0, 1, CP], [1, 1, 1, 3, ATTACH]]
    # attach composed with an atomic covering it loses the ATTACH kind
    atomic = build(("atomic", 3, 5))
    assert a.compose(atomic).to_json() == [[0, 1, 0, 5, ATOMIC]]


def test_compose_end_to_end_chain():
    # Y = "a" + two marks + "b" (o), step one removes the marks, step two removes a leading ws then maps 1:1
    first = build(("cp", 1), ("attach", 2), ("cp", 1))  # Y2 -> Z4
    second = build(("atomic", 0, 1), ("cp", 1))  # X1 -> Y2? (delete first char, keep the second)
    c = second.compose(first)
    assert (c.n_len, c.o_len) == (1, 4)
    assert c.to_original(0, 1) == MappedRange(3, 4, True)


def random_map(rng: random.Random, o_target: int, insertions: bool = True) -> OffsetMap:
    b = OffsetMapBuilder()
    while b.o < o_target or (insertions and rng.random() < 0.2):
        rem = o_target - b.o
        choice = rng.random()
        if insertions and (rem == 0 or choice < 0.1):
            b.atomic(rng.randint(1, 3), 0)  # insertion
        elif choice < 0.55:
            b.cp(rng.randint(1, rem))
        elif choice < 0.75:
            b.attach(rng.randint(1, rem))
        else:
            b.atomic(rng.randint(0, 3), rng.randint(1, rem))
        if rem == 0 and rng.random() < 0.5:
            break
    return b.build()


def all_ranges(m: OffsetMap):
    out = []
    for s in range(m.n_len + 1):
        for e in range(s, m.n_len + 1):
            out.append(m.to_original(s, e))
    return out


@pytest.mark.parametrize("insertions", [False, True])
def test_compose_random_maps(insertions):
    """Associative without insertions (no normalization step inserts text); always valid with them."""
    rng = random.Random(20261105)
    for _ in range(400):
        m1 = random_map(rng, rng.randint(0, 14), insertions)
        m2 = random_map(rng, m1.n_len, insertions)
        m3 = random_map(rng, m2.n_len, insertions)
        left = m3.compose(m2).compose(m1)
        right = m3.compose(m2.compose(m1))
        for m in (left, right):
            assert OffsetMap.from_json(m.to_json()) == m  # composed maps stay valid
            assert (m.n_len, m.o_len) == (m3.n_len, m1.o_len)
        if not insertions:
            assert all_ranges(left) == all_ranges(right)
            assert left.to_normalized(0, left.o_len) == right.to_normalized(0, right.o_len)


def simple_map(rng: random.Random, o_target: int) -> OffsetMap:
    """Only CP and non-empty ATOMIC segments, so every range maps through a plain widening rule."""
    b = OffsetMapBuilder()
    while b.o < o_target:
        rem = o_target - b.o
        if rng.random() < 0.5:
            b.cp(rng.randint(1, rem))
        else:
            b.atomic(rng.randint(1, 3), rng.randint(1, rem))
    return b.build()


def reference_to_original(a: OffsetMap, b: OffsetMap, s: int, e: int) -> tuple[int, int]:
    """Brute force over single code points. Y units joined by a shared ATOMIC segment of `a` or `b` form one group."""
    y_len = a.o_len
    parent = list(range(y_len))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for seg in list(a.segments()):
        if seg[4] == ATOMIC:
            for y in range(seg[2] + 1, seg[3]):
                parent[find(y)] = find(seg[2])
    for seg in list(b.segments()):
        if seg[4] == ATOMIC:
            for y in range(seg[0] + 1, seg[1]):
                parent[find(y)] = find(seg[0])
    z_of: list[tuple[int, int]] = [(0, 0)] * y_len
    for n0, n1, o0, o1, kind in b.segments():
        for y in range(n0, n1):
            z_of[y] = (o0 + y - n0, o0 + y - n0 + 1) if kind == CP else (o0, o1)
    touched: set[int] = set()
    for n0, n1, o0, o1, kind in a.segments():
        for x in range(max(s, n0), min(e, n1)):
            ys = [o0 + x - n0] if kind == CP else list(range(o0, o1))
            touched.update(find(y) for y in ys)
    units = [y for y in range(y_len) if find(y) in touched]
    return min(z_of[y][0] for y in units), max(z_of[y][1] for y in units)


def test_compose_matches_brute_force_reference():
    rng = random.Random(99)
    for _ in range(300):
        b = simple_map(rng, rng.randint(1, 14))
        a = simple_map(rng, b.n_len)
        c = a.compose(b)
        assert (c.n_len, c.o_len) == (a.n_len, b.o_len)
        for s in range(a.n_len):
            for e in range(s + 1, a.n_len + 1):
                got = c.to_original(s, e)
                assert (got.start, got.end) == reference_to_original(a, b, s, e), (a.to_json(), b.to_json(), s, e)


def test_compose_identity_is_neutral_on_random_maps():
    rng = random.Random(7)
    for _ in range(100):
        m = random_map(rng, rng.randint(0, 12))
        assert m.compose(OffsetMap.identity(m.o_len)).to_json() == m.to_json()
        assert OffsetMap.identity(m.n_len).compose(m).to_json() == m.to_json()


# -- JSON -------------------------------------------------------------------------------------------------------


def test_json_round_trip_and_rejection():
    m = build(("cp", 2), ("atomic", 1, 3), ("attach", 1), ("atomic", 0, 2), ("cp", 1))
    data = m.to_json()
    assert data == [[0, 2, 0, 2, 0], [2, 3, 2, 5, 1], [3, 3, 5, 6, 2], [3, 3, 6, 8, 1], [3, 4, 8, 9, 0]]
    assert OffsetMap.from_json(data) == m
    assert OffsetMap.from_json([]) == OffsetMap.identity(0)
    assert m.nbytes == 8 * 5 * m.segment_count
    with pytest.raises(ValueError):
        OffsetMap.from_json([[0, 1, 0]])
    with pytest.raises(ValueError):
        OffsetMap.from_json([[0, 1, 0, 1, 1.0]])
    with pytest.raises(ValueError):
        OffsetMap.from_json([[0, 1, 0, 2, CP]])


# -- scale ------------------------------------------------------------------------------------------------------


def test_one_million_code_points_through_both_profiles(record_property):
    sample = "".join(
        (FIXTURES / n).read_bytes().decode("utf-8") for n in ("en_short.txt", "ar_short.txt", "mixed_short.txt")
    )
    sample += (
        "a\r\nb  \u00a0 c \u0643\u0640\u064e\u062a\u064e \ufb01 \u0663\u0660 \ufe8d\u0627\u0653 "
        "Stra\u00dfe \u200d\U0001f4d8 e\u0301 \u0651\u064e\n"
    )
    text = (sample * (1_000_000 // len(sample) + 1))[:1_000_000]
    assert len(text) == 1_000_000

    def run():
        canon = apply(CANONICAL_V1, text)
        search = apply(SEARCH_AR_EN_V1, canon.text)
        return canon, search, search.map.compose(canon.map)

    t0 = time.perf_counter()
    run()
    plain = time.perf_counter() - t0
    tracemalloc.start()
    t0 = time.perf_counter()
    canon, search, total = run()
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    segments = total.segment_count
    record_property("seconds_plain", round(plain, 2))
    record_property("seconds_tracemalloc", round(elapsed, 2))
    record_property("peak_mib", round(peak / 2**20, 1))
    record_property("composed_segments", segments)
    print(
        f"1M code points: {plain:.2f} s plain, {elapsed:.2f} s under tracemalloc, "
        f"peak {peak / 2**20:.1f} MiB, composed segments {segments}"
    )
    assert plain < 60
    assert elapsed < 60
    assert peak < 512 * 2**20
    assert total.nbytes == 40 * segments
    assert segments * 4 <= len(text)  # identity runs merge: far fewer segments than code points
    assert (total.n_len, total.o_len) == (len(search.text), len(text))
    assert canon.text == unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))
    # spot check: the last word maps back into the original text
    n = total.n_len
    r = total.to_original(n - 2, n)
    assert 0 <= r.start <= r.end <= len(text)
