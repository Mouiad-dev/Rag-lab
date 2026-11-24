from pathlib import Path

import pytest
from rag_foundry_domain.documents import MediaType
from rag_foundry_domain.errors import ExtractionError, InvalidTextEncodingError
from rag_foundry_domain.extraction import BlockKind
from rag_foundry_domain.offsets import OffsetMap
from rag_foundry_ingestion.extract import EXTRACTORS, ExtractionLimits, get_extractor
from rag_foundry_ingestion.extract.base import RawBlock, finish, register
from rag_foundry_ingestion.extract.markdown import EXTRACTOR as MD
from rag_foundry_ingestion.extract.plain import EXTRACTOR as TXT
from rag_foundry_ingestion.normalize import CANONICAL_V1, apply

FIXTURES = Path(__file__).resolve().parents[2] / "packages" / "contracts" / "fixtures" / "sources"
LIMITS = ExtractionLimits()


def run(ex, text: str | bytes, encoding: str | None = None, limits: ExtractionLimits = LIMITS):
    data = text.encode("utf-8") if isinstance(text, str) else text
    return ex.extract(data, text_encoding=encoding, limits=limits)


def texts(result) -> list[str]:
    return [result.canonical_text[b.start : b.end] for b in result.blocks]


def check_blocks(result, source: str) -> None:
    """Each block maps through source_map to a source span that re-normalizes to the block text."""
    assert result.source_map.n_len == len(result.canonical_text)
    assert result.source_map.o_len == len(source)
    prev_end = 0
    for b in result.blocks:
        assert b.start >= prev_end
        assert b.end > b.start
        prev_end = b.end
        r = result.source_map.to_original(b.start, b.end)
        assert apply(CANONICAL_V1, source[r.start : r.end]).text == result.canonical_text[b.start : b.end]


def kinds(result) -> list[str]:
    return [b.kind.value for b in result.blocks]


# -- registry ----------------------------------------------------------------------------------------------------


def test_registry_and_unsupported():
    assert get_extractor(MediaType.TEXT_PLAIN) is TXT
    assert get_extractor("text/markdown") is MD
    assert set(EXTRACTORS) >= {MediaType.TEXT_PLAIN, MediaType.TEXT_MARKDOWN}
    with pytest.raises(ExtractionError) as e:
        get_extractor(MediaType.APPLICATION_PDF if MediaType.APPLICATION_PDF not in EXTRACTORS else "image/png")
    assert e.value.code == "unsupported_extractor"


# -- txt ---------------------------------------------------------------------------------------------------------


def test_txt_paragraphs_and_metadata():
    src = "one\ntwo\n\nthree\n"
    r = run(TXT, src)
    assert texts(r) == ["one\ntwo", "three"]
    assert kinds(r) == ["paragraph", "paragraph"]
    assert [b.block_id for b in r.blocks] == ["b00000", "b00001"]
    assert (r.extractor, r.extractor_version, r.profile_id) == ("txt", "1", "canonical-v1")
    assert r.source_space == "decoded_text" and r.page_starts is None and r.warnings == ()
    assert r.raw_map.is_identity
    check_blocks(r, src)


def test_txt_whitespace_only_lines_are_blank():
    src = "a\n \t \nb\n\u00a0\nc"
    r = run(TXT, src)
    assert texts(r) == ["a", "b", "c"]
    check_blocks(r, src)


def test_txt_crlf_and_cr():
    src = "one\r\ntwo\r\n\r\nthree\rfour\r"
    r = run(TXT, src)
    assert r.canonical_text == "one\ntwo\n\nthree\nfour\n"
    assert texts(r) == ["one\ntwo", "three\nfour"]
    check_blocks(r, src)
    first = r.source_map.to_original(r.blocks[0].start, r.blocks[0].end)
    assert src[first.start : first.end] == "one\r\ntwo"


def test_txt_bom_utf8_sig_offsets_start_after_bom():
    src = "abc\n\ndef"
    r = run(TXT, b"\xef\xbb\xbf" + src.encode(), "utf-8-sig")
    assert r.canonical_text == src
    assert r.blocks[0].start == 0
    check_blocks(r, src)
    # plain utf-8 keeps the BOM as a character
    r2 = run(TXT, b"\xef\xbb\xbf" + src.encode(), "utf-8")
    assert r2.canonical_text.startswith("\ufeff")


def test_txt_nfc_and_mixed_arabic_fixture():
    src = "e\u0301cole\n\n\u0645\u0631\u062d\u0628\u0627 world"
    r = run(TXT, src)
    assert texts(r) == ["\u00e9cole", "\u0645\u0631\u062d\u0628\u0627 world"]
    check_blocks(r, src)
    fixture = (FIXTURES / "mixed_short.txt").read_bytes().decode("utf-8")
    rf = run(TXT, fixture)
    assert len(rf.blocks) == 2
    check_blocks(rf, fixture)


def test_bad_encoding_and_unknown_encoding():
    with pytest.raises(InvalidTextEncodingError):
        run(TXT, b"\xff\xfe\xfa", "utf-8")
    with pytest.raises(InvalidTextEncodingError):
        run(TXT, "abc", "no-such-codec")


def test_max_chars_checked_on_raw_text():
    lim = ExtractionLimits(max_chars=5)
    assert run(TXT, "12345", limits=lim).canonical_text == "12345"
    with pytest.raises(ExtractionError) as e:
        run(TXT, "123456", limits=lim)
    assert e.value.code == "text_too_long"
    with pytest.raises(ExtractionError) as e2:
        run(MD, "123456", limits=lim)
    assert e2.value.code == "text_too_long"
    # raw length counts CRLF as two code points
    with pytest.raises(ExtractionError):
        run(TXT, "ab\r\ncd", limits=lim)


@pytest.mark.parametrize("ex", [TXT, MD])
@pytest.mark.parametrize("src", ["", "   \n\t\n", "\n\n\n"])
def test_empty_input_is_no_text_layer(ex, src):
    with pytest.raises(ExtractionError) as e:
        run(ex, src)
    assert e.value.code == "no_text_layer"


@pytest.mark.parametrize("ex", [TXT, MD])
def test_deterministic(ex):
    src = "# T\n\nbody\r\ntext\n\n```\ncode\n```\n"
    a, b = run(ex, src), run(ex, src)
    assert a == b


# -- markdown ----------------------------------------------------------------------------------------------------


def test_md_headings_paths_and_blocks():
    src = "# A\n\npara a\n\n## B\ntext b\n### C\n\n## D\n\n# E\n"
    r = run(MD, src)
    assert texts(r) == ["# A", "para a", "## B", "text b", "### C", "## D", "# E"]
    assert kinds(r) == ["heading", "paragraph", "heading", "paragraph", "heading", "heading", "heading"]
    assert [b.heading_level for b in r.blocks] == [1, None, 2, None, 3, 2, 1]
    assert [b.section_path for b in r.blocks] == [
        ("A",),
        ("A",),
        ("A", "B"),
        ("A", "B"),
        ("A", "B", "C"),
        ("A", "D"),
        ("E",),
    ]
    assert (r.extractor, r.extractor_version) == ("markdown", "1")
    check_blocks(r, src)


def test_md_arabic_and_mixed_headings():
    src = "# \u0645\u0631\u062d\u0628\u0627\n\n## Intro \u0645\u0642\u062f\u0645\u0629 ##\n\n\u0646\u0635\n"
    r = run(MD, src)
    assert r.blocks[0].section_path == ("\u0645\u0631\u062d\u0628\u0627",)
    assert r.blocks[2].section_path == ("\u0645\u0631\u062d\u0628\u0627", "Intro \u0645\u0642\u062f\u0645\u0629")
    assert texts(r)[2] == "\u0646\u0635"
    check_blocks(r, src)


@pytest.mark.parametrize(
    ("line", "heading", "title"),
    [
        ("#5", False, None),
        ("####### x", False, None),
        ("###### six", True, "six"),
        ("# foo #", True, "foo"),
        ("# foo ###   ", True, "foo"),
        ("# foo#", True, "foo#"),
        ("# foo \\#", True, "foo \\#"),
        ("   # three", True, "three"),
        ("    # four spaces", False, None),
        ("\t# tab", False, None),
        ("#", True, ""),
        ("# #", True, ""),
        ("#\ttab title", True, "tab title"),
    ],
)
def test_md_atx_edge_cases(line, heading, title):
    src = f"{line}\n"
    r = run(MD, src)
    assert len(r.blocks) == 1
    b = r.blocks[0]
    if heading:
        assert b.kind is BlockKind.HEADING
        assert b.section_path == (title,)
    else:
        assert b.kind is BlockKind.PARAGRAPH
        assert b.section_path == ()
    check_blocks(r, src)


def test_md_heading_interrupts_paragraph():
    r = run(MD, "text\n# H\nmore\n")
    assert texts(r) == ["text", "# H", "more"]
    assert r.blocks[2].section_path == ("H",)


def test_md_fence_closed_and_blank_lines_inside():
    src = "intro\n\n```py\nx = 1\n\n# not heading\n```\n\nafter\n"
    r = run(MD, src)
    assert kinds(r) == ["paragraph", "code", "paragraph"]
    assert texts(r)[1] == "```py\nx = 1\n\n# not heading\n```"
    check_blocks(r, src)


def test_md_unclosed_fence_runs_to_eof():
    src = "# T\n\n```\ncode\n# x\n\n\n"
    r = run(MD, src)
    assert kinds(r) == ["heading", "code"]
    assert texts(r)[1] == "```\ncode\n# x"
    assert r.blocks[1].section_path == ("T",)
    check_blocks(r, src)


def test_md_tilde_fence_closed_only_by_tildes():
    src = "~~~\na\n```\nb\n~~\nc\n~~~~\n\nend"
    r = run(MD, src)
    assert kinds(r) == ["code", "paragraph"]
    assert texts(r)[0] == "~~~\na\n```\nb\n~~\nc\n~~~~"
    check_blocks(r, src)


def test_md_backtick_fence_not_closed_by_tilde_or_shorter_fence():
    src = "````\na\n```\n~~~~\nb\n````\n"
    r = run(MD, src)
    assert kinds(r) == ["code"]
    assert texts(r)[0] == src.rstrip("\n")


def test_md_fence_indent_limit_and_closing_with_text():
    r = run(MD, "    ```\nx\n")  # 4 spaces: not a fence
    assert kinds(r) == ["paragraph"]
    r2 = run(MD, "```\na\n``` not closing\nb\n```\n")
    assert texts(r2) == ["```\na\n``` not closing\nb\n```"]


def test_md_crlf_bom_and_markup_verbatim():
    src = "# T\r\n\r\n*em* [l](u)\r\n- item\r\n"
    r = run(MD, b"\xef\xbb\xbf" + src.encode(), "utf-8-sig")
    assert texts(r) == ["# T", "*em* [l](u)\n- item"]
    check_blocks(r, src)


def test_md_nfc_inside_block_maps_back():
    src = "# cafe\u0301\n\ncafe\u0301 au lait\n"
    r = run(MD, src)
    assert texts(r) == ["# caf\u00e9", "caf\u00e9 au lait"]
    assert r.blocks[0].section_path == ("caf\u00e9",)  # canonical (NFC) titles
    check_blocks(r, src)


# -- review round 1: shared helper hardening ---------------------------------------------------------------------


def _finish(raw: str, blocks: list[RawBlock], page_starts=None):
    return finish(
        raw,
        OffsetMap.identity(len(raw)),
        blocks,
        extractor="t",
        extractor_version="1",
        source_space="decoded_text",
        page_starts=page_starts,
    )


def test_register_rejects_different_extractor_for_same_media_type():
    class Other:
        name = "other"
        version = "1"
        media_types = (MediaType.TEXT_PLAIN,)

        def extract(self, data, *, text_encoding, limits):
            raise AssertionError

    with pytest.raises(ValueError):
        register(Other())
    register(TXT)  # same object again is fine
    assert get_extractor(MediaType.TEXT_PLAIN) is TXT


@pytest.mark.parametrize(
    "blocks",
    [
        [RawBlock(BlockKind.PARAGRAPH, 2, 4), RawBlock(BlockKind.PARAGRAPH, 0, 1)],  # unordered
        [RawBlock(BlockKind.PARAGRAPH, 0, 3), RawBlock(BlockKind.PARAGRAPH, 2, 5)],  # overlap
        [RawBlock(BlockKind.PARAGRAPH, 0, 99)],  # out of range
        [RawBlock(BlockKind.PARAGRAPH, 1, 1)],  # empty
    ],
)
def test_finish_rejects_bad_raw_blocks(blocks):
    with pytest.raises(ExtractionError) as e:
        _finish("abcdef", blocks)
    assert e.value.code == "integrity_mismatch"


def test_finish_section_path_is_canonical():
    r = _finish("e\u0301", [RawBlock(BlockKind.PARAGRAPH, 0, 2, section_path=("e\u0301",))])
    assert r.blocks[0].section_path == ("\u00e9",)


def test_finish_page_starts_checks():
    raw = "ab\r\n\r\ncd"  # page boundary next to CRLF
    r = _finish(raw, [RawBlock(BlockKind.PAGE, 0, 2), RawBlock(BlockKind.PAGE, 6, 8)], page_starts=[0, 6])
    assert r.page_starts == (0, 4)
    raw2 = "cafe\u0301\n\nb"  # boundary after a composed mark
    r2 = _finish(raw2, [RawBlock(BlockKind.PAGE, 0, 5), RawBlock(BlockKind.PAGE, 7, 8)], page_starts=[0, 7])
    assert r2.page_starts == (0, 6)
    with pytest.raises(ExtractionError) as e:  # boundary inside "e" + combining mark collapses onto page 0
        _finish("e\u0301", [RawBlock(BlockKind.PAGE, 0, 2)], page_starts=[0, 1])
    assert e.value.code == "integrity_mismatch"
    for bad in ([0, 99], [3, 1], [-1]):
        with pytest.raises(ExtractionError):
            _finish("abcdef", [RawBlock(BlockKind.PAGE, 0, 6)], page_starts=bad)


def test_md_backtick_fence_with_backtick_in_info_string_is_not_a_fence():
    r = run(MD, "``` a`b\ntext\n```\n")
    assert kinds(r)[0] == "paragraph"
    assert texts(r)[0].startswith("``` a`b")
    r2 = run(MD, "~~~ a`b\ncode\n~~~\n")  # tilde info strings may contain backticks
    assert kinds(r2) == ["code"]
