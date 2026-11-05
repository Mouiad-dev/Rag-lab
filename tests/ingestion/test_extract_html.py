import html as stdlib_html
import time

import pytest
from rag_foundry_domain.documents import MediaType
from rag_foundry_domain.errors import ExtractionAlignmentError, ExtractionError
from rag_foundry_domain.extraction import BlockKind
from rag_foundry_domain.offsets import ATOMIC, CP
from rag_foundry_ingestion.extract import ExtractionLimits, get_extractor
from rag_foundry_ingestion.extract import html as html_mod

LIMITS = ExtractionLimits()
SRC_WS = " \t\n\r\f\x00"


@pytest.fixture
def capture(monkeypatch):
    """Run the extractor and also hand back the raw text that `finish` received."""
    seen: dict[str, str] = {}
    real = html_mod.finish

    def spy(raw, *args, **kwargs):
        seen["raw"] = raw
        return real(raw, *args, **kwargs)

    monkeypatch.setattr(html_mod, "finish", spy)

    def run(src: str, skipped: tuple[str, ...] = ()):
        """`skipped` lists the text of head/script/style/noscript nodes (aligned, then dropped from the raw text)."""
        result = html_mod.EXTRACTOR.extract(src.encode("utf-8"), text_encoding=None, limits=LIMITS)
        check_segments(src, seen["raw"], result.raw_map, skipped)
        return result

    return run


def check_segments(src: str, raw: str, raw_map, skipped: tuple[str, ...] = ()) -> None:
    assert raw_map.o_len == len(src) and raw_map.n_len == len(raw)
    for n0, n1, o0, o1, kind in raw_map.segments():
        r, s = raw[n0:n1], src[o0:o1]
        if kind == CP:
            assert r == s
        elif kind == ATOMIC and n0 == n1:  # deletion: markup or dropped whitespace/NUL
            assert s.startswith("<") or all(c in SRC_WS for c in s) or any(s in k for k in skipped), s
        elif kind == ATOMIC and o0 == o1:  # insertion: block separator
            assert r == "\n\n"
        elif kind == ATOMIC:
            if all(c in " \t\n\f" for c in r) and all(c in SRC_WS for c in s):
                assert r == " " or (r == "\n" and s in ("\r\n", "\r"))  # whitespace on both sides
            elif r == "\n" and s in ("\r\n", "\r"):
                pass
            elif r == "\ufffd" and s == "\x00":
                pass
            else:
                assert r == stdlib_html.unescape(s), (r, s)  # entity
        else:
            pytest.fail("unexpected segment kind")


def src_slice(src: str, result, start: int, end: int) -> str:
    mapped = result.source_map.to_original(start, end)
    assert mapped.exact
    return src[mapped.start : mapped.end]


def block_texts(result) -> list[str]:
    return [result.canonical_text[b.start : b.end] for b in result.blocks]


def test_registered_and_versioned():
    ex = get_extractor(MediaType.TEXT_HTML)
    assert ex is html_mod.EXTRACTOR
    assert ex.name == "html"
    assert ex.version.startswith("1+selectolax==1.0.1+py3.")


@pytest.mark.parametrize(
    ("src", "text", "slice_text"),
    [
        ('<a title="hello">hello</a>', "hello", "hello"),
        ("<!--hello-->hello", "hello", "hello"),
        ("<title>hi</title><p>hi", "hi", "hi"),
        ("<p title='a>b'>b</p>", "b", "b"),
        ("<script>x</script>x", "x", "x"),
        ("a < b", "a < b", "a < b"),
        ("a<3", "a<3", "a<3"),
        # the parser drops the first LF; the text aligns to the first source LF, the second LF is a dropped one
        ("<pre>\n\nx", "\nx", "\n\nx"),
        ("<textarea>\nx", "x", "x"),
    ],
)
def test_exact_source_offsets(capture, src, text, slice_text):
    r = capture(src, skipped=("hi", "x", "T"))
    assert r.canonical_text == text
    assert src_slice(src, r, 0, len(text)) == slice_text


def test_exact_offset_values():
    src = '<a title="hello">hello</a>'
    r = html_mod.EXTRACTOR.extract(src.encode(), text_encoding=None, limits=LIMITS)
    m = r.source_map.to_original(0, 5)
    assert (m.start, m.end, m.exact) == (17, 22, True)
    src = "<p title='a>b'>b</p>"
    r = html_mod.EXTRACTOR.extract(src.encode(), text_encoding=None, limits=LIMITS)
    m = r.source_map.to_original(0, 1)
    assert (m.start, m.end) == (15, 16)
    src = "<!--hello-->hello"
    r = html_mod.EXTRACTOR.extract(src.encode(), text_encoding=None, limits=LIMITS)
    assert (r.source_map.to_original(0, 5).start, r.source_map.to_original(0, 5).end) == (12, 17)


def test_comment_forms_and_bogus_markup(capture):
    for src in ("<!-->a", "<!--->a", "<!-- x --!>a", "<!doctype html>a", "<?php x ?>a", "</3>a", "<![CDATA[q]]>a"):
        r = capture(src)
        assert r.canonical_text == "a", src
        assert src_slice(src, r, 0, 1) == "a"


def test_tag_with_quoted_gt_and_unquoted_values(capture):
    for src in ('<a title="a>b">x</a>', "<a title='a>b'>x</a>", '<p class=a"b>x</p>', "<br/>x"):
        r = capture(src)
        assert r.canonical_text == "x", src
        assert src_slice(src, r, 0, 1) == "x"


def test_script_with_markup_text_and_style(capture):
    src = "<style>p > a { }</style><script>if (a<b) { x('</p>') }</script><p>shown"
    r = capture(src, skipped=("p > a { }", "if (a<b) { x('</p>') }"))
    assert r.canonical_text == "shown"
    assert src_slice(src, r, 0, 5) == "shown"


def test_noscript_and_template_content_is_not_extracted(capture):
    r = capture("<noscript>hidden</noscript><p>shown", skipped=("hidden",))
    assert r.canonical_text == "shown"
    with pytest.raises(ExtractionAlignmentError):  # template content is a known limit
        capture("<template><p>t</p></template><p>shown")


def test_cdata_inside_svg_fails_explicitly(capture):
    with pytest.raises(ExtractionAlignmentError) as e:
        capture("<p>a</p><svg><![CDATA[x]]></svg>")
    assert e.value.code == "alignment_failed"
    r = capture("<p>a</p><svg><path d='M0 0'/> </svg><p>b</p>")  # no text in svg: aligns
    assert block_texts(r) == ["a", "b"]


ENTITIES = ["&amp;", "&#x627;", "&nbsp;", "&#0;", "&#xD800;", "&#x110000;", "&#x80;", "&ampx", "&notit;"]


@pytest.mark.parametrize("entity", ENTITIES)
def test_entities_align_atomically(capture, entity):
    src = f"<p>a{entity}.</p>"  # '.' ends the entity name (&ampxb would be one entity)
    r = capture(src)
    expected = "a" + stdlib_html.unescape(entity) + "."
    assert r.canonical_text == expected
    inner = expected[1:-1]
    assert src_slice(src, r, 1, 1 + len(inner)) == entity
    assert src_slice(src, r, 0, len(expected)) == f"a{entity}."


def test_raw_text_elements_keep_entities_literal(capture):
    # xmp keeps "&amp;" literal; the template text is not emitted by the parser. Never map "&amp;" to 5..30.
    with pytest.raises(ExtractionAlignmentError):
        capture("<xmp>&amp;</xmp><template>amp;</template>")
    src = "<xmp>&amp;</xmp><p>z"
    r = capture(src)
    assert block_texts(r) == ["&amp;", "z"]
    assert (r.source_map.to_original(0, 5).start, r.source_map.to_original(0, 5).end) == (5, 10)
    src = "<iframe>&amp;</iframe>x"
    r = capture(src)
    assert r.canonical_text == "&amp;x"
    assert (r.source_map.to_original(0, 5).start, r.source_map.to_original(0, 5).end) == (8, 13)
    assert src_slice(src, r, 5, 6) == "x"
    src = "<noframes>&lt;</noframes><p>q"
    assert capture(src, skipped=("&lt;",)).canonical_text == "q"  # in head: aligned literally, not emitted
    assert capture("<textarea>&amp;</textarea>").canonical_text == "&"  # RCDATA decodes entities


def test_entity_text_that_looks_like_an_entity(capture):
    src = "<p>&amp;amp; &lt;b&gt;</p>"
    r = capture(src)
    assert r.canonical_text == "&amp; <b>"
    assert src_slice(src, r, 0, 5) == "&amp;amp;"


def test_nul_in_text_is_a_detached_deletion(capture):
    src = "<p>a\x00b</p>"
    r = capture(src)
    assert r.canonical_text == "ab"
    assert any(n0 == n1 and src[o0:o1] == "\x00" for n0, n1, o0, o1, _ in r.raw_map.segments())
    assert src_slice(src, r, 0, 1) == "a"


def test_cr_and_crlf(capture):
    src = "<pre>a\r\nb\rc\nd</pre>"
    r = capture(src)
    assert r.canonical_text == "a\nb\nc\nd"
    assert src_slice(src, r, 1, 2) == "\r\n"
    assert src_slice(src, r, 3, 4) == "\r"
    r = capture("<p>a\r\n\r\nb</p>")
    assert r.canonical_text == "a b"


def test_whitespace_collapses_and_blocks_are_separated(capture):
    src = "<div>\n  <p>  one \t two  </p>\n\n  <p>three</p>  </div>"
    r = capture(src)
    assert block_texts(r) == ["one two", "three"]
    assert "\n\n" in r.canonical_text
    assert src_slice(src, r, r.blocks[1].start, r.blocks[1].end) == "three"


def test_br_li_and_pre_are_blocks(capture):
    r = capture("<ul><li>a<li>b</ul><p>x<br>y<pre>  p\n  q</pre>")
    assert block_texts(r) == ["a", "b", "x", "y", "  p\n  q"]
    assert [b.kind for b in r.blocks][-1] is BlockKind.CODE


def test_headings_arabic_and_mixed(capture):
    src = (
        "<h1>\u0645\u0631\u062d\u0628\u0627 abc</h1><p>\u0646\u0635 \u0639\u0631\u0628\u064a</p>"
        "<h2>Sub \u0642\u0633\u0645</h2><p>x</p><h1>Next</h1><p>y"
    )
    r = capture(src)
    head = [(b.kind, b.heading_level, b.section_path) for b in r.blocks]
    assert head[0] == (BlockKind.HEADING, 1, ("\u0645\u0631\u062d\u0628\u0627 abc",))
    assert head[1] == (BlockKind.PARAGRAPH, None, ("\u0645\u0631\u062d\u0628\u0627 abc",))
    assert head[2][2] == ("\u0645\u0631\u062d\u0628\u0627 abc", "Sub \u0642\u0633\u0645")
    assert head[3][2] == head[2][2]
    assert head[4] == (BlockKind.HEADING, 1, ("Next",))
    assert head[5][2] == ("Next",)
    assert src_slice(src, r, r.blocks[1].start, r.blocks[1].end) == "\u0646\u0635 \u0639\u0631\u0628\u064a"


def test_foster_parenting_fails_explicitly(capture):
    with pytest.raises(ExtractionAlignmentError) as e:
        capture("<table><tr><td>a</td></tr>x</table>")
    assert e.value.code == "alignment_failed"
    r = capture("<table><tr><td>a</td><td>b</td></tr></table>")  # in source order: aligns
    assert block_texts(r) == ["a", "b"]
    # any foster-parented text fails: equal texts would otherwise swap offsets silently
    for src in (
        "<table><tr><td>a</td></tr>a</table><p>b",
        "<table>x<tr><td>y</td></tr></table>",
        "<table>x<tr><td>y</td></tr>y</table>",
        "<table>x <tr><td>y</td></tr>y</table>",
        "<table>&amp;<tr><td>&amp;</td></tr>&amp;</table>",
    ):
        with pytest.raises(ExtractionAlignmentError) as e:
            capture(src)
        assert e.value.code == "alignment_failed", src


def test_normal_tables_still_align(capture):
    cases = {
        "<table class='<table'><tr><td>a</td></tr></table>": ["a"],
        "<table title='<table>'><tr><td>a</td></tr></table>": ["a"],
        "<table><caption>cap</caption><tr><th>h</th><td>d</td></tr></table>": ["cap", "h", "d"],
        "<table><tr><td>a<table><tr><td>b</td></tr></table></td></tr></table><p>c": ["a", "b", "c"],
        "<table>\n  <tr>\n    <td>a</td>\n    <td>b</td>\n  </tr>\n</table>\n": ["a", "b"],
        "<p>x</p><table><tr><td>y</td></tr></table>": ["x", "y"],
    }
    for src, expected in cases.items():
        assert block_texts(capture(src)) == expected, src
    # `<table>` inside `<select>` is dropped by the parser: never a silent mis-alignment
    try:
        r = capture("<select><table></select><table><tr><td>t</td></tr></table>")
    except ExtractionAlignmentError:
        pass
    else:
        assert block_texts(r) == ["t"]


def test_literal_lt_after_a_tag_aligns(capture):
    src = "<textarea>&lt;</textarea>"
    r = capture(src)
    assert r.canonical_text == "<"
    m = r.source_map.to_original(0, 1)
    assert (m.start, m.end) == (10, 14)
    assert src_slice(src, r, 0, 1) == "&lt;"
    src = "<p>a</p><p><!--c-->< b"
    r = capture(src)
    assert block_texts(r) == ["a", "< b"]
    assert src_slice(src, r, r.blocks[1].start, r.blocks[1].end) == "< b"


def test_head_without_close_and_unclosed_tags(capture):
    src = "<html><head><title>T</title><meta charset=utf-8><body><p>one<p>two<b>bold<i>it"
    r = capture(src, skipped=("T",))
    assert block_texts(r) == ["one", "two" + "bold" + "it"]
    assert src_slice(src, r, r.blocks[0].start, r.blocks[0].end) == "one"
    r = capture("<head><title>T</title><p>x", skipped=("T",))
    assert r.canonical_text == "x"


def test_unterminated_markup_at_end(capture):
    for src in ("<p>a</p><!-- open", "<p>a</p><div class='x", "<p>a</p><textarea>", "<p>a</p><!"):
        r = capture(src)
        assert r.canonical_text == "a", src


def test_no_text_is_a_typed_error():
    with pytest.raises(ExtractionError) as e:
        html_mod.EXTRACTOR.extract(b"<script>x</script>", text_encoding=None, limits=LIMITS)
    assert e.value.code == "no_text_layer"


def test_same_bytes_twice_give_equal_results():
    data = b"<h1>T</h1><p>a &amp; b<p>\xd8\xa7"
    a = html_mod.EXTRACTOR.extract(data, text_encoding=None, limits=LIMITS)
    b = html_mod.EXTRACTOR.extract(data, text_encoding=None, limits=LIMITS)
    assert a.canonical_text == b.canonical_text
    assert a.blocks == b.blocks
    assert a.source_map == b.source_map
    assert a.raw_map == b.raw_map
    assert a.extractor_version == b.extractor_version


def test_pathological_inputs_finish_quickly():
    # The pinned parser is quadratic on one huge run of '<' or on tens of thousands of nested elements, so each
    # input here is about 1 MiB in total. Inputs with a long '<' run or very deep nesting now fail fast with
    # `html_too_complex` (the cost guard); the others still go through the parser and must stay quick.
    mixed = ("x" + "<" * 100_000 + "<div>" * 8_000 + "deep" + "</div>" * 8_000 + "<3" * 50_000) * 2
    cases = [
        mixed,
        "a<a " * 200_000,
        "<p>a &" * 100_000,
        "<p>" + "&amp;" * 200_000,
        "<!--" * 100_000 + "-->" + "<a b='" * 50_000,
    ]
    for src in cases:
        t0 = time.perf_counter()
        try:
            html_mod.EXTRACTOR.extract(src.encode(), text_encoding=None, limits=LIMITS)
        except ExtractionError:
            pass  # a typed failure is fine; running long or crashing is not
        assert time.perf_counter() - t0 < 5, src[:12]
    assert all(len(c) >= 500_000 for c in cases)


# -- parse-cost guard (html_too_complex) ----------------------------------------------------------------------


def _guard_raises(src: str) -> bool:
    try:
        html_mod._check_complexity(src)
    except ExtractionError as e:
        assert e.code == "html_too_complex"
        return True
    return False


def test_guard_lt_run_boundary():
    assert not _guard_raises("a" + "<" * 255 + " b")
    assert _guard_raises("text " + "<" * 256 + " more")


def test_guard_depth_boundary():
    assert not _guard_raises("<div>" * 4096 + "x")
    assert _guard_raises("<div>" * 4097 + "x")
    assert not _guard_raises(("<div>" * 4096 + "x" + "</div>" * 4096) * 3)


def test_guard_uncounted_constructs():
    for piece in ("<br>", "<img src=a>", "<p>x", "<li>x", "<td>x"):
        assert not _guard_raises(piece * 10_000), piece
    assert not _guard_raises("<svg>" + "<path d='1'/>" * 10_000 + "</svg>")
    for tag in ("script", "textarea"):
        assert not _guard_raises(f"<{tag}>" + "<div>" * 5_000 + f"</{tag}>"), tag
    assert not _guard_raises("<!--" + "<div>" * 5_000 + "-->")
    assert _guard_raises("<div/>" * 4097)


def test_guard_gives_no_negative_depth_credit():
    assert _guard_raises("</div>" * 5000 + "<div>" * 4097)


def test_guard_rejects_before_parser_on_large_input(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("parser must not be called")

    monkeypatch.setattr(html_mod, "LexborHTMLParser", boom)
    big = ExtractionLimits(max_chars=6 * 2**20)
    for src in ("<" * (5 * 2**20), "<div>" * 2**20):
        t0 = time.perf_counter()
        with pytest.raises(ExtractionError) as e:
            html_mod.EXTRACTOR.extract(src.encode(), text_encoding=None, limits=big)
        assert e.value.code == "html_too_complex"
        assert time.perf_counter() - t0 < 1
    with pytest.raises(ExtractionError) as e:
        html_mod.EXTRACTOR.extract(("<div>" * 999_000).encode(), text_encoding=None, limits=LIMITS)
    assert e.value.code == "html_too_complex"


def test_guard_sees_decoded_text_and_same_object(monkeypatch):
    seen: list[str] = []
    real = html_mod._check_complexity

    def spy(src):
        seen.append(src)
        return real(src)

    monkeypatch.setattr(html_mod, "_check_complexity", spy)
    with pytest.raises(ExtractionError) as e:
        html_mod.EXTRACTOR.extract(b"\xef\xbb\xbf" + b"<" * 256, text_encoding="utf-8-sig", limits=LIMITS)
    assert e.value.code == "html_too_complex"

    parsed: list[str] = []
    real_parser = html_mod.LexborHTMLParser

    def parser_spy(src, *args, **kwargs):
        parsed.append(src)
        return real_parser(src, *args, **kwargs)

    monkeypatch.setattr(html_mod, "LexborHTMLParser", parser_spy)
    seen.clear()
    html_mod.EXTRACTOR.extract(b"<p>hi</p>", text_encoding=None, limits=LIMITS)
    assert len(seen) == 1 and len(parsed) == 1 and seen[0] is parsed[0]


def test_guard_is_linear_on_balanced_input():
    src = "<div>a</div>" * 400_000
    t0 = time.perf_counter()
    html_mod._check_complexity(src)
    assert time.perf_counter() - t0 < 5


def test_guard_foreign_content_bypasses_are_closed():
    d5 = "<div>" * 5000
    for src in (
        "<svg><style>" + d5,
        "<math><textarea>" + d5,
        "<svg><title>" + d5,
        "<svg>" + "<div/>" * 5000,
        "<math>" + "<p/><div/>" * 5000,
        "<svg><desc>" + "<g/>" * 10 + "<foo/>" * 5000,
        "<svg></math><style>" + d5,
        "<svg><math></svg>" + "<div/>" * 5000,
        "<div><svg></div>" + "<section/>" * 5000,
        "<svg></p>" + "<section/>" * 5000,
        "<table><svg></table>" + "<article/>" * 5000,
    ):
        assert _guard_raises(src), src[:30]
    assert not _guard_raises("<svg>" + "<path d='1'/>" * 10_000 + "</svg>")
    assert not _guard_raises("<svg><title>t</title>" + "<path d='1'/>" * 300 + "</svg>")
