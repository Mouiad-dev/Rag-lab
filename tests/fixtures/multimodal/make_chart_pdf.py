"""Deterministic generator for chart_report.pdf (stdlib only, PDF 1.4 by hand).

Page 0: text paragraph. Page 1: bar chart whose values exist only as bar heights (South is tallest).
Page 2: ruled table with text cells. Page 3: one rectangle and no text layer.
No dates, IDs or random data, so repeated runs give identical bytes.
Run: python tests/fixtures/multimodal/make_chart_pdf.py
"""

from pathlib import Path

OUTPUT = Path(__file__).with_name("chart_report.pdf")
PAGE_W, PAGE_H = 612, 792


def _esc(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _text(x: int, y: int, size: int, text: str) -> str:
    return f"BT /F1 {size} Tf {x} {y} Td ({_esc(text)}) Tj ET\n"


def _rect(x: int, y: int, w: int, h: int, rgb: tuple[float, float, float]) -> str:
    r, g, b = rgb
    return f"{r} {g} {b} rg {x} {y} {w} {h} re f\n"


def _line(x1: int, y1: int, x2: int, y2: int) -> str:
    return f"0 0 0 RG 1 w {x1} {y1} m {x2} {y2} l S\n"


def _page0() -> str:
    out = _text(72, 720, 20, "Regional report")
    lines = [
        "This report summarises visitor activity across four regions.",
        "Figures are shown on the following pages.",
        "The chart page shows relative visitor counts as bars.",
    ]
    for i, line in enumerate(lines):
        out += _text(72, 680 - 20 * i, 12, line)
    return out


def _page1() -> str:
    out = _text(72, 720, 18, "Visitors by region 2025")
    base = 200
    bars = [("North", 180), ("South", 340), ("East", 120), ("West", 250)]
    out += _line(60, base, 560, base)
    for i, (label, height) in enumerate(bars):
        x = 90 + i * 120
        out += _rect(x, base, 70, height, (0.2, 0.4, 0.8))
        out += _text(x + 15, base - 20, 12, label)
    return out


def _page2() -> str:
    out = _text(72, 720, 18, "Opening hours")
    rows = [("Day", "Hours"), ("Monday", "09:00 - 17:00"), ("Saturday", "10:00 - 14:00"), ("Sunday", "Closed")]
    left, mid, right, top, row_h = 72, 232, 432, 680, 40
    for i, (a, b) in enumerate(rows):
        y = top - i * row_h
        out += _line(left, y, right, y)
        out += _text(left + 8, y - 26, 12, a)
        out += _text(mid + 8, y - 26, 12, b)
    bottom = top - len(rows) * row_h
    out += _line(left, bottom, right, bottom)
    for x in (left, mid, right):
        out += _line(x, top, x, bottom)
    return out


def _page3() -> str:
    return _rect(156, 296, 300, 200, (0.8, 0.2, 0.2))


def build_chart_pdf() -> bytes:
    contents = [_page0(), _page1(), _page2(), _page3()]
    n = len(contents)
    # objects: 1 catalog, 2 pages, 3 font, then (page, stream) pairs
    page_ids = [4 + 2 * i for i in range(n)]
    objs: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        ("<< /Type /Pages /Kids [" + " ".join(f"{p} 0 R" for p in page_ids) + f"] /Count {n} >>").encode("ascii"),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    for i, content in enumerate(contents):
        data = content.encode("ascii")
        objs.append(
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {PAGE_W} {PAGE_H}] "
                f"/Resources << /Font << /F1 3 0 R >> >> /Contents {page_ids[i] + 1} 0 R >>"
            ).encode("ascii")
        )
        objs.append(b"<< /Length " + str(len(data)).encode("ascii") + b" >>\nstream\n" + data + b"endstream")
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for num, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{num} 0 obj\n".encode("ascii") + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n".encode("ascii") + b"0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode("ascii")
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode("ascii")
    return bytes(out)


if __name__ == "__main__":
    OUTPUT.write_bytes(build_chart_pdf())
