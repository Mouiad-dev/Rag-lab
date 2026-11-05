"""Shared minimal-PDF builder for tests (ASCII Standard-14 Helvetica; optional RC4 encryption)."""

import hashlib
import struct

_PAD = bytes.fromhex("28BF4E5E4E758A4164004E56FFFA01082E2E00B6D0683E802F0CA9FE6453697A")


def rc4(key: bytes, data: bytes) -> bytes:
    s = list(range(256))
    j = 0
    for i in range(256):
        j = (j + s[i] + key[i % len(key)]) % 256
        s[i], s[j] = s[j], s[i]
    out = bytearray()
    i = j = 0
    for byte in data:
        i = (i + 1) % 256
        j = (j + s[i]) % 256
        s[i], s[j] = s[j], s[i]
        out.append(byte ^ s[(s[i] + s[j]) % 256])
    return bytes(out)


def _pad(password: bytes) -> bytes:
    return (password + _PAD)[:32]


def _escape(text: str) -> bytes:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)").encode("ascii")


def build_pdf(pages: list[str], *, user: bytes | None = None, owner: bytes | None = None) -> bytes:
    """A minimal PDF (Standard-14 Helvetica, ASCII). With `user`/`owner`: security handler V1/R2, 40-bit RC4."""
    n = len(pages)
    file_id = hashlib.md5(b"rag-foundry-test-id").digest()
    key = b""
    enc_obj = b""
    if user is not None or owner is not None:
        user = user or b""
        owner = owner if owner is not None else user
        perms = -4
        # PDF 1.7 algorithm 3 (R2): /O
        o_value = rc4(hashlib.md5(_pad(owner)).digest()[:5], _pad(user))
        # algorithm 2 (R2): encryption key
        key = hashlib.md5(_pad(user) + o_value + struct.pack("<i", perms) + file_id).digest()[:5]
        # algorithm 4 (R2): /U
        u_value = rc4(key, _PAD)
        enc_obj = b"<< /Filter /Standard /V 1 /R 2 /Length 40 /P %d /O <%s> /U <%s> >>" % (
            perms,
            o_value.hex().encode(),
            u_value.hex().encode(),
        )
    objects: list[bytes] = []  # object i+1 at index i
    first_page_obj = 4
    kids = b" ".join(b"%d 0 R" % (first_page_obj + 2 * i) for i in range(n))
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(b"<< /Type /Pages /Kids [%s] /Count %d >>" % (kids, n))
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    for i, text in enumerate(pages):
        page_no = first_page_obj + 2 * i
        stream = b"BT /F1 12 Tf 72 700 Td (%s) Tj ET" % _escape(text) if text else b""
        if key:
            obj_key = hashlib.md5(key + struct.pack("<I", page_no + 1)[:3] + b"\x00\x00").digest()[:10]
            stream = rc4(obj_key, stream)
        objects.append(
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents %d 0 R "
            b"/Resources << /Font << /F1 3 0 R >> >> >>" % (page_no + 1)
        )
        objects.append(b"<< /Length %d >>\nstream\n%s\nendstream" % (len(stream), stream))
    if enc_obj:
        objects.append(enc_obj)
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objects):
        offsets.append(len(out))
        out += b"%d 0 obj\n%s\nendobj\n" % (i + 1, body)
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    for off in offsets:
        out += b"%010d 00000 n \n" % off
    trailer = b"<< /Size %d /Root 1 0 R /ID [<%s> <%s>]" % (
        len(objects) + 1,
        file_id.hex().encode(),
        file_id.hex().encode(),
    )
    if enc_obj:
        trailer += b" /Encrypt %d 0 R" % len(objects)
    out += b"trailer\n" + trailer + b" >>\nstartxref\n%d\n%%%%EOF\n" % xref
    return bytes(out)
