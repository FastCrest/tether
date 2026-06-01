"""Tiny dependency-free text PDF writer for the Comply technical file."""

from __future__ import annotations

import textwrap
from pathlib import Path


def _escape_pdf_text(text: str) -> str:
    safe = text.encode("latin-1", "replace").decode("latin-1")
    return safe.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _pages_from_text(text: str, *, width: int = 96, height: int = 54) -> list[list[str]]:
    lines: list[str] = []
    for raw in text.splitlines():
        if not raw:
            lines.append("")
            continue
        lines.extend(textwrap.wrap(raw, width=width, replace_whitespace=False) or [""])
    return [lines[i : i + height] for i in range(0, len(lines), height)] or [[]]


def write_text_pdf(path: str | Path, *, title: str, text: str) -> Path:
    """Write a basic valid PDF containing wrapped monospace-ish text."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pages = _pages_from_text(f"{title}\n\n{text}")

    objects: list[bytes] = []
    catalog_id = 1
    pages_id = 2
    font_id = 3
    page_ids: list[int] = []
    content_ids: list[int] = []

    next_id = 4
    for _ in pages:
        page_ids.append(next_id)
        content_ids.append(next_id + 1)
        next_id += 2

    kids = " ".join(f"{pid} 0 R" for pid in page_ids)
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>".encode("ascii"))
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    for idx, lines in enumerate(pages):
        content_id = content_ids[idx]
        objects.append(
            (
                f"<< /Type /Page /Parent {pages_id} 0 R /MediaBox [0 0 612 792] "
                f"/Resources << /Font << /F1 {font_id} 0 R >> >> "
                f"/Contents {content_id} 0 R >>"
            ).encode("ascii")
        )
        stream_lines = ["BT", "/F1 9 Tf", "42 760 Td", "12 TL"]
        for line in lines:
            stream_lines.append(f"({_escape_pdf_text(line)}) Tj")
            stream_lines.append("T*")
        stream_lines.append("ET")
        stream = "\n".join(stream_lines).encode("latin-1", "replace")
        objects.append(
            b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream"
        )

    # Object numbers are fixed by insertion order: 1..N.
    chunks = [b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"]
    offsets = [0]
    for obj_num, body in enumerate(objects, start=1):
        offsets.append(sum(len(c) for c in chunks))
        chunks.append(f"{obj_num} 0 obj\n".encode("ascii") + body + b"\nendobj\n")
    xref_offset = sum(len(c) for c in chunks)
    chunks.append(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    chunks.append(b"0000000000 65535 f \n")
    for off in offsets[1:]:
        chunks.append(f"{off:010d} 00000 n \n".encode("ascii"))
    chunks.append(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root {catalog_id} 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode("ascii")
    )
    out.write_bytes(b"".join(chunks))
    return out


__all__ = ["write_text_pdf"]
