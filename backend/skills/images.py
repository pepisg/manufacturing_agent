"""Rasterize a drawing PDF page to PNG and hand it back as an image block
that the agent loop injects into the next LLM turn as a user-role message.

Why user-role: OpenAI (and therefore OpenRouter's OpenAI route) rejects
`image_url` content inside a `tool` message with a 400. The agent loop reads
the `__image__` sentinel, posts a short JSON stub as the tool result, and
appends a follow-up user message carrying the image block.
"""
from __future__ import annotations

import base64
import hashlib
import io
from pathlib import Path

import pypdfium2 as pdfium

from . import skill

_DEFAULT_SCALE = 200 / 72  # ~200 DPI; readable title block without bloating tokens


def _resolve(session, relative_path: str) -> Path:
    if not session.upload_dir:
        raise ValueError("No folder has been uploaded for this session.")
    root = Path(session.upload_dir).resolve()
    target = (root / relative_path).resolve()
    if root not in target.parents and target != root:
        raise ValueError("Path escapes upload directory.")
    if not target.is_file():
        raise ValueError(f"Not a file: {relative_path}")
    return target


def _render_pdf_page(path: Path, page: int, scale: float) -> tuple[bytes, int, int]:
    doc = pdfium.PdfDocument(str(path))
    try:
        n = len(doc)
        idx = max(0, min(page - 1, n - 1))
        pil = doc[idx].render(scale=scale).to_pil().convert("RGB")
    finally:
        doc.close()
    buf = io.BytesIO()
    pil.save(buf, format="PNG", optimize=True)
    return buf.getvalue(), pil.width, pil.height


@skill(
    name="render_part_image",
    description=(
        "Rasterize page 1 of a mechanical drawing PDF to a PNG and show it to "
        "you (the model) inline, so you can see the part's geometry, views, "
        "and dimensions. Use this whenever the user asks about a part's "
        "shape, what it looks like, how it's oriented, how features are "
        "laid out, or any visual question — title-block text alone isn't "
        "enough. After calling this you will receive the image in the next "
        "turn; answer from what you observe."
    ),
    parameters={
        "type": "object",
        "properties": {
            "relative_path": {
                "type": "string",
                "description": "Path to the PDF relative to the uploaded folder.",
            },
            "page": {
                "type": "integer",
                "description": "1-indexed page number. Default 1.",
            },
        },
        "required": ["relative_path"],
    },
)
def render_part_image(session, relative_path: str, page: int = 1) -> dict:
    try:
        path = _resolve(session, relative_path)
    except ValueError as e:
        return {"error": str(e)}
    if path.suffix.lower() != ".pdf":
        return {"error": f"Not a PDF: {relative_path}"}

    try:
        png, w, h = _render_pdf_page(path, page, _DEFAULT_SCALE)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}

    cache_dir = Path(session.upload_dir) / ".cache" / "images"
    cache_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(f"{relative_path}|{page}|{path.stat().st_mtime_ns}".encode()).hexdigest()[:16]
    out = cache_dir / f"{digest}.png"
    out.write_bytes(png)

    return {
        "__image__": "png",
        "file": relative_path,
        "page": page,
        "width": w,
        "height": h,
        "cache_path": str(out.relative_to(Path(session.upload_dir))),
        "data_url": "data:image/png;base64," + base64.b64encode(png).decode(),
    }
