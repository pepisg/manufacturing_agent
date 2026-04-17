from __future__ import annotations

from pathlib import Path

# Keep drawing text bounded for LLM context; title blocks are usually early.
DEFAULT_MAX_CHARS = 12_000


def build_part_brief(pdf_path: Path, *, max_chars: int = DEFAULT_MAX_CHARS) -> str:
    """
    Extract readable text from a blueprint PDF for agent context.
    Uses PyMuPDF only (no user configuration).
    """
    try:
        import fitz  # PyMuPDF
    except ImportError as e:
        raise ImportError("pymupdf is required for PDF extraction: pip install pymupdf") from e

    path = pdf_path.expanduser().resolve()
    doc = fitz.open(path)
    try:
        chunks: list[str] = []
        for page in doc:
            chunks.append(page.get_text() or "")
            joined = "\n".join(chunks)
            if len(joined) >= max_chars:
                break
        text = "\n".join(chunks).strip()
        if not text:
            return (
                f"(No embedded text in PDF: {path.name}; the drawing may be image-only. "
                "The browser agent should rely on folder intent and file names.)"
            )
        if len(text) > max_chars:
            text = text[:max_chars] + "\n… [truncated]"
        return text
    finally:
        doc.close()


def folder_intent_summary(material: str, manufacture_form: str) -> str:
    return (
        f"Folder intent: material category `{material}`, manufacturing form `{manufacture_form}`. "
        "Use this as coarse guidance for which JLCPCB product line to open (e.g. 3D Printing, "
        "CNC Machining, Sheet Metal). Refine using the drawing text below."
    )
