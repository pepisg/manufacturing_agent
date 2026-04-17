"""Viewer skills: when the agent wants the user to see a file, it calls
`show_pdf`. The handler returns a sentinel dict with `__viewer__` that the
agent loop surfaces to the frontend, which renders an inline PDF viewer
under the assistant message.
"""
from __future__ import annotations

from pathlib import Path
from urllib.parse import quote

from . import skill

PDF_EXTS = {".pdf"}


def _resolve(session, relative_path: str) -> Path:
    """Path-traversal-safe resolver; mirrors the pattern in cad.py/drawings.py."""
    if not session.upload_dir:
        raise ValueError("No folder has been uploaded for this session.")
    root = Path(session.upload_dir).resolve()
    target = (root / relative_path).resolve()
    if root not in target.parents and target != root:
        raise ValueError("Path escapes upload directory.")
    if not target.is_file():
        raise ValueError(f"Not a file: {relative_path}")
    return target


def _upload_url(session, relative_path: str) -> str:
    """Build a `/uploads/<session_id>/<rel>` URL with each path segment
    URL-encoded (the real paths contain spaces and brackets like
    '[Gripper Body]')."""
    rel = Path(relative_path).as_posix().lstrip("/")
    encoded = "/".join(quote(seg, safe="") for seg in rel.split("/"))
    return f"/uploads/{quote(session.session_id, safe='')}/{encoded}"


@skill(
    name="show_pdf",
    description=(
        "Open a PDF from the uploaded folder in the chat UI so the user can "
        "see it. Call this when the user asks to 'see', 'open', 'look at', "
        "or 'view' a specific drawing PDF."
    ),
    parameters={
        "type": "object",
        "properties": {
            "relative_path": {
                "type": "string",
                "description": "Path to the PDF relative to the uploaded folder.",
            },
        },
        "required": ["relative_path"],
    },
)
def show_pdf(session, relative_path: str) -> dict:
    try:
        path = _resolve(session, relative_path)
    except ValueError as e:
        return {"error": str(e)}
    if path.suffix.lower() not in PDF_EXTS:
        return {"error": f"Not a PDF: {relative_path}"}
    return {
        "__viewer__": "pdf",
        "url": _upload_url(session, relative_path),
        "title": path.name,
    }


