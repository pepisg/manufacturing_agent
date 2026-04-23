"""Viewer skills: when the agent wants the user to see a file, it calls
`show_pdf` or `show_step`. The handler returns a sentinel dict with
`__viewer__` that the agent loop surfaces to the frontend, which renders
the file inline under the assistant message (PDF via pdf.js, STEP via
three.js + occt-import-js).
"""
from __future__ import annotations

from pathlib import Path
from urllib.parse import quote

from . import skill

PDF_EXTS = {".pdf"}
STEP_EXTS = {".step", ".stp"}


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


@skill(
    name="show_step",
    description=(
        "Open a STEP file (.step/.stp) in the chat UI as an interactive 3D "
        "viewer (rotate/pan/zoom). Call this when the user asks to 'see', "
        "'view', 'open', 'look at', or 'preview' a specific STEP file or 3D "
        "model. Large assemblies may take several seconds to parse in the "
        "browser."
    ),
    parameters={
        "type": "object",
        "properties": {
            "relative_path": {
                "type": "string",
                "description": "Path to the STEP file relative to the uploaded folder.",
            },
        },
        "required": ["relative_path"],
    },
)
def show_step(session, relative_path: str) -> dict:
    try:
        path = _resolve(session, relative_path)
    except ValueError as e:
        return {"error": str(e)}
    if path.suffix.lower() not in STEP_EXTS:
        return {"error": f"Not a STEP file: {relative_path}"}
    return {
        "__viewer__": "step",
        "url": _upload_url(session, relative_path),
        "title": path.name,
    }
