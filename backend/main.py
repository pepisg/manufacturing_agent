"""FastAPI app: serves the chat UI, handles folder uploads and chat turns."""
from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import agent, skills

ROOT = Path(__file__).resolve().parent.parent
UPLOAD_ROOT = ROOT / "uploads"
FRONTEND = ROOT / "frontend"

UPLOAD_ROOT.mkdir(exist_ok=True)
skills.load_all()

app = FastAPI(title="manufacturing-agent")

_sessions: dict[str, agent.Session] = {}


def _session(session_id: str) -> agent.Session:
    s = _sessions.get(session_id)
    if s is None:
        s = agent.Session(session_id=session_id)
        _sessions[session_id] = s
    return s


class ChatRequest(BaseModel):
    session_id: str
    message: str
    model: str


class ChatResponse(BaseModel):
    reply: str
    approval: dict | None = None
    viewer: dict | None = None


@app.get("/api/models")
def models():
    return {"models": agent.AVAILABLE_MODELS, "default": agent.AVAILABLE_MODELS[0]["id"]}


@app.get("/api/session/new")
def new_session():
    return {"session_id": uuid.uuid4().hex}


@app.post("/api/upload")
async def upload(
    session_id: str = Form(...),
    files: list[UploadFile] = File(...),
    paths: list[str] = Form(...),
):
    if len(files) != len(paths):
        raise HTTPException(400, "files and paths length mismatch")
    sess = _session(session_id)

    # Fresh upload dir per upload — replaces any prior folder.
    target = UPLOAD_ROOT / session_id
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)

    saved = 0
    for f, rel in zip(files, paths):
        safe = Path(rel).as_posix().lstrip("/")
        # Block path traversal.
        if ".." in Path(safe).parts:
            continue
        dest = target / safe
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("wb") as out:
            shutil.copyfileobj(f.file, out)
        saved += 1

    sess.upload_dir = str(target)
    return {"saved": saved, "upload_dir": str(target)}


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    sess = _session(req.session_id)
    try:
        result = agent.chat(sess, req.message, model=req.model)
    except RuntimeError as e:
        raise HTTPException(500, str(e))
    return ChatResponse(reply=result.reply, approval=result.approval,
                        viewer=result.viewer)


# Serve uploaded files so the viewer skills can link to them directly.
# Must be mounted BEFORE the root mount below — the root mount is a
# catch-all and would otherwise swallow /uploads/... requests.
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_ROOT)), name="uploads")

# Serve the static frontend at /.
app.mount("/", StaticFiles(directory=str(FRONTEND), html=True), name="frontend")
