"""OpenRouter-backed chat agent with tool calling driven by the skill registry."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, NamedTuple

from openai import OpenAI

from . import skills

SYSTEM_PROMPT = (
    "You are an assistant that helps engineers analyze mechanical assemblies "
    "and get manufacturing quotes from JLCPCB. The user uploads a folder "
    "containing their design.\n"
    "\n"
    "Important: modern SolidWorks files (.SLDPRT, .SLDASM, .SLDDRW) are "
    "encrypted and cannot be read on Linux. The agent only parses STEP files "
    "(.step / .stp), which the user must export from SolidWorks (AP214 or "
    "AP242 preserves assembly structure and names). Call `list_cad_files` "
    "first; if the upload has SolidWorks files but no STEP, ask the user to "
    "export STEP and re-upload.\n"
    "\n"
    "Drawing PDFs carry a title block with Material and Manufacturing "
    "process fields — use `read_drawing_pdf` to read them, and "
    "`classify_parts_by_material` to group files into "
    "'classified/1. Parts/<N. Material>/<N. Process>/' (default "
    "dry_run=true). Before calling it with dry_run=false, ALWAYS call "
    "`ask_user_approval` first with a concise summary of the plan. "
    "`ask_user_approval` pauses the conversation and shows the user Yes/No "
    "buttons; their click comes back as the next user message.\n"
    "\n"
    "When the user asks to see / open / look at / view / preview a specific "
    "drawing PDF, call `show_pdf` with its path. It renders the PDF inline "
    "in the chat UI — call it directly instead of describing the file."
)

# Curated list of models exposed to the frontend dropdown.
AVAILABLE_MODELS: list[dict[str, str]] = [
    {"id": "anthropic/claude-opus-4.7",    "label": "Claude Opus 4.7"},
    {"id": "anthropic/claude-sonnet-4.6",  "label": "Claude Sonnet 4.6"},
    {"id": "anthropic/claude-haiku-4.5",   "label": "Claude Haiku 4.5"},
    {"id": "openai/gpt-5",                 "label": "GPT-5"},
    {"id": "openai/gpt-5-mini",            "label": "GPT-5 mini"},
    {"id": "openai/gpt-4o",                "label": "GPT-4o"},
    {"id": "openai/gpt-4o-mini",           "label": "GPT-4o mini"},
]


@dataclass
class Session:
    session_id: str
    upload_dir: str | None = None
    messages: list[dict[str, Any]] = field(default_factory=list)


def _client() -> OpenAI:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is not set. Source env.sh first.")
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=key,
        default_headers={
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "manufacturing-agent",
        },
    )


def _tool_schema() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": s.name,
                "description": s.description,
                "parameters": s.parameters,
            },
        }
        for s in skills.all_skills()
    ]


class ChatResult(NamedTuple):
    reply: str
    approval: dict[str, Any] | None = None
    viewer: dict[str, Any] | None = None


def _run_tool(session: Session, name: str, arguments: str) -> tuple[str, Any]:
    """Returns (json_for_model, raw_result). raw_result is None on error."""
    s = skills.get(name)
    if s is None:
        return json.dumps({"error": f"Unknown skill: {name}"}), None
    try:
        kwargs = json.loads(arguments or "{}")
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Bad JSON arguments: {e}"}), None
    try:
        result = s.handler(session=session, **kwargs)
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {e}"}), None
    return json.dumps(result, default=str), result


def chat(session: Session, user_message: str, model: str,
         max_tool_iters: int = 6) -> ChatResult:
    if not session.messages:
        session.messages.append({"role": "system", "content": SYSTEM_PROMPT})
    session.messages.append({"role": "user", "content": user_message})

    client = _client()
    tools = _tool_schema()

    for _ in range(max_tool_iters):
        resp = client.chat.completions.create(
            model=model,
            messages=session.messages,
            tools=tools or None,
        )
        msg = resp.choices[0].message
        session.messages.append(msg.model_dump(exclude_none=True))

        if not msg.tool_calls:
            return ChatResult(reply=msg.content or "")

        pending_approval: dict[str, Any] | None = None
        pending_viewer: dict[str, Any] | None = None
        for call in msg.tool_calls:
            json_result, raw = _run_tool(session, call.function.name,
                                         call.function.arguments)
            session.messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": json_result,
            })
            if (
                isinstance(raw, dict)
                and raw.get("__approval_request__")
                and pending_approval is None
            ):
                pending_approval = {
                    "question": raw.get("question", "Proceed?"),
                    "summary": raw.get("summary", ""),
                }
            if (
                isinstance(raw, dict)
                and raw.get("__viewer__")
                and pending_viewer is None
            ):
                pending_viewer = {
                    "kind": raw.get("__viewer__"),
                    "url": raw.get("url"),
                    "title": raw.get("title"),
                    "format": raw.get("format"),
                }

        if pending_approval is not None:
            # Surface the approval question directly; don't re-ask the model.
            text = pending_approval["question"]
            if pending_approval["summary"]:
                text = f"{text}\n\n{pending_approval['summary']}"
            return ChatResult(reply=text, approval=pending_approval,
                              viewer=pending_viewer)

        if pending_viewer is not None:
            # Show the file immediately; no need to wait for the model to
            # narrate a follow-up turn (mirrors approval's early-return).
            title = pending_viewer.get("title") or "file"
            return ChatResult(reply=f"Showing {title}.",
                              viewer=pending_viewer)

    return ChatResult(reply="(stopped: hit max tool iterations)")
