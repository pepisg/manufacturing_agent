"""ask_user_approval: sentinel skill the agent loop intercepts to pause the
conversation and ask the user for a Yes/No via UI buttons.

The handler returns a dict with `__approval_request__: True`. `backend/agent.py`
detects that flag, stops the tool loop, and surfaces the question to the
frontend as a structured reply. The user's Yes/No click is sent back as a
normal chat message, and the model — which still sees its own pending
question in the conversation — decides whether to proceed.
"""
from __future__ import annotations

from . import skill


@skill(
    name="ask_user_approval",
    description=(
        "Pause and ask the user to approve an action via Yes/No buttons. "
        "Call this BEFORE any irreversible operation (copying files, "
        "overwriting directories, etc). Pass a clear question and a short "
        "summary of what will happen if they say yes."
    ),
    parameters={
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question to show the user, e.g. 'Copy these 5 parts into classified/?'",
            },
            "summary": {
                "type": "string",
                "description": "Optional short description of what Yes would do.",
            },
        },
        "required": ["question"],
    },
)
def ask_user_approval(session, question: str, summary: str = "") -> dict:
    return {
        "__approval_request__": True,
        "question": question,
        "summary": summary,
    }
