from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jlcpc_browser.agent.task import QUOTE_URL, build_quote_task
from jlcpc_browser.context.part_brief import build_part_brief
from jlcpc_browser.discovery import PartJob

# OpenAI-compatible proxies (OpenRouter, etc.)
_DEFAULT_OPENROUTER_BASE = "https://openrouter.ai/api/v1"


def build_browser_profile(*, keep_alive: bool | None = None):
    """
    Browser defaults tuned for heavy SPAs (e.g. JLC quote): longer load waits and
    no bundled extensions (ad blockers often blank third-party sites).

    Env:
    - ``JLCPCB_HEADLESS``: ``0`` / ``false`` = visible window (often fixes blank pages);
      ``1`` / ``true`` = headless; unset = browser-use default (headless if no display).
    - ``JLCPCB_MIN_PAGE_LOAD_WAIT`` / ``JLCPCB_NETWORK_IDLE_WAIT``: seconds (float).
    - ``JLCPCB_DISABLE_BROWSER_EXTENSIONS``: default ``1`` (disable uBlock etc.). Set ``0`` to re-enable.
    """
    from browser_use.browser.profile import BrowserProfile

    hl = os.environ.get("JLCPCB_HEADLESS")
    ext_env = os.environ.get("JLCPCB_DISABLE_BROWSER_EXTENSIONS", "1").strip().lower()
    enable_extensions = ext_env in ("0", "false", "no")
    kwargs: dict[str, Any] = {
        "minimum_wait_page_load_time": float(os.environ.get("JLCPCB_MIN_PAGE_LOAD_WAIT", "1.5")),
        "wait_for_network_idle_page_load_time": float(os.environ.get("JLCPCB_NETWORK_IDLE_WAIT", "3.0")),
        "enable_default_extensions": enable_extensions,
    }

    if hl is not None and hl.strip() != "":
        lv = hl.strip().lower()
        if lv in ("0", "false", "no"):
            kwargs["headless"] = False
        elif lv in ("1", "true", "yes"):
            kwargs["headless"] = True

    if keep_alive is not None:
        kwargs["keep_alive"] = keep_alive

    return BrowserProfile(**kwargs)


def _build_chat_llm(model: str):
    """
    Configure browser-use ChatOpenAI from the environment.

    - **OpenAI (default):** set ``OPENAI_API_KEY`` only.
    - **OpenRouter:** set ``OPENROUTER_API_KEY`` and ``JLCPCB_LLM_MODEL=provider/model``;
      base URL defaults to OpenRouter unless ``JLCPCB_OPENAI_BASE_URL`` is set.
    - **Any OpenAI-compatible API:** set ``JLCPCB_OPENAI_BASE_URL`` and
      ``OPENROUTER_API_KEY`` or ``OPENAI_API_KEY``.

    Optional OpenRouter attribution: ``OPENROUTER_HTTP_REFERER``, ``OPENROUTER_APP_TITLE``.
    """
    from browser_use import ChatOpenAI

    explicit_base = os.environ.get("JLCPCB_OPENAI_BASE_URL", "").strip()
    openrouter_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    openai_key = os.environ.get("OPENAI_API_KEY", "").strip()

    if explicit_base:
        api_key = openrouter_key or openai_key
        kwargs: dict[str, Any] = {"model": model, "base_url": explicit_base.rstrip("/")}
        if api_key:
            kwargs["api_key"] = api_key
        return ChatOpenAI(**kwargs)

    if openrouter_key:
        kwargs: dict[str, Any] = {
            "model": model,
            "base_url": _DEFAULT_OPENROUTER_BASE,
            "api_key": openrouter_key,
        }
        referer = os.environ.get("OPENROUTER_HTTP_REFERER", "").strip()
        title = os.environ.get("OPENROUTER_APP_TITLE", "").strip()
        if referer or title:
            headers: dict[str, str] = {}
            if referer:
                headers["HTTP-Referer"] = referer
            if title:
                headers["X-Title"] = title
            kwargs["default_headers"] = headers
        return ChatOpenAI(**kwargs)

    return ChatOpenAI(model=model)


def _llm_kind() -> str:
    if os.environ.get("JLCPCB_OPENAI_BASE_URL", "").strip():
        return "custom_base_url"
    if os.environ.get("OPENROUTER_API_KEY", "").strip():
        return "openrouter"
    return "openai_default"


def _history_to_summary(history: Any) -> dict[str, Any]:
    """Best-effort serialization of browser-use AgentHistoryList."""
    out: dict[str, Any] = {"repr_tail": repr(history)[-8000:]}
    if hasattr(history, "model_dump"):
        try:
            out["model_dump"] = history.model_dump()
        except Exception:
            pass
    if hasattr(history, "final_result"):
        try:
            fr = history.final_result()
            if isinstance(fr, (str, dict, list, int, float, bool, type(None))):
                out["final_result"] = fr
            else:
                out["final_result"] = repr(fr)
        except Exception:
            pass
    return out


async def run_quote_agent(
    job: PartJob,
    *,
    max_steps: int = 80,
    model: str | None = None,
    browser: Any | None = None,
) -> dict[str, Any]:
    """
    Run browser-use Agent for one PartJob. Requires Python >= 3.11 and credentials:
    ``OPENAI_API_KEY`` (OpenAI), or ``OPENROUTER_API_KEY`` (OpenRouter), or
    ``JLCPCB_OPENAI_BASE_URL`` + a key for other OpenAI-compatible APIs.

    Pass ``browser`` (a shared ``Browser`` / ``BrowserSession`` with ``keep_alive=True``)
    so multiple parts reuse one session and the shopping cart accumulates.
    """
    from browser_use import Agent

    brief = build_part_brief(job.pdf_path)
    task = build_quote_task(job, brief)
    # Only CAD/mesh paths are allowlisted — never the blueprint PDF.
    paths = [str(x.resolve()) for x in job.companion_paths]

    llm_model = model or os.environ.get("JLCPCB_LLM_MODEL", "gpt-4o-mini")
    llm = _build_chat_llm(llm_model)

    # Open quote first: browser-use skips auto-navigate when the task contains
    # multiple URLs (PDF text often includes links), which leaves about:blank.
    initial_navigate = [{"navigate": {"url": QUOTE_URL, "new_tab": False}}]

    agent_kw: dict[str, Any] = {
        "task": task,
        "llm": llm,
        "available_file_paths": paths,
        "initial_actions": initial_navigate,
        "extend_system_message": (
            "Upload only the minimum CAD file(s) required for the selected product (often one; 3D: STL or STEP per form). "
            "Never the PDF. After upload: Product desc — type others, wait 1s, click others in the auto-open list; fill additional-info box; then SAVE TO CART."
        ),
    }
    if browser is not None:
        agent_kw["browser"] = browser
    else:
        agent_kw["browser_profile"] = build_browser_profile()

    agent = Agent(**agent_kw)
    history = await agent.run(max_steps=max_steps)
    return {
        "part_id": job.part_id,
        "ok": True,
        "ts": datetime.now(timezone.utc).isoformat(),
        "model": llm_model,
        "llm_kind": _llm_kind(),
        "history": _history_to_summary(history),
    }


def dry_run_record(job: PartJob) -> dict[str, Any]:
    """Discovery + PDF text only (no browser)."""
    brief = build_part_brief(job.pdf_path)
    return {
        "part_id": job.part_id,
        "material": job.material,
        "manufacture_form": job.manufacture_form,
        "pdf_path": str(job.pdf_path),
        "companion_paths": [str(p) for p in job.companion_paths],
        "part_brief_preview": brief[:2000] + ("…" if len(brief) > 2000 else ""),
        "part_brief_chars": len(brief),
    }


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
