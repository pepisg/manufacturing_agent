"""JLCPCB quoting skills.

Flow per part: launch browser-use with a minimal task (pick product line,
upload CAD, wait for the instant price to render — do NOT fill Product
desc / additional info / Save to Cart). When the agent stops (success or
timeout), we grab the live page's raw HTML via `Page.evaluate` and send a
cleaned chunk to gpt-4o-mini to extract the computed price as JSON.

Pulls the coworker's `jlcpc-browser/` package for its helpers
(`build_browser_profile`, `_build_chat_llm`, `build_part_brief`, quote URL)
via a sys.path shim. Each per-part run is wrapped in an asyncio timeout
(default 120s, overridable via `JLCPCB_QUOTE_TIMEOUT_S`).
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote as urlquote

from . import skill

# Make `import jlcpc_browser.*` resolve to the coworker's package without
# installing it. Idempotent: `insert` into sys.path is a no-op on re-import.
_JLCPC_SRC = (Path(__file__).parent / "jlcpc-browser" / "src").resolve()
if str(_JLCPC_SRC) not in sys.path:
    sys.path.insert(0, str(_JLCPC_SRC))

CLASSIFIED_PARTS_SUBPATH = Path("classified") / "1. Parts"
QUOTES_CACHE_SUBPATH = Path(".cache") / "quotes"
UPLOAD_STAGE_SUBPATH = Path(".cache") / "jlcpcb_uploads"
DEFAULT_QUOTE_TIMEOUT_S = 120.0
_ENUM_PREFIX = re.compile(r"^\d+\.\s*")
_SAFE_STEM = re.compile(r"[^a-z0-9]+")


def _quote_timeout_s() -> float:
    raw = os.environ.get("JLCPCB_QUOTE_TIMEOUT_S", "").strip()
    if not raw:
        return DEFAULT_QUOTE_TIMEOUT_S
    try:
        return max(10.0, float(raw))
    except ValueError:
        return DEFAULT_QUOTE_TIMEOUT_S


def _upload_root(session) -> Path:
    if not session.upload_dir:
        raise ValueError("No folder has been uploaded for this session.")
    return Path(session.upload_dir).resolve()


def _resolve(session, relative_path: str) -> Path:
    root = _upload_root(session)
    target = (root / relative_path).resolve()
    if root not in target.parents and target != root:
        raise ValueError("Path escapes upload directory.")
    if not target.is_file():
        raise ValueError(f"Not a file: {relative_path}")
    return target


def _strip_enum(name: str) -> str:
    """'0. Metal' -> 'Metal'; leaves non-enumerated names alone."""
    return _ENUM_PREFIX.sub("", name).strip()


def _artifact_url(session, rel: Path) -> str:
    parts = [urlquote(seg, safe="") for seg in rel.as_posix().split("/")]
    return f"/uploads/{urlquote(session.session_id, safe='')}/{'/'.join(parts)}"


def _write_artifact(session, filename: str, data: Any) -> tuple[Path, str]:
    root = _upload_root(session)
    cache_dir = root / QUOTES_CACHE_SUBPATH
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / filename
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str) + "\n",
                    encoding="utf-8")
    return path, _artifact_url(session, path.relative_to(root))


def _ts_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _safe_stem(part_id: str) -> str:
    """Lowercase a-z0-9 only. Used for the staging dir name — the upload
    filename itself is always 'solid.<ext>' so JLCPCB's dropzone sees nothing
    weird."""
    s = _SAFE_STEM.sub("-", part_id.lower()).strip("-")
    return s or "part"


def _stage_companions(job, root: Path):
    """Copy each companion file to `<root>/.cache/jlcpcb_uploads/<safe>/`
    renamed to `solid.<ext>` (with a counter suffix when two companions share
    an extension). Returns a new PartJob whose `companion_paths` point at the
    staged copies. The originals are untouched."""
    from jlcpc_browser.discovery import PartJob

    stage_dir = root / UPLOAD_STAGE_SUBPATH / _safe_stem(job.part_id)
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)

    ext_seen: dict[str, int] = {}
    staged: list[Path] = []
    for src in job.companion_paths:
        ext = src.suffix.lower()
        n = ext_seen.get(ext, 0)
        ext_seen[ext] = n + 1
        name = f"solid{ext}" if n == 0 else f"solid-{n+1}{ext}"
        dst = stage_dir / name
        shutil.copy2(src, dst)
        staged.append(dst)

    return PartJob(
        part_id=job.part_id,
        pdf_path=job.pdf_path,
        companion_paths=tuple(staged),
        material=job.material,
        manufacture_form=job.manufacture_form,
        source_dir=job.source_dir,
    )


def _job_for_pdf(pdf_path: Path, root: Path):
    """Build a PartJob from a PDF on disk. Siblings are matched via the
    shared `_part_identifier` (strips `[desc]` and `_R<nn>` suffix, lowercase),
    so a PDF named '...P050 [Wheel] _R00.pdf' pairs with a STEP named
    '...P050[Wheel]_R00.step' despite the space/bracket mismatch. For welding
    sub-parts (P050-P07 under a P050 PDF), we match any sibling whose
    identifier starts with this PDF's identifier + '-' or '.'."""
    from jlcpc_browser.discovery import PartJob

    from .drawings import _part_identifier

    ident = _part_identifier(pdf_path.name)
    companions_list: list[Path] = []
    for p in pdf_path.parent.iterdir():
        if not p.is_file() or p.suffix.lower() == ".pdf":
            continue
        p_ident = _part_identifier(p.name)
        if p_ident == ident or p_ident.startswith(ident + "-") or p_ident.startswith(ident + "."):
            companions_list.append(p)
    companions = tuple(sorted(companions_list, key=lambda p: p.name.casefold()))
    # If the PDF sits at classified/1. Parts/<material>/<process>/, extract
    # the folder intent; otherwise fall back to empty strings (task prompt
    # still has the drawing text to work with).
    material = manufacture_form = ""
    try:
        rel_parts = pdf_path.resolve().relative_to(root).parts
        if (
            len(rel_parts) >= 4
            and rel_parts[0] == "classified"
            and rel_parts[1] == "1. Parts"
        ):
            material = _strip_enum(rel_parts[2])
            manufacture_form = _strip_enum(rel_parts[3])
    except ValueError:
        pass

    return PartJob(
        part_id=pdf_path.stem,
        pdf_path=pdf_path,
        companion_paths=companions,
        material=material,
        manufacture_form=manufacture_form,
        source_dir=pdf_path.parent,
    )


# ---------------------------------------------------------------------------
# Price capture: minimal browser-use task + HTML → LLM parse
# ---------------------------------------------------------------------------


_PRICE_TASK_TEMPLATE = """You are automating a guest instant quote on JLCPCB. Start at {url}.

Inputs:
- Part id: {part_id}
- Folder intent: material `{material}`, manufacturing form `{form}`
- Drawing text (may mention material/process):
{brief}

Allowed CAD paths (upload exactly ONE — NEVER the PDF):
{paths}

Steps:
1. On the quote page, pick the correct top-level product line for this part
   (3D Printing, CNC Machining, Sheet Metal, etc.) using folder intent +
   drawing text. State which one in one short sentence before uploading.
2. Upload exactly ONE CAD file from the list above. Prefer the format the
   selected product expects (3D printing: STL or STEP; CNC / sheet metal:
   STEP).

   **CRITICAL — how to upload:** you MUST use the `upload_file` action,
   passing `index` = the index of the file-upload element (the dropzone /
   "Choose File" button / invisible `<input type=file>`) and `path` = one of
   the allowed CAD paths above VERBATIM.

   DO NOT call `click` on the upload button / dropzone — clicking it opens
   the host OS file-picker dialog, which you cannot interact with and which
   will freeze this task. DO NOT type the path into any text field. The
   ONLY way to attach the file is the `upload_file` action.

   If you accidentally clicked and an OS file-picker is now blocking the
   page, press Escape (send_keys "Escape") and retry with `upload_file`.

3. Wait for JLCPCB to compute and render the instant price. Do NOT fill
   Product desc, quantity, material, finish, or any other field.
4. As SOON as a numeric price is visible on the page, STOP and call your
   `done` action with a one-line summary.

DO NOT:
- Fill Product desc, additional info, quantity, or any other form field.
- Click Save to Cart, Add to Cart, or Checkout.
- Log in.
- Click the upload button (use `upload_file` instead).
"""


_PRICE_PARSER_PROMPT = (
    "You receive cleaned HTML from a JLCPCB instant-quote page after a CAD "
    "upload. Extract the computed prices. Return STRICT JSON (no prose, no "
    "markdown):\n"
    "{\n"
    '  "price": "<manufacturing total price WITHOUT shipping, e.g. '
    '$12.50 USD, or null>",\n'
    '  "currency": "<USD|CNY|EUR|... or null>",\n'
    '  "unit_price": "<per-unit price if shown, else null>",\n'
    '  "quantity": "<ordered qty or null>",\n'
    '  "shipping": "<shipping estimate / cost with currency, or null>",\n'
    '  "total_with_shipping": "<grand total including shipping if the '
    'page shows a combined total, else null>",\n'
    '  "lead_time": "<e.g. 3-5 business days, or null>",\n'
    '  "product_line": "<3D Printing | CNC Machining | Sheet Metal | ... '
    'or null>",\n'
    '  "notes": "<short extra context: resin/material chosen, bbox, etc., '
    'or null>"\n'
    "}\n"
    "Pricing rules:\n"
    "- `price` is the 'Total Price' / 'Order total' / manufacturing charge "
    "BEFORE shipping — NOT the shipping estimate.\n"
    "- `shipping` is the 'Shipping Estimate' / 'Shipping fee' line.\n"
    "- Ignore coupon amounts like 'Save $300.00'.\n"
    "If the page shows no numeric price, return all nulls. Never invent a "
    "price."
)


_PRICE_CURRENCY_PATTERN = re.compile(
    r"(?:[$€£¥]|\bUSD\b|\bEUR\b|\bCNY\b|\bGBP\b)\s*\d[\d,]*(?:\.\d{1,2})?"
    r"|\d[\d,]*(?:\.\d{1,2})?\s*(?:USD|EUR|CNY|GBP)\b",
    re.IGNORECASE,
)


def _clean_html_for_llm(html: str, max_chars: int = 80_000) -> str:
    """Turn the full page HTML into a focused 'price evidence' payload.

    JLCPCB's quote page is a 3MB Nuxt SPA whose visible DOM sits near the
    end of the document — naive head-truncation drops the price entirely.
    We strip scripts/styles/comments, collapse whitespace, then find every
    currency-shaped number and emit ~300 chars of context around each match.
    That keeps the LLM prompt under a few kB while still carrying every
    price label (Total Price, Shipping Estimate, Material, etc.)."""
    if not html:
        return ""
    cleaned = re.sub(r"<script\b[^>]*>.*?</script>", "", html,
                     flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<style\b[^>]*>.*?</style>", "", cleaned,
                     flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<!--.*?-->", "", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    matches = list(_PRICE_CURRENCY_PATTERN.finditer(cleaned))
    if not matches:
        return cleaned[-max_chars:] if len(cleaned) > max_chars else cleaned

    # Merge overlapping windows so adjacent prices (e.g. Material $0.32 /
    # Surface finish $0.00 / Total $0.32 inside the same panel) arrive as
    # one contiguous snippet instead of three.
    half_window = 300
    windows: list[tuple[int, int]] = []
    for m in matches:
        s = max(0, m.start() - half_window)
        e = min(len(cleaned), m.end() + half_window)
        if windows and s <= windows[-1][1]:
            windows[-1] = (windows[-1][0], max(windows[-1][1], e))
        else:
            windows.append((s, e))

    snippets = [cleaned[s:e] for s, e in windows]
    joined = "\n---\n".join(snippets)
    if len(joined) > max_chars:
        joined = joined[:max_chars]
    return joined


def _parse_price_html(html: str, job) -> dict:
    """LLM-based price extraction. Returns {} on any failure."""
    from .. import agent as agent_mod

    cleaned = _clean_html_for_llm(html)
    if not cleaned:
        return {}
    try:
        client = agent_mod._client()
    except RuntimeError:
        return {}
    try:
        resp = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[
                {"role": "system", "content": _PRICE_PARSER_PROMPT},
                {"role": "user",
                 "content": f"Part: {job.part_id}\n\nHTML:\n{cleaned}"},
            ],
            response_format={"type": "json_object"},
        )
        return json.loads(resp.choices[0].message.content or "{}")
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


async def _capture_page_html(browser) -> str:
    try:
        page = await browser.get_current_page()
        if page is None:
            return ""
        return await page.evaluate("(()=>document.documentElement.outerHTML)")
    except Exception:
        return ""


def _build_price_task(job) -> str:
    from jlcpc_browser.context.part_brief import build_part_brief

    brief = (build_part_brief(job.pdf_path) or "")[:6_000]
    paths = "\n".join(f"  - {p.resolve()}" for p in job.companion_paths) or "  (none)"
    return _PRICE_TASK_TEMPLATE.format(
        url="https://cart.jlcpcb.com/quote",
        part_id=job.part_id,
        material=job.material or "(unknown)",
        form=job.manufacture_form or "(unknown)",
        brief=brief,
        paths=paths,
    )


async def _run_price_capture(job, browser=None, timeout: float = DEFAULT_QUOTE_TIMEOUT_S,
                             max_steps: int = 25) -> dict:
    """Drive browser-use with a minimal 'upload + wait for price' task, then
    grab HTML. Creates its own Browser unless one is supplied. Returns
    {part_id, html, timed_out, agent_error}."""
    from browser_use import Agent, Browser

    from jlcpc_browser.agent.runner import _build_chat_llm, build_browser_profile

    owns_browser = browser is None
    if owns_browser:
        browser = Browser(browser_profile=build_browser_profile(keep_alive=True))

    task = _build_price_task(job)
    paths = [str(p.resolve()) for p in job.companion_paths]
    model = os.environ.get("JLCPCB_LLM_MODEL", "openai/gpt-4o-mini")
    llm = _build_chat_llm(model)

    timed_out = False
    agent_error: str | None = None
    try:
        agent = Agent(
            task=task,
            llm=llm,
            available_file_paths=paths,
            initial_actions=[
                {"navigate": {"url": "https://cart.jlcpcb.com/quote", "new_tab": False}}
            ],
            browser=browser,
        )
        try:
            await asyncio.wait_for(agent.run(max_steps=max_steps), timeout=timeout)
        except asyncio.TimeoutError:
            timed_out = True
        except Exception as e:
            agent_error = f"{type(e).__name__}: {e}"

        html = await _capture_page_html(browser)
    finally:
        if owns_browser:
            try:
                await browser.kill()
            except Exception:
                pass

    return {
        "part_id": job.part_id,
        "html": html,
        "timed_out": timed_out,
        "agent_error": agent_error,
    }


# ---------------------------------------------------------------------------
# Skills
# ---------------------------------------------------------------------------


@skill(
    name="list_classified_parts",
    description=(
        "List the parts discovered under `classified/1. Parts/` (the layout "
        "produced by `classify_parts_by_material`). Each entry has part_id, "
        "material, process, the PDF path, and companion CAD file paths. Use "
        "this to decide which parts to quote."
    ),
    parameters={"type": "object", "properties": {}, "required": []},
)
def list_classified_parts(session) -> dict:
    try:
        root = _upload_root(session)
    except ValueError as e:
        return {"error": str(e)}
    parts_root = root / CLASSIFIED_PARTS_SUBPATH
    if not parts_root.is_dir():
        return {
            "error": (f"No classified parts at {CLASSIFIED_PARTS_SUBPATH}. "
                      "Run `classify_parts_by_material` with dry_run=false first."),
        }

    # Walk the tree ourselves and build each job via `_job_for_pdf` so
    # companion matching uses the normalized identifier (tolerates
    # '...P050 [Desc]_R00.pdf' vs '...P050[Desc]_R00.step' mismatches).
    parts: list[dict] = []
    for mat_dir in sorted(parts_root.iterdir()):
        if not mat_dir.is_dir() or mat_dir.name.startswith("."):
            continue
        for proc_dir in sorted(mat_dir.iterdir()):
            if not proc_dir.is_dir() or proc_dir.name.startswith("."):
                continue
            for pdf in sorted(proc_dir.glob("*.pdf")):
                j = _job_for_pdf(pdf, root)
                parts.append({
                    "part_id": j.part_id,
                    "material": _strip_enum(j.material),
                    "process": _strip_enum(j.manufacture_form),
                    "pdf": str(j.pdf_path.relative_to(root)),
                    "companions": [str(p.relative_to(root)) for p in j.companion_paths],
                })
    return {
        "parts_root": str(parts_root.relative_to(root)),
        "count": len(parts),
        "parts": parts,
    }


@skill(
    name="quote_part",
    description=(
        "Get a JLCPCB instant-quote price for one part. Launches a browser, "
        "picks the right product line, uploads the CAD file, waits for the "
        "price to render, then STOPS. The raw page HTML is parsed by an LLM "
        "to extract price, currency, lead time, and product line. Does NOT "
        "fill Product desc, additional info, or Save to Cart. Pass the "
        "relative path of the drawing PDF under the uploaded folder. Before "
        "calling this, consider `ask_user_approval` unless the user "
        "explicitly asked to quote that exact file."
    ),
    parameters={
        "type": "object",
        "properties": {
            "relative_path": {
                "type": "string",
                "description": "Path to the drawing PDF relative to the uploaded folder.",
            },
        },
        "required": ["relative_path"],
    },
)
def quote_part(session, relative_path: str) -> dict:
    try:
        pdf = _resolve(session, relative_path)
    except ValueError as e:
        return {"error": str(e)}
    if pdf.suffix.lower() != ".pdf":
        return {"error": f"Not a PDF: {relative_path}"}

    root = _upload_root(session)
    raw_job = _job_for_pdf(pdf, root)

    if not raw_job.companion_paths:
        return {
            "part_id": raw_job.part_id,
            "ok": False,
            "error": (f"No companion CAD file found next to {pdf.name}. "
                      "Need a STEP/STL/DXF with a matching identifier."),
        }

    # Stage only what the browser uploads. Keep `raw_job` for the response so
    # the LLM conversation retains the ORIGINAL filenames (otherwise follow-up
    # turns would only see 'solid.step' and lose track of the user's files).
    staged_job = _stage_companions(raw_job, root)

    original_companions = [str(p.relative_to(root)) for p in raw_job.companion_paths]
    timeout = _quote_timeout_s()

    try:
        capture = asyncio.run(_run_price_capture(staged_job, timeout=timeout))
    except Exception as e:
        return {
            "part_id": raw_job.part_id,
            "ok": False,
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
        }

    price = _parse_price_html(capture.get("html") or "", raw_job)

    ts = _ts_slug()
    artifact = {
        "part_id": raw_job.part_id,
        "pdf": str(raw_job.pdf_path.relative_to(root)),
        "companions": original_companions,
        "uploaded_as": [p.name for p in staged_job.companion_paths],
        "timed_out": capture.get("timed_out"),
        "agent_error": capture.get("agent_error"),
        "timeout_s": timeout,
        "price": price,
        "html_len": len(capture.get("html") or ""),
    }
    artifact_url = None
    try:
        html_path = (
            root / QUOTES_CACHE_SUBPATH
            / f"{_safe_stem(raw_job.part_id)}-{ts}.html"
        )
        html_path.parent.mkdir(parents=True, exist_ok=True)
        html_path.write_text(capture.get("html") or "", encoding="utf-8")
        artifact["html_file"] = str(html_path.relative_to(root))
        _, artifact_url = _write_artifact(
            session, f"{_safe_stem(raw_job.part_id)}-{ts}.json", artifact
        )
    except Exception:
        pass

    price_visible = bool(price and price.get("price"))
    return {
        "part_id": raw_job.part_id,
        "ok": price_visible,
        "timed_out": bool(capture.get("timed_out")),
        "timeout_s": timeout,
        "pdf": str(raw_job.pdf_path.relative_to(root)),
        "companions": original_companions,
        "price": price,
        "agent_error": capture.get("agent_error"),
        "artifact_url": artifact_url,
    }


@skill(
    name="quote_all_classified",
    description=(
        "Get a JLCPCB instant-quote price for every part under "
        "`classified/1. Parts/`. Uses one shared Browser across all parts, "
        "driving each through the minimal 'upload + wait for price' task, "
        "then extracting the price from the raw page HTML via an LLM. Does "
        "NOT fill Product desc, additional info, or Save to Cart. "
        "Long-running (up to timeout_s per part, default 120s). ALWAYS "
        "call `ask_user_approval` first — present the part count and "
        "expected duration."
    ),
    parameters={"type": "object", "properties": {}, "required": []},
)
def quote_all_classified(session) -> dict:
    try:
        root = _upload_root(session)
    except ValueError as e:
        return {"error": str(e)}
    parts_root = root / CLASSIFIED_PARTS_SUBPATH
    if not parts_root.is_dir():
        return {
            "error": (f"No classified parts at {CLASSIFIED_PARTS_SUBPATH}. "
                      "Run `classify_parts_by_material` with dry_run=false first."),
        }

    raw_jobs: list = []
    for mat_dir in sorted(parts_root.iterdir()):
        if not mat_dir.is_dir() or mat_dir.name.startswith("."):
            continue
        for proc_dir in sorted(mat_dir.iterdir()):
            if not proc_dir.is_dir() or proc_dir.name.startswith("."):
                continue
            for pdf in sorted(proc_dir.glob("*.pdf")):
                raw_jobs.append(_job_for_pdf(pdf, root))

    if not raw_jobs:
        return {"error": "No PDFs found under classified/1. Parts/."}

    timeout = _quote_timeout_s()

    async def _run_all() -> list[dict]:
        from browser_use import Browser

        from jlcpc_browser.agent.runner import build_browser_profile

        browser = Browser(browser_profile=build_browser_profile(keep_alive=True))
        out: list[dict] = []
        try:
            for raw_job in raw_jobs:
                original_companions = [
                    str(p.relative_to(root)) for p in raw_job.companion_paths
                ]
                if not raw_job.companion_paths:
                    out.append({
                        "part_id": raw_job.part_id,
                        "ok": False,
                        "pdf": str(raw_job.pdf_path.relative_to(root)),
                        "companions": [],
                        "error": "no companion CAD file",
                    })
                    continue
                staged_job = _stage_companions(raw_job, root)
                try:
                    capture = await _run_price_capture(
                        staged_job, browser=browser, timeout=timeout
                    )
                    price = _parse_price_html(capture.get("html") or "", raw_job)
                    out.append({
                        "part_id": raw_job.part_id,
                        "ok": bool(price and price.get("price")),
                        "pdf": str(raw_job.pdf_path.relative_to(root)),
                        "companions": original_companions,
                        "timed_out": bool(capture.get("timed_out")),
                        "price": price,
                        "agent_error": capture.get("agent_error"),
                        "html_len": len(capture.get("html") or ""),
                    })
                except Exception as e:
                    out.append({
                        "part_id": raw_job.part_id,
                        "ok": False,
                        "pdf": str(raw_job.pdf_path.relative_to(root)),
                        "companions": original_companions,
                        "error": f"{type(e).__name__}: {e}",
                        "traceback": traceback.format_exc(),
                    })
        finally:
            try:
                await browser.kill()
            except Exception:
                pass
        return out

    try:
        records = asyncio.run(_run_all())
    except Exception as e:
        return {
            "error": f"Browser run aborted: {type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
        }

    succeeded = sum(1 for r in records if r.get("ok"))
    timed_out = sum(1 for r in records if r.get("timed_out"))
    summary = {
        "ok": succeeded == len(records),
        "part_count": len(records),
        "succeeded": succeeded,
        "failed": len(records) - succeeded,
        "timed_out": timed_out,
        "timeout_s": timeout,
        "parts": [
            {
                "part_id": r.get("part_id"),
                "ok": r.get("ok"),
                "timed_out": bool(r.get("timed_out")),
                "price": (r.get("price") or {}).get("price"),
                "error": r.get("error") or r.get("agent_error"),
            }
            for r in records
        ],
    }

    try:
        _, url = _write_artifact(session, f"run-{_ts_slug()}.json",
                                 {**summary, "records": records})
        summary["artifact_url"] = url
    except Exception:
        summary["artifact_url"] = None
    return summary
