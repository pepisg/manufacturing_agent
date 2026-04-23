"""Skill for sourcing manufacturing providers for a classified part.

Given a part file and its material + process classification, call an
OpenRouter `:online` model so the LLM runs a live web search (via Exa) and
returns a structured shortlist of shops that can produce it.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import skill


_PROVIDER_PROMPT = (
    "You are a sourcing assistant that finds real manufacturing shops for a "
    "single mechanical part. You have live web search; use it. Return ONLY a "
    "JSON object with this shape — no prose, no markdown:\n"
    "{\n"
    '  "query": "<the search phrase you used>",\n'
    '  "providers": [\n'
    "    {\n"
    '      "name": "<company name>",\n'
    '      "url": "<homepage or quote page>",\n'
    '      "country": "<country or region>",\n'
    '      "capabilities": ["<process>", "<material>", ...],\n'
    '      "moq": "<minimum order if known, else null>",\n'
    '      "lead_time": "<typical lead time if known, else null>",\n'
    '      "instant_quote": true | false | null,\n'
    '      "summary": "<one or two sentences on why this shop fits>"\n'
    "    }\n"
    "  ]\n"
    "}\n"
    "\n"
    "Rules: every provider MUST come from a web result you actually saw; "
    "never invent companies or URLs. Prefer shops that explicitly list the "
    "requested process AND the material family. Return 4-8 providers, "
    "ordered best-fit first. If the user gave a country, prioritize shops "
    "that ship from or to it. If you genuinely cannot find good matches, "
    'return {"query": "...", "providers": []}.'
)


def _part_hint(session, relative_path: str | None) -> str:
    """Best-effort filename hint; we don't parse the file, just name-drop it."""
    if not relative_path:
        return ""
    try:
        root = Path(session.upload_dir or "").resolve()
        target = (root / relative_path).resolve()
        if root not in target.parents and target != root:
            return ""
        if not target.is_file():
            return ""
        return target.name
    except Exception:
        return ""


@skill(
    name="find_providers",
    description=(
        "ONLY use when the user EXPLICITLY asks to find alternative "
        "vendors / other shops / compare suppliers / source outside "
        "JLCPCB. DO NOT call this for 'quote', 'price', 'get a quote', or "
        "'how much' requests — those map to `quote_part` / "
        "`quote_all_classified` (JLCPCB browser automation). This skill "
        "runs a web search via OpenRouter `:online` and returns a "
        "shortlist of real manufacturing shops (name, URL, country, "
        "capabilities). Required: material and manufacturing_process."
    ),
    parameters={
        "type": "object",
        "properties": {
            "relative_path": {
                "type": "string",
                "description": "Path to the part file (STEP/PDF) relative to the upload folder. Optional — used only to label the result.",
            },
            "material": {
                "type": "string",
                "description": "Material spec, e.g. '6061 Aluminum', 'PA Type 6', 'NBR 70 shore A'.",
            },
            "manufacturing_process": {
                "type": "string",
                "description": "Process, e.g. 'CNC machining', 'FDM', 'Injection molding', 'Sheet metal bending'.",
            },
            "country": {
                "type": "string",
                "description": "Preferred shop country or region (e.g. 'USA', 'EU', 'China'). Optional.",
            },
            "quantity": {
                "type": "integer",
                "description": "Target quantity for the sourcing decision. Optional.",
            },
            "notes": {
                "type": "string",
                "description": "Extra constraints: tolerance, finish, bounding-box size, certifications. Optional.",
            },
        },
        "required": ["material", "manufacturing_process"],
    },
)
def find_providers(
    session,
    material: str,
    manufacturing_process: str,
    relative_path: str | None = None,
    country: str | None = None,
    quantity: int | None = None,
    notes: str | None = None,
) -> dict:
    # Late import to avoid a skills/agent package cycle (same pattern as drawings.py).
    from .. import agent

    try:
        client = agent._client()
    except RuntimeError as e:
        return {"error": str(e)}

    part_name = _part_hint(session, relative_path)

    user_lines = [
        f"Process: {manufacturing_process}",
        f"Material: {material}",
    ]
    if part_name:
        user_lines.append(f"Part file: {part_name}")
    if country:
        user_lines.append(f"Preferred region: {country}")
    if quantity:
        user_lines.append(f"Target quantity: {quantity}")
    if notes:
        user_lines.append(f"Notes: {notes}")
    user_lines.append("Find 4-8 real manufacturing shops that can produce this part.")

    try:
        resp = client.chat.completions.create(
            # `:online` suffix enables OpenRouter's built-in web search.
            model="openai/gpt-4o-mini:online",
            messages=[
                {"role": "system", "content": _PROVIDER_PROMPT},
                {"role": "user", "content": "\n".join(user_lines)},
            ],
        )
        raw = resp.choices[0].message.content or "{}"
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}

    # The :online path doesn't always honour response_format, so tolerate a
    # fenced block or leading prose and pull the first JSON object out.
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end < start:
        return {"error": "Model did not return JSON.", "raw": raw[:2000]}
    try:
        parsed: dict[str, Any] = json.loads(raw[start : end + 1])
    except json.JSONDecodeError as e:
        return {"error": f"Bad JSON from model: {e}", "raw": raw[:2000]}

    providers = parsed.get("providers") or []
    if not isinstance(providers, list):
        providers = []

    return {
        "part_file": relative_path,
        "part_name": part_name or None,
        "material": material,
        "manufacturing_process": manufacturing_process,
        "country": country,
        "quantity": quantity,
        "query": parsed.get("query"),
        "providers": providers,
    }
