from __future__ import annotations

import argparse
import asyncio
import json
import sys
import traceback
from pathlib import Path

from jlcpc_browser.discovery import discover_parts_tree
from jlcpc_browser.agent.runner import dry_run_record, run_quote_agent, write_json


def _eprint(*args: object) -> None:
    print(*args, file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="JLCPCB quote agent: scan parts/, build PDF context, run browser-use per part.",
    )
    parser.add_argument(
        "--parts-root",
        type=Path,
        default=Path("parts"),
        help="Root containing parts/<material>/<manufacture_form>/ (default: ./parts)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only discover parts and extract PDF text; do not launch the browser agent.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/run_summary.json"),
        help="Write run summary JSON here (default: artifacts/run_summary.json)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=80,
        help="browser-use max_steps per part (default: 80)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="LLM model id (default: env JLCPCB_LLM_MODEL or gpt-4o-mini)",
    )
    parser.add_argument(
        "--no-shared-browser",
        action="store_true",
        help="Start a new browser for each part (cart will not accumulate). Default: one shared browser for all parts.",
    )
    args = parser.parse_args()

    if not args.dry_run and sys.version_info < (3, 11):
        _eprint("Agent mode requires Python 3.11+ (browser-use). Use --dry-run, or upgrade Python.")
        sys.exit(2)

    discovered = discover_parts_tree(args.parts_root)
    for w in discovered.warnings:
        _eprint("warning:", w)

    if not discovered.jobs:
        _eprint("No parts found (need parts/<material>/<manufacture_form>/*.pdf).")
        write_json(
            args.output,
            {"ok": False, "error": "no_jobs", "warnings": discovered.warnings},
        )
        sys.exit(1)

    parts_out: list[dict] = []

    if args.dry_run:
        for job in discovered.jobs:
            parts_out.append(dry_run_record(job))
        write_json(
            args.output,
            {
                "ok": True,
                "mode": "dry_run",
                "parts_root": str(args.parts_root.resolve()),
                "warnings": discovered.warnings,
                "parts": parts_out,
            },
        )
        print(json.dumps({"written": str(args.output), "parts": len(parts_out)}, indent=2))
        return

    shared_browser = not args.no_shared_browser

    async def _run_all() -> None:
        browser = None
        if shared_browser:
            from browser_use import Browser

            from jlcpc_browser.agent.runner import build_browser_profile

            browser = Browser(browser_profile=build_browser_profile(keep_alive=True))

        try:
            for job in discovered.jobs:
                try:
                    rec = await run_quote_agent(
                        job,
                        max_steps=args.max_steps,
                        model=args.model,
                        browser=browser,
                    )
                    parts_out.append(rec)
                except Exception as e:
                    parts_out.append(
                        {
                            "part_id": job.part_id,
                            "ok": False,
                            "error": str(e),
                            "traceback": traceback.format_exc(),
                        }
                    )
        finally:
            if browser is not None:
                try:
                    await browser.kill()
                except Exception:
                    pass

    asyncio.run(_run_all())

    write_json(
        args.output,
        {
            "ok": all(p.get("ok") for p in parts_out if isinstance(p, dict)),
            "mode": "agent",
            "shared_browser": shared_browser,
            "parts_root": str(args.parts_root.resolve()),
            "warnings": discovered.warnings,
            "parts": parts_out,
        },
    )
    print(json.dumps({"written": str(args.output), "parts": len(parts_out)}, indent=2))


if __name__ == "__main__":
    main()
