"""Skills for reading mechanical drawing PDFs and classifying parts by
material + manufacturing process.

We send the raw page-1 text of each drawing PDF to an LLM and let it pull
out title-block fields and classify the material. Layouts vary (A4, A3, with
or without a notes panel, different label orderings) and the LLM handles
them all; coordinate heuristics are fragile and were removed.
"""
from __future__ import annotations

import json
import os
import re
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pdfplumber

from . import skill

TITLE_BLOCK_FIELDS = (
    "material", "description", "manufacturing_process", "code",
    "finish", "weight_kg", "version", "material_class",
)

_VALID_CLASSES = {"Metal", "Plastic", "Other"}

_TITLE_BLOCK_PROMPT = (
    "You extract fields from a mechanical-engineering drawing PDF's title "
    "block. The input is the raw text of the first page of that PDF — it "
    "comes from a layout-preserving extractor, so columns may be jumbled "
    "and there is a lot of unrelated text from tolerance tables, revision "
    "panels, and ISO-5457 stamps. Find these fields and return them as "
    "strict JSON (no prose, no markdown):\n"
    "  material, description, manufacturing_process, code, finish, "
    "weight_kg, version, material_class.\n"
    "\n"
    "material — the raw material spec (e.g. 'PA Type 6', 'Galvanized Steel', "
    "'6061 Alloy', 'NBR 90 shore A'). Preserve the source spelling.\n"
    "manufacturing_process — e.g. FDM, CNC, Laser Cut, Bending Sheet Metal, "
    "Silicone moulded, SLA, Injection Molding. Preserve source spelling.\n"
    "code — the part number, typically starting with a project code like "
    "'K25-09.P041' or 'K25-01.W002'.\n"
    "weight_kg — a string matching the printed value (keep decimals as-is).\n"
    "description — short description from the title block.\n"
    "\n"
    "material_class MUST be exactly one of: 'Metal', 'Plastic', 'Other'. "
    "'Plastic' covers polymers, thermoplastics, rubbers, silicones, "
    "elastomers, and 3D-printing resins. 'Other' is wood, ceramic, glass, "
    "fabric, composite, truly unknown, or absent. Known codes: 6061 / 7075 "
    "/ 5052 / 3003 / 1060 aluminum = Metal; 304 / 316 / AISI 1020 / 1045 / "
    "Galvanized Steel = Metal; NBR / SBR / EPDM / Viton = Plastic; PA / "
    "ABS / PLA / PC / PET / PETG / HIPS = Plastic.\n"
    "\n"
    "If a field is not present in the title block, return null for that "
    "field. Never invent values. The title block usually sits at the "
    "bottom of the page."
)

# Module-level thread-pool for parallel LLM calls. 8 workers is enough for
# the typical folder size (<~50 parts) without hammering rate limits.
_LLM_POOL = ThreadPoolExecutor(max_workers=8, thread_name_prefix="pdf-llm")


def _resolve(session, relative_path: str) -> Path:
    if not session.upload_dir:
        raise ValueError("No folder has been uploaded for this session.")
    root = Path(session.upload_dir).resolve()
    target = (root / relative_path).resolve()
    if root not in target.parents and target != root:
        raise ValueError("Path escapes upload directory.")
    if not target.is_file():
        raise ValueError(f"Not a file: {relative_path}")
    return target


def _extract_pdf_text(path: Path) -> str:
    """Page-1 layout-preserving text. Title blocks always fit on page 1."""
    with pdfplumber.open(str(path)) as pdf:
        if not pdf.pages:
            return ""
        return pdf.pages[0].extract_text(layout=True) or pdf.pages[0].extract_text() or ""


def _empty_title_block() -> dict[str, Any]:
    return {f: None for f in TITLE_BLOCK_FIELDS}


def _extract_title_block(path: Path) -> dict[str, Any]:
    text = _extract_pdf_text(path)
    if not text.strip():
        return _empty_title_block()

    # Late import avoids a package-level cycle between skills and the agent.
    from .. import agent

    try:
        client = agent._client()
    except RuntimeError:
        return _empty_title_block()

    # Layout-preserving extraction is very whitespace-heavy; the title block
    # always sits at the bottom of the page, so send the tail, not the head.
    tail = text[-16_000:] if len(text) > 16_000 else text
    try:
        resp = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[
                {"role": "system", "content": _TITLE_BLOCK_PROMPT},
                {"role": "user", "content": tail},
            ],
            response_format={"type": "json_object"},
        )
        parsed = json.loads(resp.choices[0].message.content or "{}")
    except Exception:
        return _empty_title_block()

    out = _empty_title_block()
    for f in TITLE_BLOCK_FIELDS:
        v = parsed.get(f)
        if isinstance(v, (int, float)):
            v = str(v)
        if isinstance(v, str) and v.strip():
            out[f] = v.strip()
    # Never trust a class the model invented without a material string behind
    # it — welding assemblies often have no Material field and the model will
    # happily guess "Plastic" anyway.
    if not out["material"]:
        out["material_class"] = UNKNOWN_MATERIAL
    elif out["material_class"] not in _VALID_CLASSES:
        out["material_class"] = _material_class_fallback(out["material"])
    return out


@skill(
    name="read_drawing_pdf",
    description=(
        "Extract the manufacturing title-block fields from a mechanical "
        "drawing PDF: material, description, manufacturing process, code, "
        "finish, weight (kg), and version."
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
def read_drawing_pdf(session, relative_path: str) -> dict:
    path = _resolve(session, relative_path)
    if path.suffix.lower() != ".pdf":
        return {"error": f"Not a PDF: {relative_path}"}
    fields = _extract_title_block(path)
    return {"file": relative_path, **fields}


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

UNKNOWN_MATERIAL = "Other"
MATERIAL_CLASSES = ("Metal", "Plastic", UNKNOWN_MATERIAL)

# Part-file extensions we want to copy along with each part.
PART_EXTS = {".step", ".stp", ".stl", ".dxf", ".pdf", ".iges", ".igs"}

# Keyword fallback for when the LLM is unavailable or returns garbage.
# Intentionally permissive; prefer the LLM path.
_METAL_KEYWORDS = (
    "steel", "stainless", "inox", "aluminum", "aluminium", "alu ", "alu-",
    "brass", "copper", "bronze", "titanium", "iron", "zinc", "galvanized",
    "alloy", "a36", "6061", "7075", "5052", "3003", "304", "316",
)
_PLASTIC_KEYWORDS = (
    "pa ", "pa6", "pa12", "nylon", "abs", "pla", "pet", "petg", "polycarbonate",
    "pc ", "polypropylene", "pp ", "polyethylene", "pe ", "hdpe", "ldpe",
    "polystyrene", "ps ", "pvc", "acrylic", "pmma", "tpu", "tpe",
    "polyurethane", "delrin", "pom", "acetal", "peek", "resin", "silicone",
    "silicon", "rubber", "elastomer", "plastic",
    "nbr", "epdm", "neoprene", "viton", "fkm", "hnbr", "shore",
)


def _material_class_fallback(material: str | None) -> str:
    if not material:
        return UNKNOWN_MATERIAL
    text = f" {material.lower()} "
    if any(k in text for k in _METAL_KEYWORDS):
        return "Metal"
    if any(k in text for k in _PLASTIC_KEYWORDS):
        return "Plastic"
    return UNKNOWN_MATERIAL




def _process_folder(process: str | None) -> str:
    if not process:
        return "Unknown"
    # Title-case while preserving known acronyms.
    ACRONYMS = {"FDM", "CNC", "SLA", "SLS", "MJF", "DMLS", "EDM"}
    parts = []
    for token in process.split():
        up = token.upper()
        if up in ACRONYMS:
            parts.append(up)
        else:
            parts.append(token.capitalize())
    return " ".join(parts) or "Unknown"


def _part_identifier(filename: str) -> str:
    """The code-prefix of a part file, used to associate siblings that differ
    only by extension or by whitespace before the description bracket.

    Examples:
      'K25-01.W002 [Base Main Structure]_R00.step'  -> 'k25-01.w002'
      'K25-01.W002[Base Main Structure]_R00.step'   -> 'k25-01.w002'
      'K25-01.W002-P07 [Base Main Structure]_R00.STEP' -> 'k25-01.w002-p07'
    """
    stem = os.path.splitext(filename)[0]
    # Description bracket — with or without a leading space — ends the code.
    i = stem.find("[")
    if i >= 0:
        stem = stem[:i]
    # Drop a trailing '_R<nn>' revision tag if there's nothing more structural.
    stem = re.sub(r"_R\d+\s*$", "", stem)
    return stem.strip().lower()


def _match_pdf(identifier: str, pdf_index: dict[str, Path]) -> str | None:
    """Return the best PDF identifier for a given file identifier.

    Direct match first. Otherwise fall back to the longest PDF identifier
    that is a prefix (so welding sub-parts like 'k25-01.w002-p07' inherit
    from 'k25-01.w002')."""
    if identifier in pdf_index:
        return identifier
    candidates = [k for k in pdf_index
                  if identifier.startswith(k + "-") or identifier.startswith(k + ".")]
    if not candidates:
        return None
    return max(candidates, key=len)


def _enumerate_folder(name: str, prefix_map: dict[str, int]) -> str:
    idx = prefix_map.setdefault(name, len(prefix_map))
    return f"{idx}. {name}"


CLASSIFIED_DIRNAME = "classified"


@skill(
    name="classify_parts_by_material",
    description=(
        "Walk the uploaded folder, read each part's drawing PDF to determine "
        "its material (plastic / metal / other) and manufacturing process, "
        "then COPY all associated files (STEP, STL, DXF, PDF) into a fresh "
        "`classified/1. Parts/<N. Material>/<N. Process>/` tree inside the "
        "upload folder. The original files are never moved or modified. Pass "
        "dry_run=true (default) to preview the plan; dry_run=false to "
        "actually copy files. Before switching to dry_run=false, always call "
        "`ask_user_approval` with the proposed layout so the user can "
        "confirm via the Yes / No buttons in the UI."
    ),
    parameters={
        "type": "object",
        "properties": {
            "dry_run": {
                "type": "boolean",
                "description": "If true (default), return the plan without copying.",
            },
        },
        "required": [],
    },
)
def classify_parts_by_material(session, dry_run: bool = True) -> dict:
    root = Path(session.upload_dir or "").resolve()
    if not root.is_dir():
        return {"error": "No folder has been uploaded for this session."}

    classified_root = root / CLASSIFIED_DIRNAME

    # Walk the upload, collecting (path, ext) for each part file; skip any
    # previously-generated classified/ tree so re-runs are idempotent.
    all_files: list[Path] = []
    for dirpath, _, filenames in os.walk(root):
        try:
            rel = Path(dirpath).relative_to(root)
        except ValueError:
            continue
        if rel.parts and rel.parts[0] == CLASSIFIED_DIRNAME:
            continue
        for name in filenames:
            if os.path.splitext(name)[1].lower() in PART_EXTS:
                all_files.append(Path(dirpath) / name)

    pdfs = [p for p in all_files if p.suffix.lower() == ".pdf"]

    # Fan out the LLM title-block extractions across the worker pool so a
    # folder of ~30 PDFs finishes in seconds, not minutes.
    pdf_index: dict[str, Path] = {}
    pdf_fields: dict[str, dict[str, Any]] = {}
    unusable_pdfs: list[str] = []
    futures = {_LLM_POOL.submit(_extract_title_block, pdf): pdf for pdf in pdfs}
    for fut in as_completed(futures):
        pdf = futures[fut]
        rel = pdf.relative_to(root)
        try:
            fields = fut.result()
        except Exception as e:
            unusable_pdfs.append(f"{rel}: {type(e).__name__}: {e}")
            continue
        # Keep any PDF that yielded at least one identifying field. Welding
        # assemblies frequently have no material but do have code/description.
        if not any(fields.get(f) for f in TITLE_BLOCK_FIELDS
                   if f != "material_class"):
            unusable_pdfs.append(f"{rel}: no title block found")
            continue
        ident = _part_identifier(pdf.name)
        pdf_index[ident] = pdf
        pdf_fields[ident] = fields

    # Bucket every non-PDF file under its matching PDF identifier, plus the
    # PDF itself. Welding sub-parts ('...-P07') inherit their parent's PDF.
    buckets: dict[str, list[Path]] = {ident: [pdf] for ident, pdf in pdf_index.items()}
    orphans: list[Path] = []
    for f in all_files:
        if f.suffix.lower() == ".pdf":
            continue
        key = _match_pdf(_part_identifier(f.name), pdf_index)
        if key is None:
            orphans.append(f)
        else:
            buckets[key].append(f)

    material_idx: dict[str, int] = {}
    process_idx: dict[str, dict[str, int]] = {}
    for seed in MATERIAL_CLASSES:
        material_idx[seed] = len(material_idx)

    plan: list[dict] = []
    for key in sorted(buckets):
        fields = pdf_fields[key]
        material_cls = fields.get("material_class") or UNKNOWN_MATERIAL
        process_raw = fields.get("manufacturing_process")
        process = _process_folder(process_raw)

        mat_folder = _enumerate_folder(material_cls, material_idx)
        proc_map = process_idx.setdefault(material_cls, {})
        proc_folder = _enumerate_folder(process, proc_map)

        dest_dir = classified_root / "1. Parts" / mat_folder / proc_folder
        plan.append({
            "part": key,
            "material_raw": fields.get("material"),
            "material_class": material_cls,
            "process_raw": process_raw,
            "process_folder": proc_folder,
            "dest_dir": str(dest_dir.relative_to(root)),
            "files": [str(p.relative_to(root)) for p in buckets[key]],
        })

    summary = {
        "dry_run": dry_run,
        "root": str(root),
        "output_dir": str(classified_root.relative_to(root)),
        "part_count": len(plan),
        "file_count": sum(len(e["files"]) for e in plan),
        "unusable_pdfs": unusable_pdfs,
        "orphan_files": [str(p.relative_to(root)) for p in orphans],
        "plan": plan,
    }

    if dry_run:
        return summary

    # Fresh output tree per run.
    if classified_root.exists():
        shutil.rmtree(classified_root)

    copies: list[dict] = []
    for entry in plan:
        dest = root / entry["dest_dir"]
        dest.mkdir(parents=True, exist_ok=True)
        for rel in entry["files"]:
            src = root / rel
            tgt = dest / src.name
            shutil.copy2(src, tgt)
            copies.append({"file": rel, "copied_to": str(tgt.relative_to(root))})
    summary["copies"] = copies
    return summary
