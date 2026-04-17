from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class PartJob:
    """One manufacturable part: PDF blueprint + same-basename companion files."""

    part_id: str
    pdf_path: Path
    companion_paths: tuple[Path, ...]
    material: str
    manufacture_form: str
    source_dir: Path


@dataclass
class DiscoveryResult:
    jobs: list[PartJob]
    warnings: list[str] = field(default_factory=list)


def _stem_matches(pdf_stem: str, path: Path) -> bool:
    return path.stem.casefold() == pdf_stem.casefold()


def discover_process_dir(process_dir: Path) -> DiscoveryResult:
    """
    Scan a single parts/<material>/<manufacture_form>/ directory (flat layout).
    Each *.pdf defines a part; non-PDF files with the same basename are companions.
    """
    warnings: list[str] = []
    process_dir = process_dir.resolve()
    if not process_dir.is_dir():
        return DiscoveryResult([], warnings=[f"Not a directory: {process_dir}"])

    parts_parent = process_dir.parent
    material = parts_parent.name
    manufacture_form = process_dir.name

    pdfs = sorted(process_dir.glob("*.pdf"))
    pdf_stems = {p.stem.casefold() for p in pdfs}
    if len(pdfs) != len(pdf_stems):
        warnings.append(f"Duplicate PDF stems in {process_dir}")

    non_pdf = [p for p in process_dir.iterdir() if p.is_file() and p.suffix.lower() != ".pdf"]
    jobs: list[PartJob] = []

    for pdf in sorted(pdfs, key=lambda p: p.name.casefold()):
        stem = pdf.stem
        companions = tuple(
            sorted(
                (p for p in non_pdf if _stem_matches(stem, p)),
                key=lambda p: p.name.casefold(),
            )
        )
        jobs.append(
            PartJob(
                part_id=stem,
                pdf_path=pdf,
                companion_paths=companions,
                material=material,
                manufacture_form=manufacture_form,
                source_dir=process_dir,
            )
        )

    for p in non_pdf:
        if p.stem.casefold() not in pdf_stems:
            warnings.append(f"Orphan file (no matching PDF stem): {p.name}")

    return DiscoveryResult(jobs=jobs, warnings=warnings)


def discover_parts_tree(parts_root: Path) -> DiscoveryResult:
    """
    Walk parts/<material>/<manufacture_form>/ and aggregate PartJobs.
    """
    parts_root = parts_root.resolve()
    if not parts_root.is_dir():
        return DiscoveryResult([], warnings=[f"Parts root missing: {parts_root}"])

    all_warnings: list[str] = []
    all_jobs: list[PartJob] = []

    for material_dir in sorted(parts_root.iterdir()):
        if not material_dir.is_dir() or material_dir.name.startswith("."):
            continue
        for process_dir in sorted(material_dir.iterdir()):
            if not process_dir.is_dir() or process_dir.name.startswith("."):
                continue
            sub = discover_process_dir(process_dir)
            all_jobs.extend(sub.jobs)
            all_warnings.extend(sub.warnings)

    return DiscoveryResult(jobs=all_jobs, warnings=all_warnings)
