"""CAD skills for the Linux agent.

Modern SolidWorks files (2015+) are encrypted and cannot be parsed on Linux.
The workflow is: the user exports their assembly from SolidWorks to STEP
(AP214 or AP242), and these skills read that STEP with OpenCascade (OCP)
via the XDE (Extended Data Exchange) API so assembly structure and names
are preserved.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps
from OCP.IFSelect import IFSelect_RetDone
from OCP.STEPCAFControl import STEPCAFControl_Reader
from OCP.TCollection import TCollection_AsciiString, TCollection_ExtendedString
from OCP.TDataStd import TDataStd_Name
from OCP.TDF import TDF_Label, TDF_LabelSequence
from OCP.TDocStd import TDocStd_Document
from OCP.TopoDS import TopoDS_Shape
from OCP.XCAFApp import XCAFApp_Application
from OCP.XCAFDoc import XCAFDoc_DocumentTool, XCAFDoc_ShapeTool

from . import skill

STEP_EXTS = {".step", ".stp"}
SW_EXTS = {".sldprt", ".sldasm", ".slddrw"}


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


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------


# Extensions we surface in list_cad_files, grouped by bucket key.
_LIST_BUCKETS: dict[str, set[str]] = {
    "pdf":  {".pdf"},
    "step": {".step", ".stp"},
    "stl":  {".stl"},
    "dxf":  {".dxf"},
}
# Ignore anything under these folders — they're agent-generated artifacts.
_LIST_SKIP_DIRS = {"classified", ".cache", ".previews"}


@skill(
    name="list_cad_files",
    description=(
        "List the manufacturing files in the uploaded folder, grouped by "
        "type: PDF drawings, STEP (.step/.stp), STL meshes, and DXF flats. "
        "Skips agent-generated output folders (classified/, .cache/). Call "
        "this first to discover what's available."
    ),
    parameters={"type": "object", "properties": {}, "required": []},
)
def list_cad_files(session) -> dict:
    root = session.upload_dir
    if not root or not os.path.isdir(root):
        return {"error": "No folder has been uploaded for this session."}
    root_path = Path(root).resolve()

    groups: dict[str, list[dict]] = {k: [] for k in _LIST_BUCKETS}
    for dirpath, dirnames, filenames in os.walk(root_path):
        # Prune skip-dirs in place so os.walk doesn't descend into them.
        dirnames[:] = [d for d in dirnames if d not in _LIST_SKIP_DIRS]
        for name in filenames:
            ext = os.path.splitext(name)[1].lower()
            for bucket, exts in _LIST_BUCKETS.items():
                if ext in exts:
                    full = os.path.join(dirpath, name)
                    groups[bucket].append({
                        "path": os.path.relpath(full, root_path),
                        "size": os.path.getsize(full),
                    })
                    break
    for lst in groups.values():
        lst.sort(key=lambda x: x["path"])

    return {
        "root": str(root_path),
        **groups,
        "counts": {k: len(v) for k, v in groups.items()},
    }


# ---------------------------------------------------------------------------
# STEP parsing (XDE)
# ---------------------------------------------------------------------------


def _load_step(path: Path) -> tuple[TDocStd_Document, XCAFDoc_ShapeTool]:
    app = XCAFApp_Application.GetApplication_s()
    doc = TDocStd_Document(TCollection_ExtendedString("XmlXCAF"))
    app.NewDocument(TCollection_ExtendedString("MDTV-XCAF"), doc)

    reader = STEPCAFControl_Reader()
    reader.SetNameMode(True)
    reader.SetColorMode(True)
    reader.SetLayerMode(True)
    status = reader.ReadFile(str(path))
    if status != IFSelect_RetDone:
        raise RuntimeError(f"STEP reader failed (status={status}) on {path.name}")
    if not reader.Transfer(doc):
        raise RuntimeError(f"STEP transfer failed on {path.name}")

    shape_tool = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
    return doc, shape_tool


def _label_name(label: TDF_Label) -> str:
    attr = TDataStd_Name()
    if label.FindAttribute(TDataStd_Name.GetID_s(), attr):
        return attr.Get().ToExtString()
    return ""


def _volume_and_bbox(shape: TopoDS_Shape) -> dict[str, Any]:
    out: dict[str, Any] = {}
    try:
        props = GProp_GProps()
        BRepGProp.VolumeProperties_s(shape, props)
        # OCP reports in the STEP file's native units (usually mm^3).
        out["volume_mm3"] = round(props.Mass(), 3)
    except Exception:
        pass
    try:
        from OCP.Bnd import Bnd_Box
        from OCP.BRepBndLib import BRepBndLib
        box = Bnd_Box()
        BRepBndLib.Add_s(shape, box)
        xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
        out["bbox_mm"] = {
            "x": round(xmax - xmin, 3),
            "y": round(ymax - ymin, 3),
            "z": round(zmax - zmin, 3),
        }
    except Exception:
        pass
    return out


def _walk(
    shape_tool: XCAFDoc_ShapeTool,
    label: TDF_Label,
    depth: int,
    with_geom: bool,
) -> dict[str, Any]:
    node: dict[str, Any] = {
        "name": _label_name(label) or "(unnamed)",
        "is_assembly": bool(shape_tool.IsAssembly_s(label)),
        "is_reference": bool(shape_tool.IsReference_s(label)),
    }

    # If this label is a reference (component instance), resolve to the
    # referred prototype and walk that, keeping this instance's name.
    if shape_tool.IsReference_s(label):
        ref = TDF_Label()
        if shape_tool.GetReferredShape_s(label, ref):
            referred = _walk(shape_tool, ref, depth + 1, with_geom)
            proto_name = referred.get("name")
            node["component_of"] = proto_name
            # STEP instance names often look like "=>[0:1:1:2]" — prefer the
            # prototype's human name for display & BOM keying.
            if not node["name"] or node["name"].startswith("=>"):
                node["name"] = proto_name or node["name"]
            node["is_assembly"] = referred.get("is_assembly", False)
            if referred.get("children"):
                node["children"] = referred["children"]
            if "geometry" in referred:
                node["geometry"] = referred["geometry"]
        return node

    if shape_tool.IsAssembly_s(label):
        children_seq = TDF_LabelSequence()
        shape_tool.GetComponents_s(label, children_seq, False)
        children = []
        for i in range(1, children_seq.Length() + 1):
            children.append(_walk(shape_tool, children_seq.Value(i), depth + 1, with_geom))
        node["children"] = children
    else:
        node["kind"] = "part"
        if with_geom:
            try:
                shape = shape_tool.GetShape_s(label)
                if not shape.IsNull():
                    node["geometry"] = _volume_and_bbox(shape)
            except Exception as e:
                node["geometry_error"] = f"{type(e).__name__}: {e}"

    return node


def _flatten_bom(node: dict[str, Any], counts: dict[str, int]) -> None:
    if node.get("kind") == "part" or (
        node.get("is_reference") and not node.get("children")
    ):
        name = node.get("component_of") or node.get("name") or "(unnamed)"
        counts[name] = counts.get(name, 0) + 1
        return
    for c in node.get("children", []):
        _flatten_bom(c, counts)


@skill(
    name="inspect_step",
    description=(
        "Parse a STEP file (.step/.stp) and return the assembly hierarchy "
        "with component names, plus per-part volume and bounding box. Use "
        "this for any question about what's in an assembly, how parts are "
        "organized, or how big parts are."
    ),
    parameters={
        "type": "object",
        "properties": {
            "relative_path": {
                "type": "string",
                "description": "Path to the STEP file relative to the uploaded folder.",
            },
            "include_geometry": {
                "type": "boolean",
                "description": "Compute volume + bbox per part (slower on large files). Default true.",
            },
        },
        "required": ["relative_path"],
    },
)
def inspect_step(
    session, relative_path: str, include_geometry: bool = True
) -> dict:
    path = _resolve(session, relative_path)
    if path.suffix.lower() not in STEP_EXTS:
        return {"error": f"Not a STEP file: {relative_path}"}

    doc, shape_tool = _load_step(path)
    roots = TDF_LabelSequence()
    shape_tool.GetFreeShapes(roots)

    tree: list[dict[str, Any]] = []
    for i in range(1, roots.Length() + 1):
        tree.append(_walk(shape_tool, roots.Value(i), 0, include_geometry))

    bom: dict[str, int] = {}
    for r in tree:
        _flatten_bom(r, bom)

    return {
        "file": relative_path,
        "roots": tree,
        "bom": [{"name": n, "quantity": q} for n, q in sorted(bom.items())],
        "unique_parts": len(bom),
        "total_instances": sum(bom.values()),
    }
