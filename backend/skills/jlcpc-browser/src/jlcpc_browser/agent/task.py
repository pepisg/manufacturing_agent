from __future__ import annotations

from jlcpc_browser.context.part_brief import folder_intent_summary
from jlcpc_browser.discovery import PartJob


QUOTE_URL = "https://cart.jlcpcb.com/quote"


def build_quote_task(job: PartJob, part_brief_text: str) -> str:
    """
    Single self-contained instruction string for browser-use Agent.task.
    """
    intent = folder_intent_summary(job.material, job.manufacture_form)
    companions = "\n".join(f"  - {p}" for p in job.companion_paths) or "  (none)"

    companion_abs = "\n".join(f"  - {p.resolve()}" for p in job.companion_paths) or "  (none)"

    if job.companion_paths:
        upload_block = f"""### Step 3a — Upload **only what this product type needs** (usually **one** file)
The blueprint PDF is **not** uploaded. Allowed CAD paths (pick a **subset** — do **not** upload every companion file):

{companion_abs}

**Which file(s):**
- Default: upload **one** geometry file unless the form clearly has multiple required upload slots.
- **3D printing:** use **either** **.stl** **or** **.step** / **.stp** (or **both** only if the UI explicitly requires two uploads or separate fields). Prefer a single file that matches the drawing and the label on the upload control.
- **CNC / sheet metal / other:** upload **one** file in the format the active step asks for (e.g. one STEP), if present in the list above.
- **Never** upload the PDF. **Never** upload paths not listed above.

Use **upload / upload_file** (not typing paths into text fields)."""
    else:
        upload_block = """### Step 3a — Files on disk
There are **no** companion CAD files for this part. If the selected service **requires** a model file, stop and report — do not upload the PDF."""

    return f"""You are automating a **guest** instant quote on JLCPCB (no login, no payment).

Start at: {QUOTE_URL}

## Inputs (read before acting)
### Coarse routing (filesystem)
{intent}

### Part identity
- Part id (filename stem): `{job.part_id}`
- Blueprint PDF on disk (information only for you — **do not upload**): `{job.pdf_path}`
- Companion CAD files on disk (you will upload **only** what Step 3a specifies):
{companions}

### Drawing / specification text (from PDF — quantities, materials, process notes; PDF file itself is not uploaded)
{part_brief_text}

---

## Step 1 — Decide and open the **correct JLCPCB product** (do this first)
Before uploading files or filling service-specific forms, you must land in the right **top-level product** on the quote site. Examples: **3D Printing**, **CNC Machining**, **Sheet Metal**, **Standard PCB/PCBA**, **SMT Stencil** — pick what matches this job.

How to choose:
1. Use **folder intent** (`material` + `manufacture_form` above) as the primary hint (e.g. `plastic/3d` → additive / 3D printing path).
2. Use **drawing text** (material, manufacturing process callouts, title block) to confirm or refine (e.g. SLA, CNC, sheet metal).
3. If folder and drawing disagree, **prefer the drawing** for process specifics, but stay within a plausible interpretation of the folder.

Actions:
- Navigate the quote UI until you see the form for the **chosen** product line (not a different tab/service).
- **Do not** upload CAD or enter dimensions until Step 1 is satisfied and you are in that product’s quote flow.

State briefly which product you selected and why (one short sentence) before continuing.

---

## Step 2 — Sub-options inside that product (if the UI shows them)
Examples: 3D printing technology/material, CNC options, PCB layer count, etc. Match **drawing + folder** where the site allows.

---

## Step 3 — Upload, **Product desc**, then other fields
{upload_block}

### Step 3b — **Product desc** + additional info — do this **after** the file upload succeeds
After the CAD file is attached, find the **Product desc** control (and any **parent** chained dropdowns above it if the UI requires them first).

For **Product desc** (strict — the UI needs time after typing; **do not** rush):
1. **One step only:** type `others` into the Product desc field. **Stop.** Do **not** press Enter, do **not** click anything in this same step.
2. **Next step must be a real wait:** use whatever **wait / pause / sleep-for-seconds** tool your action space provides, for **at least 2.5 seconds**. This must be its **own** step — **never** bundle typing + waiting + clicking into one action list.
3. The suggestion list opens on its own — do **not** click the field again to “open” the dropdown before the list appears.
4. **Then** click the **others** / **Other** row in the visible list (explicit click only). If the field clears, repeat: type `others` → **separate** wait step (≥2.5s) → click **others**.

**After** that, a **text box for additional information** usually appears (or becomes active). Fill it with a **short, appropriate** description from the drawing: part title or description, material, and process note (e.g. SLA, CNC) — one or two sentences, plain text.

Then fill remaining required fields (qty, size, tolerance, etc.) per the drawing.

---

## Step 4 — Price and cart
1. Wait until uploads complete (no spinner; filenames visible if shown).
2. Use **Calculated price** if shown, then click **SAVE TO CART** / **Add to cart** (or equivalent). After a successful save, **stop** this task — do not keep exploring.

## Restrictions
Do **not** log in, **not** shipping/payment, **not** full checkout.

## Done when
Save-to-cart succeeded **or** hard blocker. Summarize: product chosen, file(s) uploaded (names), Product desc (type others → **separate** wait ≥2.5s → click others), additional-info text filled, cart clicked or not.
"""
