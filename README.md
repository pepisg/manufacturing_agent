# Manufacturing Agent

**Turn a CAD folder into manufacturing quotes without leaving the chat.** Drop your SolidWorks project in, and the agent reads the drawings, sorts parts by material and process, drives a real browser to get live JLCPCB quotes, and finds alternative shops for anything JLC won't make.

## What it does

1. **Upload** — drop a project folder (parts, drawing PDFs, STEP exports) into the browser.
2. **Inspect** — parses STEP files with OpenCascade to pull out assembly structure, part names, volumes, and bounding boxes. Modern `.SLDPRT/.SLDASM` are encrypted, so STEP (AP214/AP242) is required.
3. **Read drawings** — extracts Material and Manufacturing process from PDF title blocks via an LLM.
4. **Classify** — groups parts into `classified/<material>/<process>/` so each bucket can be routed to the right shop.
5. **Quote** — drives a real browser on `cart.jlcpcb.com` to quote each part, or searches the web for alternative shops when JLCPCB can't handle the material/process.
6. **View** — renders PDF drawings (pdf.js) and STEP models (three.js + occt-import-js) inline in the chat.

## Stack

- **Backend** — FastAPI (`backend/main.py`) + OpenRouter tool-calling loop (`backend/agent.py`). Each tool the LLM can call is a `@skill` in `backend/skills/`.
- **Frontend** — vanilla JS chat UI with folder drop, Yes/No approval gates for destructive actions, and inline viewers.
- **Models** — Claude Opus/Sonnet/Haiku 4.x, GPT-5/mini, GPT-4o/mini (picked from a dropdown).

## Run

```bash
cp env.sh.example env.sh          # fill in OPENROUTER_API_KEY
source env.sh
.venv/bin/uvicorn backend.main:app --reload
```

Then open `http://localhost:8000`.
