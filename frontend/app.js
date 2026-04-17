const logEl = document.getElementById("log");
const form = document.getElementById("form");
const input = document.getElementById("input");
const drop = document.getElementById("drop");
const folderInput = document.getElementById("folder");
const uploadStatus = document.getElementById("upload-status");
const modelSelect = document.getElementById("model");

let sessionId = null;

// PDF.js ships its worker separately; point at the matching CDN build.
if (window.pdfjsLib) {
  window.pdfjsLib.GlobalWorkerOptions.workerSrc =
    "https://cdn.jsdelivr.net/npm/pdfjs-dist@3.11.174/build/pdf.worker.min.js";
}

function addMsg(role, text) {
  const el = document.createElement("div");
  el.className = `msg ${role}`;
  el.textContent = text;
  logEl.appendChild(el);
  logEl.scrollTop = logEl.scrollHeight;
  return el;
}

function addApprovalButtons(msgEl) {
  const wrap = document.createElement("div");
  wrap.className = "approval";
  const yes = document.createElement("button");
  yes.type = "button";
  yes.textContent = "Yes";
  yes.className = "approve-yes";
  const no = document.createElement("button");
  no.type = "button";
  no.textContent = "No";
  no.className = "approve-no";
  wrap.appendChild(yes);
  wrap.appendChild(no);
  msgEl.appendChild(wrap);
  logEl.scrollTop = logEl.scrollHeight;

  const answer = (text) => {
    wrap.remove();
    sendChat(text);
  };
  yes.addEventListener("click", () => answer("Yes, proceed."));
  no.addEventListener("click", () => answer("No, don't proceed."));
}

async function init() {
  const [s, m] = await Promise.all([
    fetch("/api/session/new").then((r) => r.json()),
    fetch("/api/models").then((r) => r.json()),
  ]);
  sessionId = s.session_id;
  for (const { id, label } of m.models) {
    const opt = document.createElement("option");
    opt.value = id;
    opt.textContent = label;
    modelSelect.appendChild(opt);
  }
  modelSelect.value = m.default;
  addMsg("system", "Ready. Drop a folder on the left, then ask questions.");
}
init();

// --- Folder drag-and-drop + click-to-pick -----------------------------------

drop.addEventListener("click", () => folderInput.click());

drop.addEventListener("dragover", (e) => {
  e.preventDefault();
  drop.classList.add("hover");
});
drop.addEventListener("dragleave", () => drop.classList.remove("hover"));

drop.addEventListener("drop", async (e) => {
  e.preventDefault();
  drop.classList.remove("hover");
  const items = Array.from(e.dataTransfer.items)
    .map((i) => (i.webkitGetAsEntry ? i.webkitGetAsEntry() : null))
    .filter(Boolean);
  const collected = [];
  for (const entry of items) await walk(entry, "", collected);
  await upload(collected);
});

folderInput.addEventListener("change", async () => {
  const collected = Array.from(folderInput.files).map((f) => ({
    file: f,
    path: f.webkitRelativePath || f.name,
  }));
  await upload(collected);
});

function walk(entry, prefix, out) {
  return new Promise((resolve) => {
    const path = prefix ? `${prefix}/${entry.name}` : entry.name;
    if (entry.isFile) {
      entry.file((f) => {
        out.push({ file: f, path });
        resolve();
      });
    } else if (entry.isDirectory) {
      const reader = entry.createReader();
      const readBatch = () => {
        reader.readEntries(async (entries) => {
          if (entries.length === 0) return resolve();
          for (const e of entries) await walk(e, path, out);
          readBatch();
        });
      };
      readBatch();
    } else {
      resolve();
    }
  });
}

async function upload(collected) {
  if (!collected.length) return;
  uploadStatus.textContent = `Uploading ${collected.length} files…`;
  const fd = new FormData();
  fd.append("session_id", sessionId);
  for (const { file, path } of collected) {
    fd.append("files", file, file.name);
    fd.append("paths", path);
  }
  const res = await fetch("/api/upload", { method: "POST", body: fd });
  if (!res.ok) {
    uploadStatus.textContent = `Upload failed: ${res.status}`;
    return;
  }
  const data = await res.json();
  uploadStatus.textContent = `Uploaded ${data.saved} files.`;
  addMsg("system", `Uploaded ${data.saved} files.`);
}

// --- Chat -------------------------------------------------------------------

async function sendChat(text) {
  addMsg("user", text);
  const thinking = addMsg("assistant", "…");
  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: sessionId,
        message: text,
        model: modelSelect.value,
      }),
    });
    const data = await res.json();
    if (!res.ok) {
      thinking.textContent = `Error: ${data.detail || res.status}`;
      return;
    }
    thinking.textContent = data.reply || "(empty reply)";
    if (data.approval) addApprovalButtons(thinking);
    if (data.viewer) {
      const { kind, url, title } = data.viewer;
      if (kind === "pdf") renderPdfViewer(thinking, url, title);
    }
  } catch (err) {
    thinking.textContent = `Error: ${err}`;
  }
}

form.addEventListener("submit", (e) => {
  e.preventDefault();
  const text = input.value.trim();
  if (!text) return;
  input.value = "";
  sendChat(text);
});

input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    form.requestSubmit();
  }
});

// --- Viewers ---------------------------------------------------------------

function buildViewerShell(msgEl, title) {
  const wrap = document.createElement("div");
  wrap.className = "viewer";
  const header = document.createElement("div");
  header.className = "viewer-title";
  header.textContent = title || "";
  const body = document.createElement("div");
  body.className = "viewer-body";
  wrap.appendChild(header);
  wrap.appendChild(body);
  msgEl.appendChild(wrap);
  logEl.scrollTop = logEl.scrollHeight;
  return body;
}

function showViewerError(body, msg) {
  body.textContent = msg;
  body.classList.add("viewer-error");
}

function renderPdfViewer(msgEl, url, title) {
  const body = buildViewerShell(msgEl, title || "PDF");
  body.style.width = "360px";
  body.style.height = "500px";

  if (!window.pdfjsLib) {
    showViewerError(body, "PDF.js failed to load.");
    return;
  }

  const canvas = document.createElement("canvas");
  canvas.style.maxWidth = "100%";
  canvas.style.maxHeight = "100%";
  body.appendChild(canvas);

  // Render the first page at ~1.2 scale. Non-blocking; handle errors inline.
  window.pdfjsLib
    .getDocument({ url })
    .promise.then((pdf) => pdf.getPage(1))
    .then((page) => {
      const viewport = page.getViewport({ scale: 1.2 });
      canvas.width = viewport.width;
      canvas.height = viewport.height;
      const ctx = canvas.getContext("2d");
      return page.render({ canvasContext: ctx, viewport }).promise;
    })
    .catch((err) => {
      showViewerError(body, `PDF render error: ${err.message || err}`);
    });
}

