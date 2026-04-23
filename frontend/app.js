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

// Render markdown safely into an existing message bubble. Wraps tables in a
// horizontal-scroll container so they don't break the bubble layout, and
// forces links to open in a new tab.
function setMarkdownContent(el, text) {
  const src = text || "";
  let html = src;
  if (window.marked) {
    window.marked.setOptions({ gfm: true, breaks: true });
    html = window.marked.parse(src);
  }
  if (window.DOMPurify) {
    html = window.DOMPurify.sanitize(html);
  }
  el.classList.add("markdown");
  el.innerHTML = html;
  el.querySelectorAll("table").forEach((t) => {
    if (t.parentElement && t.parentElement.classList.contains("md-table-wrap")) return;
    const wrap = document.createElement("div");
    wrap.className = "md-table-wrap";
    t.parentNode.insertBefore(wrap, t);
    wrap.appendChild(t);
  });
  el.querySelectorAll("a").forEach((a) => {
    a.setAttribute("target", "_blank");
    a.setAttribute("rel", "noopener noreferrer");
  });
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
    setMarkdownContent(thinking, data.reply || "(empty reply)");
    if (data.approval) addApprovalButtons(thinking);
    if (data.viewer) {
      const { kind, url, title } = data.viewer;
      if (kind === "pdf") renderPdfViewer(thinking, url, title);
      else if (kind === "step") renderStepViewer(thinking, url, title);
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

// Module singleton — occt-import-js is ~10MB of WASM; reuse across viewers.
let _occtPromise = null;
function loadOcct() {
  if (_occtPromise) return _occtPromise;
  if (typeof window.occtimportjs !== "function") {
    return Promise.reject(new Error("occt-import-js failed to load"));
  }
  _occtPromise = window.occtimportjs({
    locateFile: (name) =>
      `https://cdn.jsdelivr.net/npm/occt-import-js@0.0.22/dist/${name}`,
  });
  return _occtPromise;
}

async function renderStepViewer(msgEl, url, title) {
  const body = buildViewerShell(msgEl, title || "STEP");
  body.style.width = "480px";
  body.style.height = "360px";
  body.style.background = "#0a0c10";
  body.style.position = "relative";

  const status = document.createElement("div");
  status.className = "viewer-loading";
  status.textContent = "Loading 3D viewer…";
  body.appendChild(status);

  if (!window.THREE) {
    showViewerError(body, "three.js failed to load.");
    return;
  }

  try {
    const [occt, buf] = await Promise.all([
      loadOcct(),
      fetch(url).then((r) => {
        if (!r.ok) throw new Error(`fetch failed: ${r.status}`);
        return r.arrayBuffer();
      }),
    ]);

    status.textContent = "Parsing STEP…";
    const parsed = occt.ReadStepFile(new Uint8Array(buf), null);
    if (!parsed || !parsed.success) {
      throw new Error("STEP parser returned no meshes");
    }

    // Build the scene from the parsed meshes.
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a0c10);
    const group = new THREE.Group();
    for (const mesh of parsed.meshes || []) {
      const pos = mesh.attributes?.position?.array;
      const idx = mesh.index?.array;
      if (!pos || !idx) continue;
      const geom = new THREE.BufferGeometry();
      geom.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
      if (mesh.attributes.normal?.array) {
        geom.setAttribute(
          "normal",
          new THREE.Float32BufferAttribute(mesh.attributes.normal.array, 3)
        );
      }
      geom.setIndex(new THREE.Uint32BufferAttribute(idx, 1));
      if (!mesh.attributes.normal?.array) geom.computeVertexNormals();
      const c = mesh.color || [0.72, 0.74, 0.78];
      const mat = new THREE.MeshStandardMaterial({
        color: new THREE.Color(c[0], c[1], c[2]),
        metalness: 0.15,
        roughness: 0.55,
      });
      group.add(new THREE.Mesh(geom, mat));
    }
    if (group.children.length === 0) {
      throw new Error("No renderable geometry found");
    }
    scene.add(group);

    // Frame the model.
    const box = new THREE.Box3().setFromObject(group);
    const size = new THREE.Vector3();
    const center = new THREE.Vector3();
    box.getSize(size);
    box.getCenter(center);
    const maxDim = Math.max(size.x, size.y, size.z) || 1;

    scene.add(new THREE.AmbientLight(0xffffff, 0.55));
    const key = new THREE.DirectionalLight(0xffffff, 0.9);
    key.position.set(1, 1, 1).multiplyScalar(maxDim);
    scene.add(key);
    const fill = new THREE.DirectionalLight(0xffffff, 0.35);
    fill.position.set(-1, -0.5, -1).multiplyScalar(maxDim);
    scene.add(fill);

    const width = body.clientWidth || 480;
    const height = body.clientHeight || 360;
    const camera = new THREE.PerspectiveCamera(
      45,
      width / height,
      Math.max(maxDim * 0.01, 0.001),
      maxDim * 100
    );
    camera.position.set(
      center.x + maxDim * 1.4,
      center.y + maxDim * 1.0,
      center.z + maxDim * 1.6
    );
    camera.lookAt(center);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));

    status.remove();
    body.appendChild(renderer.domElement);
    renderer.domElement.style.display = "block";

    const controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.target.copy(center);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.update();

    function tick() {
      controls.update();
      renderer.render(scene, camera);
      requestAnimationFrame(tick);
    }
    tick();
  } catch (err) {
    showViewerError(body, `STEP render error: ${err.message || err}`);
  }
}

