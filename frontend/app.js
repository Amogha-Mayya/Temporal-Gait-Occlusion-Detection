/* ─────────────────────────────────────────────────────────────────────────
   app.js  –  AMOGHA Gait Inference Frontend Logic
───────────────────────────────────────────────────────────────────────── */

const API = ""; // same-origin

// ── State ──────────────────────────────────────────────────────────────────
let currentJobId = null;
let pollTimer = null;
let currentRunId = null;
let currentMode = "inject";

// ── DOM refs ───────────────────────────────────────────────────────────────
const $  = id => document.getElementById(id);
const sequencePath    = $("sequencePath");
const btnBrowse       = $("btnBrowse");
const folderBrowser   = $("folderBrowser");
const fbCurrent       = $("fbCurrent");
const fbList          = $("fbList");
const fbUp            = $("fbUp");
const fbSelect        = $("fbSelect");
const fbCancel        = $("fbCancel");
const modeInject      = $("modeInject");
const modeClean       = $("modeClean");
const injectSettings  = $("injectSettings");
const occlusionType   = $("occlusionType");
const startFrame      = $("startFrame");
const endFrame        = $("endFrame");
const targetSeverity  = $("targetSeverity");
const severityBadge   = $("severityBadge");
const severityFill    = $("severityFill");
const checkpoint      = $("checkpoint");
const btnRun          = $("btnRun");
const btnRunText      = $("btnRunText");
const headerStatus    = $("headerStatus");
const emptyState      = $("emptyState");
const runningState    = $("runningState");
const resultsContent  = $("resultsContent");
const runningSubText  = $("runningSubText");
const logContent      = $("logContent");
const logBox          = $("logBox");
const resultsMeta     = $("resultsMeta");
const summaryText     = $("summaryText");
const gifImg          = $("gifImg");
const gifLabel        = $("gifLabel");
const dashboardImg    = $("dashboardImg");
const btnReset        = $("btnReset");

// ── GIF tab labels ─────────────────────────────────────────────────────────
const GIF_META = {
  paired:   { label: "Paired: Clean | Occluded", file: "paired.gif" },
  clean:    { label: "Clean Silhouette Sequence", file: "clean.gif" },
  occluded: { label: "Occluded Sequence (Model Input)", file: "occluded.gif" },
};
let activeGifTab = "paired";

// ── Mode toggle ────────────────────────────────────────────────────────────
function setMode(mode) {
  currentMode = mode;
  modeInject.classList.toggle("active", mode === "inject");
  modeClean.classList.toggle("active", mode === "clean");
  injectSettings.classList.toggle("visible", mode === "inject");
  injectSettings.classList.toggle("hidden", mode === "clean");
}
modeInject.addEventListener("click", () => setMode("inject"));
modeClean.addEventListener("click",  () => setMode("clean"));
setMode("inject"); // default

// ── Severity slider ────────────────────────────────────────────────────────
targetSeverity.addEventListener("input", () => {
  const v = parseFloat(targetSeverity.value);
  severityBadge.textContent = v.toFixed(2);
  severityFill.style.width  = (v * 100) + "%";
});

// ── Folder browser ─────────────────────────────────────────────────────────
let fbCurrentPath = null;

function openBrowser(startPath) {
  folderBrowser.style.display = "block";
  loadDir(startPath || sequencePath.value || "/home");
}

function closeBrowser() {
  folderBrowser.style.display = "none";
}

async function loadDir(path) {
  fbCurrentPath = path;
  fbCurrent.textContent = path;
  fbList.innerHTML = `<div class="fb-item" style="color:var(--text-muted)">Loading…</div>`;
  try {
    const res = await fetch(`${API}/api/browse?path=${encodeURIComponent(path)}`);
    const data = await res.json();
    if (data.error) { fbList.innerHTML = `<div class="fb-item" style="color:#f87171">${data.error}</div>`; return; }
    fbCurrentPath = data.current;
    fbCurrent.textContent = data.current;
    fbUp.onclick = () => loadDir(data.parent);
    fbList.innerHTML = "";
    if (data.entries.length === 0) {
      fbList.innerHTML = `<div class="fb-item" style="color:var(--text-muted)">— empty directory —</div>`;
      return;
    }
    data.entries.forEach(e => {
      const item = document.createElement("div");
      item.className = "fb-item";
      item.innerHTML = `
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V7z"/>
        </svg>
        ${e.name}
      `;
      item.addEventListener("click", () => loadDir(e.path));
      fbList.appendChild(item);
    });
  } catch (err) {
    fbList.innerHTML = `<div class="fb-item" style="color:#f87171">Network error: ${err.message}</div>`;
  }
}

btnBrowse.addEventListener("click", () => {
  if (folderBrowser.style.display === "none") openBrowser();
  else closeBrowser();
});
fbCancel.addEventListener("click", closeBrowser);
fbSelect.addEventListener("click", () => {
  if (fbCurrentPath) sequencePath.value = fbCurrentPath;
  closeBrowser();
});

// ── Status helpers ─────────────────────────────────────────────────────────
function setHeaderStatus(state, text) {
  const dot  = headerStatus.querySelector(".status-dot");
  const span = headerStatus.querySelector("span:last-child");
  dot.className  = `status-dot ${state}`;
  span.textContent = text;
}

function showPanel(which) {
  emptyState.style.display    = which === "empty"   ? "flex" : "none";
  runningState.style.display  = which === "running" ? "flex" : "none";
  resultsContent.style.display = which === "results" ? "block" : "none";
}

// ── Log rendering ──────────────────────────────────────────────────────────
function renderLog(lines) {
  logContent.innerHTML = "";
  lines.forEach(line => {
    const div = document.createElement("div");
    div.className = "log-line";
    if (/saved|complete|done/i.test(line)) div.classList.add("success");
    else if (/warning|warn/i.test(line)) div.classList.add("warn");
    else if (/loading|running|mode|loaded|frame/i.test(line)) div.classList.add("highlight");
    div.textContent = line;
    logContent.appendChild(div);
  });
  logBox.scrollTop = logBox.scrollHeight;
  // Update sub-text with the last meaningful line
  const meaningful = lines.filter(l => l.trim().length > 0);
  if (meaningful.length > 0) {
    runningSubText.textContent = meaningful[meaningful.length - 1].substring(0, 80);
  }
}

// ── Polling ────────────────────────────────────────────────────────────────
function startPolling(jobId) {
  pollTimer = setInterval(async () => {
    try {
      const res  = await fetch(`${API}/api/status/${jobId}`);
      const data = await res.json();
      if (data.log) renderLog(data.log);
      if (data.status === "done") {
        clearInterval(pollTimer);
        loadResults(jobId);
      } else if (data.status === "error") {
        clearInterval(pollTimer);
        const lastLine = (data.log || []).slice(-1)[0] || data.error || "Unknown error";
        showError(lastLine);
      }
    } catch (e) {
      // network hiccup – keep polling
    }
  }, 800);
}

function showError(msg) {
  runningSubText.textContent = "Error: " + msg;
  runningSubText.style.color = "#f87171";
  setHeaderStatus("error", "Error");
  btnRun.disabled = false;
  btnRunText.textContent = "Run Inference";
}

// ── Load results ───────────────────────────────────────────────────────────
async function loadResults(jobId) {
  try {
    const res  = await fetch(`${API}/api/results/${jobId}`);
    const data = await res.json();
    if (data.error) { showError(data.error); return; }

    currentRunId = data.run_id;
    setHeaderStatus("done", "Inference Complete");
    showPanel("results");

    // Meta
    resultsMeta.textContent = `Run ID: ${data.run_id}`;

    // Summary
    summaryText.textContent = data.summary || "(no summary)";

    // Dashboard
    dashboardImg.src = `${API}/api/file/${data.run_id}/dashboard.png?t=${Date.now()}`;

    // Load default GIF tab
    loadGifTab("paired");

    btnRun.disabled = false;
    btnRunText.textContent = "Run Inference";
  } catch (e) {
    showError(e.message);
  }
}

// ── GIF tab switching ──────────────────────────────────────────────────────
function loadGifTab(tab) {
  activeGifTab = tab;
  document.querySelectorAll(".gif-tab").forEach(btn => {
    btn.classList.toggle("active", btn.dataset.gif === tab);
  });
  const meta = GIF_META[tab];
  gifLabel.textContent = meta.label;
  gifImg.src = `${API}/api/file/${currentRunId}/${meta.file}?t=${Date.now()}`;
}

document.querySelectorAll(".gif-tab").forEach(btn => {
  btn.addEventListener("click", () => {
    if (currentRunId) loadGifTab(btn.dataset.gif);
  });
});

// ── Reset ──────────────────────────────────────────────────────────────────
btnReset.addEventListener("click", () => {
  currentJobId = null;
  currentRunId = null;
  if (pollTimer) clearInterval(pollTimer);
  showPanel("empty");
  setHeaderStatus("idle", "Ready");
  btnRun.disabled = false;
  btnRunText.textContent = "Run Inference";
  runningSubText.style.color = "";
  logContent.innerHTML = "";
});

// ── Run inference ──────────────────────────────────────────────────────────
btnRun.addEventListener("click", async () => {
  const seq = sequencePath.value.trim();
  if (!seq) { alert("Please enter a sequence folder path."); return; }

  const payload = {
    sequence:        seq,
    mode:            currentMode,
    occlusion_type:  occlusionType.value,
    start_frame:     parseInt(startFrame.value, 10),
    end_frame:       parseInt(endFrame.value, 10),
    target_severity: parseFloat(targetSeverity.value),
    checkpoint:      checkpoint.value,
  };

  btnRun.disabled = true;
  btnRunText.textContent = "Running…";
  showPanel("running");
  setHeaderStatus("running", "Running inference…");
  runningSubText.textContent = "Sending request…";
  logContent.innerHTML = "";

  try {
    const res  = await fetch(`${API}/api/run`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (data.error) { showError(data.error); return; }
    currentJobId = data.job_id;
    startPolling(currentJobId);
  } catch (e) {
    showError(e.message);
  }
});

// ── Init ───────────────────────────────────────────────────────────────────
showPanel("empty");
setHeaderStatus("idle", "Ready");
