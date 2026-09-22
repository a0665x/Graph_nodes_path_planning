"use strict";
const $ = id => document.getElementById(id);
const state = {image: null, source: "", mask: null, overlay: null, nodes: [], result: null,
  mode: "via", revision: 0, mapRevision: 0, ready: false, busy: false, upload: null, zoom: 1, panX: 0, panY: 0};
const canvas = $("map"), context = canvas.getContext("2d");
let settingsTimer;

async function api(path, payload) {
  const response = await fetch(path, payload === undefined ? {} : {
    method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(payload)
  });
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || `HTTP ${response.status}`);
  return result;
}
function loadImage(encoded) {
  return new Promise((resolve, reject) => {
    const image = new Image(); image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("無法讀取圖片。"));
    image.src = encoded.startsWith("data:") ? encoded : `data:image/png;base64,${encoded}`;
  });
}
function message(text, type = "") { $("message").textContent = text; $("message").className = `message ${type}`; }
function settings() {
  const resolution = Number($("resolution").value);
  return {white_threshold: Number($("white-threshold").value), wall_threshold: Number($("wall-threshold").value),
    allow_gray: !$("corridor-only").checked, robot_radius: Number($("radius").value) / resolution,
    safety_margin: Number($("margin").value) / resolution, resolution, speed: Number($("speed").value),
    clearance_weight: Number($("clearance-weight").value), gray_weight: 4};
}
function payload() { return {image: state.source, nodes: state.nodes.map(p => [...p]), settings: settings()}; }
function updateButtons() {
  $("plan").disabled = !state.ready || state.busy || state.nodes.length < 2;
  $("plan").firstElementChild.textContent = state.busy ? "正在尋找安全路徑…" : "計算安全路徑";
  $("export-route").disabled = !state.result;
  $("export-nodes").disabled = !state.nodes.length;
}
function invalidate(text = "節點已更新，請重新計算路徑。") {
  state.revision++; state.result = null; state.busy = false;
  for (const id of ["distance", "duration", "clearance"]) $(id).textContent = "—";
  $("route-status").textContent = "待規劃"; $("compute-time").textContent = "靜態地圖 · A*";
  $("metric-dot").className = "status-dot muted";
  $("segments").replaceChildren(); $("matrix").replaceChildren();
  updateButtons(); draw(); message(text, "pending");
}
function transform() {
  if (!state.image) return {scale: 1, x: 0, y: 0};
  const width = canvas.clientWidth, height = canvas.clientHeight;
  const fit = Math.min((width - 38) / state.image.width, (height - 42) / state.image.height);
  const scale = fit * state.zoom;
  return {scale, x: (width - state.image.width * scale) / 2 + state.panX,
    y: (height - state.image.height * scale) / 2 + state.panY};
}
function draw() {
  const dpr = window.devicePixelRatio || 1, width = canvas.clientWidth, height = canvas.clientHeight;
  canvas.width = Math.round(width * dpr); canvas.height = Math.round(height * dpr);
  context.setTransform(dpr, 0, 0, dpr, 0, 0); context.clearRect(0, 0, width, height);
  if (!state.image) return;
  const t = transform();
  context.save(); context.translate(t.x, t.y); context.scale(t.scale, t.scale);
  context.imageSmoothingEnabled = false; context.drawImage(state.image, 0, 0);
  if ($("show-clearance").checked && state.overlay) context.drawImage(state.overlay, 0, 0);
  if (state.result) {
    const path = state.result.path;
    context.beginPath(); context.moveTo(path[0][0], path[0][1]);
    for (const [x, y] of path.slice(1)) context.lineTo(x, y);
    // The actual A* polyline is rendered: no splines or unchecked shortcuts.
    context.strokeStyle = "#087f78"; context.lineWidth = 2.8 / t.scale;
    context.lineJoin = "round"; context.lineCap = "round"; context.stroke();
  }
  state.nodes.forEach(([x, y], i) => {
    const first = i === 0, last = i === state.nodes.length - 1;
    context.beginPath(); context.arc(x, y, 9 / t.scale, 0, Math.PI * 2);
    context.fillStyle = first ? "#087f78" : last ? "#172f36" : "#fff"; context.fill();
    context.lineWidth = 2 / t.scale; context.strokeStyle = first || last ? "#fff" : "#087f78"; context.stroke();
    context.fillStyle = first || last ? "#fff" : "#087f78";
    context.font = `600 ${9 / t.scale}px system-ui`; context.textAlign = "center"; context.textBaseline = "middle";
    context.fillText(first ? "S" : last ? "E" : String(i), x, y + .2 / t.scale);
  });
  context.restore(); $("zoom-label").textContent = `${Math.round(state.zoom * 100)}%`;
}
function renderNodes() {
  $("node-count").textContent = String(state.nodes.length).padStart(2, "0");
  const list = $("node-list"); list.replaceChildren();
  if (!state.nodes.length) {
    const note = document.createElement("li"); note.className = "node-empty";
    note.textContent = "點選白色走廊，建立起點與終點。"; list.append(note);
  }
  state.nodes.forEach(([x, y], i) => {
    const first = i === 0, last = i === state.nodes.length - 1;
    const row = document.createElement("li"); row.className = "node";
    const badge = document.createElement("span"); badge.className = "node-badge";
    badge.textContent = first ? "S" : last ? "E" : String(i).padStart(2, "0");
    const info = document.createElement("div"); info.className = "node-info";
    const title = document.createElement("strong"); title.textContent = first ? "起點" : last ? "終點" : `途經點 ${i}`;
    const coordinates = document.createElement("small"); coordinates.textContent = `x ${x}   /   y ${y}`;
    info.append(title, coordinates);
    const controls = document.createElement("div"); controls.className = "node-controls";
    for (const [symbol, label, action, disabled] of [
      ["↑", `節點 ${i + 1} 上移`, () => moveNode(i, -1), first],
      ["↓", `節點 ${i + 1} 下移`, () => moveNode(i, 1), last],
      ["×", `刪除節點 ${i + 1}`, () => { state.nodes.splice(i, 1); changedNodes(); }, false]
    ]) {
      const button = document.createElement("button"); button.textContent = symbol; button.title = label;
      button.setAttribute("aria-label", label); button.disabled = disabled; button.onclick = action; controls.append(button);
    }
    row.append(badge, info, controls); list.append(row);
  });
  updateButtons();
}
function changedNodes() { invalidate(); renderNodes(); }
function moveNode(i, delta) { const j = i + delta; [state.nodes[i], state.nodes[j]] = [state.nodes[j], state.nodes[i]]; changedNodes(); }
function table(target, headers, rows) {
  const table = document.createElement("table"), head = table.createTHead().insertRow();
  for (const label of headers) { const th = document.createElement("th"); th.textContent = label; head.append(th); }
  const body = table.createTBody();
  for (const values of rows) { const row = body.insertRow(); for (const value of values) row.insertCell().textContent = value; }
  $(target).replaceChildren(table);
}
function renderMatrix() {
  if (!state.result) return;
  const matrix = state.result[$("matrix-type").value];
  table("matrix", ["節點", ...matrix.map((_, i) => String(i + 1))],
    matrix.map((row, i) => [String(i + 1), ...row.map(value => value === null ? "—" : value.toFixed(2))]));
}
async function prepare() {
  const mapRevision = state.mapRevision;
  state.ready = false; state.mask = null; state.overlay = null; updateButtons();
  const response = await api("/api/prepare", payload());
  const image = await loadImage(response.image), mask = await loadImage(response.mask);
  if (mapRevision !== state.mapRevision) return false;
  state.image = image;
  const temp = document.createElement("canvas"); temp.width = mask.width; temp.height = mask.height;
  const ctx = temp.getContext("2d", {willReadFrequently: true}); ctx.drawImage(mask, 0, 0);
  state.mask = ctx.getImageData(0, 0, mask.width, mask.height).data;
  const overlay = ctx.createImageData(mask.width, mask.height);
  for (let i = 0; i < state.mask.length; i += 4) {
    overlay.data[i] = 170; overlay.data[i + 1] = 83; overlay.data[i + 2] = 61;
    overlay.data[i + 3] = state.mask[i] === 0 ? 85 : 0;
  }
  ctx.putImageData(overlay, 0, 0); state.overlay = temp;
  state.ready = true; $("canvas-state").textContent = $("corridor-only").checked ? "白色走廊模式" : "白色優先 · 允許灰區";
  $("map-dimensions").textContent = `${image.width} × ${image.height} px`;
  $("empty-map").hidden = true; draw(); updateButtons(); return true;
}
async function setMap(encoded, nodes, title) {
  clearTimeout(settingsTimer); state.mapRevision++; invalidate("正在建立安全通行區…");
  state.source = encoded; state.nodes = nodes; state.image = null; state.mask = null;
  state.ready = false; state.zoom = 1; state.panX = state.panY = 0;
  $("map-title").textContent = title; $("empty-map").hidden = false; renderNodes(); draw();
  const revision = state.revision;
  try {
    if (await prepare()) {
      message("地圖就緒。請在白色走廊放置節點；滾輪可縮放，右鍵可拖曳。", "pending");
      if (nodes.length >= 2) await plan();
    }
  } catch (error) { if (revision === state.revision) { $("empty-map").hidden = true; message(error.message, "error"); } }
}
async function plan() {
  if (!state.ready || state.busy || state.nodes.length < 2) return;
  invalidate("正在開始規劃…");
  const revision = state.revision; state.busy = true; updateButtons(); draw();
  $("route-status").textContent = "計算中"; message("正在計算各路段，檢查走廊與安全淨距…", "pending");
  try {
    const result = await api("/api/plan", payload());
    if (revision !== state.revision) return;
    state.result = result;
    $("distance").textContent = result.distance_m.toFixed(1);
    $("duration").textContent = result.travel_time_s.toFixed(0);
    $("clearance").textContent = result.min_clearance_m.toFixed(2);
    $("route-status").textContent = "路徑可通行";
    $("metric-dot").className = "status-dot";
    $("compute-time").textContent = `${(result.compute_ms / 1000).toFixed(2)} s · ${result.path.length.toLocaleString()} 路徑像素`;
    table("segments", ["路段", "距離 (m)", "時間 (s)", "A* 成本"],
      result.segments.map(s => [`${s.from + 1} → ${s.to + 1}`, s.distance_m.toFixed(2), s.travel_time_s.toFixed(1), s.cost.toFixed(2)]));
    renderMatrix();
    message(`已完成 ${result.segments.length} 段安全路徑。沿原始 costmap 通行，不穿越禁行區；淨距為保守下界。`);
    draw();
  } catch (error) {
    if (revision !== state.revision) return;
    $("route-status").textContent = "無可用路徑"; $("metric-dot").className = "status-dot muted";
    message(`無法規劃：${error.message}`, "error");
  } finally { if (revision === state.revision) { state.busy = false; updateButtons(); } }
}
let loadSequence = 0;
async function selectMap() {
  const sequence = ++loadSequence;
  state.mapRevision++; state.ready = false; invalidate("載入地圖中…");
  const value = $("map-source").value, title = $("map-source").selectedOptions[0].textContent;
  try {
    if (value === "upload" && state.upload) {
      await setMap(state.upload.image, [], state.upload.name); return;
    }
    const url = value.startsWith("demo:") ? `/api/demo?name=${encodeURIComponent(value.slice(5))}` : `/api/map?name=${encodeURIComponent(value.slice(5))}`;
    const result = await api(url);
    if (sequence === loadSequence) await setMap(result.image, result.nodes, title);
  } catch (error) { if (sequence === loadSequence) message(error.message, "error"); }
}
function imagePoint(event) {
  const rect = canvas.getBoundingClientRect(), t = transform();
  return [Math.round((event.clientX - rect.left - t.x) / t.scale), Math.round((event.clientY - rect.top - t.y) / t.scale)];
}
let drag = null;
canvas.addEventListener("contextmenu", event => event.preventDefault());
canvas.addEventListener("pointerdown", event => {
  drag = {x: event.clientX, y: event.clientY, panX: state.panX, panY: state.panY, moved: false,
    pan: event.button === 2 || event.button === 1 || event.pointerType === "touch"};
  canvas.setPointerCapture(event.pointerId);
});
canvas.addEventListener("pointermove", event => {
  if (drag && drag.pan) {
    const dx = event.clientX - drag.x, dy = event.clientY - drag.y;
    if (Math.hypot(dx, dy) > 5) drag.moved = true;
    if (drag.moved) { state.panX = drag.panX + dx; state.panY = drag.panY + dy; draw(); }
  }
  if (state.image) { const [x, y] = imagePoint(event); $("pointer").textContent = `x ${x} / y ${y} · 原圖像素`; }
});
canvas.addEventListener("pointercancel", () => { drag = null; });
canvas.addEventListener("pointerup", event => {
  const last = drag; drag = null;
  if (!last || last.moved || event.button === 2 || event.button === 1 || !state.ready) return;
  const point = imagePoint(event), [x, y] = point;
  if (x < 0 || y < 0 || x >= state.image.width || y >= state.image.height) return;
  if (!state.mask || state.mask[(y * state.image.width + x) * 4] === 0) {
    return message("這裡是牆壁、禁行灰區或安全邊距內。請選擇白色走廊中央。", "error");
  }
  if (state.mode === "start" && state.nodes.length) state.nodes[0] = point;
  else if (state.mode === "end" && state.nodes.length > 1) state.nodes[state.nodes.length - 1] = point;
  else {
    if (state.nodes.length >= 16) return message("最多可使用 16 個節點。", "error");
    if (state.mode === "via" && state.nodes.length > 1) state.nodes.splice(state.nodes.length - 1, 0, point);
    else state.nodes.push(point);
  }
  changedNodes();
});
canvas.addEventListener("wheel", event => { event.preventDefault(); state.zoom = Math.max(.6, Math.min(5, state.zoom * (event.deltaY < 0 ? 1.12 : .89))); draw(); }, {passive: false});
new ResizeObserver(draw).observe($("stage"));
for (const button of document.querySelectorAll("[data-mode]")) button.onclick = () => {
  state.mode = button.dataset.mode;
  for (const item of document.querySelectorAll("[data-mode]")) {
    const active = item === button; item.classList.toggle("active", active); item.setAttribute("aria-pressed", String(active));
  }
};
$("plan").onclick = plan;
$("clear-nodes").onclick = () => { state.nodes = []; changedNodes(); };
$("fit").onclick = () => { state.zoom = 1; state.panX = state.panY = 0; draw(); };
$("zoom-in").onclick = () => { state.zoom = Math.min(5, state.zoom * 1.25); draw(); };
$("zoom-out").onclick = () => { state.zoom = Math.max(.6, state.zoom / 1.25); draw(); };
$("show-clearance").onchange = draw;
$("matrix-type").onchange = renderMatrix;
$("map-source").onchange = selectMap;
$("upload-map").onclick = () => $("map-file").click();
$("map-file").onchange = async event => {
  const file = event.target.files[0]; if (!file) return;
  if (file.size > 10 * 1024 * 1024) { message("請使用 10 MiB 以下的地圖。", "error"); event.target.value = ""; return; }
  const sequence = ++loadSequence; state.mapRevision++; state.ready = false; invalidate("正在讀取地圖…");
  try {
    const bytes = new Uint8Array(await file.arrayBuffer());
    let binary = ""; for (let i = 0; i < bytes.length; i += 8192) binary += String.fromCharCode(...bytes.subarray(i, i + 8192));
    if (sequence === loadSequence) {
      // A separate upload label prevents a selected bundled map from lying about the canvas.
      $("map-source").querySelector("option[value='upload']")?.remove();
      const option = new Option(file.name, "upload", true, true); $("map-source").add(option);
      state.upload = {image: btoa(binary), name: file.name};
      await setMap(state.upload.image, [], file.name);
    }
  } catch (error) { if (sequence === loadSequence) message(error.message, "error"); }
  event.target.value = "";
};
$("import-nodes").onclick = () => $("node-file").click();
$("node-file").onchange = async event => {
  const file = event.target.files[0]; if (!file || !state.ready) return;
  const revision = state.revision;
  try {
    if (file.size > 10000) throw new Error("節點檔案過大。最多 16 個節點。");
    const result = await api("/api/points", {text: await file.text()});
    if (revision !== state.revision) return;
    for (let i = 0; i < result.nodes.length; i++) {
      const [x, y] = result.nodes[i];
      if (x < 0 || y < 0 || x >= state.image.width || y >= state.image.height || state.mask[(y * state.image.width + x) * 4] === 0)
        throw new Error(`節點 ${i + 1} 不在安全通行區；未套用此檔案。`);
    }
    state.nodes = result.nodes; changedNodes();
  } catch (error) { if (revision === state.revision) message(error.message, "error"); }
  finally { event.target.value = ""; }
};
for (const id of ["corridor-only", "radius", "margin", "resolution", "speed", "white-threshold", "wall-threshold", "clearance-weight"]) {
  $(id).addEventListener("input", () => {
    $("clearance-value").textContent = $("clearance-weight").value;
    clearTimeout(settingsTimer); state.mapRevision++; state.ready = false; invalidate("設定已變更，正在重新檢查安全通行區…");
    const revision = state.revision;
    settingsTimer = setTimeout(async () => {
      try { if (await prepare()) message("安全通行區已更新，請重新計算路徑。", "pending"); }
      catch (error) { if (revision === state.revision) message(error.message, "error"); }
    }, 250);
  });
}
function download(name, text, type) {
  const url = URL.createObjectURL(new Blob([text], {type})), a = document.createElement("a");
  a.href = url; a.download = name; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
}
$("export-nodes").onclick = () => download("points.txt", state.nodes.map(([x, y]) => `(${x}, ${y})`).join("\n") + "\n", "text/plain");
$("export-route").onclick = () => {
  if (state.result) download("safe-route.json", JSON.stringify({map: $("map-title").textContent,
    image_width: state.image.width, image_height: state.image.height, ...state.result}, null, 2), "application/json");
};
(async () => {
  try {
    const result = await api("/api/maps");
    for (const name of result.maps) $("map-source").add(new Option(name, `repo:${name}`));
    await selectMap();
  } catch (error) { message(error.message, "error"); }
})();
