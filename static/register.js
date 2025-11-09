const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const ctx = overlay.getContext("2d");
const stepInfo = document.getElementById("stepInfo");
const startBtn = document.getElementById("startReg");
const nameInput = document.getElementById("name");

let inFlight = false;
let running = false;
let facePresent = false;

const SNAP_W = 320, SNAP_H = 240, JPEG_Q = 0.6;

const steps = ["Lihat lurus ke depan", "Arahkan wajah ke KANAN", "Arahkan wajah ke KIRI", "Berkedip (tutup mata sesaat)"];
let viewW = 640, viewH = 480, dpr = 1;

// === util
function uniqueIdx(conns) {
  const s = new Set();
  conns.forEach(([a, b]) => { s.add(a); s.add(b); });
  return Array.from(s);
}

function pointsFrom(lms, idxs, W, H) {
  return idxs.map((i) => ({ x: lms[i].x * W, y: lms[i].y * H }));
}

function drawEllipseFromPoints(ctx, pts, style = { stroke: "#8B5CF6", width: 2 }) {
  if (!pts.length) return;
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const p of pts) { minX = Math.min(minX, p.x); maxX = Math.max(maxX, p.x); minY = Math.min(minY, p.y); maxY = Math.max(maxY, p.y); }
  const cx = (minX + maxX) / 2, cy = (minY + maxY) / 2, rx = Math.max(4, (maxX - minX) / 2), ry = Math.max(4, (maxY - minY) / 2);
  ctx.save(); ctx.lineWidth = style.width; ctx.strokeStyle = style.stroke;
  ctx.beginPath(); ctx.ellipse(cx, cy, rx, ry, 0, 0, Math.PI * 2); ctx.stroke(); ctx.restore();
}

function stopCamera() {
  const stream = video.srcObject;
  if (stream) stream.getTracks().forEach((track) => track.stop());
  video.srcObject = null;
}

async function openCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480, facingMode: "user" } });
  video.srcObject = stream;
  await new Promise((r) => (video.onloadedmetadata = r));
  await video.play();

  dpr = window.devicePixelRatio || 1;
  const rect = video.getBoundingClientRect();
  viewW = rect.width; viewH = rect.height;
  overlay.width = Math.round(viewW * dpr); overlay.height = Math.round(viewH * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  try {
    const fm = new FaceMesh({ locateFile: (f) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${f}` });
    fm.setOptions({ maxNumFaces: 1, refineLandmarks: true, selfieMode: true, minDetectionConfidence: 0.5, minTrackingConfidence: 0.5 });

    fm.onResults((res) => {
      ctx.clearRect(0, 0, viewW, viewH);
      const ok = res.multiFaceLandmarks && res.multiFaceLandmarks.length;
      facePresent = !!ok;
      if (!ok) return;
      const lms = res.multiFaceLandmarks[0];
      drawEllipseFromPoints(ctx, pointsFrom(lms, uniqueIdx(FACEMESH_FACE_OVAL), viewW, viewH));
      drawEllipseFromPoints(ctx, pointsFrom(lms, uniqueIdx(FACEMESH_LEFT_EYE), viewW, viewH));
      drawEllipseFromPoints(ctx, pointsFrom(lms, uniqueIdx(FACEMESH_RIGHT_EYE), viewW, viewH));
    });

    new Camera(video, { onFrame: async () => { await fm.send({ image: video }); }, width: 640, height: 480 }).start();
  } catch (err) { console.error("FaceMesh init gagal:", err); facePresent = true; }
}

function snapshot() {
  const c = document.createElement("canvas"); c.width = SNAP_W; c.height = SNAP_H;
  const cctx = c.getContext("2d"); cctx.save(); cctx.translate(c.width, 0); cctx.scale(-1, 1);
  cctx.drawImage(video, 0, 0, c.width, c.height); cctx.restore();
  return c.toDataURL("image/jpeg", JPEG_Q);
}

function renderStatus({ step, need, got }) {
  let html = `<div>Step ${step + 1}/4: <strong>${steps[Math.min(step, 3)] ?? "Selesai"}</strong></div><div style="margin-top:6px">`;
  for (let i = 0; i < 4; i++) {
    const ok = got[i] >= need[i];
    html += `<span class="tag ${ok ? "ok" : "warn"}">${["Depan", "Kanan", "Kiri", "Kedip"][i]}: ${got[i]}/${need[i]}</span>`;
  }
  html += facePresent ? '<span class="tag ok">Face detected</span>' : '<span class="tag warn">Arahkan wajah ke kamera</span>';
  stepInfo.innerHTML = html + "</div>";
}

startBtn.addEventListener("click", async () => {
  const name = nameInput.value.trim();
  if (!name) return alert("Nama wajib diisi");
  await openCamera();
  const r = await fetch("/api/start_register", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ name }) });
  const j = await r.json(); if (!j.ok) return alert(j.error || "Error");
  renderStatus(j); running = true; loopCapture();
});

async function loopCapture() {
  if (!running) return;
  if (!facePresent || inFlight) return setTimeout(loopCapture, 200);
  inFlight = true;
  const r = await fetch("/api/register_frame", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ image: snapshot() }) });
  const j = await r.json();
  if (!j.ok) { stepInfo.innerHTML = `<span class="tag err">${j.error}</span>`; running = false; inFlight = false; return; }
  renderStatus(j);
  if (j.done) { running = false; stepInfo.innerHTML += ' <span class="tag ok">Registrasi selesai ✔</span>'; stopCamera(); await listDataset(); inFlight = false; return; }
  inFlight = false; setTimeout(loopCapture, 200);
}

// === Model Training Card ===
const trainModelBtn = document.getElementById("trainModelBtn");
const trainStatus = document.getElementById("trainStatus");

if (trainModelBtn) {
  trainModelBtn.addEventListener("click", async () => {
    trainModelBtn.disabled = true;
    trainStatus.innerHTML = `<span class="tag warn">⏳ Training dimulai... mohon tunggu</span>`;
    const startRes = await fetch("/api/train", { method: "POST" });
    const startJson = await startRes.json();
    if (!startJson.ok) {
      trainStatus.innerHTML = `<span class="tag err">❌ ${startJson.error}</span>`;
      trainModelBtn.disabled = false;
      return;
    }
    const interval = setInterval(async () => {
      const res = await fetch("/api/train_status");
      const j = await res.json();
      if (!j.ok) return;
      if (j.error) {
        clearInterval(interval);
        trainStatus.innerHTML = `<span class="tag err">❌ ${j.error}</span>`;
        trainModelBtn.disabled = false;
        return;
      }
      if (!j.done) {
        trainStatus.innerHTML = `<span class="tag warn">⚙️ Training masih berjalan...</span>`;
        return;
      }
      clearInterval(interval);
      trainModelBtn.disabled = false;
      if (j.result) {
        const r = j.result;
        trainStatus.innerHTML = `
          <span class="tag ok">✅ Training selesai!</span>
          <div class="train-result-card">
            <div><strong>Classes:</strong> ${r.classes.length}</div>
            <div><strong>Best Kernel:</strong> ${r.best_kernel}</div>
            <div><strong>Best Accuracy:</strong> ${r.best_acc}%</div>
            <div><strong>Average Accuracy:</strong> ${r.avg_acc}%</div>
          </div>`;
      }
    }, 3000);
  });
}

// === Dataset tools ===
window.listDataset = async () => {
  const container = document.getElementById("datasetList");
  if (!container) return;
  container.style.opacity = 0.5;
  try {
    const res = await fetch("/api/embeddings");
    const j = await res.json();
    if (!j.ok) throw new Error(j.error || "gagal");
    const names = j.names || [];
    container.innerHTML = names.length === 0 ? '<p class="muted">(Dataset kosong)</p>' : names.map(n => `<div class="tag ok">${n}</div>`).join(" ");
  } catch (e) { container.innerHTML = '<span class="tag warn">Gagal memuat dataset</span>'; }
  setTimeout(() => container.style.opacity = 1, 150);
};

window.deleteByName = async (name) => {
  if (!name) return;
  const j = await (await fetch("/api/embeddings", { method: "DELETE", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ name }) })).json();
  if (!j.ok) return alert(j.error || "gagal");
  alert(`Deleted: ${j.deleted}`); listDataset();
};

document.addEventListener("DOMContentLoaded", listDataset);
