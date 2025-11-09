const video = document.getElementById('video');
const overlay = document.getElementById('overlay');
const ctx = overlay.getContext('2d');
const startBtn = document.getElementById('startId');
const resultArea = document.querySelector('.results-area');
const MIRRORED = true;

let running = false;
let lastFaces = [];

// === Buka Kamera ===
async function openCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' }
    });
    video.srcObject = stream;
    await new Promise(r => video.onloadedmetadata = r);
    await video.play();
    overlay.width = video.videoWidth || 640;
    overlay.height = video.videoHeight || 480;
  } catch (err) {
    console.error('Gagal akses kamera', err);
    alert('Gagal akses kamera: ' + (err.message || err));
    throw err;
  }
}

// === Ambil Snapshot dari Kamera ===
function snapshot() {
  const c = document.createElement('canvas');
  c.width = video.videoWidth || 640;
  c.height = video.videoHeight || 480;
  const cctx = c.getContext('2d');
  cctx.drawImage(video, 0, 0, c.width, c.height);
  return c.toDataURL('image/jpeg', 0.85);
}

// === Gambar Bounding Box + Label Nama ===
function drawFaces(faces) {
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  faces.forEach(f => {
    let left = f.left, right = f.right;
    if (MIRRORED) {
      left = overlay.width - f.right;
      right = overlay.width - f.left;
    }
    const top = f.top, bottom = f.bottom;
    const w = right - left, h = bottom - top;

    // Warna box berdasarkan status
    const color =
      f.name === "Unknown" ? "#ef4444" : // merah
      f.confidence >= 85 ? "#22c55e" :   // hijau
      "#eab308";                         // kuning

    ctx.lineWidth = 2;
    ctx.strokeStyle = color;
    ctx.beginPath();
    if (ctx.roundRect) ctx.roundRect(left, top, w, h, 10);
    else ctx.rect(left, top, w, h);
    ctx.stroke();

    // Label nama di atas wajah
    const text = f.name;
    const pad = 6, H = 22;
    ctx.font = 'bold 14px system-ui';
    const W = ctx.measureText(text).width + pad * 2;
    const x = Math.max(0, left + (w / 2 - W / 2));
    const y = Math.max(0, top - H - 6);
    ctx.fillStyle = color;
    ctx.fillRect(x, y, W, H);
    ctx.fillStyle = '#000';
    ctx.fillText(text, x + pad, y + 15);
  });
}

// === Update Panel Hasil di Kanan ===
function updateResultsPanel(faces) {
  if (!faces || !faces.length) {
    resultArea.innerHTML = `
      <div class="result-placeholder">
        <i class="fa-regular fa-face-meh"></i>
        <p>No face detected yet</p>
      </div>`;
    lastFaces = [];
    return;
  }

  const newNames = faces.map(f => f.name);
  const oldNames = lastFaces.map(f => f.name);
  const isNewFace = newNames.join(',') !== oldNames.join(',');
  lastFaces = faces;

  let html = "";
  faces.forEach(f => {
    const conf = f.confidence ? `${f.confidence.toFixed(1)}%` : "-";
    const status =
      f.name === "Unknown"
        ? "Unrecognized"
        : f.confidence >= 85
          ? "Recognized"
          : "Low Confidence";

    const colorClass =
      f.name === "Unknown" ? "red" :
      f.confidence >= 85 ? "green" :
      "orange";

    const detect = f.detect_time ? `${f.detect_time.toFixed(3)} s` : "-";
    const predict = f.predict_time ? `${f.predict_time.toFixed(3)} s` : "-";

    html += `
      <div class="face-card ${colorClass} ${isNewFace ? "pop" : ""}">
        <h4>${f.name}</h4>
        <p><strong>Confidence:</strong> ${conf}</p>
        <p><strong>Status:</strong> ${status}</p>
        <p><strong>Detection Time:</strong> ${detect}</p>
        <p><strong>Prediction Time:</strong> ${predict}</p>
      </div>`;
  });

  resultArea.innerHTML = html;
}

// === Loop Deteksi Wajah ===
async function tick() {
  if (!running) return;
  try {
    const r = await fetch('/api/identify', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: snapshot() })
    });
    const j = await r.json();
    if (j.ok) {
      drawFaces(j.faces || []);
      updateResultsPanel(j.faces || []);
    }
  } catch (e) {
    console.error("Error identify:", e);
  }
  requestAnimationFrame(tick);
}

// === Tombol Start ===
startBtn.addEventListener('click', async () => {
  await openCamera();
  running = true;
  tick();
});
