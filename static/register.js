// /static/register.js
const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const ctx = overlay.getContext("2d");
const stepInfo = document.getElementById("stepInfo");
const startBtn = document.getElementById("startReg");
const trainBtn = document.getElementById("train");
const nameInput = document.getElementById("name");

let inFlight = false; // cegah request numpuk
const SNAP_W = 320; // kirim lebih kecil untuk kecepatan
const SNAP_H = 240;
const JPEG_Q = 0.6; // kompresi sedang

const steps = [
  "Lihat lurus ke depan",
  "Arahkan wajah ke KANAN",
  "Arahkan wajah ke KIRI",
  "Berkedip (tutup mata sesaat)",
];

let running = false;
let facePresent = false;

// ukuran tampilan (CSS) + DPR, supaya gambar pas
let viewW = 640,
  viewH = 480,
  dpr = 1;

// === util: kumpulkan index unik dari daftar koneksi Mediapipe ===
function uniqueIdx(conns) {
  const s = new Set();
  conns.forEach(([a, b]) => {
    s.add(a);
    s.add(b);
  });
  return Array.from(s);
}

// === util: ambil titik [px] dari landmarks + index (rujuk ukuran tampilan) ===
function pointsFrom(lms, idxs, W, H) {
  return idxs.map((i) => ({ x: lms[i].x * W, y: lms[i].y * H }));
}

// === util: gambar ellipse dari kumpulan titik (pakai bbox sebagai pendekatan) ===
function drawEllipseFromPoints(
  ctx,
  pts,
  style = { stroke: "#22c55e", width: 2 }
) {
  if (!pts.length) return;
  let minX = Infinity,
    minY = Infinity,
    maxX = -Infinity,
    maxY = -Infinity;
  for (const p of pts) {
    if (p.x < minX) minX = p.x;
    if (p.x > maxX) maxX = p.x;
    if (p.y < minY) minY = p.y;
    if (p.y > maxY) maxY = p.y;
  }
  const cx = (minX + maxX) / 2;
  const cy = (minY + maxY) / 2;
  const rx = Math.max(4, (maxX - minX) / 2);
  const ry = Math.max(4, (maxY - minY) / 2);
  ctx.save();
  ctx.lineWidth = style.width || 2;
  ctx.strokeStyle = style.stroke || "#22c55e";
  ctx.beginPath();
  if (ctx.ellipse) {
    ctx.ellipse(cx, cy, rx, ry, 0, 0, Math.PI * 2);
  } else {
    const r = Math.max(rx, ry);
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
  }
  ctx.stroke();
  ctx.restore();
}

function stopCamera() {
  const stream = video.srcObject;
  if (stream) {
    stream.getTracks().forEach((track) => track.stop());
  }
  video.srcObject = null;
}

async function openCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: {
      width: { ideal: 640 },
      height: { ideal: 480 },
      facingMode: "user",
    },
  });
  video.srcObject = stream;
  await new Promise((r) => (video.onloadedmetadata = r));
  await video.play();

  // ====== ukuran overlay = ukuran tampilan (CSS) dikali DPR ======
  dpr = window.devicePixelRatio || 1;
  const rect = video.getBoundingClientRect();
  viewW = rect.width;
  viewH = rect.height;

  overlay.style.width = viewW + "px";
  overlay.style.height = viewH + "px";
  overlay.width = Math.round(viewW * dpr);
  overlay.height = Math.round(viewH * dpr);

  // koordinat menggambar kita = pixel CSS (bukan buffer DPR)
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  // === MediaPipe FaceMesh (CDN globals) ===
  try {
    const fm = new FaceMesh({
      locateFile: (file) =>
        `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`,
    });
    fm.setOptions({
      maxNumFaces: 1,
      refineLandmarks: true,
      selfieMode: true, // mode selfie (preview mirror)
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });

    fm.onResults((res) => {
      // pakai ukuran TAMPILAN agar marker presisi (clear dalam koordinat CSS)
      ctx.clearRect(0, 0, viewW, viewH);

      const ok = res.multiFaceLandmarks && res.multiFaceLandmarks.length;
      facePresent = !!ok;
      if (!ok) return;

      const lms = res.multiFaceLandmarks[0];

      // === HANYA ellipse mata, mulut, dan wajah (tanpa tesselation) ===
      const W = viewW,
        H = viewH;

      // mata kiri & kanan
      const leftEyeIdx = uniqueIdx(FACEMESH_LEFT_EYE);
      const rightEyeIdx = uniqueIdx(FACEMESH_RIGHT_EYE);
      const leftEyePts = pointsFrom(lms, leftEyeIdx, W, H);
      const rightEyePts = pointsFrom(lms, rightEyeIdx, W, H);
      drawEllipseFromPoints(ctx, leftEyePts);
      drawEllipseFromPoints(ctx, rightEyePts);

      // mulut
      const lipsIdx = uniqueIdx(FACEMESH_LIPS);
      const lipsPts = pointsFrom(lms, lipsIdx, W, H);
      drawEllipseFromPoints(ctx, lipsPts, { stroke: "#8B5CF6", width: 2 });

      // wajah (face oval)
      const ovalIdx = uniqueIdx(FACEMESH_FACE_OVAL);
      const ovalPts = pointsFrom(lms, ovalIdx, W, H);
      drawEllipseFromPoints(ctx, ovalPts, { stroke: "#8B5CF6", width: 2 });

      // bounding box kasar (referensi)
      // const xs = lms.map(p=>p.x*W);
      // const ys = lms.map(p=>p.y*H);
      // const left=Math.max(0,Math.min(...xs)), right=Math.min(W,Math.max(...xs));
      // const top=Math.max(0,Math.min(...ys)), bottom=Math.min(H,Math.max(...ys));
      // ctx.lineWidth = 1.5; ctx.strokeStyle = '#22c55e';
      // if (ctx.roundRect) { ctx.beginPath(); ctx.roundRect(left, top, right-left, bottom-top, 10); ctx.stroke(); }
      // else { ctx.strokeRect(left, top, right-left, bottom-top); }
    });

    const cam = new Camera(video, {
      onFrame: async () => {
        await fm.send({ image: video });
      },
      width: 640,
      height: 480,
    });
    cam.start();
  } catch (err) {
    console.error("FaceMesh init gagal:", err);
    facePresent = true; // fallback
  }
}

// kirim frame yang SUDAH mirror, supaya instruksi kanan/kiri sesuai preview selfie
function snapshot() {
  const c = document.createElement("canvas");
  c.width = SNAP_W;
  c.height = SNAP_H;
  const cctx = c.getContext("2d");
  cctx.save();
  cctx.translate(c.width, 0);
  cctx.scale(-1, 1);
  cctx.drawImage(video, 0, 0, c.width, c.height);
  cctx.restore();
  return c.toDataURL("image/jpeg", JPEG_Q);
}

function renderStatus({ step, need, got }) {
  let html = `<div>Step ${step + 1}/4: <strong>${
    steps[Math.min(step, 3)] ?? "Selesai"
  }</strong></div>`;
  html += '<div style="margin-top:6px">';
  for (let i = 0; i < 4; i++) {
    const ok = got[i] >= need[i];
    html += `<span class="tag ${ok ? "ok" : "warn"}">${
      ["Depan", "Kanan", "Kiri", "Kedip"][i]
    }: ${got[i]}/${need[i]}</span>`;
  }
  html += ` ${
    facePresent
      ? '<span class="tag ok">Face detected</span>'
      : '<span class="tag warn">Arahkan wajah ke kamera</span>'
  }`;
  html += "</div>";
  stepInfo.innerHTML = html;
}

async function loopCapture() {
  if (!running) {
    return;
  }
  if (!facePresent) {
    setTimeout(loopCapture, 250);
    return;
  }
  if (inFlight) {
    setTimeout(loopCapture, 120);
    return;
  } // tunggu request sebelumnya selesai

  inFlight = true;
  let nextDelay = 280; // default

  try {
    const r = await fetch("/api/register_frame", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: snapshot() }),
    });

    const ct = (r.headers.get("content-type") || "").toLowerCase();
    let j;
    if (!ct.includes("application/json")) {
      const text = await r.text();
      throw new Error(
        `Non-JSON (status ${r.status}). Cuplikan: ${text.slice(0, 200)}…`
      );
    }
    try {
      j = await r.json();
    } catch (e) {
      const text = await r.text().catch(() => "");
      throw new Error(
        `JSONDecodeError (status ${r.status}). Cuplikan: ${text.slice(0, 200)}…`
      );
    }

    if (!j.ok) {
      stepInfo.innerHTML = `<span class="tag err">${j.error || "Error"}</span>`;
      running = false;
      inFlight = false;
      return;
    }

    renderStatus(j);

    // kalau ditolak, beri jeda sedikit lebih lama
    if (j.accepted === false) {
      if (j.reason) {
        stepInfo.innerHTML += ` <span class="tag warn">${j.reason}</span>`;
      }
      nextDelay = 360;
    } else {
      // diterima → percepat sampling untuk cepat penuh kuotanya
      nextDelay = 160;
    }

    if (j.done) {
      running = false;
      trainBtn.disabled = false;
      stepInfo.innerHTML += ' <span class="tag ok">Registrasi selesai ✔</span>';

      // 🔹 Matikan kamera otomatis
      stopCamera();

      // 🔹 Perbarui daftar nama tanpa refresh
      await listDataset();

      inFlight = false;
      return;
    }
  } catch (e) {
    stepInfo.innerHTML = `<span class="tag err">Server error saat simpan frame: ${
      e.message || e
    }</span>`;
    running = false;
    inFlight = false;
    return;
  }

  inFlight = false;
  setTimeout(loopCapture, nextDelay);
}

startBtn.addEventListener("click", async () => {
  const name = nameInput.value.trim();
  if (!name) {
    alert("Nama wajib diisi");
    return;
  }
  await openCamera();
  const r = await fetch("/api/start_register", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });
  const j = await r.json();
  if (!j.ok) {
    alert(j.error || "Error");
    return;
  }
  renderStatus(j);
  running = true;
  loopCapture();
});

trainBtn.addEventListener("click", async () => {
  trainBtn.disabled = true;
  const r = await fetch("/api/train", { method: "POST" });
  const j = await r.json();
  if (!j.ok) {
    alert(j.error || "Error");
    trainBtn.disabled = false;
    return;
  }
  stepInfo.innerHTML += ` <span class="tag ok">Model dilatih (${j.classes.length} kelas)</span>`;
});

// mini tools (opsional)
window.listDataset = async () => {
  const container = document.getElementById("datasetList");
  if (!container) return;

  // efek loading ringan
  container.style.opacity = 0.5;

  try {
    const res = await fetch("/api/embeddings");
    const j = await res.json();
    if (!j.ok) throw new Error(j.error || "gagal");

    const names = j.names || [];
    container.innerHTML =
      names.length === 0
        ? '<p class="muted">(Dataset kosong)</p>'
        : names.map(n => `<div class="tag ok">${n}</div>`).join(" ");
  } catch (e) {
    console.error(e);
    container.innerHTML = '<span class="tag warn">Gagal memuat dataset</span>';
  } finally {
    // kembalikan opacity setelah 150ms biar transisinya halus
    setTimeout(() => { container.style.opacity = 1; }, 150);
  }
};

window.deleteByName = async (name) => {
  if (!name) return;
  const j = await (
    await fetch("/api/embeddings", {
      method: "DELETE",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    })
  ).json();
  if (!j.ok) return alert(j.error || "gagal");
  alert(`Deleted: ${j.deleted}`);
  listDataset(); // 🔁 refresh daftar setelah delete
};
document.addEventListener("DOMContentLoaded", listDataset);
