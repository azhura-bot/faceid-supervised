const video = document.getElementById('video');
const overlay = document.getElementById('overlay');
const ctx = overlay.getContext('2d');
const startBtn = document.getElementById('startId');
const resultArea = document.querySelector('.results-area');
const MIRRORED = true;

let running = false;
let lastFaces = []; // 🔹 untuk mendeteksi wajah baru

async function openCamera(){
  try{
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: {ideal: 640}, height: {ideal: 480}, facingMode: 'user' }
    });
    video.srcObject = stream;
    await new Promise(r=> video.onloadedmetadata = r);
    await video.play();
    overlay.width = video.videoWidth || 640;
    overlay.height = video.videoHeight || 480;
  }catch(err){
    console.error('Gagal akses kamera', err);
    alert('Gagal akses kamera: '+(err.message||err));
    throw err;
  }
}

function snapshot(){
  const c = document.createElement('canvas');
  c.width = video.videoWidth || 640; 
  c.height = video.videoHeight || 480;
  const cctx = c.getContext('2d');
  cctx.drawImage(video, 0, 0, c.width, c.height);
  return c.toDataURL('image/jpeg', 0.85);
}

function drawFaces(faces){
  ctx.clearRect(0,0,overlay.width,overlay.height);
  faces.forEach(f => {
    let left = f.left, right = f.right;
    if (MIRRORED) {
      left  = overlay.width - f.right;
      right = overlay.width - f.left;
    }
    const top = f.top, bottom = f.bottom;
    const w = right - left, h = bottom - top;

    ctx.lineWidth = 2; 
    ctx.strokeStyle = f.name === "Unknown" ? "red" : "#22c55e";
    ctx.beginPath();
    if (ctx.roundRect) ctx.roundRect(left, top, w, h, 10); else ctx.rect(left, top, w, h);
    ctx.stroke();

    const text = f.name, pad = 6, H = 22;
    ctx.font = 'bold 14px system-ui';
    const W = ctx.measureText(text).width + pad*2;
    const x = Math.max(0, left + (w/2 - W/2));
    const y = Math.max(0, top - H - 6);
    ctx.fillStyle = f.name === "Unknown" ? "red" : "#22c55e";
    ctx.fillRect(x, y, W, H);
    ctx.fillStyle = '#000'; 
    ctx.fillText(text, x+pad, y+15);
  });
}

// 🧩 Update panel hasil tanpa animasi kedap-kedip
function updateResultsPanel(faces){
  if (!faces || !faces.length){
    resultArea.innerHTML = `
      <div class="result-placeholder">
        <i class="fa-regular fa-face-meh"></i>
        <p>No face detected yet</p>
      </div>`;
    lastFaces = [];
    return;
  }

  // 🔹 Cek wajah baru (berdasarkan nama)
  const newNames = faces.map(f => f.name);
  const oldNames = lastFaces.map(f => f.name);
  const isNewFace = newNames.join(',') !== oldNames.join(',');
  lastFaces = faces;

  // 🔹 Update UI
  let html = "";
  faces.forEach(f => {
    const conf = f.confidence ? `${f.confidence.toFixed(1)}%` : "-";
    const status = f.name === "Unknown" ? "Unrecognized" : "Recognized";
    const colorClass =
      f.name === "Unknown" ? "red" :
      f.confidence >= 85 ? "green" : "orange";

    // hanya beri animasi kalau wajah baru muncul
    const animClass = isNewFace ? "pop" : "";

    html += `
      <div class="face-card ${colorClass} ${animClass}">
        <h4>${f.name}</h4>
        <p><strong>Confidence:</strong> ${conf}</p>
        <p><strong>Status:</strong> ${status}</p>
      </div>`;
  });
  resultArea.innerHTML = html;
}

async function tick(){
  if(!running) return;
  try{
    const r = await fetch('/api/identify', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({image: snapshot()})
    });
    const j = await r.json();
    if(j.ok){
      drawFaces(j.faces || []);
      updateResultsPanel(j.faces || []);
    }
  }catch(e){ console.error(e); }
  requestAnimationFrame(tick);
}

startBtn.addEventListener('click', async () => {
  await openCamera();
  running = true; 
  tick();
});
