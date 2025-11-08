import os, base64, pickle, threading
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from flask import Flask, render_template, request, jsonify
import cv2, face_recognition
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from supabase import create_client, Client

load_dotenv()

# ==== Konfigurasi Supabase ====
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

app = Flask(__name__)

THRESHOLD = 0.60
DS_LOCK = threading.Lock()

# ---------- Helpers ----------
def b64_to_image(b64_data: str):
    if "," in b64_data:
        b64_data = b64_data.split(",", 1)[1]
    img_bytes = base64.b64decode(b64_data)
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    return cv2.imdecode(img_array, cv2.IMREAD_COLOR)

# ---------- Supabase Wrappers ----------
def load_dataset() -> Tuple[np.ndarray, np.ndarray]:
    with DS_LOCK:
        data = supabase.table("embeddings").select("*").execute().data or []
        if not data:
            return np.empty((0, 128)), np.array([])
        X = np.array([r["embedding"] for r in data], dtype=np.float64)
        y = np.array([r["name"] for r in data], dtype=object)
        return X, y

def append_embedding(name: str, enc: np.ndarray):
    with DS_LOCK:
        rec = {"name": name, "embedding": enc.tolist()}
        supabase.table("embeddings").insert(rec).execute()

def delete_embeddings(by_id=None, by_name=None) -> int:
    with DS_LOCK:
        if by_id:
            res = supabase.table("embeddings").delete().eq("id", by_id).execute()
        elif by_name:
            res = supabase.table("embeddings").delete().eq("name", by_name).execute()
        else:
            return 0
        return len(res.data or [])

def list_embeddings():
    with DS_LOCK:
        res = supabase.table("embeddings").select("id,name,ts").execute()
        return res.data or []

# ---------- Pages ----------
@app.get("/")
def index_page(): return render_template("index.html")

@app.get("/register")
def register_page(): return render_template("register.html")

@app.get("/identify")
def identify_page(): return render_template("identify.html")

@app.get("/about")
def about_page(): return render_template("about.html")

# ---------- Registrasi ----------
@dataclass
class RegState:
    name: str
    step: int = 0
    need: List[int] = None
    got: List[int] = None
    def __post_init__(self):
        if self.need is None: self.need = [5,5,5,5]
        if self.got is None: self.got = [0,0,0,0]
reg_session: RegState|None = None

@app.post("/api/start_register")
def api_start_register():
    global reg_session
    name = request.json.get("name", "").strip()
    if not name:
        return jsonify({"ok": False, "error": "Nama wajib diisi"}), 400
    reg_session = RegState(name=name)
    return jsonify({"ok": True, "step": reg_session.step, "need": reg_session.need, "got": reg_session.got})

@app.post("/api/register_frame")
def api_register_frame():
    global reg_session
    if reg_session is None:
        return jsonify({"ok": False, "error": "Belum start register"}), 400

    img_b64 = request.json.get("image")
    if not img_b64:
        return jsonify({"ok": False, "error": "Frame kosong"}), 400

    img = b64_to_image(img_b64)
    if img is None:
        return jsonify({"ok": False, "error": "Gagal decode frame"}), 400

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb, model="hog")

    if len(boxes) == 0:
        return jsonify({
            "ok": True, "accepted": False, "reason": "Wajah tidak terdeteksi.",
            "step": reg_session.step, "need": reg_session.need, "got": reg_session.got
        })

    encs = face_recognition.face_encodings(rgb, boxes)
    enc = encs[0]
    step = reg_session.step
    accepted = True
    reason = "ok"

    lmarks_list = face_recognition.face_landmarks(rgb)
    if not lmarks_list:
        accepted = False
        reason = "Landmark wajah tidak terdeteksi."
    else:
        lmarks = lmarks_list[0]

        # === STEP 3 → Kedip ===
        if step == 3:
            def eye_aspect_ratio(landmarks) -> float:
                eye = landmarks.get('left_eye') or landmarks.get('right_eye')
                if not eye or len(eye) < 6: return 1.0
                def dist(p, q): return np.linalg.norm(np.array(p)-np.array(q))
                A = dist(eye[1], eye[5]); B = dist(eye[2], eye[4]); C = dist(eye[0], eye[3])
                return float((A + B) / (2.0 * C + 1e-6))
            ear = eye_aspect_ratio(lmarks)
            if ear > 0.25:
                accepted = False
                reason = "Coba berkedip (tutup mata sesaat)."

        # === STEP 1 dan 2 → Hadap Kanan / Kiri ===
        elif step in (1, 2):
            (top, right, bottom, left) = boxes[0]
            cx_face = (left + right) / 2.0
            if "nose_tip" in lmarks:
                nose = np.mean(np.array(lmarks["nose_tip"]), axis=0)
                face_width = right - left
                offset = face_width * 0.06
                if step == 1 and nose[0] < cx_face + offset:
                    accepted = False; reason = "Arahkan wajah ke KANAN (sedikit lagi)."
                if step == 2 and nose[0] > cx_face - offset:
                    accepted = False; reason = "Arahkan wajah ke KIRI (sedikit lagi)."

    if accepted:
        append_embedding(reg_session.name, enc)
        reg_session.got[step] += 1
        if reg_session.got[step] >= reg_session.need[step]:
            reg_session.step += 1

    done = reg_session.step >= len(reg_session.need)
    return jsonify({
        "ok": True,
        "accepted": accepted,
        "reason": reason,
        "step": reg_session.step,
        "need": reg_session.need,
        "got": reg_session.got,
        "done": done
    })

# ---------- Training ----------
@app.post("/api/train")
def api_train():
    X, y = load_dataset()
    if len(y) == 0:
        return jsonify({"ok": False, "error": "Dataset kosong."}), 400
    if len(set(y)) < 2:
        return jsonify({"ok": False, "error": "Minimal 2 kelas."}), 400

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    clf = make_pipeline(StandardScaler(with_mean=False), LinearSVC())
    clf.fit(X, y_enc)

    os.makedirs("models", exist_ok=True)
    with open("models/svm.pkl", "wb") as f:
        pickle.dump({"pipeline": clf, "label_encoder": le}, f)
    return jsonify({"ok": True, "classes": list(le.classes_)})

# ---------- Identifikasi (pakai SVM) ----------
@app.post("/api/identify")
def api_identify():
    try:
        img_b64 = request.json.get("image")
        if not img_b64:
            return jsonify({"ok": False, "error": "Frame kosong"}), 400
        img = b64_to_image(img_b64)
        if img is None:
            return jsonify({"ok": False, "error": "Gagal decode gambar"}), 400

        # pastikan model tersedia
        model_path = "models/svm.pkl"
        if not os.path.exists(model_path):
            return jsonify({"ok": False, "error": "Model belum dilatih. Jalankan /api/train dulu."}), 400

        with open(model_path, "rb") as f:
            model_data = pickle.load(f)
        clf = model_data["pipeline"]; le = model_data["label_encoder"]

        boxes, encs = compute_embedding(img)
        if len(boxes) == 0 or len(encs) == 0:
            return jsonify({"ok": True, "faces": []})

        results = []
        for (top, right, bottom, left), enc in zip(boxes, encs):
            try:
                pred = clf.decision_function([enc])
                idx = int(np.argmax(pred))
                name = le.inverse_transform([idx])[0]
                confidence = float(np.max(pred))
            except Exception:
                name = "Unknown"; confidence = 0.0

            results.append({
                "top": int(top), "right": int(right),
                "bottom": int(bottom), "left": int(left),
                "name": name, "confidence": round(confidence, 2)
            })

        return jsonify({"ok": True, "faces": results})
    except Exception as e:
        print("❌ Error /api/identify:", e)
        return jsonify({"ok": False, "error": str(e)}), 500

def compute_embedding(img_bgr):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb, model="hog")
    encs = face_recognition.face_encodings(rgb, boxes)
    return boxes, encs

# ---------- Dataset ----------
@app.get("/api/embeddings")
def api_embeddings_list():
    items = list_embeddings()
    unique_names = sorted(set(r["name"] for r in items if r.get("name")))
    return jsonify({"ok": True, "names": unique_names})

@app.delete("/api/embeddings")
def api_embeddings_delete():
    data = request.get_json(silent=True) or {}
    by_id = data.get("id"); by_name = data.get("name")
    n = delete_embeddings(by_id=by_id, by_name=by_name)
    return jsonify({"ok": True, "deleted": n})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
