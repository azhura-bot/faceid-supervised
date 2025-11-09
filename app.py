import os, base64, pickle, threading, time
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from collections import deque, Counter
from flask import Flask, render_template, request, jsonify
import cv2, face_recognition
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from supabase import create_client, Client

load_dotenv()

# ==== Konfigurasi Supabase ====
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

app = Flask(__name__)

STABILIZER_HISTORY = {}  
MAX_HISTORY = 5          
THRESHOLD = 70           
DS_LOCK = threading.Lock()
TRAIN_STATUS = {
    "running": False,
    "done": False,
    "error": None,
    "result": None
}

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

# ---------- Training & Evaluasi ----------
@app.post("/api/train")
def api_train():
    global TRAIN_STATUS
    if TRAIN_STATUS["running"]:
        return jsonify({"ok": False, "error": "Training sedang berjalan."}), 400

    TRAIN_STATUS = {"running": True, "done": False, "error": None, "result": None}

    def background_train():
        global TRAIN_STATUS
        try:
            X, y = load_dataset()
            if len(y) == 0:
                raise ValueError("Dataset kosong.")
            if len(set(y)) < 2:
                raise ValueError("Minimal 2 kelas untuk training.")

            le = LabelEncoder()
            y_enc = le.fit_transform(y)
            X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.3, random_state=42)

            kernels = ['linear', 'rbf', 'poly']
            results = {}

            for kernel in kernels:
                start = time.time()
                model = make_pipeline(StandardScaler(with_mean=False), SVC(kernel=kernel, probability=True))
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                duration = time.time() - start
                results[kernel] = {
                    "accuracy": round(acc * 100, 2),
                    "train_time_sec": round(duration, 3)
                }

            best_kernel = max(results, key=lambda k: results[k]["accuracy"])
            best_model = make_pipeline(StandardScaler(with_mean=False), SVC(kernel=best_kernel, probability=True))
            best_model.fit(X, y_enc)

            os.makedirs("models", exist_ok=True)
            with open("models/svm.pkl", "wb") as f:
                pickle.dump({"pipeline": best_model, "label_encoder": le}, f)

            avg_acc = np.mean([r["accuracy"] for r in results.values()])

            TRAIN_STATUS["result"] = {
                "classes": list(le.classes_),
                "kernels": results,
                "best_kernel": best_kernel,
                "best_acc": results[best_kernel]["accuracy"],
                "avg_acc": round(avg_acc, 2)
            }
            TRAIN_STATUS["done"] = True
            TRAIN_STATUS["running"] = False
        except Exception as e:
            TRAIN_STATUS["error"] = str(e)
            TRAIN_STATUS["running"] = False
            TRAIN_STATUS["done"] = True

    threading.Thread(target=background_train, daemon=True).start()
    return jsonify({"ok": True, "message": "Training dimulai di background"})

@app.get("/api/train_status")
def api_train_status():
    global TRAIN_STATUS
    return jsonify({
        "ok": True,
        "running": TRAIN_STATUS["running"],
        "done": TRAIN_STATUS["done"],
        "error": TRAIN_STATUS["error"],
        "result": TRAIN_STATUS["result"]
    })

# ---------- Identifikasi (pakai probabilitas) ----------
@app.post("/api/identify")
def api_identify():
    try:
        img_b64 = request.json.get("image")
        if not img_b64:
            return jsonify({"ok": False, "error": "Frame kosong"}), 400

        img = b64_to_image(img_b64)
        if img is None:
            return jsonify({"ok": False, "error": "Gagal decode gambar"}), 400

        model_path = "models/svm.pkl"
        if not os.path.exists(model_path):
            return jsonify({"ok": False, "error": "Model belum dilatih. Jalankan /api/train dulu."}), 400

        # 🔹 Load model
        with open(model_path, "rb") as f:
            model_data = pickle.load(f)
        clf = model_data["pipeline"]
        le = model_data["label_encoder"]

        start_detect = time.time()
        boxes, encs = compute_embedding(img)
        detect_time = time.time() - start_detect

        # Tidak ada wajah
        if len(boxes) == 0 or len(encs) == 0:
            return jsonify({"ok": True, "faces": []})

        results = []

        for (top, right, bottom, left), enc in zip(boxes, encs):
            start_pred = time.time()

            # 🔹 Gunakan probabilitas untuk confidence
            probs = clf.predict_proba([enc])[0]
            pred_time = time.time() - start_pred

            idx = int(np.argmax(probs))
            name = le.inverse_transform([idx])[0]
            confidence = float(np.max(probs)) * 100

            # 🔹 Simpan history untuk smoothing
            if name not in STABILIZER_HISTORY:
                STABILIZER_HISTORY[name] = deque(maxlen=MAX_HISTORY)
            STABILIZER_HISTORY[name].append(confidence)

            # Rata-rata smoothing confidence
            smooth_conf = np.mean(STABILIZER_HISTORY[name])

            # Majority voting untuk nama dominan
            votes = Counter()
            for n, confs in STABILIZER_HISTORY.items():
                votes[n] = np.mean(confs)
            stable_name = votes.most_common(1)[0][0]

            # 🔹 Threshold: jika kurang dari ambang → Unknown
            final_name = stable_name if smooth_conf >= THRESHOLD else "Unknown"

            results.append({
                "top": int(top), "right": int(right),
                "bottom": int(bottom), "left": int(left),
                "name": final_name,
                "confidence": round(smooth_conf, 1),
                "detect_time": round(detect_time, 3),
                "predict_time": round(pred_time, 3)
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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
