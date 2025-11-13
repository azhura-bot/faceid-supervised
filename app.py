import os, base64, pickle, threading, time, warnings
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from collections import deque
from flask import Flask, render_template, request, jsonify
import cv2, face_recognition
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import confusion_matrix
from supabase import create_client, Client

warnings.filterwarnings("ignore", category=UserWarning)
load_dotenv()

# ==== Konfigurasi Supabase ====
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

app = Flask(__name__)

# ==== Konstanta ====
STABILIZER_HISTORY = {}
MAX_HISTORY = 5
THRESHOLD = 70
DS_LOCK = threading.Lock()
TRAIN_STATUS = {"running": False, "done": False, "error": None, "result": None}

MODEL_PATH = "models/svm.pkl"



# ---------- Helpers ----------
def b64_to_image(b64_data: str):
    if "," in b64_data:
        b64_data = b64_data.split(",", 1)[1]
    img_bytes = base64.b64decode(b64_data)
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    return cv2.imdecode(img_array, cv2.IMREAD_COLOR)


def compute_embedding(img_bgr):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb, model="hog")
    encs = face_recognition.face_encodings(rgb, boxes)
    return boxes, encs



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
        if by_name:
            res = supabase.table("embeddings").delete().eq("name", by_name).execute()
        elif by_id:
            res = supabase.table("embeddings").delete().eq("id", by_id).execute()
        else:
            return 0

        # === FIX: Hapus model lama supaya nama lama hilang dari SVM ===
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)

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
        if self.need is None: self.need = [5, 5, 5, 5]
        if self.got is None: self.got = [0, 0, 0, 0]

reg_session: RegState | None = None


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
        return jsonify({"ok": True, "accepted": False, "reason": "Wajah tidak terdeteksi.",
                        "step": reg_session.step, "need": reg_session.need, "got": reg_session.got})

    encs = face_recognition.face_encodings(rgb, boxes)
    if not encs:
        return jsonify({"ok": False, "error": "Gagal mendapatkan encoding wajah"}), 400

    enc = encs[0]
    step = reg_session.step
    append_embedding(reg_session.name, enc)
    reg_session.got[step] += 1
    if reg_session.got[step] >= reg_session.need[step]:
        reg_session.step += 1

    done = reg_session.step >= len(reg_session.need)
    return jsonify({"ok": True, "accepted": True, "reason": "ok",
                    "step": reg_session.step, "need": reg_session.need,
                    "got": reg_session.got, "done": done})



# ---------- Training ----------
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

            kernels = ["linear", "rbf", "poly"]
            results = {}

            kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

            for kernel in kernels:
                start = time.time()
                model = make_pipeline(
                    StandardScaler(with_mean=False),
                    SVC(kernel=kernel, probability=True)
                )
                acc_scores = cross_val_score(model, X, y_enc, cv=kf, scoring="accuracy")
                acc_mean = np.mean(acc_scores)

                model.fit(X, y_enc)
                y_pred = model.predict(X)
                cm = confusion_matrix(y_enc, y_pred)
                correct_ratio = np.trace(cm) / np.sum(cm)

                results[kernel] = {
                    "accuracy": round(acc_mean * 100, 2),
                    "train_time_sec": round(time.time() - start, 3),
                    "confusion_matrix": round(correct_ratio * 100, 2)
                }

            best_kernel = max(results, key=lambda k: results[k]["accuracy"])
            best_model = make_pipeline(
                StandardScaler(with_mean=False),
                SVC(kernel=best_kernel, probability=True)
            )
            best_model.fit(X, y_enc)

            os.makedirs("models", exist_ok=True)
            with open(MODEL_PATH, "wb") as f:
                pickle.dump({"pipeline": best_model, "label_encoder": le}, f)

            TRAIN_STATUS["result"] = {"table": results}

        except Exception as e:
            TRAIN_STATUS["error"] = str(e)
        finally:
            TRAIN_STATUS["running"] = False
            TRAIN_STATUS["done"] = True

    threading.Thread(target=background_train, daemon=True).start()
    return jsonify({"ok": True, "message": "Training dimulai di background"})


@app.get("/api/train_status")
def api_train_status():
    return jsonify({"ok": True, **TRAIN_STATUS})



# ---------- Identifikasi ----------
@app.post("/api/identify")
def api_identify():
    try:
        img_b64 = request.json.get("image")
        if not img_b64:
            return jsonify({"ok": False, "error": "Frame kosong"}), 400

        img = b64_to_image(img_b64)

        # === FIX: Jika model sudah dihapus, langsung Unknown ===
        if not os.path.exists(MODEL_PATH):
            return jsonify({"ok": True, "faces": []})  

        with open(MODEL_PATH, "rb") as f:
            model_data = pickle.load(f)

        clf = model_data["pipeline"]
        le = model_data["label_encoder"]

        boxes, encs = compute_embedding(img)
        if len(boxes) == 0 or len(encs) == 0:
            return jsonify({"ok": True, "faces": []})

        results = []
        for (top, right, bottom, left), enc in zip(boxes, encs):

            probs = clf.predict_proba([enc])[0]
            idx = int(np.argmax(probs))
            confidence = float(np.max(probs)) * 100
            name = le.inverse_transform([idx])[0]

            # Threshold
            if confidence < THRESHOLD:
                name = "Unknown"

            results.append({
                "top": int(top), "right": int(right),
                "bottom": int(bottom), "left": int(left),
                "name": name,
                "confidence": round(confidence, 1)
            })

        return jsonify({"ok": True, "faces": results})

    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500




# ---------- Dataset Management ----------
@app.get("/api/embeddings")
def api_list_embeddings():
    try:
        data = list_embeddings()
        names = sorted(set([r["name"] for r in data])) if data else []
        return jsonify({"ok": True, "names": names})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.delete("/api/embeddings")
def api_delete_embeddings_api():
    try:
        body = request.get_json(force=True)
        name = body.get("name")
        if not name:
            return jsonify({"ok": False, "error": "Nama wajib diisi"}), 400

        deleted = delete_embeddings(by_name=name)

        return jsonify({"ok": True, "deleted": deleted, "model_reset": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500




if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
