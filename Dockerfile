# ===============================
# ⚡ Fast & Lightweight Build for Render
# ===============================
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies only for runtime (no compilers)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libopenblas-dev \
    liblapack-dev \
 && rm -rf /var/lib/apt/lists/*

# Upgrade pip & wheel
RUN pip install --no-cache-dir --upgrade pip wheel setuptools

# Copy requirements first
COPY requirements.txt .

# ✅ Preinstall from prebuilt wheels only (avoid compiling dlib)
RUN pip install --no-cache-dir --only-binary=:all: \
    numpy==1.26.4 \
    scipy \
    scikit-learn \
    opencv-python-headless \
    flask \
    gunicorn \
    supabase \
    python-dotenv

# ✅ Install face-recognition from GitHub (includes prebuilt dlib wheel)
RUN pip install --no-cache-dir \
    "git+https://github.com/ageitgey/face_recognition.git" \
    "git+https://github.com/ageitgey/face_recognition_models.git"

# Copy project files
COPY . .

# Default command
CMD gunicorn app:app --bind 0.0.0.0:${PORT:-10000}
