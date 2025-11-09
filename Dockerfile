# =======================================
# 🚀 Render-friendly lightweight build (no cmake)
# =======================================
FROM python:3.10-slim

WORKDIR /app

# Minimal runtime libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Upgrade pip and wheel
RUN pip install --no-cache-dir --upgrade pip wheel setuptools

# Copy requirements
COPY requirements.txt .

# Install lightweight dependencies first
RUN pip install --no-cache-dir \
    flask gunicorn numpy==1.26.4 scipy scikit-learn \
    opencv-python-headless supabase python-dotenv

# ✅ Install precompiled dlib wheel directly from PyPI or URL (bypass source build)
RUN pip install --no-cache-dir https://github.com/ageitgey/dlib-bin/releases/download/v19.24.2/dlib-19.24.2-cp310-cp310-manylinux_2_31_x86_64.whl

# ✅ Then install face_recognition (no need to build dlib)
RUN pip install --no-cache-dir face-recognition==1.3.0 face-recognition-models==0.3.0

# Copy your project
COPY . .

# Run app
CMD gunicorn app:app --bind 0.0.0.0:${PORT:-10000}
