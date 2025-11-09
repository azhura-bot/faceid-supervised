# =======================================
# 🚀 Render-friendly lightweight build
# =======================================
FROM python:3.10-slim

WORKDIR /app

# Minimal system libs (tanpa compiler)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Upgrade pip & wheel
RUN pip install --no-cache-dir --upgrade pip wheel setuptools

# Copy dependencies file
COPY requirements.txt .

# Install lightweight dependencies first (no build tools)
RUN pip install --no-cache-dir \
    flask gunicorn numpy==1.26.4 scipy scikit-learn \
    opencv-python-headless supabase python-dotenv

# ✅ Install prebuilt dlib & face-recognition
RUN pip install --no-cache-dir dlib-bin==19.24.6 face-recognition==1.3.0 face-recognition-models==0.3.0

# Copy all project files
COPY . .

# Default command for Render
CMD gunicorn app:app --bind 0.0.0.0:${PORT:-10000}
