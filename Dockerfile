# =======================================
# 🚀 Render-friendly lightweight build
# =======================================
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies including build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip wheel setuptools

# Copy requirements
COPY requirements.txt .

# Install all dependencies at once
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY . .

# Start app
CMD gunicorn app:app --bind 0.0.0.0:${PORT:-8080} --timeout 120 --workers 1 --threads 2