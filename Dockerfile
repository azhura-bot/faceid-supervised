# ================================
# 🧱 STAGE 1: Build Environment
# ================================
FROM python:3.10-slim AS builder

# Set work directory
WORKDIR /app

# Install system dependencies for compiling Python wheels (once)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libgl1 \
    libglib2.0-0 \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
 && rm -rf /var/lib/apt/lists/*

# Upgrade pip & preinstall dependencies
RUN pip install --no-cache-dir --upgrade pip wheel setuptools

# Copy requirements first (for caching)
COPY requirements.txt .

# Install Python dependencies (using binary wheels only)
RUN pip install --no-cache-dir --only-binary=:all: -r requirements.txt || \
    pip install --no-cache-dir -r requirements.txt

# ================================
# 🧩 STAGE 2: Runtime Image
# ================================
FROM python:3.10-slim

# Working directory
WORKDIR /app

# Copy project files
COPY . .

# Copy installed packages from builder stage
COPY --from=builder /usr/local/lib/python3.10 /usr/local/lib/python3.10
COPY --from=builder /usr/local/bin /usr/local/bin

# Install minimal system libs for runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Default command
CMD gunicorn app:app --bind 0.0.0.0:${PORT:-10000}
