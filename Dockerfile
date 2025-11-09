# =======================================
# 🚀 Ultra-lightweight build for Render Free
# =======================================
FROM python:3.10-slim

WORKDIR /app

# Install only essential runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Copy requirements first for better caching
COPY requirements.txt .

# Install in one layer to reduce memory usage
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY . .

# Start app
CMD gunicorn app:app --bind 0.0.0.0:${PORT:-10000}