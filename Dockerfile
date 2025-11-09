# Base image ringan
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy semua file ke container
COPY . .

# Install dependency sistem untuk dlib, OpenCV, dan build tools
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    g++ \
    libgl1 \
    libglib2.0-0 \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
 && rm -rf /var/lib/apt/lists/*

# Install pip dependencies
RUN pip install --upgrade pip && \
    pip install wheel setuptools && \
    pip install -r requirements.txt

# Jalankan aplikasi Flask lewat Gunicorn
CMD gunicorn app:app --bind 0.0.0.0:${PORT:-10000}
