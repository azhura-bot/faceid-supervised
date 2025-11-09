#!/usr/bin/env bash
set -o errexit

# Install requirements biasa (kecuali face-recognition)
pip install Flask numpy opencv-python-headless dlib-bin==19.24.6 scikit-learn supabase gunicorn

# Install face-recognition tapi tanpa deps agar tidak compile dlib
pip install face-recognition==1.3.0 --no-deps
