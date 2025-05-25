#!/bin/bash

# Exit on any error
set -e

echo "Starting TryOnDiffusion API..."

# Set environment variables
export PYTHONPATH=/app:$PYTHONPATH
export ENVIRONMENT=${ENVIRONMENT:-production}

# Create necessary directories
mkdir -p /app/models
mkdir -p /app/logs

# Start the FastAPI server
exec python -m uvicorn tryon_api:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --access-log 