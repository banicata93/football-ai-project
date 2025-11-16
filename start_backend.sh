#!/bin/bash
# Start Football AI Backend API on port 8000

echo "🚀 Starting Football AI Backend API on port 8000..."

# Kill any existing process on port 8000
lsof -ti:8000 | xargs kill -9 2>/dev/null || true

# Wait a moment
sleep 1

# Start the backend
cd "$(dirname "$0")"
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
