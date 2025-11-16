#!/bin/bash
# Stop all Football AI services

echo "🛑 Stopping all Football AI services..."

# Kill backend (port 8000)
echo "Stopping Backend API (port 8000)..."
lsof -ti:8000 | xargs kill -9 2>/dev/null || true

# Kill frontend (port 3000)
echo "Stopping React Frontend (port 3000)..."
lsof -ti:3000 | xargs kill -9 2>/dev/null || true

# Kill UI (port 8501)
echo "Stopping Streamlit UI (port 8501)..."
lsof -ti:8501 | xargs kill -9 2>/dev/null || true

echo "✅ All services stopped"
