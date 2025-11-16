#!/bin/bash
# Start Football AI Streamlit UI on port 8501

echo "📊 Starting Football AI Streamlit UI on port 8501..."

# Kill any existing process on port 8501
lsof -ti:8501 | xargs kill -9 2>/dev/null || true

# Wait a moment
sleep 1

# Start the UI
cd "$(dirname "$0")"
streamlit run ui/app.py --server.port 8501 --server.address 0.0.0.0
