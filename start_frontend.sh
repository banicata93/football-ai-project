#!/bin/bash
# Start Football AI React Frontend on port 3000

echo "🎨 Starting Football AI React Frontend on port 3000..."

# Kill any existing process on port 3000
lsof -ti:3000 | xargs kill -9 2>/dev/null || true

# Wait a moment
sleep 1

# Start the frontend
cd "$(dirname "$0")/frontend"
PORT=3000 npm start
