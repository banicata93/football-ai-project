#!/bin/bash
# Start all Football AI services

echo "🚀 Starting all Football AI services..."
echo ""
echo "Port Configuration:"
echo "  - Backend API:      http://localhost:8000"
echo "  - React Frontend:   http://localhost:3000"
echo "  - Streamlit UI:     http://localhost:8501"
echo ""

# Stop any existing services
./stop_all.sh

sleep 2

# Start backend in background
echo "Starting Backend API..."
./start_backend.sh > logs/backend.log 2>&1 &
BACKEND_PID=$!

# Wait for backend to start
echo "Waiting for backend to start..."
sleep 5

# Check if backend is running
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✅ Backend API started successfully"
else
    echo "❌ Backend API failed to start"
    exit 1
fi

# Start frontend in background
echo "Starting React Frontend..."
./start_frontend.sh > logs/frontend.log 2>&1 &
FRONTEND_PID=$!

# Start UI in background
echo "Starting Streamlit UI..."
./start_ui.sh > logs/ui.log 2>&1 &
UI_PID=$!

echo ""
echo "✅ All services started!"
echo ""
echo "Access points:"
echo "  - API Docs:         http://localhost:8000/docs"
echo "  - React Frontend:   http://localhost:3000"
echo "  - Streamlit UI:     http://localhost:8501"
echo ""
echo "Process IDs:"
echo "  - Backend:  $BACKEND_PID"
echo "  - Frontend: $FRONTEND_PID"
echo "  - UI:       $UI_PID"
echo ""
echo "To stop all services, run: ./stop_all.sh"
