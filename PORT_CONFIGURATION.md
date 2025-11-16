# Port Configuration Guide

## 🎯 Standard Port Configuration

All services have been standardized to use the following ports:

| Service | Port | URL | Description |
|---------|------|-----|-------------|
| **Backend API** | `8000` | http://localhost:8000 | FastAPI backend with ML models |
| **React Frontend** | `3000` | http://localhost:3000 | React web interface |
| **Streamlit UI** | `8501` | http://localhost:8501 | Streamlit dashboard |

## 🚀 Quick Start

### Start All Services
```bash
./start_all.sh
```

### Start Individual Services
```bash
# Backend only
./start_backend.sh

# Frontend only
./start_frontend.sh

# Streamlit UI only
./start_ui.sh
```

### Stop All Services
```bash
./stop_all.sh
```

## 📝 Configuration Files Updated

The following files have been updated with correct port configurations:

### Backend (Port 8000)
- ✅ `Makefile` - `run-dev` command
- ✅ `Dockerfile` - EXPOSE and HEALTHCHECK
- ✅ `api/main.py` - FastAPI app (uses uvicorn default)

### Frontend (Port 3000)
- ✅ `frontend/.env` - `REACT_APP_API_URL=http://localhost:8000`
- ✅ `frontend/.env.example` - Template with correct ports
- ✅ `frontend/src/services/api.js` - API client configuration

### Streamlit UI (Port 8501)
- ✅ `ui/api_client.py` - API URL default to `http://localhost:8000`
- ✅ `ui/app.py` - Streamlit app (uses streamlit default port)

## 🔧 Manual Start Commands

If you prefer to start services manually:

### Backend API
```bash
# Development mode with auto-reload
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Or using Makefile
make run-dev
```

### React Frontend
```bash
cd frontend
PORT=3000 npm start
```

### Streamlit UI
```bash
streamlit run ui/app.py --server.port 8501 --server.address 0.0.0.0
```

## 🐳 Docker

Docker configuration has also been updated:

```bash
# Build
docker build -t football-ai-service .

# Run
docker run -p 8000:8000 football-ai-service
```

## 🔍 Check Running Services

```bash
# Check what's running on each port
lsof -i :8000  # Backend
lsof -i :3000  # Frontend
lsof -i :8501  # UI
```

## 🛑 Kill Specific Port

```bash
# Kill backend
lsof -ti:8000 | xargs kill -9

# Kill frontend
lsof -ti:3000 | xargs kill -9

# Kill UI
lsof -ti:8501 | xargs kill -9
```

## 📊 API Endpoints

Once the backend is running on port 8000:

- **API Documentation**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health
- **Models Info**: http://localhost:8000/models
- **Predict**: http://localhost:8000/predict

## ✅ Verification

After starting all services, verify they're working:

```bash
# Check backend health
curl http://localhost:8000/health

# Check models endpoint (should return full metrics now)
curl http://localhost:8000/models

# Check frontend (should open in browser)
open http://localhost:3000

# Check Streamlit UI (should open in browser)
open http://localhost:8501
```

## 🐛 Troubleshooting

### Port Already in Use
If you get "port already in use" errors:
```bash
./stop_all.sh
# Wait a few seconds
./start_all.sh
```

### Backend Won't Start
- Check logs: `tail -f logs/backend.log`
- Verify Python dependencies: `pip install -r requirements.txt`
- Check if models are present: `ls -la models/`

### Frontend Won't Start
- Check logs: `tail -f logs/frontend.log`
- Install dependencies: `cd frontend && npm install`
- Clear cache: `cd frontend && rm -rf node_modules package-lock.json && npm install`

### UI Won't Connect to Backend
- Verify backend is running: `curl http://localhost:8000/health`
- Check `ui/api_client.py` has correct URL: `http://localhost:8000`
- Restart UI: `./stop_all.sh && ./start_ui.sh`

## 📦 Environment Variables

### Backend
No environment variables needed (uses defaults)

### Frontend
Create `frontend/.env`:
```
REACT_APP_API_URL=http://localhost:8000
PORT=3000
```

### Streamlit UI
Optional `FOOTBALL_API_URL` environment variable:
```bash
export FOOTBALL_API_URL=http://localhost:8000
streamlit run ui/app.py
```

## 🎉 All Fixed!

All port configurations have been cleaned up and standardized. The system should now work seamlessly with:
- Backend on 8000
- Frontend on 3000
- UI on 8501
