# ⚡ Quick Start Guide - Football AI Service

Get the complete system running in 5 minutes!

---

## 🎯 What You'll Get

- **Backend API** running on `http://localhost:8000`
- **Frontend Dashboard** running on `http://localhost:3000`
- **Interactive predictions** with beautiful visualizations

---

## 📋 Prerequisites

- Python 3.8+
- Node.js 16+
- npm

---

## 🚀 Start Backend (Terminal 1)

```bash
# Navigate to project root
cd football_ai_service

# Start FastAPI server
python api/main.py
```

**Expected output:**
```
INFO - ✓ Poisson model зареден
INFO - ✓ 1X2 model зареден
INFO - ✓ OU2.5 model зареден
INFO - ✓ BTTS model зареден
INFO - ✓ Ensemble model зареден
INFO - ✓ FII calculator зареден
INFO - Team data заредени за 2942 отбора
INFO - Стартиране на FastAPI сървър...
INFO - Application startup complete.
INFO - Uvicorn running on http://127.0.0.1:8000
```

✅ **Backend is ready when you see:** `Uvicorn running on http://127.0.0.1:8000`

---

## 🎨 Start Frontend (Terminal 2)

```bash
# Navigate to frontend directory
cd football_ai_service/frontend

# Install dependencies (first time only)
npm install

# Copy environment file (first time only)
cp .env.example .env

# Start development server
npm start
```

**Expected output:**
```
Compiled successfully!

You can now view football-ai-frontend in the browser.

  Local:            http://localhost:3000
  On Your Network:  http://192.168.1.x:3000
```

✅ **Frontend is ready when browser opens automatically at:** `http://localhost:3000`

---

## 🎮 Test the System

### 1. Check Health Status
- Look at the header in the frontend
- Should see green "Online" indicator with pulse dot

### 2. Make a Prediction

**Step 1:** Select Home Team
- Click "Home Team" input
- Type "Manchester United"
- Select from dropdown

**Step 2:** Select Away Team
- Click "Away Team" input  
- Type "Liverpool"
- Select from dropdown

**Step 3:** (Optional) Select League
- Choose "Premier League" from dropdown

**Step 4:** Predict
- Click "🔮 Predict Match" button
- Wait 1-2 seconds

**Step 5:** View Results
- See 1X2 donut chart
- See Over/Under 2.5 bar chart
- See BTTS radial gauges
- See FII gauge (0-10)

### 3. Explore Features

**Dark Mode:**
- Click moon/sun icon in header
- Theme toggles instantly

**View Details:**
- Click "🔍 View Full Prediction Details"
- See complete JSON response

**Try Different Teams:**
- Make predictions for various matches
- Compare results

---

## 🔍 Verify Everything Works

### Backend Checks

```bash
# Health check
curl http://localhost:8000/health

# Get teams list
curl http://localhost:8000/teams | head -20

# Test prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"home_team":"Barcelona","away_team":"Real Madrid"}'
```

### Frontend Checks

1. **Open browser:** http://localhost:3000
2. **Check console:** No errors (F12 → Console)
3. **Test search:** Type in team input, see dropdown
4. **Make prediction:** Should see charts
5. **Toggle dark mode:** Should work smoothly

---

## 🐛 Troubleshooting

### Backend Won't Start

**Problem:** Port 8000 already in use
```bash
# Kill process on port 8000
lsof -ti:8000 | xargs kill -9

# Restart backend
python api/main.py
```

**Problem:** Module not found
```bash
# Install dependencies
pip install -r requirements.txt
```

### Frontend Won't Start

**Problem:** Port 3000 already in use
```bash
# Kill process on port 3000
lsof -ti:3000 | xargs kill -9

# Or use different port
PORT=3001 npm start
```

**Problem:** npm install fails
```bash
# Clear cache
npm cache clean --force
rm -rf node_modules package-lock.json
npm install
```

### Connection Issues

**Problem:** Frontend shows "Offline"
- Check backend is running on port 8000
- Check CORS is enabled (already configured)
- Check no firewall blocking

**Problem:** CORS errors
- Backend already has CORS configured
- If still issues, check browser console

---

## 📊 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Browser (localhost:3000)                 │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │           React Frontend Dashboard                   │  │
│  │  • Team Search                                       │  │
│  │  • Prediction Form                                   │  │
│  │  • Interactive Charts (Recharts)                     │  │
│  │  • Dark Mode Toggle                                  │  │
│  └──────────────────────────────────────────────────────┘  │
│                          │                                   │
│                          │ HTTP/REST API                     │
│                          ▼                                   │
└─────────────────────────────────────────────────────────────┘
                           │
                           │
┌─────────────────────────────────────────────────────────────┐
│              FastAPI Backend (localhost:8000)               │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │           Prediction Service                         │  │
│  │  • Load Models (Poisson, XGBoost, LightGBM)        │  │
│  │  • Feature Engineering                               │  │
│  │  • Ensemble Predictions                              │  │
│  │  • FII Calculation                                   │  │
│  └──────────────────────────────────────────────────────┘  │
│                          │                                   │
│                          ▼                                   │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Trained ML Models                       │  │
│  │  • Poisson Model (baseline)                         │  │
│  │  • XGBoost 1X2 (65.5% accuracy)                     │  │
│  │  • LightGBM OU2.5 (76.1% accuracy)                  │  │
│  │  • XGBoost BTTS (77.6% accuracy)                    │  │
│  │  • Ensemble Model                                    │  │
│  │  • Stacking Ensemble (meta-learning)                │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## 🎯 Next Steps

### Explore the API

**Swagger UI:** http://localhost:8000/docs
- Interactive API documentation
- Test all endpoints
- See request/response schemas

**ReDoc:** http://localhost:8000/redoc
- Alternative documentation
- Better for reading

### Customize Frontend

```bash
# Edit colors
frontend/tailwind.config.js

# Edit components
frontend/src/components/

# Add new features
frontend/src/App.jsx
```

### Deploy to Production

**Backend:**
```bash
# Using Gunicorn
gunicorn api.main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker
```

**Frontend:**
```bash
# Build for production
cd frontend
npm run build

# Deploy to Netlify/Vercel
# Or serve with nginx
```

---

## 📚 Documentation

- **Main README:** `README.md`
- **Frontend README:** `frontend/README.md`
- **Frontend Setup:** `frontend/SETUP.md`
- **API Docs:** http://localhost:8000/docs
- **Step Completion Docs:** `STEP*_COMPLETED.md`

---

## 🎉 You're All Set!

If you can:
- ✅ See the dashboard at http://localhost:3000
- ✅ Search for teams
- ✅ Make predictions
- ✅ See beautiful charts
- ✅ Toggle dark mode

**Then everything is working perfectly!** 🚀

Enjoy using the Football AI Prediction Service! ⚽🤖

---

**Need Help?**
- Check browser console (F12)
- Check backend logs
- Review troubleshooting section above
- Check documentation files
