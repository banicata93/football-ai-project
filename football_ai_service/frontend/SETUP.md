# 🚀 Frontend Setup Guide

Complete guide to set up and run the Football AI Prediction Service frontend.

## 📋 Prerequisites

1. **Node.js & npm**
   ```bash
   # Check if installed
   node --version  # Should be 16.x or higher
   npm --version   # Should be 8.x or higher
   ```

   If not installed, download from: https://nodejs.org/

2. **Backend API Running**
   - FastAPI backend must be running on `http://localhost:8000`
   - Start backend first:
     ```bash
     cd ..
     python api/main.py
     ```

## 🛠️ Installation Steps

### Step 1: Navigate to Frontend Directory

```bash
cd frontend
```

### Step 2: Install Dependencies

```bash
npm install
```

This will install:
- React 18
- TailwindCSS 3
- Recharts 2
- Axios
- Lucide React icons
- And all other dependencies

### Step 3: Configure Environment

```bash
# Copy example env file
cp .env.example .env

# Edit .env if needed (default should work)
# REACT_APP_API_URL=http://localhost:8000
```

### Step 4: Start Development Server

```bash
npm start
```

The app will automatically open at `http://localhost:3000`

## ✅ Verification

1. **Check Backend Connection**
   - Look for green "Online" indicator in header
   - If red "Offline", check that backend is running

2. **Test Team Search**
   - Click on "Home Team" input
   - Start typing a team name
   - Dropdown should show matching teams

3. **Make a Prediction**
   - Select "Manchester United" as Home Team
   - Select "Liverpool" as Away Team
   - Click "🔮 Predict Match"
   - Should see prediction cards with charts

## 🎨 Features to Test

### Dark Mode
- Click moon/sun icon in header
- Theme should toggle and persist on reload

### Predictions
- Try different team combinations
- Check all three prediction cards (1X2, OU2.5, BTTS)
- Verify FII gauge displays correctly

### Details View
- Click "🔍 View Full Prediction Details"
- Should show JSON data

### Responsive Design
- Resize browser window
- Layout should adapt for mobile/tablet/desktop

## 🐛 Troubleshooting

### Port 3000 Already in Use

```bash
# Kill process on port 3000
lsof -ti:3000 | xargs kill -9

# Or use different port
PORT=3001 npm start
```

### Backend Connection Failed

```bash
# Check backend is running
curl http://localhost:8000/health

# If not, start backend
cd ..
python api/main.py
```

### npm install Fails

```bash
# Clear cache and retry
npm cache clean --force
rm -rf node_modules package-lock.json
npm install
```

### Tailwind Styles Not Working

```bash
# Rebuild Tailwind
npm run build:css

# Or restart dev server
npm start
```

## 📦 Build for Production

```bash
# Create optimized production build
npm run build

# Test production build locally
npm install -g serve
serve -s build
```

## 🔧 Development Tips

### Hot Reload
- Changes to `.jsx` files reload automatically
- Changes to `.css` files reload automatically
- Changes to `.env` require restart

### Component Structure
```
src/
├── components/
│   ├── Header.jsx          # Top navigation
│   ├── PredictionForm.jsx  # Team selection form
│   └── PredictionCards.jsx # Result visualizations
├── services/
│   └── api.js              # API client
├── utils/
│   └── helpers.js          # Utility functions
└── App.jsx                 # Main component
```

### Adding New Features

1. Create component in `src/components/`
2. Import in `App.jsx`
3. Add to layout
4. Test in browser

## 🎯 Next Steps

Once frontend is running:

1. **Test all features**
   - Make predictions
   - Toggle dark mode
   - Check responsive design

2. **Customize**
   - Edit colors in `tailwind.config.js`
   - Modify components as needed
   - Add new features

3. **Deploy**
   - Build for production
   - Deploy to Netlify/Vercel
   - Configure environment variables

## 📞 Support

If you encounter issues:

1. Check backend logs
2. Check browser console (F12)
3. Verify all dependencies installed
4. Ensure ports 3000 and 8000 are available

## 🎉 Success!

If you see the dashboard with team dropdowns and can make predictions, you're all set!

Enjoy using the Football AI Prediction Service! ⚽🤖
