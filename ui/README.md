# ⚽ Football AI Dashboard

**NEW STREAMLIT UI - REPLACES REACT FRONTEND**

This is the new primary frontend for the Football AI project, built with Streamlit. The React frontend on port 3001 is now deprecated.

## 🚀 Quick Start

### Prerequisites
- Football AI API running on `localhost:3000`
- Python 3.8+ with required packages

### Installation
```bash
# Install UI dependencies
pip install streamlit plotly pandas requests

# Start the backend API (in separate terminal)
python api/main.py

# Start the NEW Streamlit UI on port 3001 (replaces React)
streamlit run ui/app.py --server.port 3001
```

### Environment Configuration
```bash
# Optional: Custom API URL (default: http://localhost:3000)
export FOOTBALL_API_URL="http://your-api-server:3000"
```

## 🏗️ Architecture

```
ui/
├── app.py                      # Main Streamlit application
├── api_client.py               # Clean REST client for FastAPI backend
├── components/                 # Modular tab components
│   ├── __init__.py
│   ├── tab_overview.py         # System overview & health
│   ├── tab_single_match.py     # Single match predictions
│   ├── tab_next_round.py       # Next round predictions
│   ├── tab_league_explorer.py  # League explorer with filters
│   ├── tab_scoreline_lab.py    # Scoreline analysis & heatmaps
│   ├── tab_model_system.py     # Model info & system health
│   └── tab_api_explorer.py     # API endpoint testing
└── README.md                   # This file
```

## 📊 Dashboard Features

### 🏠 Overview Tab
- System health status (healthy/degraded/unhealthy)
- Quick stats (teams, matches, features)
- Models overview with accuracy metrics
- Real-time API connection monitoring

### ⚽ Single Match Tab
- **Input**: Home/Away teams, league, optional date
- **Hybrid 1X2**: ML + Scoreline + Poisson + Draw Specialist visualization
- **OU2.5 & BTTS**: Interactive donut charts
- **FII Score**: Football Intelligence Index gauge
- **Source Breakdown**: Model contribution weights
- **Raw JSON**: Debug response viewer

### 📅 Next Round Tab
- League selection from API
- Complete round predictions table
- Confidence-based filtering and color coding
- Statistical summaries and charts
- Hybrid model usage indicators

### 🌍 League Explorer Tab
- Browse all available leagues
- Apply filters (confidence, result type, sorting)
- Multi-league comparison
- Enhanced match filtering and statistics

### 🎯 Scoreline Lab Tab
- Detailed scoreline probability analysis
- 5x5 heatmap visualization (0-4 goals each team)
- Top 10 most probable exact scores
- Derived metrics from scoreline matrix
- Graceful fallback when scoreline data unavailable

### 📊 Models & System Tab
- Complete system health monitoring
- Model performance metrics and accuracy charts
- Top teams by Elo rating with visualizations
- Performance statistics (response times, cache rates)
- Raw API response debugging

### 🧪 API Explorer Tab
- Direct endpoint testing
- GET endpoints: /health, /stats, /models, /teams, /predict/leagues
- POST endpoints: /predict/improved, /predict with JSON editor
- Formatted response display
- Raw JSON debugging

## 🔧 Technical Details

### API Client Features
- **Error Handling**: Comprehensive HTTP error management with user-friendly messages
- **Timeout Management**: 30-second configurable timeouts
- **Environment Config**: FOOTBALL_API_URL environment variable support
- **Safe Responses**: All methods return {"ok": True/False, "data": {}, "error": "..."} format

### UI Components
- **Defensive Programming**: All field access uses .get() with defaults to prevent crashes
- **Graceful Degradation**: Handles missing API fields elegantly
- **Interactive Visualizations**: Plotly charts with hover effects and responsive design
- **Custom Styling**: Professional CSS with gradients and modern UI elements

### Data Safety
- **No Crashes**: UI never crashes due to missing API fields
- **Default Values**: Safe fallbacks for all data access patterns
- **Error Display**: User-friendly error messages instead of exceptions
- **Debug Support**: Raw JSON viewers for troubleshooting

## 🚨 Important Notes

- **React Frontend Deprecated**: The old React frontend on port 3001 is no longer used
- **Single Frontend**: This Streamlit UI is now the ONLY frontend for the project
- **Port 3001**: Run with `streamlit run ui/app.py --server.port 3001`
- **Backend Unchanged**: FastAPI backend remains on port 3000 (DO NOT MODIFY)

## 🐛 Troubleshooting

### Common Issues
1. **API Connection Failed**: Ensure FastAPI backend is running on port 3000
2. **Missing Data**: Check API response format in Raw JSON expanders
3. **Port Conflicts**: Kill any existing processes on port 3001 before starting

### Debug Mode
- Use Raw JSON expanders in each tab to inspect API responses
- Check browser console for JavaScript errors
- Enable Streamlit debug: `streamlit run ui/app.py --logger.level=debug --server.port 3001`

## 🔄 Migration from React

The React frontend has been completely replaced. Key differences:

- **Better Error Handling**: No more crashes on missing API fields
- **More Features**: Scoreline Lab, League Explorer, API Explorer
- **Better UX**: Professional styling, interactive charts, real-time status
- **Easier Maintenance**: Python-based, modular component architecture
- **Debug Tools**: Built-in API response inspection and testing tools
- Batch prediction functionality
- Real-time ESPN fixture data

### 🔧 Tab 4: Model Health
- System health monitoring
- Model information and metrics
- Training status and data freshness
- Service statistics

### 🔍 Tab 5: API Explorer
- Test any API endpoint interactively
- Configure HTTP method and JSON body
- Quick access to common endpoints
- View formatted JSON responses

## 🎨 UI Components

- **Interactive Charts:** Plotly visualizations
- **Real-time Data:** Live API integration
- **Error Handling:** Graceful error display
- **Loading States:** Progress indicators
- **Color Coding:** Probability-based styling

## 🛠️ Development

### File Structure
```
ui/
├── app.py              → Main Streamlit application
├── api_client.py       → Backend API communication
└── README.md           → This file
```

### Configuration
The dashboard connects to the backend API at `http://localhost:3000` by default. 
To change this, modify the `base_url` in `api_client.py`.

### Dependencies
- **streamlit**: Web app framework
- **plotly**: Interactive charts
- **pandas**: Data manipulation
- **requests**: HTTP client

## 🔧 Troubleshooting

### Common Issues

**"Cannot connect to API server"**
- Make sure the backend is running: `python3 api/main.py`
- Check if port 3000 is available
- Verify API health: `curl http://localhost:3000/health`

**"Module not found" errors**
- Install dependencies: `pip install streamlit plotly pandas requests`
- Make sure you're in the correct directory

**Charts not displaying**
- Update Plotly: `pip install --upgrade plotly`
- Clear browser cache
- Try refreshing the page

### Performance Tips
- Use the team search helper for accurate team names
- The dashboard caches API responses for better performance
- Large league rounds may take a few seconds to load

## 📝 Usage Examples

### Single Match Prediction
1. Go to "Predict Single Match" tab
2. Select "Premier League" from dropdown
3. Enter "Manchester City" as home team
4. Enter "Liverpool" as away team
5. Click "Predict Match"
6. View interactive charts and results

### Next Round Analysis
1. Go to "Next Round Predictions" tab
2. Select "2025-26-english-premier-league"
3. Click "Predict Next Round"
4. Review the complete round table
5. Analyze statistics and trends

### API Testing
1. Go to "API Explorer" tab
2. Select "GET" method
3. Enter "/health" endpoint
4. Click "Send Request"
5. View the JSON response

## 🎯 Features in Detail

### Interactive Visualizations
- **Bar Charts**: Show 1X2 probability distributions
- **Donut Charts**: Display OU2.5 over/under splits
- **Gauge Charts**: Visualize BTTS probabilities
- **Data Tables**: Color-coded probability tables

### Real-time Integration
- **Live API**: Real-time backend communication
- **Error Recovery**: Automatic retry on failures
- **Status Monitoring**: Connection health indicators
- **Progress Tracking**: Loading states for operations

---

Built with ❤️ using Streamlit and the Football AI API
