# ⚽ Football AI Prediction Service - Frontend

Modern, responsive React dashboard for visualizing AI-powered football match predictions.

## 🎨 Features

- **Interactive Predictions**: Beautiful visualizations using Recharts
- **Real-time Updates**: Auto-refresh health status every 60 seconds
- **Dark Mode**: Toggle between light and dark themes
- **Responsive Design**: Works perfectly on desktop and mobile
- **Team Search**: Auto-complete dropdown for easy team selection
- **Prediction History**: Stores last 5 predictions in localStorage
- **Detailed Analytics**: View full JSON prediction data

## 🚀 Quick Start

### Prerequisites

- Node.js 16+ and npm
- FastAPI backend running on `http://localhost:8000`

### Installation

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Create .env file
cp .env.example .env

# Start development server
npm start
```

The app will open at `http://localhost:3000`

## 📦 Tech Stack

- **React 18** - UI framework
- **TailwindCSS** - Styling
- **Recharts** - Data visualization
- **Axios** - HTTP client
- **Lucide React** - Icons
- **Framer Motion** - Animations (optional)

## 🎯 Components

### Header
- Logo and navigation
- Health status indicator
- Dark mode toggle
- Version badge

### PredictionForm
- Team search with auto-complete
- League selection
- Form validation
- Loading states

### PredictionCards
- **1X2 Card**: Pie chart for match outcome
- **Over/Under 2.5 Card**: Horizontal bar chart
- **BTTS Card**: Radial gauges
- **FII Card**: Football Intelligence Index with components
- **Details Panel**: Collapsible JSON view

## 🎨 Color Palette

```css
Primary:    #2563eb (Blue)
Secondary:  #16a34a (Green)
Accent:     #f59e0b (Orange)
Background: #f8fafc (Light Gray)
Dark BG:    #0f172a (Dark Blue)
```

## 📱 Responsive Breakpoints

- **Mobile**: < 768px
- **Tablet**: 768px - 1024px
- **Desktop**: > 1024px

## 🔧 API Integration

The frontend connects to these FastAPI endpoints:

- `GET /health` - Health check
- `GET /teams` - List of teams
- `GET /models` - Model information
- `GET /stats` - Service statistics
- `POST /predict` - Match prediction

## 🌙 Dark Mode

Dark mode preference is saved to localStorage and persists across sessions.

## 📊 Visualizations

### 1X2 Prediction
- Donut chart showing probabilities for Home/Draw/Away
- Color-coded by outcome
- Confidence badge

### Over/Under 2.5
- Horizontal bar chart
- Clear comparison of Over vs Under probabilities

### BTTS (Both Teams To Score)
- Dual radial gauges
- Yes/No probabilities side by side

### FII (Football Intelligence Index)
- Large radial gauge (0-10 scale)
- Component breakdown
- Color-coded by confidence level

## 🛠️ Development

```bash
# Run development server
npm start

# Build for production
npm build

# Run tests
npm test
```

## 📝 Environment Variables

Create a `.env` file:

```env
REACT_APP_API_URL=http://localhost:8000
```

## 🚢 Deployment

### Build for Production

```bash
npm run build
```

The optimized build will be in the `build/` directory.

### Deploy to Netlify/Vercel

1. Connect your repository
2. Set build command: `npm run build`
3. Set publish directory: `build`
4. Add environment variable: `REACT_APP_API_URL`

## 🎯 Future Enhancements

- [ ] Match comparison page
- [ ] Historical predictions tracking
- [ ] League heatmaps
- [ ] Model diagnostics modal
- [ ] Export predictions to PDF
- [ ] Real-time WebSocket updates
- [ ] User authentication
- [ ] Favorite teams

## 📄 License

Part of the Football AI Prediction Service project.

## 🙏 Acknowledgments

- React team for the amazing framework
- Recharts for beautiful charts
- TailwindCSS for utility-first CSS
- Lucide for clean icons
