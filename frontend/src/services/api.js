import axios from 'axios';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// API Service
export const footballAPI = {
  // Health check
  getHealth: async () => {
    const response = await api.get('/health');
    return response.data;
  },

  // Get all teams
  getTeams: async () => {
    const response = await api.get('/teams');
    return response.data;
  },

  // Get models info
  getModels: async () => {
    const response = await api.get('/models');
    return response.data;
  },

  // Get service stats
  getStats: async () => {
    const response = await api.get('/stats');
    return response.data;
  },

  // Make prediction (POST)
  predictMatch: async (homeTeam, awayTeam, league = null) => {
    const response = await api.post('/predict', {
      home_team: homeTeam,
      away_team: awayTeam,
      league: league,
    });
    return response.data;
  },

  // Make prediction (GET)
  predictMatchGet: async (homeTeam, awayTeam, league = null) => {
    const url = `/predict/${encodeURIComponent(homeTeam)}/vs/${encodeURIComponent(awayTeam)}`;
    const params = league ? { league } : {};
    const response = await api.get(url, { params });
    return response.data;
  },
};

export default api;
