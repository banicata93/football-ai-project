// Format probability to percentage
export const formatPercentage = (value) => {
  if (value === null || value === undefined) return 'N/A';
  return `${(value * 100).toFixed(1)}%`;
};

// Format decimal to 3 places
export const formatDecimal = (value, decimals = 3) => {
  if (value === null || value === undefined) return 'N/A';
  return value.toFixed(decimals);
};

// Get confidence color
export const getConfidenceColor = (level) => {
  const colors = {
    High: 'text-green-600 bg-green-100 dark:bg-green-900 dark:text-green-300',
    Medium: 'text-yellow-600 bg-yellow-100 dark:bg-yellow-900 dark:text-yellow-300',
    Low: 'text-red-600 bg-red-100 dark:bg-red-900 dark:text-red-300',
  };
  return colors[level] || colors.Medium;
};

// Get FII color based on score
export const getFIIColor = (score) => {
  if (score >= 7) return '#16a34a'; // green
  if (score >= 4) return '#f59e0b'; // yellow
  return '#dc2626'; // red
};

// Get outcome label
export const getOutcomeLabel = (outcome) => {
  const labels = {
    '1': 'Home Win',
    'X': 'Draw',
    '2': 'Away Win',
    'Over': 'Over 2.5',
    'Under': 'Under 2.5',
    'Yes': 'BTTS Yes',
    'No': 'BTTS No',
  };
  return labels[outcome] || outcome;
};

// Save to local storage
export const savePredictionHistory = (prediction) => {
  try {
    const history = JSON.parse(localStorage.getItem('predictionHistory') || '[]');
    history.unshift({
      ...prediction,
      timestamp: new Date().toISOString(),
    });
    // Keep only last 5
    const trimmed = history.slice(0, 5);
    localStorage.setItem('predictionHistory', JSON.stringify(trimmed));
  } catch (error) {
    console.error('Error saving to localStorage:', error);
  }
};

// Get prediction history
export const getPredictionHistory = () => {
  try {
    return JSON.parse(localStorage.getItem('predictionHistory') || '[]');
  } catch (error) {
    console.error('Error reading from localStorage:', error);
    return [];
  }
};

// Clear prediction history
export const clearPredictionHistory = () => {
  localStorage.removeItem('predictionHistory');
};

// Get theme from localStorage
export const getTheme = () => {
  return localStorage.getItem('theme') || 'light';
};

// Set theme to localStorage
export const setTheme = (theme) => {
  localStorage.setItem('theme', theme);
  if (theme === 'dark') {
    document.documentElement.classList.add('dark');
  } else {
    document.documentElement.classList.remove('dark');
  }
};
