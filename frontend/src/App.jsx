import React, { useState } from 'react';
import Header from './components/Header';
import PredictionForm from './components/PredictionForm';
import PredictionCards from './components/PredictionCards';
import { footballAPI } from './services/api';
import { savePredictionHistory } from './utils/helpers';
import { AlertCircle } from 'lucide-react';

function App() {
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handlePredict = async (homeTeam, awayTeam, league) => {
    setLoading(true);
    setError(null);
    setPrediction(null);

    try {
      const data = await footballAPI.predictMatch(homeTeam, awayTeam, league);
      setPrediction(data);
      savePredictionHistory(data);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to get prediction. Please try again.');
      console.error('Prediction error:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-background dark:bg-dark-bg transition-colors duration-300">
      <Header />

      <main className="container mx-auto px-4 py-8">
        {/* Hero Section */}
        <div className="text-center mb-12 animate-fade-in">
          <h1 className="text-4xl md:text-5xl font-bold text-gray-800 dark:text-white mb-4">
            AI-Powered Football Match Predictions
          </h1>
          <p className="text-lg text-gray-600 dark:text-gray-300 max-w-2xl mx-auto">
            Get accurate match predictions powered by advanced machine learning models.
            Analyze 1X2, Over/Under 2.5, BTTS, and Football Intelligence Index.
          </p>
        </div>

        {/* Main Content Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Prediction Form - Left Column */}
          <div className="lg:col-span-1">
            <PredictionForm onPredict={handlePredict} loading={loading} />

            {/* Stats Card */}
            <div className="card mt-6">
              <h3 className="text-lg font-bold text-gray-800 dark:text-white mb-4">
                📊 Quick Stats
              </h3>
              <div className="space-y-3">
                <div className="stat-card">
                  <p className="text-sm text-gray-600 dark:text-gray-300">Models</p>
                  <p className="text-2xl font-bold text-primary">6</p>
                </div>
                <div className="stat-card">
                  <p className="text-sm text-gray-600 dark:text-gray-300">Accuracy</p>
                  <p className="text-2xl font-bold text-green-600">65-78%</p>
                </div>
                <div className="stat-card">
                  <p className="text-sm text-gray-600 dark:text-gray-300">Teams</p>
                  <p className="text-2xl font-bold text-blue-600">2,942</p>
                </div>
              </div>
            </div>
          </div>

          {/* Results - Right Column */}
          <div className="lg:col-span-2">
            {/* Loading State */}
            {loading && (
              <div className="card flex flex-col items-center justify-center py-20">
                <div className="loading-spinner mb-4"></div>
                <p className="text-lg text-gray-600 dark:text-gray-300">
                  Analyzing match data...
                </p>
                <p className="text-sm text-gray-500 dark:text-gray-400 mt-2">
                  This may take a few seconds
                </p>
              </div>
            )}

            {/* Error State */}
            {error && (
              <div className="card bg-red-50 dark:bg-red-900/20 border-2 border-red-200 dark:border-red-800">
                <div className="flex items-start space-x-3">
                  <AlertCircle className="text-red-600 flex-shrink-0 mt-1" size={24} />
                  <div>
                    <h3 className="text-lg font-bold text-red-800 dark:text-red-300 mb-2">
                      Prediction Error
                    </h3>
                    <p className="text-red-700 dark:text-red-400">{error}</p>
                  </div>
                </div>
              </div>
            )}

            {/* Prediction Results */}
            {prediction && !loading && (
              <PredictionCards prediction={prediction} />
            )}

            {/* Empty State */}
            {!prediction && !loading && !error && (
              <div className="card flex flex-col items-center justify-center py-20 text-center">
                <div className="text-6xl mb-4">⚽</div>
                <h3 className="text-2xl font-bold text-gray-800 dark:text-white mb-2">
                  Ready to Predict
                </h3>
                <p className="text-gray-600 dark:text-gray-300 max-w-md">
                  Select two teams from the form on the left to get AI-powered predictions
                  for their match outcome.
                </p>
              </div>
            )}
          </div>
        </div>

        {/* Footer */}
        <footer className="mt-16 text-center text-gray-500 dark:text-gray-400 text-sm">
          <p>
            Built with ❤️ using React, TailwindCSS, and FastAPI
          </p>
          <p className="mt-2">
            Powered by XGBoost, LightGBM, and Ensemble Learning
          </p>
        </footer>
      </main>
    </div>
  );
}

export default App;
