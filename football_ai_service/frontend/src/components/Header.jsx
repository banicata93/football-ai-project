import React, { useState, useEffect } from 'react';
import { Moon, Sun, Activity } from 'lucide-react';
import { getTheme, setTheme } from '../utils/helpers';
import { footballAPI } from '../services/api';

const Header = () => {
  const [darkMode, setDarkMode] = useState(getTheme() === 'dark');
  const [isHealthy, setIsHealthy] = useState(null);

  useEffect(() => {
    setTheme(darkMode ? 'dark' : 'light');
  }, [darkMode]);

  useEffect(() => {
    // Check health on mount and every 60 seconds
    const checkHealth = async () => {
      try {
        await footballAPI.getHealth();
        setIsHealthy(true);
      } catch (error) {
        setIsHealthy(false);
      }
    };

    checkHealth();
    const interval = setInterval(checkHealth, 60000);

    return () => clearInterval(interval);
  }, []);

  const toggleDarkMode = () => {
    setDarkMode(!darkMode);
  };

  return (
    <header className="bg-white dark:bg-dark-card shadow-md sticky top-0 z-50 transition-colors duration-300">
      <div className="container mx-auto px-4 py-4">
        <div className="flex items-center justify-between">
          {/* Logo */}
          <div className="flex items-center space-x-3">
            <div className="text-4xl">⚽</div>
            <div>
              <h1 className="text-2xl font-bold text-primary dark:text-blue-400">
                Football AI Service
              </h1>
              <p className="text-sm text-gray-500 dark:text-gray-400">
                Powered by Machine Learning
              </p>
            </div>
          </div>

          {/* Navigation & Controls */}
          <div className="flex items-center space-x-6">
            {/* Health Status */}
            <div className="flex items-center space-x-2">
              <Activity 
                size={20} 
                className={isHealthy ? 'text-green-500' : 'text-red-500'}
              />
              <div className="flex items-center space-x-1">
                <div 
                  className={`w-2 h-2 rounded-full ${
                    isHealthy ? 'bg-green-500 pulse-dot' : 'bg-red-500'
                  }`}
                />
                <span className="text-sm text-gray-600 dark:text-gray-300">
                  {isHealthy === null ? 'Checking...' : isHealthy ? 'Online' : 'Offline'}
                </span>
              </div>
            </div>

            {/* Version Badge */}
            <div className="hidden md:block">
              <span className="badge bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-300">
                v1.0.0
              </span>
            </div>

            {/* Dark Mode Toggle */}
            <button
              onClick={toggleDarkMode}
              className="p-2 rounded-lg bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors duration-200"
              aria-label="Toggle dark mode"
            >
              {darkMode ? (
                <Sun size={20} className="text-yellow-500" />
              ) : (
                <Moon size={20} className="text-gray-700" />
              )}
            </button>
          </div>
        </div>

        {/* Navigation Tabs */}
        <nav className="mt-4 flex space-x-6 border-t border-gray-200 dark:border-gray-700 pt-4">
          <button className="text-primary dark:text-blue-400 font-semibold border-b-2 border-primary pb-1">
            🏠 Home
          </button>
          <button className="text-gray-600 dark:text-gray-400 hover:text-primary dark:hover:text-blue-400 transition-colors pb-1">
            🔮 Predictions
          </button>
          <button className="text-gray-600 dark:text-gray-400 hover:text-primary dark:hover:text-blue-400 transition-colors pb-1">
            👥 Teams
          </button>
          <button className="text-gray-600 dark:text-gray-400 hover:text-primary dark:hover:text-blue-400 transition-colors pb-1">
            📊 Stats
          </button>
        </nav>
      </div>
    </header>
  );
};

export default Header;
