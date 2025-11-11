import React, { useState, useEffect } from 'react';
import { Search, Loader } from 'lucide-react';
import { footballAPI } from '../services/api';

const PredictionForm = ({ onPredict, loading }) => {
  const [teams, setTeams] = useState([]);
  const [homeTeam, setHomeTeam] = useState('');
  const [awayTeam, setAwayTeam] = useState('');
  const [league, setLeague] = useState('');
  const [homeSearch, setHomeSearch] = useState('');
  const [awaySearch, setAwaySearch] = useState('');
  const [showHomeDropdown, setShowHomeDropdown] = useState(false);
  const [showAwayDropdown, setShowAwayDropdown] = useState(false);

  useEffect(() => {
    loadTeams();
    
    // Close dropdowns on click outside
    const handleClickOutside = (event) => {
      if (!event.target.closest('.team-dropdown-container')) {
        setShowHomeDropdown(false);
        setShowAwayDropdown(false);
      }
    };
    
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const loadTeams = async () => {
    try {
      const data = await footballAPI.getTeams();
      // Handle both formats: array of strings or array of objects
      const teamsList = data.teams || [];
      
      // Convert to array of objects with name and display_name
      const teamsData = teamsList.map(team => {
        if (typeof team === 'string') {
          return { name: team, display: team };
        } else {
          return {
            name: team.name,
            display: team.display_name || team.name,
            tier: team.tier,
            elo: team.elo
          };
        }
      });
      
      setTeams(teamsData);
    } catch (error) {
      console.error('Error loading teams:', error);
      setTeams([]);
    }
  };

  const filteredHomeTeams = teams.filter(team =>
    team.display.toLowerCase().includes(homeSearch.toLowerCase()) ||
    team.name.toLowerCase().includes(homeSearch.toLowerCase())
  ).slice(0, 10);

  const filteredAwayTeams = teams.filter(team =>
    (team.display.toLowerCase().includes(awaySearch.toLowerCase()) ||
     team.name.toLowerCase().includes(awaySearch.toLowerCase())) &&
    team.name !== homeTeam
  ).slice(0, 10);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (homeTeam && awayTeam && homeTeam !== awayTeam) {
      onPredict(homeTeam, awayTeam, league || null);
    }
  };

  return (
    <div className="card animate-fade-in">
      <h2 className="text-2xl font-bold mb-6 text-gray-800 dark:text-white flex items-center">
        <span className="text-3xl mr-3">🔮</span>
        Match Prediction
      </h2>

      <form onSubmit={handleSubmit} className="space-y-6">
        {/* Home Team */}
        <div className="relative team-dropdown-container">
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            Home Team
          </label>
          <div className="relative">
            <input
              type="text"
              value={homeSearch}
              onChange={(e) => {
                setHomeSearch(e.target.value);
                setHomeTeam('');
                setShowHomeDropdown(true);
              }}
              onFocus={() => setShowHomeDropdown(true)}
              placeholder="Search for home team..."
              className="input-field"
              required
            />
            <Search className="absolute right-3 top-3 text-gray-400" size={20} />
          </div>

          {/* Home Team Dropdown */}
          {showHomeDropdown && (
            <div className="absolute z-10 w-full mt-1 bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-lg shadow-lg max-h-60 overflow-y-auto">
              {filteredHomeTeams.length > 0 ? (
                filteredHomeTeams.map((team, index) => (
                  <button
                    key={index}
                    type="button"
                    onClick={() => {
                      setHomeTeam(team.name);  // Store API name
                      setHomeSearch(team.display);  // Show display name
                      setShowHomeDropdown(false);
                    }}
                    className="w-full text-left px-4 py-2 hover:bg-gray-100 dark:hover:bg-gray-600 transition-colors"
                  >
                    <div className="flex items-center justify-between">
                      <span className="font-medium">{team.display}</span>
                      {team.tier && (
                        <span className={`text-xs px-2 py-1 rounded ${
                          team.tier === 'Elite' ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-300' :
                          team.tier === 'Strong' ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-300' :
                          team.tier === 'Average' ? 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-300' :
                          'bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-300'
                        }`}>
                          {team.tier}
                        </span>
                      )}
                    </div>
                  </button>
                ))
              ) : (
                <div className="px-4 py-2 text-gray-500 dark:text-gray-400">
                  {homeSearch ? 'No teams found' : 'Start typing to search...'}
                </div>
              )}
            </div>
          )}
        </div>

        {/* Away Team */}
        <div className="relative team-dropdown-container">
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            Away Team
          </label>
          <div className="relative">
            <input
              type="text"
              value={awaySearch}
              onChange={(e) => {
                setAwaySearch(e.target.value);
                setAwayTeam('');
                setShowAwayDropdown(true);
              }}
              onFocus={() => setShowAwayDropdown(true)}
              placeholder="Search for away team..."
              className="input-field"
              required
            />
            <Search className="absolute right-3 top-3 text-gray-400" size={20} />
          </div>

          {/* Away Team Dropdown */}
          {showAwayDropdown && (
            <div className="absolute z-10 w-full mt-1 bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-lg shadow-lg max-h-60 overflow-y-auto">
              {filteredAwayTeams.length > 0 ? (
                filteredAwayTeams.map((team, index) => (
                  <button
                    key={index}
                    type="button"
                    onClick={() => {
                      setAwayTeam(team.name);  // Store API name
                      setAwaySearch(team.display);  // Show display name
                      setShowAwayDropdown(false);
                    }}
                    className="w-full text-left px-4 py-2 hover:bg-gray-100 dark:hover:bg-gray-600 transition-colors"
                  >
                    <div className="flex items-center justify-between">
                      <span className="font-medium">{team.display}</span>
                      {team.tier && (
                        <span className={`text-xs px-2 py-1 rounded ${
                          team.tier === 'Elite' ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-300' :
                          team.tier === 'Strong' ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-300' :
                          team.tier === 'Average' ? 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-300' :
                          'bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-300'
                        }`}>
                          {team.tier}
                        </span>
                      )}
                    </div>
                  </button>
                ))
              ) : (
                <div className="px-4 py-2 text-gray-500 dark:text-gray-400">
                  {awaySearch ? 'No teams found' : 'Start typing to search...'}
                </div>
              )}
            </div>
          )}
        </div>

        {/* League (Optional) */}
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            League (Optional)
          </label>
          <select
            value={league}
            onChange={(e) => setLeague(e.target.value)}
            className="input-field"
          >
            <option value="">Select league...</option>
            <option value="Premier League">Premier League</option>
            <option value="La Liga">La Liga</option>
            <option value="Bundesliga">Bundesliga</option>
            <option value="Serie A">Serie A</option>
            <option value="Ligue 1">Ligue 1</option>
          </select>
        </div>

        {/* Submit Button */}
        <button
          type="submit"
          disabled={loading || !homeTeam || !awayTeam || homeTeam === awayTeam}
          className="btn-primary w-full flex items-center justify-center space-x-2 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {loading ? (
            <>
              <Loader className="animate-spin" size={20} />
              <span>Predicting...</span>
            </>
          ) : (
            <>
              <span>🔮</span>
              <span>Predict Match</span>
            </>
          )}
        </button>
      </form>

      {/* Selected Match Preview */}
      {homeTeam && awayTeam && (
        <div className="mt-6 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
          <div className="text-center">
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
              Selected Match
            </p>
            <p className="text-lg font-bold text-gray-800 dark:text-white">
              {homeSearch} <span className="text-primary">vs</span> {awaySearch}
            </p>
            {league && (
              <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
                {league}
              </p>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default PredictionForm;
