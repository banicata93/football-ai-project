import React, { useState } from 'react';
import { PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, RadialBarChart, RadialBar } from 'recharts';
import { TrendingUp, Award, Target, ChevronDown, ChevronUp } from 'lucide-react';
import { formatPercentage, getOutcomeLabel, getConfidenceColor, getFIIColor } from '../utils/helpers';

const PredictionCards = ({ prediction }) => {
  const [showDetails, setShowDetails] = useState(false);

  if (!prediction) return null;

  const { prediction_1x2, prediction_ou25, prediction_btts, fii, match_info } = prediction;

  // 1X2 Data for Pie Chart
  const data1x2 = [
    { name: 'Home Win', value: prediction_1x2.prob_home_win, color: '#16a34a' },
    { name: 'Draw', value: prediction_1x2.prob_draw, color: '#f59e0b' },
    { name: 'Away Win', value: prediction_1x2.prob_away_win, color: '#dc2626' },
  ];

  // OU2.5 Data for Bar Chart
  const dataOU25 = [
    { name: 'Over 2.5', value: prediction_ou25.prob_over * 100, color: '#2563eb' },
    { name: 'Under 2.5', value: prediction_ou25.prob_under * 100, color: '#64748b' },
  ];

  // BTTS Data for Radial Chart
  const dataBTTS = [
    { name: 'Yes', value: prediction_btts.prob_yes * 100, fill: '#16a34a' },
    { name: 'No', value: prediction_btts.prob_no * 100, fill: '#dc2626' },
  ];

  // FII Gauge Data
  const fiiData = [
    { name: 'FII', value: fii.score, fill: getFIIColor(fii.score) },
  ];

  return (
    <div className="space-y-6 animate-slide-up">
      {/* Match Header */}
      <div className="card bg-gradient-to-r from-blue-500 to-purple-600 text-white">
        <div className="text-center">
          <div className="text-4xl mb-2">🏟️</div>
          <h2 className="text-3xl font-bold mb-2">
            {match_info.home_team} <span className="text-yellow-300">vs</span> {match_info.away_team}
          </h2>
          {match_info.league && (
            <p className="text-blue-100 text-lg">{match_info.league}</p>
          )}
          <p className="text-blue-200 text-sm mt-2">
            Prediction generated at {new Date(prediction.timestamp).toLocaleString()}
          </p>
        </div>
      </div>

      {/* Prediction Cards Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {/* 1X2 Card */}
        <div className="card">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-xl font-bold text-gray-800 dark:text-white flex items-center">
              <Award className="mr-2 text-primary" size={24} />
              1X2 Prediction
            </h3>
            <span className={`badge ${getConfidenceColor('High')}`}>
              {formatPercentage(prediction_1x2.confidence)}
            </span>
          </div>

          <ResponsiveContainer width="100%" height={200}>
            <PieChart>
              <Pie
                data={data1x2}
                cx="50%"
                cy="50%"
                innerRadius={60}
                outerRadius={80}
                paddingAngle={5}
                dataKey="value"
              >
                {data1x2.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip formatter={(value) => formatPercentage(value)} />
            </PieChart>
          </ResponsiveContainer>

          <div className="mt-4 space-y-2">
            {data1x2.map((item, index) => (
              <div key={index} className="flex items-center justify-between">
                <div className="flex items-center">
                  <div
                    className="w-3 h-3 rounded-full mr-2"
                    style={{ backgroundColor: item.color }}
                  />
                  <span className="text-sm text-gray-600 dark:text-gray-300">
                    {item.name}
                  </span>
                </div>
                <span className="font-semibold text-gray-800 dark:text-white">
                  {formatPercentage(item.value)}
                </span>
              </div>
            ))}
          </div>

          <div className="mt-4 p-3 bg-green-50 dark:bg-green-900/20 rounded-lg border border-green-200 dark:border-green-800">
            <p className="text-sm text-gray-600 dark:text-gray-300">Predicted Outcome:</p>
            <p className="text-lg font-bold text-green-700 dark:text-green-400">
              {getOutcomeLabel(prediction_1x2.predicted_outcome)}
            </p>
          </div>
        </div>

        {/* Over/Under 2.5 Card */}
        <div className="card">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-xl font-bold text-gray-800 dark:text-white flex items-center">
              <Target className="mr-2 text-blue-600" size={24} />
              Over/Under 2.5
            </h3>
            <span className={`badge ${getConfidenceColor('High')}`}>
              {formatPercentage(prediction_ou25.confidence)}
            </span>
          </div>

          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={dataOU25} layout="vertical">
              <XAxis type="number" domain={[0, 100]} />
              <YAxis type="category" dataKey="name" width={100} />
              <Tooltip formatter={(value) => `${value.toFixed(1)}%`} />
              <Bar dataKey="value" radius={[0, 10, 10, 0]}>
                {dataOU25.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>

          <div className="mt-4 space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-600 dark:text-gray-300">Over 2.5</span>
              <span className="font-semibold text-gray-800 dark:text-white">
                {formatPercentage(prediction_ou25.prob_over)}
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-600 dark:text-gray-300">Under 2.5</span>
              <span className="font-semibold text-gray-800 dark:text-white">
                {formatPercentage(prediction_ou25.prob_under)}
              </span>
            </div>
          </div>

          <div className="mt-4 p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
            <p className="text-sm text-gray-600 dark:text-gray-300">Predicted Outcome:</p>
            <p className="text-lg font-bold text-blue-700 dark:text-blue-400">
              {getOutcomeLabel(prediction_ou25.predicted_outcome)}
            </p>
          </div>
        </div>

        {/* BTTS Card */}
        <div className="card">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-xl font-bold text-gray-800 dark:text-white flex items-center">
              <TrendingUp className="mr-2 text-green-600" size={24} />
              Both Teams To Score
            </h3>
            <span className={`badge ${getConfidenceColor('High')}`}>
              {formatPercentage(prediction_btts.confidence)}
            </span>
          </div>

          <div className="grid grid-cols-2 gap-4">
            {dataBTTS.map((item, index) => (
              <div key={index} className="text-center">
                <ResponsiveContainer width="100%" height={120}>
                  <RadialBarChart
                    cx="50%"
                    cy="50%"
                    innerRadius="60%"
                    outerRadius="90%"
                    data={[item]}
                    startAngle={90}
                    endAngle={-270}
                  >
                    <RadialBar
                      background
                      dataKey="value"
                      cornerRadius={10}
                      fill={item.fill}
                    />
                  </RadialBarChart>
                </ResponsiveContainer>
                <p className="text-sm text-gray-600 dark:text-gray-300 mt-2">{item.name}</p>
                <p className="text-xl font-bold text-gray-800 dark:text-white">
                  {item.value.toFixed(1)}%
                </p>
              </div>
            ))}
          </div>

          <div className="mt-4 p-3 bg-purple-50 dark:bg-purple-900/20 rounded-lg border border-purple-200 dark:border-purple-800">
            <p className="text-sm text-gray-600 dark:text-gray-300">Predicted Outcome:</p>
            <p className="text-lg font-bold text-purple-700 dark:text-purple-400">
              {getOutcomeLabel(prediction_btts.predicted_outcome)}
            </p>
          </div>
        </div>
      </div>

      {/* FII Card - Full Width */}
      <div className="card bg-gradient-to-br from-amber-50 to-orange-50 dark:from-amber-900/20 dark:to-orange-900/20">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-2xl font-bold text-gray-800 dark:text-white flex items-center">
            <span className="text-3xl mr-2">🧠</span>
            Football Intelligence Index (FII)
          </h3>
          <span className={`badge ${getConfidenceColor(fii.confidence_level)}`}>
            {fii.confidence_level} Confidence
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* FII Gauge */}
          <div className="text-center">
            <ResponsiveContainer width="100%" height={250}>
              <RadialBarChart
                cx="50%"
                cy="50%"
                innerRadius="70%"
                outerRadius="100%"
                data={fiiData}
                startAngle={180}
                endAngle={0}
              >
                <RadialBar
                  background
                  dataKey="value"
                  cornerRadius={10}
                  fill={getFIIColor(fii.score)}
                />
              </RadialBarChart>
            </ResponsiveContainer>
            <div className="mt-[-80px] mb-[60px]">
              <p className="text-5xl font-bold" style={{ color: getFIIColor(fii.score) }}>
                {fii.score.toFixed(2)}
              </p>
              <p className="text-gray-600 dark:text-gray-300">out of 10</p>
            </div>
          </div>

          {/* FII Components */}
          <div className="space-y-3">
            <h4 className="font-semibold text-gray-700 dark:text-gray-300 mb-3">
              Index Components:
            </h4>
            {Object.entries(fii.components).map(([key, value]) => (
              <div key={key} className="flex items-center justify-between p-3 bg-white dark:bg-dark-card rounded-lg">
                <span className="text-sm text-gray-600 dark:text-gray-300 capitalize">
                  {key.replace(/_/g, ' ')}
                </span>
                <span className="font-semibold text-gray-800 dark:text-white">
                  {typeof value === 'number' ? value.toFixed(3) : value}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* View Details Button */}
      <div className="card">
        <button
          onClick={() => setShowDetails(!showDetails)}
          className="w-full flex items-center justify-between text-left"
        >
          <span className="text-lg font-semibold text-gray-800 dark:text-white">
            🔍 View Full Prediction Details
          </span>
          {showDetails ? <ChevronUp size={24} /> : <ChevronDown size={24} />}
        </button>

        {showDetails && (
          <div className="mt-4 p-4 bg-gray-50 dark:bg-gray-800 rounded-lg overflow-x-auto">
            <pre className="text-xs text-gray-700 dark:text-gray-300">
              {JSON.stringify(prediction, null, 2)}
            </pre>
          </div>
        )}
      </div>
    </div>
  );
};

export default PredictionCards;
