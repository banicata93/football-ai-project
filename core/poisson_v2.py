#!/usr/bin/env python3
"""
Poisson v2 Model for Football Predictions

Enhanced Poisson model with:
- Time-decay weighting for recent matches
- League-specific home advantage factors
- Improved attack/defense strength calculation
- Better goal distribution modeling
"""

import pandas as pd
import numpy as np
from scipy.stats import poisson
from datetime import datetime, timedelta
import json
import pickle
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PoissonV2Model:
    """
    Enhanced Poisson model for football match predictions
    
    Features:
    - Time-decay weighting (recent matches more important)
    - League-specific home advantage
    - Improved attack/defense strength calculation
    - Better goal distribution modeling
    """
    
    def __init__(self, time_decay_factor: float = 0.8, min_matches: int = 10):
        """
        Initialize Poisson v2 model
        
        Args:
            time_decay_factor: Decay factor for time weighting (0.8 = 20% decay per week)
            min_matches: Minimum matches required for team strength calculation
        """
        self.time_decay_factor = time_decay_factor
        self.min_matches = min_matches
        
        # Model parameters
        self.team_attack_strength = {}
        self.team_defense_strength = {}
        self.league_home_advantage = {}
        self.league_avg_goals = {}
        self.global_avg_goals = 2.7
        
        # Training metadata
        self.trained_leagues = set()
        self.training_date = None
        self.model_version = "v2.0"
        
        logger.info(f"🏗️ Initialized Poisson v2 model (decay={time_decay_factor})")
    
    def _calculate_time_weights(self, match_dates: pd.Series, reference_date: datetime = None) -> np.ndarray:
        """
        Calculate time-decay weights for matches
        
        Args:
            match_dates: Series of match dates
            reference_date: Reference date for weight calculation (default: today)
            
        Returns:
            Array of weights (more recent = higher weight)
        """
        if reference_date is None:
            reference_date = datetime.now()
        
        # Convert to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(match_dates):
            match_dates = pd.to_datetime(match_dates)
        
        # Calculate days difference
        days_diff = (reference_date - match_dates).dt.days
        
        # Apply exponential decay: weight = decay_factor ^ (days_diff / 7)
        weights = self.time_decay_factor ** (days_diff / 7.0)
        
        # Ensure non-negative weights
        weights = np.maximum(weights, 0.01)
        
        return weights.values
    
    def _calculate_league_factors(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate league-specific factors
        
        Args:
            df: DataFrame with match data
            
        Returns:
            Dictionary with league factors
        """
        league_factors = {}
        
        for league in df['league'].unique():
            league_data = df[df['league'] == league].copy()
            
            if len(league_data) < 50:  # Skip leagues with too few matches
                continue
            
            # Calculate average goals per match
            total_goals = league_data['home_score'] + league_data['away_score']
            avg_goals = total_goals.mean()
            
            # Calculate home advantage (home wins / total matches)
            home_wins = (league_data['home_score'] > league_data['away_score']).sum()
            home_advantage = home_wins / len(league_data)
            
            league_factors[league] = {
                'avg_goals': avg_goals,
                'home_advantage': home_advantage,
                'total_matches': len(league_data)
            }
            
            logger.info(f"📊 {league}: {avg_goals:.2f} goals/match, {home_advantage:.1%} home advantage")
        
        return league_factors
    
    def _calculate_team_strengths(self, df: pd.DataFrame, weights: np.ndarray) -> Tuple[Dict, Dict]:
        """
        Calculate attack and defense strengths for all teams using weighted data
        
        Args:
            df: Match data DataFrame
            weights: Time-decay weights for each match
            
        Returns:
            Tuple of (attack_strengths, defense_strengths) dictionaries
        """
        attack_strengths = {}
        defense_strengths = {}
        
        # Get all unique teams
        all_teams = set(df['home_team'].unique()) | set(df['away_team'].unique())
        
        for team in all_teams:
            # Home matches
            home_matches = df[df['home_team'] == team].copy()
            home_weights = weights[df['home_team'] == team]
            
            # Away matches  
            away_matches = df[df['away_team'] == team].copy()
            away_weights = weights[df['away_team'] == team]
            
            if len(home_matches) + len(away_matches) < self.min_matches:
                # Use league average for teams with insufficient data
                attack_strengths[team] = 1.0
                defense_strengths[team] = 1.0
                continue
            
            # Calculate weighted attack strength
            home_goals_weighted = (home_matches['home_score'] * home_weights).sum()
            away_goals_weighted = (away_matches['away_score'] * away_weights).sum()
            total_weight = home_weights.sum() + away_weights.sum()
            
            if total_weight > 0:
                avg_goals_scored = (home_goals_weighted + away_goals_weighted) / total_weight
                attack_strength = avg_goals_scored / self.global_avg_goals
            else:
                attack_strength = 1.0
            
            # Calculate weighted defense strength
            home_conceded_weighted = (home_matches['away_score'] * home_weights).sum()
            away_conceded_weighted = (away_matches['home_score'] * away_weights).sum()
            
            if total_weight > 0:
                avg_goals_conceded = (home_conceded_weighted + away_conceded_weighted) / total_weight
                defense_strength = avg_goals_conceded / self.global_avg_goals
            else:
                defense_strength = 1.0
            
            attack_strengths[team] = max(0.3, min(3.0, attack_strength))  # Bound values
            defense_strengths[team] = max(0.3, min(3.0, defense_strength))
        
        return attack_strengths, defense_strengths
    
    def fit(self, df: pd.DataFrame, league_column: str = 'league') -> 'PoissonV2Model':
        """
        Train the Poisson v2 model on match data
        
        Args:
            df: DataFrame with columns: home_team, away_team, home_goals, away_goals, date, league
            league_column: Name of the league column
            
        Returns:
            Self for method chaining
        """
        logger.info(f"🏋️ Training Poisson v2 model on {len(df)} matches...")
        
        # Prepare data
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        
        # Calculate time weights
        weights = self._calculate_time_weights(df['date'])
        
        # Calculate league factors
        league_factors = self._calculate_league_factors(df)
        
        # Store league information
        for league, factors in league_factors.items():
            self.league_home_advantage[league] = factors['home_advantage']
            self.league_avg_goals[league] = factors['avg_goals']
            self.trained_leagues.add(league)
        
        # Calculate team strengths
        self.team_attack_strength, self.team_defense_strength = self._calculate_team_strengths(df, weights)
        
        # Update global average
        total_goals = df['home_score'] + df['away_score']
        self.global_avg_goals = total_goals.mean()
        
        self.training_date = datetime.now()
        
        logger.info(f"✅ Trained on {len(self.team_attack_strength)} teams across {len(self.trained_leagues)} leagues")
        logger.info(f"📈 Global average goals: {self.global_avg_goals:.2f}")
        
        return self
    
    def predict_match(self, home_team: str, away_team: str, league: str = None) -> Dict[str, float]:
        """
        Predict match outcome using Poisson v2 model
        
        Args:
            home_team: Home team name
            away_team: Away team name
            league: League name (for home advantage factor)
            
        Returns:
            Dictionary with predictions: {
                'lambda_home': expected home goals,
                'lambda_away': expected away goals,
                'poisson_p_home': probability of home win,
                'poisson_p_draw': probability of draw,
                'poisson_p_away': probability of away win
            }
        """
        # Get team strengths (default to 1.0 if unknown)
        home_attack = self.team_attack_strength.get(home_team, 1.0)
        home_defense = self.team_defense_strength.get(home_team, 1.0)
        away_attack = self.team_attack_strength.get(away_team, 1.0)
        away_defense = self.team_defense_strength.get(away_team, 1.0)
        
        # Get league factors
        if league and league in self.league_home_advantage:
            home_advantage = self.league_home_advantage[league]
            league_avg_goals = self.league_avg_goals[league]
        else:
            home_advantage = 0.55  # Default home advantage
            league_avg_goals = self.global_avg_goals
        
        # Calculate expected goals (lambda parameters)
        # Home team expected goals = attack_strength * opponent_defense_weakness * league_avg * home_advantage
        home_advantage_factor = 1.0 + (home_advantage - 0.5) * 0.6  # Convert to multiplicative factor
        
        lambda_home = home_attack * (1.0 / away_defense) * league_avg_goals * home_advantage_factor * 0.5
        lambda_away = away_attack * (1.0 / home_defense) * league_avg_goals * 0.5
        
        # Ensure reasonable bounds
        lambda_home = max(0.1, min(5.0, lambda_home))
        lambda_away = max(0.1, min(5.0, lambda_away))
        
        # Calculate match outcome probabilities
        prob_home, prob_draw, prob_away = self._calculate_match_probabilities(lambda_home, lambda_away)
        
        return {
            'lambda_home': lambda_home,
            'lambda_away': lambda_away,
            'poisson_p_home': prob_home,
            'poisson_p_draw': prob_draw,
            'poisson_p_away': prob_away,
            'expected_total_goals': lambda_home + lambda_away,
            'home_advantage_used': home_advantage,
            'league_avg_goals': league_avg_goals
        }
    
    def _calculate_match_probabilities(self, lambda_home: float, lambda_away: float, max_goals: int = 8) -> Tuple[float, float, float]:
        """
        Calculate match outcome probabilities using Poisson distribution
        
        Args:
            lambda_home: Expected home team goals
            lambda_away: Expected away team goals
            max_goals: Maximum goals to consider in calculation
            
        Returns:
            Tuple of (prob_home_win, prob_draw, prob_away_win)
        """
        prob_home = 0.0
        prob_draw = 0.0
        prob_away = 0.0
        
        for home_goals in range(max_goals + 1):
            for away_goals in range(max_goals + 1):
                # Probability of this exact score
                prob_score = poisson.pmf(home_goals, lambda_home) * poisson.pmf(away_goals, lambda_away)
                
                if home_goals > away_goals:
                    prob_home += prob_score
                elif home_goals == away_goals:
                    prob_draw += prob_score
                else:
                    prob_away += prob_score
        
        # Normalize to ensure probabilities sum to 1
        total = prob_home + prob_draw + prob_away
        if total > 0:
            prob_home /= total
            prob_draw /= total
            prob_away /= total
        
        return prob_home, prob_draw, prob_away
    
    def get_team_info(self, team: str) -> Dict[str, float]:
        """
        Get team strength information
        
        Args:
            team: Team name
            
        Returns:
            Dictionary with team strengths
        """
        return {
            'attack_strength': self.team_attack_strength.get(team, 1.0),
            'defense_strength': self.team_defense_strength.get(team, 1.0),
            'is_known_team': team in self.team_attack_strength
        }
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model to file
        
        Args:
            filepath: Path to save the model
        """
        model_data = {
            'team_attack_strength': self.team_attack_strength,
            'team_defense_strength': self.team_defense_strength,
            'league_home_advantage': self.league_home_advantage,
            'league_avg_goals': self.league_avg_goals,
            'global_avg_goals': self.global_avg_goals,
            'trained_leagues': list(self.trained_leagues),
            'training_date': self.training_date.isoformat() if self.training_date else None,
            'model_version': self.model_version,
            'time_decay_factor': self.time_decay_factor,
            'min_matches': self.min_matches
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"💾 Saved Poisson v2 model to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str) -> 'PoissonV2Model':
        """
        Load a trained model from file
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded PoissonV2Model instance
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Create new instance
        model = cls(
            time_decay_factor=model_data.get('time_decay_factor', 0.8),
            min_matches=model_data.get('min_matches', 10)
        )
        
        # Restore model state
        model.team_attack_strength = model_data['team_attack_strength']
        model.team_defense_strength = model_data['team_defense_strength']
        model.league_home_advantage = model_data['league_home_advantage']
        model.league_avg_goals = model_data['league_avg_goals']
        model.global_avg_goals = model_data['global_avg_goals']
        model.trained_leagues = set(model_data['trained_leagues'])
        model.model_version = model_data.get('model_version', 'v2.0')
        
        if model_data['training_date']:
            model.training_date = datetime.fromisoformat(model_data['training_date'])
        
        logger.info(f"📂 Loaded Poisson v2 model from {filepath}")
        logger.info(f"🎯 Model covers {len(model.team_attack_strength)} teams, {len(model.trained_leagues)} leagues")
        
        return model
    
    def get_model_info(self) -> Dict:
        """
        Get model information and statistics
        
        Returns:
            Dictionary with model metadata
        """
        return {
            'model_type': 'Poisson v2',
            'version': self.model_version,
            'training_date': self.training_date.isoformat() if self.training_date else None,
            'total_teams': len(self.team_attack_strength),
            'total_leagues': len(self.trained_leagues),
            'leagues': list(self.trained_leagues),
            'global_avg_goals': self.global_avg_goals,
            'time_decay_factor': self.time_decay_factor,
            'min_matches_threshold': self.min_matches,
            'league_home_advantages': self.league_home_advantage
        }


def main():
    """
    Example usage and testing
    """
    # Create sample data for testing
    sample_data = pd.DataFrame({
        'home_team': ['Team A', 'Team B', 'Team A', 'Team C'] * 25,
        'away_team': ['Team B', 'Team C', 'Team D', 'Team A'] * 25,
        'home_score': np.random.poisson(1.5, 100),
        'away_score': np.random.poisson(1.2, 100),
        'date': pd.date_range('2024-01-01', periods=100, freq='D'),
        'league': ['Premier League'] * 50 + ['La Liga'] * 50
    })
    
    # Train model
    model = PoissonV2Model(time_decay_factor=0.8)
    model.fit(sample_data)
    
    # Make prediction
    prediction = model.predict_match('Team A', 'Team B', 'Premier League')
    
    print("🎯 Poisson v2 Prediction:")
    print(f"Expected goals: {prediction['lambda_home']:.2f} - {prediction['lambda_away']:.2f}")
    print(f"Probabilities: 1={prediction['poisson_p_home']:.3f}, X={prediction['poisson_p_draw']:.3f}, 2={prediction['poisson_p_away']:.3f}")
    
    # Save and load test
    model.save_model('test_poisson_v2.pkl')
    loaded_model = PoissonV2Model.load_model('test_poisson_v2.pkl')
    
    print("✅ Poisson v2 model test completed successfully!")


if __name__ == "__main__":
    main()
