#!/usr/bin/env python3
"""
1X2-Specific Feature Engineering

Advanced features specifically designed for 1X2 (match result) predictions:
- Match difficulty index
- Expected points (xPts) calculations
- Late goal vulnerability
- Travel fatigue proxy
- Form momentum with weighting
- Possession and shot balance
- League-specific home advantage factors
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Features1X2:
    """
    Advanced feature engineering specifically for 1X2 predictions
    """
    
    def __init__(self, lookback_days: int = 365, min_matches: int = 5):
        """
        Initialize 1X2 feature engineer
        
        Args:
            lookback_days: Days to look back for historical data
            min_matches: Minimum matches required for reliable statistics
        """
        self.lookback_days = lookback_days
        self.min_matches = min_matches
        
        # League-specific factors (will be calculated from data)
        self.league_home_advantages = {}
        self.league_avg_goals = {}
        self.league_factors = {}
        
        logger.info(f"🔧 Initialized 1X2 feature engineer (lookback={lookback_days} days)")
    
    def calculate_league_factors(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """
        Calculate league-specific factors for feature engineering
        
        Args:
            df: Historical match data
            
        Returns:
            Dictionary with league factors
        """
        logger.info("📊 Calculating league-specific factors...")
        
        league_factors = {}
        
        for league in df['league'].unique():
            league_data = df[df['league'] == league].copy()
            
            if len(league_data) < 50:  # Skip leagues with insufficient data
                continue
            
            # Home advantage calculation
            home_wins = (league_data['home_score'] > league_data['away_score']).sum()
            draws = (league_data['home_score'] == league_data['away_score']).sum()
            away_wins = (league_data['home_score'] < league_data['away_score']).sum()
            total_matches = len(league_data)
            
            home_advantage = home_wins / total_matches
            draw_rate = draws / total_matches
            
            # Average goals
            avg_goals = (league_data['home_score'] + league_data['away_score']).mean()
            
            # Goal difference variance (competitiveness indicator)
            goal_diff = league_data['home_score'] - league_data['away_score']
            competitiveness = 1.0 / (1.0 + goal_diff.std())
            
            league_factors[league] = {
                'home_advantage': home_advantage,
                'draw_rate': draw_rate,
                'avg_goals': avg_goals,
                'competitiveness': competitiveness,
                'total_matches': total_matches
            }
            
            logger.info(f"🏆 {league}: HA={home_advantage:.3f}, DR={draw_rate:.3f}, Goals={avg_goals:.2f}")
        
        self.league_factors = league_factors
        return league_factors
    
    def calculate_match_difficulty_index(self, home_team: str, away_team: str, 
                                       df: pd.DataFrame, reference_date: datetime) -> float:
        """
        Calculate match difficulty index based on team strengths
        
        Args:
            home_team: Home team name
            away_team: Away team name
            df: Historical match data
            reference_date: Reference date for calculation
            
        Returns:
            Match difficulty index (0-1, higher = more difficult/competitive)
        """
        # Get recent matches for both teams
        cutoff_date = reference_date - timedelta(days=self.lookback_days)
        recent_data = df[df['date'] >= cutoff_date].copy()
        
        # Calculate team strengths
        home_strength = self._calculate_team_strength(home_team, recent_data)
        away_strength = self._calculate_team_strength(away_team, recent_data)
        
        # Difficulty is higher when teams are more evenly matched
        strength_diff = abs(home_strength - away_strength)
        difficulty_index = 1.0 - (strength_diff / 2.0)  # Normalize to 0-1
        
        return max(0.0, min(1.0, difficulty_index))
    
    def _calculate_team_strength(self, team: str, df: pd.DataFrame) -> float:
        """
        Calculate team strength based on recent performance
        
        Args:
            team: Team name
            df: Recent match data
            
        Returns:
            Team strength (0-1)
        """
        team_matches = df[(df['home_team'] == team) | (df['away_team'] == team)].copy()
        
        if len(team_matches) < self.min_matches:
            return 0.5  # Default strength for unknown teams
        
        # Calculate points per game
        points = 0
        total_matches = 0
        
        for _, match in team_matches.iterrows():
            if match['home_team'] == team:
                # Team played at home
                if match['home_score'] > match['away_score']:
                    points += 3  # Win
                elif match['home_score'] == match['away_score']:
                    points += 1  # Draw
                # Loss = 0 points
            else:
                # Team played away
                if match['away_score'] > match['home_score']:
                    points += 3  # Win
                elif match['away_score'] == match['home_score']:
                    points += 1  # Draw
                # Loss = 0 points
            
            total_matches += 1
        
        points_per_game = points / total_matches if total_matches > 0 else 1.5
        strength = points_per_game / 3.0  # Normalize to 0-1
        
        return max(0.0, min(1.0, strength))
    
    def calculate_expected_points(self, team: str, df: pd.DataFrame, 
                                reference_date: datetime, is_home: bool) -> float:
        """
        Calculate expected points (xPts) based on recent performance
        
        Args:
            team: Team name
            df: Historical match data
            reference_date: Reference date
            is_home: Whether team is playing at home
            
        Returns:
            Expected points for this match
        """
        cutoff_date = reference_date - timedelta(days=self.lookback_days)
        recent_data = df[df['date'] >= cutoff_date].copy()
        
        # Get team's recent matches
        if is_home:
            team_matches = recent_data[recent_data['home_team'] == team].copy()
        else:
            team_matches = recent_data[recent_data['away_team'] == team].copy()
        
        if len(team_matches) < self.min_matches:
            return 1.5  # Default expected points
        
        # Calculate actual points from recent matches
        total_points = 0
        for _, match in team_matches.iterrows():
            if is_home:
                home_score, away_score = match['home_score'], match['away_score']
            else:
                home_score, away_score = match['away_score'], match['home_score']  # Flip perspective
            
            if home_score > away_score:
                total_points += 3
            elif home_score == away_score:
                total_points += 1
        
        expected_points = total_points / len(team_matches)
        return expected_points
    
    def calculate_late_goal_vulnerability(self, team: str, df: pd.DataFrame, 
                                        reference_date: datetime) -> float:
        """
        Calculate team's vulnerability to late goals (proxy for mental strength)
        
        Args:
            team: Team name
            df: Historical match data with minute-by-minute data (if available)
            reference_date: Reference date
            
        Returns:
            Late goal vulnerability index (0-1, higher = more vulnerable)
        """
        # Note: This is a simplified version since we don't have minute-by-minute data
        # In practice, this would analyze goals scored/conceded in final 15 minutes
        
        cutoff_date = reference_date - timedelta(days=self.lookback_days)
        recent_data = df[df['date'] >= cutoff_date].copy()
        
        team_matches = recent_data[(recent_data['home_team'] == team) | 
                                 (recent_data['away_team'] == team)].copy()
        
        if len(team_matches) < self.min_matches:
            return 0.5  # Default vulnerability
        
        # Proxy: teams that concede more goals are more vulnerable
        goals_conceded = 0
        total_matches = 0
        
        for _, match in team_matches.iterrows():
            if match['home_team'] == team:
                goals_conceded += match['away_score']
            else:
                goals_conceded += match['home_score']
            total_matches += 1
        
        avg_goals_conceded = goals_conceded / total_matches if total_matches > 0 else 1.0
        
        # Normalize to 0-1 (assuming max 4 goals conceded per game)
        vulnerability = min(1.0, avg_goals_conceded / 4.0)
        
        return vulnerability
    
    def calculate_travel_fatigue_proxy(self, team: str, df: pd.DataFrame, 
                                     reference_date: datetime) -> float:
        """
        Calculate travel fatigue proxy based on recent match frequency
        
        Args:
            team: Team name
            df: Historical match data
            reference_date: Reference date
            
        Returns:
            Travel fatigue index (0-1, higher = more fatigued)
        """
        # Look at matches in last 14 days
        cutoff_date = reference_date - timedelta(days=14)
        recent_data = df[df['date'] >= cutoff_date].copy()
        
        team_matches = recent_data[(recent_data['home_team'] == team) | 
                                 (recent_data['away_team'] == team)].copy()
        
        # Count matches in last 14 days
        match_count = len(team_matches)
        
        # More matches = more fatigue
        # Assuming 2+ matches in 14 days indicates high fatigue
        fatigue_index = min(1.0, match_count / 3.0)
        
        return fatigue_index
    
    def calculate_form_momentum_weighted(self, team: str, df: pd.DataFrame, 
                                       reference_date: datetime) -> float:
        """
        Calculate weighted form momentum (recent results weighted more heavily)
        
        Args:
            team: Team name
            df: Historical match data
            reference_date: Reference date
            
        Returns:
            Form momentum (-1 to 1, positive = good form)
        """
        cutoff_date = reference_date - timedelta(days=60)  # Last 2 months
        recent_data = df[df['date'] >= cutoff_date].copy()
        
        team_matches = recent_data[(recent_data['home_team'] == team) | 
                                 (recent_data['away_team'] == team)].copy()
        
        if len(team_matches) < 3:
            return 0.0  # Neutral form for insufficient data
        
        # Sort by date (most recent first)
        team_matches = team_matches.sort_values('date', ascending=False)
        
        weighted_form = 0.0
        total_weight = 0.0
        
        for i, (_, match) in enumerate(team_matches.iterrows()):
            # Weight decreases with age (most recent = weight 1.0)
            weight = 1.0 / (1.0 + i * 0.2)
            
            # Calculate result value
            if match['home_team'] == team:
                home_score, away_score = match['home_score'], match['away_score']
            else:
                home_score, away_score = match['away_score'], match['home_score']
            
            if home_score > away_score:
                result_value = 1.0  # Win
            elif home_score == away_score:
                result_value = 0.0  # Draw
            else:
                result_value = -1.0  # Loss
            
            weighted_form += result_value * weight
            total_weight += weight
        
        if total_weight > 0:
            momentum = weighted_form / total_weight
        else:
            momentum = 0.0
        
        return max(-1.0, min(1.0, momentum))
    
    def calculate_possession_balance(self, home_team: str, away_team: str, 
                                   df: pd.DataFrame, reference_date: datetime) -> float:
        """
        Calculate expected possession balance between teams
        
        Args:
            home_team: Home team name
            away_team: Away team name
            df: Historical match data
            reference_date: Reference date
            
        Returns:
            Possession balance (-1 to 1, positive favors home team)
        """
        # Proxy using goal statistics (teams that score more typically have more possession)
        cutoff_date = reference_date - timedelta(days=self.lookback_days)
        recent_data = df[df['date'] >= cutoff_date].copy()
        
        # Calculate average goals scored for each team
        home_avg_goals = self._calculate_avg_goals_scored(home_team, recent_data)
        away_avg_goals = self._calculate_avg_goals_scored(away_team, recent_data)
        
        # Balance based on goal-scoring ability
        if home_avg_goals + away_avg_goals > 0:
            balance = (home_avg_goals - away_avg_goals) / (home_avg_goals + away_avg_goals)
        else:
            balance = 0.0
        
        return max(-1.0, min(1.0, balance))
    
    def _calculate_avg_goals_scored(self, team: str, df: pd.DataFrame) -> float:
        """Calculate average goals scored by team"""
        team_matches = df[(df['home_team'] == team) | (df['away_team'] == team)].copy()
        
        if len(team_matches) == 0:
            return 1.0  # Default
        
        total_goals = 0
        for _, match in team_matches.iterrows():
            if match['home_team'] == team:
                total_goals += match['home_score']
            else:
                total_goals += match['away_score']
        
        return total_goals / len(team_matches)
    
    def calculate_shot_balance(self, home_team: str, away_team: str, 
                             df: pd.DataFrame, reference_date: datetime) -> float:
        """
        Calculate expected shot balance (proxy using goals and team strength)
        
        Args:
            home_team: Home team name
            away_team: Away team name
            df: Historical match data
            reference_date: Reference date
            
        Returns:
            Shot balance (-1 to 1, positive favors home team)
        """
        # Similar to possession balance but considering defensive strength too
        cutoff_date = reference_date - timedelta(days=self.lookback_days)
        recent_data = df[df['date'] >= cutoff_date].copy()
        
        home_attack = self._calculate_avg_goals_scored(home_team, recent_data)
        away_attack = self._calculate_avg_goals_scored(away_team, recent_data)
        
        home_defense = self._calculate_avg_goals_conceded(home_team, recent_data)
        away_defense = self._calculate_avg_goals_conceded(away_team, recent_data)
        
        # Shot balance considers both attacking and defensive capabilities
        home_net_strength = home_attack - home_defense
        away_net_strength = away_attack - away_defense
        
        if abs(home_net_strength) + abs(away_net_strength) > 0:
            balance = (home_net_strength - away_net_strength) / (abs(home_net_strength) + abs(away_net_strength) + 1e-6)
        else:
            balance = 0.0
        
        return max(-1.0, min(1.0, balance))
    
    def _calculate_avg_goals_conceded(self, team: str, df: pd.DataFrame) -> float:
        """Calculate average goals conceded by team"""
        team_matches = df[(df['home_team'] == team) | (df['away_team'] == team)].copy()
        
        if len(team_matches) == 0:
            return 1.0  # Default
        
        total_goals_conceded = 0
        for _, match in team_matches.iterrows():
            if match['home_team'] == team:
                total_goals_conceded += match['away_score']
            else:
                total_goals_conceded += match['home_score']
        
        return total_goals_conceded / len(team_matches)
    
    def create_1x2_features(self, home_team: str, away_team: str, league: str,
                           df: pd.DataFrame, reference_date: datetime) -> Dict[str, float]:
        """
        Create comprehensive 1X2-specific features for a match
        
        Args:
            home_team: Home team name
            away_team: Away team name
            league: League name
            df: Historical match data
            reference_date: Reference date for feature calculation
            
        Returns:
            Dictionary with all 1X2 features
        """
        logger.info(f"🔧 Creating 1X2 features for {home_team} vs {away_team}")
        
        features = {}
        
        # Calculate league factors if not already done
        if not self.league_factors:
            self.calculate_league_factors(df)
        
        # Basic match features
        features['match_difficulty_index'] = self.calculate_match_difficulty_index(
            home_team, away_team, df, reference_date
        )
        
        # Expected points
        features['expected_points_home'] = self.calculate_expected_points(
            home_team, df, reference_date, is_home=True
        )
        features['expected_points_away'] = self.calculate_expected_points(
            away_team, df, reference_date, is_home=False
        )
        
        # Vulnerability and fatigue
        features['late_goal_vulnerability_home'] = self.calculate_late_goal_vulnerability(
            home_team, df, reference_date
        )
        features['late_goal_vulnerability_away'] = self.calculate_late_goal_vulnerability(
            away_team, df, reference_date
        )
        
        features['travel_fatigue_home'] = self.calculate_travel_fatigue_proxy(
            home_team, df, reference_date
        )
        features['travel_fatigue_away'] = self.calculate_travel_fatigue_proxy(
            away_team, df, reference_date
        )
        
        # Form momentum
        features['form_momentum_home'] = self.calculate_form_momentum_weighted(
            home_team, df, reference_date
        )
        features['form_momentum_away'] = self.calculate_form_momentum_weighted(
            away_team, df, reference_date
        )
        
        # Balance features
        features['possession_balance'] = self.calculate_possession_balance(
            home_team, away_team, df, reference_date
        )
        features['shot_balance'] = self.calculate_shot_balance(
            home_team, away_team, df, reference_date
        )
        
        # League-specific features
        if league in self.league_factors:
            league_info = self.league_factors[league]
            features['home_advantage_league_mean'] = league_info['home_advantage']
            features['league_competitiveness'] = league_info['competitiveness']
            features['league_avg_goals'] = league_info['avg_goals']
            features['league_draw_rate'] = league_info['draw_rate']
        else:
            # Default values for unknown leagues
            features['home_advantage_league_mean'] = 0.45
            features['league_competitiveness'] = 0.5
            features['league_avg_goals'] = 2.7
            features['league_draw_rate'] = 0.25
        
        # Derived features
        features['expected_points_diff'] = features['expected_points_home'] - features['expected_points_away']
        features['form_momentum_diff'] = features['form_momentum_home'] - features['form_momentum_away']
        features['fatigue_diff'] = features['travel_fatigue_away'] - features['travel_fatigue_home']  # Positive if away more fatigued
        features['vulnerability_diff'] = features['late_goal_vulnerability_away'] - features['late_goal_vulnerability_home']
        
        logger.info(f"✅ Created {len(features)} 1X2 features")
        return features
    
    def get_feature_names(self) -> List[str]:
        """
        Get list of all 1X2 feature names
        
        Returns:
            List of feature names
        """
        return [
            'match_difficulty_index',
            'expected_points_home',
            'expected_points_away',
            'late_goal_vulnerability_home',
            'late_goal_vulnerability_away',
            'travel_fatigue_home',
            'travel_fatigue_away',
            'form_momentum_home',
            'form_momentum_away',
            'possession_balance',
            'shot_balance',
            'home_advantage_league_mean',
            'league_competitiveness',
            'league_avg_goals',
            'league_draw_rate',
            'expected_points_diff',
            'form_momentum_diff',
            'fatigue_diff',
            'vulnerability_diff'
        ]


def main():
    """
    Example usage and testing
    """
    # Create sample data
    np.random.seed(42)
    
    dates = pd.date_range('2024-01-01', '2024-12-01', freq='D')
    teams = ['Team A', 'Team B', 'Team C', 'Team D', 'Team E']
    leagues = ['Premier League', 'La Liga']
    
    sample_data = []
    for i in range(500):
        date = np.random.choice(dates)
        home_team = np.random.choice(teams)
        away_team = np.random.choice([t for t in teams if t != home_team])
        league = np.random.choice(leagues)
        
        # Simulate realistic goal distributions
        home_score = np.random.poisson(1.5)
        away_score = np.random.poisson(1.2)
        
        sample_data.append({
            'date': date,
            'home_team': home_team,
            'away_team': away_team,
            'home_score': home_score,
            'away_score': away_score,
            'league': league
        })
    
    df = pd.DataFrame(sample_data)
    
    # Test feature engineering
    feature_engineer = Features1X2(lookback_days=180)
    
    # Calculate league factors
    league_factors = feature_engineer.calculate_league_factors(df)
    print("🏆 League Factors:")
    for league, factors in league_factors.items():
        print(f"  {league}: HA={factors['home_advantage']:.3f}, Goals={factors['avg_goals']:.2f}")
    
    # Create features for a sample match
    reference_date = datetime(2024, 11, 1)
    features = feature_engineer.create_1x2_features(
        'Team A', 'Team B', 'Premier League', df, reference_date
    )
    
    print(f"\n🔧 1X2 Features for Team A vs Team B:")
    for feature_name, value in features.items():
        print(f"  {feature_name}: {value:.4f}")
    
    print(f"\n📊 Total features created: {len(features)}")
    print("✅ 1X2 feature engineering test completed successfully!")


if __name__ == "__main__":
    main()
