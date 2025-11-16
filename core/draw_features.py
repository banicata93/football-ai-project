#!/usr/bin/env python3
"""
Draw Features - Specialized Features for Draw Prediction

This module creates features specifically designed to predict draws (X) in football matches.
Focus on balance, symmetry, and equilibrium indicators that correlate with draw outcomes.

IMPORTANT: This is ADDITIVE - does not modify existing feature engineering.
"""

import sys
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.utils import setup_logging


class DrawFeatures:
    """
    Specialized feature engineering for draw prediction
    
    Creates 8 draw-oriented features:
    1. possession_symmetry - How balanced possession is expected to be
    2. shot_balance - Expected shot balance between teams
    3. pace_of_play_proxy - Match pace indicator (draws often slower)
    4. defensive_stability_delta - Difference in defensive stability
    5. form_equilibrium_index - How similar recent form is
    6. xg_balance_proxy - Expected goals balance
    7. league_draw_rate - Historical draw rate for league
    8. home_vs_away_diff_compressed - Compressed strength difference
    """
    
    def __init__(self, lookback_days: int = 180, min_matches: int = 5):
        """
        Initialize draw feature engineer
        
        Args:
            lookback_days: Days to look back for historical data
            min_matches: Minimum matches required for calculations
        """
        self.lookback_days = lookback_days
        self.min_matches = min_matches
        self.logger = setup_logging()
        
        # Cache for league statistics
        self.league_stats_cache = {}
        self.team_stats_cache = {}
        
        self.logger.info(f"🎯 Initialized Draw Features (lookback={lookback_days} days)")
    
    def create_draw_features(self, home_team: str, away_team: str, league: str,
                           df: pd.DataFrame, reference_date: datetime) -> Dict[str, float]:
        """
        Create all draw-specific features for a match
        
        Args:
            home_team: Home team name
            away_team: Away team name
            league: League name
            df: Historical match data
            reference_date: Reference date for feature calculation
            
        Returns:
            Dictionary with draw features
        """
        try:
            self.logger.debug(f"🔧 Creating draw features for {home_team} vs {away_team}")
            
            # Filter recent data
            cutoff_date = reference_date - timedelta(days=self.lookback_days)
            recent_data = df[df['date'] >= cutoff_date].copy()
            
            if len(recent_data) < self.min_matches:
                self.logger.warning(f"⚠️ Insufficient data for draw features: {len(recent_data)} matches")
                return self._get_default_features()
            
            # Calculate all draw features
            features = {}
            
            # 1. Possession Symmetry
            features['possession_symmetry'] = self._calculate_possession_symmetry(
                home_team, away_team, recent_data
            )
            
            # 2. Shot Balance
            features['shot_balance'] = self._calculate_shot_balance(
                home_team, away_team, recent_data
            )
            
            # 3. Pace of Play Proxy
            features['pace_of_play_proxy'] = self._calculate_pace_proxy(
                home_team, away_team, recent_data
            )
            
            # 4. Defensive Stability Delta
            features['defensive_stability_delta'] = self._calculate_defensive_stability_delta(
                home_team, away_team, recent_data
            )
            
            # 5. Form Equilibrium Index
            features['form_equilibrium_index'] = self._calculate_form_equilibrium(
                home_team, away_team, recent_data, reference_date
            )
            
            # 6. xG Balance Proxy
            features['xg_balance_proxy'] = self._calculate_xg_balance_proxy(
                home_team, away_team, recent_data
            )
            
            # 7. League Draw Rate
            features['league_draw_rate'] = self._calculate_league_draw_rate(
                league, recent_data
            )
            
            # 8. Home vs Away Diff Compressed
            features['home_vs_away_diff_compressed'] = self._calculate_strength_diff_compressed(
                home_team, away_team, recent_data
            )
            
            self.logger.debug(f"✅ Created {len(features)} draw features")
            return features
            
        except Exception as e:
            self.logger.error(f"❌ Error creating draw features: {e}")
            return self._get_default_features()
    
    def _calculate_possession_symmetry(self, home_team: str, away_team: str, 
                                     df: pd.DataFrame) -> float:
        """
        Calculate expected possession symmetry (proxy for balanced match)
        
        Higher values indicate more balanced possession expected -> higher draw probability
        """
        try:
            # Get team attacking/defensive stats as proxy for possession
            home_attack = self._get_team_attack_strength(home_team, df)
            away_attack = self._get_team_attack_strength(away_team, df)
            
            home_defense = self._get_team_defense_strength(home_team, df)
            away_defense = self._get_team_defense_strength(away_team, df)
            
            # Calculate expected possession balance
            home_possession_proxy = (home_attack + away_defense) / 2
            away_possession_proxy = (away_attack + home_defense) / 2
            
            # Symmetry = 1 - |difference|, normalized to 0-1
            possession_diff = abs(home_possession_proxy - away_possession_proxy)
            symmetry = max(0.0, 1.0 - possession_diff)
            
            return min(1.0, symmetry)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error calculating possession symmetry: {e}")
            return 0.5  # Neutral
    
    def _calculate_shot_balance(self, home_team: str, away_team: str, 
                              df: pd.DataFrame) -> float:
        """
        Calculate expected shot balance between teams
        
        More balanced shots -> higher draw probability
        """
        try:
            # Proxy: use goals scored/conceded as shot indicators
            home_goals_for = self._get_avg_goals_scored(home_team, df, is_home=True)
            home_goals_against = self._get_avg_goals_conceded(home_team, df, is_home=True)
            
            away_goals_for = self._get_avg_goals_scored(away_team, df, is_home=False)
            away_goals_against = self._get_avg_goals_conceded(away_team, df, is_home=False)
            
            # Expected shots proxy
            home_shots_proxy = home_goals_for + away_goals_against
            away_shots_proxy = away_goals_for + home_goals_against
            
            # Balance = 1 - |difference|/sum
            total_shots = home_shots_proxy + away_shots_proxy
            if total_shots > 0:
                shot_diff = abs(home_shots_proxy - away_shots_proxy)
                balance = 1.0 - (shot_diff / total_shots)
            else:
                balance = 0.5
            
            return max(0.0, min(1.0, balance))
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error calculating shot balance: {e}")
            return 0.5
    
    def _calculate_pace_proxy(self, home_team: str, away_team: str, 
                            df: pd.DataFrame) -> float:
        """
        Calculate pace of play proxy
        
        Lower pace often correlates with draws (more defensive, cautious play)
        """
        try:
            # Use total goals as pace proxy
            home_pace = self._get_avg_total_goals_in_matches(home_team, df)
            away_pace = self._get_avg_total_goals_in_matches(away_team, df)
            
            # Average pace
            avg_pace = (home_pace + away_pace) / 2
            
            # Invert pace (lower pace = higher draw probability)
            # Normalize assuming max pace ~4 goals per game
            pace_proxy = max(0.0, 1.0 - (avg_pace / 4.0))
            
            return min(1.0, pace_proxy)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error calculating pace proxy: {e}")
            return 0.5
    
    def _calculate_defensive_stability_delta(self, home_team: str, away_team: str, 
                                           df: pd.DataFrame) -> float:
        """
        Calculate difference in defensive stability
        
        Similar defensive stability -> higher draw probability
        """
        try:
            home_defense_var = self._get_defensive_variance(home_team, df)
            away_defense_var = self._get_defensive_variance(away_team, df)
            
            # Similarity in defensive variance
            var_diff = abs(home_defense_var - away_defense_var)
            
            # Convert to similarity (lower difference = higher similarity)
            # Normalize assuming max variance difference ~2
            similarity = max(0.0, 1.0 - (var_diff / 2.0))
            
            return min(1.0, similarity)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error calculating defensive stability delta: {e}")
            return 0.5
    
    def _calculate_form_equilibrium(self, home_team: str, away_team: str, 
                                  df: pd.DataFrame, reference_date: datetime) -> float:
        """
        Calculate form equilibrium index
        
        Similar recent form -> higher draw probability
        """
        try:
            # Get recent form (last 5 matches)
            home_form = self._get_recent_form(home_team, df, reference_date, n_matches=5)
            away_form = self._get_recent_form(away_team, df, reference_date, n_matches=5)
            
            # Form similarity
            form_diff = abs(home_form - away_form)
            
            # Convert to equilibrium (lower difference = higher equilibrium)
            # Normalize assuming max form difference ~2 (range -1 to 1)
            equilibrium = max(0.0, 1.0 - (form_diff / 2.0))
            
            return min(1.0, equilibrium)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error calculating form equilibrium: {e}")
            return 0.5
    
    def _calculate_xg_balance_proxy(self, home_team: str, away_team: str, 
                                  df: pd.DataFrame) -> float:
        """
        Calculate expected goals balance proxy
        
        Balanced xG -> higher draw probability
        """
        try:
            # Use goals as xG proxy
            home_xg_for = self._get_avg_goals_scored(home_team, df, is_home=True)
            home_xg_against = self._get_avg_goals_conceded(home_team, df, is_home=True)
            
            away_xg_for = self._get_avg_goals_scored(away_team, df, is_home=False)
            away_xg_against = self._get_avg_goals_conceded(away_team, df, is_home=False)
            
            # Expected xG in head-to-head
            home_expected_xg = (home_xg_for + away_xg_against) / 2
            away_expected_xg = (away_xg_for + home_xg_against) / 2
            
            # Balance
            total_xg = home_expected_xg + away_expected_xg
            if total_xg > 0:
                xg_diff = abs(home_expected_xg - away_expected_xg)
                balance = 1.0 - (xg_diff / total_xg)
            else:
                balance = 0.5
            
            return max(0.0, min(1.0, balance))
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error calculating xG balance proxy: {e}")
            return 0.5
    
    def _calculate_league_draw_rate(self, league: str, df: pd.DataFrame) -> float:
        """
        Calculate historical draw rate for the league
        """
        try:
            if league in self.league_stats_cache:
                return self.league_stats_cache[league]['draw_rate']
            
            league_data = df[df['league'] == league].copy() if 'league' in df.columns else df
            
            if len(league_data) == 0:
                draw_rate = 0.25  # Default draw rate
            else:
                # Count draws
                if 'result' in league_data.columns:
                    draws = (league_data['result'] == 'X').sum()
                elif 'home_score' in league_data.columns and 'away_score' in league_data.columns:
                    draws = (league_data['home_score'] == league_data['away_score']).sum()
                else:
                    draws = len(league_data) * 0.25  # Estimate
                
                draw_rate = draws / len(league_data)
            
            # Cache result
            self.league_stats_cache[league] = {'draw_rate': draw_rate}
            
            return min(1.0, max(0.0, draw_rate))
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error calculating league draw rate: {e}")
            return 0.25
    
    def _calculate_strength_diff_compressed(self, home_team: str, away_team: str, 
                                          df: pd.DataFrame) -> float:
        """
        Calculate compressed strength difference
        
        Smaller strength difference -> higher draw probability
        """
        try:
            home_strength = self._get_team_overall_strength(home_team, df, is_home=True)
            away_strength = self._get_team_overall_strength(away_team, df, is_home=False)
            
            # Strength difference
            strength_diff = abs(home_strength - away_strength)
            
            # Compress using sigmoid-like function
            # Higher compression -> higher draw probability
            compressed = 1.0 / (1.0 + strength_diff)
            
            return min(1.0, max(0.0, compressed))
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error calculating strength difference: {e}")
            return 0.5
    
    # Helper methods for team statistics
    
    def _get_team_attack_strength(self, team: str, df: pd.DataFrame) -> float:
        """Get team's attack strength"""
        try:
            team_matches = df[(df['home_team'] == team) | (df['away_team'] == team)].copy()
            
            if len(team_matches) == 0:
                return 1.0
            
            total_goals = 0
            for _, match in team_matches.iterrows():
                if match['home_team'] == team:
                    total_goals += match.get('home_score', 0)
                else:
                    total_goals += match.get('away_score', 0)
            
            return total_goals / len(team_matches)
            
        except Exception:
            return 1.0
    
    def _get_team_defense_strength(self, team: str, df: pd.DataFrame) -> float:
        """Get team's defense strength (inverted goals conceded)"""
        try:
            team_matches = df[(df['home_team'] == team) | (df['away_team'] == team)].copy()
            
            if len(team_matches) == 0:
                return 1.0
            
            total_conceded = 0
            for _, match in team_matches.iterrows():
                if match['home_team'] == team:
                    total_conceded += match.get('away_score', 0)
                else:
                    total_conceded += match.get('home_score', 0)
            
            avg_conceded = total_conceded / len(team_matches)
            # Invert (lower conceded = higher defense strength)
            return max(0.1, 3.0 - avg_conceded)
            
        except Exception:
            return 1.0
    
    def _get_avg_goals_scored(self, team: str, df: pd.DataFrame, is_home: bool) -> float:
        """Get average goals scored by team (home or away)"""
        try:
            if is_home:
                team_matches = df[df['home_team'] == team].copy()
                goals = team_matches.get('home_score', pd.Series()).sum()
            else:
                team_matches = df[df['away_team'] == team].copy()
                goals = team_matches.get('away_score', pd.Series()).sum()
            
            return goals / max(1, len(team_matches))
            
        except Exception:
            return 1.0
    
    def _get_avg_goals_conceded(self, team: str, df: pd.DataFrame, is_home: bool) -> float:
        """Get average goals conceded by team (home or away)"""
        try:
            if is_home:
                team_matches = df[df['home_team'] == team].copy()
                goals = team_matches.get('away_score', pd.Series()).sum()
            else:
                team_matches = df[df['away_team'] == team].copy()
                goals = team_matches.get('home_score', pd.Series()).sum()
            
            return goals / max(1, len(team_matches))
            
        except Exception:
            return 1.0
    
    def _get_avg_total_goals_in_matches(self, team: str, df: pd.DataFrame) -> float:
        """Get average total goals in team's matches"""
        try:
            team_matches = df[(df['home_team'] == team) | (df['away_team'] == team)].copy()
            
            if len(team_matches) == 0:
                return 2.5
            
            total_goals = 0
            for _, match in team_matches.iterrows():
                total_goals += match.get('home_score', 0) + match.get('away_score', 0)
            
            return total_goals / len(team_matches)
            
        except Exception:
            return 2.5
    
    def _get_defensive_variance(self, team: str, df: pd.DataFrame) -> float:
        """Get variance in goals conceded (defensive stability)"""
        try:
            team_matches = df[(df['home_team'] == team) | (df['away_team'] == team)].copy()
            
            if len(team_matches) < 3:
                return 1.0
            
            goals_conceded = []
            for _, match in team_matches.iterrows():
                if match['home_team'] == team:
                    goals_conceded.append(match.get('away_score', 0))
                else:
                    goals_conceded.append(match.get('home_score', 0))
            
            return np.var(goals_conceded) if len(goals_conceded) > 1 else 1.0
            
        except Exception:
            return 1.0
    
    def _get_recent_form(self, team: str, df: pd.DataFrame, reference_date: datetime, 
                        n_matches: int = 5) -> float:
        """Get recent form score (-1 to 1, where 1 = all wins, -1 = all losses)"""
        try:
            # Get recent matches
            team_matches = df[(df['home_team'] == team) | (df['away_team'] == team)].copy()
            team_matches = team_matches[team_matches['date'] < reference_date]
            team_matches = team_matches.sort_values('date', ascending=False).head(n_matches)
            
            if len(team_matches) == 0:
                return 0.0
            
            form_points = 0
            for _, match in team_matches.iterrows():
                if match['home_team'] == team:
                    home_score = match.get('home_score', 0)
                    away_score = match.get('away_score', 0)
                    
                    if home_score > away_score:
                        form_points += 1  # Win
                    elif home_score == away_score:
                        form_points += 0  # Draw
                    else:
                        form_points -= 1  # Loss
                else:
                    home_score = match.get('home_score', 0)
                    away_score = match.get('away_score', 0)
                    
                    if away_score > home_score:
                        form_points += 1  # Win
                    elif away_score == home_score:
                        form_points += 0  # Draw
                    else:
                        form_points -= 1  # Loss
            
            # Normalize to -1 to 1
            return form_points / len(team_matches)
            
        except Exception:
            return 0.0
    
    def _get_team_overall_strength(self, team: str, df: pd.DataFrame, is_home: bool) -> float:
        """Get overall team strength (combination of attack and defense)"""
        try:
            attack = self._get_team_attack_strength(team, df)
            defense = self._get_team_defense_strength(team, df)
            
            # Home advantage
            home_bonus = 0.1 if is_home else 0.0
            
            return (attack + defense) / 2 + home_bonus
            
        except Exception:
            return 1.0
    
    def _get_default_features(self) -> Dict[str, float]:
        """Get default feature values when calculation fails"""
        return {
            'possession_symmetry': 0.5,
            'shot_balance': 0.5,
            'pace_of_play_proxy': 0.5,
            'defensive_stability_delta': 0.5,
            'form_equilibrium_index': 0.5,
            'xg_balance_proxy': 0.5,
            'league_draw_rate': 0.25,
            'home_vs_away_diff_compressed': 0.5
        }
    
    def get_feature_names(self) -> List[str]:
        """Get list of all draw feature names"""
        return [
            'possession_symmetry',
            'shot_balance', 
            'pace_of_play_proxy',
            'defensive_stability_delta',
            'form_equilibrium_index',
            'xg_balance_proxy',
            'league_draw_rate',
            'home_vs_away_diff_compressed'
        ]
    
    def get_feature_descriptions(self) -> Dict[str, str]:
        """Get descriptions of all draw features"""
        return {
            'possession_symmetry': 'Expected possession balance between teams (0-1)',
            'shot_balance': 'Expected shot balance indicator (0-1)',
            'pace_of_play_proxy': 'Match pace proxy, inverted (lower pace = higher draw prob)',
            'defensive_stability_delta': 'Similarity in defensive stability (0-1)',
            'form_equilibrium_index': 'Recent form similarity between teams (0-1)',
            'xg_balance_proxy': 'Expected goals balance proxy (0-1)',
            'league_draw_rate': 'Historical draw rate for the league (0-1)',
            'home_vs_away_diff_compressed': 'Compressed strength difference (0-1)'
        }


def main():
    """
    Example usage and testing
    """
    print("🎯 Testing Draw Features")
    print("=" * 40)
    
    # Create sample data
    sample_data = []
    teams = ['Team_A', 'Team_B', 'Team_C', 'Team_D']
    
    for i in range(100):
        date = datetime.now() - timedelta(days=i)
        home_team = np.random.choice(teams)
        away_team = np.random.choice([t for t in teams if t != home_team])
        
        # Simulate match
        home_score = np.random.poisson(1.5)
        away_score = np.random.poisson(1.2)
        
        sample_data.append({
            'date': date,
            'home_team': home_team,
            'away_team': away_team,
            'home_score': home_score,
            'away_score': away_score,
            'league': 'Test_League'
        })
    
    df = pd.DataFrame(sample_data)
    
    # Test feature engineering
    draw_features = DrawFeatures(lookback_days=90)
    
    # Create features for a test match
    features = draw_features.create_draw_features(
        home_team='Team_A',
        away_team='Team_B', 
        league='Test_League',
        df=df,
        reference_date=datetime.now()
    )
    
    print(f"\n🔧 Created {len(features)} draw features:")
    for feature_name, value in features.items():
        print(f"   {feature_name}: {value:.3f}")
    
    # Show feature descriptions
    print(f"\n📊 Feature Descriptions:")
    descriptions = draw_features.get_feature_descriptions()
    for name, desc in descriptions.items():
        print(f"   {name}: {desc}")
    
    print("\n✅ Draw Features test completed!")


if __name__ == "__main__":
    main()
