#!/usr/bin/env python3
"""
Scoreline Probability Engine

Generates full scoreline probability distribution (0-0, 1-0, 0-1, 1-1, etc.)
Combines Poisson v2 + ML corrections + tempo/xG adjustments + draw specialist.

ADDITIVE module - does not modify existing models.
"""

import sys
import os
import pickle
import json
import yaml
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from scipy.stats import poisson

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.utils import setup_logging
from core.poisson_v2 import PoissonV2Model


class ScorelineProbabilityEngine:
    """
    Scoreline Probability Engine
    
    Generates complete scoreline probability matrix combining:
    1. Poisson v2 base distributions
    2. ML correction factors
    3. Tempo/xG/defense adjustments
    4. Draw specialist corrections
    5. League-level scoring priors
    """
    
    def __init__(self, config_path: str = "config/scoreline_config.yaml"):
        """
        Initialize scoreline engine
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = Path(config_path)
        self.logger = setup_logging()
        
        # Load configuration
        self.config = self._load_config()
        
        # Components
        self.poisson_v2 = None
        self.correction_model = None
        self.draw_predictor = None
        
        # Cache
        self.league_priors = {}
        
        # Initialize components
        self._initialize_components()
        
        self.logger.info("🎯 Scoreline Probability Engine initialized")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    config_data = yaml.safe_load(f)
                # Extract the nested config
                config = config_data.get('scoreline_config', {})
                # Merge with defaults
                default_config = self._get_default_config()
                default_config.update(config.get('matrix', {}))
                default_config.update(config.get('engine_settings', {}))
                return default_config
            else:
                self.logger.warning(f"⚠️ Config file not found: {self.config_path}")
                return self._get_default_config()
        except Exception as e:
            self.logger.error(f"❌ Error loading config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'matrix_max_goals': 4,
            'apply_tempo': True,
            'apply_draw_corrections': True,
            'apply_league_scaling': True,
            'min_prob': 0.00001,
            'max_prob': 0.30,
            'correction_model_version': 'v1',
            'correction_model_path': 'models/scoreline_correction_v1',
            'log_file': 'logs/scoreline_engine.log'
        }
    
    def _initialize_components(self):
        """Initialize all engine components"""
        try:
            # Load Poisson v2
            self.load_poisson_v2()
            
            # Load correction model
            self.load_scoreline_correction_model()
            
            # Load draw predictor (optional)
            self._load_draw_predictor()
            
        except Exception as e:
            self.logger.error(f"❌ Error initializing components: {e}")
    
    def load_poisson_v2(self) -> bool:
        """
        Load Poisson v2 model
        
        Returns:
            True if loaded successfully
        """
        try:
            self.poisson_v2 = PoissonV2Model()
            self.logger.info("✅ Poisson v2 loaded")
            return True
        except Exception as e:
            self.logger.error(f"❌ Error loading Poisson v2: {e}")
            return False
    
    def load_scoreline_correction_model(self) -> bool:
        """
        Load ML correction model
        
        Returns:
            True if loaded successfully
        """
        try:
            model_path = Path(self.config['correction_model_path'])
            model_file = model_path / "correction_model.pkl"
            
            if model_file.exists():
                with open(model_file, 'rb') as f:
                    self.correction_model = pickle.load(f)
                self.logger.info("✅ Scoreline correction model loaded")
                return True
            else:
                self.logger.warning(f"⚠️ Correction model not found: {model_file}")
                return False
        except Exception as e:
            self.logger.error(f"❌ Error loading correction model: {e}")
            return False
    
    def _load_draw_predictor(self):
        """Load draw predictor (optional)"""
        try:
            from core.draw_predictor import DrawPredictor
            self.draw_predictor = DrawPredictor()
            self.logger.info("✅ Draw predictor loaded for scoreline engine")
        except Exception as e:
            self.logger.warning(f"⚠️ Draw predictor not available: {e}")
            self.draw_predictor = None
    
    def compute_base_poisson_matrix(self, lambda_home: float, lambda_away: float) -> np.ndarray:
        """
        Compute base Poisson probability matrix
        
        Args:
            lambda_home: Expected home goals
            lambda_away: Expected away goals
            
        Returns:
            Probability matrix [home_goals, away_goals]
        """
        max_goals = self.config['matrix_max_goals']
        matrix = np.zeros((max_goals + 1, max_goals + 1))
        
        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                prob_home = poisson.pmf(i, lambda_home)
                prob_away = poisson.pmf(j, lambda_away)
                matrix[i, j] = prob_home * prob_away
        
        return matrix
    
    def apply_ml_corrections(self, matrix: np.ndarray, features: Dict[str, float]) -> np.ndarray:
        """
        Apply ML correction factors to base matrix
        
        Args:
            matrix: Base Poisson matrix
            features: Match features for correction
            
        Returns:
            Corrected matrix
        """
        if self.correction_model is None:
            self.logger.warning("⚠️ No correction model available")
            return matrix
        
        try:
            corrected_matrix = matrix.copy()
            max_goals = self.config['matrix_max_goals']
            
            # Prepare features for all scoreline combinations
            for i in range(max_goals + 1):
                for j in range(max_goals + 1):
                    # Create features for this scoreline
                    scoreline_features = self._create_scoreline_features(i, j, features)
                    
                    # Get correction factor
                    correction = self._predict_correction_factor(scoreline_features)
                    
                    # Apply correction
                    corrected_matrix[i, j] *= correction
            
            return corrected_matrix
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error applying ML corrections: {e}")
            return matrix
    
    def _create_scoreline_features(self, home_goals: int, away_goals: int, 
                                 base_features: Dict[str, float]) -> List[float]:
        """Create features for specific scoreline prediction"""
        features = []
        
        # Base match features
        features.extend([
            base_features.get('xg_home', 1.5),
            base_features.get('xg_away', 1.2),
            base_features.get('tempo_proxy', 0.5),
            base_features.get('defensive_stability_home', 0.5),
            base_features.get('defensive_stability_away', 0.5),
            base_features.get('form_momentum_home', 0.0),
            base_features.get('form_momentum_away', 0.0),
            base_features.get('league_scoring_rate', 2.7),
            base_features.get('draw_specialist_prob', 0.25)
        ])
        
        # Scoreline-specific features
        features.extend([
            float(home_goals),
            float(away_goals),
            float(home_goals + away_goals),  # Total goals
            float(abs(home_goals - away_goals)),  # Goal difference
            float(home_goals == away_goals),  # Is draw
            float(home_goals > away_goals),   # Home win
            float(home_goals < away_goals),   # Away win
            float((home_goals + away_goals) > 2.5),  # Over 2.5
            float(home_goals > 0 and away_goals > 0)  # BTTS
        ])
        
        return features
    
    def _predict_correction_factor(self, features: List[float]) -> float:
        """Predict correction factor using ML model"""
        try:
            if self.correction_model is None:
                return 1.0
            
            # Convert to numpy array
            X = np.array(features).reshape(1, -1)
            
            # Predict correction factor
            correction = self.correction_model.predict(X)[0]
            
            # Clip to reasonable range
            return max(0.1, min(3.0, correction))
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error predicting correction: {e}")
            return 1.0
    
    def apply_draw_corrections(self, matrix: np.ndarray, p_draw_specialist: float = None) -> np.ndarray:
        """
        Apply draw specialist corrections to matrix
        
        Args:
            matrix: Current probability matrix
            p_draw_specialist: Draw specialist probability
            
        Returns:
            Draw-corrected matrix
        """
        if not self.config['apply_draw_corrections'] or p_draw_specialist is None:
            return matrix
        
        try:
            corrected_matrix = matrix.copy()
            max_goals = self.config['matrix_max_goals']
            
            # Calculate current draw probability from matrix
            current_draw_prob = sum(matrix[i, i] for i in range(max_goals + 1))
            
            if current_draw_prob > 0:
                # Calculate adjustment factor
                draw_adjustment = p_draw_specialist / current_draw_prob
                
                # Apply to draw scorelines
                for i in range(max_goals + 1):
                    corrected_matrix[i, i] *= draw_adjustment
                
                # Normalize to maintain probability sum
                corrected_matrix = self.normalize_matrix(corrected_matrix)
            
            return corrected_matrix
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error applying draw corrections: {e}")
            return matrix
    
    def apply_tempo_adjustments(self, matrix: np.ndarray, tempo_factor: float = 1.0) -> np.ndarray:
        """
        Apply tempo-based adjustments to matrix
        
        Args:
            matrix: Current probability matrix
            tempo_factor: Tempo adjustment factor (1.0 = normal)
            
        Returns:
            Tempo-adjusted matrix
        """
        if not self.config['apply_tempo']:
            return matrix
        
        try:
            adjusted_matrix = matrix.copy()
            max_goals = self.config['matrix_max_goals']
            
            # Higher tempo -> shift probability to higher-scoring games
            # Lower tempo -> shift probability to lower-scoring games
            for i in range(max_goals + 1):
                for j in range(max_goals + 1):
                    total_goals = i + j
                    
                    if tempo_factor > 1.0:
                        # High tempo: boost higher-scoring games
                        adjustment = 1.0 + (tempo_factor - 1.0) * (total_goals / (max_goals * 2))
                    else:
                        # Low tempo: boost lower-scoring games
                        adjustment = 1.0 - (1.0 - tempo_factor) * (total_goals / (max_goals * 2))
                    
                    adjusted_matrix[i, j] *= adjustment
            
            return self.normalize_matrix(adjusted_matrix)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error applying tempo adjustments: {e}")
            return matrix
    
    def apply_league_scaling(self, matrix: np.ndarray, league: str) -> np.ndarray:
        """
        Apply league-specific scaling factors
        
        Args:
            matrix: Current probability matrix
            league: League name
            
        Returns:
            League-scaled matrix
        """
        if not self.config['apply_league_scaling']:
            return matrix
        
        try:
            # Get league scoring profile (cached)
            league_profile = self._get_league_profile(league)
            
            scaled_matrix = matrix.copy()
            max_goals = self.config['matrix_max_goals']
            
            # Apply league-specific adjustments
            for i in range(max_goals + 1):
                for j in range(max_goals + 1):
                    total_goals = i + j
                    
                    # Get league adjustment for this goal total
                    if total_goals in league_profile:
                        scaling_factor = league_profile[total_goals]
                        scaled_matrix[i, j] *= scaling_factor
            
            return self.normalize_matrix(scaled_matrix)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error applying league scaling: {e}")
            return matrix
    
    def _get_league_profile(self, league: str) -> Dict[int, float]:
        """Get league-specific scoring profile"""
        if league in self.league_priors:
            return self.league_priors[league]
        
        # Default league profile (can be learned from data)
        default_profile = {
            0: 0.8,   # Low-scoring games less common
            1: 0.9,
            2: 1.0,   # Normal
            3: 1.1,
            4: 1.0,
            5: 0.9,
            6: 0.7,   # High-scoring games less common
            7: 0.5,
            8: 0.3
        }
        
        self.league_priors[league] = default_profile
        return default_profile
    
    def normalize_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """
        Normalize matrix so all probabilities sum to 1
        
        Args:
            matrix: Probability matrix
            
        Returns:
            Normalized matrix
        """
        total_prob = np.sum(matrix)
        if total_prob > 0:
            normalized = matrix / total_prob
            
            # Apply min/max probability constraints
            min_prob = self.config['min_prob']
            max_prob = self.config['max_prob']
            
            normalized = np.clip(normalized, min_prob, max_prob)
            
            # Renormalize after clipping
            return normalized / np.sum(normalized)
        else:
            # Fallback: uniform distribution
            return np.ones_like(matrix) / matrix.size
    
    def get_scoreline_probabilities(self, home_team: str, away_team: str, 
                                  league: str = None, df: pd.DataFrame = None,
                                  features: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Get complete scoreline probability distribution
        
        Args:
            home_team: Home team name
            away_team: Away team name
            league: League name
            df: Historical data
            features: Pre-computed features
            
        Returns:
            Complete scoreline analysis
        """
        try:
            self.logger.info(f"🎯 Computing scoreline probabilities: {home_team} vs {away_team}")
            
            # Get Poisson parameters
            lambda_home, lambda_away = self._get_poisson_parameters(
                home_team, away_team, league, df
            )
            
            # Compute base Poisson matrix
            base_matrix = self.compute_base_poisson_matrix(lambda_home, lambda_away)
            
            # Get match features
            if features is None:
                features = self._create_match_features(home_team, away_team, league, df)
            
            # Apply corrections sequentially
            corrected_matrix = base_matrix.copy()
            
            # 1. ML corrections
            corrected_matrix = self.apply_ml_corrections(corrected_matrix, features)
            
            # 2. Draw corrections
            p_draw_specialist = features.get('draw_specialist_prob')
            corrected_matrix = self.apply_draw_corrections(corrected_matrix, p_draw_specialist)
            
            # 3. Tempo adjustments
            tempo_factor = features.get('tempo_proxy', 1.0)
            corrected_matrix = self.apply_tempo_adjustments(corrected_matrix, tempo_factor)
            
            # 4. League scaling
            corrected_matrix = self.apply_league_scaling(corrected_matrix, league or 'default')
            
            # 5. Final normalization
            final_matrix = self.normalize_matrix(corrected_matrix)
            
            # Convert to scoreline dictionary and compute summary metrics
            return self._build_result(final_matrix, lambda_home, lambda_away, features)
            
        except Exception as e:
            self.logger.error(f"❌ Error computing scoreline probabilities: {e}")
            return self._fallback_result(home_team, away_team)
    
    def _get_poisson_parameters(self, home_team: str, away_team: str, 
                              league: str, df: pd.DataFrame) -> Tuple[float, float]:
        """Get Poisson lambda parameters"""
        try:
            if self.poisson_v2 and df is not None:
                # Use Poisson v2 to get expected goals
                poisson_result = self.poisson_v2.predict_match(home_team, away_team, league)
                
                lambda_home = poisson_result.get('expected_home_goals', 1.5)
                lambda_away = poisson_result.get('expected_away_goals', 1.2)
            else:
                # Fallback to default values
                lambda_home = 1.5
                lambda_away = 1.2
            
            return lambda_home, lambda_away
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error getting Poisson parameters: {e}")
            return 1.5, 1.2
    
    def _create_match_features(self, home_team: str, away_team: str, 
                             league: str, df: pd.DataFrame) -> Dict[str, float]:
        """Create features for match prediction"""
        features = {
            'xg_home': 1.5,
            'xg_away': 1.2,
            'tempo_proxy': 1.0,
            'defensive_stability_home': 0.5,
            'defensive_stability_away': 0.5,
            'form_momentum_home': 0.0,
            'form_momentum_away': 0.0,
            'league_scoring_rate': 2.7,
            'draw_specialist_prob': 0.25
        }
        
        try:
            # Get draw specialist probability if available
            if self.draw_predictor and df is not None:
                draw_result = self.draw_predictor.predict_draw_probability(
                    home_team, away_team, league or 'unknown', df
                )
                features['draw_specialist_prob'] = draw_result.get('draw_probability', 0.25)
            
            # Add more sophisticated feature computation here
            # (team form, xG estimates, tempo, etc.)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error creating match features: {e}")
        
        return features
    
    def _build_result(self, matrix: np.ndarray, lambda_home: float, lambda_away: float,
                     features: Dict[str, float]) -> Dict[str, Any]:
        """Build final result dictionary"""
        max_goals = self.config['matrix_max_goals']
        
        # Convert matrix to scoreline dictionary
        scoreline_probs = {}
        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                scoreline = f"{i}-{j}"
                scoreline_probs[scoreline] = float(matrix[i, j])
        
        # Compute summary metrics
        p_home = sum(matrix[i, j] for i in range(max_goals + 1) 
                    for j in range(max_goals + 1) if i > j)
        p_draw = sum(matrix[i, i] for i in range(max_goals + 1))
        p_away = sum(matrix[i, j] for i in range(max_goals + 1) 
                    for j in range(max_goals + 1) if i < j)
        
        # BTTS probability
        btts_prob = sum(matrix[i, j] for i in range(1, max_goals + 1) 
                       for j in range(1, max_goals + 1))
        
        # Over 2.5 probability
        over25_prob = sum(matrix[i, j] for i in range(max_goals + 1) 
                         for j in range(max_goals + 1) if i + j > 2.5)
        
        return {
            'matrix': scoreline_probs,
            'summary': {
                'p_home': p_home,
                'p_draw': p_draw,
                'p_away': p_away,
                'xGF': lambda_home,
                'xGA': lambda_away,
                'btts_prob': btts_prob,
                'over25_prob': over25_prob
            },
            'features_used': features,
            'engine_version': 'scoreline_v1',
            'matrix_size': f"{max_goals + 1}x{max_goals + 1}",
            'total_combinations': (max_goals + 1) ** 2
        }
    
    def _fallback_result(self, home_team: str, away_team: str) -> Dict[str, Any]:
        """Fallback result when computation fails"""
        max_goals = self.config['matrix_max_goals']
        
        # Create uniform distribution as fallback
        uniform_prob = 1.0 / ((max_goals + 1) ** 2)
        scoreline_probs = {}
        
        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                scoreline = f"{i}-{j}"
                scoreline_probs[scoreline] = uniform_prob
        
        return {
            'matrix': scoreline_probs,
            'summary': {
                'p_home': 0.33,
                'p_draw': 0.34,
                'p_away': 0.33,
                'xGF': 1.5,
                'xGA': 1.2,
                'btts_prob': 0.5,
                'over25_prob': 0.5
            },
            'features_used': {},
            'engine_version': 'scoreline_v1_fallback',
            'matrix_size': f"{max_goals + 1}x{max_goals + 1}",
            'total_combinations': (max_goals + 1) ** 2,
            'fallback_reason': 'Computation error'
        }
    
    def get_summary_metrics(self, result: Dict[str, Any]) -> Dict[str, float]:
        """Extract summary metrics from scoreline result"""
        return result.get('summary', {})
    
    def get_most_likely_scorelines(self, result: Dict[str, Any], top_n: int = 5) -> List[Tuple[str, float]]:
        """Get most likely scorelines"""
        matrix = result.get('matrix', {})
        sorted_scorelines = sorted(matrix.items(), key=lambda x: x[1], reverse=True)
        return sorted_scorelines[:top_n]


def main():
    """
    Example usage and testing
    """
    print("🎯 Testing Scoreline Probability Engine")
    print("=" * 50)
    
    # Initialize engine
    engine = ScorelineProbabilityEngine()
    
    # Create sample data
    sample_data = []
    teams = ['Team_A', 'Team_B', 'Team_C', 'Team_D']
    
    for i in range(50):
        date = datetime.now() - pd.Timedelta(days=i)
        home_team = np.random.choice(teams)
        away_team = np.random.choice([t for t in teams if t != home_team])
        
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
    
    # Test scoreline prediction
    result = engine.get_scoreline_probabilities(
        home_team='Team_A',
        away_team='Team_B',
        league='Test_League',
        df=df
    )
    
    print(f"\n🎯 Scoreline Analysis:")
    print(f"   Engine Version: {result['engine_version']}")
    print(f"   Matrix Size: {result['matrix_size']}")
    
    print(f"\n📊 Summary Metrics:")
    summary = result['summary']
    print(f"   Home Win: {summary['p_home']:.3f}")
    print(f"   Draw: {summary['p_draw']:.3f}")
    print(f"   Away Win: {summary['p_away']:.3f}")
    print(f"   BTTS: {summary['btts_prob']:.3f}")
    print(f"   Over 2.5: {summary['over25_prob']:.3f}")
    
    print(f"\n🎯 Most Likely Scorelines:")
    top_scorelines = engine.get_most_likely_scorelines(result, top_n=5)
    for i, (scoreline, prob) in enumerate(top_scorelines, 1):
        print(f"   {i}. {scoreline}: {prob:.3f}")
    
    print("\n✅ Scoreline Engine test completed!")


if __name__ == "__main__":
    import pandas as pd
    main()
