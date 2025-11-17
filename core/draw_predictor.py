#!/usr/bin/env python3
"""
Draw Predictor - Specialized Binary Classifier for Draw Prediction

Loads the trained draw model and combines it with other draw probability sources:
- ML 1X2 probabilities (existing)
- Poisson draw probability
- League draw rate prior
- Specialized draw model prediction

IMPORTANT: This is ADDITIVE - does not modify existing prediction logic.
"""

import sys
import os
import pickle
import json
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.utils import setup_logging
from core.draw_features import DrawFeatures

# Import LGBWrapper shim for backward compatibility with pickled models
# This allows loading Draw Specialist v1 model without retraining
from core.lgb_wrapper_shim import LGBWrapper
# Make it available in __main__ namespace for pickle
sys.modules['__main__'].LGBWrapper = LGBWrapper


class DrawPredictor:
    """
    Specialized predictor for draw outcomes
    
    Combines multiple draw probability sources:
    1. Trained binary draw model (draw vs non-draw)
    2. ML 1X2 draw probability
    3. Poisson draw probability  
    4. League draw rate prior
    
    Uses weighted combination with learned weights.
    """
    
    def __init__(self, model_path: str = "models/draw_model_v1"):
        """
        Initialize draw predictor
        
        Args:
            model_path: Path to draw model directory
        """
        self.model_path = Path(model_path)
        self.logger = setup_logging()
        
        # Model components
        self.draw_model = None
        self.feature_list = None
        self.model_metrics = None
        
        # Feature engineering
        self.draw_features = DrawFeatures()
        
        # Combination weights (can be optimized)
        self.combination_weights = {
            'draw_model': 0.4,    # Specialized draw model
            'ml_1x2': 0.3,        # ML 1X2 draw probability
            'poisson': 0.2,       # Poisson draw probability
            'league_prior': 0.1   # League draw rate prior
        }
        
        # Load model
        self.is_loaded = self._load_model()
        
        if self.is_loaded:
            self.logger.info("🎯 Draw Predictor initialized successfully")
        else:
            self.logger.warning("⚠️ Draw Predictor initialized without model (fallback mode)")
    
    def _load_model(self) -> bool:
        """
        Load the trained draw model and metadata
        
        Returns:
            True if model loaded successfully
        """
        try:
            model_file = self.model_path / "draw_model.pkl"
            feature_file = self.model_path / "feature_list.json"
            metrics_file = self.model_path / "metrics.json"
            
            if not model_file.exists():
                self.logger.warning(f"⚠️ Draw model not found: {model_file}")
                return False
            
            # Load model
            with open(model_file, 'rb') as f:
                self.draw_model = pickle.load(f)
            
            # Load feature list
            if feature_file.exists():
                with open(feature_file, 'r') as f:
                    self.feature_list = json.load(f)
            else:
                self.logger.warning("⚠️ Feature list not found, using default")
                self.feature_list = self.draw_features.get_feature_names()
            
            # Load metrics
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    self.model_metrics = json.load(f)
                    
                self.logger.info(f"📊 Model metrics - Accuracy: {self.model_metrics.get('accuracy', 'N/A'):.3f}")
                self.logger.info(f"📊 Draw Recall: {self.model_metrics.get('draw_recall', 'N/A'):.3f}")
            
            self.logger.info(f"✅ Loaded draw model with {len(self.feature_list)} features")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error loading draw model: {e}")
            return False
    
    def predict_draw_probability(self, 
                               home_team: str, 
                               away_team: str, 
                               league: str,
                               df: pd.DataFrame,
                               reference_date: datetime = None,
                               p_ml_draw: float = None,
                               p_poisson_draw: float = None) -> Dict[str, float]:
        """
        Predict draw probability using combined approach
        
        Args:
            home_team: Home team name
            away_team: Away team name
            league: League name
            df: Historical match data
            reference_date: Reference date for prediction
            p_ml_draw: ML 1X2 draw probability (optional)
            p_poisson_draw: Poisson draw probability (optional)
            
        Returns:
            Dictionary with draw prediction results
        """
        if reference_date is None:
            reference_date = datetime.now()
        
        try:
            self.logger.debug(f"🎯 Predicting draw probability for {home_team} vs {away_team}")
            
            # Get all probability components
            probabilities = self._get_all_draw_probabilities(
                home_team, away_team, league, df, reference_date,
                p_ml_draw, p_poisson_draw
            )
            
            # Combine probabilities
            final_prob = self._combine_probabilities(probabilities)
            
            # Build result
            result = {
                'draw_probability': final_prob,
                'confidence': self._calculate_confidence(probabilities),
                'components': probabilities,
                'weights_used': self.combination_weights.copy(),
                'model_version': 'draw_predictor_v1',
                'is_model_loaded': self.is_loaded
            }
            
            self.logger.debug(f"✅ Draw probability: {final_prob:.3f}")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Error predicting draw probability: {e}")
            return self._fallback_prediction(p_ml_draw)
    
    def _get_all_draw_probabilities(self, 
                                  home_team: str, 
                                  away_team: str, 
                                  league: str,
                                  df: pd.DataFrame,
                                  reference_date: datetime,
                                  p_ml_draw: float = None,
                                  p_poisson_draw: float = None) -> Dict[str, float]:
        """
        Get draw probabilities from all sources
        
        Returns:
            Dictionary with probabilities from each component
        """
        probabilities = {}
        
        # 1. Specialized draw model prediction
        if self.is_loaded:
            probabilities['draw_model'] = self._predict_with_draw_model(
                home_team, away_team, league, df, reference_date
            )
        else:
            probabilities['draw_model'] = 0.25  # Default
        
        # 2. ML 1X2 draw probability
        probabilities['ml_1x2'] = p_ml_draw if p_ml_draw is not None else 0.25
        
        # 3. Poisson draw probability
        probabilities['poisson'] = p_poisson_draw if p_poisson_draw is not None else 0.25
        
        # 4. League draw rate prior
        probabilities['league_prior'] = self._get_league_draw_rate(league, df)
        
        return probabilities
    
    def _predict_with_draw_model(self, 
                               home_team: str, 
                               away_team: str, 
                               league: str,
                               df: pd.DataFrame,
                               reference_date: datetime) -> float:
        """
        Make prediction using the specialized draw model
        
        Returns:
            Draw probability from the model
        """
        try:
            # Create draw features
            features = self.draw_features.create_draw_features(
                home_team, away_team, league, df, reference_date
            )
            
            # Prepare feature vector
            feature_vector = []
            for feature_name in self.feature_list:
                feature_vector.append(features.get(feature_name, 0.5))
            
            # Make prediction
            X = np.array(feature_vector).reshape(1, -1)
            
            # Get probability (assuming binary classifier with predict_proba)
            if hasattr(self.draw_model, 'predict_proba'):
                prob_array = self.draw_model.predict_proba(X)
                # prob_array[0][1] is probability of positive class (draw)
                draw_prob = prob_array[0][1]
            else:
                # Fallback for models without predict_proba
                prediction = self.draw_model.predict(X)[0]
                draw_prob = prediction if 0 <= prediction <= 1 else 0.5
            
            return max(0.01, min(0.99, draw_prob))  # Clip to valid range
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error in draw model prediction: {e}")
            return 0.25
    
    def _get_league_draw_rate(self, league: str, df: pd.DataFrame) -> float:
        """
        Get historical draw rate for the league
        
        Returns:
            League draw rate
        """
        try:
            # Use draw features method
            return self.draw_features._calculate_league_draw_rate(league, df)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error getting league draw rate: {e}")
            return 0.25
    
    def _combine_probabilities(self, probabilities: Dict[str, float]) -> float:
        """
        Combine probabilities using weighted average
        
        Args:
            probabilities: Dictionary with component probabilities
            
        Returns:
            Final combined draw probability
        """
        try:
            weighted_sum = 0.0
            total_weight = 0.0
            
            for component, weight in self.combination_weights.items():
                if component in probabilities:
                    prob = probabilities[component]
                    # Ensure valid probability
                    prob = max(0.01, min(0.99, prob))
                    
                    weighted_sum += weight * prob
                    total_weight += weight
            
            if total_weight > 0:
                final_prob = weighted_sum / total_weight
            else:
                final_prob = 0.25  # Default
            
            return max(0.01, min(0.99, final_prob))
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error combining probabilities: {e}")
            return 0.25
    
    def _calculate_confidence(self, probabilities: Dict[str, float]) -> float:
        """
        Calculate confidence in the prediction
        
        Args:
            probabilities: Component probabilities
            
        Returns:
            Confidence score (0-1)
        """
        try:
            # Calculate agreement between components
            prob_values = list(probabilities.values())
            
            if len(prob_values) < 2:
                return 0.5
            
            # Standard deviation of probabilities (lower = higher agreement)
            std_dev = np.std(prob_values)
            
            # Convert to confidence (lower std = higher confidence)
            # Assuming max std ~0.3 for reasonable disagreement
            confidence = max(0.1, 1.0 - (std_dev / 0.3))
            
            return min(1.0, confidence)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error calculating confidence: {e}")
            return 0.5
    
    def _fallback_prediction(self, p_ml_draw: float = None) -> Dict[str, float]:
        """
        Fallback prediction when main prediction fails
        
        Args:
            p_ml_draw: ML draw probability if available
            
        Returns:
            Fallback prediction result
        """
        fallback_prob = p_ml_draw if p_ml_draw is not None else 0.25
        
        return {
            'draw_probability': fallback_prob,
            'confidence': 0.3,  # Low confidence for fallback
            'components': {
                'draw_model': 0.25,
                'ml_1x2': fallback_prob,
                'poisson': 0.25,
                'league_prior': 0.25
            },
            'weights_used': {'ml_1x2': 1.0, 'draw_model': 0.0, 'poisson': 0.0, 'league_prior': 0.0},
            'model_version': 'draw_predictor_v1_fallback',
            'is_model_loaded': False,
            'fallback_reason': 'Prediction error'
        }
    
    def update_combination_weights(self, new_weights: Dict[str, float]) -> bool:
        """
        Update combination weights
        
        Args:
            new_weights: New weight dictionary
            
        Returns:
            True if weights updated successfully
        """
        try:
            # Validate weights
            total_weight = sum(new_weights.values())
            if abs(total_weight - 1.0) > 0.01:
                self.logger.warning(f"⚠️ Weights don't sum to 1.0: {total_weight}")
                # Normalize
                for key in new_weights:
                    new_weights[key] /= total_weight
            
            self.combination_weights = new_weights.copy()
            self.logger.info(f"✅ Updated combination weights: {self.combination_weights}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error updating weights: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, any]:
        """
        Get information about the draw predictor
        
        Returns:
            Dictionary with model information
        """
        return {
            'version': '1.0',
            'type': 'draw_specialist_predictor',
            'model_loaded': self.is_loaded,
            'model_path': str(self.model_path),
            'feature_count': len(self.feature_list) if self.feature_list else 0,
            'combination_weights': self.combination_weights.copy(),
            'metrics': self.model_metrics.copy() if self.model_metrics else None,
            'components': {
                'draw_model': 'Specialized binary classifier (draw vs non-draw)',
                'ml_1x2': 'ML 1X2 draw probability',
                'poisson': 'Poisson draw probability',
                'league_prior': 'Historical league draw rate'
            }
        }


def main():
    """
    Example usage and testing
    """
    print("🎯 Testing Draw Predictor")
    print("=" * 40)
    
    # Initialize predictor
    predictor = DrawPredictor()
    
    # Create sample data
    sample_data = []
    teams = ['Team_A', 'Team_B', 'Team_C', 'Team_D']
    
    for i in range(50):
        date = datetime.now() - timedelta(days=i)
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
    
    # Test prediction
    result = predictor.predict_draw_probability(
        home_team='Team_A',
        away_team='Team_B',
        league='Test_League',
        df=df,
        p_ml_draw=0.3,
        p_poisson_draw=0.25
    )
    
    print(f"\n🎯 Draw Prediction Result:")
    print(f"   Draw Probability: {result['draw_probability']:.3f}")
    print(f"   Confidence: {result['confidence']:.3f}")
    print(f"   Model Version: {result['model_version']}")
    print(f"   Model Loaded: {result['is_model_loaded']}")
    
    print(f"\n📊 Component Probabilities:")
    for component, prob in result['components'].items():
        weight = result['weights_used'].get(component, 0)
        print(f"   {component}: {prob:.3f} (weight: {weight:.3f})")
    
    # Show model info
    print(f"\n🔧 Model Information:")
    info = predictor.get_model_info()
    for key, value in info.items():
        if key != 'components':
            print(f"   {key}: {value}")
    
    print("\n✅ Draw Predictor test completed!")


if __name__ == "__main__":
    from datetime import timedelta
    main()
