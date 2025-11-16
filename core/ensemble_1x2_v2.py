#!/usr/bin/env python3
"""
Ensemble 1X2 v2 - Optimized Ensemble for 1X2 Predictions

NEW ensemble class that uses optimized weights from ensemble_optimizer_1x2.py
This is ADDITIVE - does not modify existing ensemble logic.

Combines:
- ML 1X2 v2 (three-binary-model system)
- Poisson v2 1X2 probabilities
- Elo 1X2 probabilities (if available)

Uses optimized weights from config/ensemble_1x2_weights.yaml
"""

import os
import sys
import yaml
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.utils import setup_logging


class Ensemble1X2V2:
    """
    Optimized ensemble for 1X2 predictions using learned weights
    
    This class:
    1. Loads optimized weights from config file
    2. Applies weighted average of ML, Poisson, and Elo predictions
    3. Returns final 1X2 probability distribution
    4. Has safe fallback to ML-only predictions
    """
    
    def __init__(self, config_path: str = "config/ensemble_1x2_weights.yaml"):
        """
        Initialize ensemble with optimized weights
        
        Args:
            config_path: Path to weights configuration file
        """
        self.config_path = config_path
        self.logger = setup_logging()
        
        # Default weights (fallback)
        self.default_weights = {
            'ml': 0.6,
            'poisson': 0.3,
            'elo': 0.1
        }
        
        # Load optimized weights
        self.weights = self._load_weights()
        self.is_optimized = self.weights != self.default_weights
        
        # Performance tracking
        self.prediction_count = 0
        self.fallback_count = 0
        
        self.logger.info(f"🎯 Initialized Ensemble 1X2 v2")
        self.logger.info(f"📊 Weights - ML: {self.weights['ml']:.3f}, Poisson: {self.weights['poisson']:.3f}, Elo: {self.weights['elo']:.3f}")
        self.logger.info(f"🔧 Using {'optimized' if self.is_optimized else 'default'} weights")
    
    def _load_weights(self) -> Dict[str, float]:
        """
        Load ensemble weights from configuration file
        
        Returns:
            Dictionary with weights for each component
        """
        config_file = Path(self.config_path)
        
        if not config_file.exists():
            self.logger.warning(f"⚠️ Weights config not found: {config_file}")
            self.logger.info("🔄 Using default weights")
            return self.default_weights.copy()
        
        try:
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f)
            
            # Extract weights
            ensemble_config = config_data.get('ensemble_1x2_weights', {})
            weights_config = ensemble_config.get('weights', {})
            
            weights = {
                'ml': weights_config.get('ml', self.default_weights['ml']),
                'poisson': weights_config.get('poisson', self.default_weights['poisson']),
                'elo': weights_config.get('elo', self.default_weights['elo'])
            }
            
            # Validate weights
            total_weight = sum(weights.values())
            if abs(total_weight - 1.0) > 0.01:
                self.logger.warning(f"⚠️ Weights don't sum to 1.0 ({total_weight:.3f}), normalizing...")
                for key in weights:
                    weights[key] /= total_weight
            
            # Log metadata
            last_updated = ensemble_config.get('last_updated', 'Unknown')
            performance = ensemble_config.get('performance_metrics', {})
            
            self.logger.info(f"✅ Loaded optimized weights from {config_file}")
            self.logger.info(f"📅 Last updated: {last_updated}")
            
            if performance:
                log_loss = performance.get('log_loss', 0)
                accuracy = performance.get('accuracy', 0)
                self.logger.info(f"📊 Performance - Log Loss: {log_loss:.4f}, Accuracy: {accuracy:.4f}")
            
            return weights
            
        except Exception as e:
            self.logger.error(f"❌ Error loading weights config: {e}")
            self.logger.info("🔄 Falling back to default weights")
            return self.default_weights.copy()
    
    def reload_weights(self) -> bool:
        """
        Reload weights from configuration file
        
        Returns:
            True if weights were successfully reloaded
        """
        try:
            new_weights = self._load_weights()
            
            if new_weights != self.weights:
                old_weights = self.weights.copy()
                self.weights = new_weights
                self.is_optimized = self.weights != self.default_weights
                
                self.logger.info("🔄 Weights reloaded successfully")
                self.logger.info(f"📊 Old: ML={old_weights['ml']:.3f}, Poisson={old_weights['poisson']:.3f}, Elo={old_weights['elo']:.3f}")
                self.logger.info(f"📊 New: ML={self.weights['ml']:.3f}, Poisson={self.weights['poisson']:.3f}, Elo={self.weights['elo']:.3f}")
                return True
            else:
                self.logger.info("📊 Weights unchanged")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Error reloading weights: {e}")
            return False
    
    def predict_1x2(self, 
                    p_ml: List[float], 
                    p_poisson: Optional[List[float]] = None,
                    p_elo: Optional[List[float]] = None) -> Dict[str, float]:
        """
        Generate ensemble 1X2 prediction
        
        Args:
            p_ml: ML probabilities [home, draw, away]
            p_poisson: Poisson probabilities [home, draw, away] (optional)
            p_elo: Elo probabilities [home, draw, away] (optional)
            
        Returns:
            Dictionary with ensemble prediction
        """
        self.prediction_count += 1
        
        try:
            # Validate ML probabilities
            if not self._validate_probabilities(p_ml):
                self.logger.warning("⚠️ Invalid ML probabilities, using fallback")
                return self._fallback_prediction(p_ml)
            
            # Convert to numpy arrays
            p_ml_array = np.array(p_ml)
            
            # Handle missing components
            if p_poisson is None or not self._validate_probabilities(p_poisson):
                p_poisson_array = p_ml_array.copy()  # Fallback to ML
                poisson_available = False
            else:
                p_poisson_array = np.array(p_poisson)
                poisson_available = True
            
            if p_elo is None or not self._validate_probabilities(p_elo):
                p_elo_array = p_ml_array.copy()  # Fallback to ML
                elo_available = False
            else:
                p_elo_array = np.array(p_elo)
                elo_available = True
            
            # Calculate ensemble probabilities
            ensemble_probs = (
                self.weights['ml'] * p_ml_array +
                self.weights['poisson'] * p_poisson_array +
                self.weights['elo'] * p_elo_array
            )
            
            # Normalize to ensure probabilities sum to 1
            total_prob = np.sum(ensemble_probs)
            if total_prob > 0:
                ensemble_probs = ensemble_probs / total_prob
            else:
                # Emergency fallback
                ensemble_probs = np.array([0.33, 0.34, 0.33])
            
            # Determine predicted outcome
            predicted_class = np.argmax(ensemble_probs)
            outcome_map = {0: '1', 1: 'X', 2: '2'}
            predicted_outcome = outcome_map[predicted_class]
            
            # Calculate confidence (max probability)
            confidence = float(np.max(ensemble_probs))
            
            # Build result
            result = {
                'prob_home_win': float(ensemble_probs[0]),
                'prob_draw': float(ensemble_probs[1]),
                'prob_away_win': float(ensemble_probs[2]),
                'predicted_outcome': predicted_outcome,
                'confidence': confidence,
                'model_version': 'ensemble_1x2_v2',
                'ensemble_info': {
                    'weights_used': self.weights.copy(),
                    'components_available': {
                        'ml': True,
                        'poisson': poisson_available,
                        'elo': elo_available
                    },
                    'is_optimized': self.is_optimized
                }
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Error in ensemble prediction: {e}")
            return self._fallback_prediction(p_ml)
    
    def _validate_probabilities(self, probs: Optional[List[float]]) -> bool:
        """
        Validate probability array
        
        Args:
            probs: Probability array to validate
            
        Returns:
            True if valid
        """
        if probs is None:
            return False
        
        if len(probs) != 3:
            return False
        
        # Check for valid numbers
        for p in probs:
            if not isinstance(p, (int, float)) or np.isnan(p) or np.isinf(p):
                return False
            if p < 0 or p > 1:
                return False
        
        # Check if probabilities sum to approximately 1
        total = sum(probs)
        if abs(total - 1.0) > 0.1:  # Allow some tolerance
            return False
        
        return True
    
    def _fallback_prediction(self, p_ml: List[float]) -> Dict[str, float]:
        """
        Fallback prediction using ML probabilities only
        
        Args:
            p_ml: ML probabilities
            
        Returns:
            Fallback prediction result
        """
        self.fallback_count += 1
        
        try:
            # Ensure valid probabilities
            if not self._validate_probabilities(p_ml):
                # Emergency default
                p_ml = [0.45, 0.25, 0.30]
            
            probs = np.array(p_ml)
            
            # Normalize
            total = np.sum(probs)
            if total > 0:
                probs = probs / total
            
            predicted_class = np.argmax(probs)
            outcome_map = {0: '1', 1: 'X', 2: '2'}
            predicted_outcome = outcome_map[predicted_class]
            
            result = {
                'prob_home_win': float(probs[0]),
                'prob_draw': float(probs[1]),
                'prob_away_win': float(probs[2]),
                'predicted_outcome': predicted_outcome,
                'confidence': float(np.max(probs)),
                'model_version': 'ensemble_1x2_v2_fallback',
                'ensemble_info': {
                    'weights_used': {'ml': 1.0, 'poisson': 0.0, 'elo': 0.0},
                    'components_available': {'ml': True, 'poisson': False, 'elo': False},
                    'is_optimized': False,
                    'fallback_reason': 'Invalid input probabilities'
                }
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Error in fallback prediction: {e}")
            
            # Ultimate fallback
            return {
                'prob_home_win': 0.45,
                'prob_draw': 0.25,
                'prob_away_win': 0.30,
                'predicted_outcome': '1',
                'confidence': 0.45,
                'model_version': 'ensemble_1x2_v2_emergency',
                'ensemble_info': {
                    'weights_used': {'ml': 1.0, 'poisson': 0.0, 'elo': 0.0},
                    'components_available': {'ml': False, 'poisson': False, 'elo': False},
                    'is_optimized': False,
                    'fallback_reason': 'Emergency fallback'
                }
            }
    
    def get_statistics(self) -> Dict[str, any]:
        """
        Get ensemble usage statistics
        
        Returns:
            Dictionary with statistics
        """
        fallback_rate = self.fallback_count / max(self.prediction_count, 1)
        
        return {
            'total_predictions': self.prediction_count,
            'fallback_predictions': self.fallback_count,
            'fallback_rate': fallback_rate,
            'weights_optimized': self.is_optimized,
            'current_weights': self.weights.copy(),
            'config_path': self.config_path
        }
    
    def get_ensemble_info(self) -> Dict[str, any]:
        """
        Get detailed ensemble information
        
        Returns:
            Dictionary with ensemble metadata
        """
        return {
            'version': '2.0',
            'type': 'optimized_ensemble_1x2',
            'weights': self.weights.copy(),
            'is_optimized': self.is_optimized,
            'config_file': self.config_path,
            'statistics': self.get_statistics(),
            'components': {
                'ml_1x2_v2': 'Three binary models (homewin, draw, awaywin)',
                'poisson_v2': 'Enhanced Poisson with time-decay',
                'elo': 'Elo rating system (if available)'
            }
        }


def main():
    """
    Example usage and testing
    """
    print("🎯 Testing Ensemble 1X2 v2")
    print("=" * 40)
    
    # Initialize ensemble
    ensemble = Ensemble1X2V2()
    
    # Test predictions
    test_cases = [
        {
            'name': 'All components available',
            'p_ml': [0.6, 0.2, 0.2],
            'p_poisson': [0.5, 0.3, 0.2],
            'p_elo': [0.55, 0.25, 0.2]
        },
        {
            'name': 'ML + Poisson only',
            'p_ml': [0.4, 0.3, 0.3],
            'p_poisson': [0.45, 0.25, 0.3],
            'p_elo': None
        },
        {
            'name': 'ML only',
            'p_ml': [0.35, 0.35, 0.3],
            'p_poisson': None,
            'p_elo': None
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🧪 Test {i}: {test_case['name']}")
        
        result = ensemble.predict_1x2(
            p_ml=test_case['p_ml'],
            p_poisson=test_case.get('p_poisson'),
            p_elo=test_case.get('p_elo')
        )
        
        print(f"   Prediction: {result['predicted_outcome']}")
        print(f"   Probabilities: 1={result['prob_home_win']:.3f}, X={result['prob_draw']:.3f}, 2={result['prob_away_win']:.3f}")
        print(f"   Confidence: {result['confidence']:.3f}")
        print(f"   Model: {result['model_version']}")
    
    # Show statistics
    print(f"\n📊 Ensemble Statistics:")
    stats = ensemble.get_statistics()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print("\n✅ Ensemble 1X2 v2 test completed!")


if __name__ == "__main__":
    main()
