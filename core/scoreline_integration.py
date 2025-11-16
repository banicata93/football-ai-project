#!/usr/bin/env python3
"""
Scoreline Integration Module

Integrates Scoreline Probability Engine outputs into all prediction types:
- 1X2 predictions from scoreline matrix aggregation
- BTTS predictions from scoreline matrix analysis
- OU2.5 predictions from goal total distributions

ADDITIVE - does not modify existing models, only enhances predictions.
"""

import sys
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from scipy import stats
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.utils import setup_logging
from core.scoreline_engine import ScorelineProbabilityEngine


class ScorelineIntegrator:
    """
    Integrates scoreline probabilities into all prediction types
    
    Provides enhanced predictions by combining ML models with scoreline matrix analysis.
    """
    
    def __init__(self):
        """Initialize scoreline integrator"""
        self.logger = setup_logging()
        self.scoreline_engine = None
        
        # Integration parameters
        self.config = {
            'confidence_threshold': 0.5,
            'disagreement_threshold': 0.25,
            'variance_threshold': 0.1,
            'smoothing_factor': 0.05,
            'min_weight': 0.1,
            'max_weight': 0.7
        }
        
        self._initialize_engine()
        self.logger.info("🎯 Scoreline Integrator initialized")
    
    def _initialize_engine(self):
        """Initialize scoreline engine (lazy loading)"""
        try:
            self.scoreline_engine = ScorelineProbabilityEngine()
            self.logger.info("✅ Scoreline engine loaded for integration")
        except Exception as e:
            self.logger.warning(f"⚠️ Scoreline engine not available: {e}")
            self.scoreline_engine = None
    
    def derive_1x2_from_matrix(self, matrix: Dict[str, float]) -> Dict[str, Any]:
        """
        Derive 1X2 probabilities from scoreline matrix
        
        Args:
            matrix: Scoreline probability matrix {"0-0": prob, "1-0": prob, ...}
            
        Returns:
            1X2 probabilities with confidence metrics
        """
        try:
            p_home = 0.0
            p_draw = 0.0
            p_away = 0.0
            
            # Aggregate probabilities by result type
            for scoreline, prob in matrix.items():
                home_goals, away_goals = map(int, scoreline.split('-'))
                
                if home_goals > away_goals:
                    p_home += prob
                elif home_goals == away_goals:
                    p_draw += prob
                else:
                    p_away += prob
            
            # Normalize to ensure sum = 1
            total = p_home + p_draw + p_away
            if total > 0:
                p_home /= total
                p_draw /= total
                p_away /= total
            else:
                # Fallback to uniform distribution
                p_home = p_draw = p_away = 1/3
            
            # Calculate confidence metrics
            confidence = self._calculate_1x2_confidence(matrix)
            margin_variance = self._calculate_margin_variance(matrix)
            
            return {
                'p_home': p_home,
                'p_draw': p_draw,
                'p_away': p_away,
                'confidence': confidence,
                'margin_variance': margin_variance,
                'source': 'scoreline_matrix'
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error deriving 1X2 from matrix: {e}")
            return {
                'p_home': 1/3, 'p_draw': 1/3, 'p_away': 1/3,
                'confidence': 0.0, 'margin_variance': 0.0,
                'source': 'fallback'
            }
    
    def derive_btts_from_matrix(self, matrix: Dict[str, float]) -> Dict[str, Any]:
        """
        Derive BTTS probabilities from scoreline matrix
        
        Args:
            matrix: Scoreline probability matrix
            
        Returns:
            BTTS probabilities with confidence metrics
        """
        try:
            p_no_btts = 0.0  # At least one team scores 0
            p_yes_btts = 0.0  # Both teams score at least 1
            
            for scoreline, prob in matrix.items():
                home_goals, away_goals = map(int, scoreline.split('-'))
                
                if home_goals == 0 or away_goals == 0:
                    p_no_btts += prob
                else:
                    p_yes_btts += prob
            
            # Normalize
            total = p_no_btts + p_yes_btts
            if total > 0:
                p_no_btts /= total
                p_yes_btts /= total
            else:
                p_no_btts = p_yes_btts = 0.5
            
            # Calculate confidence
            entropy = -p_yes_btts * np.log2(p_yes_btts + 1e-10) - p_no_btts * np.log2(p_no_btts + 1e-10)
            confidence = 1 - entropy  # Lower entropy = higher confidence
            
            return {
                'p_yes': p_yes_btts,
                'p_no': p_no_btts,
                'confidence': confidence,
                'source': 'scoreline_matrix'
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error deriving BTTS from matrix: {e}")
            return {
                'p_yes': 0.5, 'p_no': 0.5,
                'confidence': 0.0, 'source': 'fallback'
            }
    
    def derive_ou25_from_matrix(self, matrix: Dict[str, float]) -> Dict[str, Any]:
        """
        Derive Over/Under 2.5 probabilities from scoreline matrix
        
        Args:
            matrix: Scoreline probability matrix
            
        Returns:
            OU2.5 probabilities with confidence metrics
        """
        try:
            p_under = 0.0  # Total goals <= 2
            p_over = 0.0   # Total goals >= 3
            
            goal_distribution = []  # For variance calculation
            
            for scoreline, prob in matrix.items():
                home_goals, away_goals = map(int, scoreline.split('-'))
                total_goals = home_goals + away_goals
                
                # Collect for distribution analysis
                goal_distribution.extend([total_goals] * int(prob * 1000))
                
                if total_goals <= 2:
                    p_under += prob
                else:
                    p_over += prob
            
            # Normalize
            total = p_under + p_over
            if total > 0:
                p_under /= total
                p_over /= total
            else:
                p_under = p_over = 0.5
            
            # Calculate goal variance for tempo adjustment
            goal_variance = np.var(goal_distribution) if goal_distribution else 0.0
            
            # Apply tempo-based stretching if high variance
            if goal_variance > self.config['variance_threshold']:
                stretch_factor = min(1.2, 1 + goal_variance * 0.5)
                # Stretch probabilities away from 0.5
                p_over = 0.5 + (p_over - 0.5) * stretch_factor
                p_under = 1 - p_over
            
            # Calculate confidence
            entropy = -p_over * np.log2(p_over + 1e-10) - p_under * np.log2(p_under + 1e-10)
            confidence = 1 - entropy
            
            return {
                'p_over': p_over,
                'p_under': p_under,
                'confidence': confidence,
                'goal_variance': goal_variance,
                'source': 'scoreline_matrix'
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error deriving OU2.5 from matrix: {e}")
            return {
                'p_over': 0.5, 'p_under': 0.5,
                'confidence': 0.0, 'goal_variance': 0.0,
                'source': 'fallback'
            }
    
    def _calculate_1x2_confidence(self, matrix: Dict[str, float]) -> float:
        """Calculate confidence for 1X2 predictions"""
        try:
            # Calculate entropy of the 1X2 distribution
            probs = []
            for result_type in ['home', 'draw', 'away']:
                prob = 0.0
                for scoreline, p in matrix.items():
                    home_goals, away_goals = map(int, scoreline.split('-'))
                    if result_type == 'home' and home_goals > away_goals:
                        prob += p
                    elif result_type == 'draw' and home_goals == away_goals:
                        prob += p
                    elif result_type == 'away' and home_goals < away_goals:
                        prob += p
                probs.append(prob)
            
            # Normalize
            total = sum(probs)
            if total > 0:
                probs = [p / total for p in probs]
            
            # Calculate entropy
            entropy = -sum(p * np.log2(p + 1e-10) for p in probs)
            max_entropy = np.log2(3)  # Maximum entropy for 3 outcomes
            
            # Confidence = 1 - normalized_entropy
            confidence = 1 - (entropy / max_entropy)
            return max(0.0, min(1.0, confidence))
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error calculating 1X2 confidence: {e}")
            return 0.5
    
    def _calculate_margin_variance(self, matrix: Dict[str, float]) -> float:
        """Calculate variance of goal margin distribution"""
        try:
            margins = []
            for scoreline, prob in matrix.items():
                home_goals, away_goals = map(int, scoreline.split('-'))
                margin = home_goals - away_goals
                margins.extend([margin] * int(prob * 1000))
            
            return np.var(margins) if margins else 0.0
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error calculating margin variance: {e}")
            return 0.0
    
    def combine_predictions(self, ml_prediction: Dict[str, Any], 
                          scoreline_prediction: Dict[str, Any],
                          prediction_type: str) -> Dict[str, Any]:
        """
        Combine ML and scoreline predictions intelligently
        
        Args:
            ml_prediction: Original ML model prediction
            scoreline_prediction: Scoreline-derived prediction
            prediction_type: '1x2', 'btts', or 'ou25'
            
        Returns:
            Combined prediction with metadata
        """
        try:
            # Calculate disagreement
            disagreement = self._calculate_disagreement(
                ml_prediction, scoreline_prediction, prediction_type
            )
            
            # Determine weights based on confidence and disagreement
            ml_confidence = ml_prediction.get('confidence', 0.5)
            scoreline_confidence = scoreline_prediction.get('confidence', 0.5)
            
            # Base weights
            if ml_confidence < self.config['confidence_threshold']:
                scoreline_weight = 0.6  # Increase scoreline weight if ML uncertain
            else:
                scoreline_weight = 0.4  # Default scoreline weight
            
            # Adjust for disagreement
            if disagreement > self.config['disagreement_threshold']:
                # High disagreement -> shrink to neutral, favor more confident model
                if scoreline_confidence > ml_confidence:
                    scoreline_weight = min(0.6, scoreline_weight + 0.1)
                else:
                    scoreline_weight = max(0.3, scoreline_weight - 0.1)
            
            # Adjust for scoreline variance (low variance = more reliable)
            if prediction_type == '1x2':
                margin_var = scoreline_prediction.get('margin_variance', 0.1)
                if margin_var < self.config['variance_threshold']:
                    scoreline_weight += 0.1  # Boost scoreline if low variance
            
            # Clamp weights
            scoreline_weight = max(self.config['min_weight'], 
                                 min(self.config['max_weight'], scoreline_weight))
            ml_weight = 1 - scoreline_weight
            
            # Combine predictions
            combined = self._weighted_combine(
                ml_prediction, scoreline_prediction, 
                ml_weight, scoreline_weight, prediction_type
            )
            
            # Add metadata
            combined.update({
                'combination_method': 'intelligent_weighting',
                'ml_weight': ml_weight,
                'scoreline_weight': scoreline_weight,
                'disagreement': disagreement,
                'ml_confidence': ml_confidence,
                'scoreline_confidence': scoreline_confidence
            })
            
            return combined
            
        except Exception as e:
            self.logger.error(f"❌ Error combining predictions: {e}")
            return ml_prediction  # Fallback to ML prediction
    
    def _calculate_disagreement(self, ml_pred: Dict, scoreline_pred: Dict, 
                              pred_type: str) -> float:
        """Calculate disagreement between ML and scoreline predictions"""
        try:
            if pred_type == '1x2':
                ml_probs = [ml_pred.get('p_home', 1/3), ml_pred.get('p_draw', 1/3), ml_pred.get('p_away', 1/3)]
                sc_probs = [scoreline_pred.get('p_home', 1/3), scoreline_pred.get('p_draw', 1/3), scoreline_pred.get('p_away', 1/3)]
            elif pred_type == 'btts':
                ml_probs = [ml_pred.get('p_yes', 0.5), ml_pred.get('p_no', 0.5)]
                sc_probs = [scoreline_pred.get('p_yes', 0.5), scoreline_pred.get('p_no', 0.5)]
            elif pred_type == 'ou25':
                ml_probs = [ml_pred.get('p_over', 0.5), ml_pred.get('p_under', 0.5)]
                sc_probs = [scoreline_pred.get('p_over', 0.5), scoreline_pred.get('p_under', 0.5)]
            else:
                return 0.0
            
            # Calculate KL divergence as disagreement measure
            kl_div = sum(p * np.log((p + 1e-10) / (q + 1e-10)) 
                        for p, q in zip(ml_probs, sc_probs))
            
            # Normalize to [0, 1]
            return min(1.0, kl_div / 2.0)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error calculating disagreement: {e}")
            return 0.0
    
    def _weighted_combine(self, ml_pred: Dict, sc_pred: Dict, 
                         ml_weight: float, sc_weight: float, pred_type: str) -> Dict:
        """Combine predictions using weighted average"""
        try:
            combined = {}
            
            if pred_type == '1x2':
                combined['p_home'] = (ml_weight * ml_pred.get('p_home', 1/3) + 
                                    sc_weight * sc_pred.get('p_home', 1/3))
                combined['p_draw'] = (ml_weight * ml_pred.get('p_draw', 1/3) + 
                                    sc_weight * sc_pred.get('p_draw', 1/3))
                combined['p_away'] = (ml_weight * ml_pred.get('p_away', 1/3) + 
                                    sc_weight * sc_pred.get('p_away', 1/3))
                
                # Normalize
                total = combined['p_home'] + combined['p_draw'] + combined['p_away']
                if total > 0:
                    combined['p_home'] /= total
                    combined['p_draw'] /= total
                    combined['p_away'] /= total
                    
            elif pred_type == 'btts':
                combined['p_yes'] = (ml_weight * ml_pred.get('p_yes', 0.5) + 
                                   sc_weight * sc_pred.get('p_yes', 0.5))
                combined['p_no'] = 1 - combined['p_yes']
                
            elif pred_type == 'ou25':
                combined['p_over'] = (ml_weight * ml_pred.get('p_over', 0.5) + 
                                    sc_weight * sc_pred.get('p_over', 0.5))
                combined['p_under'] = 1 - combined['p_over']
            
            # Combine confidence
            combined['confidence'] = (ml_weight * ml_pred.get('confidence', 0.5) + 
                                    sc_weight * sc_pred.get('confidence', 0.5))
            
            return combined
            
        except Exception as e:
            self.logger.error(f"❌ Error in weighted combine: {e}")
            return ml_pred
    
    def get_enhanced_predictions(self, home_team: str, away_team: str, 
                               league: str, df: pd.DataFrame,
                               ml_predictions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get enhanced predictions combining ML and scoreline analysis
        
        Args:
            home_team: Home team name
            away_team: Away team name
            league: League name
            df: Historical data
            ml_predictions: Original ML predictions
            
        Returns:
            Enhanced predictions with scoreline integration
        """
        try:
            if self.scoreline_engine is None:
                self.logger.warning("⚠️ Scoreline engine not available, using ML only")
                return ml_predictions
            
            # Get scoreline matrix
            scoreline_result = self.scoreline_engine.get_scoreline_probabilities(
                home_team, away_team, league, df
            )
            
            matrix = scoreline_result.get('matrix', {})
            if not matrix:
                self.logger.warning("⚠️ Empty scoreline matrix, using ML only")
                return ml_predictions
            
            # Derive predictions from scoreline matrix
            scoreline_1x2 = self.derive_1x2_from_matrix(matrix)
            scoreline_btts = self.derive_btts_from_matrix(matrix)
            scoreline_ou25 = self.derive_ou25_from_matrix(matrix)
            
            # Combine with ML predictions
            enhanced_predictions = ml_predictions.copy()
            
            # Enhance 1X2
            if '1x2' in ml_predictions:
                enhanced_predictions['1x2_enhanced'] = self.combine_predictions(
                    ml_predictions['1x2'], scoreline_1x2, '1x2'
                )
                enhanced_predictions['1x2_scoreline'] = scoreline_1x2
            
            # Enhance BTTS
            if 'btts' in ml_predictions:
                enhanced_predictions['btts_enhanced'] = self.combine_predictions(
                    ml_predictions['btts'], scoreline_btts, 'btts'
                )
                enhanced_predictions['btts_scoreline'] = scoreline_btts
            
            # Enhance OU2.5
            if 'ou25' in ml_predictions:
                enhanced_predictions['ou25_enhanced'] = self.combine_predictions(
                    ml_predictions['ou25'], scoreline_ou25, 'ou25'
                )
                enhanced_predictions['ou25_scoreline'] = scoreline_ou25
            
            # Add scoreline metadata
            enhanced_predictions['scoreline_matrix'] = matrix
            enhanced_predictions['scoreline_confidence'] = scoreline_result.get('summary', {}).get('confidence', 0.5)
            enhanced_predictions['scoreline_engine_version'] = scoreline_result.get('engine_version', 'unknown')
            
            self.logger.info(f"✅ Enhanced predictions for {home_team} vs {away_team}")
            return enhanced_predictions
            
        except Exception as e:
            self.logger.error(f"❌ Error getting enhanced predictions: {e}")
            return ml_predictions


def main():
    """
    Example usage and testing
    """
    print("🎯 Testing Scoreline Integration")
    print("=" * 50)
    
    # Initialize integrator
    integrator = ScorelineIntegrator()
    
    # Example scoreline matrix
    example_matrix = {
        "0-0": 0.089, "1-0": 0.134, "0-1": 0.098,
        "1-1": 0.156, "2-0": 0.087, "0-2": 0.067,
        "2-1": 0.112, "1-2": 0.089, "2-2": 0.078,
        "3-0": 0.045, "0-3": 0.034, "3-1": 0.067,
        "1-3": 0.045, "3-2": 0.056, "2-3": 0.043,
        "4-0": 0.023, "0-4": 0.015, "4-1": 0.034,
        "1-4": 0.023, "4-2": 0.028, "2-4": 0.019,
        "3-3": 0.034, "4-3": 0.018, "3-4": 0.014,
        "4-4": 0.009
    }
    
    # Test 1X2 derivation
    print("\n🎯 Testing 1X2 Derivation:")
    result_1x2 = integrator.derive_1x2_from_matrix(example_matrix)
    print(f"   Home: {result_1x2['p_home']:.3f}")
    print(f"   Draw: {result_1x2['p_draw']:.3f}")
    print(f"   Away: {result_1x2['p_away']:.3f}")
    print(f"   Confidence: {result_1x2['confidence']:.3f}")
    
    # Test BTTS derivation
    print("\n🎯 Testing BTTS Derivation:")
    result_btts = integrator.derive_btts_from_matrix(example_matrix)
    print(f"   BTTS Yes: {result_btts['p_yes']:.3f}")
    print(f"   BTTS No: {result_btts['p_no']:.3f}")
    print(f"   Confidence: {result_btts['confidence']:.3f}")
    
    # Test OU2.5 derivation
    print("\n🎯 Testing OU2.5 Derivation:")
    result_ou25 = integrator.derive_ou25_from_matrix(example_matrix)
    print(f"   Over 2.5: {result_ou25['p_over']:.3f}")
    print(f"   Under 2.5: {result_ou25['p_under']:.3f}")
    print(f"   Confidence: {result_ou25['confidence']:.3f}")
    print(f"   Goal Variance: {result_ou25['goal_variance']:.3f}")
    
    # Test combination
    print("\n🎯 Testing Prediction Combination:")
    ml_1x2 = {'p_home': 0.45, 'p_draw': 0.25, 'p_away': 0.30, 'confidence': 0.7}
    combined = integrator.combine_predictions(ml_1x2, result_1x2, '1x2')
    print(f"   ML: H={ml_1x2['p_home']:.3f}, D={ml_1x2['p_draw']:.3f}, A={ml_1x2['p_away']:.3f}")
    print(f"   Scoreline: H={result_1x2['p_home']:.3f}, D={result_1x2['p_draw']:.3f}, A={result_1x2['p_away']:.3f}")
    print(f"   Combined: H={combined['p_home']:.3f}, D={combined['p_draw']:.3f}, A={combined['p_away']:.3f}")
    print(f"   Weights: ML={combined['ml_weight']:.2f}, Scoreline={combined['scoreline_weight']:.2f}")
    
    print("\n✅ Scoreline Integration test completed!")


if __name__ == "__main__":
    main()
