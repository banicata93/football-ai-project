#!/usr/bin/env python3
"""
Hybrid Integration - Utility + glue layer for merging all model outputs

Provides utilities for:
- Normalizing probabilities
- Applying hybrid configuration weights
- Handling fallback logic per league
- Combining multiple prediction sources

ADDITIVE - does not modify existing model logic.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

from core.utils import setup_logging


class HybridIntegrator:
    """
    Utility and glue layer for merging multiple model outputs
    
    Handles probability normalization, weight application,
    and fallback logic for hybrid predictions.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize hybrid integrator
        
        Args:
            config: Hybrid configuration dictionary
        """
        self.config = config
        self.logger = setup_logging()
        
        # League-specific draw rates for fallback
        self.league_draw_rates = {
            'premier_league': 0.26,
            'la_liga': 0.28,
            'serie_a': 0.29,
            'bundesliga': 0.27,
            'ligue_1': 0.28,
            'eredivisie': 0.30,
            'primeira_liga': 0.29,
            'championship': 0.25,
            'default': 0.27
        }
        
        self.logger.info("🎯 Hybrid Integrator initialized")
    
    def combine_probabilities(self,
                            ml_probs: Optional[Dict[str, float]] = None,
                            scoreline_probs: Optional[Dict[str, float]] = None,
                            poisson_probs: Optional[Dict[str, float]] = None,
                            draw_specialist_prob: Optional[float] = None,
                            league: str = 'default',
                            weights: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Combine probabilities from multiple sources
        
        Args:
            ml_probs: ML 1X2 probabilities {"1": p1, "X": px, "2": p2}
            scoreline_probs: Scoreline-derived 1X2 probabilities
            poisson_probs: Poisson 1X2 probabilities
            draw_specialist_prob: Draw specialist probability (0-1)
            league: League name for league-specific weights
            weights: Custom weights (overrides config)
            
        Returns:
            Combined and normalized probabilities
        """
        try:
            self.logger.info("🔧 Combining probabilities from multiple sources")
            
            # Get effective weights
            effective_weights = self.get_effective_weights(league, weights)
            
            # Initialize combined probabilities
            combined = {'1': 0.0, 'X': 0.0, '2': 0.0}
            total_weight = 0.0
            
            # Add ML contribution
            if ml_probs and effective_weights.get('ml', 0) > 0:
                w = effective_weights['ml']
                ml_probs = self.normalize(ml_probs)
                for outcome in ['1', 'X', '2']:
                    combined[outcome] += w * ml_probs.get(outcome, 1/3)
                total_weight += w
                self.logger.debug(f"Added ML contribution with weight {w}")
            
            # Add Scoreline contribution
            if scoreline_probs and effective_weights.get('scoreline', 0) > 0:
                w = effective_weights['scoreline']
                scoreline_probs = self.normalize(scoreline_probs)
                for outcome in ['1', 'X', '2']:
                    combined[outcome] += w * scoreline_probs.get(outcome, 1/3)
                total_weight += w
                self.logger.debug(f"Added Scoreline contribution with weight {w}")
            
            # Add Poisson contribution
            if poisson_probs and effective_weights.get('poisson', 0) > 0:
                w = effective_weights['poisson']
                poisson_probs = self.normalize(poisson_probs)
                for outcome in ['1', 'X', '2']:
                    combined[outcome] += w * poisson_probs.get(outcome, 1/3)
                total_weight += w
                self.logger.debug(f"Added Poisson contribution with weight {w}")
            
            # Special handling for draw specialist
            if draw_specialist_prob is not None and effective_weights.get('draw_specialist', 0) > 0:
                w = effective_weights['draw_specialist']
                # Replace part of the draw probability with specialist prediction
                original_draw = combined['X']
                combined['X'] = (1 - w) * original_draw + w * draw_specialist_prob
                self.logger.debug(f"Enhanced draw probability with specialist weight {w}")
            
            # Normalize by total weight
            if total_weight > 0:
                for outcome in combined:
                    combined[outcome] /= total_weight
            else:
                # Fallback to uniform if no sources available
                combined = {'1': 1/3, 'X': 1/3, '2': 1/3}
                self.logger.warning("⚠️ No sources available, using uniform distribution")
            
            # Final normalization to ensure sum = 1
            combined = self.normalize(combined)
            
            self.logger.info(f"✅ Probabilities combined: H={combined['1']:.3f}, X={combined['X']:.3f}, A={combined['2']:.3f}")
            return combined
            
        except Exception as e:
            self.logger.error(f"❌ Error combining probabilities: {e}")
            return self._fallback_probabilities(league)
    
    def normalize(self, probs: Dict[str, float]) -> Dict[str, float]:
        """
        Normalize probabilities to sum to 1.0
        
        Args:
            probs: Probability dictionary
            
        Returns:
            Normalized probabilities
        """
        try:
            if not probs:
                return {'1': 1/3, 'X': 1/3, '2': 1/3}
            
            # Ensure all required keys exist
            for key in ['1', 'X', '2']:
                if key not in probs:
                    probs[key] = 0.0
            
            # Calculate total
            total = sum(probs.values())
            
            if total <= 0:
                # Fallback to uniform
                return {'1': 1/3, 'X': 1/3, '2': 1/3}
            
            # Normalize
            normalized = {k: v / total for k, v in probs.items()}
            
            # Clamp to reasonable bounds
            for key in normalized:
                normalized[key] = max(0.001, min(0.998, normalized[key]))
            
            # Re-normalize after clamping
            total = sum(normalized.values())
            normalized = {k: v / total for k, v in normalized.items()}
            
            return normalized
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error normalizing probabilities: {e}")
            return {'1': 1/3, 'X': 1/3, '2': 1/3}
    
    def apply_calibration(self, probs: Dict[str, float], 
                         calibrator: Any = None,
                         method: str = 'temperature') -> Dict[str, float]:
        """
        Apply calibration to probabilities
        
        Args:
            probs: Input probabilities
            calibrator: Calibration model (optional)
            method: Calibration method
            
        Returns:
            Calibrated probabilities
        """
        try:
            if calibrator is None:
                return probs
            
            # Convert to array format
            prob_array = np.array([probs.get('1', 1/3), probs.get('X', 1/3), probs.get('2', 1/3)])
            
            if method == 'temperature':
                calibrated = self._apply_temperature_scaling(prob_array, calibrator)
            elif hasattr(calibrator, 'predict_proba'):
                calibrated = calibrator.predict_proba(prob_array.reshape(1, -1))[0]
            elif hasattr(calibrator, 'predict'):
                calibrated = calibrator.predict(prob_array.reshape(1, -1))[0]
            else:
                calibrated = prob_array
            
            return {
                '1': float(calibrated[0]),
                'X': float(calibrated[1]),
                '2': float(calibrated[2])
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error applying calibration: {e}")
            return probs
    
    def _apply_temperature_scaling(self, probs: np.ndarray, temperature: float) -> np.ndarray:
        """Apply temperature scaling calibration"""
        try:
            # Convert to log space
            log_probs = np.log(np.maximum(probs, 1e-10))
            
            # Apply temperature
            scaled_logits = log_probs / temperature
            
            # Softmax
            exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
            return exp_logits / np.sum(exp_logits)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error in temperature scaling: {e}")
            return probs
    
    def load_weights(self, weights_file: str) -> Optional[Dict[str, Any]]:
        """
        Load weights from file
        
        Args:
            weights_file: Path to weights file
            
        Returns:
            Loaded weights or None if failed
        """
        try:
            import json
            from pathlib import Path
            
            weights_path = Path(weights_file)
            if weights_path.exists():
                with open(weights_path, 'r') as f:
                    weights = json.load(f)
                self.logger.info(f"✅ Weights loaded from {weights_file}")
                return weights
            else:
                self.logger.warning(f"⚠️ Weights file not found: {weights_file}")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Error loading weights: {e}")
            return None
    
    def get_effective_weights(self, league: str, custom_weights: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Get effective weights for combination
        
        Args:
            league: League name
            custom_weights: Custom weights to override config
            
        Returns:
            Effective weights dictionary
        """
        try:
            # Start with config weights
            if custom_weights:
                base_weights = custom_weights.get('weights', self.config.get('weights', {}))
            else:
                base_weights = self.config.get('weights', {})
            
            # Apply league-specific overrides if enabled
            if self.config.get('leagues', {}).get('use_per_league_weights', False):
                league_slug = self._get_league_slug(league)
                league_overrides = self.config.get('league_overrides', {}).get(league_slug, {})
                base_weights.update(league_overrides)
            
            # Ensure all required weights exist
            default_weights = {
                'ml': 0.45,
                'scoreline': 0.25,
                'poisson': 0.20,
                'draw_specialist': 0.10
            }
            
            effective_weights = default_weights.copy()
            effective_weights.update(base_weights)
            
            # Normalize weights (excluding draw_specialist which is handled separately)
            base_sources = ['ml', 'scoreline', 'poisson']
            base_total = sum(effective_weights.get(source, 0) for source in base_sources)
            
            if base_total > 0:
                for source in base_sources:
                    effective_weights[source] = effective_weights.get(source, 0) / base_total
            
            return effective_weights
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error getting effective weights: {e}")
            return {'ml': 0.45, 'scoreline': 0.25, 'poisson': 0.20, 'draw_specialist': 0.10}
    
    def _get_league_slug(self, league: str) -> str:
        """Convert league name to slug"""
        if not league:
            return 'default'
        
        league_mapping = {
            'premier league': 'premier_league',
            'la liga': 'la_liga',
            'serie a': 'serie_a',
            'bundesliga': 'bundesliga',
            'ligue 1': 'ligue_1',
            'eredivisie': 'eredivisie',
            'primeira liga': 'primeira_liga',
            'championship': 'championship'
        }
        
        return league_mapping.get(league.lower(), 'default')
    
    def _fallback_probabilities(self, league: str) -> Dict[str, float]:
        """Get fallback probabilities based on league"""
        try:
            league_slug = self._get_league_slug(league)
            draw_rate = self.league_draw_rates.get(league_slug, self.league_draw_rates['default'])
            
            # Simple heuristic: equal home/away, league-specific draw
            home_away_rate = (1 - draw_rate) / 2
            
            return {
                '1': home_away_rate,
                'X': draw_rate,
                '2': home_away_rate
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error getting fallback probabilities: {e}")
            return {'1': 1/3, 'X': 1/3, '2': 1/3}
    
    def hybrid_summary(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate summary of hybrid prediction
        
        Args:
            result: Hybrid prediction result
            
        Returns:
            Summary dictionary
        """
        try:
            sources_used = result.get('sources_used', {})
            components = result.get('components', {})
            weights_used = result.get('weights_used', {})
            
            summary = {
                'total_sources': sum(1 for used in sources_used.values() if used),
                'sources_available': list(source for source, used in sources_used.items() if used),
                'primary_source': max(weights_used, key=weights_used.get) if weights_used else 'unknown',
                'confidence_level': self._get_confidence_level(result.get('confidence', 0.33)),
                'prediction_strength': result.get('confidence', 0.33),
                'is_balanced': self._is_balanced_prediction(result),
                'dominant_outcome': result.get('predicted_outcome', 'X'),
                'hybrid_advantage': self._calculate_hybrid_advantage(components, weights_used)
            }
            
            return summary
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error generating hybrid summary: {e}")
            return {'error': str(e)}
    
    def _get_confidence_level(self, confidence: float) -> str:
        """Get confidence level description"""
        if confidence >= 0.8:
            return 'Very High'
        elif confidence >= 0.6:
            return 'High'
        elif confidence >= 0.5:
            return 'Medium'
        elif confidence >= 0.4:
            return 'Low'
        else:
            return 'Very Low'
    
    def _is_balanced_prediction(self, result: Dict[str, Any]) -> bool:
        """Check if prediction is balanced (no dominant outcome)"""
        try:
            probs = [
                result.get('prob_home_win', 1/3),
                result.get('prob_draw', 1/3),
                result.get('prob_away_win', 1/3)
            ]
            
            max_prob = max(probs)
            return max_prob < 0.5  # No outcome has >50% probability
            
        except Exception as e:
            return False
    
    def _calculate_hybrid_advantage(self, components: Dict[str, Any], 
                                  weights: Dict[str, float]) -> Dict[str, Any]:
        """Calculate advantage of using hybrid vs single source"""
        try:
            if not components or not weights:
                return {'advantage': 0.0, 'reason': 'insufficient_data'}
            
            # Count available sources
            available_sources = sum(1 for comp in components.values() if comp is not None)
            
            if available_sources <= 1:
                return {'advantage': 0.0, 'reason': 'single_source_only'}
            
            # Calculate disagreement between sources
            disagreement = self._calculate_source_disagreement(components)
            
            # Higher disagreement = higher potential advantage from ensemble
            advantage_score = min(disagreement * 2, 1.0)  # Cap at 1.0
            
            return {
                'advantage': advantage_score,
                'reason': 'ensemble_benefit',
                'disagreement': disagreement,
                'sources_count': available_sources
            }
            
        except Exception as e:
            return {'advantage': 0.0, 'reason': 'calculation_error'}
    
    def _calculate_source_disagreement(self, components: Dict[str, Any]) -> float:
        """Calculate disagreement between prediction sources"""
        try:
            valid_sources = []
            
            for source, probs in components.items():
                if probs and isinstance(probs, dict) and all(k in probs for k in ['1', 'X', '2']):
                    valid_sources.append([probs['1'], probs['X'], probs['2']])
            
            if len(valid_sources) < 2:
                return 0.0
            
            # Calculate pairwise KL divergences
            disagreements = []
            for i in range(len(valid_sources)):
                for j in range(i + 1, len(valid_sources)):
                    kl_div = self._kl_divergence(valid_sources[i], valid_sources[j])
                    disagreements.append(kl_div)
            
            return np.mean(disagreements) if disagreements else 0.0
            
        except Exception as e:
            return 0.0
    
    def _kl_divergence(self, p: List[float], q: List[float]) -> float:
        """Calculate KL divergence between two probability distributions"""
        try:
            kl = 0.0
            for i in range(len(p)):
                p_val = max(p[i], 1e-10)
                q_val = max(q[i], 1e-10)
                kl += p_val * np.log(p_val / q_val)
            return kl
        except Exception as e:
            return 0.0


def main():
    """Test hybrid integrator"""
    print("🎯 Testing Hybrid Integrator")
    print("=" * 50)
    
    # Test configuration
    config = {
        'weights': {
            'ml': 0.45,
            'scoreline': 0.25,
            'poisson': 0.20,
            'draw_specialist': 0.10
        },
        'leagues': {
            'use_per_league_weights': True
        }
    }
    
    # Initialize integrator
    integrator = HybridIntegrator(config)
    
    # Test data
    ml_probs = {'1': 0.45, 'X': 0.25, '2': 0.30}
    scoreline_probs = {'1': 0.42, 'X': 0.28, '2': 0.30}
    poisson_probs = {'1': 0.48, 'X': 0.22, '2': 0.30}
    draw_specialist_prob = 0.32
    
    # Test combination
    combined = integrator.combine_probabilities(
        ml_probs=ml_probs,
        scoreline_probs=scoreline_probs,
        poisson_probs=poisson_probs,
        draw_specialist_prob=draw_specialist_prob,
        league='Premier League'
    )
    
    print(f"\n🎯 Combined Probabilities:")
    print(f"   Home: {combined['1']:.3f}")
    print(f"   Draw: {combined['X']:.3f}")
    print(f"   Away: {combined['2']:.3f}")
    print(f"   Sum: {sum(combined.values()):.3f}")
    
    # Test normalization
    test_probs = {'1': 0.6, 'X': 0.3, '2': 0.2}  # Sum > 1
    normalized = integrator.normalize(test_probs)
    
    print(f"\n🔧 Normalization Test:")
    print(f"   Original: {test_probs}")
    print(f"   Normalized: {normalized}")
    print(f"   Sum: {sum(normalized.values()):.3f}")
    
    # Test weights
    weights = integrator.get_effective_weights('Premier League')
    print(f"\n⚖️ Effective Weights:")
    for source, weight in weights.items():
        print(f"   {source}: {weight:.3f}")
    
    print("\n✅ Hybrid integrator test completed!")


if __name__ == "__main__":
    main()
