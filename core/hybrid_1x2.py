#!/usr/bin/env python3
"""
Hybrid 1X2 Model

Combines multiple 1X2 prediction sources:
- ML 1X2 v2 output
- Scoreline-based 1X2 probabilities  
- Draw Specialist probability
- Poisson v2 1X2 probabilities
- Elo-based 1X2 probabilities

ADDITIVE - does not modify existing 1X2 models.
"""

import sys
import os
import yaml
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from scipy import stats

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.utils import setup_logging


class Hybrid1X2Combiner:
    """
    Combines multiple 1X2 prediction sources into a hybrid prediction
    
    Uses intelligent weighting based on source reliability, disagreement,
    and league-specific patterns.
    """
    
    def __init__(self, config_path: str = "config/hybrid_1x2_config.yaml"):
        """
        Initialize hybrid 1X2 combiner
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = Path(config_path)
        self.logger = setup_logging()
        
        # Load configuration
        self.config = self._load_config()
        
        # League-specific draw rates (can be learned from data)
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
        
        self.logger.info("🎯 Hybrid 1X2 Combiner initialized")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    config = yaml.safe_load(f)
                return config.get('hybrid_1x2_config', {})
            else:
                self.logger.warning(f"⚠️ Config file not found: {self.config_path}")
                return self._get_default_config()
        except Exception as e:
            self.logger.error(f"❌ Error loading config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'default_weights': {
                'ml': 0.5,
                'scoreline': 0.3,
                'poisson': 0.1,
                'elo': 0.1
            },
            'draw_weights': {
                'ml': 0.35,
                'scoreline': 0.25,
                'draw_specialist': 0.25,
                'poisson': 0.10,
                'league_prior': 0.05
            },
            'league_overrides': {},
            'uncertainty_thresholds': {
                'high_disagreement': 0.3,
                'low_confidence': 0.6
            },
            'normalization': {
                'method': 'softmax',  # 'linear' or 'softmax'
                'temperature': 1.0
            }
        }
    
    def combine(self, 
                ml_probs: Dict[str, float],
                scoreline_probs: Optional[Dict[str, float]] = None,
                draw_specialist_prob: Optional[float] = None,
                poisson_probs: Optional[Dict[str, float]] = None,
                elo_probs: Optional[Dict[str, float]] = None,
                meta_features: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Combine multiple 1X2 prediction sources
        
        Args:
            ml_probs: ML 1X2 probabilities {"1": p1, "X": px, "2": p2}
            scoreline_probs: Scoreline-derived 1X2 probabilities
            draw_specialist_prob: Draw specialist probability (0-1)
            poisson_probs: Poisson 1X2 probabilities
            elo_probs: Elo-based 1X2 probabilities
            meta_features: Additional info (league, confidence, etc.)
            
        Returns:
            Combined hybrid prediction with uncertainty metrics
        """
        try:
            self.logger.info("🎯 Combining 1X2 predictions from multiple sources")
            
            # Validate and prepare inputs
            ml_probs = self._validate_probs(ml_probs, "ML")
            scoreline_probs = self._validate_probs(scoreline_probs, "Scoreline") if scoreline_probs else None
            poisson_probs = self._validate_probs(poisson_probs, "Poisson") if poisson_probs else None
            elo_probs = self._validate_probs(elo_probs, "Elo") if elo_probs else None
            
            # Get league info
            league = meta_features.get('league', 'default') if meta_features else 'default'
            league_slug = self._get_league_slug(league)
            
            # Get weights for this league
            weights = self._get_weights(league_slug, meta_features)
            
            # Combine probabilities
            hybrid_probs = self._weighted_combination(
                ml_probs=ml_probs,
                scoreline_probs=scoreline_probs,
                draw_specialist_prob=draw_specialist_prob,
                poisson_probs=poisson_probs,
                elo_probs=elo_probs,
                weights=weights,
                league_slug=league_slug
            )
            
            # Calculate uncertainty metrics
            uncertainty = self._calculate_uncertainty(
                ml_probs, scoreline_probs, poisson_probs, elo_probs, hybrid_probs
            )
            
            # Build result
            result = {
                'hybrid_probs': hybrid_probs,
                'components': {
                    'ml': ml_probs,
                    'scoreline': scoreline_probs,
                    'draw_specialist': draw_specialist_prob,
                    'poisson': poisson_probs,
                    'elo': elo_probs
                },
                'weights_used': weights,
                'uncertainty': uncertainty,
                'league': league,
                'combination_method': 'weighted_average',
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(f"✅ Hybrid 1X2 combination completed")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Error in hybrid combination: {e}")
            # Fallback to ML probabilities
            return {
                'hybrid_probs': ml_probs,
                'components': {'ml': ml_probs},
                'weights_used': {'ml': 1.0},
                'uncertainty': {'entropy': 0.0, 'max_disagreement': 0.0},
                'fallback_reason': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _validate_probs(self, probs: Dict[str, float], source: str) -> Dict[str, float]:
        """Validate and normalize probabilities"""
        if not probs:
            return None
        
        # Ensure all keys exist
        required_keys = ['1', 'X', '2']
        for key in required_keys:
            if key not in probs:
                probs[key] = 0.0
        
        # Normalize to sum = 1
        total = sum(probs.values())
        if total > 0:
            probs = {k: v / total for k, v in probs.items()}
        else:
            # Uniform fallback
            probs = {'1': 1/3, 'X': 1/3, '2': 1/3}
        
        # Clamp to reasonable range
        for key in probs:
            probs[key] = max(0.01, min(0.98, probs[key]))
        
        # Renormalize after clamping
        total = sum(probs.values())
        probs = {k: v / total for k, v in probs.items()}
        
        return probs
    
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
    
    def _get_weights(self, league_slug: str, meta_features: Optional[Dict] = None) -> Dict[str, float]:
        """Get weights for combination based on league and meta features"""
        try:
            # Start with default weights
            weights = self.config['default_weights'].copy()
            
            # Apply league overrides if available
            league_overrides = self.config.get('league_overrides', {})
            if league_slug in league_overrides:
                weights.update(league_overrides[league_slug])
            
            # Adjust weights based on meta features
            if meta_features:
                weights = self._adjust_weights_by_meta(weights, meta_features)
            
            # Ensure weights sum to 1 (excluding draw-specific weights)
            base_weights = {k: v for k, v in weights.items() 
                          if k in ['ml', 'scoreline', 'poisson', 'elo']}
            total = sum(base_weights.values())
            if total > 0:
                for key in base_weights:
                    weights[key] = base_weights[key] / total
            
            return weights
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error getting weights: {e}")
            return self.config['default_weights']
    
    def _adjust_weights_by_meta(self, weights: Dict[str, float], 
                               meta_features: Dict[str, Any]) -> Dict[str, float]:
        """Adjust weights based on meta features"""
        adjusted = weights.copy()
        
        try:
            # Adjust based on ML confidence
            ml_confidence = meta_features.get('ml_confidence', 0.7)
            if ml_confidence < self.config['uncertainty_thresholds']['low_confidence']:
                # Low ML confidence -> reduce ML weight, boost others
                adjusted['ml'] *= 0.8
                adjusted['scoreline'] *= 1.2
                adjusted['poisson'] *= 1.1
            
            # Adjust based on scoreline confidence
            scoreline_confidence = meta_features.get('scoreline_confidence', 0.7)
            if scoreline_confidence > 0.8:
                # High scoreline confidence -> boost scoreline weight
                adjusted['scoreline'] *= 1.1
                adjusted['ml'] *= 0.95
            
            # Adjust based on match importance (if available)
            match_importance = meta_features.get('match_importance', 'normal')
            if match_importance == 'high':
                # Important match -> be more conservative, favor ML
                adjusted['ml'] *= 1.1
                adjusted['scoreline'] *= 0.9
            
            return adjusted
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error adjusting weights: {e}")
            return weights
    
    def _weighted_combination(self, 
                            ml_probs: Dict[str, float],
                            scoreline_probs: Optional[Dict[str, float]],
                            draw_specialist_prob: Optional[float],
                            poisson_probs: Optional[Dict[str, float]],
                            elo_probs: Optional[Dict[str, float]],
                            weights: Dict[str, float],
                            league_slug: str) -> Dict[str, float]:
        """Perform weighted combination of probabilities"""
        
        # Initialize combined probabilities
        combined = {'1': 0.0, 'X': 0.0, '2': 0.0}
        total_weight = 0.0
        
        # Add ML contribution
        if ml_probs and 'ml' in weights:
            w = weights['ml']
            for outcome in ['1', 'X', '2']:
                combined[outcome] += w * ml_probs[outcome]
            total_weight += w
        
        # Add Scoreline contribution
        if scoreline_probs and 'scoreline' in weights:
            w = weights['scoreline']
            for outcome in ['1', 'X', '2']:
                combined[outcome] += w * scoreline_probs[outcome]
            total_weight += w
        
        # Add Poisson contribution
        if poisson_probs and 'poisson' in weights:
            w = weights['poisson']
            for outcome in ['1', 'X', '2']:
                combined[outcome] += w * poisson_probs[outcome]
            total_weight += w
        
        # Add Elo contribution
        if elo_probs and 'elo' in weights:
            w = weights['elo']
            for outcome in ['1', 'X', '2']:
                combined[outcome] += w * elo_probs[outcome]
            total_weight += w
        
        # Special handling for draw probability
        draw_weights = self.config.get('draw_weights', {})
        if draw_specialist_prob is not None and 'draw_specialist' in draw_weights:
            # Replace part of the draw probability with specialist prediction
            specialist_weight = draw_weights['draw_specialist']
            combined['X'] = (1 - specialist_weight) * combined['X'] + specialist_weight * draw_specialist_prob
        
        # Add league prior for draw
        if 'league_prior' in draw_weights:
            league_draw_rate = self.league_draw_rates.get(league_slug, self.league_draw_rates['default'])
            prior_weight = draw_weights['league_prior']
            combined['X'] = (1 - prior_weight) * combined['X'] + prior_weight * league_draw_rate
        
        # Normalize to sum = 1
        if total_weight > 0:
            for outcome in combined:
                combined[outcome] /= total_weight
        
        # Final normalization
        total_prob = sum(combined.values())
        if total_prob > 0:
            combined = {k: v / total_prob for k, v in combined.items()}
        else:
            # Fallback to uniform
            combined = {'1': 1/3, 'X': 1/3, '2': 1/3}
        
        # Apply normalization method
        if self.config.get('normalization', {}).get('method') == 'softmax':
            combined = self._apply_softmax(combined)
        
        return combined
    
    def _apply_softmax(self, probs: Dict[str, float]) -> Dict[str, float]:
        """Apply softmax normalization"""
        try:
            temperature = self.config.get('normalization', {}).get('temperature', 1.0)
            
            # Convert to log space and apply temperature
            log_probs = {k: np.log(max(v, 1e-10)) / temperature for k, v in probs.items()}
            
            # Softmax
            max_log = max(log_probs.values())
            exp_probs = {k: np.exp(v - max_log) for k, v in log_probs.items()}
            
            # Normalize
            total = sum(exp_probs.values())
            return {k: v / total for k, v in exp_probs.items()}
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error in softmax: {e}")
            return probs
    
    def _calculate_uncertainty(self, 
                             ml_probs: Dict[str, float],
                             scoreline_probs: Optional[Dict[str, float]],
                             poisson_probs: Optional[Dict[str, float]],
                             elo_probs: Optional[Dict[str, float]],
                             hybrid_probs: Dict[str, float]) -> Dict[str, float]:
        """Calculate uncertainty metrics"""
        try:
            # Calculate entropy of final distribution
            entropy = -sum(p * np.log2(max(p, 1e-10)) for p in hybrid_probs.values())
            max_entropy = np.log2(3)  # Maximum entropy for 3 outcomes
            normalized_entropy = entropy / max_entropy
            
            # Calculate disagreement between sources
            sources = [ml_probs]
            if scoreline_probs:
                sources.append(scoreline_probs)
            if poisson_probs:
                sources.append(poisson_probs)
            if elo_probs:
                sources.append(elo_probs)
            
            max_disagreement = 0.0
            if len(sources) > 1:
                for i in range(len(sources)):
                    for j in range(i + 1, len(sources)):
                        disagreement = self._kl_divergence(sources[i], sources[j])
                        max_disagreement = max(max_disagreement, disagreement)
            
            # Calculate confidence (inverse of entropy)
            confidence = 1 - normalized_entropy
            
            # Calculate prediction strength (max probability)
            prediction_strength = max(hybrid_probs.values())
            
            return {
                'entropy': float(entropy),
                'normalized_entropy': float(normalized_entropy),
                'confidence': float(confidence),
                'max_disagreement': float(max_disagreement),
                'prediction_strength': float(prediction_strength),
                'num_sources': len(sources)
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error calculating uncertainty: {e}")
            return {
                'entropy': 0.0,
                'normalized_entropy': 0.0,
                'confidence': 0.5,
                'max_disagreement': 0.0,
                'prediction_strength': 0.33,
                'num_sources': 1
            }
    
    def _kl_divergence(self, p: Dict[str, float], q: Dict[str, float]) -> float:
        """Calculate KL divergence between two probability distributions"""
        try:
            kl = 0.0
            for outcome in ['1', 'X', '2']:
                p_val = max(p.get(outcome, 1e-10), 1e-10)
                q_val = max(q.get(outcome, 1e-10), 1e-10)
                kl += p_val * np.log(p_val / q_val)
            return kl
        except Exception as e:
            self.logger.warning(f"⚠️ Error calculating KL divergence: {e}")
            return 0.0
    
    def get_best_prediction(self, result: Dict[str, Any]) -> str:
        """Get the most likely outcome from hybrid result"""
        try:
            hybrid_probs = result['hybrid_probs']
            return max(hybrid_probs, key=hybrid_probs.get)
        except Exception as e:
            self.logger.warning(f"⚠️ Error getting best prediction: {e}")
            return 'X'  # Default to draw
    
    def get_confidence_level(self, result: Dict[str, Any]) -> str:
        """Get confidence level description"""
        try:
            confidence = result['uncertainty']['confidence']
            if confidence >= 0.8:
                return 'High'
            elif confidence >= 0.6:
                return 'Medium'
            elif confidence >= 0.4:
                return 'Low'
            else:
                return 'Very Low'
        except Exception as e:
            self.logger.warning(f"⚠️ Error getting confidence level: {e}")
            return 'Unknown'


def main():
    """
    Example usage and testing
    """
    print("🎯 Testing Hybrid 1X2 Combiner")
    print("=" * 50)
    
    # Initialize combiner
    combiner = Hybrid1X2Combiner()
    
    # Example inputs
    ml_probs = {'1': 0.45, 'X': 0.25, '2': 0.30}
    scoreline_probs = {'1': 0.42, 'X': 0.28, '2': 0.30}
    draw_specialist_prob = 0.32
    poisson_probs = {'1': 0.48, 'X': 0.22, '2': 0.30}
    elo_probs = {'1': 0.50, 'X': 0.20, '2': 0.30}
    
    meta_features = {
        'league': 'Premier League',
        'ml_confidence': 0.75,
        'scoreline_confidence': 0.80,
        'match_importance': 'normal'
    }
    
    # Test combination
    result = combiner.combine(
        ml_probs=ml_probs,
        scoreline_probs=scoreline_probs,
        draw_specialist_prob=draw_specialist_prob,
        poisson_probs=poisson_probs,
        elo_probs=elo_probs,
        meta_features=meta_features
    )
    
    print(f"\n🎯 Hybrid 1X2 Result:")
    hybrid = result['hybrid_probs']
    print(f"   Home (1): {hybrid['1']:.3f}")
    print(f"   Draw (X): {hybrid['X']:.3f}")
    print(f"   Away (2): {hybrid['2']:.3f}")
    
    print(f"\n📊 Component Contributions:")
    components = result['components']
    for source, probs in components.items():
        if probs and isinstance(probs, dict):
            print(f"   {source.upper()}: 1={probs['1']:.3f}, X={probs['X']:.3f}, 2={probs['2']:.3f}")
        elif probs is not None:
            print(f"   {source.upper()}: {probs:.3f}")
    
    print(f"\n🔧 Weights Used:")
    weights = result['weights_used']
    for source, weight in weights.items():
        print(f"   {source}: {weight:.3f}")
    
    print(f"\n📈 Uncertainty Metrics:")
    uncertainty = result['uncertainty']
    for metric, value in uncertainty.items():
        print(f"   {metric}: {value:.3f}")
    
    print(f"\n🎯 Best Prediction: {combiner.get_best_prediction(result)}")
    print(f"🎯 Confidence Level: {combiner.get_confidence_level(result)}")
    
    print("\n✅ Hybrid 1X2 Combiner test completed!")


if __name__ == "__main__":
    main()
