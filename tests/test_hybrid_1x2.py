#!/usr/bin/env python3
"""
Test Hybrid 1X2 System

Verifies hybrid prediction correctness, probability sum ≈ 1.0,
hybrid engine loads correctly, and validates output format.
"""

import sys
import os
import unittest
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.hybrid_1x2_predictor import Hybrid1X2Predictor
from core.hybrid_integration import HybridIntegrator


class TestHybrid1X2(unittest.TestCase):
    """Test cases for Hybrid 1X2 system"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.predictor = Hybrid1X2Predictor()
        
        # Test configuration
        self.test_config = {
            'weights': {
                'ml': 0.45,
                'scoreline': 0.25,
                'poisson': 0.20,
                'draw_specialist': 0.10
            },
            'calibration': {
                'enabled': True,
                'method': 'temperature'
            }
        }
        
        self.integrator = HybridIntegrator(self.test_config)
        
        # Test data
        self.test_features = pd.DataFrame({
            'home_elo': [1500],
            'away_elo': [1450],
            'home_form': [0.6],
            'away_form': [0.5]
        })
        
        self.test_context = {
            'home_team': 'Manchester City',
            'away_team': 'Liverpool',
            'league': 'Premier League'
        }
    
    def test_hybrid_predictor_initialization(self):
        """Test hybrid predictor initializes correctly"""
        self.assertIsNotNone(self.predictor)
        self.assertIsNotNone(self.predictor.config)
        self.assertIsNotNone(self.predictor.hybrid_integrator)
    
    def test_hybrid_predictor_status(self):
        """Test hybrid predictor status reporting"""
        status = self.predictor.get_status()
        
        self.assertIsInstance(status, dict)
        self.assertIn('is_available', status)
        self.assertIn('config_loaded', status)
        self.assertTrue(status['config_loaded'])
    
    def test_hybrid_prediction_format(self):
        """Test hybrid prediction output format"""
        result = self.predictor.predict_hybrid_1x2(self.test_features, self.test_context)
        
        # Check required fields
        required_fields = [
            'prob_home_win', 'prob_draw', 'prob_away_win',
            'predicted_outcome', 'confidence', 'calibrated', 'hybrid_used'
        ]
        
        for field in required_fields:
            self.assertIn(field, result, f"Missing required field: {field}")
        
        # Check probability types
        self.assertIsInstance(result['prob_home_win'], float)
        self.assertIsInstance(result['prob_draw'], float)
        self.assertIsInstance(result['prob_away_win'], float)
        
        # Check boolean fields
        self.assertIsInstance(result['calibrated'], bool)
        self.assertIsInstance(result['hybrid_used'], bool)
    
    def test_probability_sum(self):
        """Test that probabilities sum to approximately 1.0"""
        result = self.predictor.predict_hybrid_1x2(self.test_features, self.test_context)
        
        prob_sum = (result['prob_home_win'] + 
                   result['prob_draw'] + 
                   result['prob_away_win'])
        
        self.assertAlmostEqual(prob_sum, 1.0, places=3, 
                              msg=f"Probabilities sum to {prob_sum}, expected ~1.0")
    
    def test_probability_bounds(self):
        """Test that probabilities are within valid bounds [0, 1]"""
        result = self.predictor.predict_hybrid_1x2(self.test_features, self.test_context)
        
        probs = [result['prob_home_win'], result['prob_draw'], result['prob_away_win']]
        
        for prob in probs:
            self.assertGreaterEqual(prob, 0.0, f"Probability {prob} is negative")
            self.assertLessEqual(prob, 1.0, f"Probability {prob} exceeds 1.0")
    
    def test_predicted_outcome_consistency(self):
        """Test that predicted outcome matches highest probability"""
        result = self.predictor.predict_hybrid_1x2(self.test_features, self.test_context)
        
        probs = {
            '1': result['prob_home_win'],
            'X': result['prob_draw'],
            '2': result['prob_away_win']
        }
        
        expected_outcome = max(probs, key=probs.get)
        
        # Map to expected format
        outcome_mapping = {'1': '1', 'X': 'X', '2': '2'}
        expected_mapped = outcome_mapping.get(expected_outcome, expected_outcome)
        
        self.assertEqual(result['predicted_outcome'], expected_mapped,
                        f"Predicted outcome {result['predicted_outcome']} doesn't match highest prob {expected_mapped}")
    
    def test_confidence_calculation(self):
        """Test confidence calculation"""
        result = self.predictor.predict_hybrid_1x2(self.test_features, self.test_context)
        
        max_prob = max(result['prob_home_win'], result['prob_draw'], result['prob_away_win'])
        
        self.assertAlmostEqual(result['confidence'], max_prob, places=3,
                              msg=f"Confidence {result['confidence']} doesn't match max prob {max_prob}")
    
    def test_hybrid_integrator_normalization(self):
        """Test probability normalization"""
        # Test with unnormalized probabilities
        test_probs = {'1': 0.6, 'X': 0.3, '2': 0.2}  # Sum = 1.1
        
        normalized = self.integrator.normalize(test_probs)
        
        # Check sum
        prob_sum = sum(normalized.values())
        self.assertAlmostEqual(prob_sum, 1.0, places=3)
        
        # Check all keys present
        for key in ['1', 'X', '2']:
            self.assertIn(key, normalized)
            self.assertGreaterEqual(normalized[key], 0.0)
            self.assertLessEqual(normalized[key], 1.0)
    
    def test_hybrid_integrator_combination(self):
        """Test probability combination"""
        ml_probs = {'1': 0.45, 'X': 0.25, '2': 0.30}
        scoreline_probs = {'1': 0.42, 'X': 0.28, '2': 0.30}
        poisson_probs = {'1': 0.48, 'X': 0.22, '2': 0.30}
        
        combined = self.integrator.combine_probabilities(
            ml_probs=ml_probs,
            scoreline_probs=scoreline_probs,
            poisson_probs=poisson_probs,
            league='Premier League'
        )
        
        # Check format
        self.assertIsInstance(combined, dict)
        for key in ['1', 'X', '2']:
            self.assertIn(key, combined)
        
        # Check sum
        prob_sum = sum(combined.values())
        self.assertAlmostEqual(prob_sum, 1.0, places=3)
    
    def test_effective_weights(self):
        """Test effective weights calculation"""
        weights = self.integrator.get_effective_weights('Premier League')
        
        self.assertIsInstance(weights, dict)
        
        # Check required sources
        required_sources = ['ml', 'scoreline', 'poisson']
        for source in required_sources:
            self.assertIn(source, weights)
            self.assertIsInstance(weights[source], (int, float))
            self.assertGreaterEqual(weights[source], 0.0)
    
    def test_fallback_behavior(self):
        """Test fallback behavior when components fail"""
        # Test with empty features (should trigger fallback)
        empty_features = pd.DataFrame()
        
        result = self.predictor.predict_hybrid_1x2(empty_features, self.test_context)
        
        # Should still return valid format
        self.assertIn('prob_home_win', result)
        self.assertIn('prob_draw', result)
        self.assertIn('prob_away_win', result)
        
        # Probabilities should still sum to 1
        prob_sum = result['prob_home_win'] + result['prob_draw'] + result['prob_away_win']
        self.assertAlmostEqual(prob_sum, 1.0, places=3)
    
    def test_league_specific_behavior(self):
        """Test league-specific weight adjustments"""
        # Test different leagues
        leagues = ['Premier League', 'Serie A', 'Bundesliga']
        
        for league in leagues:
            context = self.test_context.copy()
            context['league'] = league
            
            result = self.predictor.predict_hybrid_1x2(self.test_features, context)
            
            # Should return valid result for each league
            self.assertIn('prob_home_win', result)
            prob_sum = result['prob_home_win'] + result['prob_draw'] + result['prob_away_win']
            self.assertAlmostEqual(prob_sum, 1.0, places=3)
    
    def test_sources_used_reporting(self):
        """Test sources used reporting"""
        result = self.predictor.predict_hybrid_1x2(self.test_features, self.test_context)
        
        if 'sources_used' in result:
            sources_used = result['sources_used']
            self.assertIsInstance(sources_used, dict)
            
            # Check expected sources
            expected_sources = ['ml', 'scoreline', 'poisson', 'draw_specialist']
            for source in expected_sources:
                if source in sources_used:
                    self.assertIsInstance(sources_used[source], bool)
    
    def test_components_reporting(self):
        """Test components reporting"""
        result = self.predictor.predict_hybrid_1x2(self.test_features, self.test_context)
        
        if 'components' in result:
            components = result['components']
            self.assertIsInstance(components, dict)
            
            # Check that components are properly formatted
            for source, probs in components.items():
                if probs is not None:
                    if isinstance(probs, dict):
                        # Should have 1X2 format
                        self.assertTrue(any(key in probs for key in ['1', 'X', '2']))
                    elif isinstance(probs, (int, float)):
                        # Draw specialist single probability
                        self.assertGreaterEqual(probs, 0.0)
                        self.assertLessEqual(probs, 1.0)


class TestHybridIntegration(unittest.TestCase):
    """Test cases for hybrid integration utilities"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = {
            'weights': {
                'ml': 0.45,
                'scoreline': 0.25,
                'poisson': 0.20,
                'draw_specialist': 0.10
            }
        }
        self.integrator = HybridIntegrator(self.config)
    
    def test_kl_divergence_calculation(self):
        """Test KL divergence calculation"""
        p = [0.5, 0.3, 0.2]
        q = [0.4, 0.4, 0.2]
        
        kl_div = self.integrator._kl_divergence(p, q)
        
        self.assertIsInstance(kl_div, float)
        self.assertGreaterEqual(kl_div, 0.0)  # KL divergence is non-negative
    
    def test_source_disagreement(self):
        """Test source disagreement calculation"""
        components = {
            'ml': {'1': 0.5, 'X': 0.3, '2': 0.2},
            'scoreline': {'1': 0.4, 'X': 0.4, '2': 0.2},
            'poisson': {'1': 0.6, 'X': 0.2, '2': 0.2}
        }
        
        disagreement = self.integrator._calculate_source_disagreement(components)
        
        self.assertIsInstance(disagreement, float)
        self.assertGreaterEqual(disagreement, 0.0)
    
    def test_hybrid_summary(self):
        """Test hybrid summary generation"""
        result = {
            'sources_used': {'ml': True, 'scoreline': True, 'poisson': False},
            'components': {
                'ml': {'1': 0.5, 'X': 0.3, '2': 0.2},
                'scoreline': {'1': 0.4, 'X': 0.4, '2': 0.2}
            },
            'weights_used': {'ml': 0.6, 'scoreline': 0.4},
            'confidence': 0.7,
            'predicted_outcome': '1'
        }
        
        summary = self.integrator.hybrid_summary(result)
        
        self.assertIsInstance(summary, dict)
        self.assertIn('total_sources', summary)
        self.assertIn('confidence_level', summary)
        self.assertIn('dominant_outcome', summary)


def run_tests():
    """Run all tests"""
    print("🧪 Running Hybrid 1X2 Tests")
    print("=" * 50)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestHybrid1X2))
    suite.addTests(loader.loadTestsFromTestCase(TestHybridIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 50)
    if result.wasSuccessful():
        print("✅ All tests passed!")
    else:
        print(f"❌ {len(result.failures)} test(s) failed")
        print(f"❌ {len(result.errors)} error(s) occurred")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    run_tests()
