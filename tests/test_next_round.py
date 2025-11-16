#!/usr/bin/env python3
"""
Integration tests for Next Round Prediction System

Tests the complete workflow from fixtures loading to API responses.
"""

import unittest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.fixtures_loader import FixturesLoader
from api.prediction_service import PredictionService


class TestNextRoundPrediction(unittest.TestCase):
    """Integration tests for next round prediction system"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.fixtures_loader = FixturesLoader()
        self.prediction_service = PredictionService()
    
    def test_fixtures_loader_initialization(self):
        """Test that fixtures loader initializes correctly"""
        self.assertIsNotNone(self.fixtures_loader)
        self.assertTrue(self.fixtures_loader.espn_data_dir.exists())
    
    def test_get_available_leagues(self):
        """Test getting available leagues"""
        leagues = self.fixtures_loader.get_available_leagues()
        
        # Should have leagues
        self.assertGreater(len(leagues), 0)
        
        # Each league should have required fields
        for league in leagues[:5]:  # Test first 5
            self.assertIn('name', league)
            self.assertIn('slug', league)
            self.assertIn('id', league)
            self.assertIsInstance(league['id'], int)
    
    def test_get_next_round_fixtures(self):
        """Test getting next round fixtures for a league"""
        # Get a test league
        leagues = self.fixtures_loader.get_available_leagues()
        self.assertGreater(len(leagues), 0)
        
        test_league = leagues[0]['slug']
        fixtures = self.fixtures_loader.get_next_round(test_league)
        
        if not fixtures.empty:
            # Check required columns
            required_columns = ['date', 'home_team', 'away_team', 'league', 'round_date']
            for col in required_columns:
                self.assertIn(col, fixtures.columns)
            
            # Check data types
            self.assertTrue(fixtures['date'].dtype.name.startswith('datetime'))
            self.assertIsInstance(fixtures.iloc[0]['home_team'], str)
            self.assertIsInstance(fixtures.iloc[0]['away_team'], str)
    
    def test_prediction_service_league_round(self):
        """Test prediction service league round method"""
        # Use a small test league
        test_league = 'third-round'
        
        result = self.prediction_service.predict_league_round(test_league)
        
        # Check result structure
        self.assertIn('league', result)
        self.assertIn('total_matches', result)
        self.assertIn('matches', result)
        
        self.assertEqual(result['league'], test_league)
        self.assertIsInstance(result['total_matches'], int)
        self.assertIsInstance(result['matches'], list)
        
        # If matches exist, check their structure
        if result['matches']:
            match = result['matches'][0]
            
            # Required fields
            required_fields = ['home_team', 'away_team', 'date']
            for field in required_fields:
                self.assertIn(field, match)
            
            # If predictions exist, check structure
            if match.get('predictions'):
                predictions = match['predictions']
                
                # Should have all prediction types
                self.assertIn('1x2', predictions)
                self.assertIn('ou25', predictions)
                self.assertIn('btts', predictions)
                
                # Check 1X2 structure
                pred_1x2 = predictions['1x2']
                self.assertIn('predicted_outcome', pred_1x2)
                self.assertIn('prob_home_win', pred_1x2)
                self.assertIn('prob_draw', pred_1x2)
                self.assertIn('prob_away_win', pred_1x2)
                
                # Check probabilities sum to ~1.0
                prob_sum = (pred_1x2['prob_home_win'] + 
                           pred_1x2['prob_draw'] + 
                           pred_1x2['prob_away_win'])
                self.assertAlmostEqual(prob_sum, 1.0, places=2)
                
                # Check OU2.5 structure
                pred_ou25 = predictions['ou25']
                self.assertIn('predicted_outcome', pred_ou25)
                self.assertIn('prob_over', pred_ou25)
                self.assertIn('prob_under', pred_ou25)
                
                # Check OU2.5 probabilities sum to ~1.0
                ou25_sum = pred_ou25['prob_over'] + pred_ou25['prob_under']
                self.assertAlmostEqual(ou25_sum, 1.0, places=2)
                
                # Check BTTS structure
                pred_btts = predictions['btts']
                self.assertIn('predicted_outcome', pred_btts)
                self.assertIn('prob_yes', pred_btts)
                self.assertIn('prob_no', pred_btts)
                
                # Check BTTS probabilities sum to ~1.0
                btts_sum = pred_btts['prob_yes'] + pred_btts['prob_no']
                self.assertAlmostEqual(btts_sum, 1.0, places=2)
    
    def test_league_mapping(self):
        """Test ESPN league slug to our system mapping"""
        test_cases = [
            ('2025-26-english-premier-league', 'Premier League'),
            ('2025-26-laliga', 'La Liga'),
            ('2025-26-italian-serie-a', 'Serie A'),
            ('2025-26-german-bundesliga', 'Bundesliga'),
            ('2025-26-ligue-1', 'Ligue 1')
        ]
        
        for espn_slug, expected_name in test_cases:
            mapped_name = self.prediction_service._map_espn_league_to_our_system(espn_slug)
            self.assertEqual(mapped_name, expected_name)
    
    def test_invalid_league_handling(self):
        """Test handling of invalid league names"""
        result = self.prediction_service.predict_league_round('non-existent-league')
        
        self.assertIn('error', result)
        self.assertEqual(result['total_matches'], 0)
        self.assertEqual(result['matches'], [])
    
    def test_probability_bounds(self):
        """Test that all probabilities are within valid bounds"""
        test_league = 'third-round'
        result = self.prediction_service.predict_league_round(test_league)
        
        for match in result.get('matches', []):
            if match.get('predictions'):
                predictions = match['predictions']
                
                # Check 1X2 probabilities
                pred_1x2 = predictions['1x2']
                self.assertGreaterEqual(pred_1x2['prob_home_win'], 0.0)
                self.assertLessEqual(pred_1x2['prob_home_win'], 1.0)
                self.assertGreaterEqual(pred_1x2['prob_draw'], 0.0)
                self.assertLessEqual(pred_1x2['prob_draw'], 1.0)
                self.assertGreaterEqual(pred_1x2['prob_away_win'], 0.0)
                self.assertLessEqual(pred_1x2['prob_away_win'], 1.0)
                
                # Check OU2.5 probabilities
                pred_ou25 = predictions['ou25']
                self.assertGreaterEqual(pred_ou25['prob_over'], 0.0)
                self.assertLessEqual(pred_ou25['prob_over'], 1.0)
                self.assertGreaterEqual(pred_ou25['prob_under'], 0.0)
                self.assertLessEqual(pred_ou25['prob_under'], 1.0)
                
                # Check BTTS probabilities
                pred_btts = predictions['btts']
                self.assertGreaterEqual(pred_btts['prob_yes'], 0.0)
                self.assertLessEqual(pred_btts['prob_yes'], 1.0)
                self.assertGreaterEqual(pred_btts['prob_no'], 0.0)
                self.assertLessEqual(pred_btts['prob_no'], 1.0)
    
    def test_confidence_scores(self):
        """Test that confidence scores are present and valid"""
        test_league = 'third-round'
        result = self.prediction_service.predict_league_round(test_league)
        
        for match in result.get('matches', []):
            if match.get('confidence'):
                confidence = match['confidence']
                
                # Should have confidence scores
                self.assertIn('overall', confidence)
                self.assertIn('fii_score', confidence)
                
                # Confidence should be between 0 and 1
                self.assertGreaterEqual(confidence['overall'], 0.0)
                self.assertLessEqual(confidence['overall'], 1.0)
                self.assertGreaterEqual(confidence['fii_score'], 0.0)
                self.assertLessEqual(confidence['fii_score'], 1.0)


def run_integration_test():
    """Run the integration test suite"""
    print("🧪 RUNNING NEXT-ROUND PREDICTION INTEGRATION TESTS")
    print("=" * 60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestNextRoundPrediction)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("✅ ALL INTEGRATION TESTS PASSED!")
        print(f"📊 Tests run: {result.testsRun}")
    else:
        print("❌ SOME TESTS FAILED!")
        print(f"📊 Tests run: {result.testsRun}")
        print(f"❌ Failures: {len(result.failures)}")
        print(f"⚠️  Errors: {len(result.errors)}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_integration_test()
    sys.exit(0 if success else 1)
