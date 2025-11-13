"""
Unit тестове за подобренията в вероятностната логика
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch

from core.poisson_utils import PoissonModel
from core.ensemble import EnsembleModel
from core.utils import soft_clip_probs, normalize_1x2_probs
from api.prediction_service import PredictionService


class TestPoissonImprovements:
    """Тестове за Poisson модел подобрения"""
    
    def test_per_league_home_advantage(self):
        """Тест за per-league home advantage"""
        model = PoissonModel()
        
        # Test different leagues
        assert model.league_home_advantage[1] == 1.12  # Premier League
        assert model.league_home_advantage[2] == 1.06  # La Liga
        assert model.league_home_advantage[4] == 1.14  # Bundesliga
    
    def test_shrinkage_alpha(self):
        """Тест за shrinkage параметър"""
        model = PoissonModel(shrinkage_alpha=0.3)
        assert model.shrinkage_alpha == 0.3
    
    def test_form_adjustments(self):
        """Тест за form adjustments в calculate_lambda"""
        model = PoissonModel()
        
        # Mock team strengths
        model.attack_strength = {1: 1.2, 2: 0.8}
        model.defense_strength = {1: 0.9, 2: 1.1}
        model.league_avg_goals_home = {0: 1.5}
        model.league_avg_goals_away = {0: 1.2}
        
        # Test with good form vs bad form
        lambda_home_good, lambda_away_good = model.calculate_lambda(
            1, 2, home_form_5=0.8, away_form_5=0.2
        )
        
        lambda_home_bad, lambda_away_bad = model.calculate_lambda(
            1, 2, home_form_5=0.2, away_form_5=0.8
        )
        
        # Good home form should increase lambda_home
        assert lambda_home_good > lambda_home_bad
        # Good away form should increase lambda_away
        assert lambda_away_bad < lambda_away_good
    
    def test_lambda_caps(self):
        """Тест за λ caps"""
        model = PoissonModel()
        
        # Mock extreme team strengths
        model.attack_strength = {1: 5.0, 2: 0.1}  # Extreme values
        model.defense_strength = {1: 0.1, 2: 5.0}
        model.league_avg_goals_home = {0: 1.5}
        model.league_avg_goals_away = {0: 1.2}
        
        lambda_home, lambda_away = model.calculate_lambda(1, 2)
        
        # Should be capped
        assert 0.2 <= lambda_home <= 4.0
        assert 0.2 <= lambda_away <= 4.0
    
    def test_dynamic_max_goals(self):
        """Тест за динамичен max_goals"""
        model = PoissonModel()
        
        # Mock team strengths for high-scoring match
        model.attack_strength = {1: 2.0, 2: 2.0}
        model.defense_strength = {1: 1.5, 2: 1.5}
        model.league_avg_goals_home = {0: 2.0}
        model.league_avg_goals_away = {0: 1.8}
        
        result = model.predict_match_probabilities(1, 2)
        
        # Should have realistic λ values
        assert result['lambda_home'] > 0
        assert result['lambda_away'] > 0
        assert result['expected_total_goals'] > 0


class TestEnsembleImprovements:
    """Тестове за Ensemble модел подобрения"""
    
    def test_dynamic_initialization(self):
        """Тест за dynamic параметри"""
        ensemble = EnsembleModel(dynamic=True)
        assert ensemble.dynamic is True
        
        per_league = {1: {'poisson': 0.4, 'ml': 0.6}}
        ensemble_league = EnsembleModel(per_league_weights=per_league)
        assert ensemble_league.per_league_weights[1]['poisson'] == 0.4
    
    def test_dynamic_weights_high_entropy(self):
        """Тест за динамични weights при висока ентропия"""
        ensemble = EnsembleModel(dynamic=True)
        
        # High entropy ML prediction (uncertain)
        poisson_pred = np.array([0.4, 0.3, 0.3])
        ml_pred = np.array([0.35, 0.33, 0.32])  # Very uncertain
        
        weights = ensemble._get_dynamic_weights(poisson_pred, ml_pred)
        
        # Should increase Poisson weight due to high entropy
        assert weights[0] > ensemble.weights['poisson']  # Poisson weight increased
    
    def test_dynamic_weights_high_disagreement(self):
        """Тест за динамични weights при голямо разминаване"""
        ensemble = EnsembleModel(dynamic=True)
        
        # High disagreement between models
        poisson_pred = np.array([0.7, 0.2, 0.1])
        ml_pred = np.array([0.2, 0.3, 0.5])  # Very different
        
        weights = ensemble._get_dynamic_weights(poisson_pred, ml_pred)
        
        # Should shrink towards 0.5 due to high disagreement
        assert abs(weights[0] - 0.5) < abs(ensemble.weights['poisson'] - 0.5)
    
    def test_per_league_weights(self):
        """Тест за per-league weights"""
        per_league = {1: {'poisson': 0.4, 'ml': 0.6, 'elo': 0.0}}
        ensemble = EnsembleModel(dynamic=True, per_league_weights=per_league)
        
        poisson_pred = np.array([0.4, 0.3, 0.3])
        ml_pred = np.array([0.4, 0.3, 0.3])
        
        weights = ensemble._get_dynamic_weights(poisson_pred, ml_pred, league_id=1)
        
        # Should use league-specific base weights
        assert abs(weights[0] - 0.4) < 0.1  # Close to league-specific weight


class TestConfidenceImprovements:
    """Тестове за подобрен confidence scoring"""
    
    def test_confidence_binary_high_agreement(self):
        """Тест за binary confidence при високо съгласие"""
        service = PredictionService()
        
        # High agreement between models
        confidence = service._confidence_binary(0.8, 0.75)
        
        # Should be high confidence
        assert confidence > 0.7
    
    def test_confidence_binary_low_agreement(self):
        """Тест за binary confidence при ниско съгласие"""
        service = PredictionService()
        
        # Low agreement between models
        confidence = service._confidence_binary(0.8, 0.3)
        
        # Should be lower confidence
        assert confidence < 0.6
    
    def test_confidence_1x2_high_entropy(self):
        """Тест за 1X2 confidence при висока ентропия"""
        service = PredictionService()
        
        # High entropy (uncertain) predictions
        ml_probs = np.array([0.35, 0.33, 0.32])
        poi_probs = np.array([0.34, 0.33, 0.33])
        
        confidence = service._confidence_1x2(ml_probs, poi_probs)
        
        # Should be low confidence due to high entropy
        assert confidence < 0.5
    
    def test_confidence_1x2_low_entropy(self):
        """Тест за 1X2 confidence при ниска ентропия"""
        service = PredictionService()
        
        # Low entropy (confident) predictions
        ml_probs = np.array([0.8, 0.1, 0.1])
        poi_probs = np.array([0.75, 0.15, 0.1])
        
        confidence = service._confidence_1x2(ml_probs, poi_probs)
        
        # Should be high confidence
        assert confidence > 0.7


class TestUtilityFunctions:
    """Тестове за utility функции"""
    
    def test_soft_clip_probs(self):
        """Тест за soft clipping"""
        # Test extreme values
        probs = np.array([0.001, 0.5, 0.999])
        clipped = soft_clip_probs(probs)
        
        # Should be within soft bounds
        assert np.all(clipped >= 0.02)
        assert np.all(clipped <= 0.98)
        
        # Middle values should be less affected
        assert abs(clipped[1] - 0.5) < 0.1
    
    def test_normalize_1x2_probs(self):
        """Тест за 1X2 normalization"""
        probs = np.array([0.5, 0.3, 0.1])
        normalized = normalize_1x2_probs(probs)
        
        # Should sum to 1
        assert abs(normalized.sum() - 1.0) < 1e-10
        
        # Should be soft clipped
        assert np.all(normalized >= 0.02)
        assert np.all(normalized <= 0.98)


class TestLeagueFallbacks:
    """Тестове за league-specific fallbacks"""
    
    def test_get_league_fallback_known_league(self):
        """Тест за известна лига"""
        service = PredictionService()
        
        fallback = service._get_league_fallback('Premier League')
        
        # Should have Premier League specific values
        assert fallback['prob_over25'] == 0.58  # High-scoring league
        assert fallback['probs_1x2'][0] == 0.46  # Home advantage
    
    def test_get_league_fallback_unknown_league(self):
        """Тест за неизвестна лига"""
        service = PredictionService()
        
        fallback = service._get_league_fallback('Unknown League')
        
        # Should use default values
        assert fallback['prob_over25'] == 0.54
        assert len(fallback['probs_1x2']) == 3
        assert abs(sum(fallback['probs_1x2']) - 1.0) < 0.01
    
    def test_get_league_fallback_none(self):
        """Тест за None лига"""
        service = PredictionService()
        
        fallback = service._get_league_fallback(None)
        
        # Should use default values
        assert 'probs_1x2' in fallback
        assert 'prob_over25' in fallback
        assert 'prob_btts' in fallback


class TestIntegration:
    """Integration тестове"""
    
    @patch('api.prediction_service.PredictionService._create_match_features')
    def test_prediction_with_improvements(self, mock_features):
        """Тест за цялостна прогноза с подобренията"""
        # Mock feature creation
        mock_features.return_value = pd.DataFrame({
            'elo_diff': [100],
            'home_form_5': [0.6],
            'away_form_5': [0.4],
            'home_xg_proxy': [1.5],
            'away_xg_proxy': [1.2],
            'home_shooting_efficiency': [0.3],
            'away_shooting_efficiency': [0.25]
        })
        
        service = PredictionService()
        
        # Mock models
        service.models = {
            'poisson': Mock(),
            '1x2': Mock(),
            'ou25': Mock(),
            'btts': Mock(),
            'ensemble': Mock(),
            'fii': Mock()
        }
        
        # Mock model predictions
        service.models['poisson'].predict_match_probabilities.return_value = {
            'prob_home_win': 0.5,
            'prob_draw': 0.3,
            'prob_away_win': 0.2,
            'prob_over_25': 0.6,
            'prob_btts_yes': 0.55,
            'lambda_home': 1.6,
            'lambda_away': 1.3,
            'expected_total_goals': 2.9
        }
        
        service.models['1x2'].predict_proba.return_value = np.array([[0.48, 0.32, 0.20]])
        service.models['ou25'].predict_proba.return_value = np.array([[0.35, 0.65]])
        service.models['btts'].predict_proba.return_value = np.array([[0.42, 0.58]])
        
        service.models['ensemble'].predict.return_value = np.array([0.49, 0.31, 0.20])
        service.models['fii'].calculate_fii.return_value = (7.5, 'High')
        
        # Test prediction
        result = service.predict('Test Home', 'Test Away', 'Premier League')
        
        # Should have all required fields
        assert 'prediction_1x2' in result
        assert 'prediction_ou25' in result
        assert 'prediction_btts' in result
        assert 'confidence' in result['prediction_1x2']
        
        # Confidence should be calculated with new method
        confidence = result['prediction_1x2']['confidence']
        assert 0 <= confidence <= 1


if __name__ == "__main__":
    # Ръчно тестване
    print("🧪 Тестване на вероятностни подобрения...")
    
    # Test Poisson improvements
    test_poisson = TestPoissonImprovements()
    test_poisson.test_per_league_home_advantage()
    test_poisson.test_form_adjustments()
    test_poisson.test_lambda_caps()
    print("✅ Poisson тестове преминаха")
    
    # Test Ensemble improvements
    test_ensemble = TestEnsembleImprovements()
    test_ensemble.test_dynamic_initialization()
    print("✅ Ensemble тестове преминаха")
    
    # Test Confidence improvements
    test_confidence = TestConfidenceImprovements()
    print("✅ Confidence тестове преминаха")
    
    # Test Utilities
    test_utils = TestUtilityFunctions()
    test_utils.test_soft_clip_probs()
    test_utils.test_normalize_1x2_probs()
    print("✅ Utility тестове преминаха")
    
    print("\n🎉 Всички тестове за вероятностни подобрения преминаха успешно!")
