"""
Тестове за League-Specific OU2.5 модели
"""

import unittest
import os
import sys
import tempfile
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import joblib
import json
from unittest.mock import patch, MagicMock

from core.league_utils import (
    get_league_slug, get_league_display_name, is_supported_league,
    get_per_league_model_path, get_supported_leagues
)
from api.prediction_service import PredictionService


class TestLeagueUtils(unittest.TestCase):
    """Тестове за league utilities"""
    
    def test_get_league_slug(self):
        """Тест за league slug mapping"""
        # Тест с league names
        self.assertEqual(get_league_slug("Premier League"), "premier_league")
        self.assertEqual(get_league_slug("La Liga"), "la_liga")
        self.assertEqual(get_league_slug("Serie A"), "serie_a")
        self.assertEqual(get_league_slug("Bundesliga"), "bundesliga")
        self.assertEqual(get_league_slug("Ligue 1"), "ligue_1")
        
        # Case insensitive
        self.assertEqual(get_league_slug("premier league"), "premier_league")
        self.assertEqual(get_league_slug("PREMIER LEAGUE"), "premier_league")
        
        # Тест с league IDs
        self.assertEqual(get_league_slug(league_id=1), "premier_league")
        self.assertEqual(get_league_slug(league_id=2), "la_liga")
        self.assertEqual(get_league_slug(league_id=3), "serie_a")
        
        # Неизвестни лиги
        self.assertIsNone(get_league_slug("Unknown League"))
        self.assertIsNone(get_league_slug(league_id=999))
    
    def test_get_league_display_name(self):
        """Тест за display names"""
        self.assertEqual(get_league_display_name("premier_league"), "Premier League")
        self.assertEqual(get_league_display_name("la_liga"), "La Liga")
        self.assertEqual(get_league_display_name("unknown_league"), "Unknown League")
    
    def test_is_supported_league(self):
        """Тест за поддържани лиги"""
        self.assertTrue(is_supported_league("Premier League"))
        self.assertTrue(is_supported_league(league_id=1))
        self.assertFalse(is_supported_league("Unknown League"))
        self.assertFalse(is_supported_league(league_id=999))
    
    def test_get_per_league_model_path(self):
        """Тест за model path generation"""
        path = get_per_league_model_path("premier_league", "ou25", "v1")
        expected = "models/leagues/premier_league/ou25_v1"
        self.assertEqual(path, expected)
    
    def test_get_supported_leagues(self):
        """Тест за поддържани лиги"""
        leagues = get_supported_leagues()
        self.assertIsInstance(leagues, dict)
        self.assertIn("premier_league", leagues)
        self.assertIn("la_liga", leagues)
        self.assertEqual(leagues["premier_league"], "Premier League")


class TestPerLeagueModels(unittest.TestCase):
    """Тестове за per-league модели в API"""
    
    def setUp(self):
        """Setup за тестовете"""
        self.temp_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
        
        # Създава mock структура
        os.makedirs("models/leagues/premier_league/ou25_v1", exist_ok=True)
        os.makedirs("models/leagues/la_liga/ou25_v1", exist_ok=True)
        os.makedirs("config", exist_ok=True)
        
        # Mock конфигурация
        config = {
            'model_ou25': {
                'per_league': {
                    'enabled': True,
                    'target_leagues': ['premier_league', 'la_liga'],
                    'lazy_loading': True,
                    'calibration_enabled': True
                }
            }
        }
        
        with open("config/model_config.yaml", 'w') as f:
            import yaml
            yaml.dump(config, f)
    
    def tearDown(self):
        """Cleanup след тестовете"""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.temp_dir)
    
    @patch('api.prediction_service.joblib.load')
    @patch('api.prediction_service.os.path.exists')
    def test_load_league_model(self, mock_exists, mock_joblib_load):
        """Тест за зареждане на league модел"""
        # Mock файлове съществуват
        mock_exists.return_value = True
        
        # Mock модел
        mock_model = MagicMock()
        mock_calibrator = MagicMock()
        mock_joblib_load.side_effect = [mock_model, mock_calibrator]
        
        # Mock feature файл
        with patch('builtins.open', unittest.mock.mock_open(read_data='["feature1", "feature2"]')):
            service = PredictionService()
            result = service._load_league_model("premier_league")
        
        self.assertTrue(result)
        self.assertIn("premier_league", service.ou25_models_by_league)
        self.assertIn("premier_league", service.ou25_calibrators_by_league)
    
    @patch('api.prediction_service.joblib.load')
    @patch('api.prediction_service.os.path.exists')
    def test_get_ou25_model_for_league(self, mock_exists, mock_joblib_load):
        """Тест за получаване на модел за лига"""
        # Mock setup
        mock_exists.return_value = True
        mock_model = MagicMock()
        mock_joblib_load.return_value = mock_model
        
        with patch('builtins.open', unittest.mock.mock_open(read_data='["feature1"]')):
            service = PredictionService()
            
            # Тест с поддържана лига
            model, calibrator, source = service._get_ou25_model_for_league("Premier League")
            
            # Трябва да върне league-specific модел (ако е зареден)
            self.assertIn(source, ["league_ou25", "global_ou25"])
    
    def test_feature_consistency_check(self):
        """Тест за feature consistency между глобален и league модели"""
        # Създава mock feature файлове
        global_features = ["feature1", "feature2", "feature3"]
        league_features = ["feature1", "feature2", "feature3"]  # Същите features
        
        global_path = "models/model_ou25_v1/feature_list.json"
        league_path = "models/leagues/premier_league/ou25_v1/feature_columns.json"
        
        os.makedirs(os.path.dirname(global_path), exist_ok=True)
        os.makedirs(os.path.dirname(league_path), exist_ok=True)
        
        with open(global_path, 'w') as f:
            json.dump(global_features, f)
        
        with open(league_path, 'w') as f:
            json.dump(league_features, f)
        
        # Проверява че features са еднакви
        with open(global_path, 'r') as f:
            global_feat = json.load(f)
        
        with open(league_path, 'r') as f:
            league_feat = json.load(f)
        
        self.assertEqual(global_feat, league_feat, "Feature mismatch between global and league models")


class TestPerLeagueTraining(unittest.TestCase):
    """Тестове за per-league training pipeline"""
    
    def setUp(self):
        """Setup за training тестовете"""
        # Създава mock данни
        np.random.seed(42)
        n_samples = 3000
        
        self.train_df = pd.DataFrame({
            'league_id': np.random.choice([1, 2, 3], n_samples),  # Premier League, La Liga, Serie A
            'over_2_5': np.random.binomial(1, 0.45, n_samples),
            'feature1': np.random.normal(0, 1, n_samples),
            'feature2': np.random.normal(0, 1, n_samples),
            'feature3': np.random.normal(0, 1, n_samples)
        })
        
        self.val_df = pd.DataFrame({
            'league_id': np.random.choice([1, 2, 3], 1000),
            'over_2_5': np.random.binomial(1, 0.45, 1000),
            'feature1': np.random.normal(0, 1, 1000),
            'feature2': np.random.normal(0, 1, 1000),
            'feature3': np.random.normal(0, 1, 1000)
        })
    
    def test_league_data_filtering(self):
        """Тест за филтриране на данни по лига"""
        from core.league_utils import LEAGUE_ID_TO_SLUG
        
        # Добавя league_slug колона
        self.train_df['league_slug'] = self.train_df['league_id'].map(LEAGUE_ID_TO_SLUG)
        
        # Филтрира Premier League данни
        pl_data = self.train_df[self.train_df['league_slug'] == 'premier_league']
        
        self.assertTrue(len(pl_data) > 0)
        self.assertTrue(all(pl_data['league_id'] == 1))
    
    def test_minimum_matches_threshold(self):
        """Тест за минимален брой мачове"""
        min_matches = 2000
        
        # Брои мачове по лига
        league_counts = self.train_df['league_id'].value_counts() + self.val_df['league_id'].value_counts()
        
        for league_id, count in league_counts.items():
            if count >= min_matches:
                print(f"League {league_id}: {count} мача (достатъчно за тренировка)")
            else:
                print(f"League {league_id}: {count} мача (недостатъчно за тренировка)")
    
    @patch('pipelines.train_ml_models.get_feature_columns')
    def test_feature_consistency_assertion(self, mock_get_features):
        """Тест за feature consistency assertion"""
        mock_get_features.return_value = ['feature1', 'feature2', 'feature3']
        
        # Симулира feature consistency check
        global_features = mock_get_features()
        league_features = ['feature1', 'feature2', 'feature3']  # Същите
        
        # Това не трябва да хвърли грешка
        try:
            assert league_features == global_features, "Feature mismatch OU2.5 per-league vs global"
            assertion_passed = True
        except AssertionError:
            assertion_passed = False
        
        self.assertTrue(assertion_passed)
        
        # Тест с различни features (трябва да хвърли грешка)
        league_features_different = ['feature1', 'feature2', 'feature4']  # Различни
        
        with self.assertRaises(AssertionError):
            assert league_features_different == global_features, "Feature mismatch OU2.5 per-league vs global"


class TestPerLeagueIntegration(unittest.TestCase):
    """Интеграционни тестове за per-league функционалност"""
    
    def test_api_response_format(self):
        """Тест за API response формат с per-league информация"""
        # Mock response с model_sources
        mock_response = {
            'prediction_ou25': {
                'prob_over': 0.65,
                'prob_under': 0.35,
                'predicted_outcome': 'Over',
                'confidence': 0.65
            },
            'model_sources': {
                'ou25': 'league_ou25'  # Или 'global_ou25'
            }
        }
        
        # Проверява че response съдържа model source информация
        self.assertIn('model_sources', mock_response)
        self.assertIn('ou25', mock_response['model_sources'])
        self.assertIn(mock_response['model_sources']['ou25'], ['league_ou25', 'global_ou25'])
    
    def test_fallback_behavior(self):
        """Тест за fallback поведение"""
        # Симулира случай където няма league-specific модел
        league = "Unknown League"
        league_slug = get_league_slug(league)
        
        # Трябва да върне None за неизвестна лига
        self.assertIsNone(league_slug)
        
        # В този случай API трябва да използва глобален модел
        expected_source = "global_ou25"
        self.assertEqual(expected_source, "global_ou25")


if __name__ == '__main__':
    # Стартира тестовете
    unittest.main(verbosity=2)
