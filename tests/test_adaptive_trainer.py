#!/usr/bin/env python3
"""
Тестове за Adaptive Learning Pipeline
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import unittest
import tempfile
import shutil
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from pipelines.adaptive_trainer import AdaptiveTrainer
from core.league_utils import LEAGUE_ID_TO_SLUG


class TestAdaptiveTrainer(unittest.TestCase):
    """Тестове за AdaptiveTrainer класа"""
    
    def setUp(self):
        """Настройка преди всеки тест"""
        # Създава временна директория
        self.temp_dir = tempfile.mkdtemp()
        
        # Създава тестова конфигурация
        self.test_config = {
            'enabled': True,
            'drift_threshold': 0.05,
            'retrain_min_matches': 50,  # По-малко за тестове
            'retrain_window_days': 30,
            'backup_old_models': True,
            'log_file': os.path.join(self.temp_dir, 'test_metrics.json'),
            'models_dir': os.path.join(self.temp_dir, 'models/leagues/'),
            'backup_dir': os.path.join(self.temp_dir, 'models/backups/'),
            'adaptive_log': os.path.join(self.temp_dir, 'adaptive.log'),
            'performance_metrics': {
                'primary': 'log_loss',
                'secondary': 'brier_score',
                'accuracy_threshold': 0.55
            },
            'max_concurrent_retrains': 1,
            'rollback_on_failure': True,
            'validation_split': 0.2
        }
        
        # Създава тестови метрики
        self.test_metrics = {
            'metrics_by_league': {
                'premier_league': {
                    'accuracy': 0.75,
                    'log_loss': 0.60,
                    'brier_score': 0.20,
                    'matches': 500
                },
                'la_liga': {
                    'accuracy': 0.70,
                    'log_loss': 0.65,
                    'brier_score': 0.22,
                    'matches': 400
                }
            }
        }
        
        # Запазва тестовите метрики
        os.makedirs(os.path.dirname(self.test_config['log_file']), exist_ok=True)
        with open(self.test_config['log_file'], 'w') as f:
            json.dump(self.test_metrics, f)
    
    def tearDown(self):
        """Почистване след всеки тест"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_test_trainer(self):
        """Създава тестов AdaptiveTrainer"""
        with patch('pipelines.adaptive_trainer.AdaptiveTrainer._load_config') as mock_config:
            mock_config.return_value = self.test_config
            trainer = AdaptiveTrainer()
            return trainer
    
    def test_initialization(self):
        """Тест за инициализация на AdaptiveTrainer"""
        trainer = self.create_test_trainer()
        
        self.assertIsNotNone(trainer.config)
        self.assertIsNotNone(trainer.logger)
        self.assertTrue(trainer.config['enabled'])
        self.assertEqual(trainer.config['drift_threshold'], 0.05)
    
    def test_load_current_metrics(self):
        """Тест за зареждане на текущи метрики"""
        trainer = self.create_test_trainer()
        
        metrics = trainer.load_current_metrics()
        
        self.assertIsInstance(metrics, dict)
        self.assertIn('premier_league', metrics)
        self.assertIn('la_liga', metrics)
        self.assertEqual(metrics['premier_league']['accuracy'], 0.75)
    
    def test_load_current_metrics_missing_file(self):
        """Тест за зареждане на метрики при липсващ файл"""
        trainer = self.create_test_trainer()
        trainer.config['log_file'] = 'nonexistent_file.json'
        
        metrics = trainer.load_current_metrics()
        
        self.assertEqual(metrics, {})
    
    def test_detect_drift_no_history(self):
        """Тест за drift detection без исторически данни"""
        trainer = self.create_test_trainer()
        
        drifted_leagues = trainer.detect_drift()
        
        # При липса на история, не трябва да има drift
        self.assertEqual(drifted_leagues, [])
    
    def test_detect_drift_with_history(self):
        """Тест за drift detection с исторически данни"""
        trainer = self.create_test_trainer()
        
        # Създава фалшива история
        history = {
            '2023-01-01T00:00:00': {
                'metrics': {
                    'premier_league': {
                        'accuracy': 0.80,
                        'log_loss': 0.50,  # По-добър от текущия (0.60)
                        'brier_score': 0.18,
                        'matches': 500
                    },
                    'la_liga': {
                        'accuracy': 0.75,
                        'log_loss': 0.55,  # По-добър от текущия (0.65)
                        'brier_score': 0.20,
                        'matches': 400
                    }
                }
            }
        }
        
        with patch.object(trainer, 'load_metrics_history', return_value=history):
            with patch.object(trainer, 'save_metrics_history'):
                drifted_leagues = trainer.detect_drift()
        
        # И двете лиги трябва да имат drift (log_loss се е влошил с >5%)
        self.assertIn('premier_league', drifted_leagues)
        self.assertIn('la_liga', drifted_leagues)
    
    def test_detect_drift_no_drift(self):
        """Тест за drift detection без drift"""
        trainer = self.create_test_trainer()
        
        # Създава история със същите метрики
        history = {
            '2023-01-01T00:00:00': {
                'metrics': self.test_metrics['metrics_by_league'].copy()
            }
        }
        
        with patch.object(trainer, 'load_metrics_history', return_value=history):
            with patch.object(trainer, 'save_metrics_history'):
                drifted_leagues = trainer.detect_drift()
        
        # Не трябва да има drift
        self.assertEqual(drifted_leagues, [])
    
    def test_backup_model(self):
        """Тест за backup на модел"""
        trainer = self.create_test_trainer()
        
        league_slug = 'premier_league'
        
        # Mock get_per_league_model_path за да използва temp директорията
        with patch('pipelines.adaptive_trainer.get_per_league_model_path') as mock_path:
            model_dir = os.path.join(self.temp_dir, 'models', league_slug, 'ou25_v1')
            mock_path.return_value = model_dir
            os.makedirs(model_dir, exist_ok=True)
            
            # Създава тестови файлове
            test_files = ['ou25_model.pkl', 'calibrator.pkl', 'metrics.json']
            for file_name in test_files:
                with open(os.path.join(model_dir, file_name), 'w') as f:
                    f.write('test content')
            
            # Прави backup
            backup_path = trainer.backup_model(league_slug)
            
            # Проверява че backup-ът е създаден
            self.assertTrue(os.path.exists(backup_path))
            
            # Проверява че файловете са копирани
            for file_name in test_files:
                backup_file = os.path.join(backup_path, file_name)
                self.assertTrue(os.path.exists(backup_file))
    
    def test_backup_model_nonexistent(self):
        """Тест за backup на несъществуващ модел"""
        trainer = self.create_test_trainer()
        
        backup_path = trainer.backup_model('nonexistent_league')
        
        # Backup-ът трябва да е неуспешен
        self.assertEqual(backup_path, "")
    
    def test_rollback_model(self):
        """Тест за rollback на модел"""
        trainer = self.create_test_trainer()
        
        league_slug = 'premier_league'
        
        # Създава backup
        backup_dir = os.path.join(self.temp_dir, 'backup_test')
        os.makedirs(backup_dir, exist_ok=True)
        
        test_files = ['ou25_model.pkl', 'calibrator.pkl']
        for file_name in test_files:
            with open(os.path.join(backup_dir, file_name), 'w') as f:
                f.write('backup content')
        
        # Mock get_per_league_model_path за да използва temp директорията
        with patch('pipelines.adaptive_trainer.get_per_league_model_path') as mock_path:
            model_dir = os.path.join(self.temp_dir, 'models', league_slug, 'ou25_v1')
            mock_path.return_value = model_dir
            
            # Прави rollback
            success = trainer.rollback_model(league_slug, backup_dir)
            
            # Проверява успеха
            self.assertTrue(success)
            
            # Проверява че файловете са възстановени
            for file_name in test_files:
                restored_file = os.path.join(model_dir, file_name)
                self.assertTrue(os.path.exists(restored_file))
    
    def test_rollback_model_nonexistent_backup(self):
        """Тест за rollback с несъществуващ backup"""
        trainer = self.create_test_trainer()
        
        success = trainer.rollback_model('premier_league', 'nonexistent_backup')
        
        # Rollback-ът трябва да е неуспешен
        self.assertFalse(success)
    
    @patch('pipelines.adaptive_trainer.os.path.exists')
    @patch('pipelines.adaptive_trainer.pd.read_parquet')
    def test_load_new_data(self, mock_read_parquet, mock_exists):
        """Тест за зареждане на нови данни"""
        trainer = self.create_test_trainer()
        
        # Мокира че файловете съществуват
        mock_exists.return_value = True
        
        # Мокира данни с актуални дати
        recent_dates = pd.date_range(datetime.now() - timedelta(days=15), periods=50)
        test_data = pd.DataFrame({
            'league_id': [3903] * 50,  # Premier League ID
            'over_25': np.random.randint(0, 2, 50),
            'date': recent_dates
        })
        
        mock_read_parquet.return_value = test_data
        
        # Зарежда данни
        new_data = trainer.load_new_data('premier_league', days=30)
        
        # Проверява резултата
        self.assertIsInstance(new_data, pd.DataFrame)
        self.assertTrue(len(new_data) > 0)
        self.assertTrue(all(new_data['league_id'] == 3903))
    
    def test_load_new_data_unknown_league(self):
        """Тест за зареждане на данни за неизвестна лига"""
        trainer = self.create_test_trainer()
        
        new_data = trainer.load_new_data('unknown_league')
        
        # Трябва да върне празен DataFrame
        self.assertTrue(new_data.empty)
    
    def test_adaptive_learning_cycle_disabled(self):
        """Тест за adaptive learning cycle когато е изключен"""
        trainer = self.create_test_trainer()
        trainer.config['enabled'] = False
        
        results = trainer.adaptive_learning_cycle()
        
        self.assertFalse(results['enabled'])
    
    def test_adaptive_learning_cycle_no_drift(self):
        """Тест за adaptive learning cycle без drift"""
        trainer = self.create_test_trainer()
        
        with patch.object(trainer, 'detect_drift', return_value=[]):
            results = trainer.adaptive_learning_cycle()
        
        self.assertTrue(results['enabled'])
        self.assertEqual(results['drifted_leagues'], [])
        self.assertEqual(results['retrained_leagues'], [])
        self.assertEqual(results['summary']['total_drifted'], 0)
    
    def test_adaptive_learning_cycle_with_drift(self):
        """Тест за adaptive learning cycle с drift"""
        trainer = self.create_test_trainer()
        
        drifted_leagues = ['premier_league']
        
        with patch.object(trainer, 'detect_drift', return_value=drifted_leagues):
            with patch.object(trainer, 'retrain_league_model', return_value=True):
                results = trainer.adaptive_learning_cycle()
        
        self.assertTrue(results['enabled'])
        self.assertEqual(results['drifted_leagues'], drifted_leagues)
        self.assertEqual(results['retrained_leagues'], drifted_leagues)
        self.assertEqual(results['summary']['total_retrained'], 1)
        self.assertEqual(results['summary']['success_rate'], 1.0)
    
    def test_adaptive_learning_cycle_failed_retrain(self):
        """Тест за adaptive learning cycle с неуспешен retrain"""
        trainer = self.create_test_trainer()
        
        drifted_leagues = ['premier_league']
        
        with patch.object(trainer, 'detect_drift', return_value=drifted_leagues):
            with patch.object(trainer, 'retrain_league_model', return_value=False):
                results = trainer.adaptive_learning_cycle()
        
        self.assertTrue(results['enabled'])
        self.assertEqual(results['drifted_leagues'], drifted_leagues)
        self.assertEqual(results['retrained_leagues'], [])
        self.assertEqual(results['failed_retrains'], drifted_leagues)
        self.assertEqual(results['summary']['success_rate'], 0.0)


class TestAdaptiveTrainerIntegration(unittest.TestCase):
    """Интеграционни тестове за AdaptiveTrainer"""
    
    def test_config_loading(self):
        """Тест за зареждане на конфигурация от файл"""
        # Създава временен config файл
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
adaptive_learning:
  enabled: true
  drift_threshold: 0.1
  retrain_min_matches: 100
  models_dir: "models/leagues/"
  backup_dir: "models/backups/"
  adaptive_log: "logs/adaptive.log"
  log_file: "logs/metrics.json"
  performance_metrics:
    primary: "log_loss"
""")
            config_path = f.name
        
        try:
            trainer = AdaptiveTrainer(config_path)
            self.assertTrue(trainer.config['enabled'])
            self.assertEqual(trainer.config['drift_threshold'], 0.1)
            self.assertEqual(trainer.config['retrain_min_matches'], 100)
        finally:
            os.unlink(config_path)
    
    def test_config_loading_fallback(self):
        """Тест за fallback конфигурация при липсващ файл"""
        trainer = AdaptiveTrainer('nonexistent_config.yaml')
        
        # Трябва да използва fallback конфигурация
        self.assertTrue(trainer.config['enabled'])
        self.assertEqual(trainer.config['drift_threshold'], 0.05)


def run_tests():
    """Стартира всички тестове"""
    # Създава test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Добавя тестовете
    suite.addTests(loader.loadTestsFromTestCase(TestAdaptiveTrainer))
    suite.addTests(loader.loadTestsFromTestCase(TestAdaptiveTrainerIntegration))
    
    # Стартира тестовете
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    print("🧪 СТАРТИРАНЕ НА ADAPTIVE TRAINER ТЕСТОВЕ")
    print("=" * 60)
    
    success = run_tests()
    
    if success:
        print("\n✅ Всички тестове минаха успешно!")
    else:
        print("\n❌ Някои тестове се провалиха!")
        sys.exit(1)
