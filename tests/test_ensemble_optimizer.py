#!/usr/bin/env python3
"""
Тестове за Dynamic Ensemble Optimizer
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
import yaml
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from pipelines.ensemble_optimizer import EnsembleOptimizer, optimize_ensemble_weights


class TestEnsembleOptimizer(unittest.TestCase):
    """Тестове за EnsembleOptimizer класа"""
    
    def setUp(self):
        """Настройка преди всеки тест"""
        # Създава временна директория
        self.temp_dir = tempfile.mkdtemp()
        
        # Създава тестова конфигурация
        self.test_config = {
            'ensemble': {
                'current_weights': {
                    'poisson': 0.30,
                    'ml': 0.50,
                    'elo': 0.20
                },
                'optimization': {
                    'enabled': True,
                    'min_improvement': 0.02,
                    'lookback_days': 30,
                    'weight_constraints': {
                        'min_weight': 0.1,
                        'max_weight': 0.8
                    },
                    'cross_validation_folds': 3,
                    'validation_threshold': 0.01
                },
                'backup': {
                    'enabled': True,
                    'max_backups': 5,
                    'backup_dir': os.path.join(self.temp_dir, 'backups/')
                },
                'logging': {
                    'enabled': True,
                    'log_file': os.path.join(self.temp_dir, 'ensemble.log')
                },
                'history': {
                    'optimization_count': 0
                }
            }
        }
        
        # Създава тестов config файл
        self.config_path = os.path.join(self.temp_dir, 'ensemble_weights.yaml')
        with open(self.config_path, 'w') as f:
            yaml.dump(self.test_config, f)
        
        # Създава тестови исторически данни
        self.test_data = self._create_test_data()
    
    def tearDown(self):
        """Почистване след всеки тест"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_test_data(self) -> pd.DataFrame:
        """Създава тестови данни за оптимизация"""
        np.random.seed(42)
        n_samples = 200
        
        # Симулира компонентни прогнози
        poisson_pred = np.random.beta(2, 2, n_samples)  # Poisson predictions
        ml_pred = np.random.beta(2.5, 2.5, n_samples)  # ML predictions (по-добри)
        elo_pred = np.random.beta(1.5, 1.5, n_samples)  # Elo predictions (по-слаби)
        
        # Симулира реални резултати (корелирани с ML)
        actual_result = (ml_pred + np.random.normal(0, 0.1, n_samples) > 0.5).astype(int)
        
        # Timestamps
        timestamps = [
            (datetime.now() - timedelta(days=i)).isoformat()
            for i in range(n_samples-1, -1, -1)
        ]
        
        return pd.DataFrame({
            'timestamp': timestamps,
            'poisson_prediction': poisson_pred,
            'ml_prediction': ml_pred,
            'elo_prediction': elo_pred,
            'actual_result': actual_result
        })
    
    def create_test_optimizer(self):
        """Създава тестов EnsembleOptimizer"""
        return EnsembleOptimizer(self.config_path)
    
    def test_initialization(self):
        """Тест за инициализация на EnsembleOptimizer"""
        optimizer = self.create_test_optimizer()
        
        self.assertIsNotNone(optimizer.config)
        self.assertIsNotNone(optimizer.logger)
        self.assertTrue(optimizer.opt_config['enabled'])
        self.assertEqual(optimizer.current_weights['poisson'], 0.30)
        self.assertEqual(optimizer.current_weights['ml'], 0.50)
        self.assertEqual(optimizer.current_weights['elo'], 0.20)
    
    def test_load_historical_predictions_empty(self):
        """Тест за зареждане на исторически данни при липсващ файл"""
        optimizer = self.create_test_optimizer()
        
        df = optimizer.load_historical_predictions()
        
        self.assertTrue(df.empty)
    
    @patch('pipelines.ensemble_optimizer.os.path.exists')
    def test_load_historical_predictions_with_data(self, mock_exists):
        """Тест за зареждане на исторически данни"""
        optimizer = self.create_test_optimizer()
        
        # Мокира че файлът съществува
        mock_exists.return_value = True
        
        # Създава тестов JSONL файл
        history_file = "logs/predictions_history/ou25_predictions.jsonl"
        os.makedirs(os.path.dirname(history_file), exist_ok=True)
        
        with open(history_file, 'w') as f:
            for _, row in self.test_data.iterrows():
                json_line = {
                    'timestamp': row['timestamp'],
                    'poisson_prediction': row['poisson_prediction'],
                    'ml_prediction': row['ml_prediction'],
                    'elo_prediction': row['elo_prediction'],
                    'actual_result': row['actual_result']
                }
                f.write(json.dumps(json_line) + '\n')
        
        try:
            df = optimizer.load_historical_predictions(days=30)
            
            self.assertFalse(df.empty)
            self.assertIn('poisson_prediction', df.columns)
            self.assertIn('ml_prediction', df.columns)
            self.assertIn('elo_prediction', df.columns)
            self.assertIn('actual_result', df.columns)
        finally:
            if os.path.exists(history_file):
                os.remove(history_file)
    
    def test_evaluate_component_performance(self):
        """Тест за оценка на компонентите"""
        optimizer = self.create_test_optimizer()
        
        performance = optimizer.evaluate_component_performance(self.test_data)
        
        self.assertIsInstance(performance, dict)
        
        # Проверява че всички компоненти са оценени
        expected_components = ['poisson', 'ml', 'elo']
        for component in expected_components:
            self.assertIn(component, performance)
            
            metrics = performance[component]
            self.assertIn('log_loss', metrics)
            self.assertIn('brier_score', metrics)
            self.assertIn('accuracy', metrics)
            self.assertIn('samples', metrics)
            
            # Проверява валидни стойности
            self.assertGreater(metrics['log_loss'], 0)
            self.assertGreaterEqual(metrics['brier_score'], 0)
            self.assertLessEqual(metrics['brier_score'], 1)
            self.assertGreaterEqual(metrics['accuracy'], 0)
            self.assertLessEqual(metrics['accuracy'], 1)
    
    def test_ensemble_predictions(self):
        """Тест за ensemble прогнози"""
        optimizer = self.create_test_optimizer()
        
        weights = {'poisson': 0.3, 'ml': 0.5, 'elo': 0.2}
        ensemble_pred = optimizer._ensemble_predictions(self.test_data, weights)
        
        self.assertEqual(len(ensemble_pred), len(self.test_data))
        self.assertTrue(np.all(ensemble_pred >= 0))
        self.assertTrue(np.all(ensemble_pred <= 1))
        
        # Проверява че ensemble е комбинация от компонентите
        expected = (
            0.3 * self.test_data['poisson_prediction'].values +
            0.5 * self.test_data['ml_prediction'].values +
            0.2 * self.test_data['elo_prediction'].values
        )
        np.testing.assert_array_almost_equal(ensemble_pred, expected)
    
    def test_optimize_weights_returns_valid_sum(self):
        """Тест че оптимизацията връща тегла със сума = 1.0"""
        optimizer = self.create_test_optimizer()
        
        new_weights, metrics = optimizer.optimize_weights(self.test_data)
        
        # Проверява сумата на теглата
        weights_sum = sum(new_weights.values())
        self.assertAlmostEqual(weights_sum, 1.0, places=6)
        
        # Проверява ограниченията
        for component, weight in new_weights.items():
            if weight > 0:  # Само за активни компоненти
                self.assertGreaterEqual(weight, 0.1)
                self.assertLessEqual(weight, 0.8)
        
        # Проверява метрики
        self.assertIn('current_log_loss', metrics)
        self.assertIn('new_log_loss', metrics)
        self.assertIn('improvement', metrics)
    
    def test_optimize_weights_empty_data(self):
        """Тест за оптимизация с празни данни"""
        optimizer = self.create_test_optimizer()
        
        empty_df = pd.DataFrame()
        new_weights, metrics = optimizer.optimize_weights(empty_df)
        
        # Трябва да върне текущите тегла
        self.assertEqual(new_weights, optimizer.current_weights)
        self.assertEqual(metrics, {})
    
    def test_validate_new_weights_insufficient_improvement(self):
        """Тест за валидация при недостатъчно подобрение"""
        optimizer = self.create_test_optimizer()
        
        new_weights = {'poisson': 0.31, 'ml': 0.49, 'elo': 0.20}
        metrics = {'improvement': 0.01}  # По-малко от 2%
        
        is_valid = optimizer.validate_new_weights(self.test_data, new_weights, metrics)
        
        self.assertFalse(is_valid)
    
    def test_validate_new_weights_sufficient_improvement(self):
        """Тест за валидация при достатъчно подобрение"""
        optimizer = self.create_test_optimizer()
        
        new_weights = {'poisson': 0.20, 'ml': 0.60, 'elo': 0.20}
        metrics = {'improvement': 0.05}  # 5% подобрение
        
        is_valid = optimizer.validate_new_weights(self.test_data, new_weights, metrics)
        
        self.assertTrue(is_valid)
    
    def test_validate_new_weights_invalid_sum(self):
        """Тест за валидация с невалидна сума на тегла"""
        optimizer = self.create_test_optimizer()
        
        new_weights = {'poisson': 0.30, 'ml': 0.50, 'elo': 0.30}  # Сума = 1.1
        metrics = {'improvement': 0.05}
        
        is_valid = optimizer.validate_new_weights(self.test_data, new_weights, metrics)
        
        self.assertFalse(is_valid)
    
    def test_validate_new_weights_constraint_violation(self):
        """Тест за валидация с нарушение на ограниченията"""
        optimizer = self.create_test_optimizer()
        
        new_weights = {'poisson': 0.05, 'ml': 0.85, 'elo': 0.10}  # poisson < 0.1, ml > 0.8
        metrics = {'improvement': 0.05}
        
        is_valid = optimizer.validate_new_weights(self.test_data, new_weights, metrics)
        
        self.assertFalse(is_valid)
    
    def test_backup_and_update_workflow(self):
        """Тест за backup и update workflow"""
        optimizer = self.create_test_optimizer()
        
        # Създава backup
        backup_path = optimizer.backup_old_weights()
        
        self.assertTrue(os.path.exists(backup_path))
        self.assertTrue(backup_path.endswith('.yaml'))
        
        # Тества update на конфигурацията
        new_weights = {'poisson': 0.25, 'ml': 0.55, 'elo': 0.20}
        metrics = {'improvement': 0.05, 'new_log_loss': 0.65}
        
        optimizer.update_weights_config(new_weights, metrics, backup_path)
        
        # Проверява че конфигурацията е обновена
        with open(optimizer.config_path, 'r') as f:
            updated_config = yaml.safe_load(f)
        
        self.assertEqual(
            updated_config['ensemble']['current_weights'],
            new_weights
        )
        self.assertIsNotNone(
            updated_config['ensemble']['history']['last_optimization']
        )
    
    def test_cross_validate_weights(self):
        """Тест за cross validation на тегла"""
        optimizer = self.create_test_optimizer()
        
        # Тегла които трябва да са по-добри от текущите
        better_weights = {'poisson': 0.20, 'ml': 0.60, 'elo': 0.20}
        
        cv_result = optimizer._cross_validate_weights(self.test_data, better_weights)
        
        # CV може да е успешна или неуспешна в зависимост от данните
        self.assertIsInstance(cv_result, bool)
    
    def test_cleanup_old_backups(self):
        """Тест за почистване на стари backup-и"""
        optimizer = self.create_test_optimizer()
        
        # Създава няколко backup файла
        backup_dir = Path(optimizer.backup_dir)
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        for i in range(7):  # Повече от max_backups (5)
            backup_file = backup_dir / f"ensemble_weights_202301{i:02d}_120000.yaml"
            backup_file.write_text("test content")
        
        # Извиква cleanup
        optimizer._cleanup_old_backups()
        
        # Проверява че са останали само 5 файла
        remaining_files = list(backup_dir.glob("ensemble_weights_*.yaml"))
        self.assertLessEqual(len(remaining_files), 5)


class TestEnsembleOptimizerIntegration(unittest.TestCase):
    """Интеграционни тестове за EnsembleOptimizer"""
    
    def setUp(self):
        """Настройка преди всеки тест"""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Почистване след всеки тест"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_optimize_ensemble_weights_function(self):
        """Тест за convenience функцията optimize_ensemble_weights"""
        # Създава временен config
        config_path = os.path.join(self.temp_dir, 'ensemble_weights.yaml')
        test_config = {
            'ensemble': {
                'current_weights': {'poisson': 0.30, 'ml': 0.50, 'elo': 0.20},
                'optimization': {'enabled': False}  # Изключена за теста
            }
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(test_config, f)
        
        with patch('pipelines.ensemble_optimizer.EnsembleOptimizer') as mock_optimizer_class:
            mock_optimizer = MagicMock()
            mock_optimizer.optimize_ensemble_weights.return_value = {'enabled': False}
            mock_optimizer_class.return_value = mock_optimizer
            
            result = optimize_ensemble_weights()
            
            self.assertIsInstance(result, dict)
            mock_optimizer_class.assert_called_once()
            mock_optimizer.optimize_ensemble_weights.assert_called_once()
    
    def test_integration_with_performance_monitor(self):
        """Тест за интеграция с performance monitor"""
        # Симулира извикване от performance monitor
        with patch('pipelines.ensemble_optimizer.EnsembleOptimizer') as mock_optimizer_class:
            mock_optimizer = MagicMock()
            mock_optimizer.optimize_ensemble_weights.return_value = {
                'enabled': True,
                'success': True,
                'weights_updated': True,
                'metrics': {'improvement': 0.05}
            }
            mock_optimizer_class.return_value = mock_optimizer
            
            # Извиква optimize_ensemble_weights (както в performance_monitor.py)
            try:
                from pipelines.ensemble_optimizer import optimize_ensemble_weights
                result = optimize_ensemble_weights()
                
                # Проверява че няма грешки
                self.assertIsInstance(result, dict)
                self.assertTrue(result.get('enabled', False))
                
            except Exception as e:
                self.fail(f"Integration test failed with error: {e}")
    
    def test_config_loading_fallback(self):
        """Тест за fallback конфигурация при липсващ файл"""
        optimizer = EnsembleOptimizer('nonexistent_config.yaml')
        
        # Трябва да използва fallback конфигурация
        self.assertIsNotNone(optimizer.config)
        self.assertIn('ensemble', optimizer.config)
        self.assertIn('current_weights', optimizer.config['ensemble'])


def run_tests():
    """Стартира всички тестове"""
    # Създава test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Добавя тестовете
    suite.addTests(loader.loadTestsFromTestCase(TestEnsembleOptimizer))
    suite.addTests(loader.loadTestsFromTestCase(TestEnsembleOptimizerIntegration))
    
    # Стартира тестовете
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    print("🧪 СТАРТИРАНЕ НА ENSEMBLE OPTIMIZER ТЕСТОВЕ")
    print("=" * 70)
    
    success = run_tests()
    
    if success:
        print("\n✅ Всички тестове минаха успешно!")
    else:
        print("\n❌ Някои тестове се провалиха!")
        sys.exit(1)
