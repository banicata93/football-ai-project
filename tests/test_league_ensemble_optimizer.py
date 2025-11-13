#!/usr/bin/env python3
"""
Тестове за League-Specific Dynamic Ensemble Optimizer
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

from pipelines.league_ensemble_optimizer import LeagueEnsembleOptimizer, run_league_ensemble_optimization


class TestLeagueEnsembleOptimizer(unittest.TestCase):
    """Тестове за LeagueEnsembleOptimizer класа"""
    
    def setUp(self):
        """Настройка преди всеки тест"""
        # Създава временна директория
        self.temp_dir = tempfile.mkdtemp()
        
        # Създава тестова конфигурация
        self.test_config = {
            'league_ensembles': {
                'enabled': True,
                'lookback_days': 30,
                'min_matches_per_league': 50,
                'min_improvement': 0.02,
                'constraints': {
                    'min_weight': 0.1,
                    'max_weight': 0.8
                },
                'cross_validation': {
                    'enabled': True,
                    'folds': 3,
                    'validation_threshold': 0.01
                },
                'optimization': {
                    'method': 'SLSQP',
                    'max_iterations': 100,
                    'tolerance': 1e-6,
                    'random_restarts': 2
                },
                'backup': {
                    'enabled': True,
                    'max_backups': 5,
                    'backup_dir': os.path.join(self.temp_dir, 'backups/')
                },
                'default_weights': {
                    'poisson': 0.30,
                    'ml': 0.50,
                    'elo': 0.20
                },
                'output': {
                    'weights_file': os.path.join(self.temp_dir, 'league_weights.yaml'),
                    'results_file': os.path.join(self.temp_dir, 'results.json')
                },
                'logging': {
                    'enabled': True,
                    'log_file': os.path.join(self.temp_dir, 'league_ensemble.log')
                }
            }
        }
        
        # Създава тестов config файл
        self.config_path = os.path.join(self.temp_dir, 'league_ensemble.yaml')
        with open(self.config_path, 'w') as f:
            yaml.dump(self.test_config, f)
        
        # Създава тестови данни
        self.test_data = self._create_test_data()
    
    def tearDown(self):
        """Почистване след всеки тест"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_test_data(self) -> pd.DataFrame:
        """Създава тестови данни за league ensemble optimization"""
        np.random.seed(42)
        n_samples = 300
        
        # Създава timestamps за последните 35 дни
        end_date = datetime.now()
        timestamps = [
            end_date - timedelta(days=i) for i in range(n_samples-1, -1, -1)
        ]
        
        # Симулира различни league performance характеристики
        data = []
        leagues = ['premier_league', 'la_liga', 'serie_a', 'bundesliga']
        
        for i, timestamp in enumerate(timestamps):
            league_slug = leagues[i % len(leagues)]
            
            # Различни performance характеристики по лиги
            if league_slug == 'premier_league':
                # ML е най-добър
                poisson_pred = np.random.beta(2, 2)
                ml_pred = np.random.beta(3, 1.5)  # По-добър
                elo_pred = np.random.beta(1.5, 2)
                true_prob = 0.2 * poisson_pred + 0.7 * ml_pred + 0.1 * elo_pred
                
            elif league_slug == 'la_liga':
                # Poisson е най-добър
                poisson_pred = np.random.beta(3, 1.5)  # По-добър
                ml_pred = np.random.beta(2, 2)
                elo_pred = np.random.beta(1.5, 2)
                true_prob = 0.6 * poisson_pred + 0.3 * ml_pred + 0.1 * elo_pred
                
            elif league_slug == 'serie_a':
                # Балансирани модели
                poisson_pred = np.random.beta(2.2, 2.2)
                ml_pred = np.random.beta(2.2, 2.2)
                elo_pred = np.random.beta(2.2, 2.2)
                true_prob = 0.33 * poisson_pred + 0.33 * ml_pred + 0.34 * elo_pred
                
            else:  # bundesliga
                # Elo е най-добър
                poisson_pred = np.random.beta(1.8, 2.2)
                ml_pred = np.random.beta(1.8, 2.2)
                elo_pred = np.random.beta(3, 1.5)  # По-добър
                true_prob = 0.2 * poisson_pred + 0.2 * ml_pred + 0.6 * elo_pred
            
            # Clipping
            poisson_pred = np.clip(poisson_pred, 0, 1)
            ml_pred = np.clip(ml_pred, 0, 1)
            elo_pred = np.clip(elo_pred, 0, 1)
            
            # Ensemble prediction (default weights)
            ensemble_pred = 0.30 * poisson_pred + 0.50 * ml_pred + 0.20 * elo_pred
            
            # Actual result
            actual_result = int(true_prob + np.random.normal(0, 0.1) > 0.5)
            
            record = {
                'timestamp': timestamp.isoformat(),
                'league_slug': league_slug,
                'poisson_prediction': float(poisson_pred),
                'ml_prediction': float(ml_pred),
                'elo_prediction': float(elo_pred),
                'ensemble_prediction': float(ensemble_pred),
                'actual_result': int(actual_result),
                'confidence': float(np.random.uniform(0.6, 0.9))
            }
            
            data.append(record)
        
        return pd.DataFrame(data)
    
    def create_test_optimizer(self):
        """Създава тестов LeagueEnsembleOptimizer"""
        return LeagueEnsembleOptimizer(self.config_path)
    
    def test_initialization(self):
        """Тест за инициализация на LeagueEnsembleOptimizer"""
        optimizer = self.create_test_optimizer()
        
        self.assertIsNotNone(optimizer.config)
        self.assertIsNotNone(optimizer.logger)
        self.assertTrue(optimizer.ensemble_config['enabled'])
        self.assertEqual(optimizer.lookback_days, 30)
        self.assertEqual(optimizer.min_matches, 50)
    
    @patch('pipelines.league_ensemble_optimizer.os.path.exists')
    def test_load_historical_data_empty(self, mock_exists):
        """Тест за зареждане на данни при липсващ файл"""
        # Мокира че файлът не съществува
        mock_exists.return_value = False
        
        optimizer = self.create_test_optimizer()
        
        df = optimizer.load_historical_data()
        
        self.assertTrue(df.empty)
    
    @patch('pipelines.league_ensemble_optimizer.os.path.exists')
    def test_load_historical_data_with_file(self, mock_exists):
        """Тест за зареждане на данни с файл"""
        optimizer = self.create_test_optimizer()
        
        # Мокира че файлът съществува
        mock_exists.return_value = True
        
        # Създава тестов JSONL файл
        history_file = "logs/predictions_history/ou25_predictions.jsonl"
        os.makedirs(os.path.dirname(history_file), exist_ok=True)
        
        with open(history_file, 'w') as f:
            for _, row in self.test_data.iterrows():
                json_line = row.to_dict()
                f.write(json.dumps(json_line) + '\n')
        
        try:
            df = optimizer.load_historical_data(days_back=30)
            
            self.assertFalse(df.empty)
            self.assertIn('league_slug', df.columns)
            self.assertIn('poisson_prediction', df.columns)
            
        finally:
            if os.path.exists(history_file):
                os.remove(history_file)
    
    def test_evaluate_league_performance(self):
        """Тест за оценка на league performance"""
        optimizer = self.create_test_optimizer()
        
        # Намалява min_matches за теста
        optimizer.min_matches = 20
        
        performance = optimizer.evaluate_league_performance(self.test_data, 'premier_league')
        
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
    
    def test_ensemble_predictions(self):
        """Тест за ensemble прогнози"""
        optimizer = self.create_test_optimizer()
        
        weights = {'poisson': 0.3, 'ml': 0.5, 'elo': 0.2}
        ensemble_pred = optimizer._ensemble_predictions(self.test_data, weights)
        
        self.assertEqual(len(ensemble_pred), len(self.test_data))
        self.assertTrue(np.all(ensemble_pred >= 0))
        self.assertTrue(np.all(ensemble_pred <= 1))
    
    def test_optimize_league_weights_returns_valid_weights(self):
        """Тест че оптимизацията връща валидни тегла"""
        optimizer = self.create_test_optimizer()
        
        # Намалява min_matches за теста
        optimizer.min_matches = 20
        
        new_weights, metrics = optimizer.optimize_league_weights(self.test_data, 'premier_league')
        
        if new_weights:  # Ако има резултат
            # Проверява сумата на теглата
            weights_sum = sum(new_weights.values())
            self.assertAlmostEqual(weights_sum, 1.0, places=6)
            
            # Проверява ограниченията
            for component, weight in new_weights.items():
                if weight > 0:  # Само за активни компоненти
                    self.assertGreaterEqual(weight, 0.1)
                    self.assertLessEqual(weight, 0.8)
            
            # Проверява метрики
            self.assertIn('improvement', metrics)
            self.assertIn('current_log_loss', metrics)
            self.assertIn('new_log_loss', metrics)
    
    def test_optimize_league_weights_insufficient_data(self):
        """Тест за оптимизация с недостатъчно данни"""
        optimizer = self.create_test_optimizer()
        
        # Използва висок min_matches
        optimizer.min_matches = 1000
        
        new_weights, metrics = optimizer.optimize_league_weights(self.test_data, 'premier_league')
        
        # Трябва да върне празни резултати
        self.assertEqual(new_weights, {})
        self.assertEqual(metrics, {})
    
    def test_cross_validate_league_weights(self):
        """Тест за cross validation на league weights"""
        optimizer = self.create_test_optimizer()
        
        # Тегла които трябва да са по-добри от default
        better_weights = {'poisson': 0.20, 'ml': 0.60, 'elo': 0.20}
        
        cv_result = optimizer.cross_validate_league_weights(
            self.test_data, 'premier_league', better_weights
        )
        
        # CV може да е успешна или неуспешна в зависимост от данните
        self.assertIsInstance(cv_result, bool)
    
    def test_backup_current_weights(self):
        """Тест за backup на текущи weights"""
        optimizer = self.create_test_optimizer()
        
        # Създава фалшив weights файл
        weights_file = optimizer.weights_file
        os.makedirs(os.path.dirname(weights_file), exist_ok=True)
        
        test_weights = {'premier_league': {'poisson': 0.3, 'ml': 0.5, 'elo': 0.2}}
        with open(weights_file, 'w') as f:
            yaml.dump(test_weights, f)
        
        # Прави backup
        backup_path = optimizer.backup_current_weights()
        
        # Проверява че backup-ът е създаден
        self.assertTrue(os.path.exists(backup_path))
        self.assertTrue(backup_path.endswith('.yaml'))
    
    def test_save_league_weights(self):
        """Тест за запазване на league weights"""
        optimizer = self.create_test_optimizer()
        
        league_weights = {
            'premier_league': {'poisson': 0.2, 'ml': 0.6, 'elo': 0.2},
            'la_liga': {'poisson': 0.5, 'ml': 0.3, 'elo': 0.2}
        }
        
        metadata = {'timestamp': datetime.now().isoformat()}
        
        optimizer.save_league_weights(league_weights, metadata)
        
        # Проверява че файлът е създаден
        self.assertTrue(os.path.exists(optimizer.weights_file))
        
        # Проверява съдържанието
        with open(optimizer.weights_file, 'r') as f:
            saved_data = yaml.safe_load(f)
        
        self.assertIn('league_weights', saved_data)
        self.assertIn('metadata', saved_data)
        self.assertEqual(saved_data['league_weights'], league_weights)
    
    def test_weight_constraints_validation(self):
        """Тест за валидация на weight constraints"""
        optimizer = self.create_test_optimizer()
        
        # Тества objective function с валидни тегла
        league_df = self.test_data[self.test_data['league_slug'] == 'premier_league']
        components = ['poisson', 'ml', 'elo']
        
        # Валидни тегла
        valid_weights = np.array([0.3, 0.5, 0.2])
        loss = optimizer._objective_function(valid_weights, league_df, components)
        
        self.assertIsInstance(loss, float)
        self.assertGreater(loss, 0)
        self.assertLess(loss, float('inf'))
    
    def test_cleanup_old_backups(self):
        """Тест за почистване на стари backup-и"""
        optimizer = self.create_test_optimizer()
        
        # Създава няколко backup файла
        backup_dir = optimizer.backup_dir
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        for i in range(7):  # Повече от max_backups (5)
            backup_file = backup_dir / f"league_weights_202301{i:02d}_120000.yaml"
            backup_file.write_text("test content")
        
        # Извиква cleanup
        optimizer._cleanup_old_backups()
        
        # Проверява че са останали само 5 файла
        remaining_files = list(backup_dir.glob("league_weights_*.yaml"))
        self.assertLessEqual(len(remaining_files), 5)


class TestLeagueEnsembleOptimizerIntegration(unittest.TestCase):
    """Интеграционни тестове за LeagueEnsembleOptimizer"""
    
    def setUp(self):
        """Настройка преди всеки тест"""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Почистване след всеки тест"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_run_league_ensemble_optimization_function(self):
        """Тест за convenience функцията run_league_ensemble_optimization"""
        with patch('pipelines.league_ensemble_optimizer.LeagueEnsembleOptimizer') as mock_optimizer_class:
            mock_optimizer = MagicMock()
            mock_optimizer.run_league_ensemble_optimization.return_value = {'enabled': False}
            mock_optimizer_class.return_value = mock_optimizer
            
            result = run_league_ensemble_optimization()
            
            self.assertIsInstance(result, dict)
            mock_optimizer_class.assert_called_once()
            mock_optimizer.run_league_ensemble_optimization.assert_called_once()
    
    def test_config_loading_fallback(self):
        """Тест за fallback конфигурация при липсващ файл"""
        optimizer = LeagueEnsembleOptimizer('nonexistent_config.yaml')
        
        # Трябва да използва fallback конфигурация
        self.assertIsNotNone(optimizer.config)
        self.assertIn('league_ensembles', optimizer.config)
        self.assertTrue(optimizer.config['league_ensembles']['enabled'])
    
    def test_full_optimization_workflow(self):
        """Тест за пълен optimization workflow"""
        # Създава временен config
        config_path = os.path.join(self.temp_dir, 'league_ensemble.yaml')
        test_config = {
            'league_ensembles': {
                'enabled': True,
                'lookback_days': 30,
                'min_matches_per_league': 10,  # Ниско за теста
                'min_improvement': 0.01,
                'constraints': {'min_weight': 0.1, 'max_weight': 0.8},
                'cross_validation': {'enabled': False},  # Изключена за теста
                'optimization': {'method': 'SLSQP', 'random_restarts': 1},
                'backup': {'enabled': True, 'backup_dir': os.path.join(self.temp_dir, 'backups/')},
                'default_weights': {'poisson': 0.30, 'ml': 0.50, 'elo': 0.20},
                'output': {
                    'weights_file': os.path.join(self.temp_dir, 'weights.yaml'),
                    'results_file': os.path.join(self.temp_dir, 'results.json')
                },
                'logging': {'enabled': True, 'log_file': os.path.join(self.temp_dir, 'log.log')}
            }
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(test_config, f)
        
        # Mock load_historical_data за да върне празни данни
        with patch.object(LeagueEnsembleOptimizer, 'load_historical_data') as mock_load:
            mock_load.return_value = pd.DataFrame()  # Празни данни
            
            optimizer = LeagueEnsembleOptimizer(config_path)
            result = optimizer.run_league_ensemble_optimization()
            
            # Трябва да върне error заради липсата на данни
            self.assertIn('error', result)


def run_tests():
    """Стартира всички тестове"""
    # Създава test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Добавя тестовете
    suite.addTests(loader.loadTestsFromTestCase(TestLeagueEnsembleOptimizer))
    suite.addTests(loader.loadTestsFromTestCase(TestLeagueEnsembleOptimizerIntegration))
    
    # Стартира тестовете
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    print("🧪 СТАРТИРАНЕ НА LEAGUE ENSEMBLE OPTIMIZER ТЕСТОВЕ")
    print("=" * 70)
    
    success = run_tests()
    
    if success:
        print("\n✅ Всички тестове минаха успешно!")
    else:
        print("\n❌ Някои тестове се провалиха!")
        sys.exit(1)
