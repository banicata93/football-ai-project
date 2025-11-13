#!/usr/bin/env python3
"""
Тестове за Advanced Drift Analyzer
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

from pipelines.drift_analyzer import DriftAnalyzer, run_drift_analysis


class TestDriftAnalyzer(unittest.TestCase):
    """Тестове за DriftAnalyzer класа"""
    
    def setUp(self):
        """Настройка преди всеки тест"""
        # Създава временна директория
        self.temp_dir = tempfile.mkdtemp()
        
        # Създава тестова конфигурация
        self.test_config = {
            'drift_detection': {
                'enabled': True,
                'analysis_window_days': 7,
                'baseline_window_days': 30,
                'min_samples_per_league': 20,
                'thresholds': {
                    'kl_divergence': 0.10,
                    'jensen_shannon': 0.08,
                    'psi': 0.15,
                    'ece_change': 0.03,
                    'brier_change': 0.05,
                    'league_isolation': 0.20
                },
                'severity_levels': {
                    'low': 0.5,
                    'medium': 0.8,
                    'high': 1.0,
                    'critical': 1.5
                },
                'reporting': {
                    'enabled': True,
                    'report_file': os.path.join(self.temp_dir, 'drift_report.json')
                },
                'logging': {
                    'enabled': True,
                    'log_file': os.path.join(self.temp_dir, 'drift.log')
                }
            }
        }
        
        # Създава тестов config файл
        self.config_path = os.path.join(self.temp_dir, 'drift_config.yaml')
        with open(self.config_path, 'w') as f:
            yaml.dump(self.test_config, f)
        
        # Създава тестови данни
        self.test_data = self._create_test_data()
    
    def tearDown(self):
        """Почистване след всеки тест"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_test_data(self) -> pd.DataFrame:
        """Създава тестови данни за drift анализ"""
        np.random.seed(42)
        n_samples = 500
        
        # Създава timestamps за последните 40 дни
        end_date = datetime.now()
        timestamps = [
            end_date - timedelta(days=i) for i in range(n_samples-1, -1, -1)
        ]
        
        # Симулира различни drift сценарии
        data = []
        
        for i, timestamp in enumerate(timestamps):
            # Drift ефект - променя се с времето
            drift_factor = 1.0 + 0.3 * (i / n_samples)  # Постепенно увеличение
            
            # Baseline predictions (първите 30 дни)
            if i < 400:  # Baseline период
                poisson_pred = np.random.beta(2, 2)
                ml_pred = np.random.beta(2.5, 2.5)
                elo_pred = np.random.beta(1.5, 1.5)
            else:  # Drift период (последните 7 дни)
                # Симулира drift - ML модела се влошава
                poisson_pred = np.random.beta(2, 2)
                ml_pred = np.random.beta(1.8, 2.8) * drift_factor  # Drift в ML
                elo_pred = np.random.beta(1.5, 1.5)
            
            # Ensemble prediction
            ensemble_pred = 0.3 * poisson_pred + 0.5 * ml_pred + 0.2 * elo_pred
            
            # Actual result (корелиран с ensemble)
            actual_result = int(ensemble_pred + np.random.normal(0, 0.1) > 0.5)
            
            # League rotation
            leagues = ['premier_league', 'la_liga', 'serie_a', 'bundesliga']
            league_slug = leagues[i % len(leagues)]
            
            record = {
                'timestamp': timestamp.isoformat(),
                'league_slug': league_slug,
                'poisson_prediction': float(np.clip(poisson_pred, 0, 1)),
                'ml_prediction': float(np.clip(ml_pred, 0, 1)),
                'elo_prediction': float(np.clip(elo_pred, 0, 1)),
                'ensemble_prediction': float(np.clip(ensemble_pred, 0, 1)),
                'actual_result': int(actual_result),
                'confidence': float(np.random.uniform(0.6, 0.9))
            }
            
            data.append(record)
        
        return pd.DataFrame(data)
    
    def create_test_analyzer(self):
        """Създава тестов DriftAnalyzer"""
        return DriftAnalyzer(self.config_path)
    
    def test_initialization(self):
        """Тест за инициализация на DriftAnalyzer"""
        analyzer = self.create_test_analyzer()
        
        self.assertIsNotNone(analyzer.config)
        self.assertIsNotNone(analyzer.logger)
        self.assertTrue(analyzer.config['drift_detection']['enabled'])
        self.assertEqual(analyzer.analysis_window, 7)
        self.assertEqual(analyzer.baseline_window, 30)
    
    def test_kl_divergence_calculation(self):
        """Тест за KL Divergence функцията"""
        analyzer = self.create_test_analyzer()
        
        # Тест с идентични разпределения
        p = np.array([0.5, 0.3, 0.2])
        q = np.array([0.5, 0.3, 0.2])
        kl_div = analyzer.calculate_kl_divergence(p, q)
        self.assertAlmostEqual(kl_div, 0.0, places=5)
        
        # Тест с различни разпределения
        p = np.array([0.7, 0.2, 0.1])
        q = np.array([0.3, 0.4, 0.3])
        kl_div = analyzer.calculate_kl_divergence(p, q)
        self.assertGreater(kl_div, 0.0)
        
        # Тест с edge cases
        p = np.array([1.0, 0.0, 0.0])
        q = np.array([0.0, 0.5, 0.5])
        kl_div = analyzer.calculate_kl_divergence(p, q)
        self.assertGreater(kl_div, 0.0)
    
    def test_psi_calculation(self):
        """Тест за PSI функцията"""
        analyzer = self.create_test_analyzer()
        
        # Тест с идентични разпределения
        baseline = np.random.normal(0, 1, 1000)
        current = np.random.normal(0, 1, 1000)
        psi = analyzer.calculate_psi(baseline, current)
        self.assertLess(abs(psi), 0.1)  # Трябва да е близо до 0
        
        # Тест с различни разпределения
        baseline = np.random.normal(0, 1, 1000)
        current = np.random.normal(1, 1, 1000)  # Shifted mean
        psi = analyzer.calculate_psi(baseline, current)
        self.assertGreater(abs(psi), 0.1)  # Трябва да има значителна разлика
    
    def test_ece_calculation(self):
        """Тест за ECE функцията"""
        analyzer = self.create_test_analyzer()
        
        # Perfect calibration
        y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0, 1, 0])
        y_prob = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.6, 0.4, 0.85, 0.15])
        ece = analyzer.calculate_ece(y_true, y_prob, n_bins=5)
        self.assertGreaterEqual(ece, 0.0)
        self.assertLessEqual(ece, 1.0)
        
        # Poor calibration
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_prob = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])
        ece_poor = analyzer.calculate_ece(y_true, y_prob, n_bins=5)
        self.assertGreater(ece_poor, ece)
    
    def test_prediction_drift_detection(self):
        """Тест за prediction drift detection"""
        analyzer = self.create_test_analyzer()
        
        # Разделя данните на baseline и current
        cutoff_date = self.test_data['timestamp'].max()
        cutoff_datetime = datetime.fromisoformat(cutoff_date) - timedelta(days=7)
        
        baseline_df = self.test_data[
            pd.to_datetime(self.test_data['timestamp']) < cutoff_datetime
        ]
        current_df = self.test_data[
            pd.to_datetime(self.test_data['timestamp']) >= cutoff_datetime
        ]
        
        results = analyzer.detect_prediction_drift(baseline_df, current_df)
        
        # Проверява структурата на резултатите
        self.assertIn('drift_detected', results)
        self.assertIn('drift_score', results)
        self.assertIn('components', results)
        self.assertIn('metrics', results)
        
        # Проверява че има анализ за компонентите
        self.assertIn('ml', results['components'])
        self.assertIn('poisson', results['components'])
        
        # Проверява метриките
        ml_component = results['components']['ml']
        self.assertIn('kl_divergence', ml_component)
        self.assertIn('psi', ml_component)
        
        # Drift score трябва да е положителен (заради симулирания drift)
        self.assertGreaterEqual(results['drift_score'], 0)
    
    def test_calibration_drift_detection(self):
        """Тест за calibration drift detection"""
        analyzer = self.create_test_analyzer()
        
        # Разделя данните
        cutoff_date = self.test_data['timestamp'].max()
        cutoff_datetime = datetime.fromisoformat(cutoff_date) - timedelta(days=7)
        
        baseline_df = self.test_data[
            pd.to_datetime(self.test_data['timestamp']) < cutoff_datetime
        ]
        current_df = self.test_data[
            pd.to_datetime(self.test_data['timestamp']) >= cutoff_datetime
        ]
        
        results = analyzer.detect_calibration_drift(baseline_df, current_df)
        
        # Проверява структурата
        self.assertIn('drift_detected', results)
        self.assertIn('drift_score', results)
        self.assertIn('components', results)
        
        # Проверява че има ECE и Brier анализ
        if 'ml' in results['components']:
            ml_component = results['components']['ml']
            self.assertIn('ece_change', ml_component)
            self.assertIn('brier_change', ml_component)
    
    def test_league_specific_drift_detection(self):
        """Тест за league-specific drift detection"""
        analyzer = self.create_test_analyzer()
        
        # Създава тестови данни с достатъчно samples за всяка лига
        test_df = self.test_data.copy()
        test_df['timestamp'] = pd.to_datetime(test_df['timestamp'])
        
        # Увеличава min_samples_per_league за теста
        analyzer.config['drift_detection']['min_samples_per_league'] = 10
        
        results = analyzer.detect_league_specific_drift(test_df)
        
        # Проверява структурата
        self.assertIn('drift_detected', results)
        self.assertIn('drift_score', results)
        self.assertIn('leagues', results)
        self.assertIn('cross_league_consistency', results)
        
        # Проверява че има анализ по лиги (може да е 0 ако няма достатъчно данни)
        leagues = results['leagues']
        self.assertIsInstance(leagues, dict)
        
        # Ако има лиги, проверява структурата
        for league, info in leagues.items():
            self.assertIn('drift_score', info)
            self.assertIn('samples_baseline', info)
            self.assertIn('samples_current', info)
    
    def test_drift_report_generation(self):
        """Тест за генериране на drift report"""
        analyzer = self.create_test_analyzer()
        
        # Симулира анализ резултати
        analysis_results = {
            'prediction_drift': {
                'drift_detected': True,
                'drift_score': 1.2,
                'components': {'ml': {'kl_divergence': 0.15}}
            },
            'calibration_drift': {
                'drift_detected': False,
                'drift_score': 0.5,
                'components': {'ml': {'ece_change': 0.02}}
            }
        }
        
        report = analyzer.generate_drift_report(analysis_results)
        
        # Проверява структурата на отчета
        self.assertIn('timestamp', report)
        self.assertIn('overall_drift', report)
        self.assertIn('drift_types', report)
        self.assertIn('recommendations', report)
        self.assertIn('summary', report)
        
        # Проверява overall drift
        overall = report['overall_drift']
        self.assertIn('detected', overall)
        self.assertIn('severity', overall)
        self.assertIn('score', overall)
        
        # Проверява че има препоръки при drift
        if overall['detected']:
            self.assertGreater(len(report['recommendations']), 0)
    
    @patch('pipelines.drift_analyzer.os.path.exists')
    def test_load_historical_data_no_file(self, mock_exists):
        """Тест за зареждане на данни при липсващ файл"""
        mock_exists.return_value = False
        
        analyzer = self.create_test_analyzer()
        df = analyzer.load_historical_data()
        
        self.assertTrue(df.empty)
    
    @patch('pipelines.drift_analyzer.os.path.exists')
    def test_load_historical_data_with_file(self, mock_exists):
        """Тест за зареждане на данни с файл"""
        mock_exists.return_value = True
        
        # Създава тестов JSONL файл
        history_file = "logs/predictions_history/ou25_predictions.jsonl"
        os.makedirs(os.path.dirname(history_file), exist_ok=True)
        
        with open(history_file, 'w') as f:
            for _, row in self.test_data.iterrows():
                json_line = row.to_dict()
                f.write(json.dumps(json_line) + '\n')
        
        try:
            analyzer = self.create_test_analyzer()
            df = analyzer.load_historical_data(days_back=30)
            
            self.assertFalse(df.empty)
            self.assertIn('timestamp', df.columns)
            self.assertIn('poisson_prediction', df.columns)
            
        finally:
            if os.path.exists(history_file):
                os.remove(history_file)


class TestDriftAnalyzerIntegration(unittest.TestCase):
    """Интеграционни тестове за DriftAnalyzer"""
    
    def setUp(self):
        """Настройка преди всеки тест"""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Почистване след всеки тест"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_run_drift_analysis_function(self):
        """Тест за convenience функцията run_drift_analysis"""
        with patch('pipelines.drift_analyzer.DriftAnalyzer') as mock_analyzer_class:
            mock_analyzer = MagicMock()
            mock_analyzer.run_drift_analysis.return_value = {'enabled': False}
            mock_analyzer_class.return_value = mock_analyzer
            
            result = run_drift_analysis()
            
            self.assertIsInstance(result, dict)
            mock_analyzer_class.assert_called_once()
            mock_analyzer.run_drift_analysis.assert_called_once()
    
    def test_config_loading_fallback(self):
        """Тест за fallback конфигурация при липсващ файл"""
        analyzer = DriftAnalyzer('nonexistent_config.yaml')
        
        # Трябва да използва fallback конфигурация
        self.assertIsNotNone(analyzer.config)
        self.assertIn('drift_detection', analyzer.config)
        self.assertTrue(analyzer.config['drift_detection']['enabled'])
    
    def test_full_analysis_workflow(self):
        """Тест за пълен анализ workflow"""
        # Създава временен config
        config_path = os.path.join(self.temp_dir, 'drift_config.yaml')
        test_config = {
            'drift_detection': {
                'enabled': True,
                'analysis_window_days': 7,
                'baseline_window_days': 30,
                'thresholds': {'kl_divergence': 0.10},
                'severity_levels': {'low': 0.5, 'medium': 0.8, 'high': 1.0, 'critical': 1.5},
                'reporting': {'enabled': True, 'report_file': os.path.join(self.temp_dir, 'report.json')},
                'logging': {'enabled': True, 'log_file': os.path.join(self.temp_dir, 'drift.log')}
            }
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(test_config, f)
        
        # Mock load_historical_data за да върне празни данни
        with patch.object(DriftAnalyzer, 'load_historical_data') as mock_load:
            mock_load.return_value = pd.DataFrame()  # Празни данни
            
            analyzer = DriftAnalyzer(config_path)
            result = analyzer.run_drift_analysis()
            
            # Трябва да върне error заради липсата на данни
            self.assertIn('error', result)


def run_tests():
    """Стартира всички тестове"""
    # Създава test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Добавя тестовете
    suite.addTests(loader.loadTestsFromTestCase(TestDriftAnalyzer))
    suite.addTests(loader.loadTestsFromTestCase(TestDriftAnalyzerIntegration))
    
    # Стартира тестовете
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    print("🧪 СТАРТИРАНЕ НА DRIFT ANALYZER ТЕСТОВЕ")
    print("=" * 70)
    
    success = run_tests()
    
    if success:
        print("\n✅ Всички тестове минаха успешно!")
    else:
        print("\n❌ Някои тестове се провалиха!")
        sys.exit(1)
