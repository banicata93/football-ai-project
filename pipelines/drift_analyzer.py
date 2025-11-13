#!/usr/bin/env python3
"""
Advanced Drift Analyzer за откриване на различни типове drift в ML моделите

Анализира Data Drift, Prediction Drift, Concept Drift, Feature Stability Drift
и League-Specific Drift използвайки статистически метрики и тестове.
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import pandas as pd
import numpy as np
import yaml
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import logging
from scipy import stats
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance, ks_2samp
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.metrics import brier_score_loss
import itertools

from core.utils import setup_logging


class DriftAnalyzer:
    """
    Advanced Drift Analyzer за comprehensive drift detection
    """
    
    def __init__(self, config_path: str = "config/drift_config.yaml"):
        """
        Инициализация на DriftAnalyzer
        
        Args:
            config_path: Път към конфигурационния файл
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.logger = self._setup_logging()
        
        # Drift thresholds
        self.thresholds = self.config['drift_detection']['thresholds']
        
        # Analysis windows
        self.analysis_window = self.config['drift_detection']['analysis_window_days']
        self.baseline_window = self.config['drift_detection']['baseline_window_days']
        
        self.logger.info("🔍 DriftAnalyzer инициализиран")
    
    def _load_config(self) -> Dict:
        """Зарежда конфигурацията"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            # Fallback конфигурация
            return {
                'drift_detection': {
                    'enabled': True,
                    'analysis_window_days': 7,
                    'baseline_window_days': 60,
                    'thresholds': {
                        'kl_divergence': 0.10,
                        'psi': 0.15,
                        'ece_change': 0.03
                    }
                }
            }
    
    def _setup_logging(self) -> logging.Logger:
        """Настройва logging за drift analyzer"""
        logger = setup_logging()
        
        # Добавя file handler
        log_config = self.config['drift_detection'].get('logging', {})
        log_file = log_config.get('log_file', 'logs/drift_analyzer.log')
        
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        return logger
    
    def load_historical_data(self, days_back: int = None) -> pd.DataFrame:
        """
        Зарежда исторически данни за drift анализ
        
        Args:
            days_back: Брой дни назад
        
        Returns:
            DataFrame с исторически данни
        """
        if days_back is None:
            days_back = self.baseline_window + self.analysis_window
        
        try:
            history_file = "logs/predictions_history/ou25_predictions.jsonl"
            
            if not os.path.exists(history_file):
                self.logger.warning(f"❌ Не е намерен history файл: {history_file}")
                return pd.DataFrame()
            
            # Зарежда JSONL данни
            predictions = []
            cutoff_date = datetime.now() - timedelta(days=days_back)
            
            with open(history_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        pred_date = datetime.fromisoformat(data.get('timestamp', ''))
                        
                        if pred_date >= cutoff_date:
                            predictions.append(data)
                            
                    except (json.JSONDecodeError, ValueError):
                        continue
            
            if not predictions:
                self.logger.warning(f"❌ Няма данни за последните {days_back} дни")
                return pd.DataFrame()
            
            df = pd.DataFrame(predictions)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            self.logger.info(f"📊 Заредени {len(df)} записа за drift анализ")
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Грешка при зареждане на данни: {e}")
            return pd.DataFrame()
    
    def calculate_kl_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """
        Изчислява KL Divergence между две разпределения
        
        Args:
            p, q: Probability distributions
        
        Returns:
            KL divergence стойност
        """
        try:
            # Добавя малка стойност за избягване на log(0)
            epsilon = 1e-10
            p = np.clip(p, epsilon, 1 - epsilon)
            q = np.clip(q, epsilon, 1 - epsilon)
            
            # Нормализира
            p = p / np.sum(p)
            q = q / np.sum(q)
            
            return np.sum(p * np.log(p / q))
            
        except Exception as e:
            self.logger.error(f"❌ Грешка при KL divergence: {e}")
            return float('inf')
    
    def calculate_psi(self, baseline: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
        """
        Изчислява Population Stability Index (PSI)
        
        Args:
            baseline: Baseline разпределение
            current: Текущо разпределение
            bins: Брой bins за дискретизация
        
        Returns:
            PSI стойност
        """
        try:
            # Създава bins базирани на baseline
            _, bin_edges = np.histogram(baseline, bins=bins)
            
            # Изчислява разпределенията
            baseline_counts, _ = np.histogram(baseline, bins=bin_edges)
            current_counts, _ = np.histogram(current, bins=bin_edges)
            
            # Нормализира
            baseline_pct = baseline_counts / len(baseline)
            current_pct = current_counts / len(current)
            
            # Избягва деление на 0
            epsilon = 1e-6
            baseline_pct = np.maximum(baseline_pct, epsilon)
            current_pct = np.maximum(current_pct, epsilon)
            
            # PSI формула
            psi = np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct))
            
            return psi
            
        except Exception as e:
            self.logger.error(f"❌ Грешка при PSI: {e}")
            return float('inf')
    
    def calculate_ece(self, y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
        """
        Изчислява Expected Calibration Error (ECE)
        
        Args:
            y_true: Реални резултати
            y_prob: Predicted probabilities
            n_bins: Брой bins за калибрация
        
        Returns:
            ECE стойност
        """
        try:
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            ece = 0
            total_samples = len(y_prob)
            
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                # Намира samples в този bin
                in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
                prop_in_bin = in_bin.mean()
                
                if prop_in_bin > 0:
                    accuracy_in_bin = y_true[in_bin].mean()
                    avg_confidence_in_bin = y_prob[in_bin].mean()
                    ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
            return ece
            
        except Exception as e:
            self.logger.error(f"❌ Грешка при ECE: {e}")
            return float('inf')
    
    def detect_prediction_drift(self, baseline_df: pd.DataFrame, 
                              current_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Открива prediction drift между baseline и current периоди
        
        Args:
            baseline_df: Baseline данни
            current_df: Текущи данни
        
        Returns:
            Drift анализ резултати
        """
        try:
            results = {
                'drift_detected': False,
                'drift_score': 0.0,
                'metrics': {},
                'components': {}
            }
            
            # Анализира всеки компонент
            components = ['poisson', 'ml', 'elo', 'ensemble']
            
            for component in components:
                pred_col = f'{component}_prediction'
                
                if pred_col not in baseline_df.columns or pred_col not in current_df.columns:
                    continue
                
                baseline_pred = baseline_df[pred_col].values
                current_pred = current_df[pred_col].values
                
                # KL Divergence
                try:
                    # Създава histogram bins
                    bins = np.linspace(0, 1, 21)  # 20 bins
                    baseline_hist, _ = np.histogram(baseline_pred, bins=bins, density=True)
                    current_hist, _ = np.histogram(current_pred, bins=bins, density=True)
                    
                    # Нормализира
                    baseline_hist = baseline_hist / np.sum(baseline_hist)
                    current_hist = current_hist / np.sum(current_hist)
                    
                    kl_div = self.calculate_kl_divergence(current_hist, baseline_hist)
                except Exception:
                    kl_div = 0.0
                
                # Jensen-Shannon Distance
                try:
                    js_dist = jensenshannon(baseline_hist, current_hist)
                except Exception:
                    js_dist = 0.0
                
                # Wasserstein Distance
                try:
                    wasserstein_dist = wasserstein_distance(baseline_pred, current_pred)
                except Exception:
                    wasserstein_dist = 0.0
                
                # PSI
                psi = self.calculate_psi(baseline_pred, current_pred)
                
                # Mean shift
                mean_shift = abs(np.mean(current_pred) - np.mean(baseline_pred))
                
                results['components'][component] = {
                    'kl_divergence': kl_div,
                    'jensen_shannon': js_dist,
                    'wasserstein': wasserstein_dist,
                    'psi': psi,
                    'mean_shift': mean_shift
                }
            
            # Общ drift score
            all_kl = [comp.get('kl_divergence', 0) for comp in results['components'].values()]
            all_js = [comp.get('jensen_shannon', 0) for comp in results['components'].values()]
            all_psi = [comp.get('psi', 0) for comp in results['components'].values()]
            
            results['metrics'] = {
                'avg_kl_divergence': np.mean(all_kl) if all_kl else 0,
                'max_kl_divergence': np.max(all_kl) if all_kl else 0,
                'avg_jensen_shannon': np.mean(all_js) if all_js else 0,
                'avg_psi': np.mean(all_psi) if all_psi else 0
            }
            
            # Проверява thresholds
            max_kl = results['metrics']['max_kl_divergence']
            avg_js = results['metrics']['avg_jensen_shannon']
            avg_psi = results['metrics']['avg_psi']
            
            drift_score = max(
                max_kl / self.thresholds['kl_divergence'],
                avg_js / self.thresholds['jensen_shannon'],
                avg_psi / self.thresholds['psi']
            )
            
            results['drift_score'] = drift_score
            results['drift_detected'] = drift_score > 1.0
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Грешка при prediction drift detection: {e}")
            return {'drift_detected': False, 'drift_score': 0.0, 'error': str(e)}
    
    def detect_calibration_drift(self, baseline_df: pd.DataFrame, 
                               current_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Открива calibration drift
        
        Args:
            baseline_df: Baseline данни
            current_df: Текущи данни
        
        Returns:
            Calibration drift анализ
        """
        try:
            results = {
                'drift_detected': False,
                'drift_score': 0.0,
                'components': {}
            }
            
            if 'actual_result' not in baseline_df.columns or 'actual_result' not in current_df.columns:
                return results
            
            components = ['poisson', 'ml', 'elo', 'ensemble']
            
            for component in components:
                pred_col = f'{component}_prediction'
                
                if pred_col not in baseline_df.columns or pred_col not in current_df.columns:
                    continue
                
                # ECE за baseline и current
                baseline_ece = self.calculate_ece(
                    baseline_df['actual_result'].values,
                    baseline_df[pred_col].values
                )
                
                current_ece = self.calculate_ece(
                    current_df['actual_result'].values,
                    current_df[pred_col].values
                )
                
                ece_change = abs(current_ece - baseline_ece)
                
                # Brier Score change
                baseline_brier = brier_score_loss(
                    baseline_df['actual_result'].values,
                    baseline_df[pred_col].values
                )
                
                current_brier = brier_score_loss(
                    current_df['actual_result'].values,
                    current_df[pred_col].values
                )
                
                brier_change = abs(current_brier - baseline_brier)
                
                results['components'][component] = {
                    'baseline_ece': baseline_ece,
                    'current_ece': current_ece,
                    'ece_change': ece_change,
                    'baseline_brier': baseline_brier,
                    'current_brier': current_brier,
                    'brier_change': brier_change
                }
            
            # Общ calibration drift score
            all_ece_changes = [comp.get('ece_change', 0) for comp in results['components'].values()]
            all_brier_changes = [comp.get('brier_change', 0) for comp in results['components'].values()]
            
            max_ece_change = np.max(all_ece_changes) if all_ece_changes else 0
            max_brier_change = np.max(all_brier_changes) if all_brier_changes else 0
            
            drift_score = max(
                max_ece_change / self.thresholds['ece_change'],
                max_brier_change / self.thresholds['brier_change']
            )
            
            results['drift_score'] = drift_score
            results['drift_detected'] = drift_score > 1.0
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Грешка при calibration drift detection: {e}")
            return {'drift_detected': False, 'drift_score': 0.0, 'error': str(e)}
    
    def detect_league_specific_drift(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Открива league-specific drift
        
        Args:
            df: DataFrame с данни
        
        Returns:
            League-specific drift анализ
        """
        try:
            results = {
                'drift_detected': False,
                'drift_score': 0.0,
                'leagues': {},
                'cross_league_consistency': 0.0
            }
            
            if 'league_slug' not in df.columns:
                return results
            
            # Разделя данните по периоди
            cutoff_date = df['timestamp'].max() - timedelta(days=self.analysis_window)
            baseline_df = df[df['timestamp'] < cutoff_date]
            current_df = df[df['timestamp'] >= cutoff_date]
            
            leagues = df['league_slug'].unique()
            league_drifts = {}
            
            for league in leagues:
                league_baseline = baseline_df[baseline_df['league_slug'] == league]
                league_current = current_df[current_df['league_slug'] == league]
                
                min_samples = self.config['drift_detection']['min_samples_per_league']
                if len(league_baseline) < min_samples or len(league_current) < min_samples:
                    continue
                
                # Prediction drift за тази лига
                league_drift = self.detect_prediction_drift(league_baseline, league_current)
                league_drifts[league] = league_drift['drift_score']
                
                results['leagues'][league] = {
                    'drift_score': league_drift['drift_score'],
                    'drift_detected': league_drift['drift_detected'],
                    'samples_baseline': len(league_baseline),
                    'samples_current': len(league_current)
                }
            
            if league_drifts:
                # Cross-league consistency
                drift_values = list(league_drifts.values())
                consistency = 1.0 - (np.std(drift_values) / (np.mean(drift_values) + 1e-6))
                results['cross_league_consistency'] = max(0, consistency)
                
                # Общ drift score
                max_league_drift = np.max(drift_values)
                consistency_penalty = 1.0 - results['cross_league_consistency']
                
                results['drift_score'] = max_league_drift * (1 + consistency_penalty)
                results['drift_detected'] = results['drift_score'] > self.thresholds['league_isolation']
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Грешка при league-specific drift detection: {e}")
            return {'drift_detected': False, 'drift_score': 0.0, 'error': str(e)}
    
    def generate_drift_report(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Генерира comprehensive drift отчет
        
        Args:
            analysis_results: Резултати от drift анализа
        
        Returns:
            Detailed drift report
        """
        try:
            report = {
                'timestamp': datetime.now().isoformat(),
                'analysis_period': {
                    'baseline_days': self.baseline_window,
                    'analysis_days': self.analysis_window
                },
                'overall_drift': {
                    'detected': False,
                    'severity': 'none',
                    'score': 0.0
                },
                'drift_types': {},
                'recommendations': [],
                'summary': {}
            }
            
            # Анализира всички типове drift
            all_scores = []
            
            for drift_type, results in analysis_results.items():
                if isinstance(results, dict) and 'drift_score' in results:
                    score = results['drift_score']
                    all_scores.append(score)
                    
                    # Определя severity
                    severity = 'none'
                    if score >= self.config['drift_detection']['severity_levels']['critical']:
                        severity = 'critical'
                    elif score >= self.config['drift_detection']['severity_levels']['high']:
                        severity = 'high'
                    elif score >= self.config['drift_detection']['severity_levels']['medium']:
                        severity = 'medium'
                    elif score >= self.config['drift_detection']['severity_levels']['low']:
                        severity = 'low'
                    
                    report['drift_types'][drift_type] = {
                        'detected': results.get('drift_detected', False),
                        'score': score,
                        'severity': severity,
                        'details': results
                    }
            
            # Общ drift score и severity
            if all_scores:
                overall_score = np.max(all_scores)
                report['overall_drift']['score'] = overall_score
                report['overall_drift']['detected'] = overall_score > 1.0
                
                # Общ severity
                if overall_score >= self.config['drift_detection']['severity_levels']['critical']:
                    report['overall_drift']['severity'] = 'critical'
                elif overall_score >= self.config['drift_detection']['severity_levels']['high']:
                    report['overall_drift']['severity'] = 'high'
                elif overall_score >= self.config['drift_detection']['severity_levels']['medium']:
                    report['overall_drift']['severity'] = 'medium'
                elif overall_score >= self.config['drift_detection']['severity_levels']['low']:
                    report['overall_drift']['severity'] = 'low'
            
            # Генерира препоръки
            recommendations = []
            
            if report['overall_drift']['severity'] == 'critical':
                recommendations.append("🚨 КРИТИЧЕН DRIFT: Незабавно retraining на моделите")
                recommendations.append("🔄 Активиране на emergency rollback процедури")
            elif report['overall_drift']['severity'] == 'high':
                recommendations.append("⚠️ ВИСОК DRIFT: Планиране на retraining в рамките на 24h")
                recommendations.append("📊 Увеличаване на мониторинг честотата")
            elif report['overall_drift']['severity'] == 'medium':
                recommendations.append("📈 СРЕДЕН DRIFT: Наблюдение и подготовка за retraining")
                recommendations.append("🔍 Анализ на причините за drift")
            
            report['recommendations'] = recommendations
            
            # Summary статистики
            report['summary'] = {
                'total_drift_types_analyzed': len(analysis_results),
                'drift_types_detected': sum(1 for r in report['drift_types'].values() if r['detected']),
                'max_drift_score': max(all_scores) if all_scores else 0,
                'avg_drift_score': np.mean(all_scores) if all_scores else 0
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"❌ Грешка при генериране на drift report: {e}")
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}
    
    def run_drift_analysis(self) -> Dict[str, Any]:
        """
        Стартира пълен drift анализ
        
        Returns:
            Comprehensive drift анализ резултати
        """
        if not self.config['drift_detection'].get('enabled', True):
            self.logger.info("🔒 Drift analysis е изключен")
            return {'enabled': False}
        
        self.logger.info("🔍 Започване на comprehensive drift analysis...")
        
        try:
            # Зарежда данни
            df = self.load_historical_data()
            
            if df.empty:
                self.logger.warning("❌ Няма данни за drift анализ")
                return {'error': 'No data available'}
            
            # Разделя данните на baseline и current
            cutoff_date = df['timestamp'].max() - timedelta(days=self.analysis_window)
            baseline_df = df[df['timestamp'] < cutoff_date]
            current_df = df[df['timestamp'] >= cutoff_date]
            
            if baseline_df.empty or current_df.empty:
                self.logger.warning("❌ Недостатъчно данни за сравнение")
                return {'error': 'Insufficient data for comparison'}
            
            self.logger.info(f"📊 Анализ: {len(baseline_df)} baseline vs {len(current_df)} current записа")
            
            # Различни типове drift анализ
            analysis_results = {}
            
            # 1. Prediction Drift
            self.logger.info("🔍 Анализ на Prediction Drift...")
            analysis_results['prediction_drift'] = self.detect_prediction_drift(baseline_df, current_df)
            
            # 2. Calibration Drift
            self.logger.info("🔍 Анализ на Calibration Drift...")
            analysis_results['calibration_drift'] = self.detect_calibration_drift(baseline_df, current_df)
            
            # 3. League-Specific Drift
            self.logger.info("🔍 Анализ на League-Specific Drift...")
            analysis_results['league_drift'] = self.detect_league_specific_drift(df)
            
            # Генерира comprehensive report
            drift_report = self.generate_drift_report(analysis_results)
            
            # Запазва отчета
            report_file = self.config['drift_detection']['reporting']['report_file']
            os.makedirs(os.path.dirname(report_file), exist_ok=True)
            
            # Конвертира numpy типове за JSON serialization
            def convert_for_json(obj):
                if isinstance(obj, np.bool_):
                    return bool(obj)
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {k: convert_for_json(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_for_json(item) for item in obj]
                return obj
            
            drift_report_json = convert_for_json(drift_report)
            
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(drift_report_json, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"💾 Drift report запазен в {report_file}")
            
            # Логва резултатите
            overall_severity = drift_report['overall_drift']['severity']
            overall_score = drift_report['overall_drift']['score']
            
            if overall_severity == 'critical':
                self.logger.critical(f"🚨 КРИТИЧЕН DRIFT открит: score={overall_score:.3f}")
            elif overall_severity == 'high':
                self.logger.warning(f"⚠️ ВИСОК DRIFT открит: score={overall_score:.3f}")
            elif overall_severity == 'medium':
                self.logger.info(f"📈 СРЕДЕН DRIFT открит: score={overall_score:.3f}")
            elif overall_severity == 'low':
                self.logger.info(f"📊 НИСЪК DRIFT открит: score={overall_score:.3f}")
            else:
                self.logger.info(f"✅ Няма значителен drift: score={overall_score:.3f}")
            
            return {
                'enabled': True,
                'success': True,
                'drift_report': drift_report,
                'analysis_results': analysis_results
            }
            
        except Exception as e:
            self.logger.error(f"❌ Грешка в drift analysis: {e}")
            return {'enabled': True, 'success': False, 'error': str(e)}


def run_drift_analysis() -> Dict[str, Any]:
    """
    Convenience функция за drift анализ
    
    Returns:
        Drift анализ резултати
    """
    analyzer = DriftAnalyzer()
    return analyzer.run_drift_analysis()


def main():
    """Главна функция за drift analysis"""
    logger = setup_logging()
    
    logger.info("🔍 СТАРТИРАНЕ НА ADVANCED DRIFT ANALYSIS")
    logger.info("=" * 70)
    
    try:
        analyzer = DriftAnalyzer()
        results = analyzer.run_drift_analysis()
        
        print("\n🔍 DRIFT ANALYSIS РЕЗУЛТАТИ:")
        print("=" * 60)
        
        if not results.get('enabled', True):
            print("🔒 Drift analysis е изключен")
            return
        
        if not results.get('success', False):
            error = results.get('error', 'Unknown error')
            print(f"❌ Drift analysis се провали: {error}")
            return
        
        drift_report = results.get('drift_report', {})
        overall_drift = drift_report.get('overall_drift', {})
        
        print(f"📊 Общ drift score: {overall_drift.get('score', 0):.3f}")
        print(f"🎯 Severity: {overall_drift.get('severity', 'none').upper()}")
        print(f"🔍 Drift detected: {'ДА' if overall_drift.get('detected', False) else 'НЕ'}")
        
        # Показва drift по типове
        drift_types = drift_report.get('drift_types', {})
        if drift_types:
            print("\n📈 DRIFT ПО ТИПОВЕ:")
            for drift_type, info in drift_types.items():
                status = "🔴" if info['detected'] else "🟢"
                print(f"   {status} {drift_type}: {info['score']:.3f} ({info['severity']})")
        
        # Показва препоръки
        recommendations = drift_report.get('recommendations', [])
        if recommendations:
            print("\n💡 ПРЕПОРЪКИ:")
            for rec in recommendations:
                print(f"   {rec}")
        
        logger.info("✅ Drift analysis завършен успешно")
        
    except Exception as e:
        logger.error(f"❌ Грешка в drift analysis: {e}")
        raise


if __name__ == "__main__":
    main()
