#!/usr/bin/env python3
"""
League-Specific Dynamic Ensemble Optimizer

Автоматично изчислява оптимални ensemble тегла за всяка отделна лига,
базирано на production performance от последните 30-60 дни.
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import pandas as pd
import numpy as np
import yaml
import shutil
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import logging
from scipy.optimize import minimize
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import log_loss, brier_score_loss, accuracy_score
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

from core.utils import setup_logging


class LeagueEnsembleOptimizer:
    """
    League-Specific Dynamic Ensemble Optimizer
    """
    
    def __init__(self, config_path: str = "config/league_ensemble.yaml"):
        """
        Инициализация на LeagueEnsembleOptimizer
        
        Args:
            config_path: Път към конфигурационния файл
        """
        self.config_path = config_path
        self.config = self._load_config()
        
        # Основни настройки
        self.ensemble_config = self.config['league_ensembles']
        
        self.logger = self._setup_logging()
        self.lookback_days = self.ensemble_config['lookback_days']
        self.min_matches = self.ensemble_config['min_matches_per_league']
        
        # Weight constraints
        self.constraints = self.ensemble_config['constraints']
        
        # Backup директория
        self.backup_dir = Path(self.ensemble_config['backup']['backup_dir'])
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Output файлове
        self.weights_file = self.ensemble_config['output']['weights_file']
        self.results_file = self.ensemble_config['output']['results_file']
        
        self.logger.info("🎯 LeagueEnsembleOptimizer инициализиран")
    
    def _load_config(self) -> Dict:
        """Зарежда конфигурацията"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            # Fallback конфигурация
            return {
                'league_ensembles': {
                    'enabled': True,
                    'lookback_days': 60,
                    'min_matches_per_league': 150,
                    'min_improvement': 0.02,
                    'constraints': {'min_weight': 0.1, 'max_weight': 0.8},
                    'default_weights': {'poisson': 0.30, 'ml': 0.50, 'elo': 0.20},
                    'backup': {
                        'enabled': True,
                        'max_backups': 15,
                        'backup_dir': 'config/backups/'
                    },
                    'cross_validation': {
                        'enabled': True,
                        'folds': 5,
                        'validation_threshold': 0.01
                    },
                    'optimization': {
                        'method': 'SLSQP',
                        'max_iterations': 1000,
                        'tolerance': 1e-6,
                        'random_restarts': 3
                    },
                    'output': {
                        'weights_file': 'config/league_ensemble_weights.yaml',
                        'results_file': 'logs/league_ensemble_results.json'
                    },
                    'logging': {
                        'enabled': True,
                        'log_file': 'logs/league_ensemble_optimizer.log'
                    }
                }
            }
    
    def _setup_logging(self) -> logging.Logger:
        """Настройва logging"""
        logger = setup_logging()
        
        # Добавя file handler
        log_config = self.ensemble_config.get('logging', {})
        log_file = log_config.get('log_file', 'logs/league_ensemble_optimizer.log')
        
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        return logger
    
    def load_historical_data(self, days_back: int = None) -> pd.DataFrame:
        """
        Зарежда исторически данни за league ensemble optimization
        
        Args:
            days_back: Брой дни назад
        
        Returns:
            DataFrame с исторически данни
        """
        if days_back is None:
            days_back = self.lookback_days
        
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
            
            # Филтрира записи с реални резултати
            df = df[df['actual_result'].notna()].copy()
            
            self.logger.info(f"📊 Заредени {len(df)} записа за league ensemble optimization")
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Грешка при зареждане на данни: {e}")
            return pd.DataFrame()
    
    def evaluate_league_performance(self, df: pd.DataFrame, league_slug: str) -> Dict[str, Dict[str, float]]:
        """
        Оценява performance на компонентите за дадена лига
        
        Args:
            df: DataFrame с данни
            league_slug: Лига за анализ
        
        Returns:
            Performance метрики за всеки компонент
        """
        try:
            # Филтрира по лига
            league_df = df[df['league_slug'] == league_slug].copy()
            
            if len(league_df) < self.min_matches:
                self.logger.warning(f"⚠️ Недостатъчно данни за {league_slug}: {len(league_df)} < {self.min_matches}")
                return {}
            
            results = {}
            y_true = league_df['actual_result'].values
            
            components = ['poisson', 'ml', 'elo']
            
            for component in components:
                pred_col = f'{component}_prediction'
                
                if pred_col not in league_df.columns:
                    continue
                
                y_pred = league_df[pred_col].values
                
                # Проверява за валидни стойности
                if len(y_pred) == 0 or np.any(np.isnan(y_pred)):
                    continue
                
                # Clipping за log_loss
                y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)
                
                # Изчислява метрики
                try:
                    ll = log_loss(y_true, y_pred_clipped)
                    bs = brier_score_loss(y_true, y_pred)
                    acc = accuracy_score(y_true, (y_pred > 0.5).astype(int))
                    
                    results[component] = {
                        'log_loss': ll,
                        'brier_score': bs,
                        'accuracy': acc,
                        'samples': len(y_pred)
                    }
                    
                except Exception as e:
                    self.logger.error(f"❌ Грешка при метрики за {league_slug}/{component}: {e}")
                    continue
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Грешка при оценка на {league_slug}: {e}")
            return {}
    
    def _ensemble_predictions(self, df: pd.DataFrame, weights: Dict[str, float]) -> np.ndarray:
        """
        Изчислява ensemble прогнози с дадени тегла
        
        Args:
            df: DataFrame с компонентни прогнози
            weights: Тегла за компонентите
        
        Returns:
            Ensemble прогнози
        """
        ensemble_pred = np.zeros(len(df))
        
        for component, weight in weights.items():
            pred_col = f'{component}_prediction'
            if pred_col in df.columns:
                component_pred = df[pred_col].values
                ensemble_pred += weight * component_pred
        
        return ensemble_pred
    
    def _objective_function(self, weights_array: np.ndarray, df: pd.DataFrame, 
                          components: List[str]) -> float:
        """
        Objective function за оптимизация (минимизира log_loss)
        
        Args:
            weights_array: Тегла като numpy array
            df: DataFrame с данни
            components: Списък с компоненти
        
        Returns:
            Log loss стойност
        """
        # Конвертира array в dict
        weights = dict(zip(components, weights_array))
        
        # Изчислява ensemble прогнози
        y_pred = self._ensemble_predictions(df, weights)
        y_true = df['actual_result'].values
        
        # Clipping за log_loss
        y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)
        
        try:
            return log_loss(y_true, y_pred_clipped)
        except Exception:
            return float('inf')
    
    def optimize_league_weights(self, df: pd.DataFrame, league_slug: str) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Оптимизира ensemble теглата за дадена лига
        
        Args:
            df: DataFrame с данни
            league_slug: Лига за оптимизация
        
        Returns:
            Tuple от (нови тегла, метрики)
        """
        try:
            # Филтрира по лига
            league_df = df[df['league_slug'] == league_slug].copy()
            
            if len(league_df) < self.min_matches:
                return {}, {}
            
            # Компоненти за оптимизация
            available_components = []
            for comp in ['poisson', 'ml', 'elo']:
                if f'{comp}_prediction' in league_df.columns:
                    available_components.append(comp)
            
            if len(available_components) < 2:
                return {}, {}
            
            self.logger.info(f"🎯 Оптимизиране на тегла за {league_slug}: {available_components}")
            
            # Текущи тегла като starting point
            default_weights = self.ensemble_config['default_weights']
            initial_weights = np.array([
                default_weights.get(comp, 1.0/len(available_components)) 
                for comp in available_components
            ])
            
            # Нормализира началните тегла
            initial_weights = initial_weights / initial_weights.sum()
            
            # Ограничения
            min_weight = self.constraints['min_weight']
            max_weight = self.constraints['max_weight']
            
            # League-specific constraints ако има
            league_config = self.ensemble_config.get('leagues', {}).get(league_slug, {})
            custom_constraints = league_config.get('custom_constraints', {})
            if custom_constraints:
                min_weight = custom_constraints.get('min_weight', min_weight)
                max_weight = custom_constraints.get('max_weight', max_weight)
            
            bounds = [(min_weight, max_weight) for _ in available_components]
            
            # Constraint: сумата на теглата = 1
            constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}
            
            # Multiple random restarts за по-добра оптимизация
            best_result = None
            best_loss = float('inf')
            
            n_restarts = self.ensemble_config['optimization'].get('random_restarts', 3)
            
            for restart in range(n_restarts):
                # Random starting point
                if restart > 0:
                    start_weights = np.random.dirichlet(np.ones(len(available_components)))
                    start_weights = np.clip(start_weights, min_weight, max_weight)
                    start_weights = start_weights / start_weights.sum()
                else:
                    start_weights = initial_weights
                
                # Оптимизация с scipy
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    
                    result = minimize(
                        self._objective_function,
                        start_weights,
                        args=(league_df, available_components),
                        method=self.ensemble_config['optimization']['method'],
                        bounds=bounds,
                        constraints=constraints,
                        options={
                            'maxiter': self.ensemble_config['optimization']['max_iterations'],
                            'ftol': self.ensemble_config['optimization']['tolerance']
                        }
                    )
                
                if result.success and result.fun < best_loss:
                    best_result = result
                    best_loss = result.fun
            
            if best_result is None or not best_result.success:
                self.logger.warning(f"⚠️ Оптимизацията за {league_slug} не конвергира")
                return {}, {}
            
            # Нови тегла
            new_weights_array = best_result.x
            new_weights = dict(zip(available_components, new_weights_array))
            
            # Добавя липсващите компоненти с 0 тегло
            for comp in ['poisson', 'ml', 'elo']:
                if comp not in new_weights:
                    new_weights[comp] = 0.0
            
            # Изчислява метрики
            current_loss = self._objective_function(initial_weights, league_df, available_components)
            new_loss = best_result.fun
            
            improvement = (current_loss - new_loss) / current_loss if current_loss > 0 else 0
            
            metrics = {
                'current_log_loss': current_loss,
                'new_log_loss': new_loss,
                'improvement': improvement,
                'optimization_success': best_result.success,
                'samples': len(league_df)
            }
            
            self.logger.info(
                f"🎯 {league_slug}: log_loss {current_loss:.4f} → {new_loss:.4f} "
                f"(подобрение: {improvement:.1%})"
            )
            
            return new_weights, metrics
            
        except Exception as e:
            self.logger.error(f"❌ Грешка при оптимизация на {league_slug}: {e}")
            return {}, {}
    
    def cross_validate_league_weights(self, df: pd.DataFrame, league_slug: str, 
                                    weights: Dict[str, float]) -> bool:
        """
        Cross-validation на теглата за дадена лига
        
        Args:
            df: DataFrame с данни
            league_slug: Лига
            weights: Тегла за валидация
        
        Returns:
            True ако CV е успешна
        """
        try:
            # Филтрира по лига
            league_df = df[df['league_slug'] == league_slug].copy()
            
            if len(league_df) < 100:  # Минимум за CV
                return True  # Skip CV за малки datasets
            
            cv_config = self.ensemble_config['cross_validation']
            n_folds = cv_config['folds']
            
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
            cv_scores = []
            
            for train_idx, val_idx in kf.split(league_df):
                train_df = league_df.iloc[train_idx]
                val_df = league_df.iloc[val_idx]
                
                # Ensemble прогнози за validation set
                val_pred = self._ensemble_predictions(val_df, weights)
                val_true = val_df['actual_result'].values
                
                # Log loss за validation
                val_pred_clipped = np.clip(val_pred, 1e-15, 1 - 1e-15)
                cv_score = log_loss(val_true, val_pred_clipped)
                cv_scores.append(cv_score)
            
            mean_cv_score = np.mean(cv_scores)
            
            # Сравнява с default weights
            default_weights = self.ensemble_config['default_weights']
            default_pred = self._ensemble_predictions(league_df, default_weights)
            default_true = league_df['actual_result'].values
            default_pred_clipped = np.clip(default_pred, 1e-15, 1 - 1e-15)
            default_score = log_loss(default_true, default_pred_clipped)
            
            improvement = (default_score - mean_cv_score) / default_score
            validation_threshold = cv_config['validation_threshold']
            
            if improvement >= validation_threshold:
                self.logger.info(
                    f"✅ {league_slug} CV валидация успешна: {improvement:.1%} подобрение"
                )
                return True
            else:
                self.logger.warning(
                    f"❌ {league_slug} CV валидация неуспешна: {improvement:.1%} < {validation_threshold:.1%}"
                )
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Грешка при CV за {league_slug}: {e}")
            return False
    
    def backup_current_weights(self) -> str:
        """
        Създава backup на текущите league weights
        
        Returns:
            Път към backup файла
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_filename = f"league_weights_{timestamp}.yaml"
            backup_path = self.backup_dir / backup_filename
            
            # Копира текущия weights файл ако съществува
            if os.path.exists(self.weights_file):
                shutil.copy2(self.weights_file, backup_path)
            else:
                # Създава празен backup
                with open(backup_path, 'w') as f:
                    yaml.dump({}, f)
            
            self.logger.info(f"💾 League weights backup създаден: {backup_path}")
            
            # Почиства стари backup-и
            self._cleanup_old_backups()
            
            return str(backup_path)
            
        except Exception as e:
            self.logger.error(f"❌ Грешка при създаване на backup: {e}")
            return ""
    
    def _cleanup_old_backups(self):
        """Почиства стари backup файлове"""
        try:
            max_backups = self.ensemble_config['backup'].get('max_backups', 15)
            
            # Намира всички backup файлове
            backup_files = list(self.backup_dir.glob("league_weights_*.yaml"))
            
            # Сортира по дата (най-новите първи)
            backup_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            # Изтрива излишните backup-и
            for old_backup in backup_files[max_backups:]:
                old_backup.unlink()
                self.logger.info(f"🗑️ Изтрит стар league weights backup: {old_backup}")
                
        except Exception as e:
            self.logger.error(f"❌ Грешка при почистване на backup-и: {e}")
    
    def save_league_weights(self, league_weights: Dict[str, Dict[str, float]], 
                          metadata: Dict[str, Any]):
        """
        Запазва league ensemble weights
        
        Args:
            league_weights: Тегла по лиги
            metadata: Metadata за оптимизацията
        """
        try:
            # Подготвя данните за запис
            output_data = {
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'optimization_date': metadata.get('timestamp'),
                    'total_leagues': len(league_weights),
                    'lookback_days': self.lookback_days,
                    'min_matches_per_league': self.min_matches
                },
                'league_weights': league_weights
            }
            
            # Конвертира numpy типове
            def convert_numpy_types(obj):
                if isinstance(obj, np.bool_):
                    return bool(obj)
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                return obj
            
            output_data = convert_numpy_types(output_data)
            
            # Записва weights файла
            os.makedirs(os.path.dirname(self.weights_file), exist_ok=True)
            with open(self.weights_file, 'w', encoding='utf-8') as f:
                yaml.dump(output_data, f, default_flow_style=False, allow_unicode=True)
            
            self.logger.info(f"💾 League ensemble weights запазени в {self.weights_file}")
            
        except Exception as e:
            self.logger.error(f"❌ Грешка при запазване на league weights: {e}")
    
    def run_league_ensemble_optimization(self) -> Dict[str, Any]:
        """
        Стартира пълна league ensemble optimization
        
        Returns:
            Резултати от оптимизацията
        """
        if not self.ensemble_config.get('enabled', True):
            self.logger.info("🔒 League ensemble optimization е изключена")
            return {'enabled': False}
        
        self.logger.info("🎯 Започване на league ensemble optimization...")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'enabled': True,
            'success': False,
            'leagues_optimized': 0,
            'leagues_updated': 0,
            'league_results': {},
            'backup_path': ''
        }
        
        try:
            # 1. Зарежда исторически данни
            df = self.load_historical_data()
            
            if df.empty:
                self.logger.warning("❌ Няма данни за league ensemble optimization")
                results['error'] = 'No historical data'
                return results
            
            # 2. Създава backup
            backup_path = self.backup_current_weights()
            results['backup_path'] = backup_path
            
            # 3. Анализира лиги
            available_leagues = df['league_slug'].unique()
            league_weights = {}
            optimization_metadata = {}
            
            for league_slug in available_leagues:
                self.logger.info(f"🏆 Оптимизиране на {league_slug}...")
                
                # Оценява performance
                performance = self.evaluate_league_performance(df, league_slug)
                
                if not performance:
                    self.logger.warning(f"⚠️ Пропускане на {league_slug} - недостатъчно данни")
                    continue
                
                # Оптимизира тегла
                new_weights, metrics = self.optimize_league_weights(df, league_slug)
                
                if not new_weights or not metrics:
                    continue
                
                # Проверява минималното подобрение
                improvement = metrics.get('improvement', 0)
                min_improvement = self.ensemble_config['min_improvement']
                
                if improvement < min_improvement:
                    self.logger.info(f"📊 {league_slug}: недостатъчно подобрение {improvement:.1%} < {min_improvement:.1%}")
                    continue
                
                # Cross-validation
                if self.ensemble_config['cross_validation']['enabled']:
                    if not self.cross_validate_league_weights(df, league_slug, new_weights):
                        continue
                
                # Запазва резултатите
                league_weights[league_slug] = new_weights
                results['league_results'][league_slug] = {
                    'weights': new_weights,
                    'metrics': metrics,
                    'performance': performance
                }
                
                results['leagues_updated'] += 1
                
                self.logger.info(f"✅ {league_slug} оптимизиран успешно: {improvement:.1%} подобрение")
            
            results['leagues_optimized'] = len(available_leagues)
            
            # 4. Запазва league weights
            if league_weights:
                optimization_metadata = {
                    'timestamp': results['timestamp'],
                    'leagues_optimized': results['leagues_optimized'],
                    'leagues_updated': results['leagues_updated']
                }
                
                self.save_league_weights(league_weights, optimization_metadata)
                
                # Запазва резултатите
                os.makedirs(os.path.dirname(self.results_file), exist_ok=True)
                with open(self.results_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            
            results['success'] = True
            
            self.logger.info(
                f"✅ League ensemble optimization завършена: "
                f"{results['leagues_updated']}/{results['leagues_optimized']} лиги обновени"
            )
            
        except Exception as e:
            self.logger.error(f"❌ Грешка в league ensemble optimization: {e}")
            results['error'] = str(e)
        
        return results


def run_league_ensemble_optimization() -> Dict[str, Any]:
    """
    Convenience функция за league ensemble optimization
    
    Returns:
        Резултати от оптимизацията
    """
    optimizer = LeagueEnsembleOptimizer()
    return optimizer.run_league_ensemble_optimization()


def main():
    """Главна функция за league ensemble optimization"""
    logger = setup_logging()
    
    logger.info("🎯 СТАРТИРАНЕ НА LEAGUE ENSEMBLE OPTIMIZATION")
    logger.info("=" * 70)
    
    try:
        optimizer = LeagueEnsembleOptimizer()
        results = optimizer.run_league_ensemble_optimization()
        
        print("\n🎯 LEAGUE ENSEMBLE OPTIMIZATION РЕЗУЛТАТИ:")
        print("=" * 60)
        
        if not results.get('enabled', True):
            print("🔒 League ensemble optimization е изключена")
            return
        
        if results.get('success', False):
            print("✅ Оптимизацията завършена успешно")
            
            leagues_optimized = results.get('leagues_optimized', 0)
            leagues_updated = results.get('leagues_updated', 0)
            
            print(f"\n📊 СТАТИСТИКИ:")
            print(f"   Анализирани лиги: {leagues_optimized}")
            print(f"   Обновени лиги: {leagues_updated}")
            print(f"   Success rate: {leagues_updated/leagues_optimized*100:.1f}%" if leagues_optimized > 0 else "   Success rate: 0%")
            
            # Показва резултатите по лиги
            league_results = results.get('league_results', {})
            if league_results:
                print(f"\n🏆 ОБНОВЕНИ ЛИГИ:")
                for league, data in league_results.items():
                    weights = data['weights']
                    improvement = data['metrics']['improvement']
                    print(f"   {league}: {improvement:.1%} подобрение")
                    print(f"      Poisson: {weights.get('poisson', 0):.3f}")
                    print(f"      ML: {weights.get('ml', 0):.3f}")
                    print(f"      Elo: {weights.get('elo', 0):.3f}")
        else:
            error = results.get('error', 'Unknown error')
            print(f"❌ Оптимизацията се провали: {error}")
        
        logger.info("✅ League ensemble optimization завършен")
        
    except Exception as e:
        logger.error(f"❌ Грешка в league ensemble optimization: {e}")
        raise


if __name__ == "__main__":
    main()
