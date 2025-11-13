#!/usr/bin/env python3
"""
Dynamic Ensemble Optimizer за автоматично оптимизиране на ensemble weights

Анализира production резултати и автоматично коригира теглата между
Poisson, ML и Elo моделите за подобряване на ensemble performance.
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
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss, brier_score_loss, accuracy_score
import warnings

from core.utils import setup_logging


class EnsembleOptimizer:
    """
    Dynamic Ensemble Optimizer за автоматично оптимизиране на weights
    """
    
    def __init__(self, config_path: str = "config/ensemble_weights.yaml"):
        """
        Инициализация на EnsembleOptimizer
        
        Args:
            config_path: Път към конфигурационния файл
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.logger = self._setup_logging()
        
        # Текущи тегла
        self.current_weights = self.config['ensemble']['current_weights'].copy()
        
        # Оптимизация настройки
        self.opt_config = self.config['ensemble']['optimization']
        
        # Backup директория
        self.backup_dir = Path(self.config['ensemble']['backup']['backup_dir'])
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("🎯 EnsembleOptimizer инициализиран")
    
    def _load_config(self) -> Dict:
        """Зарежда конфигурацията"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            # Fallback конфигурация
            return {
                'ensemble': {
                    'current_weights': {'poisson': 0.30, 'ml': 0.50, 'elo': 0.20},
                    'optimization': {
                        'enabled': True,
                        'min_improvement': 0.02,
                        'lookback_days': 45,
                        'weight_constraints': {'min_weight': 0.1, 'max_weight': 0.8},
                        'cross_validation_folds': 5,
                        'validation_threshold': 0.01
                    },
                    'backup': {
                        'enabled': True,
                        'max_backups': 10,
                        'backup_dir': 'config/backups/'
                    },
                    'logging': {
                        'enabled': True,
                        'log_file': 'logs/ensemble_optimizer.log'
                    },
                    'history': {
                        'optimization_count': 0
                    }
                }
            }
    
    def _setup_logging(self) -> logging.Logger:
        """Настройва logging за ensemble optimizer"""
        logger = setup_logging()
        
        # Добавя file handler за ensemble optimizer
        log_config = self.config['ensemble'].get('logging', {})
        log_file = log_config.get('log_file', 'logs/ensemble_optimizer.log')
        
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        return logger
    
    def load_historical_predictions(self, days: int = None) -> pd.DataFrame:
        """
        Зарежда исторически прогнози и резултати
        
        Args:
            days: Брой дни назад (по подразбиране от config)
        
        Returns:
            DataFrame с исторически данни
        """
        if days is None:
            days = self.opt_config['lookback_days']
        
        try:
            # Пътя към prediction history файла
            history_file = "logs/predictions_history/ou25_predictions.jsonl"
            
            if not os.path.exists(history_file):
                self.logger.warning(f"❌ Не е намерен history файл: {history_file}")
                return pd.DataFrame()
            
            # Зарежда JSONL файла
            predictions = []
            cutoff_date = datetime.now() - timedelta(days=days)
            
            with open(history_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        
                        # Проверява дата
                        pred_date = datetime.fromisoformat(data.get('timestamp', ''))
                        if pred_date >= cutoff_date:
                            predictions.append(data)
                            
                    except (json.JSONDecodeError, ValueError) as e:
                        continue
            
            if not predictions:
                self.logger.warning(f"❌ Няма налични прогнози за последните {days} дни")
                return pd.DataFrame()
            
            # Конвертира в DataFrame
            df = pd.DataFrame(predictions)
            
            # Филтрира само записи с реални резултати
            df = df[df['actual_result'].notna()].copy()
            
            self.logger.info(f"📊 Заредени {len(df)} исторически прогнози за {days} дни")
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Грешка при зареждане на исторически данни: {e}")
            return pd.DataFrame()
    
    def evaluate_component_performance(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Оценява performance на отделните компоненти
        
        Args:
            df: DataFrame с исторически данни
        
        Returns:
            Метрики за всеки компонент
        """
        if df.empty:
            return {}
        
        try:
            results = {}
            
            # Извлича компонентните прогнози и реалните резултати
            y_true = df['actual_result'].values
            
            components = ['poisson', 'ml', 'elo']
            
            for component in components:
                if f'{component}_prediction' not in df.columns:
                    self.logger.warning(f"⚠️ Липсва {component}_prediction в данните")
                    continue
                
                y_pred = df[f'{component}_prediction'].values
                
                # Проверява за валидни стойности
                if len(y_pred) == 0 or np.any(np.isnan(y_pred)):
                    self.logger.warning(f"⚠️ Невалидни данни за {component}")
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
                    
                    self.logger.info(
                        f"📈 {component.upper()}: "
                        f"log_loss={ll:.4f}, brier_score={bs:.4f}, accuracy={acc:.3f}"
                    )
                    
                except Exception as e:
                    self.logger.error(f"❌ Грешка при изчисляване на метрики за {component}: {e}")
                    continue
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Грешка при оценка на компонентите: {e}")
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
            if f'{component}_prediction' in df.columns:
                component_pred = df[f'{component}_prediction'].values
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
    
    def optimize_weights(self, df: pd.DataFrame) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Оптимизира ensemble теглата
        
        Args:
            df: DataFrame с исторически данни
        
        Returns:
            Tuple от (нови тегла, метрики)
        """
        if df.empty:
            self.logger.warning("❌ Няма данни за оптимизация")
            return self.current_weights, {}
        
        try:
            # Компоненти за оптимизация
            available_components = []
            for comp in ['poisson', 'ml', 'elo']:
                if f'{comp}_prediction' in df.columns:
                    available_components.append(comp)
            
            if len(available_components) < 2:
                self.logger.warning("❌ Недостатъчно компоненти за оптимизация")
                return self.current_weights, {}
            
            self.logger.info(f"🎯 Оптимизиране на тегла за: {available_components}")
            
            # Текущи тегла като starting point
            initial_weights = np.array([
                self.current_weights.get(comp, 1.0/len(available_components)) 
                for comp in available_components
            ])
            
            # Нормализира началните тегла
            initial_weights = initial_weights / initial_weights.sum()
            
            # Ограничения
            min_weight = self.opt_config['weight_constraints']['min_weight']
            max_weight = self.opt_config['weight_constraints']['max_weight']
            
            bounds = [(min_weight, max_weight) for _ in available_components]
            
            # Constraint: сумата на теглата = 1
            constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}
            
            # Оптимизация с scipy
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                result = minimize(
                    self._objective_function,
                    initial_weights,
                    args=(df, available_components),
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={
                        'maxiter': self.opt_config.get('max_iterations', 1000),
                        'ftol': self.opt_config.get('tolerance', 1e-6)
                    }
                )
            
            if not result.success:
                self.logger.warning(f"⚠️ Оптимизацията не конвергира: {result.message}")
                return self.current_weights, {}
            
            # Нови тегла
            new_weights_array = result.x
            new_weights = dict(zip(available_components, new_weights_array))
            
            # Добавя липсващите компоненти с 0 тегло
            for comp in ['poisson', 'ml', 'elo']:
                if comp not in new_weights:
                    new_weights[comp] = 0.0
            
            # Изчислява метрики
            current_loss = self._objective_function(initial_weights, df, available_components)
            new_loss = result.fun
            
            improvement = (current_loss - new_loss) / current_loss
            
            metrics = {
                'current_log_loss': current_loss,
                'new_log_loss': new_loss,
                'improvement': improvement,
                'optimization_success': result.success,
                'iterations': result.nit if hasattr(result, 'nit') else 0
            }
            
            self.logger.info(
                f"🎯 Оптимизация завършена: "
                f"log_loss {current_loss:.4f} → {new_loss:.4f} "
                f"(подобрение: {improvement:.1%})"
            )
            
            return new_weights, metrics
            
        except Exception as e:
            self.logger.error(f"❌ Грешка при оптимизация на тегла: {e}")
            return self.current_weights, {}
    
    def validate_new_weights(self, df: pd.DataFrame, new_weights: Dict[str, float], 
                           metrics: Dict[str, float]) -> bool:
        """
        Валидира новите тегла
        
        Args:
            df: DataFrame с данни
            new_weights: Нови тегла
            metrics: Метрики от оптимизацията
        
        Returns:
            True ако теглата са валидни
        """
        try:
            # Проверява минималното подобрение
            min_improvement = self.opt_config['min_improvement']
            improvement = metrics.get('improvement', 0)
            
            if improvement < min_improvement:
                self.logger.info(
                    f"❌ Недостатъчно подобрение: {improvement:.1%} < {min_improvement:.1%}"
                )
                return False
            
            # Проверява сумата на теглата
            weights_sum = sum(new_weights.values())
            if abs(weights_sum - 1.0) > 1e-6:
                self.logger.warning(f"⚠️ Невалидна сума на тегла: {weights_sum:.6f}")
                return False
            
            # Проверява ограниченията
            min_weight = self.opt_config['weight_constraints']['min_weight']
            max_weight = self.opt_config['weight_constraints']['max_weight']
            
            for component, weight in new_weights.items():
                if weight > 0 and (weight < min_weight or weight > max_weight):
                    self.logger.warning(
                        f"⚠️ Тегло за {component} извън ограниченията: {weight:.3f}"
                    )
                    return False
            
            # Cross-validation ако има достатъчно данни
            if len(df) >= 100:
                cv_valid = self._cross_validate_weights(df, new_weights)
                if not cv_valid:
                    return False
            
            self.logger.info("✅ Новите тегла са валидни")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Грешка при валидация на тегла: {e}")
            return False
    
    def _cross_validate_weights(self, df: pd.DataFrame, weights: Dict[str, float]) -> bool:
        """
        Cross-validation на новите тегла
        
        Args:
            df: DataFrame с данни
            weights: Тегла за валидация
        
        Returns:
            True ако CV е успешна
        """
        try:
            n_folds = self.opt_config.get('cross_validation_folds', 5)
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
            
            cv_scores = []
            
            for train_idx, val_idx in kf.split(df):
                train_df = df.iloc[train_idx]
                val_df = df.iloc[val_idx]
                
                # Ensemble прогнози за validation set
                val_pred = self._ensemble_predictions(val_df, weights)
                val_true = val_df['actual_result'].values
                
                # Log loss за validation
                val_pred_clipped = np.clip(val_pred, 1e-15, 1 - 1e-15)
                cv_score = log_loss(val_true, val_pred_clipped)
                cv_scores.append(cv_score)
            
            mean_cv_score = np.mean(cv_scores)
            std_cv_score = np.std(cv_scores)
            
            # Сравнява с текущите тегла
            current_pred = self._ensemble_predictions(df, self.current_weights)
            current_true = df['actual_result'].values
            current_pred_clipped = np.clip(current_pred, 1e-15, 1 - 1e-15)
            current_score = log_loss(current_true, current_pred_clipped)
            
            improvement = (current_score - mean_cv_score) / current_score
            validation_threshold = self.opt_config.get('validation_threshold', 0.01)
            
            if improvement >= validation_threshold:
                self.logger.info(
                    f"✅ CV валидация успешна: {improvement:.1%} подобрение "
                    f"(CV score: {mean_cv_score:.4f} ± {std_cv_score:.4f})"
                )
                return True
            else:
                self.logger.warning(
                    f"❌ CV валидация неуспешна: {improvement:.1%} подобрение "
                    f"< {validation_threshold:.1%}"
                )
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Грешка при CV валидация: {e}")
            return False
    
    def backup_old_weights(self) -> str:
        """
        Създава backup на текущите тегла
        
        Returns:
            Път към backup файла
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_filename = f"ensemble_weights_{timestamp}.yaml"
            backup_path = self.backup_dir / backup_filename
            
            # Копира текущия config файл
            shutil.copy2(self.config_path, backup_path)
            
            self.logger.info(f"💾 Backup създаден: {backup_path}")
            
            # Почиства стари backup-и
            self._cleanup_old_backups()
            
            return str(backup_path)
            
        except Exception as e:
            self.logger.error(f"❌ Грешка при създаване на backup: {e}")
            return ""
    
    def _cleanup_old_backups(self):
        """Почиства стари backup файлове"""
        try:
            max_backups = self.config['ensemble']['backup'].get('max_backups', 10)
            
            # Намира всички backup файлове
            backup_files = list(self.backup_dir.glob("ensemble_weights_*.yaml"))
            
            # Сортира по дата (най-новите първи)
            backup_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            # Изтрива излишните backup-и
            for old_backup in backup_files[max_backups:]:
                old_backup.unlink()
                self.logger.info(f"🗑️ Изтрит стар backup: {old_backup}")
                
        except Exception as e:
            self.logger.error(f"❌ Грешка при почистване на backup-и: {e}")
    
    def update_weights_config(self, new_weights: Dict[str, float], 
                            metrics: Dict[str, float], backup_path: str):
        """
        Обновява конфигурацията с новите тегла
        
        Args:
            new_weights: Нови тегла
            metrics: Метрики от оптимизацията
            backup_path: Път към backup файла
        """
        try:
            # Обновява конфигурацията
            self.config['ensemble']['current_weights'] = new_weights.copy()
            
            # Обновява историята
            self.config['ensemble']['history']['last_optimization'] = datetime.now().isoformat()
            self.config['ensemble']['history']['last_weights_update'] = datetime.now().isoformat()
            self.config['ensemble']['history']['optimization_count'] = \
                self.config['ensemble']['history'].get('optimization_count', 0) + 1
            
            # Добавя метрики
            self.config['ensemble']['last_optimization_metrics'] = {
                'timestamp': datetime.now().isoformat(),
                'metrics': metrics,
                'backup_path': backup_path,
                'old_weights': self.current_weights.copy(),
                'new_weights': new_weights.copy()
            }
            
            # Конвертира numpy типове преди запис
            def convert_numpy_types(obj):
                """Конвертира numpy типове в Python типове"""
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
            
            config_to_save = convert_numpy_types(self.config)
            
            # Записва обновената конфигурация
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_to_save, f, default_flow_style=False, allow_unicode=True)
            
            # Обновява текущите тегла
            self.current_weights = new_weights.copy()
            
            self.logger.info("💾 Конфигурацията е обновена с новите тегла")
            
            # Логва промените
            for component, weight in new_weights.items():
                old_weight = self.current_weights.get(component, 0)
                change = weight - old_weight
                self.logger.info(f"🔄 {component}: {old_weight:.3f} → {weight:.3f} ({change:+.3f})")
            
        except Exception as e:
            self.logger.error(f"❌ Грешка при обновяване на конфигурацията: {e}")
    
    def optimize_ensemble_weights(self) -> Dict[str, Any]:
        """
        Пълен цикъл на оптимизация на ensemble тегла
        
        Returns:
            Резултати от оптимизацията
        """
        if not self.opt_config.get('enabled', True):
            self.logger.info("🔒 Ensemble optimization е изключена")
            return {'enabled': False}
        
        self.logger.info("🎯 Започване на ensemble weights optimization...")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'enabled': True,
            'success': False,
            'weights_updated': False,
            'old_weights': self.current_weights.copy(),
            'new_weights': {},
            'metrics': {},
            'backup_path': ''
        }
        
        try:
            # 1. Зарежда исторически данни
            df = self.load_historical_predictions()
            
            if df.empty:
                self.logger.warning("❌ Няма исторически данни за оптимизация")
                results['error'] = 'No historical data'
                return results
            
            # 2. Оценява компонентите
            component_performance = self.evaluate_component_performance(df)
            results['component_performance'] = component_performance
            
            # 3. Оптимизира теглата
            new_weights, opt_metrics = self.optimize_weights(df)
            results['new_weights'] = new_weights
            results['metrics'] = opt_metrics
            
            if not opt_metrics:
                self.logger.warning("❌ Оптимизацията се провали")
                results['error'] = 'Optimization failed'
                return results
            
            # 4. Валидира новите тегла
            if not self.validate_new_weights(df, new_weights, opt_metrics):
                self.logger.info("❌ Новите тегла не преминаха валидацията")
                results['error'] = 'Validation failed'
                return results
            
            # 5. Създава backup
            backup_path = self.backup_old_weights()
            results['backup_path'] = backup_path
            
            # 6. Обновява конфигурацията
            self.update_weights_config(new_weights, opt_metrics, backup_path)
            
            results['success'] = True
            results['weights_updated'] = True
            
            improvement = opt_metrics.get('improvement', 0)
            self.logger.info(
                f"✅ Ensemble optimization завършена успешно: "
                f"{improvement:.1%} подобрение в log_loss"
            )
            
        except Exception as e:
            self.logger.error(f"❌ Грешка в ensemble optimization: {e}")
            results['error'] = str(e)
        
        return results


def optimize_ensemble_weights() -> Dict[str, Any]:
    """
    Convenience функция за оптимизация на ensemble тегла
    
    Returns:
        Резултати от оптимизацията
    """
    optimizer = EnsembleOptimizer()
    return optimizer.optimize_ensemble_weights()


def main():
    """Главна функция за ensemble optimization"""
    logger = setup_logging()
    
    logger.info("🎯 СТАРТИРАНЕ НА ENSEMBLE WEIGHTS OPTIMIZATION")
    logger.info("=" * 70)
    
    try:
        # Инициализира optimizer
        optimizer = EnsembleOptimizer()
        
        # Стартира оптимизация
        results = optimizer.optimize_ensemble_weights()
        
        # Показва резултатите
        print("\n🎯 ENSEMBLE OPTIMIZATION РЕЗУЛТАТИ:")
        print("=" * 60)
        
        if not results['enabled']:
            print("🔒 Ensemble optimization е изключена")
            return
        
        if results['success']:
            print("✅ Оптимизацията завършена успешно")
            
            if results['weights_updated']:
                print("\n🔄 ПРОМЕНИ В ТЕГЛАТА:")
                old_weights = results['old_weights']
                new_weights = results['new_weights']
                
                for component in ['poisson', 'ml', 'elo']:
                    old_w = old_weights.get(component, 0)
                    new_w = new_weights.get(component, 0)
                    change = new_w - old_w
                    
                    print(f"   {component.upper()}: {old_w:.3f} → {new_w:.3f} ({change:+.3f})")
                
                metrics = results.get('metrics', {})
                improvement = metrics.get('improvement', 0)
                print(f"\n📈 Подобрение в log_loss: {improvement:.1%}")
            else:
                print("📊 Теглата не са променени (недостатъчно подобрение)")
        else:
            error = results.get('error', 'Unknown error')
            print(f"❌ Оптимизацията се провали: {error}")
        
        # Запазва резултатите
        results_file = "logs/ensemble_optimization_results.json"
        os.makedirs(os.path.dirname(results_file), exist_ok=True)
        
        # Конвертира numpy bool в Python bool за JSON serialization
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
        
        results_json = convert_for_json(results)
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results_json, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Резултати запазени в {results_file}")
        logger.info("✅ Ensemble optimization завършен")
        
    except Exception as e:
        logger.error(f"❌ Грешка в ensemble optimization: {e}")
        raise


if __name__ == "__main__":
    main()
