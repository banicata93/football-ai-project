#!/usr/bin/env python3
"""
Adaptive Learning Pipeline за OU2.5 Per-League Models

Автоматизиран модул за:
- Drift detection в per-league модели
- Incremental retraining при влошаване на performance
- Backup и rollback механизми
- Автоматично мониторинг и адаптация
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import pandas as pd
import numpy as np
import joblib
import shutil
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import yaml

from core.utils import setup_logging, load_config
from core.league_utils import LEAGUE_ID_TO_SLUG, get_league_display_name, get_per_league_model_path
from core.ml_utils import prepare_features, get_feature_columns, evaluate_classification
from sklearn.model_selection import train_test_split
from sklearn.isotonic import IsotonicRegression
import lightgbm as lgb


class AdaptiveTrainer:
    """
    Adaptive Learning система за per-league OU2.5 модели
    """
    
    def __init__(self, config_path: str = "config/adaptive_config.yaml"):
        """
        Инициализация на AdaptiveTrainer
        
        Args:
            config_path: Път към конфигурационния файл
        """
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        
        # Директории
        self.models_dir = Path(self.config['models_dir'])
        self.backup_dir = Path(self.config['backup_dir'])
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Метрики история
        self.metrics_history = {}
        self.current_metrics = {}
        
        self.logger.info("🤖 AdaptiveTrainer инициализиран")
    
    def _load_config(self, config_path: str) -> Dict:
        """Зарежда конфигурацията"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config['adaptive_learning']
        except Exception as e:
            # Fallback конфигурация
            return {
                'enabled': True,
                'drift_threshold': 0.05,
                'retrain_min_matches': 300,
                'retrain_window_days': 90,
                'backup_old_models': True,
                'log_file': 'logs/model_reports/ou25_per_league_summary.json',
                'models_dir': 'models/leagues/',
                'backup_dir': 'models/backups/',
                'adaptive_log': 'logs/adaptive_learning.log'
            }
    
    def _setup_logging(self) -> logging.Logger:
        """Настройва logging за adaptive learning"""
        logger = setup_logging()
        
        # Добавя file handler за adaptive learning
        log_file = self.config.get('adaptive_log', 'logs/adaptive_learning.log')
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        return logger
    
    def load_current_metrics(self) -> Dict:
        """
        Зарежда текущите метрики на моделите
        
        Returns:
            Речник с метрики по лиги
        """
        try:
            log_file = self.config['log_file']
            with open(log_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.current_metrics = data.get('metrics_by_league', {})
            self.logger.info(f"📊 Заредени метрики за {len(self.current_metrics)} лиги")
            return self.current_metrics
            
        except FileNotFoundError:
            self.logger.warning(f"❌ Не е намерен metrics файл: {self.config['log_file']}")
            return {}
        except Exception as e:
            self.logger.error(f"❌ Грешка при зареждане на метрики: {e}")
            return {}
    
    def load_metrics_history(self) -> Dict:
        """
        Зарежда историята на метриките
        
        Returns:
            История на метриките
        """
        history_file = "logs/adaptive_learning_history.json"
        
        try:
            with open(history_file, 'r', encoding='utf-8') as f:
                self.metrics_history = json.load(f)
            self.logger.info(f"📈 Заредена история за {len(self.metrics_history)} записа")
        except FileNotFoundError:
            self.logger.info("📈 Създаване на нова история на метриките")
            self.metrics_history = {}
        except Exception as e:
            self.logger.error(f"❌ Грешка при зареждане на история: {e}")
            self.metrics_history = {}
        
        return self.metrics_history
    
    def save_metrics_history(self):
        """Запазва историята на метриките"""
        history_file = "logs/adaptive_learning_history.json"
        
        try:
            # Добавя текущите метрики към историята
            timestamp = datetime.now().isoformat()
            self.metrics_history[timestamp] = {
                'metrics': self.current_metrics.copy(),
                'timestamp': timestamp
            }
            
            # Пази само последните 30 записа
            if len(self.metrics_history) > 30:
                sorted_keys = sorted(self.metrics_history.keys())
                for old_key in sorted_keys[:-30]:
                    del self.metrics_history[old_key]
            
            os.makedirs(os.path.dirname(history_file), exist_ok=True)
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(self.metrics_history, f, indent=2, ensure_ascii=False)
            
            self.logger.info("💾 История на метриките запазена")
            
        except Exception as e:
            self.logger.error(f"❌ Грешка при запазване на история: {e}")
    
    def detect_drift(self) -> List[str]:
        """
        Открива drift в per-league моделите
        
        Returns:
            Списък с лиги, които имат drift
        """
        if not self.config['enabled']:
            self.logger.info("🔒 Adaptive learning е изключен")
            return []
        
        self.logger.info("🔍 Започване на drift detection...")
        
        # Зарежда текущи и исторически метрики
        current = self.load_current_metrics()
        history = self.load_metrics_history()
        
        if not current:
            self.logger.warning("❌ Няма текущи метрики за анализ")
            return []
        
        if not history:
            self.logger.info("📊 Няма исторически данни - запазване на текущите метрики")
            self.save_metrics_history()
            return []
        
        # Намира последните исторически метрики
        last_timestamp = max(history.keys())
        last_metrics = history[last_timestamp]['metrics']
        
        drifted_leagues = []
        drift_threshold = self.config['drift_threshold']
        primary_metric = self.config['performance_metrics']['primary']
        
        for league_slug, current_metric in current.items():
            if league_slug not in last_metrics:
                self.logger.info(f"🆕 Нова лига: {league_slug}")
                continue
            
            last_metric = last_metrics[league_slug]
            
            # Сравнява primary метрика (log_loss)
            current_value = current_metric.get(primary_metric, 0)
            last_value = last_metric.get(primary_metric, 0)
            
            if last_value == 0:  # Избягва деление на нула
                continue
            
            # Изчислява относителна промяна
            change = (current_value - last_value) / last_value
            
            if change > drift_threshold:
                league_name = get_league_display_name(league_slug)
                self.logger.warning(
                    f"📉 DRIFT DETECTED: {league_name} - "
                    f"{primary_metric} влошен с {change:.1%} "
                    f"({last_value:.3f} → {current_value:.3f})"
                )
                drifted_leagues.append(league_slug)
            else:
                self.logger.info(
                    f"✅ {get_league_display_name(league_slug)}: "
                    f"{primary_metric} промяна {change:+.1%}"
                )
        
        # Запазва текущите метрики в историята
        self.save_metrics_history()
        
        self.logger.info(f"🔍 Drift detection завършен: {len(drifted_leagues)} лиги с drift")
        return drifted_leagues
    
    def backup_model(self, league_slug: str) -> str:
        """
        Създава backup на модел за лига
        
        Args:
            league_slug: Slug на лигата
        
        Returns:
            Път към backup директорията
        """
        try:
            # Създава backup директория
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = self.backup_dir / league_slug / f"ou25_backup_{timestamp}"
            backup_path.mkdir(parents=True, exist_ok=True)
            
            # Копира модела
            model_dir = get_per_league_model_path(league_slug, 'ou25', 'v1')
            
            if os.path.exists(model_dir):
                # Копира всички файлове от модела
                for file_name in os.listdir(model_dir):
                    src = os.path.join(model_dir, file_name)
                    dst = backup_path / file_name
                    shutil.copy2(src, dst)
                
                self.logger.info(f"💾 Backup създаден: {backup_path}")
                
                # Почиства стари backup-и
                self._cleanup_old_backups(league_slug)
                
                return str(backup_path)
            else:
                self.logger.warning(f"❌ Модел не съществува: {model_dir}")
                return ""
                
        except Exception as e:
            self.logger.error(f"❌ Грешка при backup на {league_slug}: {e}")
            return ""
    
    def _cleanup_old_backups(self, league_slug: str):
        """Почиства стари backup-и"""
        try:
            league_backup_dir = self.backup_dir / league_slug
            if not league_backup_dir.exists():
                return
            
            # Намира всички backup директории
            backups = [d for d in league_backup_dir.iterdir() if d.is_dir() and d.name.startswith('ou25_backup_')]
            
            # Сортира по дата (най-новите първи)
            backups.sort(key=lambda x: x.name, reverse=True)
            
            # Изтрива излишните backup-и
            max_backups = self.config.get('max_backups_per_league', 5)
            for old_backup in backups[max_backups:]:
                shutil.rmtree(old_backup)
                self.logger.info(f"🗑️ Изтрит стар backup: {old_backup}")
                
        except Exception as e:
            self.logger.error(f"❌ Грешка при почистване на backup-и: {e}")
    
    def rollback_model(self, league_slug: str, backup_path: str) -> bool:
        """
        Възстановява модел от backup
        
        Args:
            league_slug: Slug на лигата
            backup_path: Път към backup-а
        
        Returns:
            True ако rollback-ът е успешен
        """
        try:
            model_dir = get_per_league_model_path(league_slug, 'ou25', 'v1')
            backup_dir = Path(backup_path)
            
            if not backup_dir.exists():
                self.logger.error(f"❌ Backup не съществува: {backup_path}")
                return False
            
            # Изтрива текущия модел
            if os.path.exists(model_dir):
                shutil.rmtree(model_dir)
            
            # Възстановява от backup
            os.makedirs(model_dir, exist_ok=True)
            for file_name in os.listdir(backup_dir):
                src = backup_dir / file_name
                dst = os.path.join(model_dir, file_name)
                shutil.copy2(src, dst)
            
            self.logger.info(f"🔄 Rollback успешен за {league_slug}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Грешка при rollback на {league_slug}: {e}")
            return False
    
    def load_new_data(self, league_slug: str, days: int = 90) -> pd.DataFrame:
        """
        Зарежда нови данни за лига за последните N дни
        
        Args:
            league_slug: Slug на лигата
            days: Брой дни назад
        
        Returns:
            DataFrame с нови данни
        """
        try:
            # Намира league_id от slug
            league_id = None
            for lid, slug in LEAGUE_ID_TO_SLUG.items():
                if slug == league_slug:
                    league_id = lid
                    break
            
            if league_id is None:
                self.logger.error(f"❌ Не е намерен league_id за {league_slug}")
                return pd.DataFrame()
            
            # Зарежда всички данни (в реална система би било от база данни)
            data_files = [
                "data/processed/train_poisson_predictions.parquet",
                "data/processed/val_poisson_predictions.parquet",
                "data/processed/test_poisson_predictions.parquet"
            ]
            
            all_data = []
            for file_path in data_files:
                if os.path.exists(file_path):
                    df = pd.read_parquet(file_path)
                    all_data.append(df)
            
            if not all_data:
                self.logger.error("❌ Няма налични данни файлове")
                return pd.DataFrame()
            
            # Комбинира всички данни
            combined_df = pd.concat(all_data, ignore_index=True)
            
            # Филтрира по лига
            league_data = combined_df[combined_df['league_id'] == league_id].copy()
            
            # Филтрира по дата (ако има date колона)
            if 'date' in league_data.columns:
                cutoff_date = datetime.now() - timedelta(days=days)
                league_data['date'] = pd.to_datetime(league_data['date'], errors='coerce')
                recent_data = league_data[league_data['date'] >= cutoff_date]
            else:
                # Ако няма date колона, взема последните записи
                recent_data = league_data.tail(days * 5)  # Приблизително 5 мача на ден
            
            self.logger.info(f"📊 Заредени {len(recent_data)} нови записа за {league_slug}")
            return recent_data
            
        except Exception as e:
            self.logger.error(f"❌ Грешка при зареждане на данни за {league_slug}: {e}")
            return pd.DataFrame()
    
    def retrain_league_model(self, league_slug: str) -> bool:
        """
        Извършва incremental retraining на модел за лига
        
        Args:
            league_slug: Slug на лигата
        
        Returns:
            True ако retraining-ът е успешен
        """
        try:
            league_name = get_league_display_name(league_slug)
            self.logger.info(f"🔄 Започване на retraining за {league_name}...")
            
            # 1. Създава backup
            backup_path = self.backup_model(league_slug)
            if not backup_path:
                self.logger.error(f"❌ Неуспешен backup за {league_slug}")
                return False
            
            # 2. Зарежда нови данни
            new_data = self.load_new_data(
                league_slug, 
                self.config['retrain_window_days']
            )
            
            if len(new_data) < self.config['retrain_min_matches']:
                self.logger.warning(
                    f"❌ Недостатъчно нови данни за {league_slug}: "
                    f"{len(new_data)} < {self.config['retrain_min_matches']}"
                )
                return False
            
            # 3. Подготвя данни за обучение
            feature_cols = get_feature_columns()
            
            # Подготвя features
            X, _ = prepare_features(
                new_data, feature_cols, 
                use_intelligent_imputation=False, 
                legacy_fill_na=True
            )
            y = new_data['over_25'].values
            
            # Train/validation split
            validation_split = self.config.get('validation_split', 0.2)
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=validation_split, 
                random_state=42, stratify=y
            )
            
            self.logger.info(
                f"📊 Данни за обучение: {len(X_train)} train, {len(X_val)} val"
            )
            
            # 4. Тренира нов модел
            # Използва същите параметри като оригиналния модел
            lgb_params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'random_state': 42
            }
            
            # Създава LightGBM datasets
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            
            # Тренира модела
            model = lgb.train(
                lgb_params,
                train_data,
                valid_sets=[train_data, val_data],
                valid_names=['train', 'val'],
                num_boost_round=1000,
                callbacks=[
                    lgb.early_stopping(50),
                    lgb.log_evaluation(0)  # Тих режим
                ]
            )
            
            # 5. Валидация на новия модел
            y_val_pred = model.predict(X_val)
            y_val_pred_binary = (y_val_pred > 0.5).astype(int)
            
            val_metrics = evaluate_classification(
                y_val, y_val_pred_binary, y_val_pred,
                model_name=f"{league_slug}_retrained"
            )
            
            # Проверява дали новият модел е достатъчно добър
            new_accuracy = val_metrics['accuracy']
            min_accuracy = self.config['performance_metrics']['accuracy_threshold']
            
            if new_accuracy < min_accuracy:
                self.logger.warning(
                    f"❌ Нов модел за {league_slug} има нисък accuracy: "
                    f"{new_accuracy:.3f} < {min_accuracy:.3f}"
                )
                # Rollback
                if self.config.get('rollback_on_failure', True):
                    self.rollback_model(league_slug, backup_path)
                return False
            
            # 6. Калибрация
            calibrator = IsotonicRegression(out_of_bounds='clip')
            calibrator.fit(y_val_pred, y_val)
            
            # 7. Запазва новия модел
            model_dir = get_per_league_model_path(league_slug, 'ou25', 'v1')
            os.makedirs(model_dir, exist_ok=True)
            
            # Запазва модела
            model_file = os.path.join(model_dir, 'ou25_model.pkl')
            joblib.dump(model, model_file)
            
            # Запазва калибратора
            calibrator_file = os.path.join(model_dir, 'calibrator.pkl')
            joblib.dump(calibrator, calibrator_file)
            
            # Запазва feature columns
            feature_file = os.path.join(model_dir, 'feature_columns.json')
            with open(feature_file, 'w') as f:
                json.dump(feature_cols, f)
            
            # Запазва метрики
            metrics_file = os.path.join(model_dir, 'metrics.json')
            with open(metrics_file, 'w') as f:
                json.dump({
                    'retrained_at': datetime.now().isoformat(),
                    'validation_metrics': val_metrics,
                    'training_samples': len(X_train),
                    'validation_samples': len(X_val),
                    'backup_path': backup_path
                }, f, indent=2)
            
            self.logger.info(
                f"✅ Retraining успешен за {league_name}: "
                f"accuracy={new_accuracy:.3f}, "
                f"log_loss={val_metrics.get('log_loss', 0):.3f}"
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Грешка при retraining на {league_slug}: {e}")
            
            # Rollback при грешка
            if self.config.get('rollback_on_failure', True) and backup_path:
                self.rollback_model(league_slug, backup_path)
            
            return False
    
    def adaptive_learning_cycle(self) -> Dict[str, Any]:
        """
        Пълен цикъл на adaptive learning
        
        Returns:
            Резултати от цикъла
        """
        if not self.config['enabled']:
            self.logger.info("🔒 Adaptive learning е изключен")
            return {'enabled': False}
        
        self.logger.info("🤖 Започване на adaptive learning cycle...")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'enabled': True,
            'drifted_leagues': [],
            'retrained_leagues': [],
            'failed_retrains': [],
            'summary': {}
        }
        
        try:
            # 1. Drift detection
            drifted_leagues = self.detect_drift()
            results['drifted_leagues'] = drifted_leagues
            
            if not drifted_leagues:
                self.logger.info("✅ Няма открит drift - няма нужда от retraining")
                results['summary'] = {
                    'total_drifted': 0,
                    'total_retrained': 0,
                    'success_rate': 1.0
                }
                return results
            
            # 2. Retraining за drifted лиги
            max_concurrent = self.config.get('max_concurrent_retrains', 2)
            
            if len(drifted_leagues) <= max_concurrent:
                # Sequential retraining за малко лиги
                for league_slug in drifted_leagues:
                    success = self.retrain_league_model(league_slug)
                    if success:
                        results['retrained_leagues'].append(league_slug)
                    else:
                        results['failed_retrains'].append(league_slug)
            else:
                # Parallel retraining за много лиги
                with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
                    future_to_league = {
                        executor.submit(self.retrain_league_model, league): league
                        for league in drifted_leagues
                    }
                    
                    for future in as_completed(future_to_league):
                        league = future_to_league[future]
                        try:
                            success = future.result()
                            if success:
                                results['retrained_leagues'].append(league)
                            else:
                                results['failed_retrains'].append(league)
                        except Exception as e:
                            self.logger.error(f"❌ Грешка при parallel retraining на {league}: {e}")
                            results['failed_retrains'].append(league)
            
            # 3. Summary
            total_retrained = len(results['retrained_leagues'])
            total_failed = len(results['failed_retrains'])
            success_rate = total_retrained / len(drifted_leagues) if drifted_leagues else 1.0
            
            results['summary'] = {
                'total_drifted': len(drifted_leagues),
                'total_retrained': total_retrained,
                'total_failed': total_failed,
                'success_rate': success_rate
            }
            
            # Логва резултатите
            self.logger.info(
                f"🤖 Adaptive learning завършен: "
                f"{total_retrained}/{len(drifted_leagues)} успешни retraining-и "
                f"(success rate: {success_rate:.1%})"
            )
            
            if results['retrained_leagues']:
                retrained_names = [get_league_display_name(l) for l in results['retrained_leagues']]
                self.logger.info(f"✅ Retrained лиги: {', '.join(retrained_names)}")
            
            if results['failed_retrains']:
                failed_names = [get_league_display_name(l) for l in results['failed_retrains']]
                self.logger.warning(f"❌ Failed retrains: {', '.join(failed_names)}")
            
        except Exception as e:
            self.logger.error(f"❌ Грешка в adaptive learning cycle: {e}")
            results['error'] = str(e)
        
        return results


def main():
    """Главна функция за adaptive learning"""
    logger = setup_logging()
    
    logger.info("🤖 СТАРТИРАНЕ НА ADAPTIVE LEARNING")
    logger.info("=" * 60)
    
    try:
        # Инициализира adaptive trainer
        trainer = AdaptiveTrainer()
        
        # Стартира adaptive learning cycle
        results = trainer.adaptive_learning_cycle()
        
        # Показва резултатите
        print("\n🤖 ADAPTIVE LEARNING РЕЗУЛТАТИ:")
        print("=" * 50)
        
        if not results['enabled']:
            print("🔒 Adaptive learning е изключен")
            return
        
        summary = results.get('summary', {})
        print(f"📊 Общо лиги с drift: {summary.get('total_drifted', 0)}")
        print(f"✅ Успешни retraining-и: {summary.get('total_retrained', 0)}")
        print(f"❌ Неуспешни retraining-и: {summary.get('total_failed', 0)}")
        print(f"📈 Success rate: {summary.get('success_rate', 0):.1%}")
        
        if results.get('retrained_leagues'):
            print(f"\n🔄 Retrained лиги:")
            for league_slug in results['retrained_leagues']:
                print(f"   ✅ {get_league_display_name(league_slug)}")
        
        if results.get('failed_retrains'):
            print(f"\n❌ Failed retrains:")
            for league_slug in results['failed_retrains']:
                print(f"   ❌ {get_league_display_name(league_slug)}")
        
        # Запазва резултатите
        results_file = "logs/adaptive_learning_results.json"
        os.makedirs(os.path.dirname(results_file), exist_ok=True)
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Резултати запазени в {results_file}")
        logger.info("✅ Adaptive learning завършен успешно")
        
    except Exception as e:
        logger.error(f"❌ Грешка в adaptive learning: {e}")
        raise


if __name__ == "__main__":
    main()
