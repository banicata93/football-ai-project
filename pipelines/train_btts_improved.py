#!/usr/bin/env python3
"""
Improved BTTS Model Training Pipeline
Тренира подобрен BTTS модел с нови features и по-добра калибрация
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
from typing import Dict, Tuple

import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    brier_score_loss, log_loss, classification_report, confusion_matrix
)

from core.utils import setup_logging, save_json, PerformanceTimer
from core.ml_utils import prepare_features, evaluate_classification, get_feature_importance
from core.btts_features import BTTSFeatureEngineer


class ImprovedBTTSTrainer:
    """Подобрен BTTS модел тренировъчен pipeline"""
    
    def __init__(self):
        self.logger = setup_logging()
        self.feature_engineer = BTTSFeatureEngineer()
        
    def calculate_ece(self, y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
        """Изчислява Expected Calibration Error"""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_prob[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def evaluate_model_comprehensive(self, model, X: pd.DataFrame, y: np.ndarray, 
                                   model_name: str = "Model") -> Dict:
        """Comprehensive model evaluation"""
        
        # Predictions
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X)[:, 1]
        else:
            y_prob = model.predict(X)
        
        y_pred = (y_prob > 0.5).astype(int)
        
        # Basic metrics
        metrics = {
            'accuracy': float(accuracy_score(y, y_pred)),
            'precision_yes': float(precision_score(y, y_pred, pos_label=1)),
            'precision_no': float(precision_score(y, y_pred, pos_label=0)),
            'recall_yes': float(recall_score(y, y_pred, pos_label=1)),
            'recall_no': float(recall_score(y, y_pred, pos_label=0)),
            'f1_yes': float(f1_score(y, y_pred, pos_label=1)),
            'f1_no': float(f1_score(y, y_pred, pos_label=0)),
            'f1_macro': float(f1_score(y, y_pred, average='macro')),
            'brier_score': float(brier_score_loss(y, y_prob)),
            'log_loss': float(log_loss(y, y_prob)),
            'ece': float(self.calculate_ece(y, y_prob))
        }
        
        # Threshold analysis
        threshold_metrics = {}
        for threshold in [0.45, 0.5, 0.55, 0.6]:
            y_pred_thresh = (y_prob > threshold).astype(int)
            threshold_metrics[f'threshold_{threshold}'] = {
                'accuracy': float(accuracy_score(y, y_pred_thresh)),
                'f1_macro': float(f1_score(y, y_pred_thresh, average='macro')),
                'precision_yes': float(precision_score(y, y_pred_thresh, pos_label=1)),
                'recall_yes': float(recall_score(y, y_pred_thresh, pos_label=1))
            }
        
        # Find best threshold
        best_threshold = 0.5
        best_f1 = metrics['f1_macro']
        for thresh_name, thresh_metrics in threshold_metrics.items():
            if thresh_metrics['f1_macro'] > best_f1:
                best_f1 = thresh_metrics['f1_macro']
                best_threshold = float(thresh_name.split('_')[1])
        
        metrics['best_threshold'] = best_threshold
        metrics['best_f1_macro'] = best_f1
        metrics['threshold_analysis'] = threshold_metrics
        
        self.logger.info(f"\n--- {model_name.upper()} EVALUATION ---")
        self.logger.info(f"Accuracy: {metrics['accuracy']:.3f}")
        self.logger.info(f"F1 Macro: {metrics['f1_macro']:.3f}")
        self.logger.info(f"Brier Score: {metrics['brier_score']:.4f}")
        self.logger.info(f"ECE: {metrics['ece']:.4f}")
        self.logger.info(f"Best Threshold: {best_threshold} (F1: {best_f1:.3f})")
        
        return metrics
    
    def train_improved_btts_model(self, train_df: pd.DataFrame, val_df: pd.DataFrame, 
                                 config: Dict = None) -> Tuple[object, list, Dict]:
        """
        Тренира подобрен BTTS модел
        
        Args:
            train_df: Training data
            val_df: Validation data
            config: Model configuration
            
        Returns:
            Tuple (calibrated_model, feature_cols, metrics)
        """
        self.logger.info("=" * 70)
        self.logger.info("ТРЕНИРОВКА НА ПОДОБРЕН BTTS MODEL")
        self.logger.info("=" * 70)
        
        # Default config
        if config is None:
            config = {
                'n_estimators': 400,
                'max_depth': 7,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'eval_metric': 'logloss'
            }
        
        # Feature engineering
        self.logger.info("🔧 Прилагане на BTTS feature engineering...")
        train_enhanced = self.feature_engineer.create_btts_features(train_df)
        val_enhanced = self.feature_engineer.create_btts_features(val_df)
        
        # Feature selection - комбинира базови и BTTS features
        base_features = [
            'home_elo_before', 'away_elo_before', 'elo_diff',
            'home_form_5', 'away_form_5', 'form_diff_5',
            'home_goals_scored_avg_5', 'home_goals_conceded_avg_5',
            'away_goals_scored_avg_5', 'away_goals_conceded_avg_5',
            'home_shooting_efficiency', 'away_shooting_efficiency',
            'home_xg_proxy', 'away_xg_proxy',
            'is_home', 'is_weekend'
        ]
        
        # Добавя Poisson features ако са налични
        poisson_features = [
            'poisson_lambda_home', 'poisson_lambda_away',
            'poisson_prob_btts', 'poisson_expected_goals'
        ]
        
        for feature in poisson_features:
            if feature in train_enhanced.columns:
                base_features.append(feature)
        
        # Добавя BTTS-специфични features
        btts_features = self.feature_engineer.get_btts_feature_list()
        
        # Комбинира всички features
        all_features = base_features + btts_features
        
        # Филтрира налични features
        available_features = [f for f in all_features if f in train_enhanced.columns]
        
        self.logger.info(f"Базови features: {len(base_features)}")
        self.logger.info(f"BTTS features: {len(btts_features)}")
        self.logger.info(f"Общо налични features: {len(available_features)}")
        
        # Подготвя данните
        X_train = train_enhanced[available_features].fillna(0)
        X_val = val_enhanced[available_features].fillna(0)
        
        y_train = train_enhanced['btts'].values
        y_val = val_enhanced['btts'].values
        
        self.logger.info(f"Train samples: {len(X_train)} (Yes: {y_train.sum()}, No: {len(y_train)-y_train.sum()})")
        self.logger.info(f"Val samples: {len(X_val)} (Yes: {y_val.sum()}, No: {len(y_val)-y_val.sum()})")
        
        # Тренировка на базовия XGBoost модел
        self.logger.info("\n🚀 Тренировка на XGBoost...")
        
        with PerformanceTimer("XGBoost training", self.logger):
            base_model = xgb.XGBClassifier(**config, random_state=42)
            base_model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        
        # Оценка на базовия модел
        train_metrics_base = self.evaluate_model_comprehensive(
            base_model, X_train, y_train, "Base Model Train"
        )
        val_metrics_base = self.evaluate_model_comprehensive(
            base_model, X_val, y_val, "Base Model Validation"
        )
        
        # Калибрация на модела
        self.logger.info("\n🎯 Калибриране на модела...")
        
        with PerformanceTimer("Model calibration", self.logger):
            # Използва isotonic calibration
            calibrated_model = CalibratedClassifierCV(
                base_model, 
                method='isotonic',
                cv='prefit'  # Използва отделен validation set
            )
            calibrated_model.fit(X_val, y_val)
        
        # Оценка на калибрирания модел
        train_metrics_cal = self.evaluate_model_comprehensive(
            calibrated_model, X_train, y_train, "Calibrated Model Train"
        )
        val_metrics_cal = self.evaluate_model_comprehensive(
            calibrated_model, X_val, y_val, "Calibrated Model Validation"
        )
        
        # Feature importance
        self.logger.info("\n📊 Feature Importance (Top 20):")
        get_feature_importance(base_model, available_features, top_n=20)
        
        # Cross-validation на калибрирания модел
        self.logger.info("\n🔄 Cross-validation на калибрирания модел...")
        cv_scores = cross_val_score(
            CalibratedClassifierCV(xgb.XGBClassifier(**config, random_state=42), method='isotonic'),
            X_train, y_train, 
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring='f1_macro'
        )
        
        self.logger.info(f"CV F1 Macro: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        # Обобщение на резултатите
        improvement_summary = {
            'base_model': {
                'train': train_metrics_base,
                'validation': val_metrics_base
            },
            'calibrated_model': {
                'train': train_metrics_cal,
                'validation': val_metrics_cal
            },
            'improvements': {
                'brier_score_improvement': float(val_metrics_base['brier_score'] - val_metrics_cal['brier_score']),
                'ece_improvement': float(val_metrics_base['ece'] - val_metrics_cal['ece']),
                'f1_improvement': float(val_metrics_cal['f1_macro'] - val_metrics_base['f1_macro'])
            },
            'cross_validation': {
                'f1_macro_mean': float(cv_scores.mean()),
                'f1_macro_std': float(cv_scores.std())
            },
            'feature_info': {
                'total_features': len(available_features),
                'base_features': len(base_features),
                'btts_features': len(btts_features),
                'feature_list': available_features
            }
        }
        
        self.logger.info("\n🎉 ПОДОБРЕНИЯ:")
        self.logger.info(f"   Brier Score: {improvement_summary['improvements']['brier_score_improvement']:+.4f}")
        self.logger.info(f"   ECE: {improvement_summary['improvements']['ece_improvement']:+.4f}")
        self.logger.info(f"   F1 Macro: {improvement_summary['improvements']['f1_improvement']:+.3f}")
        
        return calibrated_model, available_features, improvement_summary
    
    def save_improved_model(self, model, feature_cols: list, metrics: Dict, 
                           model_path: str = 'models/model_btts_improved'):
        """Запазва подобрения модел"""
        
        os.makedirs(model_path, exist_ok=True)
        
        # Запазва модела
        model_file = f"{model_path}/btts_model_improved.pkl"
        joblib.dump(model, model_file)
        
        # Запазва feature columns
        feature_file = f"{model_path}/feature_columns.json"
        with open(feature_file, 'w') as f:
            json.dump({'features': feature_cols}, f, indent=2)
        
        # Запазва метрики
        metrics_file = f"{model_path}/training_metrics.json"
        save_json(metrics, metrics_file)
        
        # Запазва summary
        summary_file = f"{model_path}/model_summary.md"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("# Improved BTTS Model Summary\n\n")
            f.write(f"**Training Date:** {datetime.now().isoformat()}\n\n")
            
            cal_val = metrics['calibrated_model']['validation']
            f.write("## Performance Metrics\n")
            f.write(f"- **Accuracy:** {cal_val['accuracy']:.3f}\n")
            f.write(f"- **F1 Macro:** {cal_val['f1_macro']:.3f}\n")
            f.write(f"- **Brier Score:** {cal_val['brier_score']:.4f}\n")
            f.write(f"- **ECE:** {cal_val['ece']:.4f}\n")
            f.write(f"- **Best Threshold:** {cal_val['best_threshold']}\n\n")
            
            improvements = metrics['improvements']
            f.write("## Improvements vs Base Model\n")
            f.write(f"- **Brier Score:** {improvements['brier_score_improvement']:+.4f}\n")
            f.write(f"- **ECE:** {improvements['ece_improvement']:+.4f}\n")
            f.write(f"- **F1 Macro:** {improvements['f1_improvement']:+.3f}\n\n")
            
            feature_info = metrics['feature_info']
            f.write("## Feature Engineering\n")
            f.write(f"- **Total Features:** {feature_info['total_features']}\n")
            f.write(f"- **Base Features:** {feature_info['base_features']}\n")
            f.write(f"- **BTTS Features:** {feature_info['btts_features']}\n")
        
        self.logger.info(f"✅ Подобрен модел запазен в {model_path}")


def main():
    """Основна функция за тренировка на подобрен BTTS модел"""
    logger = setup_logging()
    
    logger.info("🚀 СТАРТИРАНЕ НА IMPROVED BTTS TRAINING")
    logger.info("=" * 60)
    
    # Зарежда данните
    train_path = 'data/processed/train_features.parquet'
    val_path = 'data/processed/val_features.parquet'
    
    if not os.path.exists(train_path) or not os.path.exists(val_path):
        logger.error("❌ Training/validation data не са намерени")
        return
    
    logger.info("📊 Зареждане на данни...")
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    
    if 'btts' not in train_df.columns:
        logger.error("❌ Няма 'btts' target колона")
        return
    
    logger.info(f"✓ Train data: {len(train_df)} matches")
    logger.info(f"✓ Val data: {len(val_df)} matches")
    
    # Тренировка
    trainer = ImprovedBTTSTrainer()
    model, features, metrics = trainer.train_improved_btts_model(train_df, val_df)
    
    # Запазване
    trainer.save_improved_model(model, features, metrics)
    
    logger.info("\n✅ Improved BTTS model тренировка завършена успешно!")


if __name__ == "__main__":
    main()
