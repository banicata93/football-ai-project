"""
Training pipeline за всички ML модели
"""

import sys
import os
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import joblib
from datetime import datetime

import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV

from core.ml_utils import (
    get_feature_columns, prepare_features, evaluate_classification,
    get_feature_importance, encode_target_1x2, ModelTracker
)
from core.utils import setup_logging, load_config, save_json, PerformanceTimer
from core.league_utils import get_league_slug, get_supported_leagues, get_per_league_model_path


def train_1x2_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    config: dict
) -> tuple:
    """
    Тренировка на 1X2 модел (XGBoost multi-class)
    
    Args:
        train_df: Train dataset
        val_df: Validation dataset
        config: Конфигурация
    
    Returns:
        Tuple (model, feature_cols, metrics)
    """
    logger = setup_logging()
    
    logger.info("=" * 70)
    logger.info("ТРЕНИРОВКА НА 1X2 MODEL (XGBoost)")
    logger.info("=" * 70)
    
    # Feature columns
    feature_cols = get_feature_columns()
    
    # Prepare features
    X_train = prepare_features(train_df, feature_cols)
    X_val = prepare_features(val_df, feature_cols)
    
    # Encode target
    y_train, label_map = encode_target_1x2(train_df['result'])
    y_val, _ = encode_target_1x2(val_df['result'])
    
    logger.info(f"Features: {len(X_train.columns)}")
    logger.info(f"Train samples: {len(X_train)}")
    logger.info(f"Val samples: {len(X_val)}")
    
    # Model config
    model_config = config.get('model_1x2', {})
    params = model_config.get('hyperparameters', {})
    
    # Train XGBoost
    logger.info("\nТренировка на XGBoost...")
    
    with PerformanceTimer("XGBoost training", logger):
        model = xgb.XGBClassifier(**params, random_state=42)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_train_proba = model.predict_proba(X_train)
    
    y_val_pred = model.predict(X_val)
    y_val_proba = model.predict_proba(X_val)
    
    # Evaluation
    logger.info("\n--- TRAIN SET ---")
    train_metrics = evaluate_classification(
        y_train, y_train_pred, y_train_proba,
        labels=['1', 'X', '2'], model_name="1X2 Train"
    )
    
    logger.info("\n--- VALIDATION SET ---")
    val_metrics = evaluate_classification(
        y_val, y_val_pred, y_val_proba,
        labels=['1', 'X', '2'], model_name="1X2 Val"
    )
    
    # Feature importance
    get_feature_importance(model, X_train.columns.tolist(), top_n=15)
    
    return model, X_train.columns.tolist(), {'train': train_metrics, 'val': val_metrics}


def train_ou25_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    config: dict
) -> tuple:
    """
    Тренировка на Over/Under 2.5 модел (LightGBM binary)
    
    Args:
        train_df: Train dataset
        val_df: Validation dataset
        config: Конфигурация
    
    Returns:
        Tuple (model, feature_cols, metrics)
    """
    logger = setup_logging()
    
    logger.info("=" * 70)
    logger.info("ТРЕНИРОВКА НА OVER/UNDER 2.5 MODEL (LightGBM)")
    logger.info("=" * 70)
    
    # Feature columns
    feature_cols = get_feature_columns()
    
    # Prepare features
    X_train = prepare_features(train_df, feature_cols)
    X_val = prepare_features(val_df, feature_cols)
    
    # Target
    y_train = train_df['over_25'].values
    y_val = val_df['over_25'].values
    
    logger.info(f"Features: {len(X_train.columns)}")
    logger.info(f"Train samples: {len(X_train)} (Over: {y_train.sum()}, Under: {len(y_train)-y_train.sum()})")
    logger.info(f"Val samples: {len(X_val)} (Over: {y_val.sum()}, Under: {len(y_val)-y_val.sum()})")
    
    # Model config
    model_config = config.get('model_ou25', {})
    params = model_config.get('hyperparameters', {})
    
    # Train LightGBM
    logger.info("\nТренировка на LightGBM...")
    
    with PerformanceTimer("LightGBM training", logger):
        model = lgb.LGBMClassifier(**params, random_state=42, verbose=-1)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_train_proba = model.predict_proba(X_train)[:, 1]
    
    y_val_pred = model.predict(X_val)
    y_val_proba = model.predict_proba(X_val)[:, 1]
    
    # Evaluation
    logger.info("\n--- TRAIN SET ---")
    train_metrics = evaluate_classification(
        y_train, y_train_pred, y_train_proba,
        labels=['Under 2.5', 'Over 2.5'], model_name="OU2.5 Train"
    )
    
    logger.info("\n--- VALIDATION SET ---")
    val_metrics = evaluate_classification(
        y_val, y_val_pred, y_val_proba,
        labels=['Under 2.5', 'Over 2.5'], model_name="OU2.5 Val"
    )
    
    # Feature importance
    get_feature_importance(model, X_train.columns.tolist(), top_n=15)
    
    return model, X_train.columns.tolist(), {'train': train_metrics, 'val': val_metrics}


def train_btts_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    config: dict
) -> tuple:
    """
    Тренировка на BTTS модел (XGBoost binary) - LEGACY VERSION
    
    Args:
        train_df: Train dataset
        val_df: Validation dataset
        config: Конфигурация
    
    Returns:
        Tuple (model, feature_cols, metrics)
    """
    logger = setup_logging()
    
    logger.info("=" * 70)
    logger.info("ТРЕНИРОВКА НА BTTS MODEL (XGBoost)")
    logger.info("=" * 70)
    
    # Feature columns
    feature_cols = get_feature_columns()
    
    # Prepare features
    X_train = prepare_features(train_df, feature_cols)
    X_val = prepare_features(val_df, feature_cols)
    
    # Target
    y_train = train_df['btts'].values
    y_val = val_df['btts'].values
    
    logger.info(f"Features: {len(X_train.columns)}")
    logger.info(f"Train samples: {len(X_train)} (Yes: {y_train.sum()}, No: {len(y_train)-y_train.sum()})")
    logger.info(f"Val samples: {len(X_val)} (Yes: {y_val.sum()}, No: {len(y_val)-y_val.sum()})")
    
    # Model config
    model_config = config.get('model_btts', {})
    params = model_config.get('hyperparameters', {})
    
    # Train XGBoost
    logger.info("\nТренировка на XGBoost...")
    
    with PerformanceTimer("XGBoost training", logger):
        model = xgb.XGBClassifier(**params, random_state=42)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_train_proba = model.predict_proba(X_train)[:, 1]
    
    y_val_pred = model.predict(X_val)
    y_val_proba = model.predict_proba(X_val)[:, 1]
    
    # Evaluation
    logger.info("\n--- TRAIN SET ---")
    train_metrics = evaluate_classification(
        y_train, y_train_pred, y_train_proba,
        labels=['No', 'Yes'], model_name="BTTS Train"
    )
    
    logger.info("\n--- VALIDATION SET ---")
    val_metrics = evaluate_classification(
        y_val, y_val_pred, y_val_proba,
        labels=['No', 'Yes'], model_name="BTTS Val"
    )
    
    # Feature importance
    get_feature_importance(model, X_train.columns.tolist(), top_n=15)
    
    return model, X_train.columns.tolist(), {'train': train_metrics, 'val': val_metrics}


def train_btts_model_v2(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    config: dict
) -> tuple:
    """
    IMPROVED BTTS MODEL V2 с:
    - BTTS-specific features
    - Calibration (Isotonic Regression)
    - Improved XGBoost parameters
    - Better regularization
    
    Args:
        train_df: Train dataset
        val_df: Validation dataset
        config: Конфигурация
    
    Returns:
        Tuple (calibrated_model, feature_cols, metrics)
    """
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import brier_score_loss
    from core.ml_utils import add_btts_specific_features, get_btts_feature_columns
    
    logger = setup_logging()
    
    logger.info("=" * 70)
    logger.info("IMPROVED BTTS MODEL V2 TRAINING")
    logger.info("=" * 70)
    
    # Add BTTS-specific features
    logger.info("\n[Step 1] Adding BTTS-specific features...")
    train_df = add_btts_specific_features(train_df)
    val_df = add_btts_specific_features(val_df)
    
    # Feature columns (with BTTS-specific)
    feature_cols = get_btts_feature_columns()
    
    # Prepare features
    X_train = prepare_features(train_df, feature_cols)
    X_val = prepare_features(val_df, feature_cols)
    
    # Target
    y_train = train_df['btts'].values
    y_val = val_df['btts'].values
    
    logger.info(f"Features: {len(X_train.columns)}")
    logger.info(f"Train: {len(X_train)} (Yes: {y_train.sum()} [{y_train.mean()*100:.1f}%], No: {len(y_train)-y_train.sum()} [{(1-y_train.mean())*100:.1f}%])")
    logger.info(f"Val: {len(X_val)} (Yes: {y_val.sum()} [{y_val.mean()*100:.1f}%], No: {len(y_val)-y_val.sum()} [{(1-y_val.mean())*100:.1f}%])")
    
    # Improved XGBoost parameters
    logger.info("\n[Step 2] Training XGBoost with improved parameters...")
    xgb_params = {
        'n_estimators': 350,
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_lambda': 1.2,
        'random_state': 42,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss'
    }
    
    with PerformanceTimer("XGBoost training", logger):
        base_model = xgb.XGBClassifier(**xgb_params)
        base_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
    
    # Get uncalibrated predictions
    y_train_proba_uncal = base_model.predict_proba(X_train)[:, 1]
    y_val_proba_uncal = base_model.predict_proba(X_val)[:, 1]
    
    logger.info(f"\nUncalibrated Val Proba: mean={y_val_proba_uncal.mean():.3f}, std={y_val_proba_uncal.std():.3f}")
    logger.info(f"Actual Val Rate: {y_val.mean():.3f}")
    
    # Calibration with Isotonic Regression
    logger.info("\n[Step 3] Calibrating with Isotonic Regression...")
    calibrated_model = CalibratedClassifierCV(
        base_model,
        method='isotonic',
        cv='prefit'  # Use validation set for calibration
    )
    calibrated_model.fit(X_val, y_val)
    
    # Calibrated predictions
    y_train_proba = calibrated_model.predict_proba(X_train)[:, 1]
    y_val_proba = calibrated_model.predict_proba(X_val)[:, 1]
    
    y_train_pred = (y_train_proba > 0.5).astype(int)
    y_val_pred = (y_val_proba > 0.5).astype(int)
    
    logger.info(f"\nCalibrated Val Proba: mean={y_val_proba.mean():.3f}, std={y_val_proba.std():.3f}")
    
    # Evaluation
    logger.info("\n[Step 4] Evaluation...")
    logger.info("\n--- TRAIN SET ---")
    train_metrics = evaluate_classification(
        y_train, y_train_pred, y_train_proba,
        labels=['No', 'Yes'], model_name="BTTS V2 Train"
    )
    train_metrics['brier_score'] = brier_score_loss(y_train, y_train_proba)
    logger.info(f"Brier Score: {train_metrics['brier_score']:.4f}")
    
    logger.info("\n--- VALIDATION SET ---")
    val_metrics = evaluate_classification(
        y_val, y_val_pred, y_val_proba,
        labels=['No', 'Yes'], model_name="BTTS V2 Val"
    )
    val_metrics['brier_score'] = brier_score_loss(y_val, y_val_proba)
    logger.info(f"Brier Score: {val_metrics['brier_score']:.4f}")
    
    # Calibration check
    logger.info("\n--- CALIBRATION CHECK ---")
    for threshold in [0.4, 0.5, 0.6, 0.7]:
        mask = y_val_proba >= threshold
        if mask.sum() > 0:
            actual_rate = y_val[mask].mean()
            logger.info(f"Prob >= {threshold:.1f}: {mask.sum():4d} samples, actual Yes rate: {actual_rate*100:.1f}%")
    
    # Feature importance
    logger.info("\n--- FEATURE IMPORTANCE ---")
    get_feature_importance(base_model, X_train.columns.tolist(), top_n=20)
    
    logger.info("\n" + "=" * 70)
    logger.info("✓ BTTS V2 MODEL TRAINING COMPLETED")
    logger.info("=" * 70)
    
    return calibrated_model, X_train.columns.tolist(), {'train': train_metrics, 'val': val_metrics}


def save_model(
    model: any,
    feature_cols: list,
    metrics: dict,
    model_name: str,
    output_dir: str
):
    """
    Запазване на модел
    
    Args:
        model: Обучен модел
        feature_cols: Feature колони
        metrics: Метрики
        model_name: Име на модела
        output_dir: Директория
    """
    logger = setup_logging()
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Запазване на модела
    model_path = os.path.join(output_dir, f"{model_name}_model.pkl")
    joblib.dump(model, model_path)
    logger.info(f"Модел запазен: {model_path}")
    
    # Запазване на feature columns
    features_path = os.path.join(output_dir, "feature_columns.json")
    save_json({'features': feature_cols}, features_path)
    
    # Запазване на метрики
    metrics_path = os.path.join(output_dir, "metrics.json")
    save_json(metrics, metrics_path)
    logger.info(f"Метрики запазени: {metrics_path}")
    
    # Model info
    model_info = {
        'model_name': model_name,
        'model_type': type(model).__name__,
        'trained_date': datetime.now().isoformat(),
        'num_features': len(feature_cols),
        'train_accuracy': metrics['train'].get('accuracy', 0),
        'val_accuracy': metrics['val'].get('accuracy', 0)
    }
    
    info_path = os.path.join(output_dir, "model_info.json")
    save_json(model_info, info_path)


def train_ou25_per_league(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    config: dict,
    league_slug: str,
    min_matches: int = 2000
) -> tuple:
    """
    Тренировка на OU2.5 модел за конкретна лига
    
    Args:
        train_df: Train dataset
        val_df: Validation dataset
        config: Конфигурация
        league_slug: League slug (premier_league, la_liga, etc.)
        min_matches: Минимален брой мачове за тренировка
    
    Returns:
        Tuple (model, calibrator, feature_cols, metrics) или None ако няма достатъчно данни
    """
    logger = setup_logging()
    
    # Филтрира данни по лига (предполагаме че има league_slug колона)
    if 'league_slug' not in train_df.columns:
        # Fallback - използваме league_id ако няма league_slug
        if 'league_id' in train_df.columns:
            # Мапираме league_id към slug
            from core.league_utils import LEAGUE_ID_TO_SLUG
            train_df['league_slug'] = train_df['league_id'].map(LEAGUE_ID_TO_SLUG)
            val_df['league_slug'] = val_df['league_id'].map(LEAGUE_ID_TO_SLUG)
        else:
            logger.warning(f"No league information found for per-league training")
            return None
    
    # Филтрира по лига
    train_league = train_df[train_df['league_slug'] == league_slug].copy()
    val_league = val_df[val_df['league_slug'] == league_slug].copy()
    
    total_matches = len(train_league) + len(val_league)
    
    if total_matches < min_matches:
        logger.info(f"❌ {league_slug}: {total_matches} мача < {min_matches} (минимум)")
        return None
    
    # Ако няма validation данни, използваме част от train
    if len(val_league) == 0:
        logger.info(f"⚠️ {league_slug}: Няма val данни, използваме 30% от train")
        from sklearn.model_selection import train_test_split
        train_league, val_league = train_test_split(
            train_league, test_size=0.3, random_state=42, 
            stratify=train_league['over_25']
        )
    
    logger.info(f"🏆 {league_slug.upper()}: {len(train_league)} train + {len(val_league)} val = {total_matches} мача")
    
    # Feature columns (същите като глобалния модел)
    global_feature_cols = get_feature_columns()
    
    # Подготвя features
    X_train, _ = prepare_features(train_league, global_feature_cols, use_intelligent_imputation=False, legacy_fill_na=True)
    X_val, _ = prepare_features(val_league, global_feature_cols, use_intelligent_imputation=False, legacy_fill_na=True)
    
    # Извлича target
    y_train = train_league['over_25'].values
    y_val = val_league['over_25'].values
    
    # Проверява feature consistency
    assert list(X_train.columns) == global_feature_cols, f"Feature mismatch OU2.5 per-league vs global for {league_slug}"
    
    logger.info(f"Features: {len(global_feature_cols)} (same as global)")
    logger.info(f"Train samples: {len(X_train)}")
    logger.info(f"Val samples: {len(X_val)}")
    logger.info(f"Target distribution - Train: {y_train.mean():.3f}, Val: {y_val.mean():.3f}")
    
    # Хиперпараметри (същите като глобалния модел)
    lgb_params = config['model_ou25']['hyperparameters'].copy()
    lgb_params.update({
        'verbose': -1,
        'random_state': 42,
        'force_col_wise': True
    })
    
    # Тренировка
    logger.info("Тренировка на LightGBM модел...")
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    model = lgb.train(
        lgb_params,
        train_data,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'val'],
        num_boost_round=1000,
        callbacks=[
            lgb.early_stopping(50),
            lgb.log_evaluation(0)  # Silent
        ]
    )
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    
    # Калибрация (ако е включена)
    calibrator = None
    if config['model_ou25']['per_league']['calibration_enabled']:
        logger.info("Прилагане на isotonic calibration...")
        
        # Използваме sklearn isotonic regression за калибрация
        from sklearn.isotonic import IsotonicRegression
        
        calibrator = IsotonicRegression(out_of_bounds='clip')
        calibrator.fit(y_val_pred, y_val)
        
        # Калибрирани predictions
        y_val_pred_calibrated = calibrator.predict(y_val_pred)
    else:
        y_val_pred_calibrated = y_val_pred
    
    # Метрики
    # Конвертира predictions в binary labels
    y_train_pred_binary = (y_train_pred > 0.5).astype(int)
    y_val_pred_binary = (y_val_pred_calibrated > 0.5).astype(int)
    
    train_metrics = evaluate_classification(y_train, y_train_pred_binary, y_train_pred, model_name=f"{league_slug}_train")
    val_metrics = evaluate_classification(y_val, y_val_pred_binary, y_val_pred_calibrated, model_name=f"{league_slug}_val")
    
    metrics = {
        'train': train_metrics,
        'val': val_metrics,
        'league': league_slug,
        'total_matches': total_matches,
        'train_matches': len(train_league),
        'val_matches': len(val_league),
        'feature_count': len(global_feature_cols),
        'calibrated': calibrator is not None
    }
    
    logger.info(f"✅ {league_slug} OU2.5 модел тренирован:")
    logger.info(f"   Train Accuracy: {train_metrics['accuracy']:.4f}")
    logger.info(f"   Val Accuracy: {val_metrics['accuracy']:.4f}")
    logger.info(f"   Val Log Loss: {val_metrics['log_loss']:.4f}")
    logger.info(f"   Val Brier Score: {val_metrics.get('brier_score', 'N/A')}")
    
    return model, calibrator, global_feature_cols, metrics


def train_all_per_league_models(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    config: dict
) -> dict:
    """
    Тренировка на всички per-league OU2.5 модели
    
    Args:
        train_df: Train dataset
        val_df: Validation dataset
        config: Конфигурация
    
    Returns:
        Dictionary с резултатите
    """
    logger = setup_logging()
    
    logger.info("=" * 70)
    logger.info("ТРЕНИРОВКА НА PER-LEAGUE OU2.5 МОДЕЛИ")
    logger.info("=" * 70)
    
    per_league_config = config['model_ou25']['per_league']
    min_matches = per_league_config['min_matches_per_league']
    target_leagues = per_league_config['target_leagues']
    
    results = {}
    summary_metrics = {}
    
    for league_slug in target_leagues:
        logger.info(f"\n🏆 Тренировка за {league_slug}...")
        
        result = train_ou25_per_league(
            train_df, val_df, config, league_slug, min_matches
        )
        
        if result is not None:
            model, calibrator, feature_cols, metrics = result
            
            # Запазва модела
            model_dir = get_per_league_model_path(league_slug, 'ou25', 'v1')
            os.makedirs(model_dir, exist_ok=True)
            
            # Запазва артефактите
            joblib.dump(model, f"{model_dir}/ou25_model.pkl")
            
            if calibrator is not None:
                joblib.dump(calibrator, f"{model_dir}/calibrator.pkl")
            
            # Feature columns (същите като глобалния)
            save_json(feature_cols, f"{model_dir}/feature_columns.json")
            
            # Метрики
            save_json(metrics, f"{model_dir}/metrics.json")
            
            results[league_slug] = {
                'model_path': model_dir,
                'metrics': metrics,
                'calibrated': calibrator is not None
            }
            
            summary_metrics[league_slug] = {
                'accuracy': metrics['val']['accuracy'],
                'log_loss': metrics['val']['log_loss'],
                'brier_score': metrics['val'].get('brier_score', None),
                'matches': metrics['total_matches']
            }
            
            logger.info(f"✅ {league_slug} модел запазен в {model_dir}")
        
        else:
            logger.info(f"❌ {league_slug} пропуснат (недостатъчно данни)")
    
    # Запазва summary отчет
    os.makedirs("logs/model_reports", exist_ok=True)
    summary_report = {
        'timestamp': datetime.now().isoformat(),
        'total_leagues_trained': len(results),
        'target_leagues': target_leagues,
        'min_matches_threshold': min_matches,
        'metrics_by_league': summary_metrics,
        'global_comparison': {
            'note': 'Compare with global OU2.5 model metrics'
        }
    }
    
    save_json(summary_report, "logs/model_reports/ou25_per_league_summary.json")
    
    logger.info(f"\n📊 PER-LEAGUE SUMMARY:")
    logger.info(f"   Успешно тренирани: {len(results)}/{len(target_leagues)} лиги")
    
    for league, metrics in summary_metrics.items():
        logger.info(f"   {league}: Acc={metrics['accuracy']:.3f}, LogLoss={metrics['log_loss']:.3f}, Matches={metrics['matches']}")
    
    logger.info(f"   Summary отчет: logs/model_reports/ou25_per_league_summary.json")
    
    return results


def main():
    """Главна функция за тренировка на всички модели"""
    # Argument parsing
    parser = argparse.ArgumentParser(description='Train ML models for football predictions')
    parser.add_argument('--per-league', action='store_true', 
                       help='Train per-league OU2.5 models in addition to global models')
    parser.add_argument('--only-per-league', action='store_true',
                       help='Train only per-league OU2.5 models (skip global models)')
    
    args = parser.parse_args()
    
    logger = setup_logging()
    
    logger.info("=" * 70)
    logger.info("СТАРТИРАНЕ НА ML MODELS TRAINING PIPELINE")
    if args.per_league:
        logger.info("🏆 PER-LEAGUE РЕЖИМ АКТИВЕН")
    if args.only_per_league:
        logger.info("🏆 САМО PER-LEAGUE МОДЕЛИ")
    logger.info("=" * 70)
    
    # Зареждане на конфигурация
    config = load_config("config/model_config.yaml")
    
    # Зареждане на данни
    logger.info("\nЗареждане на данни...")
    
    train_df = pd.read_parquet("data/processed/train_poisson_predictions.parquet")
    val_df = pd.read_parquet("data/processed/val_poisson_predictions.parquet")
    test_df = pd.read_parquet("data/processed/test_poisson_predictions.parquet")
    
    logger.info(f"Train: {len(train_df)} мача")
    logger.info(f"Validation: {len(val_df)} мача")
    logger.info(f"Test: {len(test_df)} мача")
    
    # Model tracker
    tracker = ModelTracker()
    
    # Проверява режима на тренировка
    if args.only_per_league:
        # Само per-league модели
        if config['model_ou25']['per_league']['enabled']:
            per_league_results = train_all_per_league_models(train_df, val_df, config)
            logger.info(f"✅ Тренирани {len(per_league_results)} per-league OU2.5 модела")
        else:
            logger.warning("Per-league training е изключен в конфигурацията!")
    
    else:
        # Стандартни глобални модели
        # 1. Train 1X2 Model
        logger.info("\n" + "=" * 70)
        logger.info("[1/3] ТРЕНИРОВКА НА 1X2 MODEL")
        logger.info("=" * 70)
        
        model_1x2, features_1x2, metrics_1x2 = train_1x2_model(train_df, val_df, config)
        save_model(model_1x2, features_1x2, metrics_1x2, "1x2", "models/model_1x2_v1")
        tracker.add_model("1X2", metrics_1x2['train'], metrics_1x2['val'])
        
        # 2. Train Over/Under 2.5 Model
        logger.info("\n" + "=" * 70)
        logger.info("[2/3] ТРЕНИРОВКА НА OVER/UNDER 2.5 MODEL")
        logger.info("=" * 70)
        
        model_ou25, features_ou25, metrics_ou25 = train_ou25_model(train_df, val_df, config)
        save_model(model_ou25, features_ou25, metrics_ou25, "ou25", "models/model_ou25_v1")
        tracker.add_model("OU2.5", metrics_ou25['train'], metrics_ou25['val'])
        
        # 3. Train BTTS Model
        logger.info("\n" + "=" * 70)
        logger.info("[3/3] ТРЕНИРОВКА НА BTTS MODEL")
        logger.info("=" * 70)
        
        model_btts, features_btts, metrics_btts = train_btts_model(train_df, val_df, config)
        save_model(model_btts, features_btts, metrics_btts, "btts", "models/model_btts_v1")
        tracker.add_model("BTTS", metrics_btts['train'], metrics_btts['val'])
        
        # Per-league модели (ако е заявено)
        if args.per_league and config['model_ou25']['per_league']['enabled']:
            logger.info("\n" + "=" * 70)
            logger.info("[BONUS] PER-LEAGUE OU2.5 МОДЕЛИ")
            logger.info("=" * 70)
            
            per_league_results = train_all_per_league_models(train_df, val_df, config)
            logger.info(f"✅ Тренирани {len(per_league_results)} per-league OU2.5 модела")
        
        # Сравнение на модели
        logger.info("\n" + "=" * 70)
        logger.info("ОБОБЩЕНИЕ НА ВСИЧКИ МОДЕЛИ")
        logger.info("=" * 70)
        
        tracker.compare_models('accuracy')
        tracker.compare_models('log_loss')
    
    logger.info("\n" + "=" * 70)
    logger.info("ML MODELS TRAINING ЗАВЪРШЕН УСПЕШНО! ✓")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
