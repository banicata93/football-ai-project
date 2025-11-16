"""
Training pipeline за Ensemble модел и финална оценка
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import joblib
from datetime import datetime

from core.ensemble import EnsembleModel, FootballIntelligenceIndex, PredictionCombiner
from core.ml_utils import get_feature_columns, prepare_features, encode_target_1x2
from core.utils import setup_logging, load_config, save_json, PerformanceTimer
from sklearn.metrics import accuracy_score, log_loss, classification_report


def load_all_models():
    """Зареждане на всички обучени модели"""
    logger = setup_logging()
    
    logger.info("Зареждане на модели...")
    
    models = {
        'poisson': joblib.load('models/model_poisson_v1/poisson_model.pkl'),
        '1x2': joblib.load('models/model_1x2_v1/1x2_model.pkl'),
        'ou25': joblib.load('models/model_ou25_v1/ou25_model.pkl'),
        'btts': joblib.load('models/model_btts_v1/btts_model.pkl')
    }
    
    logger.info("Всички модели заредени успешно")
    
    return models


def generate_all_predictions(models: dict, df: pd.DataFrame) -> dict:
    """
    Генериране на predictions от всички модели
    
    Args:
        models: Dictionary с модели
        df: DataFrame с данни
    
    Returns:
        Dictionary с predictions
    """
    logger = setup_logging()
    
    logger.info(f"Генериране на predictions за {len(df)} мача...")
    
    predictions = {}
    
    # Poisson predictions (вече са в df)
    predictions['poisson_1x2'] = df[['poisson_prob_1', 'poisson_prob_x', 'poisson_prob_2']].values
    predictions['poisson_ou25'] = df['poisson_prob_over25'].values
    predictions['poisson_btts'] = df['poisson_prob_btts'].values
    
    # ML predictions
    feature_cols = get_feature_columns()
    X = prepare_features(df, feature_cols)
    
    # 1X2
    predictions['ml_1x2'] = models['1x2'].predict_proba(X)
    
    # OU2.5
    predictions['ml_ou25'] = models['ou25'].predict_proba(X)[:, 1]
    
    # BTTS
    predictions['ml_btts'] = models['btts'].predict_proba(X)[:, 1]
    
    logger.info("Всички predictions генерирани")
    
    return predictions


def evaluate_ensemble(
    df: pd.DataFrame,
    dataset_name: str = "Dataset"
) -> dict:
    """
    Оценка на ensemble predictions
    
    Args:
        df: DataFrame с predictions
        dataset_name: Име на dataset
    
    Returns:
        Dictionary с метрики
    """
    logger = setup_logging()
    
    logger.info(f"\n{'=' * 70}")
    logger.info(f"ОЦЕНКА НА ENSEMBLE - {dataset_name}")
    logger.info(f"{'=' * 70}")
    
    metrics = {}
    
    # 1X2 Evaluation
    y_true_1x2, _ = encode_target_1x2(df['result'])
    y_pred_1x2 = df['ensemble_pred_1x2'].values
    y_proba_1x2 = df[['ensemble_prob_1', 'ensemble_prob_x', 'ensemble_prob_2']].values
    
    # Handle NaN values
    y_proba_1x2 = np.nan_to_num(y_proba_1x2, nan=0.33)
    
    # Ensure valid probabilities
    y_proba_1x2 = np.clip(y_proba_1x2, 1e-15, 1 - 1e-15)
    y_proba_1x2 = y_proba_1x2 / y_proba_1x2.sum(axis=1, keepdims=True)
    
    acc_1x2 = accuracy_score(y_true_1x2, y_pred_1x2)
    
    # One-hot encode y_true for log_loss
    y_true_1x2_onehot = np.zeros((len(y_true_1x2), 3))
    y_true_1x2_onehot[np.arange(len(y_true_1x2)), y_true_1x2] = 1
    
    logloss_1x2 = log_loss(y_true_1x2_onehot, y_proba_1x2)
    
    metrics['1x2_accuracy'] = acc_1x2
    metrics['1x2_log_loss'] = logloss_1x2
    
    logger.info(f"\n1X2 Metrics:")
    logger.info(f"  Accuracy: {acc_1x2:.4f} ({acc_1x2*100:.2f}%)")
    logger.info(f"  Log Loss: {logloss_1x2:.4f}")
    
    # Classification report
    pred_labels = ['1' if p == 0 else 'X' if p == 1 else '2' for p in y_pred_1x2]
    true_labels = ['1' if t == 0 else 'X' if t == 1 else '2' for t in y_true_1x2]
    
    logger.info(f"\n{classification_report(true_labels, pred_labels, target_names=['1', 'X', '2'])}")
    
    # OU2.5 Evaluation
    y_true_ou25 = df['over_25'].values
    y_pred_ou25 = df['ensemble_pred_ou25'].values
    y_proba_ou25 = df['ensemble_prob_over25'].fillna(0.5).values
    y_proba_ou25 = np.clip(y_proba_ou25, 1e-15, 1 - 1e-15)
    
    acc_ou25 = accuracy_score(y_true_ou25, y_pred_ou25)
    logloss_ou25 = log_loss(y_true_ou25, y_proba_ou25)
    
    metrics['ou25_accuracy'] = acc_ou25
    metrics['ou25_log_loss'] = logloss_ou25
    
    logger.info(f"\nOver/Under 2.5 Metrics:")
    logger.info(f"  Accuracy: {acc_ou25:.4f} ({acc_ou25*100:.2f}%)")
    logger.info(f"  Log Loss: {logloss_ou25:.4f}")
    
    # BTTS Evaluation
    y_true_btts = df['btts'].values
    y_pred_btts = df['ensemble_pred_btts'].values
    y_proba_btts = df['ensemble_prob_btts'].fillna(0.5).values
    y_proba_btts = np.clip(y_proba_btts, 1e-15, 1 - 1e-15)
    
    acc_btts = accuracy_score(y_true_btts, y_pred_btts)
    logloss_btts = log_loss(y_true_btts, y_proba_btts)
    
    metrics['btts_accuracy'] = acc_btts
    metrics['btts_log_loss'] = logloss_btts
    
    logger.info(f"\nBTTS Metrics:")
    logger.info(f"  Accuracy: {acc_btts:.4f} ({acc_btts*100:.2f}%)")
    logger.info(f"  Log Loss: {logloss_btts:.4f}")
    
    # FII Statistics
    if 'fii_score' in df.columns:
        logger.info(f"\nFII Statistics:")
        logger.info(f"  Mean: {df['fii_score'].mean():.2f}")
        logger.info(f"  Median: {df['fii_score'].median():.2f}")
        logger.info(f"  Std: {df['fii_score'].std():.2f}")
        
        fii = FootballIntelligenceIndex()
        fii.get_fii_distribution(df)
    
    return metrics


def save_ensemble(
    ensemble: EnsembleModel,
    fii: FootballIntelligenceIndex,
    metrics: dict,
    output_dir: str = "models/ensemble_v1"
):
    """Запазване на ensemble модел"""
    logger = setup_logging()
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Запазване на ensemble
    ensemble_path = os.path.join(output_dir, "ensemble_model.pkl")
    joblib.dump(ensemble, ensemble_path)
    logger.info(f"Ensemble запазен: {ensemble_path}")
    
    # Запазване на FII
    fii_path = os.path.join(output_dir, "fii_model.pkl")
    joblib.dump(fii, fii_path)
    logger.info(f"FII запазен: {fii_path}")
    
    # Запазване на метрики
    metrics_path = os.path.join(output_dir, "metrics.json")
    save_json(metrics, metrics_path)
    logger.info(f"Метрики запазени: {metrics_path}")
    
    # Model info
    model_info = {
        'model_type': 'ensemble',
        'version': 'v1',
        'trained_date': datetime.now().isoformat(),
        'weights': ensemble.weights,
        'fii_weights': fii.weights,
        'fii_thresholds': fii.thresholds
    }
    
    info_path = os.path.join(output_dir, "model_info.json")
    save_json(model_info, info_path)


def main():
    """Главна функция"""
    logger = setup_logging()
    
    logger.info("=" * 70)
    logger.info("СТАРТИРАНЕ НА ENSEMBLE & FII TRAINING PIPELINE")
    logger.info("=" * 70)
    
    # Зареждане на модели
    logger.info("\n[1/6] Зареждане на модели...")
    models = load_all_models()
    
    # Зареждане на данни
    logger.info("\n[2/6] Зареждане на данни...")
    train_df = pd.read_parquet("data/processed/train_poisson_predictions.parquet")
    val_df = pd.read_parquet("data/processed/val_poisson_predictions.parquet")
    test_df = pd.read_parquet("data/processed/test_poisson_predictions.parquet")
    
    logger.info(f"Train: {len(train_df)} мача")
    logger.info(f"Validation: {len(val_df)} мача")
    logger.info(f"Test: {len(test_df)} мача")
    
    # Генериране на predictions
    logger.info("\n[3/6] Генериране на predictions...")
    
    train_preds = generate_all_predictions(models, train_df)
    val_preds = generate_all_predictions(models, val_df)
    test_preds = generate_all_predictions(models, test_df)
    
    # Инициализация на combiner
    logger.info("\n[4/6] Комбиниране на predictions...")
    combiner = PredictionCombiner()
    
    # Комбиниране
    train_df = combiner.combine_all_predictions(
        train_df,
        train_preds['poisson_1x2'], train_preds['ml_1x2'],
        train_preds['poisson_ou25'], train_preds['ml_ou25'],
        train_preds['poisson_btts'], train_preds['ml_btts']
    )
    
    val_df = combiner.combine_all_predictions(
        val_df,
        val_preds['poisson_1x2'], val_preds['ml_1x2'],
        val_preds['poisson_ou25'], val_preds['ml_ou25'],
        val_preds['poisson_btts'], val_preds['ml_btts']
    )
    
    test_df = combiner.combine_all_predictions(
        test_df,
        test_preds['poisson_1x2'], test_preds['ml_1x2'],
        test_preds['poisson_ou25'], test_preds['ml_ou25'],
        test_preds['poisson_btts'], test_preds['ml_btts']
    )
    
    # Evaluation
    logger.info("\n[5/6] Оценка на ensemble...")
    
    train_metrics = evaluate_ensemble(train_df, "TRAIN")
    val_metrics = evaluate_ensemble(val_df, "VALIDATION")
    test_metrics = evaluate_ensemble(test_df, "TEST")
    
    # Запазване
    logger.info("\n[6/6] Запазване на модели и predictions...")
    
    all_metrics = {
        'train': train_metrics,
        'validation': val_metrics,
        'test': test_metrics
    }
    
    save_ensemble(combiner.ensemble, combiner.fii, all_metrics)
    
    # Запазване на финални predictions
    train_df.to_parquet("data/processed/train_final_predictions.parquet", index=False)
    val_df.to_parquet("data/processed/val_final_predictions.parquet", index=False)
    test_df.to_parquet("data/processed/test_final_predictions.parquet", index=False)
    
    logger.info("Финални predictions запазени")
    
    # Финално обобщение
    logger.info("\n" + "=" * 70)
    logger.info("ФИНАЛНО ОБОБЩЕНИЕ")
    logger.info("=" * 70)
    
    logger.info(f"\nTRAIN SET:")
    logger.info(f"  1X2 Accuracy: {train_metrics['1x2_accuracy']*100:.2f}%")
    logger.info(f"  OU2.5 Accuracy: {train_metrics['ou25_accuracy']*100:.2f}%")
    logger.info(f"  BTTS Accuracy: {train_metrics['btts_accuracy']*100:.2f}%")
    
    logger.info(f"\nVALIDATION SET:")
    logger.info(f"  1X2 Accuracy: {val_metrics['1x2_accuracy']*100:.2f}%")
    logger.info(f"  OU2.5 Accuracy: {val_metrics['ou25_accuracy']*100:.2f}%")
    logger.info(f"  BTTS Accuracy: {val_metrics['btts_accuracy']*100:.2f}%")
    
    logger.info(f"\nTEST SET:")
    logger.info(f"  1X2 Accuracy: {test_metrics['1x2_accuracy']*100:.2f}%")
    logger.info(f"  OU2.5 Accuracy: {test_metrics['ou25_accuracy']*100:.2f}%")
    logger.info(f"  BTTS Accuracy: {test_metrics['btts_accuracy']*100:.2f}%")
    
    logger.info("\n" + "=" * 70)
    logger.info("ENSEMBLE & FII TRAINING ЗАВЪРШЕН УСПЕШНО! ✓")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
