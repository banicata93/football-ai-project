"""
Training Pipeline за Stacking Ensemble (Meta-Learning) - Simplified Version

Използва вече генерираните predictions от базовите модели.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime

from core.stacking_ensemble import StackingEnsemble, compare_ensembles
from core.utils import setup_logging

# Setup logging
logger = setup_logging()


def load_predictions():
    """Зареждане на predictions от базовите модели"""
    logger.info("="*70)
    logger.info("ЗАРЕЖДАНЕ НА PREDICTIONS")
    logger.info("="*70)
    
    # Load data with predictions
    train_df = pd.read_parquet("data/processed/train_final_predictions.parquet")
    val_df = pd.read_parquet("data/processed/val_final_predictions.parquet")
    test_df = pd.read_parquet("data/processed/test_final_predictions.parquet")
    
    logger.info(f"Train: {len(train_df)} мача")
    logger.info(f"Validation: {len(val_df)} мача")
    logger.info(f"Test: {len(test_df)} мача")
    
    return train_df, val_df, test_df


def extract_base_predictions(df):
    """
    Извличане на predictions от базовите модели
    
    Returns:
        Dictionary с predictions за всеки target
    """
    predictions = {
        '1x2': {
            'poisson': df[['poisson_prob_1', 'poisson_prob_x', 'poisson_prob_2']].values,
            # За ML predictions ще използваме ensemble (което всъщност е weighted average)
            'ml': df[['ensemble_prob_1', 'ensemble_prob_x', 'ensemble_prob_2']].values
        },
        'ou25': {
            'poisson': df['poisson_prob_over25'].values,
            'ml': df['ensemble_prob_over25'].values
        },
        'btts': {
            'poisson': df['poisson_prob_btts'].values,
            'ml': df['ensemble_prob_btts'].values
        }
    }
    
    return predictions


def train_stacking_ensemble(train_predictions, train_df, meta_model_type='xgboost'):
    """
    Обучение на stacking ensemble
    """
    logger.info("="*70)
    logger.info(f"ОБУЧЕНИЕ НА STACKING ENSEMBLE ({meta_model_type.upper()})")
    logger.info("="*70)
    
    # Създаване на stacking ensemble
    stacking = StackingEnsemble(
        meta_model_type=meta_model_type,
        calibrate=True,
        use_original_features=False
    )
    
    # Обучение на мета-модели
    
    # 1X2
    logger.info("\n1. Обучение на мета-модел за 1X2...")
    # Convert result to numeric: '1' -> 0, 'X' -> 1, '2' -> 2
    result_map = {'1': 0, 'X': 1, '2': 2}
    y_1x2 = train_df['result'].map(result_map).values
    stacking.train_1x2(train_predictions['1x2'], y_1x2)
    
    # OU2.5
    logger.info("\n2. Обучение на мета-модел за OU2.5...")
    y_ou25 = (train_df['total_goals'] > 2.5).astype(int).values
    stacking.train_ou25(train_predictions['ou25'], y_ou25)
    
    # BTTS
    logger.info("\n3. Обучение на мета-модел за BTTS...")
    y_btts = ((train_df['home_score'] > 0) & (train_df['away_score'] > 0)).astype(int).values
    stacking.train_btts(train_predictions['btts'], y_btts)
    
    logger.info("\n✓ Всички мета-модели обучени!")
    
    return stacking


def evaluate_stacking(stacking, predictions, df, split_name):
    """Оценка на stacking ensemble"""
    logger.info(f"\n{'='*70}")
    logger.info(f"ОЦЕНКА НА STACKING ENSEMBLE - {split_name.upper()}")
    logger.info(f"{'='*70}")
    
    results = {}
    
    # 1X2
    result_map = {'1': 0, 'X': 1, '2': 2}
    y_1x2 = df['result'].map(result_map).values
    metrics_1x2 = stacking.evaluate(predictions['1x2'], y_1x2, '1x2')
    results['1x2'] = metrics_1x2
    
    logger.info(f"\n1X2:")
    logger.info(f"  Accuracy: {metrics_1x2['accuracy']:.4f} ({metrics_1x2['accuracy']*100:.2f}%)")
    logger.info(f"  Log Loss: {metrics_1x2['log_loss']:.4f}")
    
    # OU2.5
    y_ou25 = (df['total_goals'] > 2.5).astype(int).values
    metrics_ou25 = stacking.evaluate(predictions['ou25'], y_ou25, 'ou25')
    results['ou25'] = metrics_ou25
    
    logger.info(f"\nOver/Under 2.5:")
    logger.info(f"  Accuracy: {metrics_ou25['accuracy']:.4f} ({metrics_ou25['accuracy']*100:.2f}%)")
    logger.info(f"  Log Loss: {metrics_ou25['log_loss']:.4f}")
    logger.info(f"  ROC AUC: {metrics_ou25['roc_auc']:.4f}")
    
    # BTTS
    y_btts = ((df['home_score'] > 0) & (df['away_score'] > 0)).astype(int).values
    metrics_btts = stacking.evaluate(predictions['btts'], y_btts, 'btts')
    results['btts'] = metrics_btts
    
    logger.info(f"\nBTTS:")
    logger.info(f"  Accuracy: {metrics_btts['accuracy']:.4f} ({metrics_btts['accuracy']*100:.2f}%)")
    logger.info(f"  Log Loss: {metrics_btts['log_loss']:.4f}")
    logger.info(f"  ROC AUC: {metrics_btts['roc_auc']:.4f}")
    
    return results


def compare_with_weighted_ensemble(stacking, test_predictions, test_df):
    """Сравнение на Stacking vs Weighted Ensemble"""
    logger.info("\n" + "="*70)
    logger.info("СРАВНЕНИЕ: STACKING vs WEIGHTED ENSEMBLE")
    logger.info("="*70)
    
    # Weighted ensemble predictions (вече са в test_df)
    weighted_preds = {
        '1x2': test_df[['ensemble_prob_1', 'ensemble_prob_x', 'ensemble_prob_2']].fillna(0.33).values,
        'ou25': test_df['ensemble_prob_over25'].fillna(0.5).values,
        'btts': test_df['ensemble_prob_btts'].fillna(0.5).values
    }
    
    # Stacking predictions
    stacking_preds = {}
    stacking_preds['1x2'] = stacking.predict_1x2(test_predictions['1x2'])
    stacking_preds['ou25'] = stacking.predict_ou25(test_predictions['ou25'])
    stacking_preds['btts'] = stacking.predict_btts(test_predictions['btts'])
    
    # True labels
    result_map = {'1': 0, 'X': 1, '2': 2}
    y_true = {
        '1x2': test_df['result'].map(result_map).values,
        'ou25': (test_df['total_goals'] > 2.5).astype(int).values,
        'btts': ((test_df['home_score'] > 0) & (test_df['away_score'] > 0)).astype(int).values
    }
    
    # Сравнение
    comparison_df = compare_ensembles(weighted_preds, stacking_preds, y_true)
    
    logger.info("\n" + str(comparison_df.to_string(index=False)))
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("РЕЗУЛТАТИ:")
    logger.info("="*70)
    
    total_improvement = 0
    for _, row in comparison_df.iterrows():
        task = row['Task']
        acc_imp = row['Accuracy_Improvement']
        ll_imp = row['LogLoss_Improvement']
        
        logger.info(f"\n{task}:")
        logger.info(f"  Accuracy improvement: {acc_imp:+.4f} ({acc_imp*100:+.2f}%)")
        logger.info(f"  Log Loss improvement: {ll_imp:+.4f}")
        
        if acc_imp > 0:
            logger.info(f"  ✓ Stacking е по-добър!")
            total_improvement += 1
        else:
            logger.info(f"  ✗ Weighted е по-добър")
    
    logger.info(f"\n{'='*70}")
    logger.info(f"Stacking е по-добър в {total_improvement}/3 задачи")
    logger.info(f"{'='*70}")
    
    return comparison_df


def main():
    """Main training pipeline"""
    
    logger.info("\n" + "="*70)
    logger.info("STACKING ENSEMBLE TRAINING PIPELINE (SIMPLIFIED)")
    logger.info("="*70)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. Load predictions
    train_df, val_df, test_df = load_predictions()
    
    # 2. Extract base predictions
    logger.info("\n" + "="*70)
    logger.info("ИЗВЛИЧАНЕ НА BASE PREDICTIONS")
    logger.info("="*70)
    
    train_predictions = extract_base_predictions(train_df)
    val_predictions = extract_base_predictions(val_df)
    test_predictions = extract_base_predictions(test_df)
    
    logger.info("✓ Predictions извлечени")
    
    # 3. Train stacking ensemble (XGBoost meta-models)
    stacking_xgb = train_stacking_ensemble(train_predictions, train_df, 'xgboost')
    
    # 4. Evaluate on validation set
    val_results_xgb = evaluate_stacking(stacking_xgb, val_predictions, val_df, 'Validation')
    
    # 5. Evaluate on test set
    test_results_xgb = evaluate_stacking(stacking_xgb, test_predictions, test_df, 'Test')
    
    # 6. Train Logistic Regression meta-models for comparison
    logger.info("\n" + "="*70)
    logger.info("ОБУЧЕНИЕ НА LOGISTIC REGRESSION META-MODELS")
    logger.info("="*70)
    
    stacking_lr = train_stacking_ensemble(train_predictions, train_df, 'logistic')
    test_results_lr = evaluate_stacking(stacking_lr, test_predictions, test_df, 'Test (LR)')
    
    # 7. Compare with weighted ensemble
    comparison_df = compare_with_weighted_ensemble(stacking_xgb, test_predictions, test_df)
    
    # 8. Save best model
    logger.info("\n" + "="*70)
    logger.info("ЗАПАЗВАНЕ НА МОДЕЛИ")
    logger.info("="*70)
    
    # Save XGBoost stacking
    output_dir = "models/stacking_v1"
    stacking_xgb.save(output_dir)
    
    # Save metrics
    all_metrics = {
        'xgboost': {
            'validation': val_results_xgb,
            'test': test_results_xgb
        },
        'logistic': {
            'test': test_results_lr
        },
        'comparison': comparison_df.to_dict('records'),
        'training_date': datetime.now().isoformat()
    }
    
    with open(f"{output_dir}/metrics.json", 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    logger.info(f"✓ Метрики запазени: {output_dir}/metrics.json")
    
    # Final summary
    logger.info("\n" + "="*70)
    logger.info("ФИНАЛЕН РЕЗУЛТАТ")
    logger.info("="*70)
    
    logger.info("\nXGBoost Stacking Ensemble (Test Set):")
    logger.info(f"  1X2 Accuracy: {test_results_xgb['1x2']['accuracy']:.4f} ({test_results_xgb['1x2']['accuracy']*100:.2f}%)")
    logger.info(f"  OU2.5 Accuracy: {test_results_xgb['ou25']['accuracy']:.4f} ({test_results_xgb['ou25']['accuracy']*100:.2f}%)")
    logger.info(f"  BTTS Accuracy: {test_results_xgb['btts']['accuracy']:.4f} ({test_results_xgb['btts']['accuracy']*100:.2f}%)")
    
    logger.info("\nLogistic Regression Stacking (Test Set):")
    logger.info(f"  1X2 Accuracy: {test_results_lr['1x2']['accuracy']:.4f} ({test_results_lr['1x2']['accuracy']*100:.2f}%)")
    logger.info(f"  OU2.5 Accuracy: {test_results_lr['ou25']['accuracy']:.4f} ({test_results_lr['ou25']['accuracy']*100:.2f}%)")
    logger.info(f"  BTTS Accuracy: {test_results_lr['btts']['accuracy']:.4f} ({test_results_lr['btts']['accuracy']*100:.2f}%)")
    
    logger.info("\n" + "="*70)
    logger.info("✓ STACKING ENSEMBLE TRAINING ЗАВЪРШЕН!")
    logger.info("="*70)
    logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
