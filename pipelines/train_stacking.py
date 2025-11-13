"""
Training Pipeline за Stacking Ensemble (Meta-Learning)

Този pipeline:
1. Зарежда всички базови модели (Poisson, ML)
2. Генерира predictions от базовите модели
3. Обучава мета-модели да комбинират тези predictions
4. Оценява и сравнява с weighted ensemble
5. Запазва най-добрия подход
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
from core.ensemble import EnsembleModel
from core.ml_utils import prepare_features
from core.utils import setup_logging

# Setup logging
logger = setup_logging()


def load_data():
    """Зареждане на train, validation и test данни"""
    logger.info("="*70)
    logger.info("ЗАРЕЖДАНЕ НА ДАННИ")
    logger.info("="*70)
    
    train_df = pd.read_parquet("data/processed/train_features.parquet")
    val_df = pd.read_parquet("data/processed/val_features.parquet")
    test_df = pd.read_parquet("data/processed/test_features.parquet")
    
    logger.info(f"Train: {len(train_df)} мача")
    logger.info(f"Validation: {len(val_df)} мача")
    logger.info(f"Test: {len(test_df)} мача")
    
    return train_df, val_df, test_df


def load_base_models():
    """Зареждане на базовите модели"""
    logger.info("="*70)
    logger.info("ЗАРЕЖДАНЕ НА БАЗОВИ МОДЕЛИ")
    logger.info("="*70)
    
    models = {}
    
    # Poisson model
    with open("models/model_poisson_v1/poisson_model.pkl", 'rb') as f:
        models['poisson'] = pickle.load(f)
    logger.info("✓ Poisson model зареден")
    
    # 1X2 model
    with open("models/model_1x2_v1/1x2_model.pkl", 'rb') as f:
        models['1x2'] = pickle.load(f)
    logger.info("✓ 1X2 model зареден")
    
    # OU2.5 model
    with open("models/model_ou25_v1/ou25_model.pkl", 'rb') as f:
        models['ou25'] = pickle.load(f)
    logger.info("✓ OU2.5 model зареден")
    
    # BTTS model
    with open("models/model_btts_v1/btts_model.pkl", 'rb') as f:
        models['btts'] = pickle.load(f)
    logger.info("✓ BTTS model зареден")
    
    # Feature columns
    with open("models/model_1x2_v1/feature_columns.json", 'r') as f:
        feature_data = json.load(f)
        feature_columns = feature_data['features']  # Правилният ключ
    
    logger.info(f"✓ Feature columns заредени: {len(feature_columns)} features")
    
    return models, feature_columns


def generate_base_predictions(df, models, feature_columns):
    """
    Генериране на predictions от базовите модели
    
    Returns:
        Dictionary с predictions за всеки target
    """
    logger.info(f"Генериране на predictions за {len(df)} мача...")
    
    # Първо генерираме Poisson predictions
    poisson_1x2 = []
    poisson_ou25 = []
    poisson_btts = []
    poisson_lambda_home = []
    poisson_lambda_away = []
    poisson_expected_goals = []
    
    for idx, row in df.iterrows():
        try:
            pred = models['poisson'].predict_match_probabilities(
                row['home_team_id'],
                row['away_team_id']
            )
            poisson_1x2.append([pred['prob_home_win'], pred['prob_draw'], pred['prob_away_win']])
            poisson_ou25.append(pred['prob_over_25'])
            poisson_btts.append(pred['prob_btts_yes'])
            poisson_lambda_home.append(pred['lambda_home'])
            poisson_lambda_away.append(pred['lambda_away'])
            poisson_expected_goals.append(pred['expected_total_goals'])
        except:
            # Fallback
            poisson_1x2.append([0.33, 0.33, 0.34])
            poisson_ou25.append(0.5)
            poisson_btts.append(0.5)
            poisson_lambda_home.append(1.5)
            poisson_lambda_away.append(1.2)
            poisson_expected_goals.append(2.7)
    
    poisson_1x2 = np.array(poisson_1x2)
    poisson_ou25 = np.array(poisson_ou25)
    poisson_btts = np.array(poisson_btts)
    
    # Добавяме Poisson features към dataframe (за ML моделите)
    df_with_poisson = df.copy()
    df_with_poisson['poisson_prob_1'] = poisson_1x2[:, 0]
    df_with_poisson['poisson_prob_x'] = poisson_1x2[:, 1]
    df_with_poisson['poisson_prob_2'] = poisson_1x2[:, 2]
    df_with_poisson['poisson_prob_over25'] = poisson_ou25
    df_with_poisson['poisson_prob_btts'] = poisson_btts
    df_with_poisson['poisson_lambda_home'] = poisson_lambda_home
    df_with_poisson['poisson_lambda_away'] = poisson_lambda_away
    df_with_poisson['poisson_expected_goals'] = poisson_expected_goals
    
    # Подготовка на features с Poisson
    X = prepare_features(df_with_poisson, feature_columns)
    
    # ML predictions
    ml_1x2 = models['1x2'].predict_proba(X)
    ml_ou25 = models['ou25'].predict_proba(X)[:, 1]
    ml_btts = models['btts'].predict_proba(X)[:, 1]
    
    # Organize predictions
    predictions = {
        '1x2': {
            'poisson': poisson_1x2,
            'ml': ml_1x2
        },
        'ou25': {
            'poisson': poisson_ou25,
            'ml': ml_ou25
        },
        'btts': {
            'poisson': poisson_btts,
            'ml': ml_btts
        }
    }
    
    logger.info("✓ Predictions генерирани")
    
    return predictions


def train_stacking_ensemble(train_predictions, train_df, meta_model_type='xgboost'):
    """
    Обучение на stacking ensemble
    
    Args:
        train_predictions: Predictions от базовите модели
        train_df: Training data
        meta_model_type: 'logistic' или 'xgboost'
    
    Returns:
        Обучен StackingEnsemble
    """
    logger.info("="*70)
    logger.info(f"ОБУЧЕНИЕ НА STACKING ENSEMBLE ({meta_model_type.upper()})")
    logger.info("="*70)
    
    # Създаване на stacking ensemble
    stacking = StackingEnsemble(
        meta_model_type=meta_model_type,
        calibrate=True,
        use_original_features=False  # Може да се експериментира с True
    )
    
    # Обучение на мета-модели
    
    # 1X2
    logger.info("\n1. Обучение на мета-модел за 1X2...")
    y_1x2 = train_df['result'].values
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
    """
    Оценка на stacking ensemble
    
    Args:
        stacking: StackingEnsemble
        predictions: Base predictions
        df: DataFrame
        split_name: 'Train', 'Validation' или 'Test'
    
    Returns:
        Dictionary с метрики
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"ОЦЕНКА НА STACKING ENSEMBLE - {split_name.upper()}")
    logger.info(f"{'='*70}")
    
    results = {}
    
    # 1X2
    y_1x2 = df['result'].values
    metrics_1x2 = stacking.evaluate(predictions['1x2'], y_1x2, '1x2')
    results['1x2'] = metrics_1x2
    
    logger.info(f"\n1X2:")
    logger.info(f"  Accuracy: {metrics_1x2['accuracy']:.4f}")
    logger.info(f"  Log Loss: {metrics_1x2['log_loss']:.4f}")
    
    # OU2.5
    y_ou25 = (df['total_goals'] > 2.5).astype(int).values
    metrics_ou25 = stacking.evaluate(predictions['ou25'], y_ou25, 'ou25')
    results['ou25'] = metrics_ou25
    
    logger.info(f"\nOver/Under 2.5:")
    logger.info(f"  Accuracy: {metrics_ou25['accuracy']:.4f}")
    logger.info(f"  Log Loss: {metrics_ou25['log_loss']:.4f}")
    logger.info(f"  ROC AUC: {metrics_ou25['roc_auc']:.4f}")
    
    # BTTS
    y_btts = ((df['home_score'] > 0) & (df['away_score'] > 0)).astype(int).values
    metrics_btts = stacking.evaluate(predictions['btts'], y_btts, 'btts')
    results['btts'] = metrics_btts
    
    logger.info(f"\nBTTS:")
    logger.info(f"  Accuracy: {metrics_btts['accuracy']:.4f}")
    logger.info(f"  Log Loss: {metrics_btts['log_loss']:.4f}")
    logger.info(f"  ROC AUC: {metrics_btts['roc_auc']:.4f}")
    
    return results


def compare_with_weighted_ensemble(
    stacking,
    weighted_ensemble,
    test_predictions,
    test_df
):
    """
    Сравнение на Stacking vs Weighted Ensemble
    """
    logger.info("\n" + "="*70)
    logger.info("СРАВНЕНИЕ: STACKING vs WEIGHTED ENSEMBLE")
    logger.info("="*70)
    
    # Generate predictions от двата ensemble подхода
    
    # Weighted ensemble predictions
    weighted_preds = {}
    
    # 1X2
    weighted_1x2 = weighted_ensemble.predict(
        test_predictions['1x2']['poisson'],
        test_predictions['1x2']['ml'],
        np.zeros((len(test_df), 3))  # Dummy Elo predictions
    )
    weighted_preds['1x2'] = weighted_1x2
    
    # OU2.5
    weighted_ou25 = (
        0.3 * test_predictions['ou25']['poisson'] +
        0.7 * test_predictions['ou25']['ml']
    )
    weighted_preds['ou25'] = weighted_ou25
    
    # BTTS
    weighted_btts = (
        0.3 * test_predictions['btts']['poisson'] +
        0.7 * test_predictions['btts']['ml']
    )
    weighted_preds['btts'] = weighted_btts
    
    # Stacking predictions
    stacking_preds = {}
    stacking_preds['1x2'] = stacking.predict_1x2(test_predictions['1x2'])
    stacking_preds['ou25'] = stacking.predict_ou25(test_predictions['ou25'])
    stacking_preds['btts'] = stacking.predict_btts(test_predictions['btts'])
    
    # True labels
    y_true = {
        '1x2': test_df['result'].values,
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
    
    for _, row in comparison_df.iterrows():
        task = row['Task']
        acc_imp = row['Accuracy_Improvement']
        ll_imp = row['LogLoss_Improvement']
        
        logger.info(f"\n{task}:")
        logger.info(f"  Accuracy improvement: {acc_imp:+.4f} ({acc_imp*100:+.2f}%)")
        logger.info(f"  Log Loss improvement: {ll_imp:+.4f}")
        
        if acc_imp > 0:
            logger.info(f"  ✓ Stacking е по-добър!")
        else:
            logger.info(f"  ✗ Weighted е по-добър")
    
    return comparison_df


def main():
    """Main training pipeline"""
    
    logger.info("\n" + "="*70)
    logger.info("STACKING ENSEMBLE TRAINING PIPELINE")
    logger.info("="*70)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. Load data
    train_df, val_df, test_df = load_data()
    
    # 2. Load base models
    models, feature_columns = load_base_models()
    
    # 3. Generate base predictions
    logger.info("\n" + "="*70)
    logger.info("ГЕНЕРИРАНЕ НА BASE PREDICTIONS")
    logger.info("="*70)
    
    logger.info("\nTrain set...")
    train_predictions = generate_base_predictions(train_df, models, feature_columns)
    
    logger.info("\nValidation set...")
    val_predictions = generate_base_predictions(val_df, models, feature_columns)
    
    logger.info("\nTest set...")
    test_predictions = generate_base_predictions(test_df, models, feature_columns)
    
    # 4. Train stacking ensemble (XGBoost meta-models)
    stacking_xgb = train_stacking_ensemble(train_predictions, train_df, 'xgboost')
    
    # 5. Evaluate on validation set
    val_results_xgb = evaluate_stacking(stacking_xgb, val_predictions, val_df, 'Validation')
    
    # 6. Evaluate on test set
    test_results_xgb = evaluate_stacking(stacking_xgb, test_predictions, test_df, 'Test')
    
    # 7. Train Logistic Regression meta-models for comparison
    logger.info("\n" + "="*70)
    logger.info("ОБУЧЕНИЕ НА LOGISTIC REGRESSION META-MODELS")
    logger.info("="*70)
    
    stacking_lr = train_stacking_ensemble(train_predictions, train_df, 'logistic')
    test_results_lr = evaluate_stacking(stacking_lr, test_predictions, test_df, 'Test (LR)')
    
    # 8. Load weighted ensemble for comparison
    logger.info("\n" + "="*70)
    logger.info("ЗАРЕЖДАНЕ НА WEIGHTED ENSEMBLE")
    logger.info("="*70)
    
    with open("models/ensemble_v1/ensemble_model.pkl", 'rb') as f:
        weighted_ensemble = pickle.load(f)
    logger.info("✓ Weighted ensemble зареден")
    
    # 9. Compare with weighted ensemble
    comparison_df = compare_with_weighted_ensemble(
        stacking_xgb,
        weighted_ensemble,
        test_predictions,
        test_df
    )
    
    # 10. Save best model
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
    logger.info(f"  1X2 Accuracy: {test_results_xgb['1x2']['accuracy']:.4f}")
    logger.info(f"  OU2.5 Accuracy: {test_results_xgb['ou25']['accuracy']:.4f}")
    logger.info(f"  BTTS Accuracy: {test_results_xgb['btts']['accuracy']:.4f}")
    
    logger.info("\nLogistic Regression Stacking (Test Set):")
    logger.info(f"  1X2 Accuracy: {test_results_lr['1x2']['accuracy']:.4f}")
    logger.info(f"  OU2.5 Accuracy: {test_results_lr['ou25']['accuracy']:.4f}")
    logger.info(f"  BTTS Accuracy: {test_results_lr['btts']['accuracy']:.4f}")
    
    logger.info("\n" + "="*70)
    logger.info("✓ STACKING ENSEMBLE TRAINING ЗАВЪРШЕН!")
    logger.info("="*70)
    logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
