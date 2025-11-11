"""
ML Utilities - Помощни функции за ML модели
"""

import logging
from typing import List, Dict, Tuple, Optional, Any

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, log_loss, roc_auc_score, 
    brier_score_loss, classification_report
)
from sklearn.calibration import CalibratedClassifierCV

from .utils import setup_logging

logger = setup_logging()


def align_features(X: pd.DataFrame, feature_list: List[str], fill_value: float = 0.0) -> pd.DataFrame:
    """
    Align features to match training data
    
    Ensures that:
    - All required features are present (fills missing with fill_value)
    - Features are in the same order as training
    - Extra features are removed
    
    Args:
        X: Input DataFrame with features
        feature_list: List of feature names from training
        fill_value: Value to fill missing features (default: 0.0)
    
    Returns:
        Aligned DataFrame with exact same features as training
    """
    # Check for missing and extra features
    missing = set(feature_list) - set(X.columns)
    extra = set(X.columns) - set(feature_list)
    
    if missing:
        logger.warning(f"Missing features (will fill with {fill_value}): {sorted(missing)}")
    
    if extra:
        logger.info(f"Extra features (will be ignored): {sorted(extra)}")
    
    # Reindex to match training features exactly
    X_aligned = X.reindex(columns=feature_list, fill_value=fill_value)
    
    logger.info(f"Feature alignment: {len(X.columns)} → {len(X_aligned.columns)} features")
    
    return X_aligned


def add_btts_specific_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавя BTTS-specific features
    
    Args:
        df: DataFrame с базови features
    
    Returns:
        DataFrame с добавени BTTS features
    """
    df = df.copy()
    
    # 1. Clean sheet rates (вероятност за 0 допуснати голове)
    df['home_clean_sheet_rate'] = (df['home_goals_conceded_avg_5'] < 0.5).astype(float)
    df['away_clean_sheet_rate'] = (df['away_goals_conceded_avg_5'] < 0.5).astype(float)
    
    # 2. Attack correlation (и двата атакуват)
    df['attack_correlation'] = (
        df['home_goals_scored_avg_5'] * df['away_goals_scored_avg_5']
    )
    
    # 3. Defense correlation (и двата допускат голове)
    df['defense_correlation'] = (
        df['home_goals_conceded_avg_5'] * df['away_goals_conceded_avg_5']
    )
    
    # 4. Form difference (absolute)
    df['form_diff_abs'] = np.abs(df['home_form_5'] - df['away_form_5'])
    
    # 5. Both teams scoring indicator (ако и двата отбелязват редовно)
    df['both_teams_scoring_indicator'] = (
        (df['home_goals_scored_avg_5'] > 0.8).astype(int) +
        (df['away_goals_scored_avg_5'] > 0.8).astype(int)
    )
    
    # 6. Defensive weakness sum
    df['defensive_weakness_sum'] = (
        df['home_goals_conceded_avg_5'] + df['away_goals_conceded_avg_5']
    )
    
    # 7. Attacking strength sum
    df['attacking_strength_sum'] = (
        df['home_goals_scored_avg_5'] + df['away_goals_scored_avg_5']
    )
    
    return df


def get_feature_columns(exclude_cols: Optional[List[str]] = None) -> List[str]:
    """
    Получаване на списък с feature колони
    
    Args:
        exclude_cols: Колони за изключване
    
    Returns:
        List от feature колони
    """
    # Основни features
    base_features = [
        # Elo features
        'home_elo_before', 'away_elo_before', 'elo_diff', 'elo_diff_normalized',
        
        # Form features
        'home_form_5', 'away_form_5', 'form_diff_5',
        
        # Goal stats (5 games)
        'home_goals_scored_avg_5', 'home_goals_conceded_avg_5',
        'away_goals_scored_avg_5', 'away_goals_conceded_avg_5',
        
        # Goal stats (10 games)
        'home_goals_scored_avg_10', 'home_goals_conceded_avg_10',
        'away_goals_scored_avg_10', 'away_goals_conceded_avg_10',
        
        # Efficiency features
        'home_shooting_efficiency', 'away_shooting_efficiency',
        'home_xg_proxy', 'away_xg_proxy',
        
        # Rest & Momentum
        'home_rest_days', 'away_rest_days', 'rest_advantage',
        'home_momentum', 'away_momentum',
        
        # Context
        'is_home', 'is_weekend', 'month', 'day_of_week',
        
        # Poisson features (ако има)
        'poisson_lambda_home', 'poisson_lambda_away',
        'poisson_prob_1', 'poisson_prob_x', 'poisson_prob_2',
        'poisson_prob_over25', 'poisson_prob_btts',
        'poisson_expected_goals'
    ]
    
    # Rolling stats features
    rolling_features = []
    for window in [5, 10]:
        for stat in ['possession', 'shots', 'shots_on_target', 'corners', 
                     'fouls', 'yellow_cards', 'pass_accuracy', 'tackles', 'interceptions']:
            rolling_features.append(f'home_{stat}_avg_{window}')
            rolling_features.append(f'away_{stat}_avg_{window}')
    
    all_features = base_features + rolling_features
    
    # Изключване на определени колони
    if exclude_cols:
        all_features = [f for f in all_features if f not in exclude_cols]
    
    return all_features


def get_btts_feature_columns() -> List[str]:
    """
    Получаване на feature колони специфични за BTTS модел
    
    Returns:
        List от BTTS feature колони
    """
    # Базови features (без Poisson features освен poisson_prob_btts)
    base_features = get_feature_columns(exclude_cols=[
        'poisson_lambda_home', 'poisson_lambda_away',
        'poisson_prob_1', 'poisson_prob_x', 'poisson_prob_2',
        'poisson_prob_over25', 'poisson_expected_goals'
    ])
    
    # BTTS-specific features
    btts_specific = [
        'home_clean_sheet_rate',
        'away_clean_sheet_rate',
        'attack_correlation',
        'defense_correlation',
        'form_diff_abs',
        'both_teams_scoring_indicator',
        'defensive_weakness_sum',
        'attacking_strength_sum'
    ]
    
    return base_features + btts_specific


def prepare_features(
    df: pd.DataFrame,
    feature_cols: List[str],
    fill_na: bool = True
) -> pd.DataFrame:
    """
    Подготовка на features за ML модел
    
    Args:
        df: DataFrame с данни
        feature_cols: Списък с feature колони
        fill_na: Дали да попълним NaN стойности
    
    Returns:
        DataFrame само с features
    """
    # Филтриране само на съществуващи колони
    available_cols = [col for col in feature_cols if col in df.columns]
    
    X = df[available_cols].copy()
    
    if fill_na:
        # Попълване на NaN с 0
        X = X.fillna(0)
    
    # Замяна на inf/-inf с големи числа
    X = X.replace([np.inf, -np.inf], [1e10, -1e10])
    
    # Clip на екстремни стойности
    for col in X.columns:
        if X[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
            # Clip на 99.9 percentile
            upper = X[col].quantile(0.999)
            lower = X[col].quantile(0.001)
            X[col] = X[col].clip(lower, upper)
    
    return X


def evaluate_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: Optional[np.ndarray] = None,
    labels: Optional[List[str]] = None,
    model_name: str = "Model"
) -> Dict[str, Any]:
    """
    Оценка на classification модел
    
    Args:
        y_true: Истински labels
        y_pred: Предсказани labels
        y_pred_proba: Предсказани вероятности
        labels: Имена на класовете
        model_name: Име на модела
    
    Returns:
        Dictionary с метрики
    """
    logger = setup_logging()
    
    metrics = {}
    
    # Accuracy
    accuracy = accuracy_score(y_true, y_pred)
    metrics['accuracy'] = accuracy
    
    logger.info(f"{model_name} Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Classification report
    if labels:
        report = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
        metrics['classification_report'] = report
        
        logger.info(f"\n{classification_report(y_true, y_pred, target_names=labels)}")
    
    # Probabilistic metrics
    if y_pred_proba is not None:
        # Log Loss
        try:
            logloss = log_loss(y_true, y_pred_proba)
            metrics['log_loss'] = logloss
            logger.info(f"{model_name} Log Loss: {logloss:.4f}")
        except:
            pass
        
        # Brier Score (за binary)
        if y_pred_proba.ndim == 1 or y_pred_proba.shape[1] == 2:
            try:
                if y_pred_proba.ndim == 2:
                    y_pred_proba_binary = y_pred_proba[:, 1]
                else:
                    y_pred_proba_binary = y_pred_proba
                
                brier = brier_score_loss(y_true, y_pred_proba_binary)
                metrics['brier_score'] = brier
                logger.info(f"{model_name} Brier Score: {brier:.4f}")
            except:
                pass
        
        # ROC AUC (за binary)
        if len(np.unique(y_true)) == 2:
            try:
                if y_pred_proba.ndim == 2:
                    y_pred_proba_binary = y_pred_proba[:, 1]
                else:
                    y_pred_proba_binary = y_pred_proba
                
                roc_auc = roc_auc_score(y_true, y_pred_proba_binary)
                metrics['roc_auc'] = roc_auc
                logger.info(f"{model_name} ROC AUC: {roc_auc:.4f}")
            except:
                pass
    
    return metrics


def get_feature_importance(
    model: Any,
    feature_names: List[str],
    top_n: int = 20
) -> pd.DataFrame:
    """
    Извличане на feature importance
    
    Args:
        model: Обучен модел (XGBoost, LightGBM, etc.)
        feature_names: Имена на features
        top_n: Брой топ features
    
    Returns:
        DataFrame с feature importance
    """
    logger = setup_logging()
    
    try:
        # XGBoost/LightGBM
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'get_score'):
            # XGBoost get_score
            importance_dict = model.get_score(importance_type='gain')
            importances = [importance_dict.get(f'f{i}', 0) for i in range(len(feature_names))]
        else:
            logger.warning("Моделът няма feature_importances_")
            return pd.DataFrame()
        
        # Създаване на DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        })
        
        # Сортиране
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        # Топ N
        top_features = importance_df.head(top_n)
        
        logger.info(f"\nТоп {top_n} най-важни features:")
        for idx, row in top_features.iterrows():
            logger.info(f"  {row['feature']:40s}: {row['importance']:.4f}")
        
        return importance_df
    
    except Exception as e:
        logger.error(f"Грешка при извличане на feature importance: {e}")
        return pd.DataFrame()


def calibrate_probabilities(
    model: Any,
    X_cal: pd.DataFrame,
    y_cal: np.ndarray,
    method: str = 'isotonic'
) -> Any:
    """
    Калибриране на вероятности
    
    Args:
        model: Обучен модел
        X_cal: Calibration features
        y_cal: Calibration labels
        method: 'isotonic' или 'sigmoid'
    
    Returns:
        Калибриран модел
    """
    logger = setup_logging()
    
    logger.info(f"Калибриране на модела с метод: {method}")
    
    calibrated_model = CalibratedClassifierCV(
        model,
        method=method,
        cv='prefit'
    )
    
    calibrated_model.fit(X_cal, y_cal)
    
    logger.info("Калибриране завършено")
    
    return calibrated_model


def encode_target_1x2(result_series: pd.Series) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Encode на 1X2 резултат
    
    Args:
        result_series: Series с резултати ('1', 'X', '2')
    
    Returns:
        Tuple (encoded array, label mapping)
    """
    label_map = {'1': 0, 'X': 1, '2': 2}
    encoded = result_series.map(label_map).values
    
    return encoded, label_map


def decode_target_1x2(encoded_array: np.ndarray) -> np.ndarray:
    """
    Decode на 1X2 резултат
    
    Args:
        encoded_array: Encoded array (0, 1, 2)
    
    Returns:
        Array с резултати ('1', 'X', '2')
    """
    reverse_map = {0: '1', 1: 'X', 2: '2'}
    decoded = np.array([reverse_map[val] for val in encoded_array])
    
    return decoded


class ModelTracker:
    """Клас за проследяване на метрики на модели"""
    
    def __init__(self):
        self.models = {}
        self.logger = setup_logging()
    
    def add_model(
        self,
        model_name: str,
        train_metrics: Dict,
        val_metrics: Dict,
        test_metrics: Optional[Dict] = None
    ):
        """
        Добавяне на модел и неговите метрики
        
        Args:
            model_name: Име на модела
            train_metrics: Train метрики
            val_metrics: Validation метрики
            test_metrics: Test метрики (optional)
        """
        self.models[model_name] = {
            'train': train_metrics,
            'validation': val_metrics,
            'test': test_metrics
        }
        
        self.logger.info(f"Модел '{model_name}' добавен в tracker")
    
    def compare_models(self, metric: str = 'accuracy'):
        """
        Сравнение на модели по дадена метрика
        
        Args:
            metric: Метрика за сравнение
        """
        self.logger.info(f"\n{'=' * 70}")
        self.logger.info(f"СРАВНЕНИЕ НА МОДЕЛИ ПО {metric.upper()}")
        self.logger.info(f"{'=' * 70}")
        
        for model_name, metrics in self.models.items():
            train_val = metrics['train'].get(metric, 'N/A')
            val_val = metrics['validation'].get(metric, 'N/A')
            test_val = metrics['test'].get(metric, 'N/A') if metrics['test'] else 'N/A'
            
            self.logger.info(
                f"{model_name:20s} - Train: {train_val}, Val: {val_val}, Test: {test_val}"
            )
    
    def get_best_model(self, metric: str = 'accuracy', dataset: str = 'validation') -> str:
        """
        Намиране на най-добрия модел
        
        Args:
            metric: Метрика за сравнение
            dataset: 'train', 'validation' или 'test'
        
        Returns:
            Име на най-добрия модел
        """
        best_model = None
        best_score = -float('inf')
        
        for model_name, metrics in self.models.items():
            score = metrics[dataset].get(metric, -float('inf'))
            if score > best_score:
                best_score = score
                best_model = model_name
        
        self.logger.info(f"Най-добър модел по {metric} на {dataset}: {best_model} ({best_score:.4f})")
        
        return best_model


if __name__ == "__main__":
    print("=== ML Utils Test ===")
    print("Модулът е готов за използване!")
