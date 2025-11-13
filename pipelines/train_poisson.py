"""
Training pipeline за Poisson baseline модел
"""

import sys
import os
from pathlib import Path

# Добавяне на root директория към path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import joblib
from datetime import datetime

from core.poisson_utils import PoissonModel
from core.utils import setup_logging, load_config, save_json, PerformanceTimer


def train_poisson_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    config: dict
) -> PoissonModel:
    """
    Тренировка на Poisson модел
    
    Args:
        train_df: Train dataset
        val_df: Validation dataset
        config: Конфигурация
    
    Returns:
        Обучен PoissonModel
    """
    logger = setup_logging()
    
    logger.info("=" * 70)
    logger.info("ТРЕНИРОВКА НА POISSON BASELINE MODEL")
    logger.info("=" * 70)
    
    # Извличане на параметри от конфигурацията
    poisson_config = config.get('poisson', {
        'home_advantage_factor': 1.15,
        'league_normalization': True,
        'min_matches_for_lambda': 5
    })
    
    # Инициализация на модел
    model = PoissonModel(
        home_advantage=poisson_config['home_advantage_factor'],
        league_normalization=poisson_config['league_normalization'],
        min_matches=poisson_config['min_matches_for_lambda']
    )
    
    # Изчисляване на team strengths от train data
    with PerformanceTimer("Изчисляване на team strengths", logger):
        model.calculate_team_strengths(train_df)
    
    logger.info(f"Attack strengths изчислени за {len(model.attack_strength)} отбора")
    logger.info(f"Defense strengths изчислени за {len(model.defense_strength)} отбора")
    
    # Статистики за strengths
    attack_values = list(model.attack_strength.values())
    defense_values = list(model.defense_strength.values())
    
    logger.info(f"\nAttack Strength статистики:")
    logger.info(f"  Min: {min(attack_values):.3f}")
    logger.info(f"  Max: {max(attack_values):.3f}")
    logger.info(f"  Mean: {np.mean(attack_values):.3f}")
    logger.info(f"  Median: {np.median(attack_values):.3f}")
    
    logger.info(f"\nDefense Strength статистики:")
    logger.info(f"  Min: {min(defense_values):.3f}")
    logger.info(f"  Max: {max(defense_values):.3f}")
    logger.info(f"  Mean: {np.mean(defense_values):.3f}")
    logger.info(f"  Median: {np.median(defense_values):.3f}")
    
    # League averages
    logger.info(f"\nLeague averages:")
    for league_id, avg in list(model.league_avg_goals_home.items())[:5]:
        avg_away = model.league_avg_goals_away.get(league_id, 0)
        logger.info(f"  League {league_id}: Home {avg:.2f}, Away {avg_away:.2f}")
    
    return model


def evaluate_model(
    model: PoissonModel,
    df: pd.DataFrame,
    dataset_name: str = "Dataset"
) -> dict:
    """
    Оценка на модела
    
    Args:
        model: Обучен Poisson модел
        df: Dataset за оценка
        dataset_name: Име на dataset
    
    Returns:
        Dictionary с метрики
    """
    logger = setup_logging()
    
    logger.info(f"\n{'=' * 70}")
    logger.info(f"ОЦЕНКА НА МОДЕЛА - {dataset_name}")
    logger.info(f"{'=' * 70}")
    
    # Прогнозиране
    with PerformanceTimer(f"Прогнозиране за {dataset_name}", logger):
        df_pred = model.predict_dataset(df)
    
    # Оценка
    metrics = model.evaluate_predictions(df_pred)
    
    # Показване на метрики
    logger.info(f"\n{dataset_name} Метрики:")
    logger.info(f"  Accuracy 1X2:        {metrics['accuracy_1x2']:.4f} ({metrics['accuracy_1x2']*100:.2f}%)")
    logger.info(f"  Accuracy Over/Under: {metrics['accuracy_over25']:.4f} ({metrics['accuracy_over25']*100:.2f}%)")
    logger.info(f"  Accuracy BTTS:       {metrics['accuracy_btts']:.4f} ({metrics['accuracy_btts']*100:.2f}%)")
    logger.info(f"  Log Loss 1X2:        {metrics['log_loss_1x2']:.4f}")
    logger.info(f"  Log Loss Over/Under: {metrics['log_loss_over25']:.4f}")
    logger.info(f"  Log Loss BTTS:       {metrics['log_loss_btts']:.4f}")
    logger.info(f"  Mean Expected Goals: {metrics['mean_expected_goals']:.2f}")
    
    return metrics, df_pred


def save_model(
    model: PoissonModel,
    metrics: dict,
    output_dir: str = "models/model_poisson_v1"
):
    """
    Запазване на модела
    
    Args:
        model: Обучен модел
        metrics: Метрики
        output_dir: Директория за запазване
    """
    logger = setup_logging()
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Запазване на модела
    model_path = os.path.join(output_dir, "poisson_model.pkl")
    joblib.dump(model, model_path)
    logger.info(f"Модел запазен: {model_path}")
    
    # Запазване на метрики
    metrics_path = os.path.join(output_dir, "metrics.json")
    save_json(metrics, metrics_path)
    logger.info(f"Метрики запазени: {metrics_path}")
    
    # Запазване на model info
    model_info = {
        'model_type': 'poisson',
        'version': 'v1',
        'trained_date': datetime.now().isoformat(),
        'home_advantage': model.home_advantage,
        'league_normalization': model.league_normalization,
        'min_matches': model.min_matches,
        'num_teams': len(model.attack_strength),
        'num_leagues': len(model.league_avg_goals_home)
    }
    
    info_path = os.path.join(output_dir, "model_info.json")
    save_json(model_info, info_path)
    logger.info(f"Model info запазен: {info_path}")


def analyze_predictions(df_pred: pd.DataFrame, top_n: int = 10):
    """
    Анализ на прогнозите
    
    Args:
        df_pred: DataFrame с прогнози
        top_n: Брой примери
    """
    logger = setup_logging()
    
    logger.info(f"\n{'=' * 70}")
    logger.info("АНАЛИЗ НА ПРОГНОЗИТЕ")
    logger.info(f"{'=' * 70}")
    
    # Най-уверени прогнози за home win
    df_sorted = df_pred.sort_values('poisson_prob_1', ascending=False)
    logger.info(f"\nНай-уверени прогнози за ПОБЕДА НА ДОМАКИНА:")
    for idx, row in df_sorted.head(top_n).iterrows():
        logger.info(
            f"  Home {row['home_team_id']} vs Away {row['away_team_id']}: "
            f"P(1)={row['poisson_prob_1']:.3f}, "
            f"Expected: {row['poisson_lambda_home']:.2f}-{row['poisson_lambda_away']:.2f}, "
            f"Actual: {row['home_score']:.0f}-{row['away_score']:.0f}"
        )
    
    # Най-уверени прогнози за draw
    df_sorted = df_pred.sort_values('poisson_prob_x', ascending=False)
    logger.info(f"\nНай-уверени прогнози за РАВЕНСТВО:")
    for idx, row in df_sorted.head(top_n).iterrows():
        logger.info(
            f"  Home {row['home_team_id']} vs Away {row['away_team_id']}: "
            f"P(X)={row['poisson_prob_x']:.3f}, "
            f"Expected: {row['poisson_lambda_home']:.2f}-{row['poisson_lambda_away']:.2f}, "
            f"Actual: {row['home_score']:.0f}-{row['away_score']:.0f}"
        )
    
    # Най-уверени прогнози за over 2.5
    df_sorted = df_pred.sort_values('poisson_prob_over25', ascending=False)
    logger.info(f"\nНай-уверени прогнози за OVER 2.5:")
    for idx, row in df_sorted.head(top_n).iterrows():
        logger.info(
            f"  Home {row['home_team_id']} vs Away {row['away_team_id']}: "
            f"P(Over)={row['poisson_prob_over25']:.3f}, "
            f"Expected total: {row['poisson_expected_goals']:.2f}, "
            f"Actual total: {row['total_goals']:.0f}"
        )
    
    # Разпределение на expected goals
    logger.info(f"\nРазпределение на EXPECTED GOALS:")
    logger.info(f"  Mean: {df_pred['poisson_expected_goals'].mean():.2f}")
    logger.info(f"  Median: {df_pred['poisson_expected_goals'].median():.2f}")
    logger.info(f"  Std: {df_pred['poisson_expected_goals'].std():.2f}")
    logger.info(f"  Min: {df_pred['poisson_expected_goals'].min():.2f}")
    logger.info(f"  Max: {df_pred['poisson_expected_goals'].max():.2f}")


def main():
    """Главна функция"""
    logger = setup_logging()
    
    logger.info("=" * 70)
    logger.info("СТАРТИРАНЕ НА POISSON MODEL TRAINING PIPELINE")
    logger.info("=" * 70)
    
    # Зареждане на конфигурация
    config = load_config("config/model_config.yaml")
    
    # Зареждане на данни
    logger.info("\n[1/5] Зареждане на данни...")
    
    train_df = pd.read_parquet("data/processed/train_features.parquet")
    val_df = pd.read_parquet("data/processed/val_features.parquet")
    test_df = pd.read_parquet("data/processed/test_features.parquet")
    
    logger.info(f"Train: {len(train_df)} мача")
    logger.info(f"Validation: {len(val_df)} мача")
    logger.info(f"Test: {len(test_df)} мача")
    
    # Тренировка на модел
    logger.info("\n[2/5] Тренировка на Poisson модел...")
    model = train_poisson_model(train_df, val_df, config)
    
    # Оценка на train set
    logger.info("\n[3/5] Оценка на train set...")
    train_metrics, train_pred = evaluate_model(model, train_df, "TRAIN")
    
    # Оценка на validation set
    logger.info("\n[4/5] Оценка на validation set...")
    val_metrics, val_pred = evaluate_model(model, val_df, "VALIDATION")
    
    # Оценка на test set
    logger.info("\n[5/5] Оценка на test set...")
    test_metrics, test_pred = evaluate_model(model, test_df, "TEST")
    
    # Запазване на модела
    all_metrics = {
        'train': train_metrics,
        'validation': val_metrics,
        'test': test_metrics
    }
    
    save_model(model, all_metrics)
    
    # Запазване на predictions
    logger.info("\nЗапазване на predictions...")
    train_pred.to_parquet("data/processed/train_poisson_predictions.parquet", index=False)
    val_pred.to_parquet("data/processed/val_poisson_predictions.parquet", index=False)
    test_pred.to_parquet("data/processed/test_poisson_predictions.parquet", index=False)
    
    # Анализ на прогнозите
    analyze_predictions(val_pred, top_n=5)
    
    # Финално обобщение
    logger.info("\n" + "=" * 70)
    logger.info("ОБОБЩЕНИЕ НА РЕЗУЛТАТИТЕ")
    logger.info("=" * 70)
    logger.info(f"\nTRAIN SET:")
    logger.info(f"  Accuracy 1X2: {train_metrics['accuracy_1x2']*100:.2f}%")
    logger.info(f"  Log Loss 1X2: {train_metrics['log_loss_1x2']:.4f}")
    
    logger.info(f"\nVALIDATION SET:")
    logger.info(f"  Accuracy 1X2: {val_metrics['accuracy_1x2']*100:.2f}%")
    logger.info(f"  Log Loss 1X2: {val_metrics['log_loss_1x2']:.4f}")
    
    logger.info(f"\nTEST SET:")
    logger.info(f"  Accuracy 1X2: {test_metrics['accuracy_1x2']*100:.2f}%")
    logger.info(f"  Log Loss 1X2: {test_metrics['log_loss_1x2']:.4f}")
    
    logger.info("\n" + "=" * 70)
    logger.info("POISSON MODEL TRAINING ЗАВЪРШЕН УСПЕШНО! ✓")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
