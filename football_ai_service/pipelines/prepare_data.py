"""
Pipeline за подготовка и обработка на ESPN данни
"""

import sys
import os
from pathlib import Path

# Добавяне на root директория към path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime

from core.data_loader import ESPNDataLoader
from core.utils import setup_logging, PerformanceTimer, save_json


def prepare_base_dataset(loader: ESPNDataLoader) -> pd.DataFrame:
    """
    Подготовка на основен dataset за тренировка
    
    Args:
        loader: ESPNDataLoader инстанция
    
    Returns:
        DataFrame с обработени данни
    """
    logger = setup_logging()
    
    with PerformanceTimer("Подготовка на base dataset", logger):
        # Зареждане на fixtures и stats
        df = loader.merge_fixtures_with_stats()
        
        # Премахване на мачове без статистики
        initial_count = len(df)
        df = df.dropna(subset=['possession', 'shots'])
        logger.info(f"Премахнати {initial_count - len(df)} мача без статистики")
        
        # Премахване на дублирани мачове
        df = df.drop_duplicates(subset=['match_id'])
        
        # Добавяне на допълнителни колони
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day_of_week'] = df['date'].dt.dayofweek
        
        # Сортиране по дата
        df = df.sort_values('date').reset_index(drop=True)
        
        logger.info(f"Подготвен dataset с {len(df)} мача")
        logger.info(f"Период: {df['date'].min()} до {df['date'].max()}")
        
        return df


def split_train_val_test(
    df: pd.DataFrame,
    train_end: str = "2024-06-30",
    val_end: str = "2024-09-30"
) -> tuple:
    """
    Разделяне на train/validation/test sets (хронологично)
    
    Args:
        df: DataFrame с данни
        train_end: Крайна дата за train set
        val_end: Крайна дата за validation set
    
    Returns:
        Tuple (train_df, val_df, test_df)
    """
    logger = setup_logging()
    
    train_end_date = pd.to_datetime(train_end)
    val_end_date = pd.to_datetime(val_end)
    
    train_df = df[df['date'] <= train_end_date].copy()
    val_df = df[(df['date'] > train_end_date) & (df['date'] <= val_end_date)].copy()
    test_df = df[df['date'] > val_end_date].copy()
    
    logger.info(f"Train set: {len(train_df)} мача ({train_df['date'].min()} до {train_df['date'].max()})")
    logger.info(f"Validation set: {len(val_df)} мача ({val_df['date'].min()} до {val_df['date'].max()})")
    logger.info(f"Test set: {len(test_df)} мача ({test_df['date'].min()} до {test_df['date'].max()})")
    
    return train_df, val_df, test_df


def save_processed_data(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str = "data/processed"
):
    """
    Запазване на обработените данни
    
    Args:
        train_df: Train DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame
        output_dir: Директория за запазване
    """
    logger = setup_logging()
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Запазване като Parquet (по-ефективно от CSV)
    train_path = os.path.join(output_dir, "train_data.parquet")
    val_path = os.path.join(output_dir, "val_data.parquet")
    test_path = os.path.join(output_dir, "test_data.parquet")
    
    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    test_df.to_parquet(test_path, index=False)
    
    logger.info(f"Train data запазен: {train_path}")
    logger.info(f"Validation data запазен: {val_path}")
    logger.info(f"Test data запазен: {test_path}")
    
    # Запазване на метаданни
    metadata = {
        'created_at': datetime.now().isoformat(),
        'train_size': len(train_df),
        'val_size': len(val_df),
        'test_size': len(test_df),
        'total_size': len(train_df) + len(val_df) + len(test_df),
        'train_period': {
            'start': str(train_df['date'].min()),
            'end': str(train_df['date'].max())
        },
        'val_period': {
            'start': str(val_df['date'].min()),
            'end': str(val_df['date'].max())
        },
        'test_period': {
            'start': str(test_df['date'].min()),
            'end': str(test_df['date'].max())
        },
        'columns': list(train_df.columns)
    }
    
    metadata_path = os.path.join(output_dir, "metadata.json")
    save_json(metadata, metadata_path)
    logger.info(f"Metadata запазен: {metadata_path}")


def generate_data_summary(df: pd.DataFrame) -> dict:
    """
    Генериране на обобщение на данните
    
    Args:
        df: DataFrame
    
    Returns:
        Dictionary с обобщение
    """
    summary = {
        'total_matches': len(df),
        'date_range': {
            'start': str(df['date'].min()),
            'end': str(df['date'].max())
        },
        'leagues': df['league_id'].nunique(),
        'teams': len(set(df['home_team_id'].unique()) | set(df['away_team_id'].unique())),
        'results': {
            'home_wins': (df['result'] == '1').sum(),
            'draws': (df['result'] == 'X').sum(),
            'away_wins': (df['result'] == '2').sum()
        },
        'goals': {
            'total': df['total_goals'].sum(),
            'average_per_match': df['total_goals'].mean(),
            'over_25_pct': (df['over_25'] == 1).mean() * 100
        },
        'btts': {
            'yes': (df['btts'] == 1).sum(),
            'no': (df['btts'] == 0).sum(),
            'yes_pct': (df['btts'] == 1).mean() * 100
        },
        'stats': {
            'avg_possession_home': df['possession'].mean(),
            'avg_shots_home': df['shots'].mean(),
            'avg_corners_home': df['corners'].mean()
        }
    }
    
    return summary


def main():
    """Главна функция"""
    logger = setup_logging()
    
    logger.info("=" * 70)
    logger.info("СТАРТИРАНЕ НА DATA PREPARATION PIPELINE")
    logger.info("=" * 70)
    
    # Инициализация на loader
    loader = ESPNDataLoader()
    
    # Подготовка на dataset
    df = prepare_base_dataset(loader)
    
    # Генериране на обобщение
    summary = generate_data_summary(df)
    logger.info("\n=== ОБОБЩЕНИЕ НА ДАННИТЕ ===")
    logger.info(f"Общо мачове: {summary['total_matches']}")
    logger.info(f"Период: {summary['date_range']['start']} до {summary['date_range']['end']}")
    logger.info(f"Лиги: {summary['leagues']}")
    logger.info(f"Отбори: {summary['teams']}")
    logger.info(f"Резултати - 1: {summary['results']['home_wins']}, "
                f"X: {summary['results']['draws']}, "
                f"2: {summary['results']['away_wins']}")
    logger.info(f"Средно голове на мач: {summary['goals']['average_per_match']:.2f}")
    logger.info(f"Over 2.5: {summary['goals']['over_25_pct']:.1f}%")
    logger.info(f"BTTS: {summary['btts']['yes_pct']:.1f}%")
    
    # Разделяне на train/val/test
    train_df, val_df, test_df = split_train_val_test(df)
    
    # Запазване на данните
    save_processed_data(train_df, val_df, test_df)
    
    # Запазване на обобщение
    summary_path = "data/processed/data_summary.json"
    save_json(summary, summary_path)
    logger.info(f"Data summary запазен: {summary_path}")
    
    logger.info("=" * 70)
    logger.info("DATA PREPARATION ЗАВЪРШЕН УСПЕШНО! ✓")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
