"""
Pipeline за генериране на features от ESPN данни
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
from core.feature_engineering import FeatureEngineer
from core.utils import setup_logging, load_config, save_json, PerformanceTimer


def main():
    """Главна функция за генериране на features"""
    
    logger = setup_logging()
    
    logger.info("=" * 70)
    logger.info("СТАРТИРАНЕ НА FEATURE GENERATION PIPELINE")
    logger.info("=" * 70)
    
    # Зареждане на конфигурация
    config = load_config("config/data_config.yaml")
    
    # Инициализация на loader
    logger.info("\n[1/5] Зареждане на данни...")
    loader = ESPNDataLoader()
    
    # Зареждане и merge на fixtures + stats
    with PerformanceTimer("Merge на fixtures и stats", logger):
        df = loader.merge_fixtures_with_stats()
    
    logger.info(f"Заредени {len(df)} мача с пълни статистики")
    
    # Премахване на мачове без резултат
    initial_count = len(df)
    df = df.dropna(subset=['home_score', 'away_score'])
    logger.info(f"Премахнати {initial_count - len(df)} мача без резултат")
    
    # Сортиране по дата
    df = df.sort_values('date').reset_index(drop=True)
    
    # Инициализация на Feature Engineer
    logger.info("\n[2/5] Инициализация на Feature Engineer...")
    feature_config = config.get('feature_params', {
        'rolling_windows': [5, 10],
        'min_matches_for_stats': 3
    })
    
    engineer = FeatureEngineer(config=feature_config)
    
    # Генериране на features
    logger.info("\n[3/5] Генериране на features...")
    with PerformanceTimer("Пълен feature engineering", logger):
        df_features = engineer.create_all_features(df)
    
    logger.info(f"Генерирани {len(df_features.columns)} колони")
    
    # Премахване на редове с твърде много NaN values
    logger.info("\n[4/5] Почистване на данни...")
    
    # Изчисляване на процент липсващи стойности
    missing_pct = df_features.isnull().sum(axis=1) / len(df_features.columns)
    max_missing = config.get('quality_params', {}).get('max_missing_stats_pct', 0.3)
    
    before_clean = len(df_features)
    df_features = df_features[missing_pct <= max_missing].copy()
    after_clean = len(df_features)
    
    logger.info(f"Премахнати {before_clean - after_clean} мача с >30% липсващи данни")
    
    # Попълване на останалите NaN с 0 или медиана
    numeric_cols = df_features.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df_features[col].isnull().sum() > 0:
            # За повечето features използваме 0
            df_features[col] = df_features[col].fillna(0)
    
    # Запазване на обработените данни
    logger.info("\n[5/5] Запазване на данни с features...")
    
    output_dir = "data/processed"
    os.makedirs(output_dir, exist_ok=True)
    
    # Запазване като Parquet
    output_path = os.path.join(output_dir, "features_full.parquet")
    df_features.to_parquet(output_path, index=False)
    logger.info(f"Features запазени: {output_path}")
    
    # Разделяне на train/val/test
    split_params = config.get('split_params', {
        'train_end_date': '2024-06-30',
        'validation_end_date': '2024-09-30'
    })
    
    train_end = pd.to_datetime(split_params['train_end_date'])
    val_end = pd.to_datetime(split_params['validation_end_date'])
    
    train_df = df_features[df_features['date'] <= train_end].copy()
    val_df = df_features[(df_features['date'] > train_end) & (df_features['date'] <= val_end)].copy()
    test_df = df_features[df_features['date'] > val_end].copy()
    
    # Запазване на splits
    train_df.to_parquet(os.path.join(output_dir, "train_features.parquet"), index=False)
    val_df.to_parquet(os.path.join(output_dir, "val_features.parquet"), index=False)
    test_df.to_parquet(os.path.join(output_dir, "test_features.parquet"), index=False)
    
    logger.info(f"Train set: {len(train_df)} мача ({train_df['date'].min()} до {train_df['date'].max()})")
    logger.info(f"Validation set: {len(val_df)} мача ({val_df['date'].min()} до {val_df['date'].max()})")
    logger.info(f"Test set: {len(test_df)} мача ({test_df['date'].min()} до {test_df['date'].max()})")
    
    # Генериране на feature summary
    feature_summary = {
        'created_at': datetime.now().isoformat(),
        'total_matches': len(df_features),
        'total_features': len(df_features.columns),
        'train_size': len(train_df),
        'val_size': len(val_df),
        'test_size': len(test_df),
        'feature_columns': list(df_features.columns),
        'date_range': {
            'start': str(df_features['date'].min()),
            'end': str(df_features['date'].max())
        },
        'elo_stats': {
            'min_elo': float(df_features['home_elo_before'].min()),
            'max_elo': float(df_features['home_elo_before'].max()),
            'mean_elo': float(df_features['home_elo_before'].mean())
        },
        'form_stats': {
            'mean_home_form_5': float(df_features['home_form_5'].mean()),
            'mean_away_form_5': float(df_features['away_form_5'].mean())
        }
    }
    
    summary_path = os.path.join(output_dir, "feature_summary.json")
    save_json(feature_summary, summary_path)
    logger.info(f"Feature summary запазен: {summary_path}")
    
    # Запазване на Elo ratings
    elo_output = os.path.join(output_dir, "elo_ratings.csv")
    engineer.elo_calculator.save_ratings(elo_output)
    
    # Статистики
    logger.info("\n" + "=" * 70)
    logger.info("ОБОБЩЕНИЕ НА FEATURES")
    logger.info("=" * 70)
    logger.info(f"Общо мачове: {len(df_features)}")
    logger.info(f"Общо features: {len(df_features.columns)}")
    logger.info(f"Период: {df_features['date'].min()} до {df_features['date'].max()}")
    logger.info(f"\nElo статистики:")
    logger.info(f"  Min: {feature_summary['elo_stats']['min_elo']:.1f}")
    logger.info(f"  Max: {feature_summary['elo_stats']['max_elo']:.1f}")
    logger.info(f"  Mean: {feature_summary['elo_stats']['mean_elo']:.1f}")
    logger.info(f"\nForm статистики:")
    logger.info(f"  Mean home form (5): {feature_summary['form_stats']['mean_home_form_5']:.3f}")
    logger.info(f"  Mean away form (5): {feature_summary['form_stats']['mean_away_form_5']:.3f}")
    
    # Топ 10 отбори по Elo
    logger.info("\n" + "=" * 70)
    logger.info("ТОП 10 ОТБОРИ ПО ELO RATING")
    logger.info("=" * 70)
    
    top_teams = engineer.elo_calculator.get_top_teams(10)
    teams_df = loader.load_teams()
    
    for i, (team_id, rating) in enumerate(top_teams, 1):
        team_name = teams_df[teams_df['team_id'] == team_id]['team_name'].values
        team_name = team_name[0] if len(team_name) > 0 else f"Team {team_id}"
        logger.info(f"{i:2d}. {team_name:30s} - Elo: {rating:.1f}")
    
    logger.info("\n" + "=" * 70)
    logger.info("FEATURE GENERATION ЗАВЪРШЕН УСПЕШНО! ✓")
    logger.info("=" * 70)
    
    # Показване на примерни features
    logger.info("\n[ПРИМЕРНИ FEATURES]")
    feature_cols = engineer.get_feature_importance_columns()
    available_cols = [col for col in feature_cols if col in df_features.columns]
    
    if available_cols:
        logger.info(f"\nПоследни 3 мача с ключови features:")
        sample = df_features[['date', 'home_team_id', 'away_team_id'] + available_cols[:10]].tail(3)
        print(sample.to_string())


if __name__ == "__main__":
    main()
