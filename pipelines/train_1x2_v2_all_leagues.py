#!/usr/bin/env python3
"""
1X2 v2 Training Pipeline - ALL LEAGUES
Trains 1X2 v2 models for ALL leagues with sufficient data (like Poisson v2)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipelines.train_1x2_v2 import Train1X2V2
from core.utils import setup_logging
import logging

logger = setup_logging()


def main():
    """Train 1X2 v2 for all leagues with sufficient data"""
    
    logger.info("="*70)
    logger.info("1X2 V2 TRAINING - ALL LEAGUES")
    logger.info("="*70)
    
    # Config for all leagues
    config = {
        'lookback_years': 3,
        'min_matches_per_league': 200,  # Lower threshold for more leagues
        'poisson_time_decay': 0.8,
        'test_size': 0.2,
        'random_state': 42,
        'model_type': 'lightgbm',
        'calibration_method': 'temperature',
        'lgb_params': {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42
        }
    }
    
    # Initialize trainer
    trainer = Train1X2V2(config)
    
    # Load data
    logger.info("\n📊 Loading data...")
    df = trainer.load_and_prepare_data()
    
    # Get all leagues with sufficient data
    league_counts = df.groupby('league_id').size()
    valid_leagues = league_counts[league_counts >= config['min_matches_per_league']]
    
    logger.info(f"\n✅ Found {len(valid_leagues)} leagues with >= {config['min_matches_per_league']} matches")
    
    # Map league_id to league names
    league_mapping = {}
    for league_id in valid_leagues.index:
        league_data = df[df['league_id'] == league_id].iloc[0]
        if 'league' in league_data:
            league_name = league_data['league']
            league_mapping[league_id] = league_name
            logger.info(f"  • {league_name}: {valid_leagues[league_id]} matches")
    
    # Train for each league
    output_base = "models/leagues"
    success_count = 0
    failed_leagues = []
    
    for league_id, league_name in league_mapping.items():
        logger.info(f"\n{'='*70}")
        logger.info(f"Training 1X2 v2 for {league_name}...")
        logger.info(f"{'='*70}")
        
        output_dir = f"{output_base}/{league_name}/1x2_v2"
        
        try:
            success = trainer.train_league(league_name, output_dir)
            if success:
                success_count += 1
                logger.info(f"✅ Successfully trained {league_name}")
            else:
                failed_leagues.append(league_name)
                logger.warning(f"⚠️ Failed to train {league_name}")
        except Exception as e:
            failed_leagues.append(league_name)
            logger.error(f"❌ Error training {league_name}: {e}")
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("TRAINING SUMMARY")
    logger.info("="*70)
    logger.info(f"✅ Successfully trained: {success_count}/{len(league_mapping)} leagues")
    if failed_leagues:
        logger.info(f"❌ Failed leagues: {', '.join(failed_leagues)}")
    logger.info("="*70)


if __name__ == "__main__":
    main()
