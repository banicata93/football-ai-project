#!/usr/bin/env python3
"""
1X2 v2 Training - 8 Major Leagues
Trains 1X2 v2 models for the 8 major European leagues
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipelines.train_1x2_v2 import Train1X2V2
from core.utils import setup_logging
import time

logger = setup_logging()


def main():
    print("="*70)
    print("1X2 V2 TRAINING - 8 MAJOR LEAGUES")
    print("="*70)
    print()
    
    # Config
    config = {
        'lookback_years': 3,
        'min_matches_per_league': 200,
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
    
    # 8 major leagues
    leagues = [
        'premier_league',
        'la_liga',
        'serie_a',
        'bundesliga',
        'ligue_1',
        'eredivisie',
        'primeira_liga',
        'championship'
    ]
    
    print(f"📊 Training 1X2 v2 for {len(leagues)} leagues:")
    for i, league in enumerate(leagues, 1):
        print(f"  {i}. {league}")
    print()
    
    # Train each league
    success_count = 0
    failed_leagues = []
    start_time = time.time()
    
    for i, league in enumerate(leagues, 1):
        league_start = time.time()
        
        print(f"\n{'='*70}")
        print(f"[{i}/{len(leagues)}] Training {league}...")
        print(f"{'='*70}")
        
        output_dir = f'models/leagues/{league}/1x2_v2'
        
        try:
            success = trainer.train_league(league, output_dir)
            
            league_time = time.time() - league_start
            
            if success:
                success_count += 1
                print(f"\n✅ {league} - SUCCESS (took {league_time:.1f}s)")
            else:
                failed_leagues.append(league)
                print(f"\n❌ {league} - FAILED (took {league_time:.1f}s)")
                
        except Exception as e:
            league_time = time.time() - league_start
            failed_leagues.append(league)
            print(f"\n❌ {league} - ERROR: {e} (took {league_time:.1f}s)")
            import traceback
            traceback.print_exc()
    
    # Summary
    total_time = time.time() - start_time
    
    print(f"\n{'='*70}")
    print(f"TRAINING SUMMARY")
    print(f"{'='*70}")
    print(f"✅ Successful: {success_count}/{len(leagues)} leagues")
    print(f"⏱️  Total time: {total_time/60:.1f} minutes")
    
    if failed_leagues:
        print(f"\n❌ Failed leagues:")
        for league in failed_leagues:
            print(f"  • {league}")
    else:
        print(f"\n🎉 ALL LEAGUES TRAINED SUCCESSFULLY!")
    
    print(f"{'='*70}")
    
    return success_count == len(leagues)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
