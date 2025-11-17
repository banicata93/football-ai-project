#!/usr/bin/env python3
"""
Quick test of 1X2 v2 - 2 leagues to verify everything works
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipelines.train_1x2_v2 import Train1X2V2
from core.utils import setup_logging

logger = setup_logging()


def main():
    print("🧪 QUICK TEST - 1X2 v2 (2 Leagues)")
    print("="*60)
    print("Testing premier_league and la_liga to verify everything works")
    print("="*60)
    
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
    
    # Test 2 leagues
    test_leagues = ['premier_league', 'la_liga']
    success_count = 0
    
    for league in test_leagues:
        print(f"\n{'='*60}")
        print(f"🎯 Testing {league}...")
        print(f"{'='*60}")
        
        output_dir = f'models/test_1x2_v2_multi/{league}'
        
        try:
            success = trainer.train_league(league, output_dir)
            if success:
                success_count += 1
                print(f"\n✅ {league} - SUCCESS")
            else:
                print(f"\n❌ {league} - FAILED")
                return False
        except Exception as e:
            print(f"\n❌ {league} - ERROR: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # Summary
    print(f"\n{'='*60}")
    print(f"TEST SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Successful: {success_count}/2 leagues")
    
    if success_count == 2:
        print(f"\n🎉 ALL TESTS PASSED!")
        print(f"✅ Ready for full training of 8 leagues")
        return True
    else:
        print(f"\n❌ TESTS FAILED")
        print(f"⚠️ Fix issues before full training")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
