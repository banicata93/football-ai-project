#!/usr/bin/env python3
"""
Quick test of 1X2 v2 - single league, small dataset
Tests the entire pipeline in ~5 minutes
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pipelines.train_1x2_v2 import Train1X2V2

def main():
    print("🧪 QUICK TEST - 1X2 v2 (Single League)")
    print("=" * 60)
    
    # Test config - only premier_league
    config = {
        'lookback_years': 1,  # Only 1 year
        'min_matches_per_league': 50,  # Lower threshold
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
        },
        'model_dir': 'models/test_1x2_v2'
    }
    
    # Initialize trainer
    trainer = Train1X2V2(config)
    
    # Train only premier_league
    print("\n🎯 Training premier_league only...")
    success = trainer.train_league('premier_league', 'models/test_1x2_v2')
    
    if success:
        print("\n🎉 TEST PASSED!")
        print("✅ premier_league trained successfully")
        print("✅ No merge errors")
        print("✅ Model saved")
        print("\n✅ Ready for full training!")
        return True
    else:
        print("\n❌ TEST FAILED")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
