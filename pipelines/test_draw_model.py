#!/usr/bin/env python3
"""
Test Draw Model Training - Small Dataset
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pipelines.train_draw_model import DrawModelTrainer

def main():
    print("🧪 Testing Draw Model Training on Small Dataset")
    print("=" * 60)
    
    # Create config with smaller dataset
    config = {
        'data': {
            'lookback_years': 1,  # Only 1 year instead of 3
            'feature_lookback_days': 90,  # Shorter lookback
            'test_size_months': 3  # Smaller test set
        },
        'model': {
            'lightgbm_params': {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': 15,  # Smaller tree
                'learning_rate': 0.1,  # Faster learning
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'random_state': 42
            },
            'training': {
                'n_estimators': 50,  # Fewer trees
                'early_stopping_rounds': 10,
                'calibration': True
            }
        },
        'model_dir': 'models/draw_model_v1_test'
    }
    
    # Initialize trainer with test config
    trainer = DrawModelTrainer(config)
    
    # Run training
    print("\n🚀 Starting training...")
    results = trainer.train_complete_pipeline()
    
    if results['success']:
        print("\n✅ TEST TRAINING SUCCESSFUL!")
        print(f"📊 Training Samples: {results['training_samples']}")
        print(f"📊 Test Samples: {results['test_samples']}")
        print(f"🔧 Features: {results['feature_count']}")
        
        test_metrics = results['test_metrics']
        print(f"\n📈 Test Metrics:")
        print(f"   Accuracy: {test_metrics['accuracy']:.3f}")
        print(f"   ROC AUC: {test_metrics['roc_auc']:.3f}")
        print(f"   F1 Score: {test_metrics['f1_score']:.3f}")
        print(f"   Draw Recall: {test_metrics.get('draw_recall', 0):.3f}")
        
        print(f"\n💾 Model saved to: {results['model_path']}")
        print("\n🎉 Test completed successfully! Ready for full training.")
        
        return True
    else:
        print(f"\n❌ TEST FAILED: {results.get('error', 'Unknown error')}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
