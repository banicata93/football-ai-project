#!/usr/bin/env python3
"""
Quick test of Draw Model - only 1000 matches
Tests the entire pipeline in ~5 minutes
"""

import sys
from pathlib import Path
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pipelines.train_draw_model import DrawModelTrainer

def main():
    print("🧪 QUICK TEST - Draw Model (1000 matches)")
    print("=" * 60)
    
    # Initialize trainer
    trainer = DrawModelTrainer()
    
    # Load data
    print("\n📂 Loading data...")
    df = trainer.load_and_prepare_data()
    print(f"✅ Loaded {len(df)} matches")
    
    # Take only first 1000 matches for quick test
    df_test = df.head(1000).copy()
    print(f"🎯 Using {len(df_test)} matches for quick test")
    
    # Create features (this is the slow part)
    print("\n🔧 Creating features...")
    features_df, feature_names = trainer.create_features(df_test)
    print(f"✅ Created {len(feature_names)} features for {len(features_df)} matches")
    
    if len(features_df) < 100:
        print(f"❌ Not enough data after feature creation: {len(features_df)}")
        return False
    
    # Split data
    print("\n📊 Splitting data...")
    train_df, test_df = trainer.split_data(features_df)
    print(f"✅ Train: {len(train_df)}, Test: {len(test_df)}")
    
    # Train model
    print("\n🏋️ Training model...")
    trainer.train_model(train_df, feature_names)
    print("✅ Model trained")
    
    # Evaluate
    print("\n📈 Evaluating...")
    test_metrics = trainer.evaluate_model(test_df)
    print(f"✅ Accuracy: {test_metrics['accuracy']:.3f}")
    print(f"✅ ROC AUC: {test_metrics['roc_auc']:.3f}")
    
    # Test saving
    print("\n💾 Testing save...")
    try:
        trainer.save_model()
        print("✅ Model saved successfully!")
        
        # Check files exist
        model_dir = Path('models/draw_model_v1')
        if (model_dir / 'draw_model.pkl').exists():
            print("✅ draw_model.pkl exists")
        if (model_dir / 'feature_list.json').exists():
            print("✅ feature_list.json exists")
        if (model_dir / 'metrics.json').exists():
            print("✅ metrics.json exists")
            
        print("\n🎉 ALL TESTS PASSED!")
        print("\n✅ Ready for full training!")
        return True
        
    except Exception as e:
        print(f"\n❌ SAVE FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
