"""
Training script за BTTS Model V2 (Improved)
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import joblib
from core.utils import setup_logging
from pipelines.train_ml_models import train_btts_model_v2

logger = setup_logging()


def main():
    """Main training function"""
    logger.info("=" * 70)
    logger.info("BTTS MODEL V2 - IMPROVED TRAINING")
    logger.info("=" * 70)
    
    # Load data
    logger.info("\nЗареждане на данни...")
    train_df = pd.read_parquet("data/processed/train_poisson_predictions.parquet")
    val_df = pd.read_parquet("data/processed/val_poisson_predictions.parquet")
    
    logger.info(f"Train: {len(train_df)} мача")
    logger.info(f"Val: {len(val_df)} мача")
    
    # Config (можеш да го игнорираш, използваме hardcoded params)
    config = {}
    
    # Train model
    logger.info("\nТренировка на BTTS V2...")
    model, feature_cols, metrics = train_btts_model_v2(train_df, val_df, config)
    
    # Save model
    output_dir = "models/model_btts_v2"
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"\nЗапазване на модел в {output_dir}...")
    
    # Save model
    joblib.dump(model, f"{output_dir}/btts_model.pkl")
    logger.info(f"✓ Model saved: {output_dir}/btts_model.pkl")
    
    # Save feature list
    import json
    with open(f"{output_dir}/feature_list.json", "w") as f:
        json.dump(feature_cols, f, indent=2)
    logger.info(f"✓ Features saved: {output_dir}/feature_list.json ({len(feature_cols)} features)")
    
    # Save metrics
    with open(f"{output_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"✓ Metrics saved: {output_dir}/metrics.json")
    
    logger.info("\n" + "=" * 70)
    logger.info("✓ BTTS V2 TRAINING COMPLETED SUCCESSFULLY!")
    logger.info("=" * 70)
    logger.info(f"\nVal Accuracy: {metrics['val']['accuracy']:.4f}")
    logger.info(f"Val Brier Score: {metrics['val']['brier_score']:.4f}")
    logger.info(f"\nModel saved to: {output_dir}/")


if __name__ == "__main__":
    main()
