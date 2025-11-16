"""
Train ALL Leagues - Comprehensive Training Script

Trains Poisson v2 and 1X2 v2 models for ALL leagues with sufficient data
"""

import sys
from pathlib import Path
import pandas as pd
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.data_loader import ESPNDataLoader
from core.data_mapper import DataMapper
from core.poisson_v2 import PoissonV2Model
from core.utils import setup_logging
import joblib
import json
from datetime import datetime


def train_all_poisson_v2(min_matches: int = 100):
    """
    Train Poisson v2 models for ALL leagues with sufficient data
    
    Args:
        min_matches: Minimum matches required per league
    """
    logger = setup_logging()
    logger.info("="*70)
    logger.info("🚀 TRAINING POISSON V2 FOR ALL LEAGUES")
    logger.info("="*70)
    
    # Initialize
    data_loader = ESPNDataLoader()
    data_mapper = DataMapper()
    
    # Load and enrich data
    logger.info("📊 Loading fixtures...")
    df = data_loader.load_fixtures()
    df = data_mapper.enrich_fixtures(df)
    
    # Get all leagues with sufficient data
    leagues_with_data = data_mapper.get_all_leagues_with_data(df, min_matches)
    
    logger.info(f"✅ Found {len(leagues_with_data)} leagues with ≥{min_matches} matches")
    logger.info("")
    
    # Paths
    models_dir = project_root / "models" / "leagues"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Train each league
    trained = 0
    skipped = 0
    failed = 0
    
    for league_id, league_info in leagues_with_data.items():
        league_slug = league_info['slug']
        league_name = league_info['name']
        match_count = league_info['matches']
        
        logger.info(f"🎯 Training: {league_name} ({league_slug})")
        logger.info(f"   Matches: {match_count}")
        
        try:
            # Create output directory
            output_dir = models_dir / league_slug / "poisson_v2"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Check if already exists
            model_path = output_dir / "poisson_model.pkl"
            if model_path.exists():
                logger.info(f"   ⏭️  Already exists, skipping")
                skipped += 1
                continue
            
            # Filter data for this league
            league_df = df[df['league_id'] == league_id].copy()
            
            if len(league_df) < min_matches:
                logger.warning(f"   ⚠️  Insufficient data: {len(league_df)} matches")
                failed += 1
                continue
            
            # Train model
            poisson = PoissonV2Model(time_decay_factor=0.8)
            poisson.fit(league_df, league_column='league')
            
            # Save model
            joblib.dump(poisson, model_path)
            
            # Generate metrics
            metrics = {
                "league": league_slug,
                "league_name": league_name,
                "league_id": int(league_id),
                "model_type": "poisson_v2",
                "trained_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "total_matches": int(match_count),
                "time_decay_factor": 0.8,
                "note": "Time-decay Poisson model for scoreline prediction"
            }
            
            metrics_path = output_dir / "metrics.json"
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            logger.info(f"   ✅ Successfully trained!")
            trained += 1
            
        except Exception as e:
            logger.error(f"   ❌ Error: {e}")
            failed += 1
        
        logger.info("")
    
    # Summary
    logger.info("="*70)
    logger.info("📊 TRAINING COMPLETE")
    logger.info("="*70)
    logger.info(f"✅ Trained: {trained} leagues")
    logger.info(f"⏭️  Skipped: {skipped} leagues (already exist)")
    logger.info(f"❌ Failed: {failed} leagues")
    logger.info(f"📦 Total: {trained + skipped} models available")
    logger.info("="*70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train models for all leagues")
    parser.add_argument("--min-matches", type=int, default=100,
                       help="Minimum matches required per league")
    
    args = parser.parse_args()
    
    train_all_poisson_v2(min_matches=args.min_matches)
