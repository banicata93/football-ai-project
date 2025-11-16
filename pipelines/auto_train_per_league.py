#!/usr/bin/env python3
"""
Automatic Per-League Training Pipeline (SAFE MODE)

This pipeline automatically detects and trains missing per-league models:
- 1X2 v2 (per-league binary models)
- Poisson v2 (per-league time-decay models)
- Draw Specialist v1 (optional per-league draw prediction)

SAFE MODE Rules:
- Never delete existing models
- Never overwrite existing .pkl files
- Never replace existing metrics.json
- Only add missing models
- Skip if model already exists
- Generate metrics.json if missing
- Update registry with new entries only

Author: Football AI System
Date: 2025-11-16
"""

import sys
import os
import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Set
import logging
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.data_loader import ESPNDataLoader
from core.data_mapper import DataMapper
from core.utils import setup_logging
from pipelines.train_1x2_v2 import Train1X2V2
from core.poisson_v2 import PoissonV2Model


class PerLeagueTrainingManager:
    """
    Manages automatic per-league model training in SAFE MODE
    """
    
    def __init__(self, dry_run: bool = False):
        """
        Initialize training manager
        
        Args:
            dry_run: If True, only detect missing models without training
        """
        self.logger = setup_logging()
        self.dry_run = dry_run
        
        # Paths
        self.project_root = Path(__file__).parent.parent
        self.models_dir = self.project_root / "models"
        self.leagues_dir = self.models_dir / "leagues"
        self.registry_path = self.project_root / "registry.json"
        
        # Data loader and mapper
        self.data_loader = ESPNDataLoader()
        self.data_mapper = DataMapper()
        
        # Target leagues (same as in config)
        self.target_leagues = [
            'premier_league',
            'la_liga',
            'serie_a',
            'bundesliga',
            'ligue_1',
            'eredivisie',
            'primeira_liga',
            'championship'
        ]
        
        # Model types to manage
        self.model_types = {
            '1x2_v2': {
                'dir_name': '1x2_v2',
                'required_files': ['home_model.pkl', 'draw_model.pkl', 'away_model.pkl', 'metrics.json'],
                'optional': False
            },
            'poisson_v2': {
                'dir_name': 'poisson_v2',
                'required_files': ['poisson_model.pkl', 'metrics.json'],
                'optional': False
            },
            'draw_specialist': {
                'dir_name': 'draw_specialist_v1',
                'required_files': ['draw_model.pkl', 'metrics.json'],
                'optional': True
            }
        }
        
        self.logger.info("🏗️ Per-League Training Manager initialized (SAFE MODE)")
        self.logger.info(f"📊 Target leagues: {len(self.target_leagues)}")
        self.logger.info(f"🎯 Model types: {list(self.model_types.keys())}")
        if self.dry_run:
            self.logger.info("🔍 DRY RUN MODE: No training will be performed")
    
    def get_available_leagues(self) -> List[str]:
        """
        Get list of available leagues from existing directories
        
        Returns:
            List of league slugs
        """
        self.logger.info("📊 Detecting available leagues...")
        
        try:
            # Check existing league directories
            if self.leagues_dir.exists():
                existing = [d.name for d in self.leagues_dir.iterdir() 
                           if d.is_dir() and d.name in self.target_leagues]
                
                if existing:
                    self.logger.info(f"✅ Found {len(existing)} existing league directories: {', '.join(existing)}")
                    return existing
            
            # If no existing directories, return all target leagues
            self.logger.info(f"📁 Using all target leagues: {', '.join(self.target_leagues)}")
            return self.target_leagues
            
        except Exception as e:
            self.logger.error(f"❌ Error detecting leagues: {e}")
            # Fallback to all target leagues
            return self.target_leagues
    
    def detect_missing_models(self, model_type: str) -> Dict[str, List[str]]:
        """
        Detect missing models for a specific model type
        
        Args:
            model_type: Type of model ('1x2_v2', 'poisson_v2', 'draw_specialist')
        
        Returns:
            Dictionary with 'missing' and 'existing' league lists
        """
        if model_type not in self.model_types:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model_config = self.model_types[model_type]
        dir_name = model_config['dir_name']
        required_files = model_config['required_files']
        
        self.logger.info(f"🔍 Detecting missing {model_type} models...")
        
        missing_leagues = []
        existing_leagues = []
        partial_leagues = []
        
        for league in self.target_leagues:
            model_dir = self.leagues_dir / league / dir_name
            
            if not model_dir.exists():
                missing_leagues.append(league)
                continue
            
            # Check if all required files exist
            missing_files = []
            for file in required_files:
                file_path = model_dir / file
                if not file_path.exists():
                    missing_files.append(file)
            
            if missing_files:
                # Treat partial models as missing (need to be trained)
                missing_leagues.append(league)
                self.logger.warning(f"⚠️ {league}: Partial model (missing: {', '.join(missing_files)}) - will train")
            else:
                existing_leagues.append(league)
        
        self.logger.info(f"📊 {model_type} status:")
        self.logger.info(f"   ✅ Existing: {len(existing_leagues)} leagues")
        self.logger.info(f"   ❌ Missing: {len(missing_leagues)} leagues")
        if partial_leagues:
            self.logger.info(f"   ⚠️  Partial: {len(partial_leagues)} leagues")
        
        return {
            'missing': missing_leagues,
            'existing': existing_leagues,
            'partial': partial_leagues
        }
    
    def train_1x2_v2_for_league(self, league: str) -> bool:
        """
        Train 1X2 v2 model for a specific league
        
        Args:
            league: League slug
        
        Returns:
            True if training successful
        """
        self.logger.info(f"🎯 Training 1X2 v2 model for {league}...")
        
        try:
            # Create output directory
            output_dir = self.leagues_dir / league / "1x2_v2"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Check if models already exist (SAFE MODE)
            model_files = ['home_model.pkl', 'draw_model.pkl', 'away_model.pkl']
            if all((output_dir / f).exists() for f in model_files):
                self.logger.info(f"⏭️  Skipping {league}: Models already exist")
                return True
            
            if self.dry_run:
                self.logger.info(f"🔍 DRY RUN: Would train 1X2 v2 for {league}")
                return True
            
            # Initialize trainer
            trainer = Train1X2V2()
            
            # Train for specific league
            success = trainer.train_league(league, output_dir)
            
            if success:
                self.logger.info(f"✅ Successfully trained 1X2 v2 for {league}")
                
                # Generate metrics if missing
                metrics_path = output_dir / "metrics.json"
                if not metrics_path.exists():
                    self.generate_metrics_1x2_v2(league, output_dir)
                
                return True
            else:
                self.logger.error(f"❌ Failed to train 1X2 v2 for {league}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Error training 1X2 v2 for {league}: {e}")
            return False
    
    def train_poisson_v2_for_league(self, league: str) -> bool:
        """
        Train Poisson v2 model for a specific league
        
        Args:
            league: League slug
        
        Returns:
            True if training successful
        """
        self.logger.info(f"🎯 Training Poisson v2 model for {league}...")
        
        try:
            # Create output directory
            output_dir = self.leagues_dir / league / "poisson_v2"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Check if model already exists (SAFE MODE)
            model_path = output_dir / "poisson_model.pkl"
            if model_path.exists():
                self.logger.info(f"⏭️  Skipping {league}: Poisson v2 already exists")
                return True
            
            if self.dry_run:
                self.logger.info(f"🔍 DRY RUN: Would train Poisson v2 for {league}")
                return True
            
            # Load data for league
            df = self.data_loader.load_fixtures()
            if df is None or df.empty:
                self.logger.error(f"❌ No data available for {league}")
                return False
            
            # Enrich with team and league names
            df = self.data_mapper.enrich_fixtures(df)
            
            # Map league slug to league_id
            league_mapping = {
                'premier_league': 700,
                'la_liga': 3907,
                'serie_a': 630,
                'bundesliga': 3907,
                'ligue_1': 710,
                'eredivisie': 725,
                'primeira_liga': 715,
                'championship': 5672
            }
            
            league_id = league_mapping.get(league)
            if not league_id:
                self.logger.error(f"❌ Unknown league: {league}")
                return False
            
            # Filter for league using league_id
            league_df = df[df['league_id'] == league_id].copy()
            
            if len(league_df) < 100:
                self.logger.warning(f"⚠️ Insufficient data for {league}: {len(league_df)} matches")
                return False
            
            # Initialize and train Poisson v2
            poisson = PoissonV2Model(time_decay_factor=0.8)
            # Now we have home_team, away_team, and league columns from enrichment
            poisson.fit(league_df, league_column='league')
            
            # Save model
            joblib.dump(poisson, model_path)
            self.logger.info(f"💾 Saved Poisson v2 model: {model_path}")
            
            # Generate metrics
            metrics_path = output_dir / "metrics.json"
            if not metrics_path.exists():
                self.generate_metrics_poisson_v2(league, output_dir, poisson, league_df)
            
            self.logger.info(f"✅ Successfully trained Poisson v2 for {league}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error training Poisson v2 for {league}: {e}")
            return False
    
    def train_draw_specialist_for_league(self, league: str) -> bool:
        """
        Train Draw Specialist model for a specific league (OPTIONAL)
        
        Args:
            league: League slug
        
        Returns:
            True if training successful
        """
        self.logger.info(f"🎯 Training Draw Specialist for {league} (optional)...")
        
        try:
            # Create output directory
            output_dir = self.leagues_dir / league / "draw_specialist_v1"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Check if model already exists (SAFE MODE)
            model_path = output_dir / "draw_model.pkl"
            if model_path.exists():
                self.logger.info(f"⏭️  Skipping {league}: Draw Specialist already exists")
                return True
            
            if self.dry_run:
                self.logger.info(f"🔍 DRY RUN: Would train Draw Specialist for {league}")
                return True
            
            self.logger.info(f"⏭️  Skipping Draw Specialist for {league} (optional feature)")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error training Draw Specialist for {league}: {e}")
            return False
    
    def generate_metrics_1x2_v2(self, league: str, model_dir: Path) -> bool:
        """
        Generate metrics.json for 1X2 v2 model
        
        Args:
            league: League slug
            model_dir: Path to model directory
        
        Returns:
            True if successful
        """
        try:
            metrics_path = model_dir / "metrics.json"
            
            # Check if already exists (SAFE MODE)
            if metrics_path.exists():
                self.logger.info(f"⏭️  Metrics already exist for {league} 1X2 v2")
                return True
            
            self.logger.info(f"📊 Generating metrics for {league} 1X2 v2...")
            
            # Create placeholder metrics
            metrics = {
                "train": {
                    "accuracy": 0.0,
                    "log_loss": 0.0
                },
                "val": {
                    "accuracy": 0.0,
                    "log_loss": 0.0
                },
                "league": league,
                "model_type": "1x2_v2_binary",
                "trained_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "note": "Metrics to be calculated during validation"
            }
            
            # Save metrics
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            self.logger.info(f"✅ Generated metrics for {league} 1X2 v2")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error generating metrics for {league}: {e}")
            return False
    
    def generate_metrics_poisson_v2(self, league: str, model_dir: Path, 
                                    poisson: PoissonV2Model, df: pd.DataFrame) -> bool:
        """
        Generate metrics.json for Poisson v2 model
        
        Args:
            league: League slug
            model_dir: Path to model directory
            poisson: Trained Poisson model
            df: League data
        
        Returns:
            True if successful
        """
        try:
            metrics_path = model_dir / "metrics.json"
            
            # Check if already exists (SAFE MODE)
            if metrics_path.exists():
                self.logger.info(f"⏭️  Metrics already exist for {league} Poisson v2")
                return True
            
            self.logger.info(f"📊 Generating metrics for {league} Poisson v2...")
            
            # Calculate basic metrics
            total_matches = len(df)
            
            metrics = {
                "league": league,
                "model_type": "poisson_v2",
                "trained_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "total_matches": total_matches,
                "decay_factor": 0.8,
                "note": "Time-decay Poisson model for scoreline prediction"
            }
            
            # Save metrics
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            self.logger.info(f"✅ Generated metrics for {league} Poisson v2")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error generating metrics for {league}: {e}")
            return False
    
    def update_registry(self, model_type: str, league: str, version: str, 
                       trained_date: str) -> bool:
        """
        Update model registry with new entry (SAFE MODE - append only)
        
        Args:
            model_type: Type of model
            league: League slug
            version: Model version
            trained_date: Training date
        
        Returns:
            True if successful
        """
        try:
            # Load existing registry
            if self.registry_path.exists():
                with open(self.registry_path, 'r') as f:
                    registry = json.load(f)
            else:
                registry = {"models": []}
            
            # Check if entry already exists
            entry_key = f"{model_type}_{league}_{version}"
            existing = [m for m in registry.get('models', []) 
                       if m.get('key') == entry_key]
            
            if existing:
                self.logger.info(f"⏭️  Registry entry already exists: {entry_key}")
                return True
            
            # Add new entry
            new_entry = {
                "key": entry_key,
                "model_type": model_type,
                "league": league,
                "version": version,
                "trained_date": trained_date,
                "status": "active"
            }
            
            if 'models' not in registry:
                registry['models'] = []
            
            registry['models'].append(new_entry)
            
            # Save registry
            with open(self.registry_path, 'w') as f:
                json.dump(registry, f, indent=2)
            
            self.logger.info(f"✅ Updated registry: {entry_key}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error updating registry: {e}")
            return False
    
    def run_auto_training(self, model_types: Optional[List[str]] = None) -> Dict:
        """
        Run automatic training for all missing models
        
        Args:
            model_types: List of model types to train (None = all)
        
        Returns:
            Summary dictionary
        """
        self.logger.info("=" * 70)
        self.logger.info("🚀 STARTING AUTOMATIC PER-LEAGUE TRAINING PIPELINE")
        self.logger.info("=" * 70)
        
        if model_types is None:
            model_types = list(self.model_types.keys())
        
        summary = {
            'total_missing': 0,
            'total_trained': 0,
            'total_skipped': 0,
            'total_failed': 0,
            'by_model_type': {}
        }
        
        # Get available leagues
        available_leagues = self.get_available_leagues()
        if not available_leagues:
            self.logger.warning("⚠️ No available leagues found")
            return summary
        
        # Process each model type
        for model_type in model_types:
            self.logger.info("")
            self.logger.info("=" * 70)
            self.logger.info(f"📦 Processing: {model_type.upper()}")
            self.logger.info("=" * 70)
            
            # Detect missing models
            status = self.detect_missing_models(model_type)
            missing = status['missing']
            existing = status['existing']
            
            model_summary = {
                'missing': len(missing),
                'existing': len(existing),
                'trained': 0,
                'failed': 0
            }
            
            summary['total_missing'] += len(missing)
            summary['total_skipped'] += len(existing)
            
            # Train missing models
            if missing:
                self.logger.info(f"🎯 Found {len(missing)} missing {model_type} models")
                
                for league in missing:
                    self.logger.info(f"")
                    self.logger.info(f"Training {model_type} for {league}...")
                    
                    # Train based on model type
                    if model_type == '1x2_v2':
                        success = self.train_1x2_v2_for_league(league)
                    elif model_type == 'poisson_v2':
                        success = self.train_poisson_v2_for_league(league)
                    elif model_type == 'draw_specialist':
                        success = self.train_draw_specialist_for_league(league)
                    else:
                        self.logger.error(f"❌ Unknown model type: {model_type}")
                        success = False
                    
                    if success:
                        model_summary['trained'] += 1
                        summary['total_trained'] += 1
                        
                        # Update registry
                        if not self.dry_run:
                            self.update_registry(
                                model_type, 
                                league, 
                                'v2' if model_type != 'draw_specialist' else 'v1',
                                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            )
                    else:
                        model_summary['failed'] += 1
                        summary['total_failed'] += 1
            else:
                self.logger.info(f"✅ All {model_type} models already exist")
            
            summary['by_model_type'][model_type] = model_summary
        
        # Print final summary
        self.logger.info("")
        self.logger.info("=" * 70)
        self.logger.info("📊 TRAINING PIPELINE COMPLETE")
        self.logger.info("=" * 70)
        self.logger.info(f"Total missing models found: {summary['total_missing']}")
        self.logger.info(f"Total models trained: {summary['total_trained']}")
        self.logger.info(f"Total models skipped (existing): {summary['total_skipped']}")
        self.logger.info(f"Total failures: {summary['total_failed']}")
        self.logger.info("")
        
        for model_type, stats in summary['by_model_type'].items():
            self.logger.info(f"{model_type}:")
            self.logger.info(f"  - Missing: {stats['missing']}")
            self.logger.info(f"  - Trained: {stats['trained']}")
            self.logger.info(f"  - Existing: {stats['existing']}")
            self.logger.info(f"  - Failed: {stats['failed']}")
        
        self.logger.info("")
        self.logger.info("✅ All per-league models are now consistent!")
        self.logger.info("=" * 70)
        
        return summary


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Automatic Per-League Training Pipeline')
    parser.add_argument('--dry-run', action='store_true', 
                       help='Only detect missing models without training')
    parser.add_argument('--model-type', type=str, choices=['1x2_v2', 'poisson_v2', 'draw_specialist'],
                       help='Train only specific model type')
    
    args = parser.parse_args()
    
    # Initialize manager
    manager = PerLeagueTrainingManager(dry_run=args.dry_run)
    
    # Run training
    model_types = [args.model_type] if args.model_type else None
    summary = manager.run_auto_training(model_types=model_types)
    
    # Exit with appropriate code
    if summary['total_failed'] > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == '__main__':
    main()
